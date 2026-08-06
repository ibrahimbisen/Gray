"""Train a stage, and feed the dashboard while it runs.

    python scripts/train.py Gray-Stand
    python scripts/train.py Gray-Push
    python scripts/train.py Gray-Push --num-envs 2048 --iterations 2000

mjlab writes its own logs under logs/rsl_rl/. The dashboard reads
progress/runs/. This runs the training and bridges between the two, so the run
appears at http://127.0.0.1:8000 the moment it starts and the curves fill in
live - rather than being something you only see once it has finished. It also
records what the run is scored on, so the dashboard can show that per run rather
than reading a task file that may since have changed.

It avoids installing the project too: mjlab normally discovers tasks through a
packaging entry point, but importing gray.tasks here registers them just the
same, and calling mjlab's launch_training directly skips the CLI.
"""

from __future__ import annotations

import _thread
import argparse
import csv
import json
import os
import shutil
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Make Ctrl-Break behave like Ctrl-C.
#
# scripts/runner.py stops a training job by sending CTRL_BREAK_EVENT, which is
# the only signal that reaches a child in its own process group on Windows. That
# raises SIGBREAK, and Python leaves SIGBREAK at its OS default: terminate
# immediately. Neither `except KeyboardInterrupt` nor `finally` below runs, so
# the run's status is never written and it shows as "running" on the dashboard
# forever - and the last metrics sync is lost with it.
#
# Measured before this line: exit code 0xC000013A, both handlers skipped.
if os.name == "nt":
    signal.signal(signal.SIGBREAK, signal.default_int_handler)

# rsl_rl records which commit a run came from, through GitPython, which refuses
# to import at all if it cannot find git. On Windows git is often on the shell's
# PATH but not the interpreter's, and losing a training run to that would be
# absurd - so point it at git if we can find it, and let it go quiet if we cannot.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
if "GIT_PYTHON_GIT_EXECUTABLE" not in os.environ:
    found = shutil.which("git") or next(
        (p for p in (r"C:\Program Files\Git\cmd\git.exe",
                     r"C:\Program Files (x86)\Git\cmd\git.exe")
         if Path(p).exists()), None)
    if found:
        os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = found

LOG_ROOT = ROOT / "logs" / "rsl_rl"

# Filming. ON for every run; --no-video is the only way off.
#
# One clip per 100 iterations, which costs about 4% of throughput. It was every
# 25 until 3 Aug 2026 - four times the intended rate and four times the cost,
# for clips nobody could see, because sync_once was throwing them away.
#
# STEPS_PER_ITER is mjlab's num_steps_per_env. It is 24 for every Gray task and
# is named here rather than written as a bare 24 in two places, because the two
# places have to agree: one sets the recording interval, the other turns the
# recorder's env-step filenames back into iteration numbers.
STEPS_PER_ITER = 24
FILM_EVERY = 100
RUNS = ROOT / "progress" / "runs"

# What each task is and how it is judged. Kept here rather than in the task file
# so the dashboard shows the same words the plan does, and so a finished run
# still carries the bar it was actually held to.
TASKS = {
    "Gray-Stand": {
        "name": "Stand still",
        "stage": 1,
        "purpose": "Can a policy hold the ride-height stance? The static check says "
                   "twelve servos at 1.96 N-m have 3.57x the torque they need, so a "
                   "failure here is the reward or the setup, not the robot.",
        "bar": "30 s without falling. Trunk within 5 mm of 190 mm. "
               "Uprightness above 0.99.",
    },
    "Gray-Push": {
        "name": "Take a push",
        "stage": 2,
        "purpose": "The first task that cannot be solved by memorising a pose - the "
                   "robot has to notice it was shoved and do something. That makes it "
                   "the first real use of the potentiometers, and it builds the reflex "
                   "walking needs, where every step is a disturbance the robot creates "
                   "for itself.",
        "bar": "Survives repeated shoves from any direction over 20 s, on ground "
               "anywhere from slippery to grippy, with mass and servo gains varied.",
    },
    "Gray-Walk": {
        "name": "Walk",
        "stage": 5,
        "purpose": "Go where it is told at 0.15-0.35 m/s and stop when told to stop. "
                   "Stages 3 and 4 are skipped on purpose - this is a straight test "
                   "of how much a policy picks up in a fixed amount of training. The "
                   "one number that decides it is the tracking band: mjlab's default "
                   "0.5 m/s would pay this robot 78% of full marks for standing "
                   "perfectly still, so it is set to 0.15 here.",
        "bar": "Walks 5 m without falling, holds commanded speed within 0.05 m/s, "
               "and drifts less than 100 mm sideways over 20 s.",
    },
}


def reward_notes() -> dict[str, str]:
    """Plain-English descriptions for every scoring term across the tasks."""
    from gray.tasks.push_env_cfg import PUSH_NOTES  # noqa: PLC0415
    from gray.tasks.stand_env_cfg import REWARD_NOTES  # noqa: PLC0415
    from gray.tasks.walk_env_cfg import WALK_NOTES  # noqa: PLC0415

    return {**REWARD_NOTES, **PUSH_NOTES, **WALK_NOTES}

# rsl_rl's tensorboard tags, and what to call them on the dashboard. Only the
# handful worth watching - the run page is a monitor, not an archive.
WATCH = {
    "Train/mean_reward": "reward",
    "Train/mean_episode_length": "episode_length",
    "Episode_Reward/height": "height_reward",
    # The walk task calls the same term ride_height. Without this line every
    # walking run shows an empty height chart, which reads as "the robot has no
    # height" rather than "nobody told the bridge the term was renamed".
    "Episode_Reward/ride_height": "height_reward",
    "Episode_Reward/tilt": "tilt_penalty",
    # Its own column, NOT tilt_penalty. Stage 1 used the name 'upright' for the
    # tilt PENALTY; the walking tasks use it for a levelness REWARD, so the same
    # tag means opposite things in old and new runs. Sharing a column would put a
    # positive number under a heading that says penalty.
    "Episode_Reward/upright": "upright_reward",
    "Episode_Termination/tipped_over": "tipped_over",
    "Episode_Termination/collapsed": "collapsed",
    "Policy/mean_std": "exploration",
    "Perf/total_fps": "steps_per_second",

    # Walking only. These are what say whether it is actually WALKING, as opposed
    # to merely scoring well. The reward is a weighted sum and can read
    # excellently while the robot creeps along on stiff legs scuffing its feet -
    # which is precisely the gait that works in simulation and falls over on a
    # real floor. Each of these isolates one way that can happen.
    #
    # `Metrics/air_time_mean` and `Metrics/peak_height_mean` used to be listed
    # here and do not exist - nothing logs them, so those two lines had been
    # quietly doing nothing. The tags below were read off a real run's event
    # file rather than guessed.
    "Metrics/walk/error_vel_xy": "speed_error",
    "Metrics/walk/error_vel_yaw": "turn_error",
    # Feet spending a sensible time in the air: the difference between a gait
    # and a shuffle.
    "Episode_Reward/stepping": "stepping",
    # How far the thighs and calves actually swing. The owner's word for what is
    # wanted is "animated"; without this the cheapest gait is tiny stiff steps.
    "Episode_Reward/leg_swing": "leg_swing",
    # A foot at the wrong height while travelling. Scuffing is free in
    # simulation and trips on carpet.
    "Episode_Reward/dragging": "dragging",
    # Ground actually put behind it. Velocity alone can be faked by rocking.
    "Episode_Reward/ground_covered": "ground_covered",
    # How hard the feet land. Printed PLA with no suspension, so this is a
    # hardware limit and not only a comfort one.
    "Metrics/landing_force_mean": "landing_force",
}


def newest_log_dir(after: float, timeout: float = 900.0) -> Path | None:
    """mjlab names its run directory by the clock, so find it rather than guess."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if LOG_ROOT.is_dir():
            for exp in LOG_ROOT.iterdir():
                for run in exp.iterdir() if exp.is_dir() else []:
                    if run.is_dir() and run.stat().st_mtime >= after - 5:
                        return run
        time.sleep(1.0)
    return None


def read_scalars(log_dir: Path) -> tuple[list[str], list[dict]]:
    """Pull the curves out of the tensorboard file training is writing to."""
    from tensorboard.backend.event_processing.event_accumulator import (  # noqa: PLC0415
        EventAccumulator,
    )

    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 100_000})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))

    # EVERY reward term, curriculum weight, metric and termination - not the
    # handful WATCH names. WATCH is still what sets the ORDER and the friendly
    # column names; everything else follows it, keeping its raw tag.
    #
    # Why this changed on 3 Aug 2026. metrics.csv carried 5 of 24 reward terms,
    # and every bug found in this reward function so far has been a term sitting
    # silently at zero - `_going_straight` off for backward commands,
    # `ground_covered` unable to pay backward at all, `stepping` paying a
    # standing robot to march. A term at zero and a term passing look exactly
    # the same from outside, and the only way to tell is to be able to see it.
    # Four times is enough.
    #
    # Cost: about 40 columns instead of 12, on a file with one row per
    # iteration. A 3000-iteration run goes from roughly 300 kB to 1 MB.
    extra = sorted(t for t in available
                   if t not in WATCH and t.split("/")[0] in
                   ("Episode_Reward", "Curriculum", "Metrics", "Episode_Termination"))

    rows: dict[int, dict] = {}
    used: list[str] = []
    for tag, name in list(WATCH.items()) + [(t, t) for t in extra]:
        if tag not in available:
            continue
        used.append(name)
        for ev in acc.Scalars(tag):
            rows.setdefault(ev.step, {"iteration": ev.step})[name] = round(ev.value, 6)
    ordered = [rows[k] for k in sorted(rows)]
    return used, ordered


def sync_once(run_dir: Path, log_dir: Path) -> None:
    """Copy metrics, videos and checkpoint markers into the shape the dashboard reads."""
    try:
        cols, rows = read_scalars(log_dir)
        if rows:
            names = ["iteration"] + [c for c in cols if c != "iteration"]
            with (run_dir / "metrics.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in names})

        # mjlab's own recorder films every 600 env-steps and names the files by
        # step. Those clips used to be left where they fell, on the grounds that
        # scripts/film_checkpoints.py would produce nicer ones - but that script
        # is a SECOND process on the graphics card, so at 4500 robots there is no
        # room to run it, and in practice nobody starts it. The result was a
        # dashboard showing one clip from iteration 0 and nothing else, for a run
        # that had nine clips sitting on disk the whole time.
        #
        # So: import them. Named `train_NNNN.mp4` by ITERATION, which is both the
        # number the rest of the dashboard speaks in and distinct from
        # film_checkpoints.py's `iter_NNNN.mp4`, so the two can coexist rather
        # than collide at "iteration 0" the way they did before.
        steps_per_iter = STEPS_PER_ITER
        for clip in sorted(log_dir.glob("videos/train/rl-video-step-*.mp4")):
            try:
                step = int(clip.stem.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            it = step // steps_per_iter
            # Only the cadence FILM_EVERY asks for. A run started before that
            # constant changed keeps recording at its old, denser interval - the
            # process cannot pick up an edit mid-flight - so without this the
            # dashboard fills with clips at a spacing nobody asked for and the
            # strip becomes unreadable. Iteration 0 always survives: it is the
            # "before" that every later clip is read against.
            if it and it % FILM_EVERY:
                continue
            out = run_dir / "videos" / f"iter_{it:04d}.mp4"
            # Size check, not just existence: the recorder writes the file as it
            # goes, so a clip copied mid-write would be a truncated one that
            # never gets corrected.
            if clip.stat().st_size and (
                    not out.exists() or out.stat().st_size != clip.stat().st_size):
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(clip, out)

        # The dashboard only counts checkpoints, so a marker per .pt is enough
        # and keeps a few hundred MB of weights out of progress/.
        for pt in log_dir.glob("model_*.pt"):
            marker = run_dir / "checkpoints" / f"{pt.stem}.npz"
            if not marker.exists():
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_bytes(b"")
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] {type(exc).__name__}: {exc}", flush=True)


def reward_ceiling(rewards) -> float:
    """The most an episode could possibly score.

    Every positive term is capped at 1.0, weighted, multiplied by the timestep
    and summed over the episode - so the largest reachable total is the positive
    weights times the episode length. See RULES.md, rule 1.
    """
    return sum(t.weight for t in rewards.values() if t.weight > 0)


def world_dials(env_cfg) -> list[dict]:
    """The five world dials, as this run will actually draw them.

    One entry per dial: the range it draws from, and whether that is the wide
    value or the narrow one. `wide` is decided by comparing against WIDE_DIALS
    rather than by trusting the `--narrow-dials` argument, so a dial changed by
    any other route still reports itself correctly.

    Returns an empty list for a task that has no dials, which is every task but
    the walk. Nothing downstream may assume the block is there.
    """
    try:
        from gray.tasks.walk_env_cfg import WIDE_DIALS  # noqa: PLC0415
    except ImportError:
        return []

    out = []
    for name, wide in WIDE_DIALS.items():
        term = getattr(env_cfg, "events", {}).get(name)
        if term is None:
            continue
        drawn = {k: tuple(term.params[k]) for k in wide}
        out.append({"name": name, "ranges": drawn,
                    "wide": drawn == {k: tuple(v) for k, v in wide.items()}})
    return out


def bridge(run_dir: Path, log_dir: Path, stop: threading.Event,
           target: float = 0.0) -> None:
    """Feed the dashboard, and stop training once the reward has nothing left to win."""
    while not stop.is_set():
        sync_once(run_dir, log_dir)
        if target:
            try:
                _, rows = read_scalars(log_dir)
                latest = rows[-1].get("reward") if rows else None
                if latest is not None and latest >= target:
                    print(f"\n[stop] reward {latest:.2f} reached the {target:.2f} "
                          f"target - every positive term is maxed, so the rest of the "
                          f"schedule would only polish. See RULES.md rule 1.",
                          flush=True)
                    stop.set()
                    # rsl_rl has no way to be asked to stop mid-learn, so raise
                    # into the main thread. The last checkpoint is at most 25
                    # iterations behind, and training's own finally block runs.
                    _thread.interrupt_main()
                    return
            except Exception as exc:  # noqa: BLE001
                print(f"[stop] could not read the reward: {exc}", flush=True)
        stop.wait(10.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", nargs="?", default="Gray-Stand", choices=sorted(TASKS),
                    help="which stage to train")
    ap.add_argument("--num-envs", type=int, default=3072,
                    help="robots trained at once. 4096 is this card's ceiling; "
                         "3072 leaves room for the video renderer")
    ap.add_argument("--iterations", type=int, default=0,
                    help="0 uses the task's own default")
    ap.add_argument("--name", default="")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--stop-at", type=float, default=0.965, metavar="FRACTION",
                    help="stop once the reward reaches this fraction of the most it "
                         "could score. 0 runs the full schedule. See RULES.md rule 1.")
    ap.add_argument("--push-speed", type=float, nargs=2, metavar=("MIN", "MAX"),
                    help="how hard the shoves are, m/s of instant trunk speed. "
                         "On 1.99 kg, 1 m/s is about 2.0 N-s.")
    ap.add_argument("--push-spin", type=float, nargs=2, metavar=("MIN", "MAX"),
                    help="how much spin each shove adds, rad/s about the vertical")
    ap.add_argument("--seed", type=int, default=0, metavar="N",
                    help="random seed. 0 keeps the task's own (42). Without this "
                         "every run of one config gives the same answer, so a "
                         "sweep cannot tell a real difference from noise - run "
                         "the same config on two seeds and the gap between them "
                         "is the noise floor everything else has to beat.")
    ap.add_argument("--reward", action="append", default=[], metavar="NAME=WEIGHT",
                    help="change one scoring weight, e.g. --reward dragging=-3. "
                         "Repeatable. The change is recorded in the run, so the "
                         "dashboard shows what this run was actually scored on. "
                         "Refused for terms a curriculum drives - use --ramp.")
    ap.add_argument("--ramp", action="append", default=[], metavar="NAME=W0,W1,W2,W3",
                    help="change every stage of a term's curriculum, e.g. "
                         "--ramp veering=-0.4,-1.0,-2.4,-4.0. One weight per "
                         "stage. Needed because a curriculum re-applies its own "
                         "weight, so --reward on a ramped term is a silent no-op.")
    ap.add_argument("--turn-std", type=float, default=0.0, metavar="RAD_PER_S",
                    help="how sharply `track_turn` scores yaw rate. 0 keeps the "
                         "task's own (0.80). This is a tolerance, not a weight, "
                         "so --reward cannot reach it: the term pays "
                         "exp(-err^2/std^2), and at std 0.80 the 0.018 rad/s "
                         "bias that produced round 0's whole straightness "
                         "failure still collected 99.95%% of full marks. Try "
                         "0.15.")
    ap.add_argument("--upright-std", type=float, default=0.0, metavar="RAD",
                    help="how sharply `upright` scores trunk tilt. 0 keeps the "
                         "task's own (0.45 rad, which is 26 DEGREES). Measured "
                         "on r1a: the robot holds 2.9 deg of tilt, and at std "
                         "0.45 holding perfectly level instead would earn it "
                         "0.8%% of one term. Try 0.15.")
    ap.add_argument("--gyro-noise", type=float, nargs=2, default=None,
                    metavar=("BIAS", "WALK"),
                    help="how wrong the heading the policy READS may be, in "
                         "radians: a per-episode offset, and a wander in rad per "
                         "root-second. Defaults 0.009 and 0.008, about half a "
                         "degree and 2 deg over a 20 s episode. '0 0' hands the "
                         "policy a perfect heading, which is the setting that "
                         "does not survive contact with a real gyro.")
    ap.add_argument("--no-heading-obs", action="store_true",
                    help="train BLIND to which way it is off the line - the 48-"
                         "input policy every run before 4 Aug 2026 used. Here so "
                         "the heading input can be measured against its own "
                         "absence rather than against a run that also differs in "
                         "iterations and robot count.")
    ap.add_argument("--crab-share", type=float, default=None,
                    help="the share of commands drawn as a PURE sideways step, "
                         "0 to 1. The task uses 0.15. It is here because it "
                         "turned out to cost turn accuracy, so it has to be "
                         "variable to be measured - 0 switches it off entirely "
                         "and gives the draw mix as it was before 5 Aug 2026.")
    ap.add_argument("--spin-share", type=float, default=None,
                    help="the share of commands drawn as a PURE turn on the "
                         "spot, 0 to 1. The task default is 0.0 - an "
                         "independent draw produces a pure spin about once in "
                         "80, which is why the 1.00 rad/s spin bar was never "
                         "passed by a robot that turns well driven. The g1 "
                         "probe of the gait batch runs this at 0.10.")
    ap.add_argument("--dive-ends", action="store_true",
                    help="install the nose_dived termination: the trunk shell "
                         "touching anything ends the attempt, and fell_over "
                         "pays its -40 for it. The trunk contact sensor is "
                         "always on; this makes it terminal. The g2 probe of "
                         "the gait batch - the owner's films show a nose-down "
                         "collapse that today costs the policy nothing.")
    ap.add_argument("--swing-target", type=float, default=0.0, metavar="M",
                    help="the height a swing is scored against, in metres, on "
                         "BOTH dragging and swing_height. The task uses 0.035, "
                         "chosen when 35 mm matched the stage 3 bar. The g3 "
                         "probe runs 0.05: a foot with 35 mm in hand has "
                         "nothing left for ground that is not a floor.")
    ap.add_argument("--narrow-dials", default="",
                    help="put named world dials back to their Gray-Push values, "
                         "comma separated, or 'all'. The walk task widens all "
                         "five - see WIDE_DIALS in gray/tasks/walk_env_cfg.py. "
                         "Batch 1 of PLAN 1.3.1 widened them together and failed "
                         "all three seeds on the four heading criteria, so this "
                         "exists to run them one at a time against a narrow "
                         "control and find which one costs the heading.")
    ap.add_argument("--with-off-track", action="store_true",
                    help="ADD the cross-track input, making the observation 50 "
                         "wide instead of 49. Off by default because it lost "
                         "ground on every criterion it was meant to help - see "
                         "off_track_obs in gray/tasks/walk_env_cfg.py. Kept "
                         "because the reasoning behind it is still the best "
                         "account of why crab drift is hard.")
    args = ap.parse_args()

    import gray.tasks  # noqa: F401  - registers the tasks
    from mjlab.scripts.train import TrainConfig, launch_training

    meta = TASKS[args.task]
    run_name = args.name or args.task.split("-")[-1].lower()

    cfg = TrainConfig.from_task(args.task)
    cfg.env.scene.num_envs = args.num_envs
    if args.iterations:
        cfg.agent.max_iterations = args.iterations
    iterations = cfg.agent.max_iterations
    cfg.agent.run_name = run_name
    if args.seed:
        cfg.agent.seed = args.seed

    shove = cfg.env.events.get("shove")
    if shove is None and (args.push_speed or args.push_spin):
        raise SystemExit(f"{args.task} has no shove event to change")
    if args.push_speed:
        was = shove.params["speed_range"]
        shove.params["speed_range"] = tuple(args.push_speed)
        print(f"shove         speed: {was} -> {tuple(args.push_speed)} m/s "
              f"(up to {args.push_speed[1] * 2.3787:.1f} N-s)")
    if args.push_spin:
        was = shove.params["spin_range"]
        shove.params["spin_range"] = tuple(args.push_spin)
        print(f"shove         spin: {was} -> {tuple(args.push_spin)} rad/s")

    # Which terms a curriculum drives, and therefore which ones --reward cannot
    # touch: the curriculum re-applies its own weight on every evaluation, so a
    # --reward on a ramped term is silently overwritten a few seconds in. That is
    # the worst kind of no-op - the run records the weight it was asked for and
    # trains on a different one.
    ramped = {c.params["reward_name"]: c
              for c in (cfg.env.curriculum or {}).values()
              if c.params.get("reward_name")}

    # Tolerances, not weights. These live in a term's `params`, so --reward and
    # --ramp cannot touch them, and until 4 Aug 2026 nothing could - which is how
    # TURN_STD sat at a value that priced the whole straightness failure at five
    # parts in ten thousand for as long as it did.
    if args.turn_std:
        turn = cfg.env.rewards.get("track_turn")
        if turn is None:
            raise SystemExit(f"{args.task} has no 'track_turn' term to sharpen")
        was = turn.params["std"]
        turn.params["std"] = args.turn_std
        print(f"tolerance     track_turn std: {was} -> {args.turn_std} rad/s")
    if args.upright_std:
        up = cfg.env.rewards.get("upright")
        if up is None:
            raise SystemExit(f"{args.task} has no 'upright' term to sharpen")
        was = up.params["std"]
        up.params["std"] = args.upright_std
        print(f"tolerance     upright std: {was} -> {args.upright_std} rad "
              f"({args.upright_std * 57.3:.0f} deg)")
    if args.no_heading_obs:
        gone = [g for g in ("actor", "critic")
                if cfg.env.observations[g].terms.pop("off_line", None) is not None]
        if not gone:
            raise SystemExit("this task has no 'off_line' observation to remove")
        print(f"observation   off_line removed from {', '.join(gone)} - "
              f"the policy trains blind to its heading error")
    if args.crab_share is not None:
        walk_cmd = cfg.env.commands.get("walk")
        if walk_cmd is None or not hasattr(walk_cmd, "rel_crab_envs"):
            raise SystemExit(f"{args.task} has no crab share to set")
        was = walk_cmd.rel_crab_envs
        walk_cmd.rel_crab_envs = args.crab_share
        print(f"command mix   pure sideways share: {was} -> {args.crab_share}")
    if args.spin_share is not None:
        walk_cmd = cfg.env.commands.get("walk")
        if walk_cmd is None or not hasattr(walk_cmd, "rel_spin_envs"):
            raise SystemExit(f"{args.task} has no spin share to set")
        was = walk_cmd.rel_spin_envs
        walk_cmd.rel_spin_envs = args.spin_share
        print(f"command mix   pure spin share: {was} -> {args.spin_share}")
    if args.dive_ends:
        from gray.tasks.walk_env_cfg import dive_termination  # noqa: PLC0415

        if "trunk" not in {s.name for s in getattr(cfg.env.scene, "sensors", ())}:
            raise SystemExit(f"{args.task} has no trunk contact sensor to read")
        cfg.env.terminations["nose_dived"] = dive_termination()
        print("termination   nose_dived: trunk contact ends the attempt")
    if args.swing_target:
        was = None
        for name in ("dragging", "swing_height"):
            term = cfg.env.rewards.get(name)
            if term is None:
                raise SystemExit(f"{args.task} has no '{name}' term to retarget")
            was = term.params["target"]
            term.params["target"] = args.swing_target
        print(f"tolerance     swing target: {was} -> {args.swing_target} m")
    if args.narrow_dials:
        # `narrow_dials` is set by walk_env_cfg while it widens them, so it holds
        # the value each dial actually had, not a second copy of the table.
        was = getattr(cfg.env, "narrow_dials", None)
        if not was:
            raise SystemExit(f"{args.task} has no widened world dials to narrow")
        wanted = (list(was) if args.narrow_dials.strip() == "all" else
                  [n.strip() for n in args.narrow_dials.split(",") if n.strip()])
        unknown = [n for n in wanted if n not in was]
        if unknown:
            raise SystemExit(
                f"no such world dial: {', '.join(unknown)}. "
                f"The five are: {', '.join(was)}")
        for name in wanted:
            cfg.env.events[name].params.update(was[name])
        print(f"world dials   back to the narrow range: {', '.join(wanted)}")
    if args.with_off_track:
        from mjlab.managers import ObservationTermCfg  # noqa: PLC0415

        from gray.tasks.walk_env_cfg import off_track_obs  # noqa: PLC0415

        for g in ("actor", "critic"):
            terms = getattr(cfg.env.observations.get(g), "terms", None)
            if terms is None:
                continue
            terms["off_track"] = ObservationTermCfg(
                func=off_track_obs, params={"command_name": "walk"})
        print("observation   off_track ADDED to actor, critic - the policy is "
              "told how far off the line it has ended up, 50 inputs not 49")
    if args.gyro_noise is not None:
        walk_cmd = cfg.env.commands.get("walk")
        if walk_cmd is None or not hasattr(walk_cmd, "gyro_bias_rad"):
            raise SystemExit(f"{args.task} has no heading observation to corrupt")
        bias, wander = args.gyro_noise
        was = (walk_cmd.gyro_bias_rad, walk_cmd.gyro_walk_rad_per_s)
        walk_cmd.gyro_bias_rad, walk_cmd.gyro_walk_rad_per_s = bias, wander
        print(f"sensor        heading noise: {was} -> {(bias, wander)}")

    for change in args.reward:
        term, _, weight = change.partition("=")
        if term not in cfg.env.rewards:
            raise SystemExit(
                f"no scoring term called {term!r}. This run has: "
                f"{', '.join(sorted(cfg.env.rewards))}")
        if term in ramped:
            raise SystemExit(
                f"{term!r} is driven by a curriculum, so --reward on it would be "
                f"overwritten within seconds and the run would record a weight it "
                f"never trained on. Use --ramp {term}=w0,w1,w2,w3 instead.")
        was = cfg.env.rewards[term].weight
        cfg.env.rewards[term].weight = float(weight)
        print(f"scoring       {term}: {was} -> {float(weight)}")

    for change in args.ramp:
        term, _, spec = change.partition("=")
        if term not in ramped:
            raise SystemExit(
                f"{term!r} has no curriculum. Ramped terms are: "
                f"{', '.join(sorted(ramped)) or 'none'}. Use --reward for the rest.")
        stages = ramped[term].params["stages"]
        want = [float(x) for x in spec.split(",") if x.strip()]
        if len(want) != len(stages):
            raise SystemExit(
                f"{term!r} has {len(stages)} ramp stages "
                f"(at steps {', '.join(str(s['step']) for s in stages)}), "
                f"but {len(want)} weights were given.")
        was = [s["weight"] for s in stages]
        for stage, w in zip(stages, want):
            stage["weight"] = w
        # The first stage is what the term starts at, so set it directly too -
        # the curriculum does not run until the first evaluation.
        cfg.env.rewards[term].weight = want[0]
        print(f"ramp          {term}: {was} -> {want}")
    cfg = TrainConfig(
        env=cfg.env, agent=cfg.agent,
        # Filming is ON for every run unless --no-video says otherwise. It is not
        # a nice-to-have: the reward is a weighted sum that can read excellently
        # while the robot creeps along scuffing its feet, and the film is the only
        # thing that shows that.
        video=not args.no_video,
        video_length=250,
        # The recorder counts calls to env.step(), NOT robot-steps - multiplying
        # by num_envs makes the interval so large it only ever fires once, at
        # step 0, which is what happened on the first real run.
        video_interval=STEPS_PER_ITER * FILM_EVERY,
        log_root=str(LOG_ROOT.relative_to(ROOT)),
        gpu_ids=[0],
    )

    # Record what this run is actually scored on, so the dashboard can show it
    # without anyone reading the task file - and so an old run still says what it
    # was scored on after the rewards have been changed.
    notes = reward_notes()
    missing = [n for n in cfg.env.rewards if n not in notes]
    if missing:
        # A term with no description shows as a blank row on the dashboard, which
        # is worse than no row - say so rather than let it slip through.
        print(f"[warn] no description for scoring terms: {', '.join(missing)}")
    scoring = sorted(
        ({"name": name, "weight": float(term.weight), "what": notes.get(name, "")}
         for name, term in cfg.env.rewards.items()),
        key=lambda r: -abs(r["weight"]),
    )

    # What the ramped terms actually end at. `scoring` above records term.weight,
    # which for a ramped term is only its stage-0 value - so a run whose veering
    # reaches -2.0 by iteration 500 recorded it as -0.2 and the dashboard showed
    # a penalty four times weaker than the one it trained on.
    ramps = [{"name": name,
              "stages": [{"step": s["step"], "weight": s["weight"]}
                         for s in c.params["stages"]]}
             for name, c in sorted(ramped.items())]

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{run_name}"
    run_dir = RUNS / run_id
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({
        "name": meta["name"],
        "purpose": meta["purpose"],
        "stage": meta["stage"],
        "stage_name": meta["name"],
        "task": args.task,
        "status": "running",
        "bar": meta["bar"],
        "started": datetime.now().isoformat(timespec="seconds"),
        "finished": None,
        "iterations_target": iterations,
        # Written as fields, not only prose. The dashboard needs them to work out
        # seconds per iteration: one iteration is num_envs x num_steps_per_env
        # robot-steps, and Perf/total_fps counts robot-steps. Deriving pace from
        # wall-clock instead folds in 40 s of startup and reads 30% slow for the
        # first few hundred iterations - exactly when someone is watching.
        "num_envs": args.num_envs,
        "num_steps_per_env": cfg.agent.num_steps_per_env,
        # Recorded so two runs of one config can be told apart. Without it a
        # sweep has no noise floor and every difference looks meaningful.
        "seed": cfg.agent.seed,
        "notes": f"{args.num_envs} robots at once, 50 Hz control."
                 + (f" Shoves {shove.params['speed_range'][0]}-"
                    f"{shove.params['speed_range'][1]} m/s from any angle, spin "
                    f"{shove.params['spin_range'][1]} rad/s." if shove else ""),
        "scoring": scoring,
        "ramps": ramps,
        # Numbers that decide what a term MEANS rather than what it is worth.
        # They are not weights, so they were invisible to every page that reads
        # `scoring` - and `track_turn`'s std is the one that let a 0.018 rad/s
        # turning bias collect 99.95% of full marks through all of round 0.
        # Recorded from the live config, so a sweep over them can be read back.
        "tolerances": {
            "track_turn_std": float(cfg.env.rewards["track_turn"].params["std"])
            if "track_turn" in cfg.env.rewards else None,
            "upright_std": float(cfg.env.rewards["upright"].params["std"])
            if "upright" in cfg.env.rewards else None,
            "gyro_bias_rad": getattr(cfg.env.commands.get("walk"),
                                     "gyro_bias_rad", None),
            "gyro_walk_rad_per_s": getattr(cfg.env.commands.get("walk"),
                                           "gyro_walk_rad_per_s", None),
            # The draw mix, recorded since 6 Aug 2026. The g1 spin probe
            # differs from its control in NOTHING but rel_spin_envs, and
            # before this the only record of that was the job note.
            "rel_crab_envs": getattr(cfg.env.commands.get("walk"),
                                     "rel_crab_envs", None),
            "rel_spin_envs": getattr(cfg.env.commands.get("walk"),
                                     "rel_spin_envs", None),
            # The swing target, for the same reason: g3 differs from its
            # control in nothing but this number.
            "swing_target_m": (
                float(cfg.env.rewards["swing_height"].params["target"])
                if "swing_height" in cfg.env.rewards else None),
        },
        # The world the run trained in, read off the live config. Added 5 Aug
        # 2026, and it is not a nicety: /dials reads the SOURCE file, so it
        # shows the task default and every run looks alike. Batch 2 of PLAN
        # 1.3.1 is five runs that differ in NOTHING BUT these five ranges, and
        # without this the only record of which was which is a job note.
        "world_dials": world_dials(cfg.env),
        # WHAT the policy reads, by name. Not how many numbers - that needs a
        # built env and this runs before one exists. The names are the useful
        # part anyway: a saved policy is a fixed-size mapping, so two runs whose
        # lists differ cannot load each other's checkpoints, and this list is
        # what says which. It has changed twice in two days.
        "observes": sorted(getattr(cfg.env.observations.get("actor"),
                                   "terms", {}) or {}),
    }, indent=2))

    ceiling = reward_ceiling(cfg.env.rewards) * cfg.env.episode_length_s
    target = ceiling * args.stop_at if args.stop_at else 0.0

    print(f"training      {args.task} - {meta['name']}")
    print(f"              {args.num_envs} robots, up to {iterations} iterations")
    print(f"reward        ceiling {ceiling:.1f} "
          f"({reward_ceiling(cfg.env.rewards):.1f}/s over "
          f"{cfg.env.episode_length_s:.0f} s)")
    print(f"              stopping at {args.stop_at:.0%} of it = {target:.1f}"
          if target else "              running the full schedule")
    print(f"dashboard     http://127.0.0.1:8000  ->  {run_id}")
    print(f"tensorboard   tensorboard --logdir {LOG_ROOT}\n")

    started = time.time()
    stop = threading.Event()
    found: dict[str, Path] = {}

    def watch():
        log_dir = newest_log_dir(started)
        if log_dir is None:
            print("[bridge] never found the log directory; dashboard will stay empty")
            return
        found["dir"] = log_dir
        print(f"[bridge] following {log_dir}", flush=True)
        bridge(run_dir, log_dir, stop, target)

    threading.Thread(target=watch, daemon=True).start()

    status = "finished"
    try:
        launch_training(task_id=args.task, args=cfg)
    except KeyboardInterrupt:
        # Either Ctrl-C, or the reward hit its target and the bridge interrupted
        # us deliberately. Those are opposite outcomes and must not read the same.
        status = "reached target" if stop.is_set() else "cancelled"
    except Exception:
        status = "failed"
        raise
    finally:
        stop.set()
        # One last pass. A short run can finish inside the bridge's poll interval,
        # and then the dashboard would show a run with no curves at all - which is
        # exactly what happened the first time.
        if "dir" in found:
            sync_once(run_dir, found["dir"])
        meta = json.loads((run_dir / "run.json").read_text())
        meta["status"] = status
        meta["finished"] = datetime.now().isoformat(timespec="seconds")
        (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
        print(f"\nrun {status}: {run_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
