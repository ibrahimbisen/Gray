"""Drive a trained policy with a command YOU choose, film it, and measure it.

verify.py pins one forward speed because that is what the bar asks for.
film_checkpoints.py films whatever random command the env happened to draw. This
is the third thing: hand the policy a specific command, hold it, and report what
the robot actually did against what it was told.

It exists to answer a fair question - "you say one policy covers sideways and
turning, so show me it walk diagonally." A video alone does not settle that,
because a robot leaning into a turn looks diagonal for a second. So every clip
comes with the ground track measured in the robot's OWN starting frame, the same
way verify.py measures drift:

    along  =  moved_x*cos(h0) + moved_y*sin(h0)
    across = -moved_x*sin(h0) + moved_y*cos(h0)

atan2(across, along) is then the true angle walked, and it can be held against
the angle commanded. Reset nudges each spawn yaw by up to +/-0.1 rad, so
measuring against world X instead would charge the policy for the test's own
randomisation.

One thing worth knowing before reading a result: a command outside the range the
policy TRAINED on proves nothing. walk_env_cfg samples vx in (0.15, 0.35) and vy
in (-0.10, 0.10), so the steepest diagonal ever sampled is about 22 degrees off
straight. Asking for 45 is asking about a region of the command space that was
never visited, and the honest answer there is "unknown", not "cannot".

    python run.py scripts/drive.py --run 25 \
        --cases "0.25,0,0; 0.25,0.10,0; 0,0.10,0"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Two trees hold the same runs under the same folder names, and they are not
# interchangeable. progress/runs/ is the dashboard's copy - run.json, metrics.csv,
# videos, and .npz checkpoints. logs/rsl_rl/ is what rsl_rl actually wrote, and
# it is the only one with the .pt files a policy can be loaded from. Run NUMBERS
# are defined by progress/runs (oldest is 1, same rule as dashboard/runs.py), so
# a number is resolved there and then mapped across by name.
RUNS = ROOT / "progress" / "runs"
LOG_ROOT = ROOT / "logs" / "rsl_rl"


def _quiet_break() -> None:
    """Ctrl-Break must raise, not terminate.

    Windows defaults SIGBREAK to hard termination, which skips `finally` and
    every `except KeyboardInterrupt` - so a cancelled job left half-written
    files behind and a run that said "running" forever. Mapped to the same
    handler as Ctrl-C.
    """
    handler = getattr(signal, "default_int_handler", None)
    if hasattr(signal, "SIGBREAK") and handler is not None:
        signal.signal(signal.SIGBREAK, handler)


def resolve_run(which: str) -> Path:
    """The rsl_rl log directory for a run number, a name, or a folder name."""
    dirs = sorted(d for d in RUNS.iterdir() if d.is_dir())
    if which.isdigit():
        n = int(which)
        if not 1 <= n <= len(dirs):
            raise SystemExit(f"run {n} does not exist - there are {len(dirs)}")
        name = dirs[n - 1].name                 # oldest is 1, same as runs.py
    else:
        exact = [d for d in dirs if d.name == which]
        matches = exact or [d for d in dirs if which in d.name]
        if not matches:
            raise SystemExit(f"no run matching {which!r}")
        name = matches[-1].name                 # the newest of the matches

    for exp in sorted(LOG_ROOT.iterdir()) if LOG_ROOT.is_dir() else []:
        candidate = exp / name
        if candidate.is_dir():
            return candidate
    raise SystemExit(
        f"run {name!r} has no rsl_rl log directory under {LOG_ROOT} - so it has "
        f"no .pt checkpoints and nothing can be loaded from it")


def parse_cases(text: str) -> list[dict]:
    out = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        bits = [b.strip() for b in part.split(",")]
        if len(bits) != 3:
            raise SystemExit(f"case {part!r} is not 'vx,vy,yaw'")
        vx, vy, yaw = (float(b) for b in bits)
        out.append({"vx": vx, "vy": vy, "yaw": yaw})
    if not out:
        raise SystemExit("no cases given")
    return out


def label(c: dict) -> str:
    name = []
    if c["vx"]:
        name.append(f"fwd{c['vx']:g}")
    if c["vy"]:
        name.append(f"side{c['vy']:+g}")
    if c["yaw"]:
        name.append(f"turn{c['yaw']:+g}")
    return "_".join(name) or "stop"


def main() -> None:
    _quiet_break()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run number, name, or folder")
    ap.add_argument("--checkpoint", help="e.g. model_2999.pt; default is the last")
    ap.add_argument("--task", default="Gray-Walk")
    ap.add_argument("--cases", default="0.25,0,0; 0.25,0.10,0; 0,0.10,0",
                    help="semicolon-separated 'vx,vy,yaw' in m/s and rad/s")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--robots", type=int, default=8,
                    help="measured over all of them; env 0 is the one filmed")
    ap.add_argument("--out", default="", help="folder for the clips")
    # Higher and further back than the checkpoint camera. A chase cam at -18
    # degrees cannot show a ground track, and a ground track is the whole point.
    ap.add_argument("--distance", type=float, default=3.4)
    ap.add_argument("--elevation", type=float, default=-38.0)
    ap.add_argument("--azimuth", type=float, default=90.0)
    args = ap.parse_args()

    log_dir = resolve_run(args.run)
    # Sorted numerically, not lexicographically. By name model_975 comes after
    # model_2999, so the last checkpoint of a 3000-iteration run would be one
    # from iteration 975 - a policy a third of the way through its training.
    ckpts = sorted(log_dir.glob("model_*.pt"),
                   key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {log_dir}")
    ckpt = (log_dir / args.checkpoint) if args.checkpoint else ckpts[-1]
    if not ckpt.is_file():
        raise SystemExit(f"no such checkpoint: {ckpt}")

    cases = parse_cases(args.cases)
    out_dir = Path(args.out) if args.out else (log_dir / "drive")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run         {log_dir.name}")
    print(f"checkpoint  {ckpt.name}")
    print(f"cases       {len(cases)}   robots {args.robots}   {args.seconds:g} s each")
    print()

    from dataclasses import asdict  # noqa: PLC0415

    import imageio.v2 as imageio  # noqa: PLC0415
    import mujoco  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    import gray.tasks  # noqa: F401,PLC0415
    from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: PLC0415

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.robots
    env_cfg.episode_length_s = args.seconds + 5.0

    cmd = env_cfg.commands.get("walk")
    if cmd is None:
        raise SystemExit(f"{args.task} has no 'walk' command to pin")
    # rel_forward_envs zeroes vy and yaw for that fraction of robots. It is 0.8
    # during training, which is why almost every checkpoint clip shows a straight
    # line. Here it must be 0, or the sideways component of a diagonal command is
    # thrown away before the policy ever sees it.
    cmd.rel_standing_envs = 0.0
    cmd.rel_forward_envs = 0.0
    if hasattr(cmd, "rel_straight_envs"):
        cmd.rel_straight_envs = 0.0             # same reason, our own version
    cmd.resampling_time_range = (1e6, 1e6)      # drawn once, never rerolled

    # Hold the posture at nominal unless the caller asked for something else.
    # Added 3 Aug 2026 with the height/pitch/roll command: without this, driving
    # a policy "straight ahead at 0.25" would also be telling it to crouch and
    # lean by amounts nobody chose, and the walked angle this script reports
    # would be measuring that instead.
    posture = env_cfg.commands.get("posture")
    if posture is not None:
        h = posture.nominal_height
        posture.ranges.height = (h, h)
        posture.ranges.pitch = (0.0, 0.0)
        posture.ranges.roll = (0.0, 0.0)
        posture.rel_nominal_envs = 1.0
        posture.resampling_time_range = (1e6, 1e6)

    agent_cfg = load_rl_cfg(args.task)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device="cuda:0"),
                             clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device="cuda:0")
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True,
                map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")

    robot = env.unwrapped.scene["robot"]
    origins = env.unwrapped.scene.env_origins
    model = env.unwrapped.sim.mj_model
    renderer = mujoco.Renderer(model, height=480, width=854)
    mj_data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = (
        args.distance, args.elevation, args.azimuth)

    results = []
    for case in cases:
        cmd.ranges.lin_vel_x = (case["vx"], case["vx"])
        cmd.ranges.lin_vel_y = (case["vy"], case["vy"])
        cmd.ranges.ang_vel_z = (case["yaw"], case["yaw"])

        obs, _ = env.reset()
        start_xy = (robot.data.root_link_pos_w[:, :2] - origins[:, :2]).clone()
        start_h = robot.data.heading_w.clone()
        fell = torch.zeros(args.robots, dtype=torch.bool, device="cuda:0")

        frames, lookat = [], None
        with torch.inference_mode():
            for _ in range(int(args.seconds * 50)):
                obs = env.step(policy(obs))[0]
                fell |= (-robot.data.projected_gravity_b[:, 2]) < 0.5

                sim = env.unwrapped.sim.data
                mj_data.qpos[:] = sim.qpos[0].detach().cpu().numpy()
                mj_data.qvel[:] = sim.qvel[0].detach().cpu().numpy()
                mujoco.mj_forward(model, mj_data)
                target = mj_data.xpos[1].copy()
                # Lag the camera so the START of the walk stays in shot. Locked
                # on the trunk, a diagonal and a straight line look identical.
                if lookat is None:
                    lookat = target.copy()
                else:
                    behind = target[:2] - lookat[:2]
                    dist = float(np.linalg.norm(behind))
                    if dist > 1.4:
                        lookat[:2] += behind * (1.0 - 1.4 / dist)
                    lookat[2] = target[2]
                cam.lookat[:] = lookat
                renderer.update_scene(mj_data, cam)
                frames.append(np.asarray(renderer.render()))

        moved = (robot.data.root_link_pos_w[:, :2] - origins[:, :2]) - start_xy
        cos_h, sin_h = torch.cos(start_h), torch.sin(start_h)
        along = moved[:, 0] * cos_h + moved[:, 1] * sin_h
        across = -moved[:, 0] * sin_h + moved[:, 1] * cos_h
        alive = ~fell
        keep = alive if alive.any() else torch.ones_like(alive)

        a, c = float(along[keep].mean()), float(across[keep].mean())
        got_deg = math.degrees(math.atan2(c, a))
        want_deg = math.degrees(math.atan2(case["vy"], case["vx"]))

        # The SPREAD, not just the mean. Eight robots gave +5.1 deg on one draw
        # and -0.4 deg on the next for the same command, which was read as a
        # constant veer until the second draw disagreed with it. A mean with no
        # spread beside it invites exactly that mistake: it looks like one
        # number when it is the centre of a wide one.
        per_deg = torch.rad2deg(torch.atan2(across[keep], along[keep]))
        spread = float(per_deg.std()) if int(keep.sum()) > 1 else 0.0
        lo, hi = float(per_deg.min()), float(per_deg.max())

        name = label(case)
        dst = out_dir / f"{name}.mp4"
        tmp = dst.with_name(dst.stem + ".writing.mp4")
        imageio.mimwrite(tmp, frames, fps=50, codec="libx264", quality=7,
                         macro_block_size=1)
        tmp.replace(dst)

        # Also drop a copy where the dashboard looks, so the clip is watchable
        # from /runs next to the checkpoint films rather than only on disk.
        # runs.py sorts videos by iteration and files without one land last, so
        # these sit after the training films instead of interleaving with them.
        published = RUNS / log_dir.name / "videos"
        if published.parent.is_dir():
            published.mkdir(parents=True, exist_ok=True)
            (published / f"drive_{name}.mp4").write_bytes(dst.read_bytes())

        row = {
            "case": name,
            "commanded": {"vx": case["vx"], "vy": case["vy"], "yaw": case["yaw"],
                          "angle_deg": round(want_deg, 1)},
            "walked": {"along_m": round(a, 3), "across_m": round(c, 3),
                       "angle_deg": round(got_deg, 1),
                       "angle_sd_deg": round(spread, 1),
                       "angle_min_deg": round(lo, 1),
                       "angle_max_deg": round(hi, 1),
                       "speed_along": round(a / args.seconds, 3),
                       "speed_across": round(c / args.seconds, 3)},
            "fell": int(fell.sum()),
            "of": args.robots,
            "video": str(dst.relative_to(ROOT)).replace("\\", "/"),
        }
        results.append(row)
        print(f"{name:17} told {case['vx']:+.2f},{case['vy']:+.2f} "
              f"({want_deg:+6.1f})   walked {a:+.2f},{c:+.2f} m "
              f"({got_deg:+6.1f} +/-{spread:4.1f}, {lo:+.0f}..{hi:+.0f})   "
              f"fell {int(fell.sum())}/{args.robots}")

    (out_dir / "drive.json").write_text(
        json.dumps({"run": log_dir.name, "checkpoint": ckpt.name,
                    "seconds": args.seconds, "robots": args.robots,
                    "cases": results}, indent=2), encoding="utf-8")
    print(f"\nclips and drive.json in {out_dir.relative_to(ROOT)}")
    env.close()


if __name__ == "__main__":
    main()
