"""Score a trained policy against its stage's pass bar.

    python scripts/verify.py Gray-Stand
    python scripts/verify.py Gray-Push --seconds 30 --robots 128

Training does not answer this on its own. The reward is a weighted sum, which
can read excellently while one term quietly fails, and an episode is shorter
than the bar - so a policy that drifts after twenty seconds still shows a clean
training curve. This measures the things each bar actually names, over the full
duration, across many robots at once.

A stage is passed because the number in the bar was met, not because the curve
went up.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

LOG_ROOT = ROOT / "logs" / "rsl_rl"

# Which log folder each task writes to, and what its bar actually asks for.
TASKS = {
    "Gray-Stand": {"experiment": "gray_stand", "stage": 1, "seconds": 30.0,
                   "bar_survive": 1.00, "bar_err_mm": 5.0, "bar_upright": 0.99},
    # Being shoved every two to four seconds by an unknown amount, on unknown
    # ground, is not something to expect a perfect score against. The bar is nine
    # in ten, and the height tolerance is wider because the robot is allowed to
    # be knocked off its height as long as it comes back.
    "Gray-Push": {"experiment": "gray_push", "stage": 2, "seconds": 20.0,
                  "bar_survive": 0.90, "bar_err_mm": 20.0, "bar_upright": 0.95},
    # Walking is measured differently from the two above. Standing still and
    # taking a shove are judged on holding a pose; walking is judged on GOING
    # SOMEWHERE, so height alone says nothing - a robot that sits down and never
    # moves holds its height perfectly.
    #
    # The bar, from the plan: walks 5 m without falling, holds the commanded
    # speed within 0.05 m/s, and drifts under 100 mm sideways.
    #
    # The command is pinned to one forward speed for the test rather than left to
    # resample: a bar that says "5 m in a straight line" cannot be measured while
    # the robot is being told to turn every few seconds.
    #
    # 25 s, not 20. At the commanded 0.25 m/s, 20 s covers exactly 5.0 m - so the
    # distance check would only pass on PERFECT tracking, making it quietly
    # stricter than the speed check beside it, and any robot at the low edge of
    # its own tolerance would fail a bar it actually met. 25 s means a robot
    # holding 0.20 m/s - the slowest the speed bar allows - still makes the 5 m.
    "Gray-Walk": {"experiment": "gray_walk", "stage": 5, "seconds": 25.0,
                  "bar_survive": 0.90, "bar_err_mm": 40.0, "bar_upright": 0.90,
                  "walk": True,
                  "test_speed": 0.25,        # m/s forward, held for the whole test
                  # Backward is now inside the box, so it has to be inside the
                  # bar too - a range you train over and never check is a range
                  # you are guessing about. Set at -0.35, the EDGE of the box,
                  # on the owner's call, stated twice, rather than in the middle:
                  # a policy can pass forward and fail backward on the asymmetry
                  # alone, and that is the strict direction to be wrong in.
                  #
                  # Wired up 4 Aug 2026. The walk test now runs TWICE, once each
                  # way, and every criterion below is scored on the WORSE of the
                  # two passes. Verify takes about twice as long as a result.
                  "test_speed_back": -0.35,
                  "bar_distance_m": 5.0,
                  "bar_speed_err": 0.05,     # m/s, mean absolute
                  # Straightness is an ANGLE, not a distance. It was written as
                  # "under 100 mm" until 4 Aug 2026, and that was wrong twice
                  # over:
                  #
                  #   It was unreachable. Over the 6.07 m this test covers, 100
                  #   mm sideways is 0.94 degrees of heading held open-loop for
                  #   25 s. The three seeds of round 0 came in at 19.2, 26.9 and
                  #   22.9 degrees, which reads as "21x over the bar" in mm and
                  #   as "20 degrees off" in the unit that describes it.
                  #
                  #   It grew with the stopwatch, and faster than the fault did.
                  #   The measured failure is a steady turn, not a fixed offset:
                  #   under a constant yaw bias b the sideways distance goes as
                  #   b*v*T^2/2 while the ANGLE goes as b*T/2. So doubling the
                  #   test quadrupled the millimetres and only doubles the
                  #   degrees. Degrees is the better unit, NOT an invariant one -
                  #   this bar means 5 deg AT 25 s, and moving `seconds` moves
                  #   what it asks for. The only reading that does not care about
                  #   duration is the yaw bias itself, deg/s, which is
                  #   heading_change_deg over the run length in the per-robot
                  #   dump. Worth switching to if `seconds` ever changes.
                  #
                  # 4.0 degrees, set 4 Aug 2026 on the owner's call, FROM A
                  # MEASUREMENT rather than a guess. It was 5.0 for one day as an
                  # admitted placeholder, chosen before any policy could sense
                  # its own heading and therefore before anyone knew what was
                  # reachable.
                  #
                  # What eight passing runs actually did, once the heading input
                  # existed:
                  #
                  #     3.21  3.23  3.38  3.42  3.47  3.54  3.64  3.75
                  #
                  # against a run-to-run noise of 0.72 deg sd. 4.0 sits about
                  # half a sigma above the worst of them: tight enough that a
                  # real regression fails it, loose enough that a normal run does
                  # not fail on luck. 5.0 left room for the robot to get
                  # noticeably worse and still pass, which is a bar that has
                  # stopped doing its job.
                  #
                  # For the record, what this replaced: "drifts under 100 mm
                  # sideways", which over the 6 m the test covers is 0.94 deg of
                  # heading held for 25 seconds. Nothing could pass it and
                  # nothing ever did.
                  "bar_drift_deg": 7.0,   # 4.0 until 6 Aug 2026 - see below

                  # ---- crab and spin, added 4 Aug 2026 (PLAN.md 1.2) --------
                  #
                  # WHAT THIS FIXES. The command box was widened on 3 Aug to
                  # sideways +/-0.20 m/s and turn +/-1.0 rad/s, and every run
                  # since has trained on both. Nothing has ever MEASURED either.
                  # The test pinned lin_vel_y and ang_vel_z to zero and ran
                  # forward and backward, so two of the four things the policy is
                  # commanded to do were graded by nothing at all - which is the
                  # same hole that let "walks backward" sit in the skill library
                  # for weeks while the real robot covered 0.00 m when asked.
                  #
                  # Both are set at the EDGE of the box, not the middle, on the
                  # same reasoning as test_speed_back: the edge is where a policy
                  # that has only half-learned the command shows it.
                  "test_side": 0.20,         # m/s pure sideways, no forward
                  "test_turn": 1.00,         # rad/s pure spin, no translation
                  #
                  # WHERE THE FIVE NUMBERS BELOW COME FROM. Three of them are
                  # derived from the forward bars, so crab and spin are asked for
                  # the same thing in their own units rather than a fresh guess.
                  # Two are set from measurement. What the three unseen-seed runs
                  # that closed 1.1 - r5a, r5b, r5c - actually score, measured
                  # 4 Aug 2026 with --no-record so it could not overwrite their
                  # verdicts, is written beside each.
                  #
                  # bar_side_distance_m, 3.75 m. Same derivation as the 5 m
                  # forward bar: the slowest speed the speed bar still allows,
                  # held for the whole test. (0.20 - 0.05) * 25 s = 3.75 m.
                  #     r5: 4.52  3.93  4.32   - already clear of it.
                  #
                  # bar_side_speed_err, 0.05 m/s. The same tolerance as forward.
                  # Crabbing is not asked to be more accurate or allowed to be
                  # less; it is the same criterion.
                  #     r5: 0.041  0.036  0.042 - already clear of it.
                  #
                  # bar_side_drift_deg, 5.0 deg since 5 Aug 2026, on the owner's
                  # call. It was 4.0 for a day - the SAME bar as forward, set on
                  # the reasoning that "how far off the line" is one question and
                  # a degree does not care which way the line points. That was my
                  # guess, made before anything had ever measured a crab step,
                  # and I wrote at the time that making it looser needed a
                  # reason. Four attempts later the reason is measured.
                  #
                  # WHAT WAS TRIED AGAINST 4.0, and all of it is in the git log:
                  #   weights      `wandering` -1.0 -> -3.0 -> -6.0. -6.0 measured
                  #                4.98 against -3.0's 4.95. Spent.
                  #   exposure     a pure-sideways share of the draws, 0 -> 0.15.
                  #                Helps by 0.6-0.9 deg. Not enough alone.
                  #   sensing      `off_track`, a 50th input reporting cross-track
                  #                distance. Lost ground on every criterion and
                  #                unbounded nearly stopped the robot walking.
                  #   length       550 -> 1500 iterations. This is what made crab
                  #                drift MEASURABLE - the spread fell from 6.37
                  #                to 0.82 - but the mean barely moved.
                  #
                  # WHERE 5.0 COMES FROM. With the sideways share and 1500
                  # iterations, the three gate seeds score about 3.7, 4.3 and
                  # 4.7. The bar sits just above the worst of them, which is the
                  # same method that set bar_drift_deg at 4.0 once forward drift
                  # became reachable. A bar ON the mean fails half of all future
                  # runs and has stopped doing its job.
                  #
                  # WHY LOOSER THAN FORWARD IS RIGHT, rather than a concession.
                  # Forward steering is observable and correctable: the robot is
                  # off the line because it points wrong, and `off_line` tells it
                  # so. A crab step holds its heading and drifts fore and aft, so
                  # the same input reads near zero while the error grows. And the
                  # authority is not equal either - lateral motion comes from one
                  # hip servo per leg against two joints for fore-aft.
                  #
                  # WHAT IT COSTS, in millimetres rather than degrees. Over the
                  # 5 m this test crabs, 4.0 deg is 350 mm of unwanted fore-aft
                  # drift and 5.0 deg is 437 mm. On the 1 m sidestep the robot
                  # will actually be asked for, the difference is under 20 mm.
                  #
                  # bar_turn_err, 0.20 rad/s. The forward speed bar is 0.05
                  # against a 0.25 m/s command - a fifth of what was asked for.
                  # A fifth of 1.0 rad/s is 0.20.
                  #     r5: 0.510  0.437  0.453 - it turns at about HALF the rate
                  #     it is told to, on every seed. Worst robot 1.15-1.32 rad/s
                  #     of error against a 1.0 command, which is a robot turning
                  #     the wrong way.
                  #
                  # bar_turn_wander_m, 0.10 m. From measurement, because this is
                  # the one the robot already does well and a measured bar is
                  # only honest when there is something passing to measure.
                  #     r5: 0.05  0.03  0.03  - twice the worst of the three,
                  #     which catches a regression without failing on luck.
                  #
                  # ---------------------------------------------------------
                  # FOUR BARS MOVED ON 6 AUG 2026, on the owner's call, after
                  # driving the robot and reading every bar out loud. The
                  # reasoning above is kept because it is how each number was
                  # reached; what changed is not the measurement but what the
                  # robot is FOR. In the owner's words, three of these were
                  # "false tests" - they measure a way the robot will never be
                  # asked to move.
                  #
                  #   drift        4.0 -> 7.0 deg. "I am holding the
                  #                controller. I steer it constantly. A slow
                  #                curve over a hundred metres does not
                  #                matter." Lower is still better; this is
                  #                what counts as passing.
                  #   side_drift   5.0 -> 15.0 deg. "I am not going to be
                  #                crabbing almost at all."
                  #   turn_err     0.20 -> 0.45 rad/s. Against the 1.00 rad/s
                  #                command that is a full circle in about
                  #                11.5 s, and the owner's line was "under 10
                  #                or 11 or 12 seconds is fine, because I will
                  #                be rotating WHILE walking, not on the
                  #                spot".
                  #   turn_wander  0.10 -> 0.50 m. "I only turn one full turn
                  #                at a time, not for 25 seconds."
                  #
                  # The seven not listed did not move. They are the ones that
                  # say it walks at all: staying up, ride height, uprightness,
                  # distance, and the two speed tolerances. A bar loosened
                  # because the test was wrong is a correction; a bar loosened
                  # because the robot cannot reach it is a lie, and none of
                  # these were the second kind - the robot was already inside
                  # three of the four when they moved.
                  "bar_side_distance_m": 3.75,
                  "bar_side_speed_err": 0.05,
                  "bar_side_drift_deg": 15.0,
                  "bar_turn_err": 0.45,      # rad/s, mean absolute
                  "bar_turn_wander_m": 0.50},
}


def trained_under(log_dir: Path) -> dict:
    """What the run was actually trained with, as mjlab recorded it at launch.

    The dump carries python tags for functions, tuples and enums. Reconstructing
    those objects is neither possible nor wanted - unsafe_load dies on the first
    enum it cannot import, and all that is needed here is the numbers. So unknown
    tags are read as the plain structure underneath them.
    """
    import yaml  # noqa: PLC0415

    path = log_dir / "params" / "env.yaml"
    if not path.exists():
        return {}

    class Tolerant(yaml.SafeLoader):
        pass

    def plain(loader, _suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)

    Tolerant.add_multi_constructor("", plain)
    try:
        return yaml.load(path.read_text(), Loader=Tolerant) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[drift] could not read what this run trained with: {exc}")
        return {}


def report_drift(log_dir: Path, cfg) -> list[str]:
    """Say so when the task has changed since the run was trained.

    Scoring a policy against a task that has moved under it is not a verdict on
    the policy - it silently answers a different question. This caught exactly
    that: a run trained on 0.6 m/s box-shaped shoves was scored against 1.2 m/s
    ones from any angle, and 'failed' a bar it had never been trained for.
    """
    was = trained_under(log_dir)
    if not was:
        return []
    drift = []

    def numbers(x):
        """Strip a params tree down to comparable plain numbers."""
        if isinstance(x, dict):
            return {k: numbers(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [numbers(v) for v in x]
        return round(x, 6) if isinstance(x, (int, float)) else str(x)

    old_shove = (was.get("events") or {}).get("shove") or {}
    new_shove = cfg.events.get("shove")
    if old_shove and new_shove is not None:
        old_p, new_p = numbers(old_shove.get("params", {})), numbers(new_shove.params)
        if old_p != new_p:
            drift.append(f"  shove    trained {old_p}")
            drift.append(f"           testing {new_p}")

    old_rew = was.get("rewards") or {}
    for name, term in cfg.rewards.items():
        before = (old_rew.get(name) or {}).get("weight")
        if isinstance(before, (int, float)) and abs(before - term.weight) > 1e-9:
            drift.append(f"  {name:<9}trained {before}   testing {term.weight}")
    for name in old_rew:
        if name not in cfg.rewards:
            drift.append(f"  {name:<9}was a scoring term then, and is not now")

    # Weights are not the only thing that moves a task, and until 6 Aug 2026
    # they were the only thing compared - the gait work changes reward PARAMS
    # (the swing target), TERMINATIONS (a dive ends the attempt) and the
    # command DRAW (the spin share), and every one of those would have scored
    # an older policy against a changed task without a word of warning.
    def plain(params: dict) -> dict:
        """Only the plain numbers. Strings, asset configs and sensor handles
        serialise differently between the live config and the yaml record, so
        comparing them reports drift that never happened."""
        keep = {}
        for k, v in dict(params).items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                keep[k] = round(float(v), 6)
            elif (isinstance(v, (list, tuple)) and v
                    and all(isinstance(x, (int, float)) for x in v)):
                keep[k] = [round(float(x), 6) for x in v]
        return keep

    for name, term in cfg.rewards.items():
        old_p = plain((old_rew.get(name) or {}).get("params") or {})
        new_p = plain(term.params)
        moved = {k for k in old_p if k in new_p and old_p[k] != new_p[k]}
        if moved:
            for k in sorted(moved):
                drift.append(f"  {name:<9}{k}: trained {old_p[k]}   "
                             f"testing {new_p[k]}")

    old_term = set(was.get("terminations") or {})
    new_term = set(cfg.terminations)
    if old_term:
        for name in sorted(new_term - old_term):
            drift.append(f"  {name:<9}ends an attempt now, and did not when "
                         f"this run trained")
        for name in sorted(old_term - new_term):
            drift.append(f"  {name:<9}ended an attempt then, and does not now")

    # The draw mix and the box. rel_standing_envs is NOT compared - main() has
    # already zeroed it for the test by the time this runs, so comparing it
    # would cry drift on every single run.
    old_walk = (was.get("commands") or {}).get("walk") or {}
    new_walk = cfg.commands.get("walk")
    if old_walk and new_walk is not None:
        for field in ("rel_straight_envs", "rel_crab_envs", "rel_spin_envs",
                      "straight_min_speed", "crab_min_speed", "spin_min_rate"):
            before, now = old_walk.get(field), getattr(new_walk, field, None)
            if (isinstance(before, (int, float)) and not isinstance(before, bool)
                    and isinstance(now, (int, float))
                    and abs(before - now) > 1e-9):
                drift.append(f"  draw     {field}: trained {before}   "
                             f"testing {now}")
        old_box = old_walk.get("ranges") or {}
        for field in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            before = old_box.get(field)
            now = getattr(getattr(new_walk, "ranges", None), field, None)
            if before is not None and now is not None \
                    and numbers(before) != numbers(tuple(now)):
                drift.append(f"  box      {field}: trained {numbers(before)}   "
                             f"testing {numbers(tuple(now))}")

    if drift:
        print("WARNING - the task has changed since this run was trained:")
        print("\n".join(drift))
        print("  The result below answers how this policy copes with today's task,")
        print("  not whether it passed the one it was trained for.\n")
    # Returned as well as printed. Until now this only reached a job log nobody
    # reads, so a verdict measured against a task that had moved sat unmarked
    # beside verdicts that had not - and walk_m3100_b is exactly that case.
    return drift


# How often the trunk's position is written down during a pass, in control steps.
# 50 steps is one second, so a 25 s pass leaves 26 marks along the floor.
#
# Until 4 Aug 2026 there were TWO: where it started and where it stopped. Two
# points always describe a straight line, so the test could not tell a robot
# walking a steady 20 degrees off from one curving away from one that lurched
# once and then walked true - and those three have three different causes. It
# also meant a robot that swung half a metre out and came back scored zero drift.
SAMPLE_EVERY = 50

# Nose-down beyond this is counted as a dive, in radians. 0.20 rad is 11.5
# degrees - past every pitch the posture command can ask for during a test
# (verify pins it level), and past the wobble of a clean gait, but well short
# of the 45.8 degree tilt that ends an episode. NOSE DOWN IS NEGATIVE,
# measured by scripts/measure_pitch_sign.py on 6 Aug 2026.
DIVE_RAD = 0.20


def run_pass(env, robot, origins, policy, torch, *, seconds, robots, target,
             device, leg=None, cmd=None, diag=False):
    """One measured episode, handed back unreduced.

    Nothing is averaged in here on purpose. The walk test runs this four times -
    forward, backward, sideways, spinning - and has to score the WORSE of the
    two straight-line passes on every criterion they share. That is only
    possible if every pass returns the raw per-robot numbers and the scoring
    rules live in exactly one place downstream, rather than being written out
    once per direction and drifting apart.

    `leg` is the command being held: {label, vx, vy, wz, kind}. It is one dict
    rather than three arguments because the pinning below has to set all three
    ranges together - pinning lin_vel_x and leaving lin_vel_y at the trained
    +/-0.20 would have every robot crabbing at a random speed during what is
    supposed to be a straight-line test.
    """
    if cmd is not None and leg is not None:
        # The command term reads its ranges at resample time, and resample
        # happens on reset - so setting them here and resetting below is what
        # makes each pass a different test rather than a repeat.
        cmd.ranges.lin_vel_x = (leg["vx"], leg["vx"])
        cmd.ranges.lin_vel_y = (leg["vy"], leg["vy"])
        cmd.ranges.ang_vel_z = (leg["wz"], leg["wz"])
        # THIS LINE IS WHY THE CRAB AND SPIN PASSES MEASURE ANYTHING.
        # `rel_straight_envs` REDRAWS lin_vel_x and then zeroes lin_vel_y and
        # ang_vel_z - see gray/tasks/walk_command.py. Left at 1.0 it would wipe
        # the sideways and turn commands out before the policy ever saw them,
        # and both passes would silently be a third and fourth copy of "stand
        # still going forwards at 0".
        cmd.rel_straight_envs = 1.0 if leg["kind"] == "straight" else 0.0

    # The gait collection, only under --gait-diag. Four things a film shows and
    # no recorded number does: SIGNED pitch (error_pitch is an absolute value,
    # so "nose down" was invisible to every chart), the height each swing
    # actually peaks at (the reward stores only its cost), dives, and when the
    # first fall comes - training is 20 s long, and stable-inside-the-horizon
    # is not stable.
    gait = None
    if diag:
        from gray.tasks.posture_command import trunk_pitch_roll  # noqa: PLC0415

        feet = env.unwrapped.scene["feet"]
        foot_ids, foot_names = robot.find_sites(".*_foot")
        zeros4 = torch.zeros(robots, len(foot_ids), device=device)
        gait = {
            "trunk_pitch_roll": trunk_pitch_roll,
            "feet": feet, "foot_ids": foot_ids, "foot_names": foot_names,
            # The same latch swing_height in the task keeps: track a foot's
            # peak while it is airborne, bank it on the step it lands.
            "peak": zeros4.clone(), "peak_sum": zeros4.clone(),
            "peak_n": zeros4.clone(), "peak_max": zeros4.clone(),
            "pitch_sum": torch.zeros(robots, device=device),
            "pitch_min": torch.full((robots,), 99.0, device=device),
            "pitch_max": torch.full((robots,), -99.0, device=device),
            "dive_steps": torch.zeros(robots, device=device),
            "first_fall": torch.full((robots,), -1.0, device=device),
        }

    obs, _ = env.reset()
    heights, uprights, path = [], [], []
    # Body-frame velocity along x and y, and yaw rate, every step. All three are
    # kept for every pass rather than the one the pass is "about": the crab pass
    # needs y, the spin pass needs yaw, and keeping them together is what lets
    # score_pass hold the scoring rules in one place instead of one branch per
    # direction. It is three floats per robot per step - about 1 MB for the whole
    # test - so there is nothing to save by being clever.
    vx_b, vy_b, wz_b = [], [], []
    fell = torch.zeros(robots, dtype=torch.bool, device=device)
    start_xy = (robot.data.root_link_pos_w[:, :2] - origins[:, :2]).clone()
    # The heading each robot STARTED on. Drift is measured against this, not
    # against world X: reset nudges the spawn yaw by up to +/-0.1 rad, and a
    # robot walking perfectly straight from a 0.1 rad start would show 0.6 m of
    # world-frame lateral offset over 6 m. That is the test's randomisation, not
    # the policy's error, and charging the policy for it makes the bar unfair by
    # an amount nobody could see.
    start_heading = robot.data.heading_w.clone()
    path.append(start_xy.clone())

    # Heights are measured above the LOCAL ground, not the env origin - on
    # the flat floor the two are the same number, on the slope batches only
    # the first one means anything. Same table read the task's terms use.
    from gray.tasks.walk_env_cfg import ground_height_under  # noqa: PLC0415

    with torch.inference_mode():
        for step in range(int(seconds * 50)):
            obs = env.step(policy(obs))[0]
            root = robot.data.root_link_pos_w
            h = root[:, 2] - ground_height_under(env.unwrapped, root[:, :2])
            up = -robot.data.projected_gravity_b[:, 2]   # 1.0 is dead level
            heights.append(h.clone())
            uprights.append(up.clone())
            if leg is not None:
                vx_b.append(robot.data.root_link_lin_vel_b[:, 0].clone())
                vy_b.append(robot.data.root_link_lin_vel_b[:, 1].clone())
                wz_b.append(robot.data.root_link_ang_vel_b[:, 2].clone())
                if (step + 1) % SAMPLE_EVERY == 0:
                    path.append(
                        (robot.data.root_link_pos_w[:, :2] - origins[:, :2]).clone())
            fell |= (up < 0.7) | (h < target * 0.55)
            if gait is not None:
                pitch, _roll = gait["trunk_pitch_roll"](robot)
                gait["pitch_sum"] += pitch
                gait["pitch_min"] = torch.minimum(gait["pitch_min"], pitch)
                gait["pitch_max"] = torch.maximum(gait["pitch_max"], pitch)
                gait["dive_steps"] += (pitch < -DIVE_RAD).float()
                gait["first_fall"] = torch.where(
                    fell & (gait["first_fall"] < 0),
                    torch.full_like(gait["first_fall"], (step + 1) / 50.0),
                    gait["first_fall"])
                sites = robot.data.site_pos_w[:, gait["foot_ids"], :]
                feet_h = sites[..., 2] - ground_height_under(
                    env.unwrapped, sites[..., :2])
                in_air = gait["feet"].data.found == 0
                gait["peak"] = torch.where(
                    in_air, torch.maximum(gait["peak"], feet_h), gait["peak"])
                # A pure read of the air-time state - the task's own terms are
                # its writers, and they run inside env.step above.
                landed = gait["feet"].compute_first_contact(
                    dt=env.unwrapped.step_dt)
                gait["peak_sum"] += gait["peak"] * landed.float()
                gait["peak_n"] += landed.float()
                gait["peak_max"] = torch.maximum(
                    gait["peak_max"], gait["peak"] * landed.float())
                gait["peak"] = torch.where(
                    landed, torch.zeros_like(gait["peak"]), gait["peak"])

    end_xy = (robot.data.root_link_pos_w[:, :2] - origins[:, :2]).clone()
    return {
        "gait": gait,
        "heights": torch.stack(heights),
        "uprights": torch.stack(uprights),
        "vx_b": torch.stack(vx_b) if vx_b else None,
        "vy_b": torch.stack(vy_b) if vy_b else None,
        "wz_b": torch.stack(wz_b) if wz_b else None,
        "fell": fell,
        "start_xy": start_xy,
        "end_xy": end_xy,
        "start_heading": start_heading,
        "end_heading": robot.data.heading_w.clone(),
        "path": torch.stack(path) if len(path) > 1 else None,
        "leg": leg,
    }


def score_pass(raw, spec, torch, *, seconds, robots, target, label=""):
    """Turn one pass's raw numbers into the criterion rows, and the per-robot dump.

    Returns (checks, per_robot). `checks` rows are
    (key, name, measured, bar, direction, worst, format, unit, note).

    `key` is a STABLE id and is deliberately not derived from `name`. The name
    has the bar number baked into it - "trunk within 40 mm of target" becomes
    "within 5 mm" the moment the tolerance changes - so it can never be used to
    match the same criterion across two runs. The key can.
    """
    fell = raw["fell"]
    alive = ~fell
    any_alive = bool(alive.any())
    survived = float(alive.float().mean())

    h, up = raw["heights"], raw["uprights"]
    err_mm = ((h[:, alive] - target).abs() * 1000) if any_alive else h.abs() * 1e6
    up_alive = up[:, alive] if any_alive else up

    tag = f", {label}" if label else ""
    checks = [
        ("survive", f"stayed up for {seconds:.0f} s", survived,
         spec["bar_survive"], "ge",
         float(robots - int(fell.sum())) / robots,
         "{:.0%}", "fraction", f"{int(fell.sum())} of {robots} fell{tag}"),
        ("height_err", f"trunk within {spec['bar_err_mm']:.0f} mm of target",
         float(err_mm.mean()), spec["bar_err_mm"], "le",
         float(err_mm.max()), "{:.2f} mm", "mm",
         f"worst {float(err_mm.max()):.1f} mm{tag}"),
        ("upright", f"uprightness above {spec['bar_upright']}",
         float(up_alive.mean()), spec["bar_upright"], "ge",
         float(up_alive.min()), "{:.4f}", "ratio",
         f"worst {float(up_alive.min()):.4f}{tag}"),
    ]
    per_robot = {}

    # Walking has more, and they are the ones that actually say it walked.
    # Measured only on the robots still standing: a robot that fell at second two
    # travelled no further, and averaging it in would report the FALL as a speed
    # error rather than as the fall it already is.
    leg = raw["leg"]
    if leg is None or raw["vx_b"] is None:
        return checks, per_robot

    keep = alive if any_alive else torch.ones_like(alive)

    # Rotate the displacement into each robot's own starting frame, so "forward"
    # means the way it was pointing and "sideways" means off that line - rather
    # than both being measured against an arbitrary world axis.
    moved = raw["end_xy"] - raw["start_xy"]
    cos_h, sin_h = torch.cos(raw["start_heading"]), torch.sin(raw["start_heading"])
    along = moved[:, 0] * cos_h + moved[:, 1] * sin_h
    across = -moved[:, 0] * sin_h + moved[:, 1] * cos_h

    # Wrapped. A robot that turns through more than half a circle is not turning
    # a small amount the other way, and the raw difference says it is: round 0
    # already has robots at -127 deg, so the unwrapped version was one bad run
    # away from reporting the opposite of what happened.
    turned = raw["end_heading"] - raw["start_heading"]
    head_err_deg = torch.rad2deg(torch.atan2(torch.sin(turned), torch.cos(turned)))

    if leg["kind"] == "spin":
        # A SPIN HAS NO LINE TO HOLD, so distance, speed and drift do not apply
        # and are not reported - a "covered 5 m" row on a robot told to turn on
        # the spot would be a bar it is meant to fail. What is being asked is
        # "did it turn at the rate it was told, without wandering off", so that
        # is what the two rows below say.
        #
        # Rate rather than total angle turned on purpose. Over 25 s at 1.0 rad/s
        # the robot goes round exactly four times, and heading only ever reads
        # modulo one turn - so a total is unmeasurable without integrating, and
        # the integral of the rate is the rate. A robot spinning at half speed
        # shows as 0.5 rad/s of error every step, which is the fault itself.
        turn_err = (raw["wz_b"][:, keep] - leg["wz"]).abs()
        # How far it slid while spinning. A quadruped told to turn in place and
        # nothing else should end roughly where it started; the trained reward
        # has no term for this at all, so it is worth measuring before assuming.
        wander = torch.sqrt(along ** 2 + across ** 2)[keep]
        checks += [
            ("turn_err",
             f"turn rate within {spec['bar_turn_err']:.2f} rad/s of the command",
             float(turn_err.mean()), spec["bar_turn_err"], "le",
             float(turn_err.max()), "{:.3f} rad/s", "rad/s",
             f"worst {float(turn_err.max()):.3f} rad/s{tag}"),
        ]
        # "Stayed on the spot" only means something if the robot was told to stay
        # on it. With --test-turn-speed the leg is an ARC: the robot is ordered
        # to travel, so distance from the start is obedience, not error, and the
        # check is dropped rather than reported as a huge failure. Turn rate is
        # still measured, and it is the number the arc exists to get.
        if not leg["vx"] and not leg["vy"]:
            checks.append(
                ("turn_wander",
                 f"stayed within {spec['bar_turn_wander_m']:.2f} m of the spot",
                 float(wander.mean()), spec["bar_turn_wander_m"], "le",
                 float(wander.max()), "{:.2f} m", "m",
                 f"worst {float(wander.max()):.2f} m{tag}"))
        per_robot = {
            "command": [leg["vx"], leg["vy"], leg["wz"]],
            "alive": [bool(x) for x in alive.tolist()],
            "turn_rate_mean": [round(x, 4)
                               for x in raw["wz_b"].mean(dim=0).tolist()],
            "wander_m": [round(x, 4)
                         for x in torch.sqrt(along ** 2 + across ** 2).tolist()],
            "heading_change_deg": [round(x, 3) for x in head_err_deg.tolist()],
        }
        return checks, per_robot

    # ---- a straight line, in whatever direction it was pointed ------------
    #
    # One set of rules for forward, backward and sideways, rather than three.
    # The commanded direction is a UNIT VECTOR in the robot's own frame, so
    # "progress" is distance covered the way it was told and "off the line" is
    # the component at right angles to that - and both mean the same thing
    # whether the command was 0.25 ahead, 0.35 behind or 0.20 to the left.
    #
    # Checked against what this replaced: forward gives (1, 0), so progress is
    # `along` and the off-line term is `across`, exactly as before. Backward
    # gives (-1, 0), so progress is `-along`, which is the old sign flip.
    speed = (leg["vx"] ** 2 + leg["vy"] ** 2) ** 0.5
    ux, uy = leg["vx"] / speed, leg["vy"] / speed
    progress = along * ux + across * uy
    off_line = -along * uy + across * ux
    angle_deg = torch.rad2deg(torch.atan2(off_line, progress))
    # Speed along the commanded direction, from the body-frame velocity - not
    # the distance divided by the clock, which would hide a robot that lurched
    # forward and stopped.
    v_along = raw["vx_b"] * ux + raw["vy_b"] * uy
    speed_err = (v_along[:, keep] - speed).abs()

    fwd, ang = progress[keep], angle_deg[keep]

    # Crab is its own three bars, under its own keys. Merging them into the
    # forward ones would take the worse of two things that are not comparable:
    # 5 m at 0.25 m/s forward and 5 m at 0.20 m/s sideways are different asks,
    # and one number covering both could only ever report the harder one.
    side = leg["kind"] == "crab"
    pre = "side_" if side else ""
    bar_d = spec["bar_side_distance_m" if side else "bar_distance_m"]
    bar_s = spec["bar_side_speed_err" if side else "bar_speed_err"]
    bar_a = spec["bar_side_drift_deg" if side else "bar_drift_deg"]
    what = "crabbed" if side else "covered"

    # How many ended up on the SAME SIDE of the commanded line. That question is
    # the whole reason the count is printed: all robots off one way is a gait
    # that is not symmetric, an even spread either way is a heading that
    # random-walks because nothing observes it, and the two want opposite fixes.
    #
    # Off the LINE, not the robot's own left. For the forward pass they are the
    # same thing, which is what this used to say. For the crab pass they are not:
    # "left" is the direction it was TOLD to go, so every robot obeying the
    # command counted as off to the left and the row read "64 of 64 left" when
    # nothing was wrong with any of them.
    one_side = int((off_line[keep] > 0).sum())

    checks += [
        (f"{pre}distance", f"{what} {bar_d:.1f} m",
         float(fwd.mean()), bar_d, "ge",
         float(fwd.min()), "{:.2f} m", "m",
         f"worst {float(fwd.min()):.2f} m{tag}"),
        (f"{pre}speed_err",
         f"speed within {bar_s:.2f} m/s of the command",
         float(speed_err.mean()), bar_s, "le",
         float(speed_err.max()), "{:.3f} m/s", "m/s",
         f"worst {float(speed_err.max()):.3f} m/s{tag}"),
        (f"{pre}drift", f"under {bar_a:.1f} deg off the line",
         float(ang.abs().mean()), bar_a, "le",
         float(ang.abs().max()), "{:.2f} deg", "deg",
         f"worst {float(ang.abs().max()):.1f} deg, "
         f"{one_side} of {int(keep.sum())} the same way{tag}"),
    ]

    # Every one of these was already computed and then thrown away by a .mean().
    # Keeping them is what makes "is this a bias or a spread?" answerable at all,
    # and that question decides whether the next fix is the gait or the reward.
    per_robot = {
        "command": [leg["vx"], leg["vy"], leg["wz"]],
        "test_speed": leg["vx"],
        "alive": [bool(x) for x in alive.tolist()],
        "along_m": [round(x, 4) for x in along.tolist()],
        "across_m": [round(x, 4) for x in across.tolist()],
        "progress_m": [round(x, 4) for x in progress.tolist()],
        "angle_deg": [round(x, 3) for x in angle_deg.tolist()],
        "heading_change_deg": [round(x, 3) for x in head_err_deg.tolist()],
    }
    return checks, per_robot


def worse_of(rows_a, rows_b):
    """Merge two passes' criterion rows, keeping the worse of each pair.

    Worse means "further the wrong way for THAT criterion" - lower for a `ge`
    bar, higher for a `le` one - so a policy that walks forward beautifully and
    backward badly is reported as the second thing. Taking the mean of the two
    would let one direction pay for the other, which is exactly the asymmetry
    the backward pass was added to catch.
    """
    by_key = {row[0]: row for row in rows_b}
    out = []
    for row in rows_a:
        other = by_key.get(row[0])
        if other is None:
            out.append(row)
            continue
        how = row[4]
        keep_a = (row[2] <= other[2]) if how == "ge" else (row[2] >= other[2])
        out.append(row if keep_a else other)
    # Rows only the LATER pass has are kept, not dropped. Every pass used to
    # measure the same six things, so this could not come up; the crab and spin
    # passes have criteria the forward pass does not, and without this line
    # `side_distance`, `turn_err` and the rest would be computed, printed by
    # their own pass, and then silently thrown away before anything was scored.
    seen = {row[0] for row in rows_a}
    out += [row for row in rows_b if row[0] not in seen]
    return out


def gait_stats(raw, torch, *, seconds) -> dict:
    """Reduce one pass's gait collection to per-robot numbers, and say them.

    Signs are kept - that is the entire point. `error_pitch` in the metrics is
    an absolute value, so nothing recorded before this could show which WAY
    the trunk pitched. Nose down is NEGATIVE (measured, see
    scripts/measure_pitch_sign.py).
    """
    g = raw["gait"]
    steps = float(seconds * 50)
    pitch_mean = g["pitch_sum"] / steps
    peak_mean = g["peak_sum"] / torch.clamp(g["peak_n"], min=1.0)
    trunk_min = raw["heights"].min(dim=0).values
    fell_n = int(raw["fell"].sum())
    first = g["first_fall"][g["first_fall"] > 0]

    def deg(x: float) -> str:
        return f"{x * 57.2958:+.1f} deg"

    dived = int((g["dive_steps"] > 0).sum())
    print(f"    pitch       mean {deg(float(pitch_mean.mean()))}   "
          f"worst dive {deg(float(g['pitch_min'].min()))}   "
          f"(nose down is negative)")
    print(f"    swing peak  mean {float(peak_mean.mean()) * 1000:.1f} mm   "
          f"highest {float(g['peak_max'].max()) * 1000:.1f} mm")
    print(f"    dives       {dived} of {len(trunk_min)} robots past "
          f"{DIVE_RAD * 57.2958:.1f} deg nose-down; worst spent "
          f"{int(g['dive_steps'].max())} steps there")
    print(f"    trunk       lowest {float(trunk_min.min()) * 1000:.1f} mm")
    print(f"    falls       {fell_n} of {len(trunk_min)}"
          + (f", earliest at {float(first.min()):.1f} s" if len(first) else ""))

    return {
        "pitch_mean_rad": [round(x, 4) for x in pitch_mean.tolist()],
        "pitch_min_rad": [round(x, 4) for x in g["pitch_min"].tolist()],
        "pitch_max_rad": [round(x, 4) for x in g["pitch_max"].tolist()],
        "dive_steps": [int(x) for x in g["dive_steps"].tolist()],
        "dive_threshold_rad": DIVE_RAD,
        "first_fall_s": [round(x, 2) for x in g["first_fall"].tolist()],
        "min_trunk_height_m": [round(x, 4) for x in trunk_min.tolist()],
        "feet": list(g["foot_names"]),
        "swing_peak_mean_m": [[round(x, 4) for x in row]
                              for row in peak_mean.tolist()],
        "swing_peak_max_m": [[round(x, 4) for x in row]
                             for row in g["peak_max"].tolist()],
        "swing_count": [[int(x) for x in row] for row in g["peak_n"].tolist()],
    }


def _dump_diagnostic(root, run_id, switched_off, per_robot, paths, ckpt,
                     robots, seconds) -> None:
    """Write a diagnostic run's numbers somewhere they cannot be mistaken for a verdict.

    Its own file, named after what was switched off, outside progress/runs/ - so
    no dashboard page picks it up and shows it as a result.
    """
    import json  # noqa: PLC0415

    out = root / "progress" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    if "servo_strength" in switched_off:
        parts.append("same-robot")
    if "nudge_base" in switched_off:
        parts.append("same-start")
    if "norecord" in switched_off:
        parts.append("measured")
    if "gait" in switched_off:
        parts.append("gait")
    parts += [t for t in switched_off if t.startswith("slope")]
    path = out / f"{run_id}__{'_'.join(parts) or 'off'}.json"
    path.write_text(json.dumps({
        "run": run_id, "checkpoint": str(ckpt).replace("\\", "/"),
        "robots": robots, "seconds": seconds,
        "switched_off": switched_off,
        "per_robot": per_robot, "paths": paths}, indent=1))
    print(f"diagnostic written: {path.relative_to(root)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", nargs="?", default="Gray-Stand", choices=sorted(TASKS))
    ap.add_argument("--run", help="run folder; default is the newest")
    ap.add_argument("--checkpoint", help="e.g. model_599.pt; default is the last")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 uses the bar's own")
    ap.add_argument("--robots", type=int, default=64)
    # Two diagnostic switches. Neither is part of the bar - a run scored with
    # either of them on is NOT recorded, because passing in an easier world is
    # not passing. They exist to answer one question the pass/fail number cannot:
    # WHERE the 23 deg spread in heading comes from.
    ap.add_argument("--same-robot", action="store_true",
                    help="every robot identical: no friction, mass, centre of "
                         "mass, servo gain or joint friction draw, and no shoves")
    ap.add_argument("--same-start", action="store_true",
                    help="every robot starts in the same pose and heading: no "
                         "spawn nudge")
    # Measure without judging. The bar is the world as it is, so this is NOT a
    # diagnostic switch - nothing is made easier and every number is real. What
    # it does is refuse to write a verdict onto the run.
    #
    # It exists because of the order the crab and spin bars had to be set in. A
    # bar has to come from a measurement, and the only policies to measure were
    # r5a/r5b/r5c - the three that had already PASSED and closed 1.1. Re-scoring
    # them against a criterion that did not exist when they trained would have
    # overwritten three "passed" verdicts with a "not passed" against a bar that
    # was still a placeholder at the time, and the record of 1.1 closing would
    # have been destroyed to find out a number.
    ap.add_argument("--no-record", action="store_true",
                    help="print the result and write nothing to the run - for "
                         "measuring a policy against a bar that does not exist "
                         "yet")
    # WHERE the test is taken, as opposed to what it has to reach. A bar says
    # how well the robot must turn; this says how fast it is asked to. They were
    # the same edit until 5 Aug 2026, and that hid the fault this exists to
    # find: `test_turn` sat at 1.00 rad/s, the exact edge of WALK_TURN, so the
    # policy was scored at the one turn rate it had almost never practised -
    # about 3.5% of draws. A bar cannot be judged sensible until the point it
    # is measured at can be moved on its own.
    #
    # FORCES --no-record. A verdict written against a moved test point would
    # claim the stage bar while measuring something else, and the run record
    # would carry no sign of it.
    ap.add_argument("--test-turn", type=float,
                    help="rad/s for the turning pass, instead of the task's. "
                         "Implies --no-record.")
    ap.add_argument("--test-side", type=float,
                    help="m/s for the sideways pass, instead of the task's. "
                         "Implies --no-record.")
    ap.add_argument("--test-turn-speed", type=float,
                    help="m/s forward DURING the turning pass, making it an arc "
                         "instead of a spin from standstill. The bar uses 0. "
                         "Implies --no-record.")
    # The gait, measured. Added 6 Aug 2026, the day the owner's film beat the
    # instruments: feet that barely lift, a nose that points down, a collapse a
    # short way into a walk - and not one recorded number could show any of it.
    # error_pitch is an absolute value, the reward stores only the cost of a
    # swing and not its height, and nothing anywhere writes down WHEN a robot
    # fell, only whether. These two switches are those instruments.
    ap.add_argument("--gait-diag", action="store_true",
                    help="collect signed pitch, swing peak heights, dives and "
                         "first-fall times per pass, print them, and write "
                         "them into the diagnostic file. Implies --no-record.")
    ap.add_argument("--ladder", action="store_true",
                    help="replace the four passes with a forward speed ladder "
                         "- 0.15 / 0.25 / 0.35, plus 0.45 which is OUTSIDE "
                         "the trained box and labelled as such - to find the "
                         "speed the dive starts at. Implies --gait-diag.")
    ap.add_argument("--slope-deg", type=float, default=0.0,
                    help="measure on the slope world instead of the flat "
                         "floor - the same 16 m pyramid train.py --slope-deg "
                         "builds. Implies --no-record: the stage bar is a "
                         "flat-floor bar until the plan says otherwise.")
    args = ap.parse_args()

    spec = dict(TASKS[args.task])
    spec.setdefault("test_turn_speed", 0.0)
    moved = []
    for key, val in (("test_turn", args.test_turn), ("test_side", args.test_side),
                     ("test_turn_speed", args.test_turn_speed)):
        if val is not None:
            moved.append(f"{key} {spec[key]} -> {val}")
            spec[key] = val
    if moved:
        args.no_record = True
        print(f"test point   {' and '.join(moved)} - not recorded")
    if args.ladder:
        args.gait_diag = True
    if args.gait_diag:
        args.no_record = True
        print("gait diagnostic - measured, not judged, not recorded")
    if args.slope_deg:
        args.no_record = True
        print(f"terrain      a {args.slope_deg:g} deg slope - not recorded")
    seconds = args.seconds or spec["seconds"]
    exp_root = LOG_ROOT / spec["experiment"]
    if not exp_root.is_dir():
        raise SystemExit(f"no runs under {exp_root}")
    log_dir = (exp_root / args.run if args.run
               else max(exp_root.iterdir(), key=lambda p: p.stat().st_mtime))
    ckpts = sorted(log_dir.glob("model_*.pt"),
                   key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {log_dir}")
    ckpt = (log_dir / args.checkpoint) if args.checkpoint else ckpts[-1]

    import torch  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    import gray.tasks  # noqa: F401,PLC0415
    from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: PLC0415

    target = float(yaml.safe_load(
        (ROOT / "progress" / "stance" / "stance.yaml").read_text())["trunk_height_m"])

    # Play mode turns off observation noise, but the disturbances stay: a policy
    # that only survives when nothing pushes it has not passed this stage.
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.robots
    env_cfg.episode_length_s = seconds + 5.0
    if args.slope_deg:
        from gray.tasks.walk_env_cfg import apply_slope  # noqa: PLC0415

        apply_slope(env_cfg, args.slope_deg)

    # Give the policy the observation it was TRAINED with, not today's.
    #
    # A saved policy is a fixed-size mapping. The task gained `off_line` - which
    # way the robot is off the line it was sent along - on 4 Aug 2026, and any
    # run trained without it is 48 numbers wide against the task's 49. Loading
    # one into the other does not degrade, it raises: "size mismatch for
    # mlp.0.weight, [256, 48] vs [256, 49]", eight seconds in, after the training
    # that produced it has already finished. Three ablation runs died that way
    # before this existed.
    #
    # train.py records the actor's term names on the run. Trusting that rather
    # than guessing from the checkpoint's width means this also covers the next
    # observation change, and says plainly when a run needs something the task no
    # longer offers - which is the one case that genuinely cannot be scored.
    trained_obs = None
    try:
        import json  # noqa: PLC0415

        meta = json.loads(
            (ROOT / "progress" / "runs" / log_dir.name / "run.json").read_text())
        trained_obs = meta.get("observes") or None
    except Exception:  # noqa: BLE001
        pass
    if trained_obs:
        for group in ("actor", "critic"):
            terms = getattr(env_cfg.observations.get(group), "terms", None)
            if terms is None:
                continue
            for name in [n for n in terms if n not in trained_obs]:
                # The critic is not loaded here, but keeping the two groups in
                # step costs nothing and stops a future critic-side check from
                # tripping over a mismatch this already knows about.
                terms.pop(name)
                if group == "actor":
                    print(f"observation   {name} removed - this run trained "
                          f"without it")
        missing = [n for n in trained_obs
                   if n not in (getattr(env_cfg.observations.get("actor"),
                                        "terms", {}) or {})]
        if missing:
            raise SystemExit(
                f"this run trained on observations the task no longer has: "
                f"{', '.join(missing)}. Its policy cannot be scored against "
                f"today's task - the inputs it expects do not exist.")

    # The diagnostic switches, applied before the world is built.
    #
    # WHY THESE EXIST. Round 0 ended 19-31 deg off the line with a spread of
    # 20-29 deg across 64 robots given the identical command. Two explanations
    # fit that equally well and want opposite fixes:
    #
    #   The robot is not the same robot twice. Every reset redraws foot friction
    #   0.4-1.2, mass +/-20%, centre of mass +/-15 mm, per-joint servo gain
    #   +/-30% and joint friction - and a lopsided gain draw is a constant
    #   turning torque for that whole episode. The policy is told its turn RATE
    #   but never which way it is now pointing, so it damps the wobble and lets
    #   the steady part integrate for 25 s. If that is the story, the spread is
    #   the world's, and the fix is to let the policy sense its heading.
    #
    #   Or the gait is simply crooked, and would be crooked on one fixed robot.
    #   Then a heading input buys much less and the gait is what needs work.
    #
    # Switching the draws off separates them, in minutes, with no training.
    switched_off = []
    if args.same_robot:
        switched_off += ["ground_grip", "how_heavy", "where_the_weight_is",
                         "servo_strength", "gearbox_drag", "shove"]
    if args.same_start:
        switched_off += ["nudge_pose", "nudge_base"]
    switched_off = [n for n in switched_off if env_cfg.events.pop(n, None) is not None]
    if switched_off:
        print("DIAGNOSTIC RUN - not a verdict. Switched off: "
              + ", ".join(switched_off))
        print("  The bar assumes an unknown robot on unknown ground. This is an "
              "easier world,\n  so whatever it scores is a measurement and not a "
              "pass.\n")

    # Walking is tested against ONE held command, not the random stream it trained
    # on. "5 m in a straight line" is not measurable while the robot is being told
    # to turn every few seconds, and a resample mid-test would show up as a speed
    # error the policy never made.
    cmd = None
    if spec.get("walk"):
        cmd = env_cfg.commands.get("walk")
        if cmd is None:
            raise SystemExit("Gray-Walk has no 'walk' command to pin for the test")
        # All three ranges, and rel_straight_envs, are set per pass by run_pass -
        # there are four passes now and two of them are NOT straight lines.
        # Pinning them here was what made the crab and spin passes impossible:
        # a single rel_straight_envs = 1.0 zeroes sideways and turn for every
        # robot, so a "turn at 1.0 rad/s" test would have measured a robot
        # standing still and reported it as a pass.
        cmd.rel_standing_envs = 0.0      # nobody is told to stand still
        cmd.rel_forward_envs = 0.0       # mjlab's version forces the speed
        # Longer than the test, so it is drawn once and never changes under us.
        cmd.resampling_time_range = (1e6, 1e6)

    # The posture command has to be pinned too, and for exactly the same reason.
    # It arrived on 3 Aug 2026 and is drawn at random like any other command - so
    # left alone, a robot being tested on "walk 5 m in a straight line" would
    # spend the test being told to crouch to 150 mm and lean 20 degrees, and the
    # result would be a measurement of something nobody asked about.
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
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")

    robot = env.unwrapped.scene["robot"]
    origins = env.unwrapped.scene.env_origins

    # Walking is measured in EVERY direction it is commanded in, and the two
    # straight-line passes are scored on the worse of the two. All of them are
    # inside the trained command box, so a policy that only works forward has
    # passed a quarter of a test. Standing and being shoved have no direction
    # and run once.
    #
    # A robot that fell drags every later sample down with it, so steadiness is
    # measured over the ones still standing. Whether they fell is its own check
    # and is not being softened here - that filtering lives in score_pass.
    #
    # FOUR passes since 4 Aug 2026, not two. Sideways and turning have been
    # inside the trained command box since 3 Aug and were graded by nothing, so
    # half of what the policy is commanded to do had never been measured. Verify
    # takes about twice as long as a result - four 25 s passes instead of two.
    if spec.get("walk"):
        legs = [
            {"label": "forward", "kind": "straight",
             "vx": spec["test_speed"], "vy": 0.0, "wz": 0.0},
            {"label": "backward", "kind": "straight",
             "vx": spec["test_speed_back"], "vy": 0.0, "wz": 0.0},
            {"label": "sideways", "kind": "crab",
             "vx": 0.0, "vy": spec["test_side"], "wz": 0.0},
            {"label": "turning", "kind": "spin",
             # `test_turn_speed` is 0 for the stage bar: a spin from standstill.
             # It exists because that is NOT the only way a robot turns, and the
             # owner reported on 6 Aug 2026 that Gray turns well when driven -
             # driving means turning WHILE MOVING, which nothing here measured.
             # A pure spin needs |vx| and |vy| both near zero at the same time,
             # which the command draw produces about once in 80 - the same
             # exposure fault the crab share was added to fix. Being able to
             # measure the arc separately is how that gets told apart from the
             # robot being unable to rotate.
             "vx": spec.get("test_turn_speed", 0.0), "vy": 0.0,
             "wz": spec["test_turn"]},
        ]
    else:
        legs = [None]

    # The speed ladder. Same pass machinery, different questions: at which
    # commanded speed does the nose-dive start, and does it start INSIDE the
    # trained box or only past its edge. 0.45 is past the edge on purpose and
    # says so in its label - outside the box a fall proves nothing about the
    # policy, only about where the box ends, and the pilot HUD makes the same
    # distinction with the same words.
    if spec.get("walk") and args.ladder:
        box_hi = float(cmd.ranges.lin_vel_x[1])
        legs = []
        # Rungs scaled to whatever the box is, plus one past its edge. Fixed
        # rungs stopped meaning anything the day the box went from 0.35 to
        # 0.9 - three of the four sat in the slowest third of it.
        rungs = [round(box_hi * f, 2) for f in (0.25, 0.5, 0.75, 1.0, 1.15)]
        for v in rungs:
            out = v > box_hi + 1e-9
            legs.append({"label": f"fwd_{v:.2f}" + ("_OUT_OF_BOX" if out else ""),
                         "kind": "straight", "vx": v, "vy": 0.0, "wz": 0.0,
                         "out_of_box": out})

    print(f"task        {args.task}")
    print(f"checkpoint  {ckpt.relative_to(ROOT)}")
    print(f"tested      {args.robots} robots, {seconds:.0f} s each"
          + (f", {len(legs)} passes" if len(legs) > 1 else "")
          + f", target trunk height {target*1000:.1f} mm\n")
    drift_lines = report_drift(log_dir, env_cfg)

    checks, per_robot, paths = None, {}, {}
    # Everything measured happens under inference_mode, INCLUDING the resets.
    # The second pass's reset is the reason: mjlab's command manager zeroes its
    # own metric tensors on reset, and those tensors became inference tensors
    # during the first pass's stepping. Resetting outside inference mode then
    # dies on "inplace update to inference tensor". Nothing here needs gradients,
    # so the whole block goes inside rather than the loop alone.
    with torch.inference_mode():
        for leg in legs:
            label = leg["label"] if leg else ""
            if label:
                held = (f"{leg['wz']:+.2f} rad/s" if leg["kind"] == "spin"
                        else f"{leg['vx']:+.2f}, {leg['vy']:+.2f} m/s")
                print(f"pass        {label} at {held}")
            raw = run_pass(env, robot, origins, policy, torch,
                           seconds=seconds, robots=args.robots, target=target,
                           device="cuda:0", leg=leg, cmd=cmd,
                           diag=args.gait_diag)
            rows, dump = score_pass(raw, spec, torch, seconds=seconds,
                                    robots=args.robots, target=target, label=label)
            checks = rows if checks is None else worse_of(checks, rows)
            if dump:
                per_robot[label] = dump
            if raw.get("gait") is not None:
                stats = gait_stats(raw, torch, seconds=seconds)
                if leg and leg.get("out_of_box"):
                    stats["out_of_box"] = True
                per_robot.setdefault(label, {}).update(stats)
            if raw["path"] is not None:
                paths[label] = {
                    "command": [leg["vx"], leg["vy"], leg["wz"]],
                    "test_speed": leg["vx"],
                    "sample_every_steps": SAMPLE_EVERY,
                    "control_hz": 50,
                    "start_heading_rad": [round(x, 5)
                                          for x in raw["start_heading"].tolist()],
                    "xy_m": [[[round(c, 4) for c in p] for p in frame]
                             for frame in raw["path"].tolist()],
                }
    if len(legs) > 1:
        print()

    passed = True
    # 44, not 32. "turn rate within 0.20 rad/s of the command" is 41 characters
    # and pushed every column right of it out of line.
    print(f"{'check':<44} {'measured':>12}  {'bar':>7}")
    print("-" * 92)
    for _key, name, got, bar, how, _worst, fmt, _unit, note in checks:
        ok = got >= bar if how == "ge" else got <= bar
        passed &= ok
        print(f"{name:<44} {fmt.format(got):>12}  {bar:>7}   "
              f"{'PASS' if ok else 'FAIL'}  {note}")

    print()
    print(f"STAGE {spec['stage']} {'PASSED' if passed else 'NOT PASSED'}")

    # Record it on the run itself. Training finishing and a stage being passed
    # are different things, and the dashboard shows them as different things.
    #
    # The prose line is kept byte-identical - it is what older runs have and what
    # the run page prints. The STRUCTURE goes alongside it. Flattening six
    # measurements, six bars and six verdicts into one sentence is what stopped
    # any page being able to answer "how far off are we, and is it closing".
    # A diagnostic is not a verdict and must not overwrite one. Recording it
    # would put "19 deg, measured on a robot that never varies" on the run page
    # next to the number the bar was actually judged on, and nothing on the page
    # would say which was which.
    if switched_off or args.no_record:
        why = (f"diagnostic run ({', '.join(switched_off)} switched off)"
               if switched_off else "--no-record: measured, not judged")
        print(f"\nnot recorded - {why}")
        tags = (switched_off or ["norecord"]) \
            + (["gait"] if args.gait_diag else []) \
            + ([f"slope{args.slope_deg:g}"] if args.slope_deg else [])
        _dump_diagnostic(ROOT, log_dir.name, tags,
                         per_robot, paths, ckpt, args.robots, seconds)
        env.close()
        return 0 if passed else 1

    try:
        from dashboard import runs as runs_mod  # noqa: PLC0415

        detail = " · ".join(
            f"{name}: {fmt.format(got)}"
            for _k, name, got, _b, _h, _w, fmt, _u, _n in checks)

        rows = []
        for key, name, got, bar, how, worst, fmt, unit, note in checks:
            ok = got >= bar if how == "ge" else got <= bar
            # Normalised distance to the bar, in the same direction for every
            # criterion: 1.0 is exactly at the bar, above 1.0 passes. That is
            # what makes drift and distance comparable on one axis. None rather
            # than infinity when a division would blow up - the page shows
            # "unknown", which is true, instead of a number that is not.
            if how == "ge":
                ratio = (got / bar) if bar else None
                margin = got - bar
            else:
                ratio = (bar / got) if got else None
                margin = bar - got
            rows.append({
                "key": key, "name": name,
                "measured": got, "bar": bar,
                "unit": unit,
                "better": "higher" if how == "ge" else "lower",
                "passed": bool(ok),
                "worst": worst,
                "margin": margin,
                "ratio": ratio,
                "note": note,
                "format": fmt,
            })

        runs_mod.set_verdict(
            log_dir.name, "passed" if passed else "not passed",
            f"{args.robots} robots x {seconds:.0f} s - {detail}",
            checks=rows,
            context={
                "task": args.task,
                "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
                "robots": args.robots,
                "seconds": seconds,
                "target_height_m": target,
                "test_speed": spec.get("test_speed"),
                "test_speed_back": spec.get("test_speed_back"),
                "test_side": spec.get("test_side"),
                "test_turn": spec.get("test_turn"),
                # The command each pass held, not just its name. A page showing
                # "sideways: 4.1 m" has to be able to say sideways at WHAT - and
                # the day test_side moves, every older verdict still says what it
                # was measured at rather than inheriting today's number.
                "passes": [{"label": lg["label"], "kind": lg["kind"],
                            "command": [lg["vx"], lg["vy"], lg["wz"]]}
                           for lg in legs if lg],
                # 64 numbers per pass instead of a mean and a max. The mean says
                # "20 degrees off"; only the 64 say whether that is every robot
                # leaning the same way, which is a gait that is not symmetric, or
                # an even spread either way, which is a heading nothing observes.
                # Those need opposite fixes, and the mean cannot tell them apart.
                "per_robot": per_robot,
                # Whether the task moved under the policy since it trained. A
                # verdict measured against a changed task answers a different
                # question, and the page has to be able to say so.
                "drift": drift_lines,
            })
        print(f"recorded on the run: {log_dir.name}")

        # The walked paths go in their own file, not in run.json. They are ~7000
        # numbers per pass against run.json's few hundred, and every dashboard
        # page reads run.json on every request.
        if paths:
            import json  # noqa: PLC0415

            out = runs_mod.RUNS / log_dir.name / "verify_paths.json"
            if out.parent.is_dir():
                out.write_text(json.dumps(
                    {"checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
                     "robots": args.robots, "seconds": seconds,
                     "passes": paths}, indent=1))
                print(f"walked paths: {out.relative_to(ROOT)}")
    except Exception as exc:  # noqa: BLE001
        print(f"could not record the verdict: {exc}")

    env.close()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
