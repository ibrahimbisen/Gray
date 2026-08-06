"""Drive a trained policy yourself, live, with the keyboard.

    Playground\\pilot.bat                    the newest walk run, flat floor
    Playground\\pilot.bat --run 25           a run by number, name, or folder
    Playground\\pilot.bat --playground       hills, a bowl, rough ground, waves
    Playground\\pilot.bat --slope-deg 10     one uniform hill

verify.py scores a policy against a bar. film_checkpoints.py films it. drive.py
pins one command and writes an mp4. None of them let you STEER it, and steering
is the thing that will actually be happening when the Joy-Cons arrive.

A policy takes three numbers - forward m/s, sideways m/s, turn rad/s - and this
puts a keyboard on the other end of them. See Playground/control.py: the same
three numbers will come off a gamepad later without anything here changing.

What it is good for: does it respond, does it recover, does it look like a dog.
What it is NOT: a result. The reward is a weighted sum and the eye is easily
fooled by a robot that looks busy. scripts/verify.py stays the judge.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Run numbers, names and log folders are resolved exactly the way drive.py does
# it, by importing that code rather than copying it. Two files disagreeing about
# which run "25" is would be a bad afternoon.
from scripts.drive import resolve_run  # noqa: E402

from Playground.control import KEYS, Pilot  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="walk",
                    help="run number, name, or folder. Default: the newest walk run")
    ap.add_argument("--checkpoint", help="e.g. model_2999.pt; default is the last")
    ap.add_argument("--task", default="Gray-Walk")
    ap.add_argument("--device", default="")
    ap.add_argument("--keys", action="store_true",
                    help="ignore the gamepad and drive from the numpad")
    ap.add_argument("--no-panel", action="store_true",
                    help="do not open the settings window beside the simulator")
    # The ground. Added 6 Aug 2026, the day the slope work landed - the owner
    # asked for somewhere to DRIVE it on hills, which no script offered.
    ap.add_argument("--playground", action="store_true",
                    help="drive on the mixed terrain field instead of a flat "
                         "floor: flat, gentle and steep hills, a bowl, rough "
                         "ground and waves, side by side, each in two "
                         "difficulties. You spawn on the flat and find the "
                         "rest by walking.")
    ap.add_argument("--slope-deg", type=float, default=0.0, metavar="DEG",
                    help="drive on one uniform slope of this angle instead - "
                         "the same world the slope batches train on. Ignored "
                         "when --playground is given.")
    args = ap.parse_args()

    log_dir = resolve_run(args.run)
    # Sorted numerically, not lexicographically - by name model_975 sorts after
    # model_2999, so "the last checkpoint" of a 3000-iteration run would be one
    # from a third of the way through. Same trap as drive.py.
    ckpts = sorted(log_dir.glob("model_*.pt"),
                   key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {log_dir}")
    ckpt = (log_dir / args.checkpoint) if args.checkpoint else ckpts[-1]
    if not ckpt.is_file():
        raise SystemExit(f"no such checkpoint: {ckpt}")

    from dataclasses import asdict  # noqa: PLC0415

    import torch  # noqa: PLC0415

    import gray.tasks  # noqa: F401,PLC0415
    from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: PLC0415

    from Playground.hud import PilotViewer  # noqa: PLC0415

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1

    ground = "the flat floor"
    if args.playground or args.slope_deg:
        from gray.tasks.walk_env_cfg import (  # noqa: PLC0415
            apply_slope, playground_terrain)

        if args.playground:
            env_cfg.scene.terrain = playground_terrain()
            ground = ("the terrain field - flat, two hills, a bowl, rough "
                      "ground and waves")
        else:
            apply_slope(env_cfg, args.slope_deg)
            # A driven robot is not being scored, and a truncation mid-drive
            # looks like the robot vanishing. Walking off the hill onto the
            # apron is allowed here; the terminations that catch a FALL stay.
            env_cfg.terminations.pop("out_of_bounds", None)
            ground = f"a {args.slope_deg:g} deg slope"
    # An hour. Long enough that the episode never quietly ends mid-drive; short
    # enough to stay a sane integer number of steps. Falling still resets, which
    # is wanted - it puts the robot back on its feet without touching anything.
    env_cfg.episode_length_s = 3600.0

    cmd = env_cfg.commands.get("walk")
    if cmd is None:
        raise SystemExit(f"{args.task} has no 'walk' command to steer")

    # What the policy was actually trained on, read from the task's own config
    # rather than written down a second time. The HUD holds the command against
    # this and says so when you drive outside it.
    trained = {"fwd": tuple(cmd.ranges.lin_vel_x),
               "side": tuple(cmd.ranges.lin_vel_y),
               "turn": tuple(cmd.ranges.ang_vel_z)}

    # Three settings, all of which would otherwise fight the keyboard for
    # control of the command, every one of them silently:
    #   resampling_time_range  the command is rerolled every 5-10 s in training
    #   rel_standing_envs      a share of robots have their command zeroed each
    #                          step, whatever it was set to
    #   rel_world_envs         a share have theirs overwritten from a world-frame
    #                          copy the keyboard never writes to
    cmd.resampling_time_range = (1e6, 1e6)
    cmd.rel_standing_envs = 0.0
    cmd.rel_world_envs = 0.0

    # THE POSTURE COMMAND HAS TO BE PINNED TOO, and it was not until 6 Aug
    # 2026. It is a second command stream - ride height, pitch, roll - drawn
    # at random every 5 to 10 seconds, and the sticks do not write to it. So
    # a driven robot was being ordered to crouch to 150 mm, lean 15 deg
    # nose-up, then 8 deg nose-down, then roll 20 deg, on its own schedule,
    # while the owner steered. It obeyed. From the outside that reads as a
    # robot that cannot hold its body level - the owner reported exactly
    # that, nose way up walking forward and way down walking backward.
    #
    # verify.py has pinned this since it was written, which is why the
    # numbers never showed what the driving showed.
    posture = env_cfg.commands.get("posture")
    if posture is not None:
        h = posture.nominal_height
        posture.ranges.height = (h, h)
        posture.ranges.pitch = (0.0, 0.0)
        posture.ranges.roll = (0.0, 0.0)
        posture.rel_nominal_envs = 1.0
        posture.resampling_time_range = (1e6, 1e6)

    agent_cfg = load_rl_cfg(args.task)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=device),
                             clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True,
                map_location=device)
    policy = runner.get_inference_policy(device=device)

    # A gamepad if there is one, the numpad if there is not. The pad is better
    # in every way that matters here - a stick reports where it IS, so the
    # command follows your thumb instead of being nudged a step at a time.
    pilot, legend, found = None, KEYS, ""
    if not args.keys:
        from Playground.pad import (  # noqa: PLC0415
            PAD_KEYS, PadPilot, describe, find_pad)
        jid = find_pad()
        if jid is not None:
            pilot = PadPilot(trained, jid)
            legend, found = PAD_KEYS, describe(jid)
    if pilot is None:
        pilot = Pilot(trained)

    from Playground.panel import Panel, load_settings  # noqa: PLC0415

    # Whatever the panel was left set to last time - which stick, which button,
    # how big the deadzone. Applied before anything reads the pad.
    load_settings(pilot)

    print(f"run         {log_dir.name}")
    print(f"checkpoint  {ckpt.name}")
    print(f"ground      {ground}")
    print(f"driving     {found or 'no gamepad found - using the numpad'}")
    print(f"trained on  {trained['fwd'][0]:g} to {trained['fwd'][1]:g} m/s forward, "
          f"{trained['side'][0]:+g} to {trained['side'][1]:+g} sideways, "
          f"{trained['turn'][0]:+g} to {trained['turn'][1]:+g} rad/s turn")
    print("            outside that, whatever it does means nothing\n")
    print(legend)

    # Swapping to another checkpoint without restarting. The runner is already
    # built, so loading a different one is two calls - but they must happen on
    # the physics thread, which is why this is handed over as a function rather
    # than done where it is asked for.
    def load_policy(path):
        runner.load(str(path), load_cfg={"actor": True}, strict=True,
                    map_location=device)
        return runner.get_inference_policy(device=device)

    # The keyboard is only wired up when there is no pad. Wired up alongside
    # one, every key press would be overwritten by the stick position on the
    # very next step, which looks like a broken keyboard rather than a
    # deliberate choice.
    on_key = None if hasattr(pilot, "poll") else pilot.key_callback
    viewer = PilotViewer(env, policy, pilot, key_callback=on_key,
                         load_policy=load_policy,
                         loaded_name=f"{log_dir.name} / {ckpt.name}")

    if args.no_panel:
        viewer.run()
    else:
        # Tk gets the MAIN thread and the simulator gets a worker. The other way
        # round - tk built on a background thread while MuJoCo holds a GLFW
        # window - kills the process outright on Windows, with no exception and
        # no traceback: both windows just never appear.
        #
        # catch_sigint is off because signal handlers can only be installed on
        # the main thread, and the panel closing must not take a running
        # simulation down with it, hence the join.
        sim = threading.Thread(target=viewer.run, kwargs={"catch_sigint": False},
                               daemon=True, name="sim")
        sim.start()
        Panel(pilot, viewer, log_dir, ckpt.name).run()
        sim.join()

    env.close()


if __name__ == "__main__":
    main()
