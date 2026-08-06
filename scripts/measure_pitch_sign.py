r"""Which sign is nose-down? Ask the simulator, not a comment.

Two texts disagree. gray/tasks/walk_env_cfg.py says nose-down is NEGATIVE,
measured on 5 Aug 2026. dashboard/plan.py's /dials row still says nose-down is
positive. One of them is wrong, and POSE_PITCH was sign-flipped on the strength
of the first, so the answer decides whether the command range points into the
10 degrees of nose-down travel or the 20 of nose-up.

The measurement, so it cannot be argued with:

1. Build the walk task, one robot, play mode.
2. Overwrite the trunk orientation with a pure +10 degree rotation about +y.
3. PROVE the nose is down with geometry that owes nothing to any convention:
   the front foot sites must sit LOWER in the world than the back ones.
4. Print what trunk_pitch_roll reports for that proven attitude.

Step 3 is the point. Reading projected_gravity back would only restate the
quaternion; two foot heights are a fact about the world.

Run it (the venv's python.exe is blocked - use the base interpreter):

    set PYTHONPATH=%CD%\.venv\Lib\site-packages;%CD%
    <uv-python>\python.exe scripts\measure_pitch_sign.py

Never while a training run holds the card.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    import torch  # noqa: PLC0415

    import gray.tasks  # noqa: F401,PLC0415
    from gray.tasks.posture_command import trunk_pitch_roll  # noqa: PLC0415
    from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg  # noqa: PLC0415

    env_cfg = load_env_cfg("Gray-Walk", play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(env_cfg, device="cuda:0")
    env.reset()
    robot = env.scene["robot"]
    ids, names = robot.find_sites(".*_foot")

    deg = 10.0
    th = math.radians(deg)
    failed = False

    def look(axis: str) -> tuple[float, float, list[float]]:
        """Rotate the trunk +10 deg about one world axis and read it back."""
        pos = robot.data.root_link_pos_w.clone()
        pos[:, 2] += 0.05  # a little air, so nothing touches while we look
        # wxyz, a pure rotation. Yaw is overwritten along with everything
        # else, so the reset nudge cannot leak into the reading.
        x = math.sin(th / 2) if axis == "x" else 0.0
        y = math.sin(th / 2) if axis == "y" else 0.0
        quat = torch.tensor([[math.cos(th / 2), x, y, 0.0]], device=pos.device)
        robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
        env.sim.forward()
        p, r = trunk_pitch_roll(robot)
        return p.item(), r.item(), robot.data.site_pos_w[0, ids, 2].tolist()

    def side(prefix: tuple[str, ...], foot_z: list[float]) -> list[float]:
        return [z for n, z in zip(names, foot_z)
                if n.lower().startswith(prefix)]

    # --- pitch: +10 deg about +y. Front feet against back feet. ---
    pitch, roll, foot_z = look("y")
    print(f"\n+{deg:.0f} deg about +y:")
    for name, z in zip(names, foot_z):
        print(f"    {name:16s} {z:+.4f}")
    front, back = side(("f",), foot_z), side(("b",), foot_z)
    nose_down = max(front) < min(back)
    print(f"front feet {'LOWER' if nose_down else 'HIGHER'} than back "
          f"-> the nose is physically {'DOWN' if nose_down else 'UP'}")
    print(f"trunk_pitch_roll: pitch {pitch:+.4f} rad "
          f"({math.degrees(pitch):+.1f} deg), roll {roll:+.4f} rad")
    if nose_down == (pitch < 0):
        print("VERDICT: NOSE DOWN IS NEGATIVE. The task file and _REACH are "
              "right; the /dials row (plan.py _COMMAND_DIALS) was the wrong "
              "text.")
    else:
        failed = True
        print("VERDICT: NOSE DOWN IS POSITIVE. The 5 Aug flip pointed "
              "POSE_PITCH the wrong way - the task file is the wrong text, "
              "and the range change is a training-set change.")

    # --- roll: +10 deg about +x. Right feet against left feet. ---
    pitch, roll, foot_z = look("x")
    print(f"\n+{deg:.0f} deg about +x:")
    for name, z in zip(names, foot_z):
        print(f"    {name:16s} {z:+.4f}")
    right = side(("fr", "br"), foot_z)
    left = side(("fl", "bl"), foot_z)
    right_down = max(right) < min(left)
    print(f"right feet {'LOWER' if right_down else 'HIGHER'} than left "
          f"-> the right side is physically {'DOWN' if right_down else 'UP'}")
    print(f"trunk_pitch_roll: pitch {pitch:+.4f} rad, roll {roll:+.4f} rad "
          f"({math.degrees(roll):+.1f} deg)")
    sign = "POSITIVE" if (right_down == (roll > 0)) else "NEGATIVE"
    print(f"VERDICT: RIGHT SIDE DOWN IS {sign}.")

    env.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
