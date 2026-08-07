"""How far out can each leg actually swing? Measured, not read off a range.

THE QUESTION. gray.xml gives the four hips two different limit pairs, and they
group by DIAGONAL rather than by side:

    front-right   -40 deg to +65 deg        back-left   -40 to +65
    front-left    -65 deg to +40 deg        back-right  -65 to +40

robot.yaml, which is measured against the CAD, says a hip is -65 to +40 with
"+ swings the leg OUT". So two legs look mirrored - and they are the same two
the build notes record as having exported 180 degrees rotated once already.
The owner assembled the robot from one hip part used twice, rotated, which is
exactly how a frame ends up flipped in the export.

WHY THE NUMBERS ALONE CANNOT ANSWER IT. If a joint's frame is rotated 180
degrees, then "+65 in that frame" can be the SAME physical motion as "-65" in
another. Mirrored numbers on a mirrored frame describe a symmetric robot. So
the range table proves nothing either way.

WHAT THIS MEASURES INSTEAD. For each leg: drive that hip to each end of its
travel, leave every other joint in the stance, and record where the FOOT ends
up - its distance from the robot's centreline, in millimetres. Geometry, in
the world, with no convention in the path.

    both diagonals reach the same        the robot is symmetric, the numbers
                                         only look odd. Nothing to fix.
    one diagonal reaches further out     the model is lopsided, and every run
                                         so far trained a robot with one leg
                                         pair that can swing wider than the
                                         other.

Run it with the card free:

    <uv-python>\\python.exe scripts\\measure_hip_travel.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LEGS = (("fr", "front-right"), ("fl", "front-left"),
        ("br", "back-right"), ("bl", "back-left"))


def main() -> int:
    import torch  # noqa: PLC0415

    import gray.tasks  # noqa: F401,PLC0415
    from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg  # noqa: PLC0415

    cfg = load_env_cfg("Gray-Walk", play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    env.reset()
    robot = env.scene["robot"]

    # Hold the trunk still and level, well clear of the floor, so nothing the
    # legs do can move the body and confuse the reading.
    pos = robot.data.root_link_pos_w.clone()
    pos[:, 2] = 0.6
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=pos.device)
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))

    site_ids, site_names = robot.find_sites(".*_foot")
    stance = robot.data.default_joint_pos.clone()
    lo, hi = robot.data.joint_pos_limits[0, :, 0], robot.data.joint_pos_limits[0, :, 1]

    print(f"\n{'leg':<12} {'limit':>16}   {'foot out from centre':>22}")
    print("-" * 56)
    reach = {}
    for tag, label in LEGS:
        jid, _ = robot.find_joints(f"{tag}hip")
        j = jid[0]
        foot = site_names.index(f"{tag}_foot")
        out = []
        for end, name in ((lo[j], "lower"), (hi[j], "upper")):
            q = stance.clone()
            q[0, j] = end
            robot.write_joint_position_to_sim(q)
            env.sim.forward()
            # Sideways distance from the trunk's centreline, in the body frame.
            # The trunk is pinned level and unrotated, so world y IS body y.
            y = (robot.data.site_pos_w[0, site_ids[foot], 1]
                 - robot.data.root_link_pos_w[0, 1]).item()
            out.append((abs(y), math.degrees(end.item()), name))
        widest = max(out)
        reach[tag] = widest[0]
        for width, deg, name in out:
            mark = "  <- widest" if (width, deg, name) == widest else ""
            print(f"{label if name == 'lower' else '':<12} "
                  f"{deg:>+8.1f} deg      {width * 1000:>10.1f} mm{mark}")
        print()

    print("-" * 56)
    diag_a = (reach["fr"] + reach["bl"]) / 2 * 1000
    diag_b = (reach["fl"] + reach["br"]) / 2 * 1000
    print(f"front-right + back-left reach out   {diag_a:6.1f} mm")
    print(f"front-left  + back-right reach out  {diag_b:6.1f} mm")
    gap = abs(diag_a - diag_b)
    print()
    if gap < 5.0:
        print(f"SYMMETRIC - the two diagonals agree to {gap:.1f} mm. The mirrored "
              f"limit numbers describe mirrored frames, which is correct. The "
              f"leg throw the owner saw is the POLICY's, not the model's.")
    else:
        print(f"LOPSIDED - the diagonals differ by {gap:.1f} mm. The model lets "
              f"one pair of legs swing wider than the other, and every run so "
              f"far trained against that. Fix the two hip ranges in the model, "
              f"then retrain.")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
