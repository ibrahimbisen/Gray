"""Prove ground_height_under against the world it claims to describe.

The 4 degree rung produced a robot that barely moves, on the hill AND on the
flat floor - the signature of movement being fined. The prime suspect is the
heightfield table read feeding `dragging` a wrong ground height once a robot
leaves the spawn platform.

The proof, sensor-free: drop a probe of known geometry - the robot itself -
at points across the hill, let it settle, and compare where its FEET rest
(world z, physics ground truth) against ground_height_under at the same
(x, y). If the table read is right, resting feet sit within a toe's width of
the read. If it is transposed, offset or misscaled, the disagreement grows
with distance from the centre.

    set PYTHONPATH=... ; python scripts\check_ground_read.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    import torch  # noqa: PLC0415

    import gray.tasks  # noqa: F401,PLC0415
    from gray.tasks.walk_env_cfg import (  # noqa: PLC0415
        apply_slope,
        ground_height_under,
    )
    from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg  # noqa: PLC0415

    cfg = load_env_cfg("Gray-Walk", play=True)
    cfg.scene.num_envs = 1
    apply_slope(cfg, 10.0)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    env.reset()
    robot = env.scene["robot"]
    ids, names = robot.find_sites(".*_foot")
    origin = env.scene.env_origins[0]
    print(f"env origin: ({origin[0]:+.3f}, {origin[1]:+.3f}, {origin[2]:+.3f})")

    worst = 0.0
    for dx, dy in ((0.0, 0.0), (2.0, 0.0), (-2.0, 0.0), (0.0, 2.0),
                   (0.0, -3.0), (3.0, 3.0), (-4.0, 2.0)):
        pos = robot.data.root_link_pos_w.clone()
        pos[:, 0] = origin[0] + dx
        pos[:, 1] = origin[1] + dy
        ground = ground_height_under(env, pos[:, :2])
        pos[:, 2] = ground + 0.25
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=pos.device)
        robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
        zero = torch.zeros(1, 12, device=pos.device)
        with torch.inference_mode():
            for _ in range(50):  # one second, let it land and settle
                env.step(zero)
        feet_w = robot.data.site_pos_w[0, ids]
        read = ground_height_under(env, feet_w[:, :2])
        gap = (feet_w[:, 2] - read)
        worst = max(worst, float(gap.abs().max()))
        print(f"at (+{dx:.0f}, +{dy:.0f}): resting feet sit "
              f"{', '.join(f'{g * 1000:+.0f}' for g in gap.tolist())} mm "
              f"above the table's ground")

    print()
    if worst < 0.06:
        print(f"TABLE READ AGREES WITH THE PHYSICS - worst gap "
              f"{worst * 1000:.0f} mm (a foot's radius). Look elsewhere.")
    else:
        print(f"TABLE READ IS WRONG - worst gap {worst * 1000:.0f} mm. "
              f"The reward terms were fed a different hill than the one "
              f"the robot stood on.")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
