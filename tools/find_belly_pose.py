#!/usr/bin/env python3
"""Find a belly-down resting pose: trunk on the floor, legs folded, nothing clipping.

    python tools/find_belly_pose.py                  # search, report, render
    python tools/find_belly_pose.py --write          # also write progress/belly_pose.json

WHY THIS EXISTS. From-scratch training starts every episode with the robot lying on
its belly and learning to stand up. That start pose has to be one the real machine
could actually be put into, which means three things at once:

  1. the TRUNK is resting on the ground, not the knees or the feet
  2. no part passes through another - checked at triangle level against the real
     meshes, not against the loose collision boxes
  3. it is STABLE: left alone under gravity with the servos holding, it stays put

The pose is not typed in. Typed-in angles are how the residual task ended up
spawning the robot standing while its gait phase said mid-stride, which threw the
trunk 330 mm backwards at every reset and put a third of the fleet on the floor
before it had taken a step. Here the pose is searched for and then SETTLED under
gravity, so what comes out is whatever the physics will actually hold.

Every candidate is scored on the settled result, never on the commanded angles: a
pose the servos cannot hold is not a pose.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_joint_limits import SelfCollision, _part_name  # noqa: E402

MJCF = "sim/models/gray.xml"
OUT_JSON = "progress/belly_pose.json"
LEGS = ("fl", "fr", "br", "bl")
SEGMENTS = ("hip", "top", "bottom")

# Long enough for the servos to drive to the target and for the robot to stop moving.
# At 50 Hz control over 2 ms physics that is 4 s, well past the ~1 s it takes to fall
# and settle from a few centimetres up.
SETTLE_STEPS = 2000

# Below this the robot has stopped moving rather than merely slowed down.
STILL_MM_S = 2.0


def _joint_addr(model: mujoco.MjModel) -> dict[str, int]:
    return {f"{leg}_{seg}": model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{seg}")]
        for leg in LEGS for seg in SEGMENTS}


def _act_addr(model: mujoco.MjModel) -> dict[str, int]:
    return {f"{leg}_{seg}": mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_{seg}")
        for leg in LEGS for seg in SEGMENTS}


def settle(model: mujoco.MjModel, data: mujoco.MjData, targets: dict[str, float],
           acts: dict[str, int], clearance_m: float = 0.005) -> dict:
    """Hold `targets` and let the robot settle. Returns what it settled into.

    THE JOINTS ARE PLACED BEFORE THE ROBOT IS DROPPED, and the drop height is computed
    from the resulting shape rather than fixed. An earlier version spawned at a fixed
    60 mm with the legs still at their default angles and only the servo TARGETS
    folded: the legs were through the floor on the first step, the solver ejected the
    robot, and all 25 candidate poses landed upside down with an identical
    uprightness of -1.000. A pose search that flips the robot every time measures
    nothing.
    """
    mujoco.mj_resetData(model, data)
    addr = _joint_addr(model)
    for name, a in addr.items():
        data.qpos[a] = targets[name]
    for name, idx in acts.items():
        data.ctrl[idx] = targets[name]

    # Drop the trunk so the lowest collision geom starts just clear of the floor.
    mujoco.mj_forward(model, data)
    lowest = min(
        (float(data.geom_xpos[g][2]) - float(model.geom_rbound[g]))
        for g in range(model.ngeom)
        if model.geom_group[g] == 3
    )
    data.qpos[2] += clearance_m - lowest

    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)

    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    # Uprightness: +1 is level and belly-down, -1 is on its back.
    R = data.xmat[trunk].reshape(3, 3)
    upright = float(R[2, 2])
    speed = float(np.linalg.norm(data.qvel[:3])) * 1000.0

    # Which named geoms are touching the floor, and how much of the trunk is down.
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    touching = set()
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == floor or c.geom2 == floor:
            other = c.geom2 if c.geom1 == floor else c.geom1
            touching.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
                         or f"geom{other}")
    return {
        "trunk_height_mm": float(data.xpos[trunk][2]) * 1000.0,
        "upright": upright,
        "speed_mm_s": speed,
        "touching": sorted(touching),
        "belly_down": "base_link_collision" in touching,
        "still": speed < STILL_MM_S,
        "qpos": data.qpos.copy(),
    }


def search(model: mujoco.MjModel, sc: SelfCollision, coarse: int = 5) -> list[dict]:
    """Sweep symmetric leg folds and keep every pose that actually lies down."""
    data = mujoco.MjData(model)
    acts = _act_addr(model)
    addr = _joint_addr(model)
    limit = float(model.jnt_range[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fl_hip")][1])

    # Searched symmetrically: the same thigh and knee fold on all four legs, and the
    # hips mirrored left/right so the robot cannot walk itself sideways while settling.
    # An asymmetric search is a much larger space and a resting pose has no reason to
    # be asymmetric.
    grid = np.linspace(-limit * 0.9, limit * 0.9, coarse)
    found = []
    for thigh, knee in itertools.product(grid, grid):
        for hip in np.linspace(0.0, limit * 0.6, 3):
            targets = {}
            for leg in LEGS:
                # Mirror the hips so left and right fold the same way outward.
                sign = 1.0 if leg in ("fl", "bl") else -1.0
                targets[f"{leg}_hip"] = sign * hip
                targets[f"{leg}_top"] = thigh
                targets[f"{leg}_bottom"] = knee
            r = settle(model, data, targets, acts)
            if not (r["belly_down"] and r["still"] and r["upright"] > 0.9):
                continue
            r["targets"] = {k: float(v) for k, v in targets.items()}
            # Only now the expensive check: does the settled pose clip?
            r["clips"] = bool(sc.hits(r["qpos"]))
            found.append(r)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coarse", type=int, default=5, help="grid points per axis")
    ap.add_argument("--write", action="store_true", help=f"write {OUT_JSON}")
    ap.add_argument("--render", default="progress/belly_pose.png")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(MJCF)
    sc = SelfCollision(model)
    print(f"searching for a belly-down pose ({args.coarse}x{args.coarse}x3 folds, "
          f"each settled for {SETTLE_STEPS * model.opt.timestep:.1f} s under gravity)")
    print(f"self-collision checked at triangle level over {len(sc.pairs)} part pairs\n")

    poses = search(model, sc, args.coarse)
    clean = [p for p in poses if not p["clips"]]
    print(f"{len(poses)} pose(s) settled belly-down and still; "
          f"{len(clean)} of them with no part inside another")
    if not clean:
        print("\nNo clean belly pose found on this grid. Widen --coarse, or the legs "
              "cannot fold clear of the trunk and the pose has to be asymmetric.")
        if poses:
            print("Closest (settles belly-down but clips):")
            p = poses[0]
            print(f"  trunk {p['trunk_height_mm']:.1f} mm, touching {p['touching']}")
        raise SystemExit(1)

    # Lowest trunk wins: that is the most committed to the ground, and the least
    # likely to be a knees-down crouch that happens to brush the belly.
    best = min(clean, key=lambda p: p["trunk_height_mm"])
    print(f"\nbest belly pose:")
    print(f"  trunk height   {best['trunk_height_mm']:.1f} mm")
    print(f"  uprightness    {best['upright']:.3f}   (1.0 = level)")
    print(f"  residual speed {best['speed_mm_s']:.2f} mm/s  (settled)")
    print(f"  touching floor {', '.join(best['touching'])}")
    print(f"  parts clipping NONE")
    print(f"  joint targets (deg):")
    for leg in LEGS:
        vals = "  ".join(f"{seg}={np.degrees(best['targets'][f'{leg}_{seg}']):+7.1f}"
                         for seg in SEGMENTS)
        print(f"     {leg}   {vals}")

    if args.render:
        os.makedirs(os.path.dirname(args.render), exist_ok=True)
        import imageio.v2 as iio
        data = mujoco.MjData(model)
        data.qpos[:] = best["qpos"]
        mujoco.mj_forward(model, data)
        r = mujoco.Renderer(model, height=620, width=940)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.distance, cam.azimuth, cam.elevation = 0.75, 130, -20
        cam.lookat[:] = [0.0, 0.0, 0.04]
        r.update_scene(data, cam)
        iio.imwrite(args.render, r.render())
        print(f"\nrendered to {args.render}")

    if args.write:
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump({
                "targets_rad": best["targets"],
                "trunk_height_mm": best["trunk_height_mm"],
                "upright": best["upright"],
                "touching": best["touching"],
                "note": "settled under gravity with the servos holding these targets; "
                        "self-collision checked at triangle level against the real "
                        "meshes. See tools/find_belly_pose.py.",
            }, fh, indent=2)
        print(f"written to {OUT_JSON}")


if __name__ == "__main__":
    main()
