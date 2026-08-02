#!/usr/bin/env python3
"""Turn a set of joint angles into a checked, rendered, saved pose.

    python tools/render_pose.py --name resting --hip 55 --top -21 --bottom 39 \
        --note "powered off and limp: 4 hips and 4 toes on the floor"

Angles are DEGREES in the pose editor's PHYSICAL convention, which is what the panel's
sliders show:

    hip     +  leg swings OUT, away from the body
    top     +  leg swings FORWARD, towards the nose
    bottom  +  foot lifts UP

The per-leg sign needed to make that true is measured, not written down - the legs are
mirrored and the raw joint angles disagree with each other.

WHAT IT DOES, in order:
  1. sets the pose and lowers the robot until its lowest part rests on the floor
  2. checks every part against every other AT TRIANGLE LEVEL on the real meshes
  3. reports what is actually touching the ground, so a claim like "8 points of
     contact" can be checked rather than assumed
  4. optionally settles it under gravity, to see whether it holds
  5. renders three views
  6. saves the angles to progress/poses.json and the description to reference/POSES.md

The render is generated FROM the saved numbers, so the picture can never drift out of
step with the data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_joint_limits import SelfCollision, _part_name  # noqa: E402
from pose_editor import (LEGS, MJCF, OUT_JSON, SEGMENTS, level_feet,  # noqa: E402
                         lowest_point, measure_signs)

RENDER_DIR = "reference/poses"
NOTES = "reference/POSES.md"
SETTLE_STEPS = 2000


def build(model, data, signs, angles: dict[str, float]) -> None:
    """Place the robot in `angles` (physical degrees) and sit it on the floor."""
    mujoco.mj_resetData(model, data)
    for leg in LEGS:
        for seg in SEGMENTS:
            name = f"{leg}_{seg}"
            adr = model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            data.qpos[adr] = np.radians(angles[seg]) * signs[name]
    mujoco.mj_forward(model, data)
    data.qpos[2] -= lowest_point(model, data)
    mujoco.mj_forward(model, data)


def touching(model, data, within_mm: float = 3.0) -> list[str]:
    """Every part whose lowest point is on, or within `within_mm` of, the floor.

    MEASURED GEOMETRICALLY, not read off MuJoCo's contact list. A pose that has been
    placed exactly on the ground has zero overlap, and MuJoCo only reports a contact
    once two shapes actually interpenetrate - so asking it produced "touching the
    floor: nothing" for a robot visibly resting on the ground. Distance to the floor
    answers the question the operator is actually asking, which is which parts carry
    the weight.
    """
    out = []
    for g in range(model.ngeom):
        if model.geom_group[g] != 3:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not name or name == "floor":
            continue
        pos = data.geom_xpos[g]
        R = data.geom_xmat[g].reshape(3, 3)
        size = model.geom_size[g]
        kind = model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            corners = np.array([[x, y, z]
                                for x in (-size[0], size[0])
                                for y in (-size[1], size[1])
                                for z in (-size[2], size[2])])
            low = float(((R @ corners.T).T + pos)[:, 2].min())
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            half = np.array([0.0, 0.0, size[1]])
            ends = np.array([(R @ half) + pos, (R @ -half) + pos])
            low = float(ends[:, 2].min() - size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            low = float(pos[2] - size[0])
        else:
            continue
        if low * 1000.0 <= within_mm:
            out.append(name.replace("_collision", ""))
    return sorted(out)


def clipping(sc: SelfCollision, model, data) -> list[tuple[str, str]]:
    sc.data.qpos[:] = data.qpos
    T = sc._place()
    clash = []
    for a, b in sc.pairs:
        sc.managers[a].set_transform(str(a), T[a])
        if sc.managers[a].in_collision_single(sc.meshes[b], transform=T[b]):
            clash.append((_part_name(model, a), _part_name(model, b)))
    return clash


def settle(model, data, limp: bool = False) -> dict:
    """Let gravity have it. `limp` switches the servos off first.

    Limp is what a powered-down robot actually is: a position servo with no
    current holds nothing, so the legs fall where the linkage and gravity put
    them. Settling with the servos ON answers a different question - whether the
    pose can be HELD - and the two give very different answers.
    """
    if limp:
        model.actuator_gainprm[:, 0] = 0.0
        model.actuator_biasprm[:, 1] = 0.0
        model.actuator_biasprm[:, 2] = 0.0
    for leg in LEGS:
        for seg in SEGMENTS:
            name = f"{leg}_{seg}"
            adr = model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            data.ctrl[aid] = data.qpos[adr]
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    return {
        "trunk_mm": float(data.xpos[trunk][2]) * 1000.0,
        "level": float(data.xmat[trunk].reshape(3, 3)[2, 2]),
        "moved_mm_s": float(np.linalg.norm(data.qvel[:3])) * 1000.0,
        "touching": touching(model, data),
    }


def render(model, data, name: str) -> list[str]:
    import imageio.v2 as iio
    os.makedirs(RENDER_DIR, exist_ok=True)
    r = mujoco.Renderer(model, height=620, width=920)
    out = []
    for tag, az, el, dist in (("iso", 135, -22, 0.95),
                              ("side", 90, -6, 0.95),
                              ("front", 180, -8, 0.9)):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        cam.lookat[:] = [0.0, 0.0, 0.05]
        r.update_scene(data, cam)
        path = f"{RENDER_DIR}/{name}_{tag}.png"
        iio.imwrite(path, r.render())
        out.append(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--hip", type=float, required=True)
    ap.add_argument("--top", type=float, required=True)
    ap.add_argument("--bottom", type=float, required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--settle", action="store_true",
                    help="also let gravity have it, and report where it ends up")
    ap.add_argument("--limp", action="store_true",
                    help="settle with the servos OFF, as a powered-down robot")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    sc = SelfCollision(model)
    signs = measure_signs(model)
    angles = {"hip": args.hip, "top": args.top, "bottom": args.bottom}

    build(model, data, signs, angles)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    height = float(data.xpos[trunk][2]) * 1000.0
    clash = clipping(sc, model, data)
    on_floor = touching(model, data)

    print(f"\n=== {args.name} ===")
    print(f"  hips {args.hip:+.1f} out   thighs {args.top:+.1f} forward   "
          f"knees {args.bottom:+.1f} up")
    print(f"  trunk height      {height:.1f} mm")
    print(f"  clipping          {'NONE - all %d pairs clear' % len(sc.pairs) if not clash else '%d PAIR(S)' % len(clash)}")
    for a, b in clash[:6]:
        print(f"                      {a} into {b}")
    print(f"  touching the floor ({len(on_floor)}): {', '.join(on_floor) or 'nothing'}")

    raw = {}
    for leg in LEGS:
        for seg in SEGMENTS:
            name = f"{leg}_{seg}"
            adr = model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            raw[name] = float(data.qpos[adr])

    if args.settle:
        settled = settle(model, data, limp=args.limp)
        print(f"  under gravity{' (LIMP)' if args.limp else '':7s} trunk {settled['trunk_mm']:.1f} mm, "
              f"level {settled['level']:+.3f}, "
              f"{'still' if settled['moved_mm_s'] < 2 else 'still moving'}")
        print(f"                    ends up on: {', '.join(settled['touching'])}")
        build(model, data, signs, angles)      # back to the commanded pose to render

    shots = render(model, data, args.name)
    print(f"  rendered          {', '.join(shots)}")

    poses = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as fh:
            poses = json.load(fh)
    poses[args.name] = {
        "angles_rad": raw,
        "physical_deg": angles,
        "trunk_height_mm": height,
        "self_collision_free": not clash,
        "touching_floor": on_floor,
        "note": args.note,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(poses, fh, indent=2)
    print(f"  saved to          {OUT_JSON}")


if __name__ == "__main__":
    main()
