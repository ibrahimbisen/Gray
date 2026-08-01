#!/usr/bin/env python3
"""Render what each joint actually does, and how much of it the walk uses.

    python scripts/render_joint_atlas.py

Writes to progress/joints/:

    hip.mp4  top.mp4  bottom.mp4   one joint sweeping its FULL travel
    ranges.json                    used-vs-available travel, per joint

WHY THIS EXISTS
---------------
Watching the walk, the legs barely seem to articulate - and the measurements agree,
but not for the reason it looks like. The servos have 270 degrees of travel and the
crawl gait asks for at most 55, while the four abduction joints are asked for
EXACTLY NOTHING: they hold one angle for the entire walk. Four of the twelve servos
are, today, dead weight.

Seeing that is the point. These clips pose the model kinematically - no physics, no
gravity, trunk pinned - so each joint's axis of rotation is unmistakable, and the
overlaid figures say how much of that travel the gait ever calls on.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gray.gait import GaitGenerator, GaitParams          # noqa: E402
from gray.kinematics import LEGS, joint_vector, load_legs  # noqa: E402

MJCF = "sim/models/gray.xml"
OUT = os.path.join("progress", "joints")
SEGMENTS = ("hip", "top", "bottom")
PLAIN = {
    "hip": ("Hip / abduction", "swings the whole leg out sideways, about the "
                               "fore-aft axis"),
    "top": ("Thigh / hip pitch", "swings the leg forwards and backwards"),
    "bottom": ("Knee", "folds the lower leg, sharing the thigh's axis"),
}
# Sweep the front-left leg: it is nearest the default camera and its motion is not
# hidden behind the trunk.
LEG = "fl"


def used_ranges() -> dict:
    """How much of each joint the classical gait actually calls on."""
    legs = load_legs()
    params = GaitParams()
    gait = GaitGenerator(legs, params)
    ts = np.linspace(0.0, 4.0 * params.period, 800)
    q = np.array([joint_vector(gait.joint_angles(t, 1.0)) for t in ts])

    out = {}
    for i, name in enumerate([f"{l}_{s}" for l in LEGS for s in SEGMENTS]):
        lo, hi = float(np.degrees(q[:, i].min())), float(np.degrees(q[:, i].max()))
        out[name] = {"used_min_deg": round(lo, 2), "used_max_deg": round(hi, 2),
                     "used_span_deg": round(hi - lo, 2)}
    return out


def main() -> int:
    import mujoco
    import imageio.v2 as iio

    os.makedirs(OUT, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)

    legs = load_legs()
    params = GaitParams()
    stand = joint_vector(GaitGenerator(legs, params).stand())

    joint_names = [f"{l}_{s}" for l in LEGS for s in SEGMENTS]
    ranges = used_ranges()

    renderer = mujoco.Renderer(model, height=560, width=760)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 0.62, -12, 35
    cam.lookat[:] = [0.09, 0.09, 0.30]

    # Show the joint axes and hide the collision primitives, so what is on screen is
    # the real geometry plus the thing being explained.
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.frame = mujoco.mjtFrame.mjFRAME_NONE
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    manifest = {"legs": LEG, "joints": {}}

    for seg in SEGMENTS:
        name = f"{LEG}_{seg}"
        idx = joint_names.index(name)
        lo, hi = model.jnt_range[idx + 1]        # +1: index 0 is the free joint

        # Full travel out and back, so the axis reads clearly in a short loop.
        sweep = np.concatenate([
            np.linspace(stand[idx], hi, 45),
            np.linspace(hi, lo, 90),
            np.linspace(lo, stand[idx], 45),
        ])

        frames = []
        for value in sweep:
            q = stand.copy()
            q[idx] = value
            # Pose it kinematically. No stepping, no gravity - the trunk stays put
            # and only the joint under discussion moves.
            data.qpos[:3] = [0.0, 0.0, 0.30]
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            data.qpos[7:] = q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam, opt)
            frames.append(renderer.render())

        path = os.path.join(OUT, f"{seg}.mp4")
        iio.mimwrite(path, frames, fps=30, quality=8, macro_block_size=1)

        title, how = PLAIN[seg]
        span_used = ranges[name]["used_span_deg"]
        span_avail = float(np.degrees(hi - lo))
        manifest["joints"][seg] = {
            "title": title,
            "how_it_moves": how,
            "video": f"/media/progress/joints/{seg}.mp4",
            "servo_min_deg": round(float(np.degrees(lo)), 1),
            "servo_max_deg": round(float(np.degrees(hi)), 1),
            "servo_span_deg": round(span_avail, 1),
            "gait_span_deg": span_used,
            "percent_used": round(span_used / span_avail * 100, 1),
            "count": 4,
        }
        print(f"{title:20s} sweeps {span_avail:.0f} deg, the walk uses "
              f"{span_used:.1f} deg ({span_used/span_avail*100:.1f}%)  -> {path}")

    manifest["per_joint"] = ranges
    manifest["note"] = (
        "The four abduction servos never move during the walk: the crawl gait only "
        "swings feet fore-aft and up-down, so a foot is never asked to move "
        "sideways. Four of the twelve servos are currently dead weight."
    )
    with open(os.path.join(OUT, "ranges.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nwrote {OUT}/ranges.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
