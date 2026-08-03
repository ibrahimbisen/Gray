"""Pose the robot by hand, and read the numbers back.

The joint travel currently in gray/config/robot.yaml was stated from memory of a
robot that is in pieces. Applied to this model it produces a robot standing on
nearly straight legs at 308 mm, which is not how a quadruped stands - so the
numbers need checking against the geometry rather than against memory.

This drives the model directly: twelve dials, a live render, and a readout of
what the pose actually is. When a leg hits the body, that is the mechanical limit,
and it is measured rather than recalled.

**Zero is the sitting pose** - the pose the CAD was in when it was exported. That
is a real, reproducible reference the owner chose, unlike anything inferred from
the geometry, and every dial reads as degrees away from sitting.

Directions are physical, and measured off the model rather than assumed, because
the legs are mirrored and half of them turn the opposite way for the same command:

    hip    + swings the leg OUT, away from the body
    thigh  + swings the leg FORWARD, towards the nose
    calf   + lifts the foot UP
"""

from __future__ import annotations

import io
import math
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

LEGS = ("fr", "fl", "br", "bl")
SEGS = ("hip", "thigh", "calf")
_LOCK = threading.Lock()
_STATE: dict | None = None


def _build() -> dict:
    import mujoco  # noqa: PLC0415
    from find_stance import Robot, joint_signs  # noqa: PLC0415

    rb = Robot()
    signs = joint_signs(rb)   # which way each joint is physically positive
    renderer = mujoco.Renderer(rb.m, height=560, width=880)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    return {"rb": rb, "signs": signs, "renderer": renderer,
            "cam": cam, "mujoco": mujoco}


def state() -> dict:
    global _STATE
    with _LOCK:
        if _STATE is None:
            _STATE = _build()
        return _STATE


def to_raw(s: dict, physical_deg: dict[str, float],
           invert: dict[str, bool] | None = None) -> dict[tuple[str, str], float]:
    """Degrees away from sitting -> the model's own joint angles.

    `invert` flips individual joints. The direction of each joint is measured off
    the model at startup, but that measurement assumes every leg is built the same
    way round - and on this robot some are not. Where the measurement disagrees
    with what the owner sees, the owner is right.
    """
    invert = invert or {}
    out = {}
    for leg in LEGS:
        for seg in SEGS:
            name = f"{leg}_{seg}"
            v = float(physical_deg.get(name, 0.0))
            flip = -1.0 if invert.get(name) else 1.0
            out[(leg, seg)] = flip * s["signs"][(leg, seg)] * math.radians(v)
    return out


def pose_report(physical_deg: dict[str, float], azimuth: float = 125.0,
                elevation: float = -12.0, distance: float = 1.05,
                invert: dict[str, bool] | None = None) -> tuple[bytes, dict]:
    """Render the pose and measure it. Returns (png, facts)."""
    s = state()
    rb, mj = s["rb"], s["mujoco"]

    with _LOCK:
        angles = to_raw(s, physical_deg, invert)
        rb.set_pose(0.6, angles)

        # Drop it until the lowest point of the robot rests on the floor, so the
        # trunk height means "how tall it stands" rather than "where I parked it".
        lowest = min(
            (float(rb.d.geom_xpos[g][2]) for g in range(rb.m.ngeom)
             if rb.m.geom_type[g] != mj.mjtGeom.mjGEOM_PLANE),
            default=0.0,
        )
        rb.d.qpos[2] -= lowest
        mj.mj_forward(rb.m, rb.d)

        feet = {leg: rb.foot_world(leg) for leg in LEGS}
        floor = min(f[2] for f in feet.values())
        rb.d.qpos[2] -= floor
        mj.mj_forward(rb.m, rb.d)

        # A leg touching the body IS the mechanical limit. Ignore the floor.
        floor_geoms = {g for g in range(rb.m.ngeom)
                       if rb.m.geom_type[g] == mj.mjtGeom.mjGEOM_PLANE}
        clashes = []
        for i in range(rb.d.ncon):
            c = rb.d.contact[i]
            if c.geom1 in floor_geoms or c.geom2 in floor_geoms:
                continue
            b1 = rb.m.geom_bodyid[c.geom1]
            b2 = rb.m.geom_bodyid[c.geom2]
            n1 = mj.mj_id2name(rb.m, mj.mjtObj.mjOBJ_BODY, b1)
            n2 = mj.mj_id2name(rb.m, mj.mjtObj.mjOBJ_BODY, b2)
            pair = tuple(sorted((n1 or "?", n2 or "?")))
            if pair not in clashes:
                clashes.append(pair)

        feet = {leg: rb.foot_world(leg) for leg in LEGS}
        trunk = float(rb.d.xpos[rb.base][2])
        span_x = float(max(f[0] for f in feet.values()) - min(f[0] for f in feet.values()))
        span_y = float(max(f[1] for f in feet.values()) - min(f[1] for f in feet.values()))

        s["cam"].azimuth = azimuth
        s["cam"].elevation = elevation
        s["cam"].distance = distance
        s["cam"].lookat[:] = (0, 0, max(trunk, 0.05) / 2)
        s["renderer"].update_scene(rb.d, s["cam"])
        frame = s["renderer"].render()

    import imageio.v2 as imageio  # noqa: PLC0415
    buf = io.BytesIO()
    imageio.imwrite(buf, np.asarray(frame), format="png")

    return buf.getvalue(), {
        "trunk_mm": round(trunk * 1000, 1),
        "stance_x_mm": round(span_x * 1000, 1),
        "stance_y_mm": round(span_y * 1000, 1),
        "feet_mm": {leg: [round(float(v) * 1000, 1) for v in p] for leg, p in feet.items()},
        "clashes": [" against ".join(p) for p in clashes],
        "colliding": bool(clashes),
    }


def defaults() -> dict:
    """Everything the page needs to draw itself."""
    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load((ROOT / "gray" / "config" / "robot.yaml").read_text())
    lim = cfg["joint_limits"]
    return {
        "legs": list(LEGS),
        "segments": list(SEGS),
        "current": {k: list(lim[k]) for k in SEGS},
        "convention": lim["convention"],
        "servo_travel_deg": cfg["servo"]["travel_deg"],
        "zero": "the sitting pose, as exported from SolidWorks",
        "invert": cfg.get("joint_direction", {}).get("invert", []),
    }


def save_directions(inverted: list[str]) -> list[str]:
    """Record which joints turn the opposite way to what the model measured.

    tools/prepare_model.py reads this and negates those joints' axes, so the
    correction lives in the model rather than only in this page.
    """
    import yaml  # noqa: PLC0415

    path = ROOT / "gray" / "config" / "robot.yaml"
    text = path.read_text()
    cfg = yaml.safe_load(text)
    clean = sorted({j for j in inverted if j.count("_") == 1})
    cfg["joint_direction"] = {
        "note": "Joints whose measured direction disagreed with the owner's eye. "
                "prepare_model.py negates the axis of each one.",
        "invert": clean,
    }
    head = text.split("\njoint_direction:")[0].rstrip() + "\n\n"
    path.write_text(head + yaml.safe_dump(
        {"joint_direction": cfg["joint_direction"]}, sort_keys=False))
    return clean


def save_limits(new: dict[str, list[float]]) -> dict:
    """Write measured travel back to robot.yaml, keeping the file's comments."""
    import yaml  # noqa: PLC0415

    path = ROOT / "gray" / "config" / "robot.yaml"
    cfg = yaml.safe_load(path.read_text())
    for seg in SEGS:
        if seg in new:
            lo, hi = sorted(float(v) for v in new[seg])
            cfg["joint_limits"][seg] = [round(lo, 1), round(hi, 1)]
    cfg["joint_limits"]["source"] = "measured in the pose editor against the CAD"
    text = path.read_text()
    head = text.split("joint_limits:")[0]
    path.write_text(head + yaml.safe_dump(
        {"joint_limits": cfg["joint_limits"]}, sort_keys=False))
    return cfg["joint_limits"]
