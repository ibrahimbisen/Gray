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


# Collision meshes are thinned, because FCL walks a bounding-volume tree and this
# runs on every dial movement. Not thinned too far, though: a decimated surface
# reports overlaps a millimetre or two off, and that is the same size as a real
# graze. 12,000 triangles holds the error well under the 1.5 mm margin below and
# still answers in well under a tenth of a second.
COLLISION_TRIANGLES = 12_000


def _collision_meshes(rb) -> dict:
    """One thinned mesh per link, in the link's own frame."""
    import trimesh  # noqa: PLC0415

    import mujoco  # noqa: PLC0415

    meshes = {}
    for b in range(rb.m.nbody):
        parts = []
        for g in range(rb.m.body_geomadr[b], rb.m.body_geomadr[b] + rb.m.body_geomnum[b]):
            if rb.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = rb.m.geom_dataid[g]
            adr, num = rb.m.mesh_vertadr[mid], rb.m.mesh_vertnum[mid]
            fadr, fnum = rb.m.mesh_faceadr[mid], rb.m.mesh_facenum[mid]
            verts = rb.m.mesh_vert[adr:adr + num].astype(float)
            faces = rb.m.mesh_face[fadr:fadr + fnum].astype(int)
            # A mesh sits at an offset inside its body. Without applying it, every
            # part ends up centred on its link's origin and nothing lines up.
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, rb.m.geom_quat[g])
            verts = verts @ rot.reshape(3, 3).T + rb.m.geom_pos[g]
            parts.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))
        if not parts:
            continue
        mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
        mesh.merge_vertices()
        if len(mesh.faces) > COLLISION_TRIANGLES:
            mesh = mesh.simplify_quadric_decimation(face_count=COLLISION_TRIANGLES)
        meshes[b] = mesh
    return meshes


def _build() -> dict:
    import mujoco  # noqa: PLC0415
    import trimesh  # noqa: PLC0415
    from find_stance import Robot, joint_signs  # noqa: PLC0415

    rb = Robot()
    signs = joint_signs(rb)   # which way each joint is physically positive
    renderer = mujoco.Renderer(rb.m, height=560, width=880)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    meshes = _collision_meshes(rb)
    manager = trimesh.collision.CollisionManager()
    for b, mesh in meshes.items():
        manager.add_object(str(b), mesh)

    s = {"rb": rb, "signs": signs, "renderer": renderer, "cam": cam,
         "mujoco": mujoco, "meshes": meshes, "manager": manager, "baseline": {}}

    # Parts that already overlap at the sitting pose are overlapping by design -
    # the knee pivot has the thigh and calf interleaved, and they never separate.
    # Record how deep that is, so only NEW penetration counts as a clash.
    rb.set_pose(0.6, {})
    s["baseline"] = _penetration(s)
    return s


def _body_lows(s: dict) -> dict[int, float]:
    """The lowest point of every link, in metres. Real vertices, not geom centres."""
    rb = s["rb"]
    return {b: float((mesh.vertices @ rb.d.xmat[b].reshape(3, 3).T
                      + rb.d.xpos[b])[:, 2].min())
            for b, mesh in s["meshes"].items()}


def lowest_point(s: dict) -> float:
    """Height of the lowest point on the whole robot, in metres."""
    lows = _body_lows(s)
    return min(lows.values()) if lows else 0.0


def touching(s: dict, tol: float = 0.004) -> tuple[list[str], list[str]]:
    """Which legs reach the ground, and which links are the ones actually on it.

    Not the same question as "is the toe down". Folded up, this robot rests on the
    side of its calves with its toes in the air, and calling that zero feet down
    is true but useless.
    """
    rb, mj = s["rb"], s["mujoco"]
    legs, parts = [], []
    for b, low in _body_lows(s).items():
        if low > tol:
            continue
        name = mj.mj_id2name(rb.m, mj.mjtObj.mjOBJ_BODY, b) or str(b)
        parts.append(name)
        leg = next((l for l in LEGS if name.lower().startswith(l)), None)
        if leg and leg not in legs:
            legs.append(leg)
    return legs, parts


def _penetration(s: dict) -> dict[tuple[str, str], float]:
    """How deeply each pair of links overlaps, in metres, at the current pose.

    MuJoCo cannot answer this: it skips any parent-child pair outright, so a calf
    folding into its own thigh is never reported, and where it does test meshes it
    tests their convex hulls - far too fat to trust on a bent leg.
    """
    rb = s["rb"]
    manager = s["manager"]
    for b in s["meshes"]:
        t = np.eye(4)
        t[:3, :3] = rb.d.xmat[b].reshape(3, 3)
        t[:3, 3] = rb.d.xpos[b]
        manager.set_transform(str(b), t)

    hit, names, data = manager.in_collision_internal(return_names=True, return_data=True)
    if not hit:
        return {}
    depth: dict[tuple[str, str], float] = {}
    for c in data:
        pair = tuple(sorted(c.names))
        depth[pair] = max(depth.get(pair, 0.0), float(getattr(c, "depth", 0.0) or 0.0))
    for pair in names:
        depth.setdefault(tuple(sorted(pair)), 0.0)
    return depth


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

        # Drop it until the lowest point of the robot rests on the floor, so trunk
        # height means "how tall it stands" rather than "where I parked it".
        # Measured on the actual mesh vertices - a geom's centre is not its bottom,
        # and whatever touches down first is not always a foot.
        rb.d.qpos[2] -= lowest_point(s)
        mj.mj_forward(rb.m, rb.d)

        # A part touching another part IS the mechanical limit. Measured on the
        # real triangles, and only counting overlap beyond what the assembly
        # already has at rest.
        MARGIN = 0.0015  # 1.5 mm of new overlap before it counts
        now = _penetration(s)
        clashes = []
        for pair, d in sorted(now.items(), key=lambda kv: -kv[1]):
            if d - s["baseline"].get(pair, 0.0) <= MARGIN:
                continue
            n1 = mj.mj_id2name(rb.m, mj.mjtObj.mjOBJ_BODY, int(pair[0])) or pair[0]
            n2 = mj.mj_id2name(rb.m, mj.mjtObj.mjOBJ_BODY, int(pair[1])) or pair[1]
            clashes.append((n1, n2, (d - s["baseline"].get(pair, 0.0)) * 1000))

        feet = {leg: rb.foot_world(leg) for leg in LEGS}
        trunk = float(rb.d.xpos[rb.base][2])
        legs_down, parts_down = touching(s)
        toes_down = [leg for leg, p in feet.items() if p[2] < 0.004]
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
        "clashes": [f"{a} into {b} by {d:.1f} mm" for a, b, d in clashes],
        "colliding": bool(clashes),
        "legs_down": len(legs_down),
        "toes_down": len(toes_down),
        "on_ground": sorted(parts_down),
        "toe_height_mm": {leg: round(float(p[2]) * 1000, 1) for leg, p in feet.items()},
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
