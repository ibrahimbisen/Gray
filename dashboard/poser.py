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
import re
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

# ---------------------------------------------------------------------------
# EVERY MuJoCo CALL RUNS ON ONE THREAD, AND IT IS THIS ONE.
#
# `mujoco.Renderer` in _build() creates an OpenGL context, and an OpenGL context
# belongs to the thread that created it. Nothing else may touch it.
#
# That was free while the dashboard answered one request at a time. server.py now
# answers on a pool of threads - it had to, because one slow call used to freeze
# every page - so a render arrives on whichever thread is idle. The first request
# builds the context on thread A; the second one lands on thread B and the driver
# refuses it:
#
#     GLFWError: (65544) WGL: Failed to make context current: The handle is invalid
#
# The pose editor stops redrawing and the hero render never arrives. A lock does
# NOT fix it: the problem is not two threads at once, it is the wrong thread at
# all. One worker, for the life of the process, is what a GL context wants.
#
# Submit from OUTSIDE _LOCK. The worker takes _LOCK itself, so a caller holding
# it while waiting on the worker would deadlock the pair.
# ---------------------------------------------------------------------------
_GL: "concurrent.futures.ThreadPoolExecutor | None" = None
_GL_LOCK = threading.Lock()


def _on_gl(fn, *args, **kwargs):
    """Run fn on the one thread that owns the OpenGL context, and wait for it."""
    global _GL  # noqa: PLW0603
    if _GL is None:
        with _GL_LOCK:
            if _GL is None:
                import atexit  # noqa: PLC0415
                import concurrent.futures  # noqa: PLC0415

                _GL = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="poser-gl")
                # A backstop for anything that is not the dashboard - a script,
                # a test. serve() calls shutdown() properly; this catches the
                # rest, and does nothing if the executor has already gone.
                atexit.register(shutdown)
    return _GL.submit(fn, *args, **kwargs).result()


def shutdown() -> None:
    """Give the renderer back on the thread that owns it, before the process ends.

    The same rule as everywhere else in this block, at the one moment it is
    easiest to forget. Left alone, Python joins the worker at interpreter exit and
    then collects what is left on the MAIN thread - so the renderer's destructor
    tore down a GL context from a thread that never owned it, and the process died
    with an access violation (0xC0000005) straight after printing "Stopped.".

    CALLED EXPLICITLY, from serve(). An atexit hook is too late: CPython runs
    threading._shutdown() BEFORE atexit callbacks, so by the time an atexit
    handler asks the executor to do one last thing, the executor is already dead
    and the request is refused. The server owns this process, so the server says
    when the renderer goes.

    Safe to call twice, and safe to call when nothing was ever rendered.
    """
    global _STATE, _GL  # noqa: PLW0603
    if _GL is None:
        return

    def _drop() -> None:
        global _STATE  # noqa: PLW0603
        s, _STATE = _STATE, None
        renderer = (s or {}).get("renderer")
        close = getattr(renderer, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001, S110
                pass          # exiting anyway; the OS reclaims it either way

    try:
        _GL.submit(_drop).result(timeout=5)
    except Exception:  # noqa: BLE001, S110
        pass
    _GL.shutdown(wait=True)
    _GL = None
# A separate lock for robot.yaml. _LOCK guards the MuJoCo model and is held for
# the length of a render; a save must not queue behind one, and read-then-write
# must not interleave with the other save.
_FILE_LOCK = threading.Lock()


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


def stance_deg() -> dict[str, float]:
    """The standing pose from robot.yaml, in the pose editor's own convention.

    The same twelve numbers every task spawns the robot in. Read from the file
    rather than repeated here, so the hero picture cannot drift away from the
    pose the robot is actually trained in.
    """
    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load((ROOT / "gray" / "config" / "robot.yaml").read_text())
    stated = (cfg.get("stance") or {}).get("angles_deg") or {}
    return {f"{leg}_{seg}": float(stated.get(f"{leg}_{seg}", 0.0))
            for leg in LEGS for seg in SEGS}


# ------------------------------------------------------------------ the frame --
#
# Which way is which, drawn on the robot itself. Every number the policy reads or
# writes is in this frame - projected_gravity, base_ang_vel, the velocity command,
# every reward that mentions a direction - and it is the one thing about the robot
# that no amount of staring at a render will tell you.
#
# Right-handed, fixed to the trunk, moving with it. X out of the nose, Z up, and
# Y is then forced: left. Each rotation is named after the axis it turns about,
# positive by the right-hand rule.

# The last number is which side of its own arc the rotation's name sits on. It is
# hand-set per axis rather than derived: at this camera the three arcs and the
# three axis labels are close enough that any single rule puts two of them on top
# of each other, and an unreadable label is worse than a magic number.
AXES = (
    # key   axis           length  arc r  colour  label         turning about it
    ("x", (1.0, 0.0, 0.0), 0.270, 0.058, "--s2", "+X forward", "roll",
     "leans left or right, nose stays put", -1),
    ("y", (0.0, 1.0, 0.0), 0.245, 0.058, "--s3", "+Y left", "pitch",
     "nose tips up or down", +1),
    ("z", (0.0, 0.0, 1.0), 0.230, 0.058, "--s1", "+Z up", "yaw",
     "nose swings left or right, trunk stays level", -1),
)


def _projector(cam, width: int, height: int, fovy_deg: float):
    """World point -> pixel in the rendered image.

    MuJoCo will not hand this back, so it is rebuilt from the camera it was
    given: azimuth and elevation make the viewing direction, distance and lookat
    put the eye behind it, and the vertical field of view sets the scale. Same
    convention as the free camera - azimuth 0 looks along +X, negative elevation
    looks down.
    """
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    fwd = np.array([math.cos(el) * math.cos(az),
                    math.cos(el) * math.sin(az),
                    math.sin(el)])
    eye = np.asarray(cam.lookat, dtype=float) - cam.distance * fwd
    right = np.cross(fwd, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    focal = (height / 2) / math.tan(math.radians(fovy_deg) / 2)

    def project(p) -> list[float] | None:
        v = np.asarray(p, dtype=float) - eye
        depth = float(v @ fwd)
        if depth <= 1e-6:                       # behind the camera
            return None
        return [round(width / 2 + focal * float(v @ right) / depth, 1),
                round(height / 2 - focal * float(v @ up) / depth, 1)]

    return project


def _perp(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors at right angles to `axis`, and to each other."""
    seed = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(seed, axis)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def frame_hero(azimuth: float = 128.0, elevation: float = -24.0,
               distance: float = 1.18) -> tuple[bytes, dict]:
    """The robot standing, with its own axes and rotations drawn on it.

    Returns (png, overlay). The overlay is pixel coordinates in the image's own
    frame, so the page draws the arrows as SVG on top: crisp text at any size,
    and no second render when the picture is resized.
    """
    s = state()
    rb, mj = s["rb"], s["mujoco"]

    with _LOCK:
        rb.set_pose(0.6, to_raw(s, stance_deg()))
        rb.d.qpos[2] -= lowest_point(s)         # drop it onto the floor
        mj.mj_forward(rb.m, rb.d)

        trunk = float(rb.d.xpos[rb.base][2])
        # The middle IMU's spot, which is where projected_gravity and base_ang_vel
        # are defined. Every arrow starts here because that is where the policy
        # thinks it is standing.
        # Copied, not viewed: rb.d.xpos is live MuJoCo memory and the next render
        # would move it underneath the arrows.
        origin = np.array(rb.d.xpos[rb.base], dtype=float)

        cam = s["cam"]
        cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
        # Aimed above the trunk, not at half its height like the pose editor: the
        # +Z arm and the yaw ring stand well over the robot, and aiming at the
        # body alone leaves the picture bottom-heavy with empty floor.
        cam.lookat[:] = (0.0, 0.0, max(trunk, 0.05) * 0.88)
        s["renderer"].update_scene(rb.d, cam)
        frame = s["renderer"].render()
        h, w = frame.shape[0], frame.shape[1]
        project = _projector(cam, w, h, float(rb.m.vis.global_.fovy))

        axes = []
        for key, vec, length, radius, colour, label, turn, meaning, side in AXES:
            a = np.asarray(vec, dtype=float)
            u, v = _perp(a)
            # Out at the far end of the arm, clear of the trunk and of the other
            # two arcs. Bunched near the origin they overlap into a knot.
            centre = origin + a * (length * 0.86)
            # Open at 300 degrees rather than closed, so the gap and the arrowhead
            # together say which way positive goes.
            arc = [project(centre + radius * (math.cos(t) * u + math.sin(t) * v))
                   for t in np.radians(np.linspace(-150, 150, 61))]
            axes.append({
                "key": key,
                "colour": colour,
                "label": label,
                "turn": turn,
                "meaning": meaning,
                "tip": project(origin + a * length),
                "far": project(origin + a * (length * 1.13)),
                "arc": [p for p in arc if p],
                # Where the rotation's name goes: off the arc, clear of the ink.
                "turn_at": project(centre + (side * radius * 1.75) * v),
            })
        centre_px = project(origin)

    import imageio.v2 as imageio  # noqa: PLC0415
    buf = io.BytesIO()
    imageio.imwrite(buf, np.asarray(frame), format="png")

    return buf.getvalue(), {
        "w": w, "h": h,
        "origin": centre_px,
        "axes": axes,
        "trunk_mm": round(trunk * 1000, 1),
        "pose": "the standing pose in robot.yaml, dropped onto the floor",
    }


_HERO: tuple[tuple, bytes, dict] | None = None


def frame_hero_cached() -> tuple[bytes, dict]:
    """The same picture, rendered once.

    Nothing about it moves: one fixed pose, one fixed camera. It is on the first
    tab of /robot, so without this every visit would pay for a MuJoCo render -
    and the first one also builds the collision meshes, which takes seconds.
    Keyed on the two files it is drawn from, so editing either redraws it.
    """
    global _HERO
    stamp = tuple(
        p.stat().st_mtime if p.exists() else 0.0
        for p in (ROOT / "sim" / "models" / "gray.xml",
                  ROOT / "gray" / "config" / "robot.yaml"))
    if _HERO is None or _HERO[0] != stamp:
        # On the GL thread, like every other render. Two pages ask for this one -
        # /robot leads with it and /summary's sheet header now carries it - so on
        # a cold cache two threads can arrive together; the single worker
        # serialises them and the second finds the cache warm.
        png, overlay = _on_gl(frame_hero)
        _HERO = (stamp, png, overlay)
    return _HERO[1], _HERO[2]


def pose_report(physical_deg: dict[str, float], azimuth: float = 125.0,
                elevation: float = -12.0, distance: float = 1.05,
                invert: dict[str, bool] | None = None) -> tuple[bytes, dict]:
    """Render the pose and measure it. Returns (png, facts).

    A thin shell over _pose_report so the render happens on the GL thread. See
    the note beside _on_gl: this is the entry the pose editor hits on every
    drag, several times a second, from whatever thread the server picked.
    """
    return _on_gl(_pose_report, physical_deg, azimuth, elevation, distance, invert)


def _pose_report(physical_deg: dict[str, float], azimuth: float,
                 elevation: float, distance: float,
                 invert: dict[str, bool] | None) -> tuple[bytes, dict]:
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


ROBOT_YAML = ROOT / "gray" / "config" / "robot.yaml"

# A line that starts a new top-level thing: a key, or a comment that introduces
# the block below it. Anything indented belongs to the block above.
_TOP = re.compile(r"^\S")


def _replace_block(text: str, key: str, block) -> str:
    """Put one top-level block back into the file, and touch nothing else.

    robot.yaml is 250 lines, and 200 of them are comments that record where each
    number came from. A dump of the whole file deletes every one of them, so
    only the block that changed gets rewritten, in place.

    The two save functions used to do `text.split(key)[0] + dump(block)`. That
    keeps the head of the file and THROWS AWAY EVERYTHING BELOW THE BLOCK. Save
    the joint reversals, then save the joint travel, and the reversals are gone -
    the second save cut the file at `joint_limits:` and the reversals sat under
    it. Nothing said so.

    A key that is not in the file yet goes on the end.
    """
    import yaml  # noqa: PLC0415

    body = yaml.safe_dump({key: block}, sort_keys=False, allow_unicode=True)
    lines = text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if ln.startswith(f"{key}:")), None)
    if start is None:
        return text.rstrip("\n") + "\n\n" + body

    end = start + 1
    while end < len(lines) and not _TOP.match(lines[end]):
        end += 1
    # The blank lines above the next block are the gap between the two, not the
    # tail of this one. Leave them where they are.
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return "".join(lines[:start]) + body + "".join(lines[end:])


def save_directions(inverted: list[str]) -> list[str]:
    """Record which joints turn the opposite way to what the model measured.

    tools/prepare_model.py reads this and negates those joints' axes, so the
    correction lives in the model rather than only in this page.
    """
    clean = sorted({j for j in inverted if j.count("_") == 1})
    block = {
        "note": "Joints whose measured direction disagreed with the owner's eye. "
                "prepare_model.py negates the axis of each one.",
        "invert": clean,
    }
    with _FILE_LOCK:
        text = ROBOT_YAML.read_text(encoding="utf-8")
        ROBOT_YAML.write_text(
            _replace_block(text, "joint_direction", block), encoding="utf-8")
    return clean


def save_limits(new: dict[str, list[float]]) -> dict:
    """Write measured travel back to robot.yaml, keeping the file's comments."""
    import yaml  # noqa: PLC0415

    with _LOCK:
        text = ROBOT_YAML.read_text(encoding="utf-8")
        limits = yaml.safe_load(text)["joint_limits"]
        for seg in SEGS:
            if seg in new:
                lo, hi = sorted(float(v) for v in new[seg])
                limits[seg] = [round(lo, 1), round(hi, 1)]
        limits["source"] = "measured in the pose editor against the CAD"
        ROBOT_YAML.write_text(
            _replace_block(text, "joint_limits", limits), encoding="utf-8")
    return limits
