"""Check a URDF exported from SolidWorks against everything Gray needs.

Run it after every export. It answers one question: is this model good enough to
build on, and if not, exactly which number is wrong.

    python tools/check_urdf.py sim/models/gray.urdf
    python tools/check_urdf.py sim/models/gray.urdf --json

Every check that fails prints the measured value, because "joint axes are wrong"
is not actionable and "fl_hip axis is 0.86 deg off Z" is.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# What the robot is supposed to be. The names are load-bearing: everything
# downstream indexes legs by these prefixes.
LEGS = ("fl", "fr", "bl", "br")
SEGMENTS = ("hip", "top", "bottom")
EXPECTED_LINKS = ("base_link",) + tuple(f"{l}_{s}" for l in LEGS for s in SEGMENTS)
EXPECTED_JOINTS = tuple(f"{l}_{s}" for l in LEGS for s in SEGMENTS)

# Mass bounds, kg. Printed structure plus 12 servos plus battery, Pi and wiring.
# Anything outside this means the CAD is missing components or double-counting.
MASS_MIN, MASS_MAX = 1.2, 2.6

# A joint axis should be a coordinate axis. The old export had axes 0.86 deg off,
# which is CAD misalignment leaking into the physics.
AXIS_TOL_DEG = 0.10


@dataclass
class Check:
    key: str
    title: str
    why: str
    passed: bool = False
    detail: str = ""
    rows: list[str] = field(default_factory=list)


def _vec(text: str) -> tuple[float, float, float]:
    parts = [float(p) for p in text.split()]
    return parts[0], parts[1], parts[2]


def _norm(v):
    return math.sqrt(sum(c * c for c in v))


def _rpy_matrix(r: float, p: float, y: float):
    """URDF fixed-axis roll-pitch-yaw, applied X then Y then Z."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _mat_mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _mat_vec(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def _angle_to_nearest_axis_deg(v) -> tuple[float, str]:
    """Smallest angle between v and any of +/-X, +/-Y, +/-Z, plus that axis name."""
    n = _norm(v)
    if n == 0:
        return 180.0, "?"
    unit = [c / n for c in v]
    best, name = 180.0, "?"
    for i, axis in enumerate("xyz"):
        for sign in (1, -1):
            dot = max(-1.0, min(1.0, unit[i] * sign))
            ang = math.degrees(math.acos(abs(dot))) if abs(dot) > 0 else 90.0
            ang = math.degrees(math.acos(max(-1.0, min(1.0, unit[i] * sign))))
            if ang < best:
                best, name = ang, ("+" if sign > 0 else "-") + axis.upper()
    return best, name


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------


def check_parses(root, urdf_path, checks):
    c = Check(
        "parses",
        "Parses as a URDF",
        "If this fails nothing else can run.",
    )
    c.passed = root is not None and root.tag == "robot"
    c.detail = f"robot name: {root.get('name')!r}" if c.passed else "not a <robot> document"
    checks.append(c)


def check_structure(root, checks):
    c = Check(
        "structure",
        "13 links, 12 joints, named correctly",
        "Everything downstream finds a leg by its name prefix.",
    )
    links = [l.get("name") for l in root.findall("link")]
    joints = [j.get("name") for j in root.findall("joint")]
    missing_l = [n for n in EXPECTED_LINKS if n not in links]
    missing_j = [n for n in EXPECTED_JOINTS if n not in joints]
    extra_l = [n for n in links if n not in EXPECTED_LINKS]
    c.passed = not (missing_l or missing_j)
    c.detail = f"{len(links)} links, {len(joints)} joints"
    for n in missing_l:
        c.rows.append(f"missing link: {n}")
    for n in missing_j:
        c.rows.append(f"missing joint: {n}")
    for n in extra_l:
        c.rows.append(f"unexpected link: {n} (harmless, but nothing will use it)")
    checks.append(c)


def check_axes(root, checks):
    c = Check(
        "axes",
        "Joint axes are exactly on a coordinate axis",
        "A tilted axis is CAD misalignment leaking into the physics. Mate to "
        "reference planes, not to faces.",
    )
    worst = 0.0
    for j in root.findall("joint"):
        axis_el = j.find("axis")
        if axis_el is None:
            c.rows.append(f"{j.get('name')}: no <axis> at all")
            worst = 180.0
            continue
        ang, nearest = _angle_to_nearest_axis_deg(_vec(axis_el.get("xyz", "0 0 1")))
        worst = max(worst, ang)
        if ang > AXIS_TOL_DEG:
            c.rows.append(f"{j.get('name')}: {ang:.2f} deg off {nearest}")
    c.passed = worst <= AXIS_TOL_DEG
    c.detail = f"worst offset {worst:.3f} deg (limit {AXIS_TOL_DEG:.2f})"
    checks.append(c)


def check_limits(root, checks):
    c = Check(
        "limits",
        "All 12 joints have travel limits",
        "Without limits the simulated robot folds itself through its own body and "
        "learns a gait the real one physically cannot do.",
    )
    ok = 0
    wide: list[str] = []
    for j in root.findall("joint"):
        if j.get("type") in ("fixed", "floating"):
            continue
        lim = j.find("limit")
        if lim is None:
            c.rows.append(f"{j.get('name')}: no <limit>")
            continue
        lo, hi = lim.get("lower"), lim.get("upper")
        eff, vel = lim.get("effort"), lim.get("velocity")
        missing = [k for k, v in (("lower", lo), ("upper", hi), ("effort", eff), ("velocity", vel)) if v is None]
        if missing:
            c.rows.append(f"{j.get('name')}: missing {', '.join(missing)}")
            continue
        if float(hi) <= float(lo):
            c.rows.append(f"{j.get('name')}: upper {hi} is not above lower {lo}")
            continue
        if float(eff) <= 0 or float(vel) <= 0:
            c.rows.append(f"{j.get('name')}: effort/velocity must be above zero")
            continue
        span = math.degrees(float(hi) - float(lo))
        if span > 270.5:
            c.rows.append(f"{j.get('name')}: {span:.0f} deg of travel, more than the servo's 270")
            continue
        if span >= 269.5:
            wide.append(j.get("name"))
        ok += 1
    c.passed = ok == 12
    c.detail = f"{ok} of 12 joints fully specified"
    if wide:
        # Not a failure, but the full servo sweep is almost never the real limit -
        # the leg hits itself or the body long before 270 degrees.
        c.rows.append(
            f"{len(wide)} joint(s) are set to the servo's full 270 deg sweep. That is "
            "the exporter default, not a measured mechanical stop. Drag each joint in "
            "the assembly until it collides and use that angle instead."
        )
    checks.append(c)


def check_mass(root, checks):
    c = Check(
        "mass",
        "Mass is complete and believable",
        "The old model claimed 1.254 kg while 12 servos alone weigh 720 g. Mass "
        "that is wrong makes every torque and every reward wrong.",
    )
    total = 0.0
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            c.rows.append(f"{link.get('name')}: no <inertial> at all")
            continue
        m_el = inertial.find("mass")
        m = float(m_el.get("value")) if m_el is not None else 0.0
        if m <= 0:
            c.rows.append(f"{link.get('name')}: mass is {m}")
        total += m
    c.passed = MASS_MIN <= total <= MASS_MAX and not c.rows
    c.detail = f"total {total*1000:.0f} g (expected {MASS_MIN*1000:.0f}-{MASS_MAX*1000:.0f} g)"
    if not (MASS_MIN <= total <= MASS_MAX):
        c.rows.append(
            f"total mass {total:.3f} kg is outside the expected band - components "
            "are missing from the CAD, or counted twice"
        )
    checks.append(c)


def check_inertia(root, checks):
    c = Check(
        "inertia",
        "Inertia tensors are physically possible",
        "An impossible tensor makes the simulator behave in ways no real robot can, "
        "and it usually fails silently.",
    )
    bad = 0
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        i = inertial.find("inertia")
        if i is None:
            c.rows.append(f"{link.get('name')}: no <inertia>")
            bad += 1
            continue
        ixx, iyy, izz = (float(i.get(k, 0)) for k in ("ixx", "iyy", "izz"))
        if min(ixx, iyy, izz) <= 0:
            c.rows.append(f"{link.get('name')}: a diagonal term is zero or negative")
            bad += 1
        elif not (ixx + iyy >= izz and iyy + izz >= ixx and izz + ixx >= iyy):
            c.rows.append(f"{link.get('name')}: fails the triangle inequality")
            bad += 1
    c.passed = bad == 0
    c.detail = f"{bad} bad tensors" if bad else "all 13 links valid"
    checks.append(c)


def check_meshes(root, urdf_path, checks):
    c = Check(
        "meshes",
        "Every referenced mesh file exists",
        "A missing mesh means an invisible link, or a crash on load.",
    )
    base = urdf_path.parent
    seen, missing = 0, 0
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        seen += 1
        rel = fn.replace("package://", "").replace("model://", "")
        if not (base / rel).exists():
            c.rows.append(f"missing: {fn}")
            missing += 1
    c.passed = seen > 0 and missing == 0
    c.detail = f"{seen - missing} of {seen} found"
    checks.append(c)


def check_collision(root, checks):
    c = Check(
        "collision",
        "Every link has collision geometry",
        "A link with no collision shape passes through the floor and through the "
        "other legs. Feet especially.",
    )
    without = [l.get("name") for l in root.findall("link") if l.find("collision") is None]
    c.passed = not without
    c.detail = f"{len(root.findall('link')) - len(without)} of {len(root.findall('link'))} links"
    for n in without:
        c.rows.append(f"{n}: no <collision>")
    checks.append(c)


def check_zup(root, checks):
    """Feet must be below the trunk and spread out sideways, not stacked vertically.

    The old export was Y-up, so the robot loaded lying on its side with its feet
    19 mm apart instead of 216 mm. This catches that without needing a simulator.
    """
    c = Check(
        "zup",
        "The robot is Z-up",
        "The old export was Y-up: loaded raw, the robot lies on its side with its "
        "feet 19 mm apart instead of 216 mm.",
    )
    # Walk the joint tree accumulating full transforms - translation AND rotation.
    # Summing the translations alone gives the wrong answer, because each joint
    # origin is expressed in its parent's rotated frame.
    parent_of, xyz_of, rot_of = {}, {}, {}
    for j in root.findall("joint"):
        child = j.find("child").get("link")
        parent_of[child] = j.find("parent").get("link")
        o = j.find("origin")
        xyz_of[child] = _vec(o.get("xyz", "0 0 0")) if o is not None else (0.0, 0.0, 0.0)
        rpy = _vec(o.get("rpy", "0 0 0")) if o is not None else (0.0, 0.0, 0.0)
        rot_of[child] = _rpy_matrix(*rpy)

    def world_xyz(link):
        """Position of a link's own frame, at the zero configuration."""
        chain, seen = [], set()
        while link in parent_of and link not in seen:
            seen.add(link)
            chain.append(link)
            link = parent_of[link]
        pos = (0.0, 0.0, 0.0)
        rot = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        for node in reversed(chain):  # root outwards
            offset = _mat_vec(rot, xyz_of[node])
            pos = (pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])
            rot = _mat_mul(rot, rot_of[node])
        return pos

    feet = [world_xyz(f"{l}_bottom") for l in LEGS]
    if len(feet) < 4:
        c.detail = "could not locate the four foot links"
        checks.append(c)
        return
    spread_x = max(f[0] for f in feet) - min(f[0] for f in feet)
    spread_y = max(f[1] for f in feet) - min(f[1] for f in feet)
    spread_z = max(f[2] for f in feet) - min(f[2] for f in feet)
    footprint = max(spread_x, spread_y)
    c.passed = footprint > 0.10 and spread_z < 0.05 and min(f[2] for f in feet) < 0
    c.detail = (
        f"footprint {spread_x*1000:.0f} x {spread_y*1000:.0f} mm, "
        f"feet vary {spread_z*1000:.0f} mm in height"
    )
    if footprint <= 0.10:
        c.rows.append(
            f"the four feet span only {footprint*1000:.0f} mm - the model is almost "
            "certainly rotated onto its side"
        )
    if spread_z >= 0.05:
        c.rows.append(
            f"the feet differ by {spread_z*1000:.0f} mm in height - they should be "
            "level with each other"
        )
    if min(f[2] for f in feet) >= 0:
        c.rows.append("the feet are not below the trunk - Z is not up")
    checks.append(c)


def check_mujoco(urdf_path, checks):
    c = Check(
        "mujoco",
        "MuJoCo loads it and keeps the trunk free",
        "MuJoCo silently welds a root link to the world. Last time that quietly "
        "deleted 723 g and the model reported 901 g instead of 1625 g.",
    )
    try:
        import mujoco  # noqa: PLC0415
    except ImportError:
        c.detail = "MuJoCo is not installed - skipped"
        c.passed = True
        checks.append(c)
        return
    try:
        model = mujoco.MjModel.from_xml_path(str(urdf_path))
    except Exception as exc:  # noqa: BLE001
        c.detail = "MuJoCo refused to load it"
        c.rows.append(str(exc).strip().splitlines()[0])
        checks.append(c)
        return
    sim_mass = float(sum(model.body_mass))
    has_free = any(model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE for i in range(model.njnt))
    c.passed = has_free and sim_mass > MASS_MIN
    c.detail = f"simulator sees {sim_mass*1000:.0f} g, {model.njnt} joints, {model.nbody} bodies"
    if not has_free:
        c.rows.append(
            "no free joint - the trunk is welded to the world and its mass has been "
            "dropped. The MJCF needs a floating base."
        )
    checks.append(c)


# --------------------------------------------------------------------------


def run_checks(urdf_path: Path) -> list[Check]:
    checks: list[Check] = []
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:  # noqa: BLE001
        c = Check("parses", "Parses as a URDF", "If this fails nothing else can run.")
        c.rows.append(str(exc))
        return [c]

    check_parses(root, urdf_path, checks)
    check_structure(root, checks)
    check_zup(root, checks)
    check_axes(root, checks)
    check_limits(root, checks)
    check_mass(root, checks)
    check_inertia(root, checks)
    check_meshes(root, urdf_path, checks)
    check_collision(root, checks)
    check_mujoco(urdf_path, checks)
    return checks


def as_dict(urdf_path: Path, checks: list[Check]) -> dict:
    return {
        "urdf": str(urdf_path),
        "exists": urdf_path.exists(),
        "passed": sum(1 for c in checks if c.passed),
        "total": len(checks),
        "checks": [
            {
                "key": c.key,
                "title": c.title,
                "why": c.why,
                "passed": c.passed,
                "detail": c.detail,
                "rows": c.rows,
            }
            for c in checks
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("urdf", nargs="?", default="sim/models/gray.urdf")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    path = Path(args.urdf)
    if not path.exists():
        if args.json:
            print(json.dumps({"urdf": str(path), "exists": False, "passed": 0, "total": 10, "checks": []}))
        else:
            print(f"No URDF at {path}")
            print("Export one from SolidWorks first, then run this again.")
        return 1

    checks = run_checks(path)
    if args.json:
        print(json.dumps(as_dict(path, checks), indent=2))
        return 0 if all(c.passed for c in checks) else 1

    passed = sum(1 for c in checks if c.passed)
    print(f"\n{path}   {passed}/{len(checks)} checks pass\n")
    for c in checks:
        mark = "PASS" if c.passed else "FAIL"
        print(f"  [{mark}]  {c.title}")
        if c.detail:
            print(f"          {c.detail}")
        for r in c.rows:
            print(f"          - {r}")
        if not c.passed:
            print(f"          why it matters: {c.why}")
        print()
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
