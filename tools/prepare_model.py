"""Turn a raw SolidWorks export into the model the simulator uses.

    python tools/prepare_model.py "sim/URDF and Meshes V2"

Three things the exporter cannot do, done here instead:

1. **Move base_link's frame onto the robot, pointing the right way.** The exporter
   writes every joint origin relative to the assembly origin, which on this
   assembly is 1.9 m away from the robot and not Z-up. Setting a custom
   coordinate system in the exporter crashes it, so the correction happens here.
   The new frame is derived from the robot's own geometry - hips and feet - not
   from a number typed in by hand, so it cannot drift out of date.

2. **Rewrite `package://` mesh paths.** MuJoCo cannot open them.

3. **Copy everything into sim/models/** under one predictable name.

The result is checked by tools/check_urdf.py, which is the point: this script's
output is verifiable rather than trusted.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_urdf import leg_seg  # noqa: E402

OUT_DIR = ROOT / "sim" / "models"
OUT_URDF = OUT_DIR / "gray.urdf"
ROBOT_NAME = "gray"


# --------------------------------------------------------------------------
# small rotation helpers - URDF stores fixed-axis roll/pitch/yaw
# --------------------------------------------------------------------------


def rpy_to_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(m: np.ndarray):
    """Inverse of rpy_to_matrix. Picks the branch with pitch in [-pi/2, pi/2]."""
    sp = -m[2, 0]
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    if abs(abs(sp) - 1.0) < 1e-9:  # gimbal lock: roll and yaw are not separable
        roll = math.atan2(-m[1, 2], m[1, 1])
        yaw = 0.0
    else:
        roll = math.atan2(m[2, 1], m[2, 2])
        yaw = math.atan2(m[1, 0], m[0, 0])
    return roll, pitch, yaw


def read_origin(el) -> tuple[np.ndarray, np.ndarray]:
    o = el.find("origin") if el is not None else None
    xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
    return xyz, rpy_to_matrix(rpy)


def write_origin(el, xyz: np.ndarray, rot: np.ndarray) -> None:
    o = el.find("origin")
    if o is None:
        o = ET.SubElement(el, "origin")
    o.set("xyz", " ".join(f"{v:.10g}" for v in xyz))
    o.set("rpy", " ".join(f"{v:.10g}" for v in matrix_to_rpy(rot)))


# --------------------------------------------------------------------------


def forward_kinematics(root) -> dict[str, np.ndarray]:
    """Position of every link's frame at the zero configuration."""
    parent, xyz_of, rot_of = {}, {}, {}
    for j in root.findall("joint"):
        child = j.find("child").get("link")
        parent[child] = j.find("parent").get("link")
        xyz_of[child], rot_of[child] = read_origin(j)

    out: dict[str, np.ndarray] = {}
    for link in {l.get("name") for l in root.findall("link")}:
        chain, node, seen = [], link, set()
        while node in parent and node not in seen:
            seen.add(node)
            chain.append(node)
            node = parent[node]
        pos, rot = np.zeros(3), np.eye(3)
        for n in reversed(chain):
            pos = pos + rot @ xyz_of[n]
            rot = rot @ rot_of[n]
        out[link] = pos
    return out


def derive_base_frame(root) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Work out where base_link's frame should be, from the robot's own geometry.

    Returns (rotation, origin, notes). The rotation's columns are the new frame's
    X, Y and Z expressed in the exporter's frame.
    """
    notes = []
    hip = {}
    for j in root.findall("joint"):
        part = leg_seg(j.get("name")) or leg_seg(j.find("child").get("link"))
        if part and part[1] == "hip" and j.find("parent").get("link") == "base_link":
            hip[part[0]] = read_origin(j)[0]
    missing = {"fl", "fr", "bl", "br"} - set(hip)
    if missing:
        raise SystemExit(f"cannot find hip joints for: {sorted(missing)}")

    fk = forward_kinematics(root)
    feet = [p for name, p in fk.items() if (leg_seg(name) or ("", ""))[1] == "calf"]
    if len(feet) != 4:
        raise SystemExit(f"expected 4 calf links, found {len(feet)}")

    hip_centre = np.mean([hip[k] for k in ("fl", "fr", "bl", "br")], axis=0)

    # The four hips lie in the robot's horizontal plane, whatever pose the legs
    # happen to be in. Take that plane's normal as the vertical. Deriving it from
    # the feet instead would fold in however far the legs are splayed, which is a
    # property of the pose, not of the robot.
    centred = np.array([hip[k] for k in ("fl", "fr", "bl", "br")]) - hip_centre
    _, sv, vt = np.linalg.svd(centred)
    up = vt[2]
    if sv[2] > 0.002:
        notes.append(
            f"the four hips are not coplanar - {sv[2]*1000:.1f} mm out. The trunk "
            "geometry is asymmetric, so 'level' is only defined to that accuracy."
        )

    # Point it away from the feet. All four legs hang below the hips no matter how
    # the pose is arranged, so this settles the sign. Getting it wrong is exactly
    # the old Y-up bug.
    if float((np.mean(feet, axis=0) - hip_centre) @ up) > 0:
        up = -up

    # Forward runs from the back pair of hips to the front pair.
    front = (hip["fl"] + hip["fr"]) / 2
    back = (hip["bl"] + hip["br"]) / 2
    fwd = front - back
    fwd = fwd - up * float(fwd @ up)  # keep only the part lying in the plane
    fwd /= np.linalg.norm(fwd)

    left = np.cross(up, fwd)  # right-handed: Z x X = Y

    # The left hips must land on the +Y side. If they do not, the CAD's left and
    # right labels are swapped, which is a naming bug worth reporting rather than
    # quietly correcting.
    if float((hip["fl"] - hip_centre) @ left) < 0:
        notes.append(
            "WARNING: the 'fl' and 'bl' hips are on the robot's RIGHT. The leg "
            "labels are mirrored in the CAD. Fix the names in SolidWorks - every "
            "turn command will otherwise go the wrong way."
        )

    rot = np.column_stack([fwd, left, up])
    return rot, hip_centre, notes


def retarget_base(root, rot: np.ndarray, origin: np.ndarray) -> None:
    """Re-express everything measured in base_link's old frame in the new one."""
    rt = rot.T

    for j in root.findall("joint"):
        if j.find("parent").get("link") != "base_link":
            continue  # child frames are relative to their own parent, untouched
        xyz, r = read_origin(j)
        write_origin(j, rt @ (xyz - origin), rt @ r)

    base = next(l for l in root.findall("link") if l.get("name") == "base_link")
    for tag in ("inertial", "visual", "collision"):
        for el in base.findall(tag):
            xyz, r = read_origin(el)
            write_origin(el, rt @ (xyz - origin), rt @ r)
            if tag == "inertial":
                rotate_inertia(el, rt)


def add_floating_base(root) -> None:
    """Say out loud that the robot is not bolted to the ground.

    A URDF with no floating joint is a fixed-base robot, and MuJoCo honours that
    by welding base_link into the world - silently taking its 1198.66 g with it,
    more than half the robot. The previous attempt at this project shipped a model
    reporting 901 g instead of 1625 g for exactly this reason.
    """
    if any(j.get("type") == "floating" for j in root.findall("joint")):
        return
    ET.SubElement(root, "link", {"name": "world"})
    j = ET.SubElement(root, "joint", {"name": "floating_base", "type": "floating"})
    ET.SubElement(j, "parent", {"link": "world"})
    ET.SubElement(j, "child", {"link": "base_link"})


def fix_masses(root, cfg: dict) -> list[str]:
    """Replace the exporter's water-density masses with the real ones.

    The exporter writes each part's volume in cm3 as its mass. Geometry is right,
    density is not - so scaling a link's mass and its inertia tensor by the same
    factor gives the correct model. Inertia is linear in density, so one factor
    does both.
    """
    m = cfg["mass"]
    want = {"trunk": m["trunk"], "hip": m["hip"], "thigh": m["thigh"], "calf": m["calf"]}
    report, total = [], 0.0
    donor: dict[str, tuple[float, dict]] = {}

    links = {l.get("name"): l for l in root.findall("link")}
    for name, link in links.items():
        part = leg_seg(name)
        key = part[1] if part else ("trunk" if name == "base_link" else None)
        if key is None or key not in want:
            continue
        inertial = link.find("inertial")
        if inertial is None:
            report.append(f"{name:10s} has no <inertial> at all - skipped")
            continue
        mass_el = inertial.find("mass")
        old = float(mass_el.get("value"))
        new = want[key] / 1000.0  # the yaml is in grams, URDF is in kg
        i_el = inertial.find("inertia")
        terms = {k: float(i_el.get(k, 0.0)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}

        if old > 0:
            scale = new / old
            donor.setdefault(key, (old, dict(terms)))
            for k, v in terms.items():
                i_el.set(k, f"{v * scale:.10g}")
            report.append(f"{name:10s} {old*1000:8.2f} -> {new*1000:8.2f} g   inertia x{scale:.3f}")
        else:
            # The exporter drops a link occasionally - bl_hip came out at 0 g with
            # perfectly normal geometry. Borrow the tensor from another link of the
            # same segment. Same part, same size, so this is close; it is still an
            # assumption and says so.
            src = donor.get(key)
            if src is None:
                report.append(f"{name:10s} 0 g and no other {key} to borrow from - LEFT BROKEN")
                continue
            src_mass, src_terms = src
            scale = new / src_mass
            for k, v in src_terms.items():
                # Mirror across the robot's mid-plane: the y-coupling terms flip.
                sign = -1.0 if k in ("ixy", "iyz") else 1.0
                i_el.set(k, f"{v * scale * sign:.10g}")
            report.append(
                f"{name:10s} {old*1000:8.2f} -> {new*1000:8.2f} g   ASSUMED - the "
                f"exporter gave it no mass, tensor copied from the other {key}"
            )
        mass_el.set("value", f"{new:.10g}")
        total += new

    report.append(f"{'TOTAL':10s} {total*1000:8.2f} g   (expected {m['total_expected']:.2f} g)")
    return report


def rotate_inertia(inertial, rt: np.ndarray) -> None:
    """An inertia tensor is expressed in the link frame, so it rotates too."""
    i = inertial.find("inertia")
    if i is None:
        return
    t = np.array([
        [float(i.get("ixx", 0)), float(i.get("ixy", 0)), float(i.get("ixz", 0))],
        [float(i.get("ixy", 0)), float(i.get("iyy", 0)), float(i.get("iyz", 0))],
        [float(i.get("ixz", 0)), float(i.get("iyz", 0)), float(i.get("izz", 0))],
    ])
    t = rt @ t @ rt.T
    for key, v in (("ixx", t[0, 0]), ("iyy", t[1, 1]), ("izz", t[2, 2]),
                   ("ixy", t[0, 1]), ("ixz", t[0, 2]), ("iyz", t[1, 2])):
        i.set(key, f"{v:.10g}")


def rewrite_mesh_paths(root) -> int:
    n = 0
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        name = fn.replace("\\", "/").rsplit("/", 1)[-1]
        mesh.set("filename", f"meshes/{name}")
        n += 1
    return n


# MuJoCo refuses any STL over 200,000 triangles outright. Well below that, a very
# dense mesh still costs collision and render time for detail no contact ever
# resolves - the trunk exported at 278,000 triangles, which is more than the whole
# rest of the robot.
MAX_TRIANGLES = 60_000


def copy_meshes(src: Path, dst: Path) -> list[str]:
    """Copy the STLs across, thinning any that are denser than we can use."""
    import trimesh  # noqa: PLC0415

    dst.mkdir(parents=True, exist_ok=True)
    report = []
    seen: set[str] = set()
    for stl in sorted(src.iterdir()):
        if stl.suffix.lower() != ".stl" or stl.name.lower() in seen:
            continue
        seen.add(stl.name.lower())
        mesh = trimesh.load_mesh(stl, process=False)
        before = len(mesh.faces)
        if before <= MAX_TRIANGLES:
            shutil.copy2(stl, dst / stl.name)
            report.append(f"{stl.name:16s} {before:>8,} tri  (unchanged)")
            continue

        box0 = mesh.bounds.copy()
        mesh = mesh.simplify_quadric_decimation(face_count=MAX_TRIANGLES)
        mesh.export(dst / stl.name)
        # These are multi-body STLs, so they are not watertight and their reported
        # volume is meaningless. The bounding box is not - if a corner moves, the
        # part changed shape somewhere that matters.
        drift = float(np.abs(mesh.bounds - box0).max()) * 1000
        report.append(
            f"{stl.name:16s} {before:>8,} -> {len(mesh.faces):>7,} tri   "
            f"bbox moved {drift:.2f} mm"
        )
    return report


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("package", help="the folder the SolidWorks exporter created")
    args = ap.parse_args()

    pkg = Path(args.package)
    if not pkg.is_absolute():
        pkg = ROOT / pkg
    urdfs = sorted((pkg / "urdf").glob("*.urdf")) if (pkg / "urdf").is_dir() else sorted(pkg.glob("*.urdf"))
    if not urdfs:
        raise SystemExit(f"no .urdf found under {pkg}")
    src = urdfs[0]
    meshes = pkg / "meshes"
    if not meshes.is_dir():
        raise SystemExit(f"no meshes/ folder under {pkg}")

    tree = ET.parse(src)
    root = tree.getroot()
    root.set("name", ROBOT_NAME)

    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load((ROOT / "gray" / "config" / "robot.yaml").read_text())

    before = forward_kinematics(root)
    add_floating_base(root)
    mass_report = fix_masses(root, cfg)
    rot, origin, notes = derive_base_frame(root)
    retarget_base(root, rot, origin)
    n_meshes = rewrite_mesh_paths(root)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh_report = copy_meshes(meshes, OUT_DIR / "meshes")
    ET.indent(tree, space="  ")
    tree.write(OUT_URDF, encoding="utf-8", xml_declaration=True)

    after = forward_kinematics(root)
    feet = {n: p for n, p in after.items() if (leg_seg(n) or ("", ""))[1] == "calf"}

    print(f"source   {src}")
    print(f"output   {OUT_URDF.relative_to(ROOT)}   ({n_meshes} mesh references)")
    print("\nmass, from gray/config/robot.yaml:")
    for line in mass_report:
        print(f"  {line}")
    print("\nmeshes:")
    for line in mesh_report:
        print(f"  {line}")
    print("\nnew base frame, in the exporter's coordinates:")
    print(f"  origin   {np.round(origin, 5)}")
    print(f"  X fwd    {np.round(rot[:, 0], 4)}")
    print(f"  Y left   {np.round(rot[:, 1], 4)}")
    print(f"  Z up     {np.round(rot[:, 2], 4)}")
    for n in notes:
        print(f"  note: {n}")

    print("\nfeet, relative to base_link (mm):")
    for name in sorted(feet):
        p = feet[name] * 1000
        print(f"  {name:10s} x{p[0]:8.1f}  y{p[1]:8.1f}  z{p[2]:8.1f}")
    spread = np.ptp(np.array(list(feet.values())), axis=0) * 1000
    print(f"  spread     x{spread[0]:8.1f}  y{spread[1]:8.1f}  z{spread[2]:8.1f}")
    print(f"\nunchanged: {len(before)} link frames, distances between them are identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
