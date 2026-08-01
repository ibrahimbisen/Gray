#!/usr/bin/env python3
"""Repair Gray's SolidWorks-exported URDF into something trainable.

The source URDF at robot/++Main Body URDF/gray.urdf is structurally correct - 13
links, 12 joints, correct tree, trustworthy inertia axes - but cannot be trained
against as it stands:

  1. ALL 12 JOINTS ARE `continuous`. No position limits, no effort ceiling, no
     velocity ceiling. The simulator believes every joint can spin through 360 deg
     at unbounded torque, so a policy trained on it learns gaits the DS3218MG
     physically cannot perform. This is the single biggest blocker.
  2. Joint axes are near-unit but sloppy, e.g. (0, -0.01495, -0.99989). Snapping to
     exact unit vectors keeps the leg IK closed-form and clean.
  3. Masses cover only the printed structure plus 8 of 12 servos - no battery, Pi,
     electronics or fasteners. Replaced from gray/config/robot.yaml.
  4. Collision geometry reuses the visual meshes, one of which is 6.4 MB. Contact
     resolution against high-poly meshes is slow and numerically noisy; replaced
     with primitives - a sphere at each foot, oriented boxes elsewhere.

Run:
    python tools/fix_urdf.py                # writes sim/models/gray.urdf
    python tools/fix_urdf.py --report       # print what changed, write nothing
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET  # construction only; parsing uses defusedxml
from dataclasses import dataclass

import numpy as np
import trimesh
import yaml
from defusedxml import minidom as defused_minidom
from defusedxml.ElementTree import parse as safe_parse

SRC_URDF = "robot/++Main Body URDF/gray.urdf"
MESH_DIR = "robot/++Main Body URDF/meshes"
ROBOT_YAML = "gray/config/robot.yaml"
OUT_URDF = "sim/models/gray.urdf"

# Shanks get a contact sphere at the foot tip; everything else gets an oriented box.
# Foot contact drives the entire gait, so it is the one surface worth being precise
# about. Other surfaces only matter for self-collision and crash detection.
SHANK_SUFFIX = "_bottom"
FOOT_RADIUS_M = 0.012  # refine against the real foot pad during Phase 4

# Visual meshes are decimated and re-exported next to the output URDF. Two reasons:
# the source meshes total ~21 MB (base_link alone is 6.4 MB) and live behind Git LFS
# under a path with '++' and spaces that does not resolve from sim/models/; and the
# Windows training box clones with LFS skipped, so it must not depend on them.
VISUAL_MESH_DIR = "meshes"
# Per-component, not per-link - see export_visual_mesh. 4000 for a whole link was far
# too aggressive: base_link is ~130k faces of thin resin frame plus four chunky baked
# servos, and a 97% cut deleted the frame outright.
VISUAL_FACE_BUDGET = 9000
MIN_COMPONENT_FACES = 400   # keep small parts (bearings, servo horns) recognisable


@dataclass
class JointFix:
    name: str
    axis_before: np.ndarray
    axis_after: np.ndarray

    @property
    def snap_deg(self) -> float:
        a = self.axis_before / np.linalg.norm(self.axis_before)
        return float(np.degrees(np.arccos(np.clip(a @ self.axis_after, -1.0, 1.0))))


def snap_axis(axis: np.ndarray) -> np.ndarray:
    """Snap a near-axis-aligned vector to the exact nearest signed unit axis."""
    i = int(np.argmax(np.abs(axis)))
    out = np.zeros(3)
    out[i] = float(np.sign(axis[i]))
    return out


def foot_tip(mesh: trimesh.Trimesh) -> np.ndarray:
    """Distal-most point of a shank - where it touches the ground.

    A link's own origin is where its parent joint attaches, so for a shank that is
    the knee. The foot is therefore simply the vertex cluster furthest from the
    origin. Measuring radially rather than along a bounding-box axis matters here
    because these shanks sit diagonally in their link frames - the bbox is
    163 x 124 mm for a part that is really a straight ~160 mm leg.
    """
    v = np.asarray(mesh.vertices)
    r = np.linalg.norm(v, axis=1)
    far = v[r > r.max() - 0.004]
    return far.mean(axis=0)


def export_visual_mesh(mesh: trimesh.Trimesh, name: str, out_dir: str) -> tuple[str, int]:
    """Decimate a link mesh and write it beside the output URDF.

    Decimates each connected component SEPARATELY. These are multi-body STLs - the
    CAD exporter merged every link's sub-assembly (resin frame plus the servos baked
    into it) into a single file - and running quadric decimation across the union
    damages them in two distinct ways:

      * The budget is spent where the faces are. base_link's thin 57 cm3 frame was
        collapsed away entirely while its four chunky baked-in servos survived, so
        the trunk rendered as four floating blocks with nothing joining them.
      * Edges get collapsed BETWEEN disconnected components, producing triangles that
        bridge the gap between them. The thighs came out with 130 mm edges on a 170 mm
        part against a 0.96 mm median - the spikes visible in any viewer.

    Splitting first fixes both: no edge can span two components, and each component
    gets a share of the budget in proportion to its own complexity, with a floor so
    that small parts do not disappear.

    This is cosmetic. Collision geometry is primitive and mass properties come from
    the undecimated meshes via robot.yaml, so nothing here touches the physics.
    """
    os.makedirs(out_dir, exist_ok=True)

    parts = mesh.split(only_watertight=False)
    if len(parts) == 0:
        parts = [mesh]
    total = sum(len(p.faces) for p in parts)

    kept = []
    for part in parts:
        budget = max(MIN_COMPONENT_FACES,
                     int(VISUAL_FACE_BUDGET * len(part.faces) / max(total, 1)))
        if len(part.faces) > budget:
            try:
                part = part.simplify_quadric_decimation(face_count=budget)
            except Exception:
                pass  # decimation backend unavailable; ship this component as-is
        kept.append(part)

    m = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]
    path = os.path.join(out_dir, f"{name}.STL")
    m.export(path)
    return path, len(m.faces)


def add_collision(link: ET.Element, name: str, mesh: trimesh.Trimesh) -> str:
    """Attach primitive collision geometry to a link. Returns a description."""
    col = ET.SubElement(link, "collision")
    col.set("name", f"{name}_collision")
    origin = ET.SubElement(col, "origin")
    geom = ET.SubElement(col, "geometry")

    if name.endswith(SHANK_SUFFIX):
        tip = foot_tip(mesh)
        origin.set("xyz", " ".join(f"{x:.6f}" for x in tip))
        origin.set("rpy", "0 0 0")
        ET.SubElement(geom, "sphere").set("radius", f"{FOOT_RADIUS_M}")
        return f"sphere r={FOOT_RADIUS_M*1000:.0f}mm at ({tip[0]*1000:.0f}, " \
               f"{tip[1]*1000:.0f}, {tip[2]*1000:.0f})mm"

    obb = mesh.bounding_box_oriented
    ext = np.asarray(obb.primitive.extents, dtype=float)
    T = np.asarray(obb.primitive.transform, dtype=float)
    rpy = trimesh.transformations.euler_from_matrix(T, "sxyz")
    origin.set("xyz", " ".join(f"{x:.6f}" for x in T[:3, 3]))
    origin.set("rpy", " ".join(f"{x:.6f}" for x in rpy))
    ET.SubElement(geom, "box").set("size", " ".join(f"{x:.6f}" for x in ext))
    return f"box {ext[0]*1000:.0f} x {ext[1]*1000:.0f} x {ext[2]*1000:.0f} mm"


# SolidWorks exported this assembly Y-up: base_link's thin axis (25 mm of a
# 197 x 25 x 158 mm plate) lies along Y, and the two front hips differ along Z.
# URDF and MuJoCo are Z-up, so loaded as-is the robot lies on its side - two legs
# stick vertically up, two down, and the feet span only 19 mm laterally. Rotating
# the base frame by +90 deg about X maps +Y to +Z: all four feet then sit below the
# trunk with a 216 mm stance. Verified empirically against both rotation signs.
R_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]])


def _rot_origin(el: ET.Element, R: np.ndarray) -> None:
    """Re-express one <origin> in a frame rotated by R."""
    if el is None:
        return
    xyz = np.array([float(v) for v in (el.get("xyz") or "0 0 0").split()])
    rpy = np.array([float(v) for v in (el.get("rpy") or "0 0 0").split()])
    el.set("xyz", " ".join(f"{v:.9f}" for v in R @ xyz))
    M = trimesh.transformations.euler_matrix(*rpy, "sxyz")
    M[:3, :3] = R @ M[:3, :3]
    el.set("rpy", " ".join(f"{v:.9f}" for v in
                           trimesh.transformations.euler_from_matrix(M, "sxyz")))


def rotate_base_frame(root: ET.Element, R: np.ndarray) -> None:
    """Rotate base_link's frame so the robot is Z-up.

    Only base_link's own frame moves, so exactly two things must be re-expressed:
    everything defined in that frame (its inertial, visual and collision origins),
    and the origins of the joints that attach its children. Child link frames are
    untouched - they are defined relative to their parent joint, which we have
    already corrected. Mesh files are never rewritten; the compensating rotation
    rides on each origin's rpy.
    """
    base = next(l for l in root.findall("link") if l.get("name") == "base_link")
    for tag in ("inertial", "visual", "collision"):
        for el in base.findall(tag):
            _rot_origin(el.find("origin"), R)

    inertia = base.find("inertial/inertia")
    if inertia is not None:
        I = np.array([
            [float(inertia.get("ixx")), float(inertia.get("ixy")), float(inertia.get("ixz"))],
            [float(inertia.get("ixy")), float(inertia.get("iyy")), float(inertia.get("iyz"))],
            [float(inertia.get("ixz")), float(inertia.get("iyz")), float(inertia.get("izz"))],
        ])
        I = R @ I @ R.T
        for key, (a, b) in {"ixx": (0, 0), "iyy": (1, 1), "izz": (2, 2),
                            "ixy": (0, 1), "ixz": (0, 2), "iyz": (1, 2)}.items():
            inertia.set(key, f"{I[a, b]:.9e}")

    for j in root.findall("joint"):
        if j.find("parent").get("link") == "base_link":
            _rot_origin(j.find("origin"), R)
            axis = j.find("axis")
            a = np.array([float(v) for v in axis.get("xyz").split()])
            axis.set("xyz", " ".join(f"{v:.0f}" for v in R @ a))


def prettify(root: ET.Element) -> str:
    raw = ET.tostring(root, encoding="unicode")
    doc = defused_minidom.parseString(raw)
    lines = doc.toprettyxml(indent="  ").splitlines()
    return "\n".join(ln for ln in lines if ln.strip()) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair Gray's URDF for training.")
    ap.add_argument("--report", action="store_true", help="print changes, write nothing")
    ap.add_argument("--out", default=OUT_URDF)
    args = ap.parse_args()

    with open(ROBOT_YAML) as fh:
        cfg = yaml.safe_load(fh)
    servo = cfg["servo"]
    limit, effort, vel = servo["limit_rad"], servo["effort_nm"], servo["velocity_rad_s"]

    root = safe_parse(SRC_URDF).getroot()
    root.set("name", "gray")

    # MuJoCo reads this block when importing the URDF. Without discardvisual=false it
    # throws the visual meshes away and renders only collision primitives, which makes
    # every training video a pile of grey boxes.
    mj = ET.Element("mujoco")
    ET.SubElement(mj, "compiler", {
        "meshdir": ".", "discardvisual": "false", "balanceinertia": "true",
        "strippath": "false",
    })
    root.insert(0, mj)

    # --- joints: continuous -> revolute, real limits, clean axes -------------------
    fixes: list[JointFix] = []
    for j in root.findall("joint"):
        axis_el = j.find("axis")
        before = np.array([float(x) for x in axis_el.get("xyz").split()])
        after = snap_axis(before)
        fixes.append(JointFix(j.get("name"), before, after))

        j.set("type", "revolute")
        axis_el.set("xyz", " ".join(f"{v:.0f}" for v in after))

        lim = j.find("limit")
        if lim is None:
            lim = ET.SubElement(j, "limit")
        lim.set("lower", f"{-limit:.6f}")
        lim.set("upper", f"{limit:.6f}")
        lim.set("effort", f"{effort}")
        lim.set("velocity", f"{vel}")

        # Gearbox friction and backlash: without damping, position-controlled hobby
        # servos simulate as ideal and the policy learns to exploit that.
        dyn = j.find("dynamics")
        if dyn is None:
            dyn = ET.SubElement(j, "dynamics")
        dyn.set("damping", "0.02")
        dyn.set("friction", "0.01")

    # --- links: masses, inertia, primitive collision -------------------------------
    mass_changes, collisions, visuals = [], [], []
    for link in root.findall("link"):
        name = link.get("name")
        spec = cfg["links"][name]

        inertial = link.find("inertial")
        old_mass = float(inertial.find("mass").get("value"))
        inertial.find("mass").set("value", f"{spec['mass_kg']:.6f}")
        mass_changes.append((name, old_mass, spec["mass_kg"]))

        ie = inertial.find("inertia")
        for key, val in spec["inertia"].items():
            ie.set(key, f"{val:.9e}")

        for old in link.findall("collision"):
            link.remove(old)
        mesh = trimesh.load(f"{MESH_DIR}/{name}.STL", force="mesh")
        collisions.append((name, add_collision(link, name, mesh)))

        # Re-export the visual mesh next to the output URDF, decimated. The source
        # path ('robot/++Main Body URDF/meshes/') does not resolve from sim/models/.
        out_dir = os.path.join(os.path.dirname(args.out) or ".", VISUAL_MESH_DIR)
        path, faces = export_visual_mesh(mesh, name, out_dir)
        visuals.append((name, len(mesh.faces), faces, os.path.getsize(path)))
        for vis in link.findall("visual"):
            mesh_el = vis.find("geometry/mesh")
            if mesh_el is not None:
                mesh_el.set("filename", f"{VISUAL_MESH_DIR}/{name}.STL")

    rotate_base_frame(root, R_YUP_TO_ZUP)

    # --- report --------------------------------------------------------------------
    print("FRAME    base_link rotated +90 deg about X  (Y-up export -> Z-up)\n")
    print("JOINTS   continuous -> revolute on all 12")
    print(f"  limits  +/-{limit:.4f} rad  (+/-{np.degrees(limit):.0f} deg, "
          f"{servo['range_deg']} deg servo)")
    print(f"  effort  {effort} N.m      velocity {vel} rad/s")
    print(f"  damping 0.02             friction 0.01\n")
    print(f"  {'joint':11s} {'axis before':>30s}      {'after':>12s} {'snap':>7s}")
    for f in sorted(fixes, key=lambda f: -f.snap_deg):
        b = f.axis_before
        a = f.axis_after
        print(f"  {f.name:11s} ({b[0]:7.4f}, {b[1]:8.5f}, {b[2]:8.5f})  ->  "
              f"({a[0]:2.0f},{a[1]:2.0f},{a[2]:2.0f}) {f.snap_deg:6.2f} deg")

    print("\nMASSES")
    for n, b, a in mass_changes:
        print(f"  {n:11s} {b*1000:7.1f} g  ->  {a*1000:7.1f} g   ({a/b:4.2f}x)")
    tb = sum(m for _, m, _ in mass_changes)
    ta = sum(m for _, _, m in mass_changes)
    print(f"  {'TOTAL':11s} {tb*1000:7.1f} g  ->  {ta*1000:7.1f} g   ({ta/tb:4.2f}x)")

    print("\nCOLLISION  (was: high-poly visual meshes)")
    for n, desc in collisions:
        print(f"  {n:11s} {desc}")

    print("\nVISUAL MESHES  decimated and re-exported to sim/models/meshes/")
    tf_in = sum(v[1] for v in visuals)
    tf_out = sum(v[2] for v in visuals)
    tb = sum(v[3] for v in visuals)
    for n, fin, fout, nbytes in visuals:
        print(f"  {n:11s} {fin:7d} -> {fout:6d} faces   {nbytes/1024:7.1f} KB")
    print(f"  {'TOTAL':11s} {tf_in:7d} -> {tf_out:6d} faces   {tb/1024:7.1f} KB"
          f"   (source was ~21 MB)")

    if args.report:
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(prettify(root))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
