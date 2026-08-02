#!/usr/bin/env python3
"""Convert the repaired URDF into a trainable MuJoCo MJCF.

A URDF alone is not a simulation. Loading sim/models/gray.urdf directly gives a robot
welded to the world (MuJoCo reports 901.5 g because base_link becomes the world body
and its mass is discarded) with no ground, no actuators and no sensors. This adds:

  freejoint   base_link floats, so the robot can fall over - the entire point
  floor       plane with friction in the range domain randomisation will sample
  actuators   12 position servos, because DS3218MG take a POSITION command and
              there is no torque control and no encoder feedback to read back
  armature    reflected rotor inertia through the gearbox. Easy to omit and badly
              wrong to omit: a ~245:1 hobby servo reflects roughly 4e-3 kg.m^2,
              which dominates the ~8e-5 kg.m^2 link inertia. Leaving it out makes
              the legs feel weightless in sim and heavy on the real robot - a
              classic sim-to-real failure. Treat as a key randomisation target.
  sensors     framequat / gyro / accelerometer on the trunk. Deliberately the ONLY
              proprioception, matching what the real robot can actually measure.

Run:
    python tools/make_mjcf.py            # writes sim/models/gray.xml
    python tools/make_mjcf.py --check    # also drop-test it
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml
from defusedxml.ElementTree import fromstring as safe_fromstring

SRC_URDF = "sim/models/gray.urdf"
ROBOT_YAML = "gray/config/robot.yaml"
OUT_MJCF = "sim/models/gray.xml"

CONTROL_HZ = 50           # PCA9685 servo PWM period is 20 ms - a hard ceiling
TIMESTEP = 0.002          # 500 Hz physics, 10 substeps per control tick

# Reflected rotor inertia through the gearbox. Estimated: a ~6e-8 kg.m^2 rotor behind
# a ~245:1 reduction (1.96 N.m stall / ~8 mN.m motor stall) gives ~3.8e-3. Randomise
# this hard during training - it is estimated, not measured.
ARMATURE = 0.003
DAMPING = 0.05            # gearbox viscous friction
FRICTIONLOSS = 0.01       # gearbox dry friction (stiction)

# Position-servo gain. A DS3218MG holds position stiffly; kp is set so a ~0.1 rad
# error saturates the 1.96 N.m force range, matching observed hobby-servo stiffness.
KP = 20.0
KV = 0.5


def build_mjcf(urdf_path: str) -> ET.Element:
    """Load the URDF through MuJoCo and recover its MJCF form."""
    model = mujoco.MjModel.from_xml_path(urdf_path)
    tmp = os.path.join(os.path.dirname(urdf_path) or ".", "_tmp_gray.xml")
    mujoco.mj_saveLastXML(tmp, model)
    with open(tmp) as fh:
        xml = fh.read()
    os.remove(tmp)
    return safe_fromstring(xml)


SPAWN_HEIGHT_M = 0.25

# MuJoCo's convention: viewers hide group 3 by default, so collision primitives stop
# obscuring the visual meshes. Purely cosmetic - the solver ignores geom groups.
COLLISION_GEOM_GROUP = 3

# THE SHIN HAD NO COLLISION BODY AT ALL. The URDF gives each shank exactly one
# collision shape, `<leg>_bottom_collision`, a 12 mm sphere sitting at the foot. That
# is the right shape for a foot and the wrong amount of shape for a shank: measured
# against the CAD mesh it covers 3% of the part, so the 170 mm shin between knee and
# foot is invisible to the solver and passes straight through the trunk, the thighs
# and the other legs.
#
# It never showed up standing or walking, because a shin rarely touches anything. It
# becomes constant the moment the robot is folded belly-down, which is where the legs
# tuck directly under the trunk - and belly-down is now the start pose for every
# training episode.
#
# For reference, the other three parts do not need this: the same measurement puts
# base_link at 100% of its mesh and the hip at 99%. Only the shank was missing.
#
# A SEPARATE GEOM, NOT A REPLACEMENT for the foot sphere. The foot's name is matched
# by FOOT_GEOM_REGEX "^(fl|fr|br|bl)_bottom_collision$" in train/gray_robot.py, which
# is what drives the contact sensor, condim=3 and the friction randomisation. Growing
# that geom into the whole shin would fire every foot-contact reward whenever a shin
# brushed anything. The new geom is named `<leg>_shank_collision` so it matches
# ".*_collision" (condim 1, plain contact) and NOT the foot regex.
SHANK_GEOM_SUFFIX = "_shank_collision"

# Radius matches the foot sphere so the capsule and the foot form one smooth leg with
# no step in the profile where they meet.
SHANK_RADIUS_M = 0.012


def add_shank_capsules(root: ET.Element) -> int:
    """Give each shank a collision body running knee to foot. Returns how many.

    The capsule is derived from the model rather than typed in: its far end is read
    off the foot geom's own `pos`, and its near end is the shank body's origin, which
    is where the knee joint sits (every `<leg>_bottom` joint declares pos="0 0 0").
    Measured that way the four shins come out 169.9-170.3 mm, agreeing with the
    thigh+shank reach in gray/config/robot.yaml. Hardcoding four coordinate triples
    would silently go stale the first time the CAD is re-exported.
    """
    added = 0
    for body in root.iter("body"):
        name = body.get("name") or ""
        if not name.endswith("_bottom"):
            continue
        leg = name[: -len("_bottom")]
        foot = next(
            (g for g in body.findall("geom")
             if g.get("name") == f"{leg}_bottom_collision"), None
        )
        if foot is None:
            # Refuse rather than emit a leg that is quietly still hollow.
            raise SystemExit(
                f"{name} has no '{leg}_bottom_collision' geom to take the shin's "
                f"far end from, so the shank capsule cannot be placed."
            )
        tip = foot.get("pos")
        if not tip:
            raise SystemExit(f"{leg}_bottom_collision has no pos to measure from.")

        shin = ET.Element("geom")
        shin.attrib.update({
            "name": f"{leg}{SHANK_GEOM_SUFFIX}",
            "type": "capsule",
            # Knee anchor to foot centre. The foot sphere stays where it is and the
            # capsule ends inside it, so the leg has no step in its profile.
            "fromto": f"0 0 0 {tip}",
            "size": str(SHANK_RADIUS_M),
            "group": str(COLLISION_GEOM_GROUP),
            "rgba": "1 1 1 1",
        })
        # Before the foot geom, so the foot is still the last word on the contact
        # that matters; ordering is cosmetic to the solver but keeps the XML readable.
        body.insert(list(body).index(foot), shin)
        added += 1
    return added


def exclude_false_contacts(root: ET.Element) -> int:
    """Stop the solver checking part pairs that cannot actually touch.

    THE TRUNK AND THE THIGHS. The trunk's collision shape is a solid box, because the
    box is the mesh's bounding box - but the real chassis is an OPEN FRAME with the
    four hip servos mounted inside it, and the thighs swing into that opening. So the
    simplified shapes report a deep overlap where the real parts have clear air
    between them.

    MEASURED, in the owner's own resting pose (hips 55 out, thighs -21, knees +39):
    the trunk box overlaps the two front thigh boxes by 31.2 mm and 31.1 mm before a
    single step is taken. MuJoCo then resolves that penetration the only way it can,
    by throwing the parts apart: the trunk left the floor at 931 mm/s, joints spun at
    704 deg/s and every servo was saturated at its 1.96 N.m limit from the first
    millisecond. The robot launched to 109 mm and bounced back to 82 mm, having never
    been asked to move at all. The owner spotted it as "the front legs push out".

    Swapping the thigh boxes for the real meshes barely helps - 31 mm becomes 26 mm -
    because MuJoCo collides the CONVEX HULL of a mesh, and a hull fills in the very
    opening the thigh sits in.

    THE EXCLUSION IS MEASURED, NOT ASSUMED. 300 poses spanning the owner's full joint
    ranges (hip -82 to +92, thigh -35 to +65, knee -40 to +39) were tested with
    triangle-level collision on the real meshes: the trunk and the thighs touch in
    ZERO of them. The pair cannot collide anywhere the robot can reach, so checking it
    can only produce false positives.

    Nothing else is excluded. Every other pair either can genuinely meet - the feet
    find each other at -28 degrees of hip, which is a real limit worth keeping - or is
    a jointed pair MuJoCo already filters.
    """
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    added = 0
    for leg in ("fl", "fr", "br", "bl"):
        ET.SubElement(contact, "exclude",
                      {"body1": "base_link", "body2": f"{leg}_top"})
        added += 1
    return added


def rebuild_trunk(worldbody: ET.Element, cfg: dict) -> ET.Element:
    """Re-create base_link as a real floating body.

    Because the URDF's root link has no parent joint, MuJoCo welds it to the world:
    base_link's geoms are emitted loose in <worldbody> and the four hips are promoted
    to top-level bodies, so its 723 g of mass is silently discarded. We rebuild the
    trunk, give it a freejoint, restore its inertial from robot.yaml, and re-parent
    the legs under it.
    """
    spec = cfg["links"]["base_link"]

    trunk = ET.Element("body")
    trunk.attrib.update({"name": "base_link", "pos": f"0 0 {SPAWN_HEIGHT_M}"})
    ET.SubElement(trunk, "freejoint", {"name": "root"})

    I = spec["inertia"]
    inertial = ET.SubElement(trunk, "inertial")
    inertial.attrib.update({
        "pos": " ".join(f"{v:.8f}" for v in spec["com_m"]),
        "mass": f"{spec['mass_kg']:.6f}",
        # MJCF fullinertia order is ixx iyy izz ixy ixz iyz
        "fullinertia": " ".join(f"{I[k]:.9e}" for k in
                                ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")),
    })

    # Adopt base_link's loose geoms and the promoted leg bodies.
    #
    # Every geom sitting loose in <worldbody> got there because MuJoCo welded the
    # root link to the world, so all of them belong to the trunk - the only geom the
    # world genuinely owns is the floor added just above. Matching on a "base_link"
    # name prefix was not enough: the URDF's VISUAL meshes are emitted with no name
    # attribute at all, so the trunk's visual mesh was left behind and rendered as a
    # ghost welded at the origin while the robot walked away from it. Harmless to
    # physics (contype=0, density=0) but wrong, and obvious the moment you open a
    # viewer.
    adopted_geoms, adopted_bodies = 0, 0
    for child in list(worldbody):
        if child.tag == "geom" and child.get("name") != "floor":
            worldbody.remove(child)
            trunk.append(child)
            adopted_geoms += 1
        elif child.tag == "body":
            worldbody.remove(child)
            trunk.append(child)
            adopted_bodies += 1

    if adopted_bodies != 4:
        raise SystemExit(f"expected 4 legs to re-parent, found {adopted_bodies}")
    worldbody.append(trunk)
    return trunk


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Gray's MJCF.")
    ap.add_argument("--check", action="store_true", help="run a settle + drop test")
    ap.add_argument("--out", default=OUT_MJCF)
    args = ap.parse_args()

    with open(ROBOT_YAML) as fh:
        cfg = yaml.safe_load(fh)
    servo = cfg["servo"]
    limit, effort = servo["limit_rad"], servo["effort_nm"]

    root = build_mjcf(SRC_URDF)
    root.set("model", "gray")

    # --- simulation options ---------------------------------------------------------
    for tag, attrs in [
        ("compiler", {"angle": "radian", "autolimits": "true"}),
        ("option", {"timestep": str(TIMESTEP), "integrator": "implicitfast",
                    "cone": "elliptic", "impratio": "10"}),
    ]:
        el = root.find(tag)
        if el is None:
            el = ET.Element(tag)
            root.insert(0, el)
        el.attrib.update(attrs)

    # Offscreen framebuffer large enough for checkpoint renders and training GIFs;
    # MuJoCo defaults to 640x480 and errors out above it.
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1920", "offheight": "1080"})

    worldbody = root.find("worldbody")

    # --- ground + lighting ----------------------------------------------------------
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    tex = ET.SubElement(asset, "texture")
    tex.attrib.update({"name": "grid", "type": "2d", "builtin": "checker",
                       "rgb1": ".18 .20 .23", "rgb2": ".22 .24 .27",
                       "width": "512", "height": "512"})
    mat = ET.SubElement(asset, "material")
    mat.attrib.update({"name": "grid", "texture": "grid", "texrepeat": "8 8",
                       "reflectance": "0.1"})

    light = ET.Element("light")
    light.attrib.update({"pos": "0 0 2", "dir": "0 0 -1", "directional": "true"})
    worldbody.insert(0, light)
    floor = ET.Element("geom")
    floor.attrib.update({"name": "floor", "type": "plane", "size": "0 0 0.05",
                         "material": "grid", "condim": "3",
                         "friction": "0.8 0.005 0.0001"})
    worldbody.insert(1, floor)

    trunk = rebuild_trunk(worldbody, cfg)

    # --- separate collision shapes from the ones you look at ---------------------------
    # Every link carries two geoms: the decimated mesh, and a box/capsule primitive the
    # solver actually uses for contact. The URDF importer leaves both in group 0, so a
    # viewer draws the primitives ON TOP of the meshes and the robot appears to be made
    # of plain boxes. MuJoCo's convention is collision geometry in group 3, which
    # viewers hide by default; this is purely how it is drawn and changes no physics.
    hidden = 0
    for g in root.iter("geom"):
        if (g.get("name") or "").endswith("_collision"):
            g.set("group", str(COLLISION_GEOM_GROUP))
            hidden += 1

    shins = add_shank_capsules(root)
    excluded = exclude_false_contacts(root)

    # --- joint dynamics --------------------------------------------------------------
    joints = [j for j in root.iter("joint") if j.get("name")]
    for j in joints:
        j.set("armature", str(ARMATURE))
        j.set("damping", str(DAMPING))
        j.set("frictionloss", str(FRICTIONLOSS))
        j.set("range", f"{-limit:.6f} {limit:.6f}")

    # --- position actuators, one per joint --------------------------------------------
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    for j in joints:
        name = j.get("name")
        p = ET.SubElement(act, "position")
        p.attrib.update({
            "name": name, "joint": name, "kp": str(KP), "kv": str(KV),
            "ctrlrange": f"{-limit:.6f} {limit:.6f}",
            "forcerange": f"{-effort} {effort}",
        })

    # --- sensors: exactly what the real robot can measure -------------------------------
    sens = root.find("sensor")
    if sens is None:
        sens = ET.SubElement(root, "sensor")
    site = ET.SubElement(trunk, "site")
    site.attrib.update({"name": "imu", "pos": "0 0 0", "size": "0.01"})
    for tag, nm in [("framequat", "imu_quat"), ("gyro", "imu_gyro"),
                    ("accelerometer", "imu_acc")]:
        s = ET.SubElement(sens, tag)
        s.set("name", nm)
        s.set("objtype" if tag == "framequat" else "site", "site" if tag == "framequat" else "imu")
        if tag == "framequat":
            s.set("objname", "imu")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    xml = ET.tostring(root, encoding="unicode")
    with open(args.out, "w") as fh:
        fh.write(xml)

    m = mujoco.MjModel.from_xml_path(args.out)
    print(f"wrote {args.out}")
    print(f"  bodies {m.nbody}  joints {m.njnt}  dofs {m.nv}  actuators {m.nu}  "
          f"geoms {m.ngeom}  ({hidden + shins} collision geoms in group "
          f"{COLLISION_GEOM_GROUP}, hidden by default, "
          f"{shins} of them shank capsules added here)")
    print(f"  {excluded} false contact pair(s) excluded "
          f"(trunk against each thigh - see exclude_false_contacts)")
    print(f"  total mass {m.body_mass.sum()*1000:.1f} g   "
          f"(robot.yaml says {cfg['total_mass_kg']*1000:.1f} g)")
    print(f"  control {CONTROL_HZ} Hz over {TIMESTEP*1000:.0f} ms physics "
          f"= {int(1/CONTROL_HZ/TIMESTEP)} substeps/tick")

    if args.check:
        drop_test(m)


def drop_test(m: mujoco.MjModel) -> None:
    """Let the robot settle under gravity and report whether the model is sane."""
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)

    z = d.qpos[2]
    quat = d.qpos[3:7]
    # gravity direction in body frame; z-component ~ -1 means upright
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    up = R.reshape(3, 3)[2, 2]
    speed = np.linalg.norm(d.qvel[:3])

    print("\ndrop test (3 s under gravity, zero control):")
    print(f"  trunk height   {z*1000:7.1f} mm")
    print(f"  uprightness    {up:7.3f}   (1.0 = level, -1.0 = upside down)")
    print(f"  residual speed {speed*1000:7.1f} mm/s")
    ok = abs(speed) < 0.05 and np.isfinite(z)
    print(f"  -> {'settled, model is stable' if ok else 'NOT settled - investigate'}")
    if not np.all(np.isfinite(d.qpos)):
        print("  !! non-finite state: the model exploded")


if __name__ == "__main__":
    main()
