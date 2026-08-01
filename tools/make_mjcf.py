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
    adopted_geoms, adopted_bodies = 0, 0
    for child in list(worldbody):
        if child.tag == "geom" and (child.get("name") or "").startswith("base_link"):
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
          f"geoms {m.ngeom}")
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
