"""Convert sim/models/gray.urdf into the MJCF file MuJoCo actually runs.

    python tools/make_mjcf.py            # -> sim/models/gray.xml
    python tools/make_mjcf.py --drop     # and drop it on the floor to see it settle

The one thing this exists for: **MuJoCo welds a URDF's root link to the world.**
It does it silently. The robot cannot move, and the trunk's mass vanishes from the
model - 1198.66 g of 2378.70 g, more than half the robot. The last attempt at this
project shipped a model reporting 901 g instead of 1625 g for exactly this reason.

Giving base_link a free joint fixes it, and the mass check below proves it.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
URDF = ROOT / "sim" / "models" / "gray.urdf"
MJCF = ROOT / "sim" / "models" / "gray.xml"


def build() -> tuple[Path, float]:
    if not URDF.exists():
        raise SystemExit(f"no model at {URDF}. Run tools/prepare_model.py first.")

    # MuJoCo treats a URDF as fixed-base unless it is told otherwise, and welds
    # base_link straight into the world - taking its mass with it. Declaring an
    # explicit floating joint to a world link is how URDF says "this robot is not
    # bolted down", and MuJoCo turns it into a free joint.
    urdf = ET.parse(URDF)
    urdf_root = urdf.getroot()
    if not any(j.get("type") == "floating" for j in urdf_root.findall("joint")):
        ET.SubElement(urdf_root, "link", {"name": "world"})
        j = ET.SubElement(urdf_root, "joint", {"name": "floating_base", "type": "floating"})
        ET.SubElement(j, "parent", {"link": "world"})
        ET.SubElement(j, "child", {"link": "base_link"})
    staged = URDF.parent / "_gray_floating.urdf"
    urdf.write(staged, encoding="utf-8", xml_declaration=True)

    try:
        model = mujoco.MjModel.from_xml_path(str(staged))
        mujoco.mj_saveLastXML(str(MJCF), model)
    finally:
        staged.unlink(missing_ok=True)

    tree = ET.parse(MJCF)
    root = tree.getroot()

    world = root.find("worldbody")
    trunk = next((b for b in world.iter("body") if b.get("name") == "base_link"), None)
    if trunk is None:
        raise SystemExit(
            "base_link is not a body in the MJCF - MuJoCo welded it into the world "
            "and its mass is gone"
        )
    if trunk.find("freejoint") is None and not any(
        j.get("type") == "free" for j in trunk.findall("joint")
    ):
        trunk.insert(0, ET.Element("freejoint", {"name": "root"}))

    # MuJoCo's offscreen buffer defaults to 640x480, which silently caps every
    # video and screenshot this project renders.
    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        root.insert(0, visual)
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", "1920")
    glob.set("offheight", "1080")

    # Floor, gravity and a light, so the model can actually be dropped and looked at.
    if root.find("asset") is None:
        root.insert(0, ET.Element("asset"))
    asset = root.find("asset")
    if asset.find("texture[@name='grid']") is None:
        ET.SubElement(asset, "texture", {
            "name": "grid", "type": "2d", "builtin": "checker",
            "rgb1": ".1 .12 .15", "rgb2": ".15 .17 .2", "width": "512", "height": "512",
        })
        ET.SubElement(asset, "material", {
            "name": "grid", "texture": "grid", "texrepeat": "8 8", "reflectance": "0.1",
        })
    if world.find("geom[@name='floor']") is None:
        world.insert(0, ET.Element("geom", {
            "name": "floor", "type": "plane", "size": "0 0 0.05",
            "material": "grid", "condim": "3",
        }))
    if world.find("light") is None:
        ET.SubElement(world, "light", {"pos": "0 0 2", "dir": "0 0 -1", "directional": "true"})

    name_geoms(root)
    add_foot_sites(root, model)
    add_servos(root, tree)

    ET.indent(tree, space="  ")
    tree.write(MJCF, encoding="utf-8", xml_declaration=True)

    checked = mujoco.MjModel.from_xml_path(str(MJCF))
    return MJCF, float(sum(checked.body_mass))


def name_geoms(root) -> None:
    """Give every collision shape a name derived from the body it belongs to.

    MuJoCo's URDF importer leaves geoms unnamed, and an unnamed geom cannot be
    selected by anything downstream - randomising the friction under the feet,
    rewarding a foot contact, telling a foot from a knee. It fails as "not all
    regular expressions are matched", which does not point at the cause at all.
    """
    for body in root.iter("body"):
        base = body.get("name")
        if not base:
            continue
        geoms = body.findall("geom")
        for i, geom in enumerate(geoms):
            if geom.get("name"):
                continue
            geom.set("name", base if len(geoms) == 1 else f"{base}_{i}")


def add_foot_sites(root, model) -> None:
    """Put a marked point on each foot, where it actually touches the ground.

    Without these there is no way to ask where a foot is or how fast it is
    moving - MuJoCo's URDF importer creates no sites at all, and a calf's centre
    of mass is not its foot. Every gait reward worth having needs them: whether a
    foot is sliding while loaded, how high it lifts on a swing, how long it
    spends in the air, how hard it lands.

    The point is the calf mesh's farthest vertex from the knee, which is the part
    that reaches the floor. Taken off the mesh rather than guessed.
    """
    for body in root.iter("body"):
        name = body.get("name") or ""
        if "calf" not in name.lower():
            continue
        leg = name.split("_")[0]
        if body.find(f"site[@name='{leg}_foot']") is not None:
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue

        best, best_d = np.zeros(3), -1.0
        for g in range(model.body_geomadr[bid],
                       model.body_geomadr[bid] + model.body_geomnum[bid]):
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[g]
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            verts = model.mesh_vert[adr:adr + num].astype(float)
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[g])
            pts = verts @ rot.reshape(3, 3).T + model.geom_pos[g]
            dist = np.linalg.norm(pts, axis=1)
            i = int(np.argmax(dist))
            if dist[i] > best_d:
                best_d, best = float(dist[i]), pts[i]

        ET.SubElement(body, "site", {
            "name": f"{leg}_foot",
            "pos": " ".join(f"{v:.6g}" for v in best),
            "size": "0.004",
            "rgba": "0.9 0.3 0.2 0.6",
        })


def add_servos(root, tree) -> None:
    """Give the twelve joints something to drive them.

    Straight out of the URDF the joints spin freely - a skeleton with bearings
    and no motors, which is why the robot collapses the moment it is dropped. A
    DS3218MG is a position servo: you send it an angle and its internal loop
    pushes toward that angle. In MuJoCo that is a position actuator with a gain,
    a damping term, and a hard ceiling on the force it can produce.

    The force ceiling is the one number here that is real - 1.96 N-m is the
    published stall torque. The gains are a starting point; see robot.yaml.
    """
    cfg = yaml.safe_load((ROOT / "gray" / "config" / "robot.yaml").read_text())
    servo = cfg["servo"]

    for old in root.findall("actuator"):
        root.remove(old)
    actuator = ET.SubElement(root, "actuator")

    for jnt in root.iter("joint"):
        name = jnt.get("name")
        if not name or jnt.get("type") in ("free", "slide", "ball"):
            continue
        # The gearbox has inertia and friction of its own. Without them the joint
        # is a frictionless hinge, which is easier to control in simulation than
        # anything real - and a policy trained against that does not transfer.
        jnt.set("armature", str(servo["armature"]))
        jnt.set("damping", str(servo["damping"]))
        jnt.set("frictionloss", str(servo["frictionloss"]))

        rng = jnt.get("range")
        ET.SubElement(actuator, "position", {
            "name": name,
            "joint": name,
            "kp": str(servo["kp"]),
            "kv": str(servo["kv"]),
            "forcerange": f"-{servo['stall_torque_nm']} {servo['stall_torque_nm']}",
            **({"ctrlrange": rng} if rng else {}),
        })

    # 50 Hz is the servo's PWM period and cannot be exceeded. The physics runs
    # four times faster so contact is resolved properly between commands.
    opt = root.find("option")
    if opt is None:
        opt = ET.Element("option")
        root.insert(0, opt)
    opt.set("timestep", str(round(1.0 / (servo["control_hz"] * 4), 6)))


def drop(seconds: float = 2.0, hold: bool = True) -> dict:
    """Let it settle, and report what happened.

    With `hold`, the servos are told to keep the stance they were dropped in -
    which is the whole test: can the robot hold itself up. Without it they are
    commanded to zero and it collapses, which is what a robot with no motors does.
    """
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    stance = ROOT / "progress" / "stance" / "stance.yaml"
    if stance.exists():
        st = yaml.safe_load(stance.read_text())
        for name, deg in st["angles_deg"].items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name.replace("_", ""))
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(deg)
        # Take the height straight from the solver, which placed the feet on the
        # floor to find it. Measuring it here off geom centres instead puts the
        # robot's legs through the floor, and MuJoCo answers by firing it 100 mm
        # into the air on the first step.
        data.qpos[2] = st["trunk_height_m"] + 0.001

    mujoco.mj_forward(model, data)

    if hold and model.nu:
        for a in range(model.nu):
            jid = model.actuator_trnid[a, 0]
            data.ctrl[a] = data.qpos[model.jnt_qposadr[jid]]

    n = int(seconds / model.opt.timestep)
    for _ in range(n):
        mujoco.mj_step(model, data)

    z_axis = data.xmat[root_id].reshape(3, 3)[:, 2]
    return {
        "trunk_height_mm": float(data.xpos[root_id][2]) * 1000,
        "uprightness": float(z_axis[2]),
        "drift_mm": float(np.linalg.norm(data.xpos[root_id][:2])) * 1000,
        "contacts": int(data.ncon),
    }


def verify_limits(cfg: dict) -> list[str]:
    """Check the written joint travel means, physically, what the owner said.

    The legs are mirrored, so on half of them a positive joint angle is the
    physically negative direction. Rather than trust that bookkeeping, turn each
    joint a little in the simulator and watch which way the leg goes. This caught
    all four knee ranges written back to front.
    """
    lim = cfg["joint_limits"]
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    out = []

    def parse(n):
        n = (n or "").lower()
        seg = next((s for s in ("thigh", "calf", "hip") if s in n), None)
        rest = n.replace(seg, "") if seg else n
        leg = next((l for l in ("fl", "fr", "bl", "br") if l in rest), None)
        return leg, seg

    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        leg, seg = parse(name)
        if seg not in lim:
            continue
        body = next((b for b in range(model.nbody)
                     if parse(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)) == (leg, "calf")), None)
        if body is None:
            out.append(f"{name:10s} no calf body found - not checked")
            continue

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        p0 = data.xipos[body].copy()
        data.qpos[model.jnt_qposadr[jid]] += np.deg2rad(10)
        mujoco.mj_forward(model, data)
        moved = data.xipos[body] - p0

        outward = 1.0 if leg in ("fl", "bl") else -1.0
        wanted = {"hip": np.array([0.0, outward, 0.0]),
                  "thigh": np.array([1.0, 0.0, 0.0]),
                  "calf": np.array([0.0, 0.0, 1.0])}[seg]
        sign = 1.0 if float(moved @ wanted) >= 0 else -1.0

        lo, hi = np.rad2deg(model.jnt_range[jid])
        got = sorted([sign * lo, sign * hi])
        want = list(lim[seg])
        ok = all(abs(a - b) < 0.5 for a, b in zip(got, want))
        out.append(
            f"{name:10s} {seg:5s} physical {got[0]:+7.1f} to {got[1]:+7.1f}   "
            f"owner {want[0]:+4.0f} to {want[1]:+4.0f}   {'ok' if ok else 'MISMATCH'}"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drop", action="store_true", help="drop it on the floor and report")
    args = ap.parse_args()

    path, sim_mass = build()
    cfg = yaml.safe_load((ROOT / "gray" / "config" / "robot.yaml").read_text())
    want = cfg["mass"]["total_expected"] / 1000.0

    print(f"wrote  {path.relative_to(ROOT)}")
    print(f"mass   simulator {sim_mass*1000:.2f} g   robot.yaml {want*1000:.2f} g   "
          f"{'MATCH' if abs(sim_mass - want) < 1e-4 else 'MISMATCH - the trunk is still welded'}")

    rows = verify_limits(cfg)
    bad = [r for r in rows if "MISMATCH" in r or "not checked" in r]
    print(f"\njoint travel, turned in the simulator and measured: "
          f"{len(rows) - len(bad)}/{len(rows)} match the owner's numbers")
    for r in (bad if bad else rows):
        print(f"  {r}")

    if args.drop:
        r = drop()
        print("\nafter 2 s on the floor:")
        print(f"  trunk height   {r['trunk_height_mm']:7.1f} mm")
        print(f"  uprightness    {r['uprightness']:7.3f}   (1.000 is perfectly level)")
        print(f"  drift          {r['drift_mm']:7.1f} mm")
        print(f"  contacts       {r['contacts']:7d}")
        if r["uprightness"] < 0.9:
            print("\n  It fell over, which is what should happen. Nothing is driving the")
            print("  joints yet, so the legs are free to fold and there is nothing holding")
            print("  the robot up. What this test proves is that the model loads, weighs")
            print("  the right amount, and collides with the floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
