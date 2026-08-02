#!/usr/bin/env python3
"""Pose Gray by hand in a 3D window, and save the poses you set.

    python tools/pose_editor.py

A window opens with the robot and a slider for every one of the twelve joints. Drag
them and the robot moves exactly as the real one would: each joint obeys its own
range, and nothing else moves. Set the shape you want, then press a key to save it.

## TYPING EXACT ANGLES

The viewer's sliders cannot be typed into and are awkward to land on a round number,
so this also reads commands from the TERMINAL you launched it from. Type there, press
enter, and the robot moves - the sliders follow along, so both ways of driving it stay
in step.

    hip 90            all four hips 90 degrees OUT
    top -45           all four thighs 45 degrees BACK
    bottom 60         all four feet 60 degrees UP
    fl_hip 90         one named joint
    front top -30     both front thighs      (front, back, left, right)
    all 0             every joint to zero
    signs             show which way positive goes on each leg
    invert bl         flip the direction of one leg (or: invert hip, invert fl_top)
    raw hip 90        set the RAW joint angle, ignoring the direction fix
    show              print every angle
    check             test for parts passing through each other
    physics / pose    switch mode, same as SPACE
    save belly        save the current pose  (or: save standing)
    load belly        bring a saved pose back
    quit              close

## POSITIVE MEANS THE SAME THING ON EVERY LEG

The legs are mirrored, so the raw joint angles do NOT agree with each other: measured
on this model, +10 degrees of knee lifts the front-RIGHT foot 26 mm and drops the
front-LEFT one 22 mm. Commanding all four to the same number would fold the robot into
a heap.

So a positive number here is a physical direction, not a joint angle:

    hip     +  swings the leg OUT, away from the body
    top     +  swings the leg FORWARD, towards the nose
    bottom  +  lifts the FOOT UP

and the per-leg sign needed to achieve that is MEASURED when this starts, by nudging
each joint and watching which way the foot actually goes. It is not written down,
because writing it down is how it was wrong: the mirroring runs front/back on the hips
and left/right on the thighs and knees, and an earlier version of this file assumed
left/right for all three and flipped two legs the wrong way.

`signs` prints what it measured. `invert` flips any joint, leg or family if you want
the opposite convention. `raw` bypasses the whole thing and sets the joint angle
directly, which is what the sliders show and what gets saved to file.

Angles are DEGREES. Every joint is clamped to the servo's real travel of +/-135
degrees, because that is the DS3218MG's actual range - asking for 200 would describe a
pose the robot cannot be put into.

    KEYS
      SPACE  switch between POSING and PHYSICS
      B   save the current pose as BELLY   (the pose every training episode starts in)
      S   save the current pose as STANDING (what it is learning to reach)
      C   check this pose for parts passing through each other, and print the result
      R   reset every joint back to zero
      P   print the current joint angles to the terminal

Saved to progress/poses.json.

## TWO MODES, AND POSING IS THE DEFAULT

POSING mode runs NO physics at all. Each joint goes exactly where its slider says and
stays there - nothing sags, nothing drifts, and the robot cannot push itself around
the floor while you work. The body is automatically lowered so its lowest part rests
on the ground, so what you see is the pose as it would actually sit.

PHYSICS mode turns gravity and the servos back on and lets go. This is the test: a
pose that collapses here is not a pose the robot can start from, and it is far better
to find that out now than after a night of training.

The viewer's own pause button does not apply. This script drives the simulation
itself, so SPACE here is the control that matters.

## HOW TO DRIVE IT

The sliders are on the RIGHT of the window, under "Control". There is one per joint,
named like `fl_hip`, `fl_top`, `fl_bottom`:

    <leg>_hip     swings the whole leg out sideways
    <leg>_top     the shoulder, swings the leg forwards and backwards
    <leg>_bottom  the knee

and the legs are fl = front left, fr = front right, bl = back left, br = back right.

## WHY GRAVITY STARTS OFF

With gravity off the servos simply hold whatever you dial in, so the robot stays where
you put it and you can build a pose in peace. Press G and it has to hold itself up for
real. A pose that collapses the moment you do that is not a pose the robot can start
from, and it is much better to find that out here than after a night of training.

## WHAT THE COLLISION CHECK IS ACTUALLY CHECKING

C tests the REAL part meshes against each other, triangle by triangle - not the
simplified blocks the physics engine uses at speed. It also ignores the pairs that are
joined together, because parts that share a pivot interlock by design: measured on
this robot, the hip and thigh overlap 2.8 mm in the correctly assembled pose. Only
parts that are NOT joined to each other can meaningfully pass through one another, and
those are the 66 pairs it looks at.
"""

from __future__ import annotations

import json
import os
import sys

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_joint_limits import SelfCollision, _part_name  # noqa: E402

MJCF = "sim/models/gray.xml"
OUT_JSON = "progress/poses.json"
LEGS = ("fl", "fr", "br", "bl")
SEGMENTS = ("hip", "top", "bottom")


def lowest_point(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """World z of the lowest point of any collision geom.

    Computed from each shape's real extent - box corners, capsule end caps, sphere
    surface - and NOT from geom_rbound. rbound is the radius of a BOUNDING SPHERE, so
    for a flat slab like Gray's chassis it reports a part that reaches 127 mm below
    its own centre when the chassis is only 12.5 mm thick. An earlier version of this
    project used rbound here and concluded the shanks hung 104 mm below the belly,
    which is not true of any pose.
    """
    lowest = float("inf")
    for g in range(model.ngeom):
        if model.geom_group[g] != 3:
            continue
        pos = data.geom_xpos[g]
        R = data.geom_xmat[g].reshape(3, 3)
        size = model.geom_size[g]
        kind = model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            corners = np.array([[x, y, z]
                                for x in (-size[0], size[0])
                                for y in (-size[1], size[1])
                                for z in (-size[2], size[2])])
            lowest = min(lowest, float(((R @ corners.T).T + pos)[:, 2].min()))
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            half = np.array([0.0, 0.0, size[1]])
            ends = np.array([(R @ half) + pos, (R @ -half) + pos])
            lowest = min(lowest, float(ends[:, 2].min() - size[0]))
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            lowest = min(lowest, float(pos[2] - size[0]))
        elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        else:
            lowest = min(lowest, float(pos[2] - model.geom_rbound[g]))
    return lowest


def level_feet(model: mujoco.MjModel, data: mujoco.MjData,
               qadr: dict[str, int], tol_mm: float = 0.5,
               max_iter: int = 60) -> dict[str, float]:
    """Nudge each leg's shoulder until all four feet sit on the floor.

    WHY THIS IS NEEDED AT ALL: setting every joint to zero does NOT stand the robot
    up square. Measured on this model, the four feet then sit at 0.0, 2.5, 8.2 and
    18.5 mm - the legs are not identical in the CAD, and the foot positions differ by
    about 10 mm between them. So "reset" produced a robot balanced on one foot with
    another 18 mm in the air, which is a poor thing to call a starting pose.

    Solved per leg on the KNEE, because that is the joint with leverage on foot
    HEIGHT. Measured: +10 degrees of knee moves the foot about 25 mm up, while the
    same at the shoulder moves it 30 mm FORWARD and only about 3 mm up. Solving on
    the shoulder barely closed the gap - 18.5 mm became 17.2 mm - because it was
    pushing the foot along the floor rather than down onto it. Returns the raw knee
    angles it chose.
    """
    feet = {leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                   f"{leg}_bottom_collision") for leg in LEGS}
    limit = float(model.jnt_range[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fl_bottom")][1])

    def foot_z(leg: str) -> float:
        mujoco.mj_forward(model, data)
        return float(data.geom_xpos[feet[leg]][2] - model.geom_size[feet[leg]][0])

    # Target: the height of the foot that is already lowest, so the body does not have
    # to move and no leg is asked to reach further than it can.
    for leg in LEGS:
        data.qpos[qadr[f"{leg}_bottom"]] = 0.0
    mujoco.mj_forward(model, data)
    target = min(foot_z(leg) for leg in LEGS)

    # SCANNED, NOT BISECTED. Foot height against shoulder angle is a rotation, so it
    # rises, peaks and falls again - it is not monotonic and bisection has no bracket
    # to work with. The first version bisected anyway and drove three shoulders to
    # their +/-135 degree end stops, lifting those feet 280 mm into the air.
    #
    # A scan also lets the nearest-to-zero solution win, so a leg that is already
    # nearly right is nudged rather than swung round to a second solution on the far
    # side of the arc.
    angles = np.linspace(-limit, limit, 1081)          # 0.25 degree steps
    chosen: dict[str, float] = {}
    for leg in LEGS:
        adr = qadr[f"{leg}_bottom"]
        # Score every angle, then take the best - ranked on how close the foot lands
        # and, among those that land equally well, on how little the joint had to move.
        scored = []
        for angle in angles:
            data.qpos[adr] = angle
            err_mm = abs(foot_z(leg) - target) * 1000.0
            scored.append((max(err_mm, tol_mm), abs(angle), float(angle)))
        _, _, best_angle = min(scored)
        data.qpos[adr] = best_angle
        chosen[f"{leg}_bottom"] = best_angle
    mujoco.mj_forward(model, data)
    return chosen


# Which legs each group word covers. "front"/"back" are the ends of the robot;
# "left"/"right" are its own left and right, so bl is back-LEFT.
GROUPS = {
    "all": LEGS,
    "front": ("fl", "fr"),
    "back": ("bl", "br"),
    "left": ("fl", "bl"),
    "right": ("fr", "br"),
}

# What a POSITIVE command means, physically, for each joint family. Chosen so that the
# same number does the same visible thing on all four legs, which is not what the raw
# joint angles do - the legs are mirrored, so raw +10 degrees lifts one foot and drops
# the one opposite it.
NATURAL = {
    "hip": ("out", 1),      # + swings the leg away from the body   (body +Y is left)
    "top": ("forward", 0),  # + swings the leg towards the nose     (body +X is forward)
    "bottom": ("up", 2),    # + lifts the foot                      (body +Z is up)
}


def measure_signs(model: mujoco.MjModel) -> dict[str, float]:
    """Per-joint sign that makes a positive command mean the direction in NATURAL.

    MEASURED AT STARTUP, NOT WRITTEN DOWN. An earlier version of this file hardcoded
    which legs to negate and got it wrong on two of them: it assumed the mirroring ran
    left/right for every joint, and in this model the hips are paired front/back while
    the thighs and knees are paired left/right. `mirror hip 120` was folding the back
    legs inward while the front legs went outward.

    Nothing here is assumed. Each joint is nudged +10 degrees, the foot's movement is
    read in the trunk's own frame, and the sign is whatever makes that movement point
    the way NATURAL says it should. If the URDF is ever re-exported with different axes
    this simply re-measures and stays correct.
    """
    data = mujoco.MjData(model)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    signs: dict[str, float] = {}

    def foot(leg: str) -> np.ndarray:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                              f"{leg}_bottom_collision")
        R = data.xmat[trunk].reshape(3, 3)
        return R.T @ (data.geom_xpos[g] - data.xpos[trunk])

    for leg in LEGS:
        # "Out" is away from the robot's centreline, so it is +Y for the two left legs
        # and -Y for the two right ones.
        outward = 1.0 if leg in ("fl", "bl") else -1.0
        for seg in SEGMENTS:
            direction, axis = NATURAL[seg]
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{seg}")
            adr = model.jnt_qposadr[jid]
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            before = foot(leg)
            mujoco.mj_resetData(model, data)
            data.qpos[adr] = np.radians(10.0)
            mujoco.mj_forward(model, data)
            moved = (foot(leg) - before)[axis]
            wanted = outward if direction == "out" else 1.0
            signs[f"{leg}_{seg}"] = 1.0 if moved * wanted >= 0 else -1.0
    return signs


def parse_command(line: str, joints: list[str]) -> tuple[dict[str, float], str]:
    """(joint -> degrees, message). An empty dict means nothing to apply."""
    parts = line.strip().lower().split()
    if not parts:
        return {}, ""

    def number(token: str) -> float | None:
        try:
            return float(token)
        except ValueError:
            return None

    # <group> <segment> <deg>
    if len(parts) == 3 and parts[0] in GROUPS and parts[1] in SEGMENTS:
        deg = number(parts[2])
        if deg is None:
            return {}, f"'{parts[2]}' is not a number"
        legs = GROUPS[parts[0]]
        return ({f"{leg}_{parts[1]}": deg for leg in legs},
                f"{parts[0]} {parts[1]} -> {deg:g} deg")

    if len(parts) == 2:
        deg = number(parts[1])
        if deg is None:
            return {}, f"'{parts[1]}' is not a number"
        # <segment> <deg>  - all four legs
        if parts[0] in SEGMENTS:
            return ({f"{leg}_{parts[0]}": deg for leg in LEGS},
                    f"all {parts[0]} -> {deg:g} deg")
        # <joint> <deg>
        if parts[0] in joints:
            return {parts[0]: deg}, f"{parts[0]} -> {deg:g} deg"
        # all <deg>
        if parts[0] == "all":
            return ({j: deg for j in joints}, f"every joint -> {deg:g} deg")
        return {}, f"'{parts[0]}' is not a joint or a group"

    return {}, f"could not read '{line.strip()}'"


def _load() -> dict:
    if not os.path.exists(OUT_JSON):
        return {}
    try:
        with open(OUT_JSON, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save(poses: dict) -> None:
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(poses, fh, indent=2)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    sc = SelfCollision(model)

    joints = [f"{leg}_{seg}" for leg in LEGS for seg in SEGMENTS]
    qadr = {j: model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints}
    aid = {j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
           for j in joints}
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    poses = _load()
    # Posing is the default. See the module docstring: physics running while you drag
    # sliders means the robot sags, drifts and shoves itself around the floor.
    state = {"physics": False, "limp": False}

    # Stashed so LIMP can switch the servos off and back on. A position actuator holds
    # its joint through gainprm[0] (how hard it pulls towards the target) and
    # biasprm[1:3] (the position and velocity feedback); zeroing all three leaves the
    # joint with nothing driving it but its own friction, which is what a powered-down
    # servo actually is.
    servo_gain = model.actuator_gainprm.copy()
    servo_bias = model.actuator_biasprm.copy()

    def set_limp(on: bool) -> None:
        state["limp"] = on
        if on:
            model.actuator_gainprm[:, 0] = 0.0
            model.actuator_biasprm[:, 1] = 0.0
            model.actuator_biasprm[:, 2] = 0.0
            print("\n  LIMP - servos off. Under physics the legs will just flop.")
        else:
            model.actuator_gainprm[:] = servo_gain
            model.actuator_biasprm[:] = servo_bias
            print("\n  servos back on - they will hold the commanded angles")

    def angles() -> dict[str, float]:
        return {j: float(data.qpos[qadr[j]]) for j in joints}

    def show_angles() -> None:
        print("\n  joint angles (degrees):")
        for leg in LEGS:
            row = "  ".join(
                f"{seg}={np.degrees(data.qpos[qadr[f'{leg}_{seg}']]):+7.1f}"
                for seg in SEGMENTS)
            print(f"    {leg}   {row}")

    def report() -> None:
        mujoco.mj_forward(model, data)
        R = data.xmat[trunk].reshape(3, 3)
        touching = sorted({
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                               c.geom2 if c.geom1 == floor else c.geom1) or "?")
            .replace("_collision", "")
            for c in (data.contact[i] for i in range(data.ncon))
            if floor in (c.geom1, c.geom2)
        })
        print(f"    trunk centre {data.xpos[trunk][2] * 1000:6.1f} mm   "
              f"level {R[2, 2]:+.3f}   "
              f"on the floor: {', '.join(touching) if touching else 'nothing'}")

    def check(): # -> tuple[bool, str]
        """(clean, message). The message goes on the PANEL, not just the terminal.

        It used to only print. From the panel the button then looked dead, because the
        answer appeared in a console window behind the 3D view - the check was running
        and reporting correctly to somewhere nobody was looking.
        """
        clash = []
        sc.data.qpos[:] = data.qpos
        T = sc._place()
        for a, b in sc.pairs:
            sc.managers[a].set_transform(str(a), T[a])
            if sc.managers[a].in_collision_single(sc.meshes[b], transform=T[b]):
                clash.append((_part_name(model, a), _part_name(model, b)))
        if clash:
            listed = ", ".join(f"{a} into {b}" for a, b in clash[:3])
            more = f" and {len(clash) - 3} more" if len(clash) > 3 else ""
            msg = f"CLIPPING - {len(clash)} pair(s): {listed}{more}"
        else:
            msg = f"No clipping. All {len(sc.pairs)} part pairs are clear."
        print(f"    {msg}")
        return (not clash), msg

    def key_callback(keycode: int) -> None:
        key = chr(keycode).upper() if 0 < keycode < 0x110000 else ""
        if keycode == 32:                      # SPACE
            state["physics"] = not state["physics"]
            if state["physics"]:
                # Hand the current pose over to the solver as its starting state, and
                # tell the servos to hold it, so letting go is a fair test of THIS
                # pose rather than of whatever the actuators were last commanded.
                data.qvel[:] = 0.0
                for j in joints:
                    data.ctrl[aid[j]] = data.qpos[qadr[j]]
                print("\n  PHYSICS - gravity on"
                      + (", servos LIMP - the legs will flop" if state["limp"]
                         else ", servos holding. Does it stay up?"))
            else:
                # KEEP WHATEVER PHYSICS PRODUCED. Coming back to posing used to snap
                # the robot to the old slider values, throwing away the settle you had
                # just watched - which made "drop it and then adjust from there"
                # impossible. The joints, and the body's attitude, are read back out
                # and become the new starting point.
                data.qvel[:] = 0.0
                for j in joints:
                    data.ctrl[aid[j]] = data.qpos[qadr[j]]
                print("\n  POSING - keeping the pose physics left it in.")
        elif key == "C":
            print("\n  checking the real part meshes against each other:")
            check()
            report()
        elif key == "R":
            for j in joints:
                data.qpos[qadr[j]] = 0.0
                data.ctrl[aid[j]] = 0.0
            data.qvel[:] = 0.0
            print("\n  all joints back to zero")
        elif key == "P":
            show_angles()
            report()
        elif key in ("B", "S"):
            name = "belly" if key == "B" else "standing"
            print(f"\n  saving as {name.upper()} ...")
            clean, _ = check()
            report()
            poses[name] = {
                "angles_rad": angles(),
                "trunk_height_mm": float(data.xpos[trunk][2]) * 1000.0,
                "self_collision_free": clean,
                "checked_under_physics": state["physics"],
            }
            _save(poses)
            show_angles()
            print(f"  saved to {OUT_JSON}"
                  + ("" if clean else "  (WARNING: parts are passing through each"
                                      " other in this pose)"))

    limit = float(model.jnt_range[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fl_hip")][1])
    limit_deg = np.degrees(limit)
    signs = measure_signs(model)
    # Kept so the panel's "reverse" tickbox is a toggle against what was measured,
    # rather than a second sign that has to be tracked separately.
    base_signs = dict(signs)

    def apply(targets: dict[str, float], raw: bool = False) -> None:
        """Write typed angles into ctrl, which is also what the sliders write to, so
        typing and dragging drive the same values and never disagree.

        `raw=False` means the number is a PHYSICAL command - out, forward, up - and the
        measured per-joint sign is applied so all four legs do the same visible thing.
        `raw=True` writes the joint angle straight through, which is what the sliders
        and the saved pose files hold.
        """
        for name, deg in targets.items():
            sign = 1.0 if raw else signs[name]
            clamped = float(np.clip(np.radians(deg) * sign, -limit, limit))
            if abs(np.degrees(clamped)) < abs(deg) - 1e-6:
                print(f"    {name} clamped to {np.degrees(clamped):+.1f} deg "
                      f"(the servo only turns +/-{limit_deg:.0f})")
            data.ctrl[aid[name]] = clamped
            if not state["physics"]:
                data.qpos[qadr[name]] = clamped

    def run_panel(viewer) -> None:
        """Open the slider window and let it drive everything.

        SINGLE THREADED ON PURPOSE. The first version ran this panel in its own thread
        while the main thread stepped the simulation and synced the 3D view, and MuJoCo
        aborted with "attempting to copy mjData while stack is in use" - the viewer
        copies mjData when it draws, and the panel was reading the same data to update
        its status line. Tk's event loop now owns the model, the physics and the redraw.
        """
        from pose_panel import PosePanel

        # What the group sliders last commanded, so a per-leg slider can be offset
        # from the group rather than fighting it.
        group = {seg: 0.0 for seg in SEGMENTS}
        per = {j: 0.0 for j in joints}
        # The robot's attitude in the world, not a joint. Applied in posing mode only:
        # under physics the solver decides which way up it is.
        body_rpy = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}

        def push(name: str) -> None:
            seg = name.split("_", 1)[1]
            apply({name: group[seg] + per[name]})

        def set_deg(key: str, deg: float) -> None:
            if key.startswith("body_"):
                body_rpy[key[len("body_"):]] = deg
            elif key.startswith("group_"):
                seg = key[len("group_"):]
                group[seg] = deg
                for leg in LEGS:
                    push(f"{leg}_{seg}")
            else:
                per[key] = deg
                push(key)

        def get_deg(key: str) -> float:
            return per.get(key, 0.0)

        def set_invert(key: str, on: bool) -> None:
            want = -1.0 if on else 1.0
            base = base_signs[key]
            signs[key] = base * want
            push(key)

        def status() -> str:
            mujoco.mj_forward(model, data)
            R = data.xmat[trunk].reshape(3, 3)
            touching = sorted({
                (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                   c.geom2 if c.geom1 == floor else c.geom1) or "?")
                .replace("_collision", "")
                for c in (data.contact[i] for i in range(data.ncon))
                if floor in (c.geom1, c.geom2)
            })
            return (f"trunk {data.xpos[trunk][2] * 1000:.0f} mm   "
                    f"level {R[2, 2]:+.2f}   "
                    f"resting on: {', '.join(touching) if touching else 'nothing'}")

        def do_save(name: str) -> None:
            key_callback(ord("B") if name == "belly" else ord("S"))

        def do_load(name: str) -> dict | None:
            saved = _load().get(name)
            if not saved:
                print(f"    no pose called '{name}' saved yet")
                return None
            out = {}
            for j, rad in saved["angles_rad"].items():
                # Stored raw, shown physical, so the panel reads back what it wrote.
                per[j] = float(np.degrees(rad)) * signs[j]
                out[j] = per[j]
            for seg in SEGMENTS:
                group[seg] = 0.0
            for j in joints:
                push(j)
            print(f"    loaded {name}")
            return out

        def do_mode(physics: bool) -> dict | None:
            """Switch mode. Returns slider values to adopt, or None to leave them.

            Leaving physics hands back what the robot actually settled into, so the
            sliders agree with what is on screen. Without this the next slider touch
            would yank the robot back to its pre-physics pose.
            """
            if physics == state["physics"]:
                return None
            key_callback(32)
            if physics:
                return None
            for seg in SEGMENTS:
                group[seg] = 0.0
            for j in joints:
                per[j] = float(np.degrees(data.qpos[qadr[j]])) * signs[j]
            # The body will have tipped, so read its attitude back too.
            quat = data.qpos[3:7]
            roll, pitch, yaw = Rotation.from_quat(
                [quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz", degrees=True)
            body_rpy.update(roll=float(roll), pitch=float(pitch), yaw=float(yaw))
            out = dict(per)
            out.update({f"body_{k}": v for k, v in body_rpy.items()})
            return out

        def do_reset() -> dict:
            """Everything back to zero, INCLUDING the body's attitude.

            Reset used to leave the body wherever it had been rolled or pitched
            to, so 'reset' produced a robot with straight legs lying on its side.
            Levelling it means the auto-drop puts all four feet on the floor,
            which is what a reset is expected to look like.
            """
            for seg in SEGMENTS:
                group[seg] = 0.0
            for j in joints:
                per[j] = 0.0
                push(j)
            body_rpy.update(roll=0.0, pitch=0.0, yaw=0.0)
            # Square it up on the floor. Zeroing the joints leaves one foot
            # 18 mm in the air, because the legs differ in the CAD.
            for j in joints:
                data.qpos[qadr[j]] = 0.0
            quat = np.array([1.0, 0.0, 0.0, 0.0])
            data.qpos[3:7] = quat
            for name, rad in level_feet(model, data, qadr).items():
                per[name] = float(np.degrees(rad)) * signs[name]
                push(name)
            out = dict(per)
            out.update({f"body_{k}": 0.0 for k in body_rpy})
            return out

        def tick() -> None:
            if state["physics"]:
                mujoco.mj_step(model, data)
            else:
                # No physics: joints go exactly where the sliders say, and the robot
                # is dropped onto the floor so the pose is judged where it would rest.
                for j in joints:
                    data.qpos[qadr[j]] = data.ctrl[aid[j]]
                # Attitude from the body sliders. Roll then pitch then yaw about the
                # world axes, and MuJoCo wants (w,x,y,z) where scipy hands back
                # (x,y,z,w) - getting that order wrong silently tips the robot over.
                quat = Rotation.from_euler(
                    "xyz",
                    [body_rpy["roll"], body_rpy["pitch"], body_rpy["yaw"]],
                    degrees=True,
                ).as_quat()
                data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                data.qpos[2] -= lowest_point(model, data)
                mujoco.mj_forward(model, data)
            viewer.sync()

        PosePanel(
            limit_deg,
            set_deg=set_deg, get_deg=get_deg,
            set_invert=set_invert,
            get_invert=lambda k: signs[k] != base_signs[k],
            on_check=lambda: check()[1],
            on_save=do_save, on_load=do_load,
            on_mode=do_mode,
            set_limp=set_limp,
            get_limp=lambda: state["limp"],
            on_reset=do_reset,
            get_status=status,
            on_tick=tick,
            is_alive=viewer.is_running,
        ).run()

    print(__doc__)
    print(f"  {len(sc.pairs)} part pairs will be checked for clipping")
    if poses:
        print(f"  already saved in {OUT_JSON}: {', '.join(sorted(poses))}")
    print("\n  POSING mode. Use the control panel window - sliders, number boxes, "
          "reverse tickboxes and buttons.")
    print("  Close either window to quit.\n")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback,
        show_left_ui=True, show_right_ui=True,
    ) as viewer:
        # The panel's event loop drives the simulation and the redraw from here on,
        # and it returns when either window is closed.
        run_panel(viewer)


if __name__ == "__main__":
    main()
