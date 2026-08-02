#!/usr/bin/env python3
"""Measure how far each joint can actually turn before the robot hits itself.

    python tools/measure_joint_limits.py                 # measure and print
    python tools/measure_joint_limits.py --write         # also write progress/joint_limits.json

WHY THIS EXISTS. Every joint in sim/models/gray.xml is given +/-2.3562 rad, which is
the DS3218MG's own travel of 270 degrees. That is what the SERVO can turn; nobody ever
checked what the ROBOT can turn. A policy handed the servo's range is free to command
poses that would jam the real machine, and the machine is in pieces so none of this
can be checked by hand.

## TWO WRONG WAYS TO MEASURE IT, BOTH TRIED AND BOTH RECORDED HERE

1. ASK MUJOCO FOR CONTACTS AS THE MODEL SHIPS. It filters contacts between a body and
   its direct parent, which is sensible in general - parts sharing a joint overlap at
   the pivot - and fatal here, because thigh-vs-shank IS the knee's limit. Measured:
   the front-left knee folded to -134.6 degrees, straight through its own thigh,
   reporting zero contacts. Every joint measures as fully free.

2. FORCE THOSE PAIRS ON AND USE MESH COLLISION. MuJoCo collides the CONVEX HULL of a
   mesh, and the hulls of two parts that share a pivot always interpenetrate. Measured
   at the assembled neutral pose: hip against thigh overlaps 29.7-30.0 mm, thigh
   against shank 21.6-21.9 mm. Sweeping the knee across its whole travel then gives
   18-25 mm of overlap AT EVERY ANGLE with no peak anywhere - a real jam is invisible
   inside 20 mm of permanent hull overlap. The hull is also nowhere near the part:
   305 cm3 against the thigh's real 99 cm3.

## WHAT THIS DOES INSTEAD

Triangle-level collision on the ACTUAL part geometry, via trimesh/FCL, with MuJoCo
used only for forward kinematics. No convex hulls anywhere. The meshes are read out of
the compiled MuJoCo model rather than off disk, so the vertices are guaranteed to be in
the same frame the geoms are placed in - loading the STL separately invites a silent
frame mismatch that would make every number wrong and none of them obviously wrong.

Mesh collision at this fidelity is far too slow to train with. That is the whole point
of baking the answer into joint limits: measure once, slowly, against the real shapes;
then train fast with cheap shapes, because a joint that cannot reach a colliding angle
does not need its contacts checked at runtime.

## WHAT IT REPORTS

A per-segment range (hip, thigh, knee) that is collision-free for EVERY sampled pose of
the other two joints on that leg, taken as the worst case over all four legs. A range
measured with the other joints parked is a range that fails the moment they move.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import mujoco
import numpy as np
import trimesh
from trimesh.collision import CollisionManager

MJCF = "sim/models/gray.xml"
OUT_JSON = "progress/joint_limits.json"

LEGS = ("fl", "fr", "br", "bl")
SEGMENTS = ("hip", "top", "bottom")
SEGMENT_LABEL = {
    "hip": "hip - swings the leg out sideways",
    "top": "thigh - shoulder pitch",
    "bottom": "knee / calf",
}

# Sweep resolution. 2 degrees is already finer than a servo horn can be fitted - the
# output spline steps in coarser increments than that - so a finer grid would report
# precision the hardware cannot act on.
STEP_DEG = 2.0

# How far inside the measured range to stop, in degrees. The meshes are the CAD, not
# the printed parts, and print tolerance plus servo horn spline error is a couple of
# degrees at the joint. Reporting the raw contact angle as the limit would put the
# policy's ceiling exactly on the point where the real robot binds.
SAFETY_MARGIN_DEG = 3.0


def _mujoco_meshes(model: mujoco.MjModel) -> dict[int, trimesh.Trimesh]:
    """Every visual mesh, in its geom's local frame, keyed by GEOM ID.

    Read out of MuJoCo rather than off disk on purpose. MuJoCo re-frames a mesh when
    it compiles it (it shifts vertices so the origin sits at the centre of mass), so an
    STL loaded straight from the file sits in a different frame from the geom that
    references it. Using the compiled vertices makes the frame question disappear.

    KEYED BY ID, NOT NAME, and that is not a style choice. The mesh geoms in
    sim/models/gray.xml carry no name attribute at all, so an earlier version of this
    keyed on mj_id2name's None, fell back to a synthetic string, and then looked that
    string back up with mj_name2id - which returns -1. Every part therefore reported
    body_id of geom[-1], every pair looked like a same-body pair, all of them were
    filtered out, and the sweep measured ZERO pairs while confidently printing that
    all three joints could use 98% of their travel.
    """
    out: dict[int, trimesh.Trimesh] = {}
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        f0, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts = model.mesh_vert[v0:v0 + nv].reshape(-1, 3).astype(np.float64)
        faces = model.mesh_face[f0:f0 + nf].reshape(-1, 3).astype(np.int64)
        out[int(gid)] = trimesh.Trimesh(verts, faces, process=False)
    return out


def _pairs_to_test(model: mujoco.MjModel, meshes: dict) -> list[tuple[int, int]]:
    """Part pairs whose contact is a real interference rather than an assembly detail.

    EXCLUDES DIRECTLY-JOINTED PAIRS, and that exclusion is measured rather than
    assumed. Exactly 10 of the 78 possible pairs overlap in the correctly ASSEMBLED
    neutral pose, and all 10 are joint pairs: base_link with each hip, each hip with
    its thigh, and two of the four thigh/shank pairs. Parts that share a pivot
    interlock by design, so "do they touch" is meaningless for them.

    Depth was tried as a way to separate the interlock from a jam, and it does not
    work on this geometry. Sweeping the front-right knee against its own thigh gives
    0.00 mm at -135 deg, 13.28 at -120, 1.26 at -105, 4.91 at -90, 20.01 at +60 and
    5.11 at +135: no monotone rise into a jam, just noise. The meshes are the
    DECIMATED visual meshes placed at joint frames approximated from the CAD export,
    and neither is accurate enough at the pivot to support that question.

    The knee is also the wrong joint to ask it of. It is driven through a push-rod
    whose ratio has never been measured (docs/PROJECT_NOTES.md lists it as open), so
    its real travel is set by the linkage geometry, not by the shank reaching its own
    thigh.

    The remaining 68 pairs have NO overlap at rest, so any contact between them is a
    genuine interference: a thigh reaching the body, a shank reaching the body, a leg
    reaching another leg. Those are what actually bound gross motion and they are what
    this measures.
    """
    gids = list(meshes)
    parent = {int(b): int(model.body_parentid[b]) for b in range(model.nbody)}
    out = []
    for a, b in itertools.combinations(gids, 2):
        ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
        if ba == bb:
            continue
        if parent.get(ba) == bb or parent.get(bb) == ba:
            continue
        out.append((a, b))
    return out


def _part_name(model: mujoco.MjModel, gid: int) -> str:
    """A readable name for an unnamed mesh geom: the body it belongs to."""
    return mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) or f"geom{gid}"


class SelfCollision:
    """Triangle-level self-intersection test for a given joint configuration."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.meshes = _mujoco_meshes(model)
        self.pairs = _pairs_to_test(model, self.meshes)
        # One manager per part. FCL builds a bounding-volume hierarchy per mesh once,
        # and only the transform changes per query, which is what makes a sweep of
        # thousands of poses affordable at ~8000 faces a part.
        self.managers = {}
        for gid, mesh in self.meshes.items():
            cm = CollisionManager()
            cm.add_object(str(gid), mesh)
            self.managers[gid] = cm

    def _place(self) -> dict[int, np.ndarray]:
        mujoco.mj_forward(self.model, self.data)
        out = {}
        for gid in self.meshes:
            T = np.eye(4)
            T[:3, :3] = self.data.geom_xmat[gid].reshape(3, 3)
            T[:3, 3] = self.data.geom_xpos[gid]
            out[gid] = T
        return out

    def hits(self, qpos: np.ndarray) -> bool:
        """True if any two parts intersect in this configuration."""
        self.data.qpos[:] = qpos
        T = self._place()
        for a, b in self.pairs:
            self.managers[a].set_transform(str(a), T[a])
            if self.managers[a].in_collision_single(self.meshes[b], transform=T[b]):
                return True
        return False


def measure(sc: SelfCollision, grid: int, step_deg: float,
            parked_span: float = 0.35) -> dict:
    """Per-segment collision-free range.

    PARKED_SPAN IS WHY THIS WORKS AT ALL. The first version parked the other two
    joints across their FULL travel and demanded the swept angle be clear for every
    combination. At +/-135 degrees on both, the legs collide with the body and each
    other whatever the third joint does, so every angle failed and all three segments
    reported BLOCKED AT ZERO - including the assembled neutral pose, which is plainly
    not blocked.

    The other joints are now parked across a band around neutral instead. That answers
    the question actually being asked - how far can this joint swing while the leg is
    somewhere reasonable - rather than an unanswerable one about simultaneous extremes.
    The box that comes out is then verified by sampling, because a set of per-joint
    ranges measured one at a time is not automatically safe when all three move.
    """
    model = sc.model
    limit = float(model.jnt_range[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fl_hip")][1])
    sweep = np.arange(-limit, limit + 1e-9, np.radians(step_deg))
    parked = np.linspace(-limit * parked_span, limit * parked_span, grid)

    addr = {f"{leg}_{seg}": model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{seg}")]
        for leg in LEGS for seg in SEGMENTS}

    results = {}
    for seg in SEGMENTS:
        others = [s for s in SEGMENTS if s != seg]
        clear = np.ones(len(sweep), dtype=bool)
        for leg in LEGS:
            for combo in itertools.product(parked, repeat=len(others)):
                q = model.qpos0.copy()
                for o, v in zip(others, combo):
                    q[addr[f"{leg}_{o}"]] = v
                for k, ang in enumerate(sweep):
                    if not clear[k]:
                        continue
                    q[addr[f"{leg}_{seg}"]] = ang
                    if sc.hits(q):
                        clear[k] = False
        # The usable range is the unbroken run of clear angles containing zero.
        # Clear pockets beyond a jam are unreachable: the joint would have to pass
        # through the jam to get to them.
        zero = int(np.argmin(np.abs(sweep)))
        if not clear[zero]:
            results[seg] = {"blocked_at_zero": True}
            continue
        lo = hi = zero
        while lo > 0 and clear[lo - 1]:
            lo -= 1
        while hi < len(sweep) - 1 and clear[hi + 1]:
            hi += 1
        margin = np.radians(SAFETY_MARGIN_DEG)
        raw_lo, raw_hi = float(sweep[lo]), float(sweep[hi])
        results[seg] = {
            "blocked_at_zero": False,
            "contact_lo_rad": raw_lo,
            "contact_hi_rad": raw_hi,
            # What to actually give the policy: pulled in by the safety margin, and
            # never widened past where contact was found.
            "lo_rad": min(0.0, raw_lo + margin),
            "hi_rad": max(0.0, raw_hi - margin),
            "servo_lo_rad": -limit,
            "servo_hi_rad": limit,
        }
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help=f"write {OUT_JSON}")
    ap.add_argument("--grid", type=int, default=3,
                    help="parked positions per other joint (default 3)")
    ap.add_argument("--step", type=float, default=STEP_DEG,
                    help=f"sweep resolution in degrees (default {STEP_DEG})")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(MJCF)
    sc = SelfCollision(model)
    print(f"triangle-level self-collision on the real part meshes (no convex hulls)")
    print(f"  {len(sc.meshes)} parts, {len(sc.pairs)} pairs tested per pose")
    print(f"  sweeping every {args.step:.0f} deg, other joints parked at "
          f"{args.grid} positions each, worst case over all 4 legs")
    print(f"  reported limits pulled in {SAFETY_MARGIN_DEG:.0f} deg from first "
          f"contact, for print and horn tolerance\n")

    res = measure(sc, args.grid, args.step)

    # Verify the box. Ranges measured one joint at a time are not automatically safe
    # when all twelve move together, and a limit that is only true in isolation is
    # worse than no limit at all - it reads as a guarantee.
    rng = np.random.default_rng(0)
    addr = {f"{leg}_{seg}": model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{seg}")]
        for leg in LEGS for seg in SEGMENTS}
    usable = {s: r for s, r in res.items() if not r.get("blocked_at_zero")}
    clashes = 0
    trials = 300 if usable else 0
    for _ in range(trials):
        q = model.qpos0.copy()
        for leg in LEGS:
            for seg, r in usable.items():
                q[addr[f"{leg}_{seg}"]] = rng.uniform(r["lo_rad"], r["hi_rad"])
        if sc.hits(q):
            clashes += 1

    print(f"{'joint':32s} {'servo can turn':>16s} {'robot can turn':>18s} {'usable':>10s}")
    print("-" * 80)
    for seg in SEGMENTS:
        r = res[seg]
        if r.get("blocked_at_zero"):
            print(f"{SEGMENT_LABEL[seg]:32s} {'':>16s} {'BLOCKED AT ZERO':>18s}")
            continue
        servo_lo = np.degrees(r["servo_lo_rad"])
        servo_hi = np.degrees(r["servo_hi_rad"])
        servo_span = servo_hi - servo_lo
        lo, hi = np.degrees(r["lo_rad"]), np.degrees(r["hi_rad"])
        span = hi - lo
        servo_txt = f"{servo_lo:.0f} to {servo_hi:.0f}"
        robot_txt = f"{lo:+.0f} to {hi:+.0f}"
        usable_txt = f"{span:.0f} of {servo_span:.0f}"
        print(f"{SEGMENT_LABEL[seg]:32s} {servo_txt:>16s} {robot_txt:>18s} "
              f"{usable_txt:>10s} deg  ({span / servo_span * 100:.0f}%)")

    if trials:
        pct = clashes / trials * 100
        print(f"\nverification: {trials} random poses drawn from these ranges, all "
              f"twelve joints moving at once")
        if clashes:
            print(f"  {clashes} of them ({pct:.0f}%) had parts passing through each "
                  f"other. THE RANGES ARE NOT SAFE AS A BOX - they hold one joint at "
                  f"a time but not together. Narrow them, or treat them as guidance "
                  f"rather than as limits.")
        else:
            print(f"  none of them collided - the ranges are safe to use together")

    if args.write:
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump({
                "segments": res,
                "step_deg": args.step,
                "parked_grid": args.grid,
                "safety_margin_deg": SAFETY_MARGIN_DEG,
                "method": "triangle-level trimesh/FCL on the compiled MuJoCo meshes; "
                          "MuJoCo used for kinematics only. Convex hulls and MuJoCo's "
                          "own contact filtering both give wrong answers here - see "
                          "tools/measure_joint_limits.py.",
            }, fh, indent=2)
        print(f"\nwritten to {OUT_JSON}")


if __name__ == "__main__":
    main()
