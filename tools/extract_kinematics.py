#!/usr/bin/env python3
"""Extract Gray's per-leg kinematic parameters from the MJCF into robot.yaml.

Why this is a separate step: gray/kinematics.py has to run on the Raspberry Pi,
which will never have MuJoCo installed. So the geometry is resolved once here, on a
machine that does, and written to config as plain numbers. The same numbers then
drive both the simulator and the real robot - one source of truth, no sim/real drift.

Structure recovered from the model:

    trunk --[mount + Rx90]--> hip --[q0 about X]--> thigh --[q1 about Z]-->
    shank --[q2 about Z]--> foot

  * q0 is abduction, rotating about the trunk's fore-aft axis.
  * q1 and q2 are both about the hip frame's Z, which the Y-up->Z-up correction
    maps onto the robot's lateral axis - so they are hip-pitch and knee, and the
    leg below the abduction joint is a planar 2-link chain.
  * Because q1 and q2 share an axis, the foot's offset ALONG that axis is constant
    regardless of joint angle. That constant is the abduction offset, and it is what
    makes a closed-form solution possible.

The four legs are NOT exact mirrors - the CAD assembly carries a few mm and a
degree or two of variation per leg - so every leg is captured independently rather
than derived from one canonical leg and flipped.

Run:
    python tools/extract_kinematics.py            # appends to gray/config/robot.yaml
    python tools/extract_kinematics.py --report   # print only
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

MJCF = "sim/models/gray.xml"
ROBOT_YAML = "gray/config/robot.yaml"
LEGS = ("fl", "fr", "br", "bl")


def leg_params(m: mujoco.MjModel, leg: str) -> dict:
    def bid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

    def jid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

    def gid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)

    hip, top, bot = bid(f"{leg}_hip"), bid(f"{leg}_top"), bid(f"{leg}_bottom")

    mount_pos = m.body_pos[hip].copy()          # hip mount in trunk frame
    mount_quat = m.body_quat[hip].copy()        # includes the Y-up -> Z-up rotation
    abduct_axis = m.jnt_axis[jid(f"{leg}_hip")].copy()
    pitch_axis = m.jnt_axis[jid(f"{leg}_top")].copy()
    knee_axis = m.jnt_axis[jid(f"{leg}_bottom")].copy()

    hip_to_thigh = m.body_pos[top].copy()       # hip origin -> hip-pitch joint
    thigh_to_knee = m.body_pos[bot].copy()      # hip-pitch joint -> knee joint
    knee_to_foot = m.geom_pos[gid(f"{leg}_bottom_collision")].copy()

    # The pitch axis is +/-Z in this frame, so planar geometry lives in XY and the
    # Z components simply sum to a constant lateral offset.
    thigh_len = float(np.linalg.norm(thigh_to_knee[:2]))
    shank_len = float(np.linalg.norm(knee_to_foot[:2]))
    lateral_offset = float(hip_to_thigh[2] + thigh_to_knee[2] + knee_to_foot[2])

    # Direction each segment points when its joint is at zero, measured in-plane.
    thigh_zero = float(np.arctan2(thigh_to_knee[1], thigh_to_knee[0]))
    shank_zero = float(np.arctan2(knee_to_foot[1], knee_to_foot[0]))

    return {
        "mount_pos": mount_pos.tolist(),
        "mount_quat": mount_quat.tolist(),
        "abduct_axis": abduct_axis.tolist(),
        "pitch_sign": float(np.sign(pitch_axis[2])),
        "knee_sign": float(np.sign(knee_axis[2])),
        "hip_to_thigh": hip_to_thigh.tolist(),
        "thigh_len": thigh_len,
        "shank_len": shank_len,
        "lateral_offset": lateral_offset,
        "thigh_zero_rad": thigh_zero,
        "shank_zero_rad": shank_zero,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(MJCF)
    params = {leg: leg_params(m, leg) for leg in LEGS}

    print(f"{'leg':4s} {'thigh mm':>9s} {'shank mm':>9s} {'lat off mm':>11s} "
          f"{'thigh0 deg':>11s} {'shank0 deg':>11s} {'pitch':>6s} {'knee':>5s}")
    for leg, p in params.items():
        print(f"{leg:4s} {p['thigh_len']*1000:9.2f} {p['shank_len']*1000:9.2f} "
              f"{p['lateral_offset']*1000:11.2f} "
              f"{np.degrees(p['thigh_zero_rad']):11.2f} "
              f"{np.degrees(p['shank_zero_rad']):11.2f} "
              f"{p['pitch_sign']:6.0f} {p['knee_sign']:5.0f}")

    tl = np.array([p["thigh_len"] for p in params.values()])
    sl = np.array([p["shank_len"] for p in params.values()])
    print(f"\n  thigh {tl.mean()*1000:.2f} mm  (spread {np.ptp(tl)*1000:.2f} mm)")
    print(f"  shank {sl.mean()*1000:.2f} mm  (spread {np.ptp(sl)*1000:.2f} mm)")
    print(f"  max leg reach {(tl+sl).max()*1000:.1f} mm")
    print(f"\n  Per-leg spread is CAD assembly variation, not real: the printed parts")
    print(f"  are identical (all four shanks are the same 46.4 cm3 mesh). Kept per-leg")
    print(f"  anyway so the model matches the URDF exactly.")

    if args.report:
        return

    with open(ROBOT_YAML) as fh:
        text = fh.read()
    marker = "\nlegs:\n"
    if marker in text:
        text = text[: text.index(marker)]

    lines = [text.rstrip("\n"), "", "# Per-leg kinematics. GENERATED by tools/extract_kinematics.py",
             "# Chain: trunk -[mount]-> hip -[q0 about X]-> thigh -[q1 about Z]->",
             "#        shank -[q2 about Z]-> foot", "legs:"]
    for leg, p in params.items():
        lines.append(f"  {leg}:")
        for k, v in p.items():
            if isinstance(v, list):
                lines.append(f"    {k}: [{', '.join(f'{x:.9f}' for x in v)}]")
            else:
                lines.append(f"    {k}: {v:.9f}")
    with open(ROBOT_YAML, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  wrote leg parameters to {ROBOT_YAML}")


if __name__ == "__main__":
    main()
