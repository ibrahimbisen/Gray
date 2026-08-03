"""Find a standing pose, and work out whether the servos can hold it.

    python scripts/find_stance.py                  # sweep heights, pick the best
    python scripts/find_stance.py --height 0.19    # solve one height
    python scripts/find_stance.py --render         # and film it

Two questions, in order:

1. **Where do the joints have to be** for the robot to stand with its feet under
   its hips? The model's zero pose is the sprawl the CAD happened to be in when
   it was exported - feet 557 mm apart front to back - so a standing pose has to
   be solved for, not assumed.

2. **Can 12 servos at 1.96 N-m hold that pose?** Each foot carries roughly a
   quarter of 2.378 kg. Push that force back through the leg's Jacobian and you
   get the torque every joint must produce just to stand still, before anything
   moves. If that exceeds the servo, no amount of training helps, and this is a
   calculation rather than a six-hour run.

The answer is a trade: stand tall and the legs are nearly straight, so the levers
are short and the torque is low - but there is no room left to absorb a stumble.
Crouch and you gain that room at the cost of torque. The sweep prints both.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MJCF = ROOT / "sim" / "models" / "gray.xml"
OUT = ROOT / "progress" / "stance"
LEGS = ("fr", "fl", "br", "bl")


def parse(name: str):
    n = (name or "").lower()
    seg = next((s for s in ("thigh", "calf", "hip") if s in n), None)
    rest = n.replace(seg, "") if seg else n
    leg = next((l for l in LEGS if l in rest), None)
    return leg, seg


class Robot:
    """The model, plus the few things MuJoCo does not hand you directly."""

    def __init__(self) -> None:
        if not MJCF.exists():
            raise SystemExit(f"no model at {MJCF}. Run tools/make_mjcf.py first.")
        self.m = mujoco.MjModel.from_xml_path(str(MJCF))
        self.d = mujoco.MjData(self.m)

        self.joints: dict[tuple[str, str], int] = {}
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            leg, seg = parse(mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j))
            if leg and seg:
                self.joints[(leg, seg)] = j

        self.calf = {}
        self.hip_body = {}
        for b in range(self.m.nbody):
            leg, seg = parse(mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b))
            if seg == "calf" and leg:
                self.calf[leg] = b
            if seg == "hip" and leg:
                self.hip_body[leg] = b

        self.foot_local = {leg: self._foot_point(b) for leg, b in self.calf.items()}
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.mass = float(sum(self.m.body_mass))

    def _foot_point(self, body: int) -> np.ndarray:
        """The point on the calf that touches the ground: its farthest vertex from
        the knee. Taken off the actual mesh rather than guessed at."""
        best, best_d = np.zeros(3), -1.0
        for g in range(self.m.body_geomadr[body],
                       self.m.body_geomadr[body] + self.m.body_geomnum[body]):
            if self.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = self.m.geom_dataid[g]
            adr, num = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
            verts = self.m.mesh_vert[adr:adr + num].astype(float)
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, self.m.geom_quat[g])
            pts = verts @ rot.reshape(3, 3).T + self.m.geom_pos[g]
            dist = np.linalg.norm(pts, axis=1)
            i = int(np.argmax(dist))
            if dist[i] > best_d:
                best_d, best = float(dist[i]), pts[i]
        return best

    # -- kinematics -------------------------------------------------------

    def set_pose(self, height: float, angles: dict[tuple[str, str], float]) -> None:
        self.d.qpos[:] = 0
        self.d.qpos[0:3] = (0.0, 0.0, height)
        self.d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for key, q in angles.items():
            self.d.qpos[self.m.jnt_qposadr[self.joints[key]]] = q
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)

    def foot_world(self, leg: str) -> np.ndarray:
        b = self.calf[leg]
        return self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ self.foot_local[leg]

    def hip_world(self, leg: str) -> np.ndarray:
        return self.d.xpos[self.hip_body[leg]].copy()

    def limits(self, leg: str, seg: str) -> tuple[float, float]:
        return tuple(self.m.jnt_range[self.joints[(leg, seg)]])


def reference_pose(rb: Robot, tries: int = 300, seed: int = 0) -> dict[tuple[str, str], float]:
    """The pose where every leg hangs straight down, fully extended.

    This is the zero the owner's travel limits were measured from - "legs hanging
    straight". The CAD was exported in a different pose entirely, so the two zeros
    do not line up, and applying the limits around the exported zero puts the
    reachable window nowhere near the ground.

    Solved per leg with many random starts. A single local solve lands in a wrong
    basin and reports that the leg cannot point downwards at all, which is untrue.
    """
    rng = np.random.default_rng(seed)
    park = 0.5  # trunk held high so the legs hang free
    out: dict[tuple[str, str], float] = {}

    for leg in LEGS:
        keys = [(leg, s) for s in ("hip", "thigh", "calf")]

        def residual(x, keys=keys, leg=leg):
            rb.set_pose(park, dict(zip(keys, x)))
            hip, foot = rb.hip_world(leg), rb.foot_world(leg)
            return np.array([20 * (foot[0] - hip[0]), 20 * (foot[1] - hip[1]),
                             (hip[2] - foot[2]) - 0.6])

        best = None
        for _ in range(tries):
            try:
                s = least_squares(residual, rng.uniform(-np.pi, np.pi, 3),
                                  xtol=1e-13, ftol=1e-13, max_nfev=400)
            except Exception:  # noqa: BLE001
                continue
            rb.set_pose(park, dict(zip(keys, s.x)))
            v = rb.foot_world(leg) - rb.hip_world(leg)
            if np.hypot(v[0], v[1]) > 0.004:
                continue
            if best is None or -v[2] > best[0]:
                best = (-v[2], s.x.copy())
        if best is None:
            raise SystemExit(f"the {leg} leg cannot be pointed straight down at all")
        for k, v in zip(keys, best[1]):
            out[k] = float(np.arctan2(np.sin(v), np.cos(v)))  # wrap to (-pi, pi]
    return out


def joint_signs(rb: Robot) -> dict[tuple[str, str], float]:
    """Which way is physically positive, per joint. Measured, not assumed."""
    signs = {}
    for (leg, seg), jid in rb.joints.items():
        rb.set_pose(0.5, {})
        p0 = rb.d.xipos[rb.calf[leg]].copy()
        rb.d.qpos[rb.m.jnt_qposadr[jid]] += np.deg2rad(10)
        mujoco.mj_forward(rb.m, rb.d)
        moved = rb.d.xipos[rb.calf[leg]] - p0
        outward = 1.0 if leg in ("fl", "bl") else -1.0
        want = {"hip": np.array([0.0, outward, 0.0]),
                "thigh": np.array([1.0, 0.0, 0.0]),
                "calf": np.array([0.0, 0.0, 1.0])}[seg]
        signs[(leg, seg)] = 1.0 if float(moved @ want) >= 0 else -1.0
    return signs


def travel(rb: Robot, ref, signs, owner) -> dict[tuple[str, str], tuple[float, float]]:
    """The owner's measured travel, placed around the reference pose."""
    out = {}
    for key, zero in ref.items():
        lo_d, hi_d = owner[key[1]]
        a = zero + signs[key] * np.deg2rad(lo_d)
        b = zero + signs[key] * np.deg2rad(hi_d)
        out[key] = (min(a, b), max(a, b))
    return out


def solve(rb: Robot, height: float, outboard: float = 0.0):
    """Angles that put each foot on the floor, under its own hip."""
    # The model's own limits. They are the owner's, measured in the pose editor
    # against this geometry, and written in by tools/prepare_model.py - so there
    # is no second copy here to drift out of date.
    keys = [(leg, seg) for leg in LEGS for seg in ("hip", "thigh", "calf")]
    lo = np.array([rb.limits(*k)[0] for k in keys])
    hi = np.array([rb.limits(*k)[1] for k in keys])
    x0 = np.clip(np.zeros(len(keys)), lo + 1e-4, hi - 1e-4)

    def residual(x):
        rb.set_pose(height, dict(zip(keys, x)))
        out = []
        for leg in LEGS:
            hip, foot = rb.hip_world(leg), rb.foot_world(leg)
            # Plant each foot outboard of its own hip, on whichever side that hip
            # is. Under the hip puts the feet inside the body's own width.
            side = 1.0 if hip[1] >= 0 else -1.0
            target = np.array([hip[0], hip[1] + side * outboard, 0.0])
            out.extend(foot - target)
        return np.array(out)

    sol = least_squares(residual, x0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, max_nfev=4000)
    rb.set_pose(height, dict(zip(keys, sol.x)))
    return dict(zip(keys, sol.x)), float(np.abs(residual(sol.x)).max())


def holding_torque(rb: Robot) -> dict[tuple[str, str], float]:
    """Torque each joint must produce to hold the pose, static.

    Every foot pushes up with a quarter of the robot's weight. Carried back
    through that leg's Jacobian, tau = J^T F, that is what the servo has to hold.
    """
    share = rb.mass * 9.81 / 4.0
    force = np.array([0.0, 0.0, share])
    out = {}
    for leg in LEGS:
        jacp = np.zeros((3, rb.m.nv))
        mujoco.mj_jac(rb.m, rb.d, jacp, None, rb.foot_world(leg), rb.calf[leg])
        tau = jacp.T @ force
        for seg in ("hip", "thigh", "calf"):
            dof = rb.m.jnt_dofadr[rb.joints[(leg, seg)]]
            out[(leg, seg)] = float(abs(tau[dof]))
    return out


def render(rb: Robot, path: Path, seconds: float = 4.0) -> None:
    import imageio.v2 as imageio  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    r = mujoco.Renderer(rb.m, height=720, width=1280)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    # Frame the whole robot, from the floor to the top of the trunk.
    top = float(rb.d.xpos[rb.base][2]) + 0.06
    cam.distance, cam.elevation = 1.05, -10
    cam.lookat[:] = (0, 0, top / 2)

    qpos = rb.d.qpos.copy()
    frames = []
    for i in range(int(seconds * 30)):
        rb.d.qpos[:] = qpos          # hold the pose; this is a look, not a sim
        rb.d.qvel[:] = 0
        mujoco.mj_forward(rb.m, rb.d)
        cam.azimuth = 120 + 360 * i / (seconds * 30)
        r.update_scene(rb.d, cam)
        frames.append(np.asarray(r.render()))
    imageio.mimwrite(path, frames, fps=30, codec="libx264", quality=8, macro_block_size=1)
    imageio.imwrite(path.with_suffix(".png"), frames[len(frames) // 3])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--height", type=float, help="solve one trunk height, in metres")
    ap.add_argument("--render", action="store_true", help="film the chosen pose")
    args = ap.parse_args()

    rb = Robot()
    cfg = yaml.safe_load((ROOT / "gray" / "config" / "robot.yaml").read_text())
    servo, owner = cfg["servo"], {k: cfg["joint_limits"][k] for k in ("hip", "thigh", "calf")}
    cap = servo["stall_torque_nm"]

    print(f"model    {rb.mass*1000:.1f} g, {len(rb.joints)} hinges")
    print(f"servo    {cap} N-m stall, {servo['no_load_speed_rad_s']} rad/s")

    stance = cfg.get("stance", {})
    outboard = float(stance.get("foot_outboard_mm", 0.0)) / 1000.0
    print(f"stance   feet planted {outboard*1000:.0f} mm outboard of each hip")

    # How tall it can stand: climb until the legs run out of travel.
    tall = 0.0
    for h in np.arange(0.06, 0.40, 0.002):
        if solve(rb, float(h), outboard)[1] <= 0.004:
            tall = float(h)
    ride = tall * (1 - cfg["stance"]["ride_drop"]) if "stance" in cfg else tall * 0.6
    print(f"\ntallest it can stand   {tall*1000:.1f} mm")
    print(f"ride height, 40% down  {ride*1000:.1f} mm\n")

    heights = ([args.height] if args.height
               else sorted({round(tall * f, 4) for f in np.arange(0.5, 1.001, 0.05)}
                           | {round(ride, 4)}))
    print(f"{'height':>8} {'of max':>7}  {'reach err':>9}  {'worst joint':>12}  "
          f"{'torque':>7}  {'margin':>7}  verdict")
    print("-" * 76)

    best = None
    for h in heights:
        angles, err = solve(rb, h, outboard)
        frac = h / tall
        if err > 0.004:
            print(f"{h*1000:7.0f}mm {frac:6.0%}  {err*1000:8.1f}mm   {'-':>12}  "
                  f"{'-':>7}  {'-':>7}  out of joint travel")
            continue
        tau = holding_torque(rb)
        (leg, seg), worst = max(tau.items(), key=lambda kv: kv[1])
        margin = cap / worst if worst else float("inf")
        ok = margin >= 1.5
        mark = "  <- ride height" if abs(h - ride) < 1e-6 else ""
        print(f"{h*1000:7.0f}mm {frac:6.0%}  {err*1000:8.1f}mm   {leg + '_' + seg:>12}  "
              f"{worst:6.2f}   {margin:6.2f}x  {'holds' if ok else 'TOO WEAK'}{mark}")
        if abs(h - ride) < 1e-6 and ok:
            best = (h, angles, tau, margin)

    if best is None:
        print("\nNo height works. Either the mass is wrong or the servos are too weak.")
        return 1

    h, angles, tau, margin = best
    print(f"\nchosen   trunk at {h*1000:.0f} mm, worst joint at {margin:.2f}x the servo")
    print("\njoint angles, degrees:")
    print(f"  {'leg':>4} {'hip':>8} {'thigh':>8} {'calf':>8}   "
          f"{'| torque, N-m':>14} {'hip':>7} {'thigh':>7} {'calf':>7}")
    for leg in LEGS:
        a = [np.rad2deg(angles[(leg, s)]) for s in ("hip", "thigh", "calf")]
        t = [tau[(leg, s)] for s in ("hip", "thigh", "calf")]
        print(f"  {leg:>4} {a[0]:8.1f} {a[1]:8.1f} {a[2]:8.1f}   {'|':>14} "
              f"{t[0]:7.2f} {t[1]:7.2f} {t[2]:7.2f}")

    solve(rb, h, outboard)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stance.yaml").write_text(yaml.safe_dump({
        "trunk_height_m": round(float(h), 4),
        "trunk_height_pct_of_max": round(100 * float(h) / float(tall), 1),
        "max_standing_height_m": round(float(tall), 4),
        "worst_margin_x": round(float(margin), 3),
        "angles_deg": {f"{leg}_{seg}": round(float(np.rad2deg(v)), 3)
                       for (leg, seg), v in angles.items()},
        "holding_torque_nm": {f"{leg}_{seg}": round(float(v), 4) for (leg, seg), v in tau.items()},
        "servo_stall_nm": float(cap),
        "zero": "the sitting pose, as exported from SolidWorks",
    }, sort_keys=False))
    print(f"\nwrote {(OUT / 'stance.yaml').relative_to(ROOT)}")

    if args.render:
        out = OUT / "stance.mp4"
        render(rb, out)
        print(f"wrote {out.relative_to(ROOT)} and {out.with_suffix('.png').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
