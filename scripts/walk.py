#!/usr/bin/env python3
"""Run Gray's classical gait - in simulation now, on the real robot in Phase 5.

    python scripts/walk.py --gait crawl --duration 8
    python scripts/walk.py --gait trot --speed 1.0 --render out.png
    python scripts/walk.py --stand-only
    python scripts/walk.py --view              # live 3D window, orbit with the mouse

The controller here is entirely deterministic: gray.gait decides where each foot
belongs, gray.kinematics converts that to joint angles, and those go straight to the
position servos. Both modules are shared with the hardware path, so what is tuned
here is what will run on the robot.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from gray.gait import GAIT_PATTERNS, GaitGenerator, GaitParams  # noqa: E402
from gray.kinematics import LEGS, joint_vector, load_legs        # noqa: E402

MJCF = "sim/models/gray.xml"
SETTLE_S = 0.5   # hold the stance before walking, so we measure gait not drop-in


def run(args) -> dict:
    import mujoco

    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)

    legs = load_legs()
    params = GaitParams(
        pattern=args.gait, stance_height=args.height, step_length=args.step_length,
        step_height=args.step_height, period=args.period, width_scale=args.width,
    )
    gait = GaitGenerator(legs, params)

    # Actuator order follows the model, which may not match LEGS order.
    act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
           for i in range(model.nu)}
    order = [act[f"{leg}_{seg}"] for leg in LEGS for seg in ("hip", "top", "bottom")]

    # Start standing, at the height the gait expects, and let it settle.
    stand = joint_vector(gait.stand())
    data.qpos[7:] = stand
    data.qpos[2] = args.height + 0.015
    data.ctrl[order] = stand
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    steps_per_tick = max(1, int(round(1.0 / args.control_hz / dt)))
    frames, samples = [], []
    renderer = None
    if args.render:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance, cam.elevation, cam.azimuth = 0.9, -12, 120

    # Live window. Measurement is unaffected: samples are still only taken over
    # (SETTLE_S, total], exactly as headless. Once that window has passed the robot
    # simply keeps walking so it can be watched for as long as the window is open.
    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                              show_right_ui=False)
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.9, -12, 120

    t, fell = 0.0, False
    total = SETTLE_S + args.duration
    wall_start = time.perf_counter()
    try:
        while True:
            if viewer is not None:
                if not viewer.is_running():
                    break
            elif t >= total:
                break

            walk_t = max(0.0, t - SETTLE_S)
            speed = 0.0 if t < SETTLE_S or args.stand_only else args.speed
            try:
                data.ctrl[order] = joint_vector(gait.joint_angles(walk_t, speed))
            except ValueError as exc:
                return {"error": str(exc)}

            for _ in range(steps_per_tick):
                mujoco.mj_step(model, data)
            t += steps_per_tick * dt

            up = _uprightness(mujoco, data)
            measuring = t <= total
            if SETTLE_S < t <= total:
                samples.append((t - SETTLE_S, data.qpos[0], data.qpos[1],
                                data.qpos[2], up))
            if up < 0.2:
                fell = fell or measuring   # a stumble while spectating is not a result
                break
            if renderer and len(frames) < args.frames and \
                    len(samples) % max(1, int(args.control_hz / 8)) == 0:
                cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.08]
                renderer.update_scene(data, cam)
                frames.append(renderer.render())

            if viewer is not None:
                viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.08]
                viewer.sync()
                # Play at true speed rather than as fast as the CPU allows.
                ahead = wall_start + t - time.perf_counter()
                if ahead > 0:
                    time.sleep(ahead)
    finally:
        if viewer is not None:
            viewer.close()

    if not samples:
        return {"error": "no samples"}
    s = np.array(samples)
    return {
        "gait": args.gait, "fell": fell, "elapsed": float(s[-1, 0]),
        "distance": float(s[-1, 1] - s[0, 1]),
        "drift": float(s[-1, 2] - s[0, 2]),
        "speed": float((s[-1, 1] - s[0, 1]) / max(s[-1, 0], 1e-9)),
        "height_mean": float(s[:, 3].mean()), "height_std": float(s[:, 3].std()),
        "upright_min": float(s[:, 4].min()),
        "support_min": min(gait.support_count(x) for x in np.linspace(0, params.period, 200)),
        "frames": frames,
    }


def _uprightness(mujoco, data) -> float:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.qpos[3:7])
    return float(R.reshape(3, 3)[2, 2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gait", default="crawl", choices=sorted(GAIT_PATTERNS))
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--speed", type=float, default=1.0, help="stride scale; <0 reverses")
    ap.add_argument("--height", type=float, default=GaitParams.stance_height)
    ap.add_argument("--step-length", type=float, default=GaitParams.step_length)
    ap.add_argument("--step-height", type=float, default=GaitParams.step_height)
    ap.add_argument("--period", type=float, default=GaitParams.period)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument("--stand-only", action="store_true")
    ap.add_argument("--render", metavar="PNG", help="write a contact-sheet PNG")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--view", action="store_true",
                    help="open a live 3D window; keeps walking until you close it")
    args = ap.parse_args()

    r = run(args)
    if "error" in r:
        print(f"FAILED: {r['error']}")
        raise SystemExit(1)

    print(f"gait            {r['gait']}  ({r['support_min']} feet down at worst)")
    print(f"ran             {r['elapsed']:.2f} s"
          f"{'  -- FELL OVER' if r['fell'] else ''}")
    print(f"distance        {r['distance']*1000:+8.1f} mm forward")
    print(f"lateral drift   {r['drift']*1000:+8.1f} mm")
    print(f"speed           {r['speed']*1000:+8.1f} mm/s")
    print(f"trunk height    {r['height_mean']*1000:8.1f} mm  "
          f"(sd {r['height_std']*1000:.1f})")
    print(f"min uprightness {r['upright_min']:8.3f}")

    if args.render and r["frames"]:
        from PIL import Image
        fr = r["frames"]
        cols = min(4, len(fr))
        rows = (len(fr) + cols - 1) // cols
        h, w = fr[0].shape[:2]
        sheet = Image.new("RGB", (w * cols, h * rows))
        for i, f in enumerate(fr):
            sheet.paste(Image.fromarray(f), ((i % cols) * w, (i // cols) * h))
        sheet.save(args.render)
        print(f"\nwrote {args.render}")


if __name__ == "__main__":
    main()
