"""Write one fake training run, so the dashboard is built against real files.

    python scripts/seed_demo_run.py          # create it
    python scripts/seed_demo_run.py --clean  # remove it

There is no training yet, and a dashboard built against imagined data is how the
last one reached 4,500 lines of HTML. This writes the exact layout described in
dashboard/runs.py - including a real video rendered from the real model - so the
page can be checked before a single line of training code exists.

The run is clearly labelled a demo. Delete it the moment a real run exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RUN_ID = "0000-00-00_demo"
FOLDER = ROOT / "progress" / "runs" / RUN_ID
ITERATIONS = 300
FILM_EVERY = 60


def write_meta() -> None:
    started = datetime.now() - timedelta(minutes=42)
    (FOLDER / "run.json").write_text(json.dumps({
        "name": "DEMO - not a real run",
        "purpose": "Placeholder so the dashboard can be built and checked before "
                   "any training exists. Delete it once a real run appears.",
        "stage": 1,
        "stage_name": "Stand still",
        "task": "Gray-Stand",
        "status": "running",
        "bar": "30 s without falling. Trunk height within 5 mm of target. "
               "Uprightness above 0.99.",
        "started": started.isoformat(timespec="seconds"),
        "finished": None,
        "iterations_target": ITERATIONS,
        "notes": "Numbers below are generated, not measured.",
    }, indent=2))


def write_metrics() -> None:
    """A plausible learning curve. Shaped, not random, so a broken chart is obvious."""
    rows = []
    for i in range(0, 211, 5):
        t = i / ITERATIONS
        rows.append({
            "iteration": i,
            "reward": round(-2.0 + 6.5 * (1 - math.exp(-4 * t)), 4),
            "episode_length": round(40 + 460 * (1 - math.exp(-3.2 * t)), 1),
            "uprightness": round(0.55 + 0.44 * (1 - math.exp(-5 * t)), 4),
            "trunk_height_mm": round(70 + 145 * (1 - math.exp(-3.6 * t)), 1),
            "falls_per_100": round(88 * math.exp(-4.5 * t) + 1.5, 2),
        })
    with (FOLDER / "metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_videos() -> list[str]:
    """Film the real model, so the video panel is showing Gray and not a placeholder."""
    made = []
    try:
        import imageio.v2 as imageio  # noqa: PLC0415
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        print(f"  no video: {exc}. The dashboard will show its empty state.")
        return made

    mjcf = ROOT / "sim" / "models" / "gray.xml"
    if not mjcf.exists():
        print("  no video: run tools/make_mjcf.py first")
        return made

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    renderer = mujoco.Renderer(model, height=360, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.azimuth, cam.elevation, cam.lookat[2] = 1.4, 135, -15, 0.1

    for it in range(0, ITERATIONS, FILM_EVERY):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        lowest = min((float(data.geom_xpos[g][2]) for g in range(model.ngeom)
                      if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE), default=0.0)
        data.qpos[2] += 0.03 - lowest
        frames = []
        for step in range(int(2.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            if step % max(1, int(1 / (30 * model.opt.timestep))) == 0:
                renderer.update_scene(data, cam)
                frames.append(renderer.render())
        out = FOLDER / "videos" / f"iter_{it:04d}.mp4"
        imageio.mimwrite(out, [np.asarray(f) for f in frames], fps=30,
                         codec="libx264", quality=7, macro_block_size=1)
        made.append(out.name)
        print(f"  filmed {out.name}  ({len(frames)} frames)")
    return made


def write_checkpoints() -> None:
    """Empty stand-ins. The dashboard only counts these; it never opens them."""
    for it in range(0, ITERATIONS, FILM_EVERY):
        (FOLDER / "checkpoints" / f"iter_{it:04d}.npz").write_bytes(b"")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean", action="store_true", help="delete the demo run")
    args = ap.parse_args()

    if args.clean:
        shutil.rmtree(FOLDER, ignore_errors=True)
        print(f"removed {FOLDER.relative_to(ROOT)}")
        return 0

    for sub in ("videos", "checkpoints"):
        (FOLDER / sub).mkdir(parents=True, exist_ok=True)
    write_meta()
    write_metrics()
    write_checkpoints()
    videos = write_videos()

    print(f"wrote {FOLDER.relative_to(ROOT)}")
    print(f"  run.json, metrics.csv, {len(videos)} video(s), "
          f"{ITERATIONS // FILM_EVERY} checkpoint stubs")
    print("\nRemove it with:  python scripts/seed_demo_run.py --clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
