"""Film every saved checkpoint, so you can watch the robot learn.

    python scripts/film_checkpoints.py                  # every checkpoint of the newest run
    python scripts/film_checkpoints.py --every 50       # thin them out
    python scripts/film_checkpoints.py --run 2026-08-02_22-56-35_stand_v1

Training saves a checkpoint every 25 iterations regardless, so the films can be
made afterwards rather than during. That is better than filming as it trains:
the recorder inside mjlab counts calls to env.step() rather than robot-steps,
which is easy to get wrong by a factor of num_envs, and rendering mid-run
competes with training for the same card.

Each film is one robot holding its stance for a few seconds, written straight
into progress/runs/<run>/videos/ where the dashboard picks it up.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

LOG_ROOT = ROOT / "logs" / "rsl_rl" / "gray_stand"
RUNS = ROOT / "progress" / "runs"


def iteration_of(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", help="run folder name; default is the newest")
    ap.add_argument("--every", type=int, default=1, help="film 1 in N checkpoints")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--envs", type=int, default=1)
    ap.add_argument("--watch", action="store_true",
                    help="keep running alongside training, filming each new checkpoint")
    args = ap.parse_args()

    if not LOG_ROOT.is_dir():
        raise SystemExit(f"no training logs under {LOG_ROOT}")
    log_dir = (LOG_ROOT / args.run if args.run
               else max(LOG_ROOT.iterdir(), key=lambda p: p.stat().st_mtime))
    run_dir = RUNS / log_dir.name
    if not run_dir.is_dir():
        raise SystemExit(f"no dashboard run at {run_dir}")

    def pending() -> list[Path]:
        found = sorted(log_dir.glob("model_*.pt"), key=iteration_of)
        found = [c for i, c in enumerate(found) if i % args.every == 0]
        return [c for c in found
                if not (run_dir / "videos" / f"iter_{iteration_of(c):04d}.mp4").exists()]

    if not args.watch and not pending():
        raise SystemExit(f"nothing new to film in {log_dir}")

    import imageio.v2 as imageio  # noqa: PLC0415
    import mujoco  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    import gray.tasks  # noqa: F401,PLC0415
    from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: PLC0415

    env_cfg = load_env_cfg("Gray-Stand", play=True)
    env_cfg.scene.num_envs = args.envs
    agent_cfg = load_rl_cfg("Gray-Stand")
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device="cuda:0"),
                             clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device="cuda:0")

    model = env.unwrapped.sim.mj_model
    renderer = mujoco.Renderer(model, height=480, width=854)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 1.0, -12.0, 125.0

    data = mujoco.MjData(model)
    steps = int(args.seconds * 50)
    every = 1  # the env already runs at 50 Hz; film every control step

    out_dir = run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run      {log_dir.name}")
    print(f"mode     {'watching for new checkpoints' if args.watch else 'one pass'}")
    print(f"output   {out_dir.relative_to(ROOT)}\n", flush=True)

    def film(ckpt: Path) -> None:
        it = iteration_of(ckpt)
        dst = out_dir / f"iter_{it:04d}.mp4"
        runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")

        obs, _ = env.reset()
        frames = []
        with torch.inference_mode():
            for i in range(steps):
                obs = env.step(policy(obs))[0]
                if i % every:
                    continue
                # Copy the first robot's state into a plain MjData to render it.
                # The simulator's state lives on the GPU, so it has to come back
                # to the host first.
                sim_data = env.unwrapped.sim.data
                data.qpos[:] = sim_data.qpos[0].detach().cpu().numpy()
                data.qvel[:] = sim_data.qvel[0].detach().cpu().numpy()
                mujoco.mj_forward(model, data)
                cam.lookat[:] = data.xpos[1]
                renderer.update_scene(data, cam)
                frames.append(np.asarray(renderer.render()))
        # Write beside the target then move, so the dashboard never serves a
        # half-written file to a browser that is polling every ten seconds.
        # Keep the .mp4 on the end: imageio picks its writer from the extension
        # and cannot open a file called .part at all.
        tmp = dst.with_name(dst.stem + ".writing.mp4")
        imageio.mimwrite(tmp, frames, fps=50, codec="libx264", quality=7,
                         macro_block_size=1)
        tmp.replace(dst)
        print(f"  iteration {it:>4}  ->  {dst.name}  ({len(frames)} frames)", flush=True)

    try:
        while True:
            todo = pending()
            for ckpt in todo:
                try:
                    film(ckpt)
                except Exception as exc:  # noqa: BLE001
                    # A checkpoint being written as we read it is normal when
                    # this runs alongside training; try it again next pass.
                    print(f"  iteration {iteration_of(ckpt):>4}  skipped: "
                          f"{type(exc).__name__}: {str(exc)[:70]}", flush=True)
            if not args.watch:
                break
            if not todo:
                print("  (waiting for the next checkpoint)", flush=True)
            time.sleep(30.0)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        env.close()

    print("\nReload the dashboard to see them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
