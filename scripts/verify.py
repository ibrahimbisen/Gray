"""Score a trained policy against its stage's pass bar.

    python scripts/verify.py Gray-Stand
    python scripts/verify.py Gray-Push --seconds 30 --robots 128

Training does not answer this on its own. The reward is a weighted sum, which
can read excellently while one term quietly fails, and an episode is shorter
than the bar - so a policy that drifts after twenty seconds still shows a clean
training curve. This measures the things each bar actually names, over the full
duration, across many robots at once.

A stage is passed because the number in the bar was met, not because the curve
went up.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

LOG_ROOT = ROOT / "logs" / "rsl_rl"

# Which log folder each task writes to, and what its bar actually asks for.
TASKS = {
    "Gray-Stand": {"experiment": "gray_stand", "stage": 1, "seconds": 30.0,
                   "bar_survive": 1.00, "bar_err_mm": 5.0, "bar_upright": 0.99},
    # Being shoved every two to four seconds by an unknown amount, on unknown
    # ground, is not something to expect a perfect score against. The bar is nine
    # in ten, and the height tolerance is wider because the robot is allowed to
    # be knocked off its height as long as it comes back.
    "Gray-Push": {"experiment": "gray_push", "stage": 2, "seconds": 20.0,
                  "bar_survive": 0.90, "bar_err_mm": 20.0, "bar_upright": 0.95},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", nargs="?", default="Gray-Stand", choices=sorted(TASKS))
    ap.add_argument("--run", help="run folder; default is the newest")
    ap.add_argument("--checkpoint", help="e.g. model_599.pt; default is the last")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 uses the bar's own")
    ap.add_argument("--robots", type=int, default=64)
    args = ap.parse_args()

    spec = TASKS[args.task]
    seconds = args.seconds or spec["seconds"]
    exp_root = LOG_ROOT / spec["experiment"]
    if not exp_root.is_dir():
        raise SystemExit(f"no runs under {exp_root}")
    log_dir = (exp_root / args.run if args.run
               else max(exp_root.iterdir(), key=lambda p: p.stat().st_mtime))
    ckpts = sorted(log_dir.glob("model_*.pt"),
                   key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {log_dir}")
    ckpt = (log_dir / args.checkpoint) if args.checkpoint else ckpts[-1]

    import torch  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    import gray.tasks  # noqa: F401,PLC0415
    from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: PLC0415

    target = float(yaml.safe_load(
        (ROOT / "progress" / "stance" / "stance.yaml").read_text())["trunk_height_m"])

    # Play mode turns off observation noise, but the disturbances stay: a policy
    # that only survives when nothing pushes it has not passed this stage.
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.robots
    env_cfg.episode_length_s = seconds + 5.0
    agent_cfg = load_rl_cfg(args.task)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device="cuda:0"),
                             clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device="cuda:0")
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")

    robot = env.unwrapped.scene["robot"]
    origins = env.unwrapped.scene.env_origins

    obs, _ = env.reset()
    heights, uprights = [], []
    fell = torch.zeros(args.robots, dtype=torch.bool, device="cuda:0")
    with torch.inference_mode():
        for _ in range(int(seconds * 50)):
            obs = env.step(policy(obs))[0]
            h = robot.data.root_link_pos_w[:, 2] - origins[:, 2]
            up = -robot.data.projected_gravity_b[:, 2]   # 1.0 is dead level
            heights.append(h.clone())
            uprights.append(up.clone())
            fell |= (up < 0.7) | (h < target * 0.55)

    h = torch.stack(heights)
    up = torch.stack(uprights)
    # A robot that fell drags every later sample down with it, so steadiness is
    # measured over the ones still standing. Whether they fell is its own check
    # and is not being softened here.
    alive = ~fell
    survived = float(alive.float().mean())
    err_mm = ((h[:, alive] - target).abs() * 1000) if bool(alive.any()) else h.abs() * 1e6
    up_alive = up[:, alive] if bool(alive.any()) else up

    print(f"task        {args.task}")
    print(f"checkpoint  {ckpt.relative_to(ROOT)}")
    print(f"tested      {args.robots} robots, {seconds:.0f} s each, "
          f"target trunk height {target*1000:.1f} mm\n")

    checks = [
        (f"stayed up for {seconds:.0f} s", survived, spec["bar_survive"], "ge",
         f"{int(fell.sum())} of {args.robots} fell", "{:.0%}"),
        (f"trunk within {spec['bar_err_mm']:.0f} mm of target",
         float(err_mm.mean()), spec["bar_err_mm"], "le",
         f"worst {float(err_mm.max()):.1f} mm", "{:.2f} mm"),
        (f"uprightness above {spec['bar_upright']}",
         float(up_alive.mean()), spec["bar_upright"], "ge",
         f"worst {float(up_alive.min()):.4f}", "{:.4f}"),
    ]

    passed = True
    print(f"{'check':<32} {'measured':>12}  {'bar':>7}")
    print("-" * 78)
    for name, got, bar, how, note, fmt in checks:
        ok = got >= bar if how == "ge" else got <= bar
        passed &= ok
        print(f"{name:<32} {fmt.format(got):>12}  {bar:>7}   "
              f"{'PASS' if ok else 'FAIL'}  {note}")

    print()
    print(f"STAGE {spec['stage']} {'PASSED' if passed else 'NOT PASSED'}")
    env.close()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
