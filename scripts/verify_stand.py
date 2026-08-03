"""Score a trained policy against stage 1's pass bar.

    python scripts/verify_stand.py
    python scripts/verify_stand.py --seconds 60 --robots 64

The bar is:

    30 s without falling. Trunk within 5 mm of 164 mm. Uprightness above 0.99.

Training does not answer this on its own. Episodes are 10 s long, so a policy
that drifts after twenty would still show a clean training curve - and the
reward is a weighted sum, which can look excellent while one term quietly fails.
This measures the three things the bar actually names, over the full duration,
across many robots at once.

A stage is not passed because the curve went up. It is passed because the number
in the bar was met.
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

LOG_ROOT = ROOT / "logs" / "rsl_rl" / "gray_stand"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", help="run folder; default is the newest")
    ap.add_argument("--checkpoint", help="e.g. model_599.pt; default is the last")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--robots", type=int, default=64)
    args = ap.parse_args()

    log_dir = (LOG_ROOT / args.run if args.run
               else max(LOG_ROOT.iterdir(), key=lambda p: p.stat().st_mtime))
    ckpts = sorted(log_dir.glob("model_*.pt"),
                   key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    ckpt = (log_dir / args.checkpoint) if args.checkpoint else ckpts[-1]

    import torch  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    import gray.tasks  # noqa: F401,PLC0415
    from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: PLC0415

    target = float(yaml.safe_load(
        (ROOT / "progress" / "stance" / "stance.yaml").read_text())["trunk_height_m"])

    env_cfg = load_env_cfg("Gray-Stand", play=True)
    env_cfg.scene.num_envs = args.robots
    agent_cfg = load_rl_cfg("Gray-Stand")
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device="cuda:0"),
                             clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device="cuda:0")
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")

    robot = env.unwrapped.scene["robot"]
    origins = env.unwrapped.scene.env_origins
    steps = int(args.seconds * 50)

    obs, _ = env.reset()
    heights, uprights = [], []
    fell = torch.zeros(args.robots, dtype=torch.bool, device="cuda:0")
    with torch.inference_mode():
        for _ in range(steps):
            obs = env.step(policy(obs))[0]
            h = robot.data.root_link_pos_w[:, 2] - origins[:, 2]
            # projected gravity's z component is -1 when perfectly level.
            up = -robot.data.projected_gravity_b[:, 2]
            heights.append(h.clone())
            uprights.append(up.clone())
            fell |= (up < 0.7) | (h < target * 0.55)

    h = torch.stack(heights)          # (steps, robots)
    up = torch.stack(uprights)
    err_mm = ((h - target).abs() * 1000)

    print(f"checkpoint  {ckpt.relative_to(ROOT)}")
    print(f"tested      {args.robots} robots, {args.seconds:.0f} s each, "
          f"target trunk height {target*1000:.1f} mm\n")

    checks = [
        (f"stayed up for {args.seconds:.0f} s",
         int((~fell).sum()), args.robots, f"{int(fell.sum())} fell"),
        ("trunk within 5 mm of target",
         float(err_mm.mean()), 5.0, f"worst {float(err_mm.max()):.2f} mm"),
        ("uprightness above 0.99",
         float(up.mean()), 0.99, f"worst {float(up.min()):.4f}"),
    ]

    passed = True
    print(f"{'check':<30} {'measured':>12}  {'bar':>8}   ")
    print("-" * 72)
    for name, got, bar, note in checks:
        if "within" in name:
            ok = got <= bar
            shown = f"{got:.2f} mm"
        elif "stayed up" in name:
            ok = got == bar
            shown = f"{int(got)}/{int(bar)}"
        else:
            ok = got >= bar
            shown = f"{got:.4f}"
        passed &= ok
        print(f"{name:<30} {shown:>12}  {bar:>8}   {'PASS' if ok else 'FAIL'}  {note}")

    print()
    print("STAGE 1 PASSED" if passed else "STAGE 1 NOT PASSED")
    env.close()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
