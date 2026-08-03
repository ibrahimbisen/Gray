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


def trained_under(log_dir: Path) -> dict:
    """What the run was actually trained with, as mjlab recorded it at launch.

    The dump carries python tags for functions, tuples and enums. Reconstructing
    those objects is neither possible nor wanted - unsafe_load dies on the first
    enum it cannot import, and all that is needed here is the numbers. So unknown
    tags are read as the plain structure underneath them.
    """
    import yaml  # noqa: PLC0415

    path = log_dir / "params" / "env.yaml"
    if not path.exists():
        return {}

    class Tolerant(yaml.SafeLoader):
        pass

    def plain(loader, _suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)

    Tolerant.add_multi_constructor("", plain)
    try:
        return yaml.load(path.read_text(), Loader=Tolerant) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[drift] could not read what this run trained with: {exc}")
        return {}


def report_drift(log_dir: Path, cfg) -> None:
    """Say so when the task has changed since the run was trained.

    Scoring a policy against a task that has moved under it is not a verdict on
    the policy - it silently answers a different question. This caught exactly
    that: a run trained on 0.6 m/s box-shaped shoves was scored against 1.2 m/s
    ones from any angle, and 'failed' a bar it had never been trained for.
    """
    was = trained_under(log_dir)
    if not was:
        return
    drift = []

    def numbers(x):
        """Strip a params tree down to comparable plain numbers."""
        if isinstance(x, dict):
            return {k: numbers(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [numbers(v) for v in x]
        return round(x, 6) if isinstance(x, (int, float)) else str(x)

    old_shove = (was.get("events") or {}).get("shove") or {}
    new_shove = cfg.events.get("shove")
    if old_shove and new_shove is not None:
        old_p, new_p = numbers(old_shove.get("params", {})), numbers(new_shove.params)
        if old_p != new_p:
            drift.append(f"  shove    trained {old_p}")
            drift.append(f"           testing {new_p}")

    old_rew = was.get("rewards") or {}
    for name, term in cfg.rewards.items():
        before = (old_rew.get(name) or {}).get("weight")
        if isinstance(before, (int, float)) and abs(before - term.weight) > 1e-9:
            drift.append(f"  {name:<9}trained {before}   testing {term.weight}")
    for name in old_rew:
        if name not in cfg.rewards:
            drift.append(f"  {name:<9}was a scoring term then, and is not now")

    if drift:
        print("WARNING - the task has changed since this run was trained:")
        print("\n".join(drift))
        print("  The result below answers how this policy copes with today's task,")
        print("  not whether it passed the one it was trained for.\n")


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
    report_drift(log_dir, env_cfg)

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
