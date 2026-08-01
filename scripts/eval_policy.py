#!/usr/bin/env python3
"""Score one trained policy against the Phase 2 gait, in plain MuJoCo.

    python scripts/eval_policy.py                          # newest checkpoint
    python scripts/eval_policy.py --checkpoint logs/.../model_1500.pt
    python scripts/eval_policy.py --video out.mp4

This is the dual-sim check docs/PROJECT_NOTES.md requires before any hardware
rollout: a policy re-scored in a second engine, against the baseline it has to beat.
See train/evaluate.py for why the second engine is worth the trouble.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.evaluate import (  # noqa: E402
    find_checkpoints,
    format_row,
    load_policy,
    rollout,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", help="model_<n>.pt (default: newest found)")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--speed", type=float, default=0.0529,
                    help="commanded forward velocity, m/s")
    ap.add_argument("--seed", type=int, default=0,
                    help="same seed -> same walk, exactly, forever")
    ap.add_argument("--video", metavar="MP4")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the Phase 2 comparison run")
    args = ap.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        found = find_checkpoints()
        if not found:
            print("no checkpoints under logs/rsl_rl/gray_residual/ - has it trained?")
            raise SystemExit(1)
        checkpoint = found[-1]

    common = dict(speed_ms=args.speed, duration=args.duration, seed=args.seed)

    print(f"checkpoint  {checkpoint}")
    print(f"commanded   {args.speed*1000:.1f} mm/s   seed {args.seed}   "
          f"{args.duration:.0f} s\n")

    policy = load_policy(checkpoint)
    trained = rollout(policy, video=args.video, **common)

    if not args.no_baseline:
        print(format_row("Phase 2", rollout(None, **common)))
    print(format_row(f"iter {policy.iteration}", trained))

    if trained.get("video"):
        print(f"\nwrote {trained['video']}")


if __name__ == "__main__":
    main()
