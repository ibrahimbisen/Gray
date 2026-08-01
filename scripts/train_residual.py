#!/usr/bin/env python3
"""Train the Phase 3 residual policy on top of Gray's classical gait.

    python scripts/train_residual.py Gray-Residual-Flat
    python scripts/train_residual.py Gray-Residual-Flat --env.scene.num-envs 1024
    python scripts/train_residual.py Gray-Residual-Flat --agent.max-iterations 50

This is a thin shim around mjlab's own trainer. All it adds is importing train.tasks,
which registers Gray with mjlab's task registry; everything after that - argument
parsing, logging, checkpointing - is mjlab's. Run `play_residual.py` to watch a
checkpoint, and `scripts/eval_policy.py` to score one in plain MuJoCo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train.tasks  # noqa: F401,E402  (import registers Gray-Residual-Flat)
from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  main()
