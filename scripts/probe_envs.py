"""Find how many robots this GPU can actually simulate at once.

    python scripts/probe_envs.py
    python scripts/probe_envs.py --sizes 2048 4096 6144

Guessing costs a failed training run and the minutes it takes to get there.
MuJoCo-Warp captures the whole physics step into a CUDA graph, and when the card
cannot hold it the failure is "Warp CUDA error 600: device not ready" at the
first reset, which does not look like running out of memory at all.

Each size is tried in its own process, because a CUDA graph failure leaves the
context unusable and anything after it in the same process fails too.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")


def try_size(n: int) -> dict:
    """Build the env, reset it, step it once. Reports VRAM actually used."""
    import torch  # noqa: PLC0415

    import gray.tasks  # noqa: F401,PLC0415
    from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg  # noqa: PLC0415

    free_before, total = torch.cuda.mem_get_info()
    cfg = load_env_cfg("Gray-Stand")
    cfg.scene.num_envs = n
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    env.reset()
    env.step(torch.zeros((n, env.action_manager.total_action_dim), device="cuda:0"))
    free_after, _ = torch.cuda.mem_get_info()
    env.close()
    return {
        "num_envs": n,
        "ok": True,
        "used_gb": round((free_before - free_after) / 1e9, 2),
        "total_gb": round(total / 1e9, 2),
        "free_left_gb": round(free_after / 1e9, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[1024, 2048, 3072, 4096, 6144, 8192])
    ap.add_argument("--child", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        try:
            print("RESULT " + json.dumps(try_size(args.child)))
            return 0
        except Exception as exc:  # noqa: BLE001
            print("RESULT " + json.dumps({
                "num_envs": args.child, "ok": False,
                "error": f"{type(exc).__name__}: {str(exc)[:120]}",
            }))
            return 1

    print(f"{'robots':>8}  {'result':>8}  {'VRAM used':>10}  {'left':>8}")
    print("-" * 46)
    best = None
    for n in sorted(args.sizes):
        proc = subprocess.run(
            [sys.executable, __file__, "--child", str(n)],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        r = json.loads(line[len("RESULT "):]) if line else {
            "num_envs": n, "ok": False, "error": "no result - the process died"}
        if r["ok"]:
            best = r
            print(f"{n:8,}  {'ok':>8}  {r['used_gb']:9.2f}G  {r['free_left_gb']:7.2f}G")
        else:
            print(f"{n:8,}  {'FAILED':>8}  {r.get('error', '')[:40]}")
            break

    if best:
        print(f"\nlargest that works: {best['num_envs']:,} robots, "
              f"{best['used_gb']:.2f} GB of {best['total_gb']:.2f} GB")
        print(f"  python scripts/train_stand.py --num-envs {best['num_envs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
