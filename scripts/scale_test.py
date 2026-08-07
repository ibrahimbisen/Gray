"""How much does another thousand robots cost, in seconds and in megabytes?

    python scripts/scale_test.py                    5000 to 10000
    python scripts/scale_test.py 5000 8000 12000    whatever you like

THE QUESTION. Task Manager showed 5000 robots using 5.3 GB of a 12 GB card at
99 percent utilisation, which says the card is short of TIME and not of MEMORY.
So there is room for more robots - but more robots only help if the cost per
iteration grows slower than the batch does. This measures both, per size.

WHY IT DOES NOT JUST TIME THE RUN. A training process spends roughly 40 seconds
building the model, compiling kernels and allocating before it steps once. Over
a 20-iteration run that startup is most of the wall clock, and dividing it out
gives a number that describes the compiler rather than the physics. So this
polls the run's own metrics.csv once a second, throws away the first few rows,
and takes the rate from the steady part.

MEMORY IS SAMPLED THE SAME WAY, from nvidia-smi once a second, and reported as
the PEAK - allocation climbs while the rollout buffers fill, so the first
reading is always an underestimate.

Run it with the card free. Never beside a training job: two CUDA processes on
one 12 GB card is what crashed this machine on 5 Aug 2026.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "progress" / "runs"
PY = sys.executable
ITERATIONS = 20
# Rows to drop before measuring. The first iterations carry the tail of the
# allocator warming up and read slower than the steady state.
WARMUP_ROWS = 5


def gpu_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return -1


def one(n_envs: int) -> dict:
    name = f"scale_{n_envs}"
    before = gpu_mib()
    argv = [PY, str(ROOT / "scripts" / "train.py"), "Gray-Walk",
            "--num-envs", str(n_envs), "--iterations", str(ITERATIONS),
            "--name", name, "--no-video", "--seed", "1301",
            "--reward", "even_stance=0.0"]
    proc = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)

    csv, peak, samples = None, before, []
    started = time.time()
    while proc.poll() is None:
        time.sleep(1.0)
        peak = max(peak, gpu_mib())
        if csv is None:
            hits = sorted(RUNS.glob(f"*_{name}/metrics.csv"),
                          key=lambda p: p.stat().st_mtime)
            # Only a directory made since this call started - an older run of
            # the same size would otherwise be measured instead of this one.
            if hits and hits[-1].stat().st_mtime > started:
                csv = hits[-1]
        if csv is not None and csv.exists():
            try:
                rows = sum(1 for _ in csv.open()) - 1
            except OSError:
                continue
            if rows > 0:
                samples.append((time.time(), rows))
    proc.wait()

    # Rate from the steady part: last sample against the first one past warmup.
    per_iter = float("nan")
    usable = [s for s in samples if s[1] > WARMUP_ROWS]
    if len(usable) >= 2 and usable[-1][1] > usable[0][1]:
        per_iter = ((usable[-1][0] - usable[0][0])
                    / (usable[-1][1] - usable[0][1]))
    return {"envs": n_envs, "per_iter": per_iter, "peak_mib": peak,
            "idle_mib": before, "rows": samples[-1][1] if samples else 0}


def main() -> int:
    sizes = ([int(a) for a in sys.argv[1:]]
             or [5000, 6000, 7000, 8000, 9000, 10000])
    print(f"{ITERATIONS} iterations per size, no video, seed 1301.")
    print(f"card at rest: {gpu_mib()} MiB\n")
    print(f"{'robots':>7}{'s / iter':>10}{'peak MiB':>10}{'robots/s':>11}"
          f"{'vs 5000':>9}")
    print("-" * 47)
    base = None
    out = []
    for n in sizes:
        r = one(n)
        out.append(r)
        if base is None:
            base = r["per_iter"]
        rate = n / r["per_iter"] if r["per_iter"] == r["per_iter"] else 0.0
        rel = (r["per_iter"] / base) if base and base == base else float("nan")
        print(f"{n:>7}{r['per_iter']:>10.2f}{r['peak_mib']:>10}"
              f"{rate:>11,.0f}{rel:>8.2f}x")
    print("\nrobots/s is the number that matters: it is how much experience the "
          "card\ncollects per second. If it stops rising, more robots buy a "
          "better gradient\nper iteration and nothing else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
