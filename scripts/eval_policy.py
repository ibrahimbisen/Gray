#!/usr/bin/env python3
"""Score trained policies against the Phase 2 gait, in plain MuJoCo.

    python scripts/eval_policy.py                     # newest checkpoint, cheap screen
    python scripts/eval_policy.py --checkpoint 950    # one checkpoint by number
    python scripts/eval_policy.py --compare 950 1100  # several, side by side
    python scripts/eval_policy.py --compare 950 1100 --full     # under randomisation
    python scripts/eval_policy.py --measure-baseline  # rewrite progress/baseline.json
    python scripts/eval_policy.py --video out.mp4     # one deterministic clip

This is the dual-sim check docs/PROJECT_NOTES.md requires before any hardware
rollout: a policy re-scored in a second engine, against the baseline it has to beat.
See train/evaluate.py for why the second engine is worth the trouble, and for why
one seed at one speed on the nominal robot is not an evaluation.

WHAT THE TWO GRIDS ARE
----------------------
The default SCREEN is 5 seeds x 3 commands on the nominal robot: about 15 rollouts,
a few seconds, enough to rank checkpoints without lying about the error bars.

--full is the GAUNTLET: 5 seeds x 5 commands x 8 randomised robots, 200 rollouts.
Mass, COM, friction, servo gains, armature, zero offsets and 10-40 ms of command
latency all move, over the same ranges training uses. This is the grid that answers
the question PROJECT_NOTES actually asks - "beat Phase 2 under full randomisation" -
and it is the only one whose worst case means anything.

Everything is reported as mean +/- sd [worst]. The worst column is not decoration:
the real robot is one draw from this distribution and nobody gets to pick which.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# For BASELINE_JSON only. The baseline is NOT per-run - it is the same hand-written
# gait every time, and the thing every run is measured against - so it stays at the
# top level while summary.csv, videos and policies moved under progress/runs/.
import progress_store  # noqa: E402
from train.evaluate import (  # noqa: E402
    FULL_COMMANDS,
    FULL_SEEDS,
    NOMINAL,
    REPORT_HEADER,
    SCREEN_COMMANDS,
    SCREEN_SEEDS,
    evaluate,
    find_checkpoints,
    format_paired,
    format_report,
    format_row,
    format_sweep,
    load_policy,
    measure_baseline,
    paired_delta,
    resolve_checkpoint,
    rollout,
    sample_draws,
)


def _grid(full: bool) -> dict:
    if full:
        return dict(commands=FULL_COMMANDS, seeds=FULL_SEEDS, draws=sample_draws())
    return dict(commands=SCREEN_COMMANDS, seeds=SCREEN_SEEDS, draws=(NOMINAL,))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", help="path, or just the iteration number")
    ap.add_argument("--compare", nargs="+", metavar="CKPT",
                    help="score several checkpoints on the same grid")
    ap.add_argument("--run-dir", help="logs/rsl_rl/gray_residual/<run> (default: newest)")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--full", action="store_true",
                    help="the gauntlet: 5 commands x 5 seeds x 8 randomised robots")
    ap.add_argument("--jobs", type=int, default=None,
                    help="worker processes (default: one per core, capped at 16)")
    ap.add_argument("--sweep", action="store_true",
                    help="also print the per-command breakdown")
    ap.add_argument("--measure-baseline", action="store_true",
                    help=f"re-measure the Phase 2 gait over 30 seeds and write "
                         f"{progress_store.BASELINE_JSON}")
    ap.add_argument("--baseline-seeds", type=int, default=30)
    ap.add_argument("--video", metavar="MP4",
                    help="render one deterministic clip (seed 0, nominal robot)")
    ap.add_argument("--speed", type=float, default=0.0529,
                    help="commanded m/s for --video only")
    ap.add_argument("--seed", type=int, default=0, help="for --video only")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the Phase 2 comparison run")
    args = ap.parse_args()

    if args.measure_baseline:
        blob = measure_baseline(seeds=tuple(range(args.baseline_seeds)),
                                duration=args.duration, jobs=args.jobs)
        print_baseline(blob)
        return

    names = args.compare or ([args.checkpoint] if args.checkpoint else [])
    if not names:
        found = find_checkpoints(args.run_dir)
        if not found:
            print("no checkpoints under logs/rsl_rl/gray_residual/ - has it trained?")
            raise SystemExit(1)
        names = [found[-1]]

    paths = [resolve_checkpoint(n, args.run_dir) for n in names]
    grid = _grid(args.full)
    n = len(grid["commands"]) * len(grid["seeds"]) * len(grid["draws"])

    print(f"grid        {len(grid['commands'])} commands x {len(grid['seeds'])} seeds "
          f"x {len(grid['draws'])} draws = {n} rollouts per policy   "
          f"({'FULL randomisation' if args.full else 'nominal dynamics'})")
    print(f"commands    {', '.join(f'{c*1000:+.1f}' for c in grid['commands'])} mm/s")
    print(f"duration    {args.duration:.0f} s\n")

    reports = []
    if not args.no_baseline:
        reports.append(evaluate(None, duration=args.duration, jobs=args.jobs,
                                progress=True, label="Phase 2 gait", **grid))
    for path in paths:
        policy = load_policy(path)
        reports.append(evaluate(policy, duration=args.duration, jobs=args.jobs,
                                progress=True, **grid))

    print(REPORT_HEADER)
    print("-" * len(REPORT_HEADER))
    for report in reports:
        print(format_report(report))
    print("\nmean +/- sd [worst case across the grid]")

    if args.sweep:
        for report in reports:
            print(f"\n{report.label}")
            print(format_sweep(report))

    # Paired verdicts. Every report ran the same commands, seeds and draws, so each
    # rollout has an exact counterpart in every other report and the comparison can
    # be made rollout by rollout. Without that, the spread across randomised robots
    # (sd ~10 mm/s) buries the difference between two checkpoints (~2 mm/s) and the
    # harness cannot tell them apart - which is the failure it exists to fix.
    if len(reports) > 1:
        print("\npaired comparisons on speed tracking, matched rollout by rollout")
        print("  (2 x standard error either side; if that spans zero it says so)")
        for i, report in enumerate(reports):
            for other in reports[i + 1:]:
                print(format_paired(report.label, other.label,
                                    paired_delta(report, other)))

    if args.video:
        path = paths[-1]
        clip = rollout(load_policy(path), speed_ms=args.speed, duration=args.duration,
                       seed=args.seed, video=args.video)
        print()
        print(format_row("clip", clip))
        print(f"wrote {clip.get('video')}")


def print_baseline(blob: dict) -> None:
    """The Phase 2 gait as a distribution. This is what replaces '675.4 mm'."""
    print()
    print("=" * 78)
    print("PHASE 2 CLASSICAL GAIT - re-measured as a distribution")
    print("=" * 78)
    print(f"  {blob['n_seeds']} seeds x {len(blob['commands_ms'])} commands, "
          f"{blob['duration_s']:.0f} s each, plus the same grid over "
          f"{blob['n_draws']} randomised robots")
    print()
    ref = blob["nominal"]["per_command"][str(blob["nominal"]["reference_command_mms"])]
    print(f"  at the reference command {blob['nominal']['reference_command_mms']:.1f} "
          f"mm/s, nominal robot:")
    for key, unit, digits in (("distance_mm", "mm", 1), ("speed_mms", "mm/s", 1),
                              ("drift_abs_mm", "mm", 1), ("upright_min", "", 3),
                              ("foot_force_p99_n", "N", 1)):
        s = ref.get(key)
        if s:
            print(f"    {key:<18} {s['mean']:8.{digits}f} +/- {s['sd']:.{digits}f} sd "
                  f"(sem {s['sem']:.{digits}f}, worst {s['worst']:.{digits}f}) "
                  f"{unit}")
    print()
    print(f"  across the whole command sweep, nominal:")
    print(f"    speed-tracking MAE  {blob['track_mae_mms']:.1f} mm/s")
    print(f"    top speed           {blob['top_speed_mms']:.1f} mm/s "
          f"(the stride scale saturates - asking for more does not get more)")
    print()
    dr = blob["randomised"]["overall"]
    dr_ref = blob["randomised"]["reference"]
    print(f"  under full randomisation ({blob['n_draws']} draws):")
    print(f"    distance @52.9      {dr_ref['distance_mm']['mean']:.1f} +/- "
          f"{dr_ref['distance_mm']['sd']:.1f} sd, "
          f"worst {dr_ref['distance_mm']['worst']:.1f} mm")
    print(f"    speed-tracking MAE  {dr['speed_error_mms']['mean']:.1f} +/- "
          f"{dr['speed_error_mms']['sd']:.1f} sd, "
          f"worst {dr['speed_error_mms']['worst']:.1f} mm/s")
    print(f"    |drift|             {dr['drift_abs_mm']['mean']:.1f} +/- "
          f"{dr['drift_abs_mm']['sd']:.1f} sd, "
          f"worst {dr['drift_abs_mm']['worst']:.1f} mm")
    print(f"    falls               {dr['fall_rate']*100:.1f}% of rollouts")
    print()
    print(f"  the figure this replaces: {blob['superseded']['distance_mm']} mm from a "
          f"single seed.")
    print(f"  it sits {abs(blob['superseded']['distance_mm'] - blob['distance_mm']) / max(blob['distance_mm_sd'], 1e-9):.2f} "
          f"standard deviations from the mean - one draw, not a benchmark.")
    print()
    print(f"  written to {progress_store.BASELINE_JSON}")
    print("=" * 78)


if __name__ == "__main__":
    main()
