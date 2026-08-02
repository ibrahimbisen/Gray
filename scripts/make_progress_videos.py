#!/usr/bin/env python3
"""Render one video per training checkpoint, and score each one properly.

    python scripts/make_progress_videos.py
    python scripts/make_progress_videos.py --no-video      # numbers only, much faster
    python scripts/make_progress_videos.py --full          # score under randomisation

Produces, under progress/:

    videos/baseline_phase2.mp4     the hand-written gait, for reference
    videos/iter_0050.mp4 ...       one per checkpoint
    policies/iter_0050.npz ...     the policy behind each clip, ~200 KB each
    summary.csv                    the numbers behind each clip

THE CLIP AND THE SCORE ARE TWO DIFFERENT THINGS, ON PURPOSE
-----------------------------------------------------------
Every clip is one deterministic rollout at the same seed, the same commanded speed
and the same nominal robot, so a side-by-side reel shows the policy improving and
nothing else. That part is unchanged - old clips remain reproducible.

The NUMBERS beside each clip are no longer that single rollout. A single rollout was
what made this harness score checkpoint 950 a loser: the classical gait's own
distance has a standard deviation of about 86 mm at one seed, which is larger than
most of the differences anyone was reading off this file. Each row is now a grid -
5 seeds x 3 commands by default, 5 x 5 x 8 randomised robots with --full - and every
column that varies carries its spread and its worst case alongside the mean. Rank
checkpoints on `track_mae_mms` (lower is better) rather than on distance at one
speed: distance at one speed cannot see a policy that has stopped tracking the
command, and the command sweep can.

The "baseline" row here is the classical gait scored on the SAME grid as every
checkpoint, so the comparison in this file is like for like. It is not the
authoritative baseline - that is progress/baseline.json, 30 seeds x 5 commands,
written by `python scripts/eval_policy.py --measure-baseline`.

Safe to re-run - existing clips are skipped, so it can be run repeatedly while
training is still going and it will only pick up what is new. summary.csv is
rewritten each time rather than appended to: there is exactly one row per
checkpoint, a fresh score replaces the old one for that checkpoint, and the file
is swapped into place in one go so an interrupted run cannot leave it half written.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.evaluate import (  # noqa: E402
    FULL_COMMANDS,
    FULL_SEEDS,
    NOMINAL,
    SCREEN_COMMANDS,
    SCREEN_SEEDS,
    Report,
    evaluate,
    find_checkpoints,
    load_policy,
    rollout,
    sample_draws,
    save_policy_npz,
)
# The same "mean +/- sd [worst]" formatting eval_policy.py prints, so the two
# commands cannot disagree about how a number is presented.
from train.evaluate import _fmt, _pct  # noqa: E402

OUT = "progress"

# One column per training objective, so the dashboard can rank checkpoints by each
# goal separately - the fastest walk and the softest walk are rarely the same one.
#
# The bare column names (distance_mm, speed_mms, ...) are the MEAN at the reference
# command 52.9 mm/s, which is exactly what the old single-rollout columns measured,
# so historical rows stay comparable. Their _sd and _worst partners are the new part:
# a mean with no spread beside it is how this file gave the wrong answer twice.
#
# track_mae_mms and top_speed_mms are sweep-wide and have no single-command
# equivalent. They are the two that separate a residual policy from the gait
# underneath it - the gait's stride scale saturates at 1.0, so it cannot go faster
# than about 60 mm/s however fast you ask, and a single-speed score is blind to that.
FIELDS = [
    "iteration",
    "distance_mm", "distance_mm_sd", "distance_mm_worst",
    "speed_mms", "speed_mms_sd", "speed_mms_worst",
    "drift_mm", "drift_mm_sd", "drift_mm_worst",
    "upright_min", "upright_min_worst",
    "height_mm", "fell", "fall_rate",
    "foot_force_p99_n", "foot_force_p99_n_sd", "foot_force_p99_n_worst",
    "joint_acc_rms", "joint_acc_rms_worst",
    "power_mean_w", "cost_of_transport",
    "track_mae_mms", "track_mae_mms_sd", "track_mae_mms_worst",
    "top_speed_mms", "top_speed_mms_worst",
    "n_trials", "grid",
    "video",
]


def _key(tag: str) -> str:
    """One row per checkpoint: the name a row is filed under.

    "baseline" in any spelling is the hand-written gait; everything else is a
    checkpoint number, so "50" and "050" are the same row, not two.
    """
    tag = (tag or "").strip()
    if tag.lower() == "baseline":
        return "baseline"
    try:
        return str(int(float(tag)))
    except ValueError:
        return tag


def _sort_key(row: dict) -> tuple:
    """Baseline first, then checkpoints in numerical order."""
    tag = _key(row.get("iteration", ""))
    if tag == "baseline":
        return (0, 0.0, "")
    try:
        return (1, float(tag), "")
    except ValueError:
        return (2, 0.0, tag)          # anything unexpected, kept at the end


def _read_summary(csv_path: str) -> list[dict]:
    """Rows already in summary.csv. A missing or unreadable file means none."""
    if not os.path.exists(csv_path):
        return []
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
            # Older files may be missing the newer columns; fill those with blanks
            # so the rewritten file still has every column in every row.
            return [{f: (raw.get(f) or "") for f in FIELDS}
                    for raw in csv.DictReader(fh)
                    if _key(raw.get("iteration", ""))]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        # A damaged or oddly encoded file must not take the whole run down with
        # it - say so and carry on, and the rewrite below puts it right.
        print(f"could not read {csv_path} ({exc}); starting a fresh one")
        return []


def _write_summary(csv_path: str, rows: list[dict]) -> None:
    """Write the whole file in one go, then swap it in.

    Writing to a temporary file next to the real one and renaming it means the
    real file is either the old complete version or the new complete version -
    never a half-written mixture, however the run ends.
    """
    tmp_path = csv_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, csv_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _merge_summary(csv_path: str, new_rows: list[dict]):
    """Existing rows plus the new ones, the fresh score winning per checkpoint.

    Returns (rows in display order, how many are brand new, how many replaced an
    existing row, how many duplicate rows an earlier run had left behind). A
    re-run therefore updates rows in place instead of piling up copies.
    """
    old_rows = _read_summary(csv_path)
    merged: dict[str, dict] = {}
    for row in old_rows:
        merged[_key(row["iteration"])] = row
    duplicates = len(old_rows) - len(merged)

    added = replaced = 0
    for row in new_rows:
        key = _key(row["iteration"])
        previous = merged.get(key)
        if previous is not None:
            replaced += 1
            # A --no-video run re-scores a checkpoint without rendering, so it has
            # no clip name to offer. The clip from the earlier run is still sitting
            # in progress/videos, so keep pointing at it rather than blanking the
            # link and losing the video from the dashboard.
            if not row.get("video") and previous.get("video"):
                row = dict(row, video=previous["video"])
        else:
            added += 1
        merged[key] = row

    return sorted(merged.values(), key=_sort_key), added, replaced, duplicates


def _stat(summary: dict, name: str, field: str, digits: int = 1):
    """One number out of an aggregate, or blank if it was never measurable."""
    stats = summary.get(name)
    if not stats:
        return ""
    return round(stats[field], digits)


def _row(tag: str, report: Report, video: str) -> dict:
    """One summary.csv row: means at the reference command, spread, worst case."""
    ref = report.reference
    over = report.overall
    top = report.top_speed_mms() or {}
    fall_rate = over["fall_rate"]

    return {
        "iteration": tag,
        "distance_mm": _stat(ref, "distance_mm", "mean"),
        "distance_mm_sd": _stat(ref, "distance_mm", "sd"),
        "distance_mm_worst": _stat(ref, "distance_mm", "worst"),
        "speed_mms": _stat(ref, "speed_mms", "mean"),
        "speed_mms_sd": _stat(ref, "speed_mms", "sd"),
        "speed_mms_worst": _stat(ref, "speed_mms", "worst"),
        # |drift|, not signed drift: wandering left is no better than wandering
        # right, and averaging the signed value over seeds would cancel the two out
        # into a flattering near-zero.
        "drift_mm": _stat(ref, "drift_abs_mm", "mean"),
        "drift_mm_sd": _stat(ref, "drift_abs_mm", "sd"),
        "drift_mm_worst": _stat(ref, "drift_abs_mm", "worst"),
        "upright_min": _stat(over, "upright_min", "mean", 3),
        "upright_min_worst": _stat(over, "upright_min", "worst", 3),
        "height_mm": _stat(ref, "height_mm", "mean"),
        # A flag for "this checkpoint is degenerate", which is what the dashboard
        # uses it for - not "it fell once in fifteen tries". The honest number is
        # in fall_rate beside it.
        "fell": int(fall_rate >= 0.5),
        "fall_rate": round(fall_rate, 3),
        # Impact and smoothness are taken over the WHOLE sweep: the legs have to
        # survive every command the robot will be given, so the worst case that
        # matters is the worst case anywhere in the envelope.
        "foot_force_p99_n": _stat(over, "foot_force_p99_n", "mean", 2),
        "foot_force_p99_n_sd": _stat(over, "foot_force_p99_n", "sd", 2),
        "foot_force_p99_n_worst": _stat(over, "foot_force_p99_n", "worst", 2),
        "joint_acc_rms": _stat(over, "joint_acc_rms", "mean", 2),
        "joint_acc_rms_worst": _stat(over, "joint_acc_rms", "worst", 2),
        # Power and cost of transport are taken at the reference command only.
        # Cost of transport is power divided by speed, so at the 0 mm/s command it
        # diverges - averaging it across the sweep would let one near-stationary
        # rollout swamp every other number in the column.
        "power_mean_w": _stat(ref, "power_mean_w", "mean", 3),
        "cost_of_transport": _stat(ref, "cost_of_transport", "mean", 2),
        "track_mae_mms": _stat(over, "speed_error_mms", "mean"),
        "track_mae_mms_sd": _stat(over, "speed_error_mms", "sd"),
        "track_mae_mms_worst": _stat(over, "speed_error_mms", "worst"),
        "top_speed_mms": round(top["mean"], 1) if top else "",
        "top_speed_mms_worst": round(top["worst"], 1) if top else "",
        "n_trials": over["n_trials"],
        "grid": f"{len(report.commands)}c x {len(report.seeds)}s x "
                f"{len(report.draws)}d",
        "video": video,
    }


HEADER = (f"{'iter':>8}  {'distance @52.9 mm':>22}  {'speed MAE mm/s':>22}  "
          f"{'top speed mm/s':>22}  {'|drift| @52.9 mm':>22}  falls")


def _line(tag: str, report: Report) -> str:
    """Same columns as scripts/eval_policy.py, so the two never disagree."""
    return (f"{tag:>8}  {_fmt(report.reference.get('distance_mm'))}  "
            f"{_fmt(report.overall.get('speed_error_mms'))}  "
            f"{_fmt(report.top_speed_mms())}  "
            f"{_fmt(report.reference.get('drift_abs_mm'))}  "
            f"{_pct(report.overall['fall_rate'], 5)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", help="logs/rsl_rl/gray_residual/<run> (default: newest)")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--speed", type=float, default=0.0529,
                    help="commanded m/s for the CLIP; the score always sweeps")
    ap.add_argument("--seed", type=int, default=0, help="identical for every clip")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--force", action="store_true", help="re-render existing clips")
    ap.add_argument("--full", action="store_true",
                    help="score on the gauntlet - 5 commands x 5 seeds x 8 "
                         "randomised robots - instead of the cheap nominal screen")
    ap.add_argument("--jobs", type=int, default=None,
                    help="worker processes (default: one per core, capped at 16)")
    ap.add_argument("--no-video", action="store_true",
                    help="score every checkpoint and print the trend, without "
                         "rendering - much faster, and uses no GPU")
    args = ap.parse_args()

    checkpoints = find_checkpoints(args.run_dir)
    if not checkpoints:
        print("no checkpoints yet under logs/rsl_rl/gray_residual/")
        raise SystemExit(1)

    vid_dir = os.path.join(OUT, "videos")
    pol_dir = os.path.join(OUT, "policies")
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(pol_dir, exist_ok=True)

    grid = (dict(commands=FULL_COMMANDS, seeds=FULL_SEEDS, draws=sample_draws())
            if args.full else
            dict(commands=SCREEN_COMMANDS, seeds=SCREEN_SEEDS, draws=(NOMINAL,)))
    per_policy = len(grid["commands"]) * len(grid["seeds"]) * len(grid["draws"])
    scoring = dict(duration=args.duration, jobs=args.jobs, **grid)
    clip = dict(speed_ms=args.speed, duration=args.duration, seed=args.seed,
                fps=args.fps, width=args.width, height=args.height)

    print(f"scoring on {len(grid['commands'])} commands x {len(grid['seeds'])} seeds "
          f"x {len(grid['draws'])} draws = {per_policy} rollouts per checkpoint "
          f"({'FULL randomisation' if args.full else 'nominal dynamics'})")
    print(f"clips are one deterministic rollout: seed {args.seed}, "
          f"{args.speed*1000:.1f} mm/s, nominal robot\n")

    rows: list[dict] = []

    baseline_mp4 = None if args.no_video else os.path.join(vid_dir,
                                                           "baseline_phase2.mp4")
    if args.force or args.no_video or not os.path.exists(baseline_mp4 or ""):
        if baseline_mp4:
            rollout(None, video=baseline_mp4, **clip)
        report = evaluate(None, label="Phase 2 gait", **scoring)
        rows.append(_row("baseline", report, baseline_mp4 or ""))
        print(HEADER)
        print(_line("gait", report))
        print("-" * len(HEADER), flush=True)

    for ckpt in checkpoints:
        policy = load_policy(ckpt)
        tag = f"iter_{policy.iteration:04d}"
        mp4 = None if args.no_video else os.path.join(vid_dir, f"{tag}.mp4")
        npz = os.path.join(pol_dir, f"{tag}.npz")

        if not args.force and mp4 and os.path.exists(mp4) and os.path.exists(npz):
            continue

        save_policy_npz(policy, npz)
        if mp4:
            rollout(policy, video=mp4, **clip)
        report = evaluate(policy, **scoring)
        rows.append(_row(str(policy.iteration), report, mp4 or ""))
        print(_line(str(policy.iteration), report), flush=True)

    csv_path = os.path.join(OUT, "summary.csv")

    if not rows:
        # Nothing new, but tidy away any duplicate rows an earlier run left behind.
        merged, _, _, duplicates = _merge_summary(csv_path, [])
        if duplicates:
            _write_summary(csv_path, merged)
            print(f"removed {duplicates} duplicate row(s) from {csv_path}")
        print("nothing new to render")
        return

    merged, added, replaced, duplicates = _merge_summary(csv_path, rows)
    _write_summary(csv_path, merged)

    tidied = f", {duplicates} duplicate(s) removed" if duplicates else ""
    print(f"\nmean +/- sd [worst case across the grid]")
    print(f"{len(rows)} row(s) scored -> {vid_dir}")
    print(f"policies      -> {pol_dir}  (small enough to commit)")
    print(f"numbers       -> {csv_path}  ({len(merged)} row(s) in total: "
          f"{added} new, {replaced} re-scored{tidied})")


if __name__ == "__main__":
    main()
