#!/usr/bin/env python3
"""Render one video per training checkpoint, all from identical conditions.

    python scripts/make_progress_videos.py

Produces, under progress/:

    videos/baseline_phase2.mp4     the hand-written gait, for reference
    videos/iter_0050.mp4 ...       one per checkpoint
    policies/iter_0050.npz ...     the policy behind each clip, ~200 KB each
    summary.csv                    the numbers behind each clip

Every clip uses the SAME seed and the SAME commanded speed, so nothing differs
between them except how much the policy has learned. That is the whole point: a
side-by-side is then honest evidence of progress rather than a lucky take.

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
    find_checkpoints,
    load_policy,
    rollout,
    save_policy_npz,
)

OUT = "progress"
# One column per training objective, so the dashboard can rank checkpoints by each
# goal separately - the fastest walk and the softest walk are rarely the same one.
FIELDS = ["iteration", "distance_mm", "speed_mms", "drift_mm", "upright_min",
          "height_mm", "fell", "foot_force_p99_n", "joint_acc_rms",
          "power_mean_w", "cost_of_transport", "video"]


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


def _row(tag: str, r: dict, video: str) -> dict:
    return {
        "iteration": tag,
        "distance_mm": round(r["distance"] * 1000, 1),
        "speed_mms": round(r["speed"] * 1000, 1),
        "drift_mm": round(r["drift"] * 1000, 1),
        "upright_min": round(r["upright_min"], 3),
        "height_mm": round(r["height_mean"] * 1000, 1),
        "fell": int(r["fell"]),
        "foot_force_p99_n": round(r["foot_force_p99"], 2),
        "joint_acc_rms": round(r["joint_acc_rms"], 2),
        "power_mean_w": round(r["power_mean_w"], 3),
        "cost_of_transport": round(r["cost_of_transport"], 2),
        "video": video,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", help="logs/rsl_rl/gray_residual/<run> (default: newest)")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--speed", type=float, default=0.0529, help="commanded m/s")
    ap.add_argument("--seed", type=int, default=0, help="identical for every clip")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--force", action="store_true", help="re-render existing clips")
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

    common = dict(speed_ms=args.speed, duration=args.duration, seed=args.seed,
                  fps=args.fps, width=args.width, height=args.height)
    rows: list[dict] = []

    header = (f"{'iter':>8}  {'distance':>11}  {'speed':>11}  {'drift':>10}  "
              f"{'upright':>7}  fell")

    baseline_mp4 = None if args.no_video else os.path.join(vid_dir,
                                                           "baseline_phase2.mp4")
    if args.force or args.no_video or not os.path.exists(baseline_mp4 or ""):
        b = rollout(None, video=baseline_mp4, **common)
        rows.append(_row("baseline", b, baseline_mp4 or ""))
        print(header)
        print(f"{'gait':>8}  {b['distance']*1000:>9.1f} mm  "
              f"{b['speed']*1000:>8.1f} mm/s  {b['drift']*1000:>7.1f} mm  "
              f"{b['upright_min']:>7.3f}  {int(b['fell'])}")
        print("-" * len(header), flush=True)

    for i, ckpt in enumerate(checkpoints, 1):
        policy = load_policy(ckpt)
        tag = f"iter_{policy.iteration:04d}"
        mp4 = None if args.no_video else os.path.join(vid_dir, f"{tag}.mp4")
        npz = os.path.join(pol_dir, f"{tag}.npz")

        if not args.force and mp4 and os.path.exists(mp4) and os.path.exists(npz):
            continue

        save_policy_npz(policy, npz)
        r = rollout(policy, video=mp4, **common)
        if "error" in r:
            print(f"{policy.iteration:>8}  FAILED: {r['error']}")
            continue
        rows.append(_row(str(policy.iteration), r, mp4 or ""))
        print(f"{policy.iteration:>8}  {r['distance']*1000:>9.1f} mm  "
              f"{r['speed']*1000:>8.1f} mm/s  {r['drift']*1000:>7.1f} mm  "
              f"{r['upright_min']:>7.3f}  {int(r['fell'])}", flush=True)

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
    print(f"\n{len(rows)} clip(s) -> {vid_dir}")
    print(f"policies      -> {pol_dir}  (small enough to commit)")
    print(f"numbers       -> {csv_path}  ({len(merged)} row(s) in total: "
          f"{added} new, {replaced} re-scored{tidied})")


if __name__ == "__main__":
    main()
