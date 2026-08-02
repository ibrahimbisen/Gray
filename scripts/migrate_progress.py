#!/usr/bin/env python3
"""Move the one set of pre-runs/ results into progress/runs/<name>/.

    python scripts/migrate_progress.py 2026-08-01_first-attempt           # dry run
    python scripts/migrate_progress.py 2026-08-01_first-attempt --apply

progress/summary.csv, progress/videos/ and progress/policies/ used to sit at the
top level, where every training run wrote to the same three places and the second
run silently overwrote the first run's measurements. progress_store.py now files
them under progress/runs/<run>/. This carries the one set of results that
predates that layout across, so nothing measured is lost to the change.

WHY THE NAME IS AN ARGUMENT INSTEAD OF BEING LOOKED UP
------------------------------------------------------
Every other path in this project is derived from the training run's own log
directory, so no name has to be chosen or remembered. These results are the
exception: the run that produced them was deleted out of
logs/rsl_rl/gray_residual/ before this layout existed, and its timestamp went
with it. Attaching them to one of the surviving run directories would file 2350
iterations of a dead run's numbers under a run that has done about 60 - a wrong
answer the dashboard would then repeat as fact. A descriptive name
(2026-08-01_first-attempt) records the date, which is known, and claims no
provenance that is not.

progress/baseline.json and progress/joints/ are deliberately left where they are:
the hand-written gait and the joint atlas belong to no single run.

Safe to re-run. Nothing moves without --apply, and a destination that already
holds results is refused rather than merged into - a half-migrated folder mixing
two runs' clips is the exact failure this layout exists to prevent.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)

import progress_store  # noqa: E402

SRC_SUMMARY = os.path.join(ROOT, "progress", "summary.csv")
SRC_VIDEOS = os.path.join(ROOT, "progress", "videos")
SRC_POLICIES = os.path.join(ROOT, "progress", "policies")


def _rel(path: str) -> str:
    """Repo-relative with forward slashes, so a printed path and a cell in
    summary.csv always name the same file the same way."""
    return progress_store.media_relpath(ROOT, path)


def _files_in(directory: str) -> list[str]:
    """File names directly inside `directory`; empty if it is not there."""
    if not os.path.isdir(directory):
        return []
    return sorted(n for n in os.listdir(directory)
                  if os.path.isfile(os.path.join(directory, n)))


def _new_video_cell(old: str, videos_dir: str) -> str:
    """One summary.csv video cell, repointed at the moved clip.

    Only the folder changed, so the file name in the old cell is what identifies
    the clip. The old cells were written on Windows and hold backslashes, so the
    separator is normalised here rather than trusting whichever os.path module
    happens to be loaded. A blank cell stays blank: progress_store.media_relpath
    treats a missing clip as an empty cell, never an error.
    """
    name = os.path.basename(old.replace("\\", "/").rstrip("/"))
    if not name:
        return ""
    return _rel(os.path.join(videos_dir, name))


def _read_csv(csv_path: str) -> tuple[list[list[str]], str]:
    """Every row of summary.csv, plus the line ending the file already uses.

    Read as rows rather than through the column names in
    scripts/make_progress_videos.py: importing that module pulls in torch and
    MuJoCo, and a file-moving script has no business loading either. Preserving
    the existing terminator keeps the diff to the one column being rewritten.
    """
    with open(csv_path, "rb") as fh:
        terminator = "\r\n" if b"\r\n" in fh.read() else "\n"
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.reader(fh)), terminator


def _rewrite_video_column(csv_path: str, videos_dir: str,
                          check_dir: str, apply: bool) -> tuple[int, list[str]]:
    """Point every video cell at `videos_dir`. Returns (cells, missing clips).

    `check_dir` is where the clips can be found right now, which differs from
    `videos_dir` during a dry run - the videos have not moved yet, but the names
    are the same either way, so the missing-clip report is the real one.
    """
    rows, terminator = _read_csv(csv_path)
    if not rows:
        return 0, []
    header = rows[0]
    if "video" not in header:
        raise SystemExit(f"REFUSING: {_rel(csv_path)} has no 'video' column")
    col = header.index("video")

    rewritten = 0
    missing = []
    for row in rows[1:]:
        if col >= len(row):
            continue
        cell = _new_video_cell(row[col], videos_dir)
        row[col] = cell
        rewritten += 1
        if cell and not os.path.exists(os.path.join(check_dir,
                                                    os.path.basename(cell))):
            missing.append(cell)

    if apply:
        # Written beside the real file and renamed in, so an interruption leaves
        # the complete old file or the complete new one, never a mixture. Same
        # rule scripts/make_progress_videos.py follows for this file.
        tmp_path = csv_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8", newline="") as fh:
                csv.writer(fh, lineterminator=terminator).writerows(rows)
            os.replace(tmp_path, csv_path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
    return rewritten, missing


def _move(src: str, dst: str, apply: bool) -> None:
    """Move a file or a whole folder to a path that does not exist yet.

    shutil.move drops the source INSIDE dst when dst is an existing directory,
    which would produce progress/runs/<name>/videos/videos - so an empty leftover
    destination folder is removed first. A non-empty one never reaches here; it
    is refused in main().
    """
    if not apply:
        return
    if os.path.isdir(dst) and not os.listdir(dst):
        os.rmdir(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)


def _occupied(paths: dict) -> list[str]:
    """What the destination already holds, as printable lines. Empty means safe."""
    found = []
    if os.path.exists(paths["summary_csv"]):
        found.append(_rel(paths["summary_csv"]))
    for key in ("videos_dir", "policies_dir"):
        n = len(_files_in(paths[key]))
        if n:
            found.append(f"{_rel(paths[key])} ({n} files)")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name",
                    help="destination run name under progress/runs/")
    ap.add_argument("--apply", action="store_true",
                    help="actually move the files (default is a dry run)")
    args = ap.parse_args()

    name = args.name.strip()
    # The name becomes a directory, so a separator in it would scatter the
    # results somewhere other than progress/runs/ and quietly succeed.
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        print(f"REFUSING: {args.name!r} is not a usable run folder name")
        return 1

    paths = progress_store.run_paths(ROOT, name)
    sources = [
        ("summary.csv", SRC_SUMMARY, paths["summary_csv"]),
        ("videos/", SRC_VIDEOS, paths["videos_dir"]),
        ("policies/", SRC_POLICIES, paths["policies_dir"]),
    ]
    present = [(label, src, dst) for label, src, dst in sources
               if os.path.exists(src)]

    if not present:
        # Re-running after a successful move is not an error: there is simply
        # nothing left at the top level to carry across.
        print("nothing to migrate: progress/ has no top-level summary.csv, "
              "videos/ or policies/")
        return 0

    occupied = _occupied(paths)
    if occupied:
        print(f"REFUSING: {_rel(paths['dir'])} already holds results:")
        for line in occupied:
            print(f"  {line}")
        print("  pick a different name, or clear that folder deliberately")
        return 1

    mode = "moving" if args.apply else "would move (dry run, pass --apply)"
    print(f"{mode} into {_rel(paths['dir'])}")
    for label, src, dst in present:
        n = len(_files_in(src)) if os.path.isdir(src) else 1
        print(f"  {_rel(src)} -> {_rel(dst)}  ({n} files)")
    for label, src, dst in sources:
        if (label, src, dst) not in present:
            print(f"  {_rel(src)} is already gone - skipped")

    for _label, src, dst in present:
        _move(src, dst, args.apply)

    # The clips are checked where they actually are: already moved under --apply,
    # still at the top level during a dry run.
    csv_path = paths["summary_csv"] if args.apply else SRC_SUMMARY
    check_dir = paths["videos_dir"] if args.apply else SRC_VIDEOS
    if os.path.exists(csv_path):
        cells, missing = _rewrite_video_column(
            csv_path, paths["videos_dir"], check_dir, args.apply)
        verb = "rewrote" if args.apply else "would rewrite"
        print(f"{verb} {cells} video cells to point at "
              f"{progress_store.media_relpath(ROOT, paths['videos_dir'])}/")
        if missing:
            print(f"WARNING: {len(missing)} video cells name a file that is "
                  f"not on disk:")
            for cell in missing:
                print(f"  {cell}")

    if args.apply:
        # progress/ can hold videos/ and policies/ with no top-level summary.csv,
        # and moving just those two is a success: counted like everything else,
        # not reported as a traceback over files that arrived safely.
        if os.path.exists(paths["summary_csv"]):
            with open(paths["summary_csv"], "rb") as fh:
                lines = sum(1 for _ in fh)
            print(f"summary.csv lines: {lines}")
        print(f"videos:            {len(_files_in(paths['videos_dir']))}")
        print(f"policies:          {len(_files_in(paths['policies_dir']))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
