"""Where a training run's measured results live on disk.

ONE rule, in ONE place, because two programs have to agree on it:
`scripts/make_progress_videos.py` writes these files and `dashboard/collect.py`
reads them. When they disagreed, the dashboard showed the previous run's scores
underneath the current run's round counter - the same page reporting two
different runs as though they were one.

    progress/
      baseline.json                  the hand-written gait. NOT per-run: it is
                                     the same walk every time and is the thing
                                     every run is measured against.
      joints/                        the joint atlas. Also not per-run - it
                                     describes the machine, not an attempt.
      runs/
        2026-08-01_20-14-14/         named after the training run that made it
          summary.csv                one row per scored checkpoint
          videos/                    one clip per scored checkpoint
          policies/                  the policy behind each clip, ~200 KB each

The run name is the training run's own directory name under
logs/rsl_rl/gray_residual/, so nothing has to be chosen, typed or remembered:
a new run writes to a folder that does not exist yet and therefore starts
empty. Previous runs are never touched and never deleted.

Stdlib only. dashboard/collect.py is polled every few seconds and must not grow
a torch or MuJoCo import by way of this file.
"""

from __future__ import annotations

import os

# Kept with forward slashes: they are joined with os.path.join (which accepts
# them on Windows) and also printed into user-facing messages, where a
# backslash reads as a typo.
EXPERIMENT_DIR = "logs/rsl_rl/gray_residual"
PROGRESS_DIR = "progress"
RUNS_DIR = "progress/runs"

# Not per-run, and deliberately so - see the module docstring.
BASELINE_JSON = "progress/baseline.json"
JOINTS_DIR = "progress/joints"


def _newest_mtime(directory: str) -> float:
    """Most recent mtime of any file under `directory` (0.0 if there are none)."""
    newest = 0.0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                continue
    return newest


def run_name(run_dir: str) -> str:
    """The run's folder name, given any path to it.

    Tolerates a trailing separator, which is what shell completion produces:
    `logs/rsl_rl/gray_residual/2026-08-01_20-14-14/` must not come back "".
    """
    return os.path.basename(os.path.normpath(run_dir))


def newest_run_name(root: str = ".") -> str | None:
    """The training run that was written to most recently, or None if none exist.

    By content mtime rather than by folder name, matching how the dashboard
    picks the live run - a resumed run keeps its original timestamped name, so
    sorting the names alphabetically would pick the wrong one.
    """
    exp_dir = os.path.join(root, *EXPERIMENT_DIR.split("/"))
    if not os.path.isdir(exp_dir):
        return None
    candidates = [
        os.path.join(exp_dir, name)
        for name in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, name))
    ]
    if not candidates:
        return None
    return os.path.basename(max(candidates, key=_newest_mtime))


def run_paths(root: str, name: str) -> dict:
    """Every path belonging to one run. Creates nothing - see `ensure_run_dirs`."""
    base = os.path.join(root, *RUNS_DIR.split("/"), name)
    return {
        "name": name,
        "dir": base,
        "summary_csv": os.path.join(base, "summary.csv"),
        "videos_dir": os.path.join(base, "videos"),
        "policies_dir": os.path.join(base, "policies"),
    }


def ensure_run_dirs(root: str, name: str) -> dict:
    """`run_paths`, with the video and policy folders created."""
    paths = run_paths(root, name)
    os.makedirs(paths["videos_dir"], exist_ok=True)
    os.makedirs(paths["policies_dir"], exist_ok=True)
    return paths


def list_runs(root: str = ".") -> list[str]:
    """Run names that have scored results on disk, newest first.

    Reads progress/runs/, not the training logs, so a run whose checkpoints have
    since been deleted still appears - the whole point of keeping these folders
    is that the measurements outlive the checkpoints they came from.
    """
    runs_dir = os.path.join(root, *RUNS_DIR.split("/"))
    if not os.path.isdir(runs_dir):
        return []
    names = [
        name for name in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, name))
    ]
    # Names are timestamps (2026-08-01_20-14-14), so a plain reverse sort is
    # newest-first for every run this project has made. mtime is the tie-break
    # for anything hand-named that does not follow the pattern.
    names.sort(key=lambda n: (n, _newest_mtime(os.path.join(runs_dir, n))),
               reverse=True)
    return names


def media_relpath(root: str, path: str) -> str:
    """An absolute path turned into the repo-relative, forward-slash form that
    summary.csv stores and the dashboard's /media/ route serves.

    Returns "" for a path outside the repo rather than raising: a missing clip
    is a blank cell on the page, never a crash in the scorer.
    """
    if not path:
        return ""
    try:
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    except ValueError:          # different drive on Windows
        return ""
    if rel.startswith(".."):
        return ""
    return rel.replace(os.sep, "/")
