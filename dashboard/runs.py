"""What a training run leaves on disk, and how to read it back.

This file is the contract. Training writes exactly this layout and the dashboard
reads exactly this layout - neither side guesses.

    progress/runs/<id>/
        run.json          what this run is and why it exists
        metrics.csv       one row per iteration, appended as it trains
        checkpoints/      iter_0150.npz - the trained policies
        videos/           iter_0150.mp4 - the robot at that checkpoint

Nothing here imports torch or MuJoCo. The dashboard has to open even when the
training environment is broken, because that is exactly when you need to look at it.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "progress" / "runs"

# What can have happened to a run. "finished" and "reached target" both mean the
# training completed normally - the second means it stopped early because the
# reward had nothing left to win (RULES.md rule 1). Neither says whether the
# stage was passed: that is the verifier's word, carried separately as `verdict`.
STATUS = ("running", "finished", "reached target", "cancelled", "failed",
          "interrupted")

# How long a run marked "running" can go without writing anything before we stop
# believing it. A killed process never gets to update its own status, and a card
# that says "running" three hours later is worse than no card.
STALE_AFTER_S = 180.0


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _read_metrics(path: Path) -> tuple[list[str], list[dict]]:
    """The metrics file is being appended to while we read it, so tolerate a
    half-written last line rather than failing the whole page."""
    if not path.exists():
        return [], []
    try:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return [], []
    if not rows:
        return [], []
    clean = []
    for r in rows:
        if any(v is None for v in r.values()):
            continue  # truncated line
        out = {}
        for k, v in r.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        clean.append(out)
    cols = [k for k in clean[0] if isinstance(clean[0][k], float)] if clean else []
    return cols, clean


def _iteration_of(name: str) -> int | None:
    stem = Path(name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def _age(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return ""
    secs = (datetime.now() - then).total_seconds()
    for limit, div, unit in ((60, 1, "s"), (3600, 60, "min"), (86400, 3600, "h")):
        if secs < limit:
            return f"{int(secs / div)} {unit} ago"
    return f"{int(secs / 86400)} d ago"


def read_run(folder: Path) -> dict:
    meta = _read_json(folder / "run.json")
    cols, rows = _read_metrics(folder / "metrics.csv")

    videos = sorted(
        ({"file": f"/media/runs/{folder.name}/videos/{p.name}",
          "iteration": _iteration_of(p.name),
          "name": p.stem}
         # ".writing.mp4" is a film still being encoded. Showing it hands the
         # browser a truncated file that will never finish loading.
         for p in (folder / "videos").glob("*.mp4")
         if not p.name.endswith(".writing.mp4")),
        key=lambda v: (v["iteration"] is None, v["iteration"] or 0),
    )
    checkpoints = sorted(
        _iteration_of(p.name) or 0 for p in (folder / "checkpoints").glob("*.npz")
    )

    latest = rows[-1] if rows else {}
    target = meta.get("iterations_target") or 0
    done = int(latest.get("iteration", len(rows)))

    # A run that says it is running but has not written for a while was killed
    # without getting the chance to say so.
    # Runs made before the vocabulary was settled say "done" and "stopped".
    status = {"done": "finished", "stopped": "cancelled"}.get(
        meta.get("status", "unknown"), meta.get("status", "unknown"))
    if status == "running":
        newest = max((p.stat().st_mtime for p in folder.rglob("*") if p.is_file()),
                     default=0.0)
        if newest and (datetime.now().timestamp() - newest) > STALE_AFTER_S:
            status = "interrupted"

    # The distinguishing part of the folder name: 2026-08-03_00-56-36_push_v4
    # matters to a person as "push_v4".
    variant = folder.name.split("_", 2)[-1] if folder.name.count("_") >= 2 else folder.name

    return {
        "id": folder.name,
        "name": meta.get("name") or folder.name,
        "variant": variant,
        "purpose": meta.get("purpose", ""),
        "stage": meta.get("stage"),
        "stage_name": meta.get("stage_name", ""),
        "task": meta.get("task", ""),
        "status": status,
        "bar": meta.get("bar", ""),
        "notes": meta.get("notes", ""),
        "started": meta.get("started"),
        "started_age": _age(meta.get("started")),
        "finished": meta.get("finished"),
        "iterations_done": done,
        "iterations_target": target,
        "progress": (done / target) if target else 0.0,
        "metric_columns": cols,
        "metrics": rows,
        "latest": latest,
        "videos": videos,
        "checkpoints": checkpoints,
        # Written by scripts/verify.py. Training finishing is not the same as the
        # stage being passed, so the two are shown separately and never merged.
        "verdict": meta.get("verdict", ""),
        "verdict_detail": meta.get("verdict_detail", ""),
        "notes": meta.get("notes", ""),
        # What this run was scored on, recorded at launch. Kept per-run rather
        # than read from the task, so an old run still says what IT was scored on
        # after the rewards have been changed.
        "scoring": meta.get("scoring", []),
    }


def all_runs() -> list[dict]:
    """Newest first, WITH every metric row. An empty progress/runs/ is normal.

    Heavy - one call is megabytes once there are a few runs. The pages use
    `all_summaries()` for the list and `detail()` for the one being looked at.
    Kept because it is the honest "give me everything" call.
    """
    if not RUNS.is_dir():
        return []
    runs = [read_run(p) for p in RUNS.iterdir() if p.is_dir()]
    runs.sort(key=lambda r: (r["started"] or "", r["id"]), reverse=True)
    return runs


# ---------------------------------------------------------------------------
# Light reads.
#
# The training centre polls every few seconds and can hold a hundred runs. Every
# metric row of every run in that payload was 1.6 MB at sixteen runs and grows
# forever - so the list carries only what a list shows, and the full curves are
# fetched for the one run being looked at.
# ---------------------------------------------------------------------------

# Parsed metrics, keyed by file mtime. Re-reading sixteen CSVs on every poll is
# wasted work when none of them have changed - and the one that HAS changed is
# the running one, whose mtime moves, so it is never served stale.
_METRIC_CACHE: dict[str, tuple[float, list[str], list[dict]]] = {}


def _metrics_cached(path: Path) -> tuple[list[str], list[dict]]:
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return [], []
    hit = _METRIC_CACHE.get(str(path))
    if hit and hit[0] == stamp:
        return hit[1], hit[2]
    cols, rows = _read_metrics(path)
    _METRIC_CACHE[str(path)] = (stamp, cols, rows)
    return cols, rows


def _duration(started: str | None, finished: str | None) -> str:
    if not started:
        return ""
    try:
        a = datetime.fromisoformat(started)
        b = datetime.fromisoformat(finished) if finished else datetime.now()
    except ValueError:
        return ""
    secs = max(0, int((b - a).total_seconds()))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def summary(folder: Path) -> dict:
    """One run, without its metric history. Everything a list or card needs."""
    full = read_run(folder)
    rows = full.pop("metrics", [])
    full["samples"] = len(rows)
    full["videos"] = len(full["videos"])
    full["checkpoints"] = len(full["checkpoints"])
    full["scoring"] = len(full["scoring"])
    full["duration"] = _duration(full["started"], full["finished"])
    return full


def all_summaries() -> list[dict]:
    """Newest first, no metric rows. This is what the run list polls."""
    if not RUNS.is_dir():
        return []
    out = []
    for p in sorted(RUNS.iterdir()):
        if not p.is_dir():
            continue
        meta = _read_json(p / "run.json")
        cols, rows = _metrics_cached(p / "metrics.csv")
        latest = rows[-1] if rows else {}
        target = meta.get("iterations_target") or 0
        done = int(latest.get("iteration", len(rows)))
        status = {"done": "finished", "stopped": "cancelled"}.get(
            meta.get("status", "unknown"), meta.get("status", "unknown"))
        if status == "running":
            newest = max((f.stat().st_mtime for f in p.rglob("*") if f.is_file()),
                         default=0.0)
            if newest and (datetime.now().timestamp() - newest) > STALE_AFTER_S:
                status = "interrupted"
        out.append({
            "id": p.name,
            "name": meta.get("name") or p.name,
            "variant": p.name.split("_", 2)[-1] if p.name.count("_") >= 2 else p.name,
            "task": meta.get("task", ""),
            "stage": meta.get("stage"),
            "stage_name": meta.get("stage_name", ""),
            "purpose": meta.get("purpose", ""),
            "notes": meta.get("notes", ""),
            "bar": meta.get("bar", ""),
            "status": status,
            "verdict": meta.get("verdict", ""),
            "verdict_detail": meta.get("verdict_detail", ""),
            "started": meta.get("started"),
            "started_age": _age(meta.get("started")),
            "finished": meta.get("finished"),
            "duration": _duration(meta.get("started"), meta.get("finished")),
            "iterations_done": done,
            "iterations_target": target,
            "progress": (done / target) if target else 0.0,
            "metric_columns": cols,
            "samples": len(rows),
            "latest": latest,
            "videos": sum(1 for f in (p / "videos").glob("*.mp4")
                          if not f.name.endswith(".writing.mp4")),
            "checkpoints": sum(1 for _ in (p / "checkpoints").glob("*.npz")),
            "scoring": len(meta.get("scoring", [])),
        })
    out.sort(key=lambda r: (r["started"] or "", r["id"]), reverse=True)
    return out


def detail(run_id: str, points: int = 0) -> dict | None:
    """One run in full, for the panel actually being looked at."""
    folder = RUNS / run_id
    if not folder.is_dir() or ".." in run_id or "/" in run_id or "\\" in run_id:
        return None
    run = read_run(folder)
    run["duration"] = _duration(run["started"], run["finished"])
    if points and len(run["metrics"]) > points:
        run["metrics"] = _thin(run["metrics"], points)
    return run


def _thin(rows: list[dict], keep: int) -> list[dict]:
    """Even sample, always keeping the last row.

    The last row is the current value the page prints beside the chart. Dropping
    it to make the arithmetic neat would make the headline number disagree with
    the end of its own line.
    """
    if keep < 2 or len(rows) <= keep:
        return rows
    step = (len(rows) - 1) / (keep - 1)
    picked = [rows[int(i * step)] for i in range(keep - 1)]
    picked.append(rows[-1])
    return picked


def series(run_ids: list[str], metric: str, points: int = 400) -> list[dict]:
    """The same metric from several runs, for laying them over each other.

    This is the view that answers "is it getting better?", which no single run
    page can. Runs that never recorded the metric come back with an empty list
    rather than being dropped - "this run has no reward curve" is information.
    """
    out = []
    for rid in run_ids:
        folder = RUNS / rid
        if not folder.is_dir():
            continue
        meta = _read_json(folder / "run.json")
        _, rows = _metrics_cached(folder / "metrics.csv")
        pairs = [{"x": r.get("iteration"), "y": r.get(metric)}
                 for r in rows
                 if isinstance(r.get(metric), float) and r.get("iteration") is not None]
        out.append({
            "id": rid,
            "label": rid.split("_", 2)[-1] if rid.count("_") >= 2 else rid,
            "task": meta.get("task", ""),
            "status": meta.get("status", ""),
            "verdict": meta.get("verdict", ""),
            "points": _thin(pairs, points) if points else pairs,
            "n": len(pairs),
        })
    return out


def metrics_available() -> list[str]:
    """Every metric name any run has recorded, so the compare view can offer them."""
    names: set[str] = set()
    if RUNS.is_dir():
        for p in RUNS.iterdir():
            if p.is_dir():
                cols, _ = _metrics_cached(p / "metrics.csv")
                names.update(c for c in cols if c != "iteration")
    return sorted(names)


def active(runs: list[dict]) -> dict | None:
    """The one to show at the top: whatever is running, else the newest."""
    return next((r for r in runs if r["status"] == "running"), runs[0] if runs else None)


def set_verdict(run_id: str, verdict: str, detail: str) -> None:
    """Record what the verifier decided, on the run itself."""
    path = RUNS / run_id / "run.json"
    if not path.exists():
        return
    meta = json.loads(path.read_text())
    meta["verdict"] = verdict
    meta["verdict_detail"] = detail
    path.write_text(json.dumps(meta, indent=2))
