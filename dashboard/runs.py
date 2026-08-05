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
import re
from datetime import datetime
from math import isfinite
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
                f = float(v)
            except (TypeError, ValueError):
                out[k] = v
                continue
            # NaN and inf are valid floats and invalid JSON. json.dumps writes
            # them as bare NaN / Infinity, which JSON.parse rejects outright - so
            # ONE diverged reward turns the whole dashboard into "Could not
            # load". A diverged RL run logs NaN as a matter of course, so this is
            # the normal case for exactly the run you most want to look at.
            # Dropped to None here, at the only place metrics enter the system.
            out[k] = f if isfinite(f) else None
        clean.append(out)
    # Scan EVERY row for the numeric columns, not just the first. train.py writes
    # "" for a metric missing from a row, and a metric absent from row 0 would
    # otherwise never appear in metric_columns - so it would get no chart for the
    # whole run, silently.
    cols: list[str] = []
    for row in clean:
        for k, v in row.items():
            if isinstance(v, float) and k not in cols:
                cols.append(k)
    return cols, clean


def _iterations_done(latest: dict, rows: list) -> int:
    """How far the run got, from its last metrics row. The row count if it cannot say.

    `iteration` comes straight out of metrics.csv, and _read_metrics above turns
    a NaN into None and keeps an unparseable cell as a string. int() raises on
    both. all_summaries() is called by nav_state(), which every endpoint calls,
    so one torn last row in one run's csv answered EVERY request with a 500.
    """
    try:
        return int(float(latest["iteration"]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return len(rows)


def _newest_mtime(folder: Path) -> float:
    """When anything in this run was last written, 0.0 if nothing could be read.

    Every stat is guarded individually. rglob lists names, and a file can be
    renamed or removed between being listed and being stat'd - which is exactly
    what filming does every time it finishes one (`.writing.mp4` -> `.mp4`), and
    what the checkpoint marker writer does. An unguarded FileNotFoundError here
    escapes the request handler, and socketserver closes the connection with no
    response at all: the whole dashboard goes to "Could not load", and only ever
    while a run is training.
    """
    newest = 0.0
    try:
        entries = list(folder.rglob("*"))
    except OSError:
        return 0.0
    for f in entries:
        try:
            if f.is_file():
                newest = max(newest, f.stat().st_mtime)
        except OSError:
            continue        # it went away mid-scan; it cannot be the newest
    return newest


def _iteration_of(name: str) -> int | None:
    """The iteration a checkpoint film was taken at, or None if it is not one.

    Only `iter_NNNN` counts. This used to scrape every digit out of the stem,
    which is fine while every file is called iter_0150 and wrong the moment one
    is not: scripts/drive.py writes clips named for the command that made them,
    and `drive_1_straight` was read as iteration 1, then sorted to the bottom of
    a sixty-six clip strip and labelled "iteration 1". Present, and impossible
    to find. A file that does not follow the convention has no iteration.
    """
    m = re.fullmatch(r"iter_(\d+)", Path(name).stem)
    return int(m.group(1)) if m else None


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
    done = _iterations_done(latest, rows)

    # A run that says it is running but has not written for a while was killed
    # without getting the chance to say so.
    # Runs made before the vocabulary was settled say "done" and "stopped".
    status = {"done": "finished", "stopped": "cancelled"}.get(
        meta.get("status", "unknown"), meta.get("status", "unknown"))
    if status == "running":
        newest = _newest_mtime(folder)
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
        "num_envs": meta.get("num_envs"),
        "num_steps_per_env": meta.get("num_steps_per_env"),
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
        "verdict_checks": meta.get("verdict_checks", []),
        "verdict_structured": bool(meta.get("verdict_checks")),
        "verdict_at": meta.get("verdict_at", ""),
        "verdict_context": meta.get("verdict_context", {}),
        "notes": meta.get("notes", ""),
        # What this run was scored on, recorded at launch. Kept per-run rather
        # than read from the task, so an old run still says what IT was scored on
        # after the rewards have been changed.
        "scoring": meta.get("scoring", []),
        # The three things `scoring` alone cannot say, all recorded by train.py
        # and none of them reaching a page until 4 Aug 2026:
        #
        #   ramps       a curriculum term's weight CLIMBS. `scoring` holds its
        #               stage-0 value, so the page showed twitching at -0.05
        #               while the run trained it to -0.25, and veering at -0.2
        #               against the -2.0 it reached by iteration 500. A weight
        #               five times weaker than the real one, printed as fact.
        #   tolerances  the std inside a term decides what it MEANS, not what it
        #               is worth. track_turn at std 0.80 scored a 0.018 rad/s
        #               bias at 99.95% of full marks - invisible in a weights
        #               table, and the whole of round 0's straightness failure.
        #   observes    which inputs the policy read. It decides whether a
        #               checkpoint can even be loaded, and it changed twice in
        #               two days.
        "ramps": meta.get("ramps", []),
        "tolerances": meta.get("tolerances", {}),
        "observes": meta.get("observes", []),
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
        done = _iterations_done(latest, rows)
        status = {"done": "finished", "stopped": "cancelled"}.get(
            meta.get("status", "unknown"), meta.get("status", "unknown"))
        if status == "running":
            newest = _newest_mtime(p)
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
            # The structured result, so the overview can compare criteria across
            # runs without re-reading every run.json. Six rows per run - a few KB
            # across the whole list, against the metric arrays this call already
            # refuses to carry.
            "verdict_checks": meta.get("verdict_checks", []),
            "verdict_structured": bool(meta.get("verdict_checks")),
            "verdict_at": meta.get("verdict_at", ""),
            "verdict_context": meta.get("verdict_context", {}),
            "started": meta.get("started"),
            "started_age": _age(meta.get("started")),
            "finished": meta.get("finished"),
            "duration": _duration(meta.get("started"), meta.get("finished")),
            "iterations_done": done,
            "iterations_target": target,
        # Which draw of the dice this run got. In the summary as well as the
        # detail because "is that difference real or is it the seed" is asked of
        # a LIST of runs, and re-reading every run.json to answer it is what the
        # summary exists to avoid.
        "seed": meta.get("seed"),
        "num_envs": meta.get("num_envs"),
        "num_steps_per_env": meta.get("num_steps_per_env"),
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
    # A sequential number, oldest = 1. Names are not unique and never were - a
    # cancelled-and-requeued job keeps its name, so the list can show four runs
    # called walk_m3100_h with nothing to tell them apart but a relative
    # timestamp. The number is what a person can actually say out loud.
    total = len(out)
    for i, run in enumerate(out):
        run["number"] = total - i
    return out


def detail(run_id: str, points: int = 0) -> dict | None:
    """One run in full, for the panel actually being looked at."""
    # An empty id makes `RUNS / ""` resolve to RUNS itself, which is a directory,
    # contains no ".." or slash, and so passed every check - returning a
    # fabricated 200 for a run called "runs" instead of a 404.
    if not run_id or ".." in run_id or "/" in run_id or "\\" in run_id:
        return None
    folder = RUNS / run_id
    if not folder.is_dir():
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


# Bumped when the shape of verdict_checks changes, so a reader can tell rather
# than guess. Absent means "prose only" - every run verified before 3 Aug 2026.
VERDICT_SCHEMA = 1


def set_verdict(run_id: str, verdict: str, detail: str,
                checks: list[dict] | None = None,
                context: dict | None = None) -> None:
    """Record what the verifier decided, on the run itself.

    `detail` is the prose line and is written exactly as before. `checks` is the
    same result with its structure intact - one row per criterion, with the
    measured value, the bar, the direction and the verdict as separate fields.
    The prose cannot be compared across runs or charted; the structure can.

    The three-argument call still works, so nothing that used it breaks.
    """
    path = RUNS / run_id / "run.json"
    if not path.exists():
        # verify.py passes the MJLAB log directory name, which matches the
        # progress run id only because both are stamped in the same second. Say
        # so rather than returning silently - a verdict that vanishes with no
        # error is indistinguishable from a verifier that never ran.
        print(f"[verdict] no run.json at {path} - verdict NOT recorded. "
              f"The mjlab log dir and the progress run id have diverged.")
        return
    meta = json.loads(path.read_text())
    meta["verdict"] = verdict
    meta["verdict_detail"] = detail
    if checks is not None:
        meta["verdict_checks"] = checks
        meta["verdict_schema"] = VERDICT_SCHEMA
        meta["verdict_at"] = datetime.now().isoformat(timespec="seconds")
    if context is not None:
        meta["verdict_context"] = context
    path.write_text(json.dumps(meta, indent=2))


def verdict_of(meta: dict) -> dict:
    """One run's verdict in a fixed shape, old format or new.

    The single place the old/new fork is handled. Every page reads this rather
    than testing for the field itself, so "this run predates structured
    verdicts" is expressed once.

    Deliberately does NOT parse verdict_detail. It looks parseable and it lies:
    "stayed up for 30 s: 100%" carries no bar at all, and "covered 5 m: 6.03 m"
    hides the bar inside the name, which changes whenever the bar does. A regex
    over that yields a page that is confidently wrong.
    """
    checks = meta.get("verdict_checks")
    return {
        "verdict": meta.get("verdict", ""),
        "detail": meta.get("verdict_detail", ""),
        "at": meta.get("verdict_at", ""),
        "schema": meta.get("verdict_schema", 0),
        "structured": bool(checks),
        "checks": checks or [],
        "context": meta.get("verdict_context", {}),
    }
