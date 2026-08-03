"""How far the project is from the bars it has to meet.

This is the join nobody else can do. `runs.py` owns what is on disk and imports
nothing but stdlib on purpose. `plan.py` owns the plan and does not know the runs
directory exists. `scripts/verify.py` owns the numeric bars and imports torch.
Answering "how close is R1 to passing" needs all three, so the join lives here,
in its own file, importing lightly.

**Three rules, and they are the whole design.**

1. **No percentages. Anywhere.** Ten of twelve criteria met reads as "83% done"
   for a robot whose sideways drift is 24 times its bar. The page reports three
   separate numbers - met, failing, never measured - and the worst criterion by
   how far over it is. There is no project score, no health index, no gauge.

2. **Distance to bar is a multiple, not a fraction.** `ratio` is normalised so
   1.0 is exactly at the bar and above 1.0 passes, whichever direction the
   criterion runs. A failing criterion is reported as `1 / ratio` times over. It
   makes drift (24x) and speed (1.5x) comparable on one axis, and it does not
   let 4% of the way there look nearly finished.

3. **A measurement is a fact; a verdict is a comparison.** Measurements never
   expire. Verdicts expire the moment the bar moves - which has already happened
   here, since the trunk tolerance is 5 mm for standing, 20 for shoving and 40
   for walking. So `best` is computed from raw measured values ignoring every
   bar, and `passes_current_bar` re-judges it against today's number.

Never-measured is its own category and never a zero. Most subsections have no
verifier at all, and "not measured" and "measured and failed" are opposite
things that a single number would merge.
"""

from __future__ import annotations

from pathlib import Path

from dashboard import plan, runs, skills

ROOT = Path(__file__).resolve().parent.parent

# What each criterion is, keyed by the stable id verify.py writes. Kept here
# rather than in scripts/verify.py so the dashboard can read it without pulling
# in torch - the same reason plan.rewards() imports the task configs lazily.
#
# `bar_field` names the entry in verify.TASKS that holds today's number.
CRITERIA = {
    "survive":    {"name": "stayed up",        "unit": "fraction",
                   "better": "higher", "bar_field": "bar_survive"},
    "height_err": {"name": "trunk height",     "unit": "mm",
                   "better": "lower",  "bar_field": "bar_err_mm"},
    "upright":    {"name": "uprightness",      "unit": "ratio",
                   "better": "higher", "bar_field": "bar_upright"},
    "distance":   {"name": "ground covered",   "unit": "m",
                   "better": "higher", "bar_field": "bar_distance_m"},
    "speed_err":  {"name": "speed tracking",   "unit": "m/s",
                   "better": "lower",  "bar_field": "bar_speed_err"},
    "drift":      {"name": "sideways drift",   "unit": "mm",
                   "better": "lower",  "bar_field": "bar_drift_mm"},
}


def _verify_tasks() -> tuple[dict, str]:
    """verify.TASKS, read live. Degrades to empty rather than blanking the page.

    Imported inside the function: scripts/verify.py pulls in torch at call time
    and this module must stay importable when the training environment is
    broken - which is exactly when someone opens the dashboard.
    """
    try:
        import importlib  # noqa: PLC0415
        import sys  # noqa: PLC0415

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        mod = importlib.import_module("scripts.verify")
        return dict(mod.TASKS), ""
    except Exception as exc:  # noqa: BLE001
        return {}, f"could not read the bars from scripts/verify.py: {exc}"


def current_bars(task: str) -> dict[str, float]:
    """Today's numeric bar per criterion for one task. Empty if it has no verifier."""
    tasks, _ = _verify_tasks()
    spec = tasks.get(task)
    if not spec:
        return {}
    out = {}
    for key, meta in CRITERIA.items():
        value = spec.get(meta["bar_field"])
        if value is not None:
            out[key] = float(value)
    return out


def _ratio(measured: float, bar: float, better: str) -> float | None:
    """1.0 at the bar, above 1.0 passing, in both directions. None if undefined."""
    try:
        if better == "higher":
            return (measured / bar) if bar else None
        return (bar / measured) if measured else None
    except (TypeError, ZeroDivisionError):
        return None


def best_for_task(task: str, summaries: list[dict] | None = None) -> dict:
    """Every criterion for one task, aggregated across every run that measured it.

    `summaries` is injectable so one request can walk progress/runs/ once and
    hand the same list to every consumer.
    """
    if summaries is None:
        summaries = runs.all_summaries()
    bars = current_bars(task)

    mine = [r for r in summaries
            if r.get("task") == task and r.get("verdict_structured")]
    mine.sort(key=lambda r: (r.get("verdict_at") or r.get("started") or "", r["id"]))
    prose_only = [r for r in summaries
                  if r.get("task") == task and r.get("verdict")
                  and not r.get("verdict_structured")]

    criteria = []
    for key, meta in CRITERIA.items():
        points = []
        for run in mine:
            for check in run.get("verdict_checks") or []:
                if check.get("key") != key:
                    continue
                points.append({
                    "run": run["id"],
                    "label": run.get("variant") or run["id"],
                    "at": run.get("verdict_at") or run.get("started") or "",
                    "value": check.get("measured"),
                    "bar": check.get("bar"),
                    "worst": check.get("worst"),
                    "passed": check.get("passed"),
                    "format": check.get("format", "{}"),
                    # A run scored against a task that had moved under it answers
                    # a different question. It still counts, but it is marked.
                    "drifted": bool((run.get("verdict_context") or {}).get("drift")),
                })
        if not points:
            continue

        values = [p["value"] for p in points if isinstance(p["value"], (int, float))]
        if not values:
            continue
        best = max(values) if meta["better"] == "higher" else min(values)
        best_point = next(p for p in points if p["value"] == best)
        latest = points[-1]
        bar = bars.get(key)
        bars_seen = sorted({p["bar"] for p in points if p["bar"] is not None})

        criteria.append({
            "key": key,
            "name": meta["name"],
            "text": latest.get("format", ""),
            "unit": meta["unit"],
            "better": meta["better"],
            "bar": bar,
            "best": best,
            "best_run": best_point["run"],
            "best_label": best_point["label"],
            "latest": latest["value"],
            "latest_run": latest["run"],
            "latest_label": latest["label"],
            "worst_seen": (max if meta["better"] == "lower" else min)(
                [p["worst"] for p in points
                 if isinstance(p["worst"], (int, float))] or [None]),
            "n": len(points),
            "history": points,
            # Judged against TODAY'S bar, from the best measurement ever taken.
            "passes_current_bar": (
                None if bar is None else
                (best >= bar if meta["better"] == "higher" else best <= bar)),
            "ratio": _ratio(best, bar, meta["better"]) if bar is not None else None,
            "latest_ratio": (_ratio(latest["value"], bar, meta["better"])
                             if bar is not None else None),
            "bars_seen": bars_seen,
            # The bar moved between runs, so their verdicts are not comparable
            # even though their measurements are.
            "stale": bar is not None and any(b != bar for b in bars_seen),
            "drifted_runs": sum(1 for p in points if p["drifted"]),
        })

    measured = {c["key"] for c in criteria}
    # "Never measured" and "measured, but only as prose" are different facts and
    # must not share a row. Saying drift was never measured is simply false -
    # three runs measured it - and a page that says so about a number you can
    # read in a log is worse than one that says nothing.
    unmeasured = [{"key": k, "name": CRITERIA[k]["name"], "bar": bars[k],
                   "unit": CRITERIA[k]["unit"],
                   "state": "prose_only" if prose_only else "never"}
                  for k in bars if k not in measured]

    return {
        "task": task,
        "verified_runs": len(mine),
        "prose_only_runs": len(prose_only),
        "criteria": criteria,
        "unmeasured": unmeasured,
        "has_verifier": bool(bars),
    }


def subsection_progress(key: str, summaries: list[dict] | None = None) -> dict:
    """One subsection of the plan, with whatever has actually been measured for it."""
    if summaries is None:
        summaries = runs.all_summaries()
    sub = next((s for s in skills.SUBSECTIONS if s["key"] == key), None)
    if sub is None:
        return {}
    tasks = plan.SUBSECTION_TASKS.get(key, [])

    criteria: list[dict] = []
    unmeasured: list[dict] = []
    verified = 0
    for task in tasks:
        got = best_for_task(task, summaries)
        verified += got["verified_runs"]
        for c in got["criteria"]:
            criteria.append({**c, "task": task})
        for u in got["unmeasured"]:
            unmeasured.append({**u, "task": task})

    met = [c for c in criteria if c["passes_current_bar"] is True]
    failing = [c for c in criteria if c["passes_current_bar"] is False]
    worst = None
    if failing:
        rated = [c for c in failing if c["ratio"]]
        if rated:
            worst = min(rated, key=lambda c: c["ratio"])

    live = next((s for s in (skills.load()["subsections"]) if s["key"] == key), {})
    return {
        "key": key,
        "id": sub["id"],
        "name": sub["name"],
        "kind": sub["kind"],
        "state": sub["state"],
        "rule": sub["rule"],
        "what": sub["what"],
        # How its skill rows are covered, and what it is actually authored
        # against. Two different questions, both on the page.
        "skill_count": live.get("count", 0),
        "coverage": live.get("coverage", {}),
        "blocked_skills": live.get("blocked", []),
        "trains_with": trains_with(key),
        "bar_prose": plan.STAGE2["bars"].get(key, ""),
        "trigger": sub.get("trigger", ""),
        "gap": sub.get("gap", ""),
        "tasks": tasks,
        # No verifier at all. Not zero, not 0% - unmeasurable, and the page must
        # say which. Seven of the nine subsections are in this state.
        "verifiable": bool(tasks),
        "criteria": criteria,
        "unmeasured": unmeasured,
        # THREE numbers, never summed into one and never divided into a fraction.
        "met": len(met),
        "failing": len(failing),
        "never_measured": len(unmeasured) + len(plan.UNVERIFIED_CLAUSES.get(key, [])),
        "unverified_clauses": plan.UNVERIFIED_CLAUSES.get(key, []),
        "verified_runs": verified,
        "worst": worst,
    }


# What stops each subsection being worked on today. Four values and no more -
# the vocabulary IS the panel. "Blocked" as free text turns a decision back into
# a paragraph; four values turn nine subsections into "what could I start now".
BLOCKED_NOTHING = "nothing - could be started today"
BLOCKED_RUN = "another training run"
BLOCKED_HARDWARE = "a measurement that needs the physical robot"
BLOCKED_PARTS = "hardware that has not been bought"

BLOCKERS = {
    "walk": BLOCKED_NOTHING,      # training right now
    "recover": BLOCKED_NOTHING,   # nothing depends on anything else
    "tool": BLOCKED_NOTHING,      # needs scene objects, but that is code
    "robust": BLOCKED_RUN,        # a dial turned during R1
    "quality": BLOCKED_NOTHING,   # a checklist, always available
    "chain": BLOCKED_RUN,         # tests the R1 -> R2 handover; needs both
    "measure": BLOCKED_RUN,       # numbers read off a trained policy
    "flight": BLOCKED_HARDWARE,   # un-parks on the measured servo speed
    "see": BLOCKED_PARTS,         # needs a camera nobody has bought
}


def blockers() -> dict[str, str]:
    return dict(BLOCKERS)


def startable() -> list[dict]:
    """What could be picked up today, with why. Computed, not written down."""
    out = []
    for sub in skills.SUBSECTIONS:
        if BLOCKERS.get(sub["key"]) != BLOCKED_NOTHING:
            continue
        if sub["state"].startswith("in progress"):
            continue
        out.append({"what": f"{sub['id']} {sub['name']}",
                    "why": sub["rule"] + " - nothing blocks it"})
    out.append({
        "what": "The parts session",
        "why": "closes the largest guess in stage 1 and needs no hardware at all - "
               "the slicer already knows the gram weight of every printed part",
    })
    out.append({
        "what": "Stage 3.1 - buy and fit the sensors",
        "why": "the long pole, and step 3.3 measures the three numbers stage 1 is "
               "currently guessing",
    })
    return out


def trains_with(key: str) -> dict:
    """What a subsection is actually AUTHORED against, as opposed to filed under.

    A run has one reward function and a handful of command sliders. Its skill
    rows are the checklist, not the thing being optimised - so the page shows
    both, side by side, because conflating them is what makes 69 rows look like
    69 pieces of work.

    Imported lazily: this pulls in torch. Degrades to "not written yet", which
    is also the honest answer for R2 and R3 - their tasks do not exist.
    """
    task_for = {"walk": "Gray-Walk"}      # only R1 has a task today
    task = task_for.get(key)
    if not task:
        return {"exists": False, "terms": [], "commands": [],
                "note": "no task has been written for this yet"}
    try:
        import importlib  # noqa: PLC0415
        import sys  # noqa: PLC0415

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        mod = importlib.import_module("gray.tasks.walk_env_cfg")
        cfg = mod.walk_env_cfg()
    except Exception as exc:  # noqa: BLE001
        return {"exists": False, "terms": [], "commands": [],
                "note": f"could not read the task: {exc}"}

    terms = sorted(
        ({"name": n, "weight": float(t.weight),
          "sign": "+" if t.weight > 0 else "-"} for n, t in cfg.rewards.items()),
        key=lambda t: (t["sign"] != "+", -abs(t["weight"])))
    cmd = cfg.commands.get("walk")
    commands = []
    if cmd is not None:
        commands = [
            {"name": "forward", "range": list(cmd.ranges.lin_vel_x), "unit": "m/s"},
            {"name": "sideways", "range": list(cmd.ranges.lin_vel_y), "unit": "m/s"},
            {"name": "turn", "range": list(cmd.ranges.ang_vel_z), "unit": "rad/s"},
        ]
    return {
        "exists": True, "task": task, "terms": terms, "commands": commands,
        "paid": sum(1 for t in terms if t["sign"] == "+"),
        "fined": sum(1 for t in terms if t["sign"] == "-"),
        "note": "",
    }


def overview(summaries: list[dict] | None = None) -> dict:
    """The whole project against its bars. One call, no disk reads of its own."""
    if summaries is None:
        summaries = runs.all_summaries()
    _, error = _verify_tasks()

    subs = [subsection_progress(s["key"], summaries) for s in skills.SUBSECTIONS]

    verified = [r for r in summaries if r.get("verdict")]
    structured = [r for r in verified if r.get("verdict_structured")]

    # The three-second answer: the failing criterion that is furthest over its
    # bar, anywhere. One criterion, one number, one multiple.
    headline = None
    for sub in subs:
        w = sub.get("worst")
        if not w:
            continue
        if headline is None or w["ratio"] < headline["ratio"]:
            headline = {**w, "sub": sub["id"], "sub_name": sub["name"]}
    if headline:
        headline["times_over"] = (1.0 / headline["ratio"]) if headline["ratio"] else None

    # Blind spots, grouped by WHAT WOULD FIX THEM. An earlier version listed
    # every gap as its own row and printed the same fact eight times - once per
    # subsection with no verifier - while also claiming criteria were "never
    # measured" that three runs had measured, just not in a comparable format.
    # Grouped by remedy it is four lines instead of twenty-six, and each one
    # says what to do.
    no_verifier = [s["id"] for s in subs if not s["verifiable"]]
    prose_only = sorted({u["name"] for s in subs for u in s["unmeasured"]
                         if u.get("state") == "prose_only"})
    never = sorted({u["name"] for s in subs for u in s["unmeasured"]
                    if u.get("state") != "prose_only"})
    clauses = [c for s in subs for c in s["unverified_clauses"]]
    blind = []
    if prose_only:
        blind.append({
            "what": f"{len(prose_only)} criteria measured, but not comparably",
            "detail": ", ".join(prose_only),
            "why": "these runs were verified before the structured format, so the "
                   "numbers exist in the job logs but cannot be charted or compared",
            "fix": "re-run scripts/verify.py on those runs - the checkpoints are "
                   "still on disk and it takes about two minutes each",
        })
    if never:
        blind.append({
            "what": f"{len(never)} criteria never measured",
            "detail": ", ".join(never),
            "why": "the bar defines them and no run has produced the number",
            "fix": "finish a run with the current verifier",
        })
    if clauses:
        blind.append({
            "what": f"{len(clauses)} clauses of the bar nothing can ask for",
            "detail": "; ".join(c.split(" - ")[0] for c in clauses),
            "why": "the command vector has no height, pitch or roll, and nothing "
                   "per foot - so these cannot be commanded, let alone measured",
            "fix": "widen the command vector, which changes the observation and "
                   "forces a retrain",
        })
    if no_verifier:
        blind.append({
            "what": f"{len(no_verifier)} subsections have no verifier at all",
            "detail": ", ".join(no_verifier),
            "why": "scripts/verify.py only knows Stand, Push and Walk. For the "
                   "rest there is no bar to measure against, so their progress is "
                   "genuinely unknown rather than zero",
            "fix": "write a verifier for R2 first - it is the next run",
        })
    blind.append({
        "what": "3 numbers in the model have no datasheet",
        "detail": "servo stiffness and speed under load, backlash, loop latency",
        "why": "nobody has measured them and no supplier publishes them, so every "
               "run is trained against an estimate",
        "fix": "stage 3.3, on the physical robot. Until then they are randomised "
               "rather than guessed, which is the correct handling",
    })

    return {
        "subsections": subs,
        "headline": headline,
        "blind_spots": blind,
        "runs_total": len(summaries),
        "runs_verified": len(verified),
        "runs_structured": len(structured),
        "runs_passed": sum(1 for r in verified if r["verdict"] == "passed"),
        # Stated so the page can be honest about its own denominator: most run
        # folders are smoke tests and cancellations, and reading 25 folders as 25
        # attempts would overstate everything.
        "bars_source": "scripts/verify.py TASKS, read live",
        "error": error,
    }
