"""What the run training right now is actually doing, distilled.

The Runs page prints forty-four metrics per run. Every one is correct and not one
of them is a conclusion. This module turns them into the four things a person
actually wants while a run is in flight:

1. **Is it on track to pass the bar?** - the same criteria scripts/verify.py
   scores, estimated live off metrics.csv. Eleven of them since 4 Aug 2026;
   six before that. The count lives in progress.CRITERIA and is deliberately
   not written down here as a number - it was, and that number went stale the
   day crab and spin were added, so this file's panel reported "6 met, 0
   failing" on runs whose verdict was NOT PASSED.
2. **Why did the last one fail?** - the newest recorded verdict, failures only.
3. **What is it doing right now?** - eight plain-English readings.
4. **Where do the points go?** - the reward, split into the terms that earned it.

**The one rule this file exists to enforce: a live estimate is not a verdict.**

verify.py runs AFTER training, on a checkpoint, on a fixed test. Nothing here can
do that. What is below reads the training logs, which measure something adjacent -
averaged over the whole command box rather than a single 0.25 m/s test, over 20 s
episodes rather than 25, over 4500 robots rather than 64. So every row carries a
`source`, and the page prints it:

    live         estimated from training as it runs. Indicative, not a verdict.
    verified     verify.py measured it on a checkpoint. This is the real thing.
    unmeasured   nothing logs it during training. Says so; borrows nothing.

Seven of the eleven walk criteria are `unmeasured` - ground covered and sideways
drift, plus all five of the crab and spin ones, which training logs nothing for
at all: a training episode draws a command from the whole box, so there is no
"the crab pass" or "the turn pass" to read a number off.
Both quantities exist inside gray/tasks/walk_env_cfg.py (`ground_covered` and
`wandering` compute them and throw them away after scoring), so this is a missing
log line and not a missing measurement. Until it is added, those rows are blank
and show the last VERIFIED number instead. They are not estimated from a nearby
metric: drift is the criterion the last three walk runs failed, and a guessed
drift figure is worse than an empty one.

The thresholds in READS are editorial - one engineer's line between "fine" and
"watch this". They are NOT bars. The bars are in scripts/verify.py and only ever
appear in `bar_board`.
"""

from __future__ import annotations

import math

from dashboard import progress, runs

# ============================================================== the bar board ===

# How each of verify.py's criteria can be read off a training run, live.
#
# `column` is what to look for in metrics.csv. `read` converts the row into the
# criterion's own unit - the unit progress.CRITERIA declares, so the number can
# sit beside a verified one without conversion. `caveat` is printed on the row:
# every one of these measures something ADJACENT to what verify.py measures, and
# the difference is the reader's to weigh, not this file's to hide.
#
# A criterion absent from this dict is absent on purpose. See the module
# docstring - `distance` and `drift` are not estimated from anything.


def _survive(row: dict) -> float | None:
    """Fraction of episodes that ended by running out of time rather than falling.

    Episode_Termination/* are mean counts per logging window, so the ratio between
    them is the useful quantity and the absolute numbers are not. Falling is
    tipped_over (the trunk went over) plus collapsed (it sank onto its belly);
    everything else that ends an episode is the clock.
    """
    out = row.get("Episode_Termination/time_out")
    fell = row.get("tipped_over")
    sank = row.get("collapsed") or 0.0
    if not isinstance(out, (int, float)) or not isinstance(fell, (int, float)):
        return None
    total = out + fell + sank
    return (out / total) if total > 0 else None


def _upright(row: dict) -> float | None:
    """Uprightness, from the posture command's own pitch and roll errors.

    verify.py measures `-projected_gravity_b[:, 2]`, which IS the cosine of the
    angle between the trunk's own up and the world's. Pitch and roll errors are
    that same tilt resolved onto two axes, so the cosine of their magnitude is the
    same number - to within the small-angle approximation, which at the 0.12 rad
    seen in practice is worth about 0.0001.
    """
    pitch = row.get("Metrics/posture/error_pitch")
    roll = row.get("Metrics/posture/error_roll")
    if not isinstance(pitch, (int, float)) or not isinstance(roll, (int, float)):
        return None
    return math.cos(min(math.pi / 2, math.hypot(pitch, roll)))


def _height_mm(row: dict) -> float | None:
    metres = row.get("Metrics/posture/error_height")
    return metres * 1000 if isinstance(metres, (int, float)) else None


def _speed_err(row: dict) -> float | None:
    value = row.get("speed_error")
    return value if isinstance(value, (int, float)) else None


# `judge` says whether the live estimate may be marked passing or failing against
# the bar. It is not a confidence rating - it is a measured fact, and these are
# the measurements, taken by reading the last metrics row of every run that also
# carries a verdict and setting the two side by side:
#
#   criterion    live -> verified          on runs #25, #34, #36
#   survive      1.00 -> 1.00              exact
#   height_err   11.0 -> 6.27  (1.75x)     pessimistic, and 6x inside the bar
#   upright     0.992 -> 0.998             pessimistic by 0.006
#   speed_err   0.112 -> 0.029  (3.9x)     pessimistic by 1.9x to 3.9x - AND THE
#                                          BAR SITS BETWEEN THE TWO
#
# The first three err toward failing, by margins far smaller than their distance
# to the bar, so a live pass is a real pass. Speed does not: 0.113 against a 0.05
# bar reads as 2.3x over, and verify.py then measured 0.031 and PASSED it. Marking
# that row failing would have raised a false alarm on all three runs. It is
# reported and not judged - which is the honest answer, and it is why this flag
# exists rather than a comment saying "roughly".
LIVE_PROXY = {
    "survive": {
        "read": _survive, "judge": True,
        "caveat": "share of episodes ending on the clock rather than on the floor, "
                  "not the share of robots that survived a fixed 25 s test",
    },
    "height_err": {
        "read": _height_mm, "judge": True,
        "caveat": "against the height it was TOLD to hold, which moves; "
                  "verify.py measures against one fixed target. Reads about 1.75x "
                  "high, and the bar is six times away",
    },
    "upright": {
        "read": _upright, "judge": True,
        "caveat": "from the mean pitch and roll error, small-angle",
    },
    "speed_err": {
        "read": _speed_err, "judge": False,
        "why": "the bar is one steady 0.25 m/s run; this is averaged over the whole "
               "command box, turns and stops included, and measures 2x to 4x higher. "
               "#36 read 0.113 here and verify.py then measured 0.031 and passed it",
        "caveat": "over the whole command box, not the 0.25 m/s test",
    },
}


def bar_board(run: dict, summaries: list[dict] | None = None) -> dict:
    """The six criteria for this run's task: live where possible, honest where not.

    Returns three counts and a row per criterion. The counts are kept separate and
    are never averaged - progress.py's rule 1, for the same reason: a single
    percentage lets a criterion fourteen times over its bar disappear into it.
    """
    task = run.get("task") or ""
    bars = progress.current_bars(task)
    if not bars:
        return {"task": task, "has_verifier": False, "rows": [],
                "met": 0, "failing": 0, "unknown": 0}

    if summaries is None:
        summaries = runs.all_summaries()
    history = progress.best_for_task(task, summaries)
    verified = {c["key"]: c for c in history["criteria"]}
    # THIS RUN'S OWN verdict, if it has been verified. It outranks everything
    # else on the row: it is the criterion measured the way the bar defines it,
    # on this run's own checkpoint. The live estimate stays on the row as the
    # second reading, so the card reads the same whether you are watching a run
    # go or looking at one from last week - the numbers just get better.
    own = {c.get("key"): c for c in (run.get("verdict_checks") or [])}

    latest = (run.get("metrics") or [{}])[-1] if run.get("metrics") else (
        run.get("latest") or {})

    rows = []
    for key, meta in progress.CRITERIA.items():
        bar = bars.get(key)
        if bar is None:
            continue  # this task has no such criterion (standing does not walk)

        proxy = LIVE_PROXY.get(key)
        live_value = proxy["read"](latest) if proxy else None
        was = proxy["read"]((run.get("metrics") or [{}])[0]) if (
            proxy and run.get("metrics")) else None
        live_ratio = (progress._ratio(live_value, bar, meta["better"])
                      if live_value is not None else None)

        mine = own.get(key)
        seen = verified.get(key)
        row = {
            "key": key,
            "name": meta["name"],
            "better": meta["better"],
            "at_start": was,
            "caveat": proxy["caveat"] if proxy else "",
            "why_unjudged": (proxy or {}).get("why", ""),
            # What the verifier last measured for this criterion on ANY run of
            # this task. Used for the rows nothing logs during training.
            "last": None if not seen else {
                "value": seen["latest"],
                "run": seen["latest_run"],
                "label": seen["latest_label"],
                "ratio": seen["latest_ratio"],
                "worst": seen["worst_seen"],
                "format": seen["text"],
                "n": seen["n"],
            },
        }

        if mine:
            # The bar this run was scored against, and the unit it was scored in
            # - both off the verdict, not off today's config. The walk drift bar
            # changed from millimetres to degrees on 4 Aug 2026, so a run's own
            # number and today's bar are not always in the same unit, and reading
            # the unit from anywhere but the run's own row prints 2115 mm as
            # 2115 degrees.
            row.update({
                "unit": mine.get("unit") or meta["unit"],
                "bar": mine.get("bar", bar),
                "measured": mine.get("measured"),
                "ratio": mine.get("ratio"),
                "passed": mine.get("passed"),
                "source": "verified",
                "judged": True,
                "note": mine.get("note", ""),
                "worst": mine.get("worst"),
                # The training estimate, behind the verdict, as the ghost.
                "other": None if live_value is None else {
                    "value": live_value, "ratio": live_ratio,
                    "what": "what training reported at the end of the run",
                },
            })
        else:
            row.update({
                "unit": meta["unit"],
                "bar": bar,
                "measured": live_value,
                # progress._ratio is the single implementation of "1.0 at the
                # bar, above 1.0 passing, in both directions". Duplicating it
                # here would be a second place for the sign convention to be got
                # wrong.
                "ratio": live_ratio,
                "source": "live" if live_value is not None else "unmeasured",
                # A live number that may not be judged is still worth printing -
                # it is the only reading there is. It just does not get a tick.
                "judged": bool(proxy and proxy.get("judge") and live_value is not None),
                "note": "",
                "worst": None,
                "other": None if not seen or seen["latest_ratio"] is None else {
                    "value": seen["latest"], "ratio": seen["latest_ratio"],
                    "what": f"last verified, on {seen['latest_label']}",
                },
            })
            # A ratio of None means the comparison has no answer. progress._ratio
            # returns it for a lower-is-better criterion measured at exactly 0.0,
            # because the bar is divided by that measurement. Comparing None to
            # 1.0 raises, and this function feeds both /api/live and
            # /api/run/<id>, so one zero in one metrics row took the Control page
            # and the Runs page down together.
            #
            # It is not scored as a pass. A drift of exactly 0.000 is a hole in
            # the log far more often than it is a perfect run, and this file's
            # rule is that a live estimate is never a verdict.
            if row["ratio"] is None:
                row["judged"] = False
            row["passed"] = (row["ratio"] >= 1.0) if row["judged"] else None
        rows.append(row)

    return {
        "task": task,
        "has_verifier": True,
        "rows": rows,
        "met": sum(1 for r in rows if r["passed"] is True),
        "failing": sum(1 for r in rows if r["passed"] is False),
        "unknown": sum(1 for r in rows if r["passed"] is None),
        # The two ways of not knowing, kept apart. "Nothing measures this" and
        # "this is measured but is not comparable to the bar" are different facts
        # and a reader can act on the second one.
        "reported": sum(1 for r in rows
                        if r["passed"] is None and r["source"] == "live"),
        "unmeasured": sum(1 for r in rows if r["source"] == "unmeasured"),
    }


# ============================================================= the last verdict ===


def last_verdict(summaries: list[dict], skip: str = "") -> dict | None:
    """The newest run that has a verdict, and what it failed on.

    `skip` drops the run being looked at, so a finished run does not report itself
    as "the last one". Only failures get a row; passing criteria collapse to a
    count, because a list of six ticks is not why anyone opened the page.
    """
    for summary in summaries:
        if summary["id"] == skip or not summary.get("verdict"):
            continue
        result = runs.verdict_of(summary)
        checks = result["checks"]
        failed = [c for c in checks if not c.get("passed")]
        return {
            "id": summary["id"],
            "number": summary.get("number"),
            "label": summary.get("variant") or summary["id"],
            "task": summary.get("task", ""),
            "verdict": result["verdict"],
            "age": summary.get("started_age", ""),
            "at": result["at"],
            "structured": result["structured"],
            # Kept verbatim for the prose-only case. Deliberately NOT parsed:
            # runs.verdict_of's docstring sets out why a regex over this string
            # produces a page that is confidently wrong.
            "detail": result["detail"],
            "passed_count": sum(1 for c in checks if c.get("passed")),
            "total": len(checks),
            "failed": [{
                "key": c.get("key", ""),
                "name": c.get("name", ""),
                "measured": c.get("measured"),
                "bar": c.get("bar"),
                "worst": c.get("worst"),
                "unit": c.get("unit", ""),
                "format": c.get("format", "{}"),
                "note": c.get("note", ""),
                "ratio": c.get("ratio"),
            } for c in failed],
            # The verdict was measured against a task that had since moved.
            "drifted": bool((result["context"] or {}).get("drift")),
        }
    return None


# ================================================================== the readings ===

# Eight plain-English conclusions. `metric` is the column judged; `warn` and `bad`
# are where this file draws the line, and they are EDITORIAL - see the module
# docstring. A reading whose metric the run never logged is dropped rather than
# printed as zero: an older run with sixteen columns should show four readings,
# not eight readings four of which are lies.
#
# `evidence` is the numbers behind the claim, printed on the same line. The claim
# is never the only thing on the row - colour and a word are not enough on their
# own, and a reader who disagrees with a threshold can see what it was drawn from.

READS = [
    {"id": "up", "title": "staying up", "metric": "tipped_over", "better": "lower",
     "warn": 0.02, "bad": 0.15,
     "says": {"good": "nothing is tipping over",
              "warn": "a few are going over",
              "bad": "they are falling over"},
     "evidence": ["tipped_over", "collapsed", "episode_length"]},

    {"id": "level", "title": "staying level", "metric": "upright_reward",
     "better": "higher", "warn": 0.80, "bad": 0.60,
     "says": {"good": "the trunk is holding level",
              "warn": "the trunk is leaning",
              "bad": "the trunk is not level"},
     "evidence": ["upright_reward", "Metrics/posture/error_roll",
                  "Metrics/posture/error_pitch"]},

    {"id": "moving", "title": "going somewhere", "metric": "ground_covered",
     "better": "higher", "warn": 0.50, "bad": 0.20,
     "says": {"good": "putting real ground behind it",
              "warn": "moving, but not much",
              "bad": "not getting anywhere"},
     "evidence": ["ground_covered", "Episode_Reward/track_speed"]},

    # NOT drawn at the 0.05 m/s bar, on purpose. This column is measured over the
    # whole command box and reads two to four times the figure verify.py gets on
    # its steady test - #36 read 0.113 here and verified at 0.031. Drawing the
    # line at the bar would paint this row red on a run that passes. 0.15 and 0.30
    # are editorial, off what runs that went on to pass actually read.
    {"id": "speed", "title": "obeying speed", "metric": "speed_error",
     "better": "lower", "warn": 0.15, "bad": 0.30,
     "says": {"good": "tracking the commanded speed across the box",
              "warn": "off the commanded speed",
              "bad": "not tracking the speed at all"},
     "evidence": ["speed_error", "turn_error"]},

    {"id": "gait", "title": "walking, not shuffling", "metric": "stepping",
     "better": "higher", "warn": 0.15, "bad": 0.05,
     "says": {"good": "real steps, feet leaving the floor",
              "warn": "short, shuffling steps",
              "bad": "the feet are barely lifting"},
     "evidence": ["stepping", "leg_swing"]},

    {"id": "feet", "title": "feet clearing the floor", "metric": "dragging",
     "better": "higher", "warn": -0.05, "bad": -0.15,
     "says": {"good": "not dragging or scuffing",
              "warn": "a foot is scuffing",
              "bad": "it is dragging a leg"},
     "evidence": ["dragging", "Episode_Reward/skidding",
                  "Episode_Reward/swing_height"]},

    {"id": "servos", "title": "kind to the servos", "metric": "Episode_Reward/effort",
     "better": "higher", "warn": -0.30, "bad": -0.80,
     "says": {"good": "torque and jitter within budget",
              "warn": "working the servos hard",
              "bad": "this gait would cook the servos"},
     "evidence": ["Episode_Reward/effort", "Episode_Reward/jitter",
                  "Episode_Reward/joint_shock", "Episode_Reward/end_stops"]},
]


def _state(value: float, better: str, warn: float, bad: float) -> str:
    if better == "lower":
        return "bad" if value > bad else "warn" if value > warn else "good"
    return "bad" if value < bad else "warn" if value < warn else "good"


def readings(run: dict) -> list[dict]:
    rows_all = run.get("metrics") or []
    if not rows_all:
        return []
    now, first = rows_all[-1], rows_all[0]

    out = []
    for spec in READS:
        value = now.get(spec["metric"])
        if not isinstance(value, (int, float)):
            continue
        state = _state(value, spec["better"], spec["warn"], spec["bad"])
        out.append({
            "id": spec["id"],
            "title": spec["title"],
            "state": state,
            "says": spec["says"][state],
            "evidence": [{"key": k, "value": now.get(k), "was": first.get(k)}
                         for k in spec["evidence"]
                         if isinstance(now.get(k), (int, float))],
        })

    # Whether it is still learning is not a threshold on one column, it is a
    # trend on the score - so it is computed rather than declared, and it goes
    # last because it is the row that says whether the rest will change.
    trend = learning(rows_all)
    if trend:
        out.append(trend)
    return out


def learning(rows: list[dict]) -> dict | None:
    """Is the score still climbing, or has it settled?

    Compared over the last fifth of the run rather than against the start: every
    run climbs steeply out of iteration zero, so "up 129 since the start" is true
    of a run that stopped improving an hour ago.
    """
    scores = [(r.get("iteration"), r.get("reward")) for r in rows
              if isinstance(r.get("reward"), (int, float))]
    if len(scores) < 10:
        return None
    window = max(2, len(scores) // 5)
    recent, before = scores[-1][1], scores[-window][1]
    span = (scores[-1][0] or 0) - (scores[-window][0] or 0)
    gained = recent - before
    # Relative to the size of the number, so it reads the same at reward 6 and at
    # reward 130. Two per cent over the last fifth is the line.
    rate = (gained / abs(before)) if before else 0.0
    state = "good" if rate > 0.02 else "warn" if rate > -0.02 else "bad"
    says = ("still climbing" if state == "good"
            else "settled - the score has stopped moving" if state == "warn"
            else "the score is going backwards")
    return {
        "id": "learning", "title": "still learning", "state": state, "says": says,
        "evidence": [
            {"key": "reward", "value": recent, "was": scores[0][1]},
            {"key": "gained over %d iters" % span, "value": gained, "was": None},
        ],
    }


# ============================================================== where points go ===

# scripts/train.py's WATCH map renames some Episode_Reward tags on the way into
# metrics.csv, so a reward term's column is not always "Episode_Reward/<name>".
# These are those renames, from train.py:128-173. Without them the six biggest
# earning terms in a walk run all read as missing.
COLUMN_ALIAS = {
    "ride_height": "height_reward",
    "height": "height_reward",
    "upright": "upright_reward",
    "tilt": "tilt_penalty",
    "stepping": "stepping",
    "leg_swing": "leg_swing",
    "dragging": "dragging",
    "ground_covered": "ground_covered",
}


def _column(term: str) -> str:
    return COLUMN_ALIAS.get(term, f"Episode_Reward/{term}")


def budget(run: dict) -> dict:
    """The reward, split into the terms that made it. Points per second.

    Episode_Reward/<term> is that term's contribution in points per second, so the
    terms sum to the reward divided by the episode length - which is the check at
    the bottom of this function, and it is worth having: if the sum stops matching
    the score, a term has been renamed and this page is quietly missing it.

    Each reward is capped at 1.0 before its weight, so a positive term's ceiling
    is its weight and `headroom` is what is left on the table. Penalties have no
    such ceiling - that is the point of them - so they carry none.
    """
    terms = run.get("scoring") or []
    rows_all = run.get("metrics") or []
    if not terms or not rows_all:
        return {"earning": [], "leaking": [], "total": None}
    now, first = rows_all[-1], rows_all[0]

    earning, leaking = [], []
    for term in terms:
        value = now.get(_column(term["name"]))
        if not isinstance(value, (int, float)):
            continue
        row = {
            "name": term["name"],
            "what": term.get("what", ""),
            "weight": term["weight"],
            "value": value,
            "was": first.get(_column(term["name"])),
        }
        if term["weight"] > 0:
            row["cap"] = term["weight"]
            row["headroom"] = max(0.0, term["weight"] - value)
            earning.append(row)
        else:
            leaking.append(row)

    earning.sort(key=lambda r: -r["value"])
    leaking.sort(key=lambda r: r["value"])

    earned = sum(r["value"] for r in earning)
    lost = sum(r["value"] for r in leaking)
    ceiling = sum(t["weight"] for t in terms if t["weight"] > 0)
    score = now.get("reward")
    # The episode length the score implies. Not read from anywhere - it is what
    # the two numbers say between them, and it is the check that the split above
    # accounts for the whole score.
    seconds = (score / (earned + lost)) if (score and (earned + lost)) else None
    return {
        "earning": earning,
        "leaking": leaking,
        "earned": earned,
        "lost": lost,
        "net": earned + lost,
        "ceiling": ceiling,
        "reward": score,
        "seconds": seconds,
        "accounted": (seconds is not None and 5.0 < seconds < 120.0),
    }


# ===================================================================== assembly ===


def readout(detail: dict, summaries: list[dict] | None = None) -> dict:
    """The on-track card for ONE run, from a detail dict already in hand.

    Split out of `live_state` so the Runs page can put the same card on whichever
    run is open - the question "did this run pass, and if not what did it fail
    on" is the same question there as it is on the Control page, and the answer
    should not be assembled twice.

    Takes the detail rather than an id on purpose: /api/run/<id> has already read
    it off disk, and re-reading a 1 MB metrics.csv to draw one card is the cost
    that made /api/monitor refuse to carry metric rows at all.
    """
    if summaries is None:
        summaries = runs.all_summaries()
    # `number` and `duration` are computed in all_summaries(), not in read_run(),
    # so a run fetched by id has neither - which is why the Runs page's own
    # header prints a run's name with no number in front of it while the sidebar
    # two inches to its left prints "#47". Stamped here, where the summaries are
    # already in hand, rather than making the caller walk the directory again.
    mine = next((s for s in summaries if s["id"] == detail.get("id")), None)
    if mine:
        detail.setdefault("number", mine.get("number"))
        detail.setdefault("duration", mine.get("duration", ""))
    return {
        "board": bar_board(detail, summaries),
        "readings": readings(detail),
        "budget": budget(detail),
        "training": detail.get("status") == "running",
    }


def live_state(points: int = 240) -> dict:
    """Everything the Control page needs about the run in front of it.

    Falls back to the newest run when nothing is training, so the page reads the
    same five minutes after a run ends as it did five minutes before.
    """
    summaries = runs.all_summaries()
    if not summaries:
        return {"run": None, "board": None, "readings": [], "budget": None,
                "previous": None}

    live = next((r for r in summaries if r["status"] == "running"), None)
    subject = live or summaries[0]
    detail = runs.detail(subject["id"], points=points) or {}
    detail["number"] = subject.get("number")
    detail["duration"] = subject.get("duration", "")
    previous = last_verdict(summaries, skip=subject["id"])
    spend = budget(detail)
    picked = charts(detail, previous, spend)

    # Only the columns actually drawn. Sending all forty-four for 240 rows is
    # 372 KB every five seconds, which is /api/monitor's old 1.6 MB bug at a
    # third the size - and the page charts four of them. Trimming here rather
    # than in the browser, because the cost is the wire, not the render.
    keep = {"iteration"} | {c["key"] for c in picked}
    curves = [{k: v for k, v in row.items() if k in keep}
              for row in (detail.get("metrics") or [])]

    return {
        "run": {k: detail.get(k) for k in (
            "id", "number", "name", "variant", "task", "status", "purpose", "bar",
            "notes", "started_age", "duration", "iterations_done",
            "iterations_target", "progress", "num_envs", "num_steps_per_env",
            "metric_columns",
            "verdict", "verdict_detail", "verdict_structured")},
        "training": live is not None,
        # The curves for the four charts: thinned to `points` rows, and only the
        # columns those four charts draw. See `keep` above.
        "metrics": curves,
        "board": bar_board(detail, summaries),
        "readings": readings(detail),
        "budget": spend,
        "previous": previous,
        # Four charts, not forty-four. Any other metric is one click away on
        # /runs, and the link is on the page.
        "charts": picked,
    }


# The four. `note` is why this one is on screen and the other forty are not; the
# page prints it, because a chart with no reason to be there is just decoration.
#
# None of these carries a bar line. `reward` and `episode_length` have no bar at
# all, and `speed_error` has one that it must not be drawn against - see
# LIVE_PROXY. A dashed rule at 0.05 under a curve sitting at 0.11 says "failing"
# to every reader who does not stop to read the caveat beside it.
CHARTS = [
    {"key": "reward", "note": "the score, against the most an episode can earn"},
    {"key": "episode_length", "note": "how long they survive. 1,000 is the full 20 s"},
    {"key": "speed_error", "note": "how far off the commanded speed, across the "
                                   "whole command box - not the 0.05 m/s bar"},
]


def charts(run: dict, previous: dict | None = None,
           spend: dict | None = None) -> list[dict]:
    """The three fixed charts, plus one chosen for what the last run failed on."""
    columns = set(run.get("metric_columns") or [])
    picked = [dict(c) for c in CHARTS]

    # The one reference line on the page: what an episode could score if every
    # reward term maxed out and no penalty was charged. It is not a pass bar - it
    # is the top of the axis the score is climbing, and training stops at 96.5%
    # of it (RULES.md rule 1). Worth drawing because "reward 130" says nothing on
    # its own and "130 of 190" says how much room is left.
    if spend and spend.get("ceiling") and spend.get("seconds"):
        picked[0]["bar"] = spend["ceiling"] * spend["seconds"]
        picked[0]["barLabel"] = "everything, every step"

    # The fourth is chosen, not fixed. If the last verdict failed on something,
    # show the live term that pays for it - so a repeat is visible while there is
    # still time to stop the run. For drift this is the only live signal there is,
    # and it is a reward term rather than millimetres, which the note says.
    fourth = _repeat_chart(previous)
    if fourth and fourth["key"] not in {p["key"] for p in picked}:
        picked.append(fourth)
    return [p for p in picked if p["key"] in columns]


# What pays for each criterion, for the "is it repeating the last failure" chart.
# These are reward terms, NOT the criterion's own unit - a rising `wandering`
# penalty means drift is getting worse, but it does not say how many degrees.
REPEAT_TERM = {
    "drift": ("Episode_Reward/wandering",
              "the drift penalty - the only live signal for the criterion #%s "
              "failed on. A reward term, not degrees off the line"),
    "distance": ("ground_covered",
                 "ground actually put behind it - what #%s fell short on"),
    "speed_err": ("speed_error", "speed tracking - what #%s failed on"),
    "survive": ("tipped_over", "robots going over - what #%s failed on"),
    "upright": ("upright_reward", "how level the trunk is - what #%s failed on"),
    "height_err": ("height_reward", "holding ride height - what #%s failed on"),
}


def _repeat_chart(previous: dict | None) -> dict | None:
    if not previous or not previous.get("failed"):
        return None
    for check in previous["failed"]:
        entry = REPEAT_TERM.get(check.get("key", ""))
        if entry:
            key, note = entry
            return {"key": key, "note": note % previous.get("number", "?")}
    return None
