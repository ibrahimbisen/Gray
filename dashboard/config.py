"""Every number that can be changed, where it lives, and how to change it.

The /dials page answers "what gets varied, and over what range". This one
answers a different question the owner asked on 6 Aug 2026: what is there to
change AT ALL. It is the difference between a list of the dials in use and the
whole dashboard of the machine.

NOTHING HERE IS WRITTEN DOWN TWICE. Every value is read from the file that
owns it, at the moment the page is drawn:

    the task constants   parsed out of gray/tasks/walk_env_cfg.py with `ast`,
                         so a renamed or retyped constant shows up as missing
                         rather than as a stale number
    the flags            parsed out of scripts/train.py's own argparse calls,
                         so a flag added there appears here with its help text
                         and its default, and one deleted disappears
    the reward weights   read from the newest run's run.json, which train.py
                         writes from the live config as the run starts

A page that keeps its own copy of these goes stale in a week, and this project
has already lost a day to exactly that - see the note at the top of plan.py's
rewards().

PARSED, NOT IMPORTED, for the same reason the queue scripts parse: importing
gray.tasks pulls in mjlab, and the dashboard runs in a plain interpreter that
has no GPU and no business building an environment to draw a table.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK = ROOT / "gray" / "tasks" / "walk_env_cfg.py"
TRAIN = ROOT / "scripts" / "train.py"
VERIFY = ROOT / "scripts" / "verify.py"
RUNS = ROOT / "progress" / "runs"

# What each task constant is, in plain words. The VALUE is never written here -
# only the explanation, which the source cannot give in a form a page can show.
# A constant with no entry still appears, marked as undocumented, because a
# silent omission is how a knob goes unnoticed for a month.
MEANS = {
    "WALK_SPEED": ("how fast it can be told to walk, forward and backward",
                   "m/s"),
    "WALK_SIDE": ("how fast it can be told to crab sideways", "m/s"),
    "WALK_TURN": ("how fast it can be told to turn", "rad/s"),
    "POSE_HEIGHT": ("how high the trunk can be told to ride", "m"),
    "POSE_PITCH": ("how far it can be told to lean, nose down NEGATIVE", "rad"),
    "POSE_ROLL": ("how far it can be told to bank, right side down NEGATIVE",
                  "rad"),
    "TRACK_STD": ("how sharply speed is scored. Smaller is stricter, and too "
                  "small pays nothing for a command it cannot reach - which "
                  "teaches a robot to stop trying", "m/s"),
    "TURN_STD": ("how sharply turn rate is scored. Same trap, on the other "
                 "axis", "rad/s"),
    "SWING_TARGET": ("the height a foot should reach at the top of its swing",
                     "m"),
    "SWING_SPREAD": ("how much the leg joints should move in a stride", "rad"),
    "MOVING": ("the smallest command that still counts as being told to move",
               "m/s"),
    "FILTER_STEPS": ("how many control steps the speed reading is smoothed "
                     "over", "steps"),
    "DRIFT_FREE_M": ("how far off the line is free before drift is charged",
                     "m"),
    "OFF_TRACK_CLIP_M": ("the largest cross-track error the policy is shown",
                         "m"),
}

# The bars live in verify.py's TASKS table. Named here so the page can show
# them beside the things that produce them.
BAR_MEANS = {
    "bar_survive": "the share of robots still standing at the end",
    "bar_err_mm": "how far the trunk may sit from its ride height",
    "bar_upright": "how level it must stay",
    "bar_distance_m": "how far it must walk in the test",
    "bar_speed_err": "how close it must hold the commanded speed",
    "bar_drift_deg": "how far off the line it may end up, walking",
    "bar_side_distance_m": "how far it must crab",
    "bar_side_speed_err": "how close it must hold a sideways command",
    "bar_side_drift_deg": "how far off the line it may end up, crabbing",
    "bar_turn_err": "how close it must hold a turn command",
    "bar_turn_wander_m": "how far it may wander while turning on the spot",
    "test_speed": "the forward speed the test is taken at",
    "test_speed_back": "the backward speed the test is taken at",
    "test_side": "the sideways speed the test is taken at",
    "test_turn": "the turn rate the test is taken at",
    "seconds": "how long each pass runs",
}


def _literal(node) -> object | None:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def _fmt(v) -> str:
    if isinstance(v, tuple):
        return " to ".join(f"{x:g}" if isinstance(x, (int, float)) else str(x)
                           for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def task_constants() -> list[dict]:
    """Module-level numbers in the walk task, with their comments as the why."""
    src = TASK.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        name = getattr(node.targets[0], "id", "")
        if not name or not name.isupper():
            continue
        value = _literal(node.value)
        if not isinstance(value, (int, float, tuple)):
            continue
        what, unit = MEANS.get(name, ("", ""))
        out.append({"name": name, "value": _fmt(value), "unit": unit,
                    "what": what, "where": "gray/tasks/walk_env_cfg.py",
                    "documented": bool(what)})
    return out


def train_flags() -> list[dict]:
    """Every --flag scripts/train.py accepts, with its default and its help."""
    tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "add_argument"):
            continue
        if not node.args:
            continue
        flag = _literal(node.args[0])
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        default = _literal(kw["default"]) if "default" in kw else None
        action = _literal(kw["action"]) if "action" in kw else None
        help_text = _literal(kw["help"]) if "help" in kw else ""
        out.append({
            "flag": flag,
            "default": ("off" if action == "store_true"
                        else "the task's own" if default in (None, 0, "")
                        else _fmt(default)),
            "help": " ".join((help_text or "").split()),
        })
    return sorted(out, key=lambda f: f["flag"])


def bars() -> list[dict]:
    """The walk bars, out of verify.py's own table."""
    tree = ast.parse(VERIFY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "TASKS" for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            continue
        for key, val in zip(node.value.keys, node.value.values):
            if _literal(key) != "Gray-Walk" or not isinstance(val, ast.Dict):
                continue
            out = []
            for k, v in zip(val.keys, val.values):
                name = _literal(k)
                value = _literal(v)
                if not isinstance(name, str) or name not in BAR_MEANS:
                    continue
                out.append({"name": name, "value": _fmt(value),
                            "what": BAR_MEANS[name]})
            return out
    return []


def newest_run() -> dict:
    """The newest run's own record - the only live copy of the weights."""
    best, meta = None, {}
    for d in RUNS.iterdir() if RUNS.is_dir() else []:
        f = d / "run.json"
        if not f.is_file():
            continue
        if best is None or f.stat().st_mtime > best:
            try:
                meta, best = json.loads(f.read_text()), f.stat().st_mtime
                meta["run"] = d.name
            except (OSError, ValueError):
                continue
    return meta


def state() -> dict:
    meta = newest_run()
    return {
        "constants": task_constants(),
        "flags": train_flags(),
        "bars": bars(),
        "weights": [{"name": t.get("name"), "weight": t.get("weight"),
                     "what": t.get("what", "")}
                    for t in (meta.get("scoring") or [])],
        "tolerances": meta.get("tolerances") or {},
        "world_dials": meta.get("world_dials") or [],
        "from_run": meta.get("run", ""),
    }
