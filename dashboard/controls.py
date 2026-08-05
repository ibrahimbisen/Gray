"""The pad map, as something the dashboard can read and change.

`Playground/settings.json` is already the map. `panel.py` loads it into
`PadPilot` when the simulator opens, and writes it back when the settings window
closes. Everything here reads and writes THAT file, in exactly the shape
`panel.py` writes it, so there is one map and not two.

**What the file cannot say is what a button is called.** `pad.py` records the
Switch button order in a comment - Y=0, B=1, A=2, X=3, L=4, R=5, ZL=6, ZR=7,
measured on the PowerA pad. A comment is not readable by anything, so it is
written out as data below, and `pad.py` now points at this file. If a different
pad is ever used, this table is the one line to change.

**Everything else is read out of the source.** The defaults, the deadzone and
the hard stops are regexed out of `Playground/pad.py` and `Playground/control.py`
rather than copied, and the trained ranges come from `plan.dials()`, which
already reads `walk_env_cfg.py` the same way. A number copied into a second file
is a number that goes stale - that is the rule this whole module exists to obey.

**Two writers.** The tkinter panel writes this file when it closes, and so does
the dashboard. Whoever goes last wins. That is fine when only one of them is
open, which is the only way they get used, but it is why saving from here while
the panel is open is a bad idea. The page says so.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dashboard import plan

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = ROOT / "Playground" / "settings.json"
PAD_SRC = ROOT / "Playground" / "pad.py"
CTL_SRC = ROOT / "Playground" / "control.py"

# The wish list: things the controller should be able to do that nothing does
# yet. Its own file, because `panel.py` rewrites settings.json from a fixed set
# of keys and would silently drop anything extra kept in there.
#
# NOTHING READS THIS BUT THE PAGE. An entry here is a note to whoever writes the
# code, not a feature - which is why every one of them is shown as `not built`
# and cannot be put on a button. The moment one IS built it moves into ACTIONS
# above and comes off this list.
WANTED = ROOT / "Playground" / "wanted.json"

# What a wanted entry can be. A number is something a stick makes; an action is
# something a button does. There is no third kind - the pad has sticks and
# buttons and that is all.
KINDS = ("action", "number")

NAME_MAX = 40
DOES_MAX = 140

# Where on the controller a wanted thing should go, once somebody builds it.
# `control` is the data-name of the shape on the drawing, so the page can put a
# marker on it without a second table of coordinates.
#
# The two stick entries per stick are the point of this list. A stick has two
# directions and the right one only uses ONE of them - side to side turns, up
# and down sends nothing at all. That spare axis is the obvious home for
# anything that wants a continuous number rather than a button.
# Four fields: the id, the full label for a dropdown, the shape it marks on the
# drawing, and a SHORT label for the menu that opens on that shape - where
# "right stick, up / down" would be repeating the thing you just clicked on.
PLACES = (
    ("", "not decided yet", "", ""),
    ("left-stick-updown", "left stick, up / down", "left", "up / down"),
    ("left-stick-side", "left stick, side to side", "left", "side to side"),
    ("right-stick-updown", "right stick, up / down", "right", "up / down"),
    ("right-stick-side", "right stick, side to side", "right", "side to side"),
    ("l3", "left stick click (L3)", "left", "click (L3)"),
    ("r3", "right stick click (R3)", "right", "click (R3)"),
    ("zl", "ZL", "ZL", "the button"), ("l", "L", "L", "the button"),
    ("zr", "ZR", "ZR", "the button"), ("r", "R", "R", "the button"),
    ("y", "Y", "Y", "the button"), ("x", "X", "X", "the button"),
    ("a", "A", "A", "the button"), ("b", "B", "B", "the button"),
    # Four ways, four places. Windows still reports them as ONE value - a POV
    # hat angle, not four button numbers - which is why they are unmeasured
    # here even though they are obviously four separate things to press.
    ("dpad-up", "d-pad up", "dpad-up", "the button"),
    ("dpad-down", "d-pad down", "dpad-down", "the button"),
    ("dpad-left", "d-pad left", "dpad-left", "the button"),
    ("dpad-right", "d-pad right", "dpad-right", "the button"),
    ("minus", "minus", "minus", "the button"),
    ("plus", "plus", "plus", "the button"),
    ("home", "home", "home", "the button"),
    ("capture", "capture", "capture", "the button"),
    ("key-1", "button 1 (top of the console)", "1", "the button"),
    ("key-2", "button 2 (top of the console)", "2", "the button"),
    ("key-3", "button 3 (top of the console)", "3", "the button"),
    ("key-4", "button 4 (top of the console)", "4", "the button"),
    ("key-5", "button 5 (top of the console)", "5", "the button"),
)

# Which places something already built is using, so the page can say so rather
# than letting two things be planned onto one axis without a word.
_TAKEN_BY_COMMAND = {"fwd": "left-stick-updown", "side": "left-stick-side",
                     "turn": "right-stick-side"}

# Windows' axis names, not the pad's. X and Y are the left stick; the right
# stick lands on some pair of Z, R, U, V depending on the pad.
AXES = ("X", "Y", "Z", "R", "U", "V")

# The three numbers a policy takes. Nothing else crosses the interface.
COMMANDS = (
    ("fwd", "forward", "m/s", "WALK_SPEED"),
    ("side", "sideways", "m/s", "WALK_SIDE"),
    ("turn", "turn", "rad/s", "WALK_TURN"),
)

# The three things a button can be made to do. Adding a fourth means writing
# code in PadPilot.poll that does it - this list is not a wish list, it is what
# is built. The wish list is `wanted()`, further down.
#
# Two descriptions each, because they are read in two places and the short one
# is not enough on its own. `says` is the callout on the drawing, where there is
# room for three words beside a button. `does` is the sentence in the table.
ACTIONS = (
    ("stop", "stop", "Stand still. Every one of the three numbers goes to zero."),
    ("reset", "put it back on its feet",
     "Stand it back up where it started, after it has gone over."),
    ("unlock", "hold - past the trained range",
     "While held, full stick reaches the hard stops instead of the trained "
     "range - for when driving it somewhere it has never been is the point."),
)

# Button number to Switch name, measured with `Playground\pad.bat` on the PowerA
# pad. These are the ones somebody has actually pressed and watched a number
# come back for. Everything else on the controller is unnumbered until the owner
# says otherwise - see BUTTONS below.
BUTTON_NAMES = {0: "Y", 1: "B", 2: "A", 3: "X",
                4: "L", 5: "R", 6: "ZL", 7: "ZR"}

# Numbers the owner set by hand, over the top of the measured ones. Its own file
# so `panel.py` cannot drop it, and separate from the measured table so the page
# can always say WHICH of the two a number came from.
#
# This exists because the measured table was a cage. A control with no measured
# number could not be given a job at all - which is wrong twice over: the pad can
# be probed, and this controller is being BUILT, so the owner decides the wiring
# rather than discovering it.
BUTTONS = ROOT / "Playground" / "buttons.json"

# Every control that can hold a button number, and what to call it. The `data-
# name` on the drawing is the key, so a control on the page and a control here
# are the same thing by construction. L3 and R3 have no shape of their own -
# they are the sticks pressed in - so they are reached through the stick's menu.
CONTROLS = (
    ("ZL", "ZL"), ("L", "L"), ("ZR", "ZR"), ("R", "R"),
    ("Y", "Y"), ("X", "X"), ("A", "A"), ("B", "B"),
    ("minus", "minus"), ("plus", "plus"),
    ("home", "home"), ("capture", "capture"),
    ("dpad-up", "d-pad up"), ("dpad-down", "d-pad down"),
    ("dpad-left", "d-pad left"), ("dpad-right", "d-pad right"),
    ("L3", "left stick click (L3)"), ("R3", "right stick click (R3)"),
    # Five extra buttons across the top of the console. Not on a real Switch -
    # they are part of what is being built, which is why they are numbered
    # rather than named after anything.
    ("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5"),
)

# joyGetPosEx reports this many.
BUTTON_MAX = 15

# What the panel's deadzone slider allows, so the two editors agree on the range.
DEADZONE_MAX = 0.4


# --- reading the source files ----------------------------------------------


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _number(src: str, name: str) -> float | None:
    hit = re.search(rf"^{name}\s*=\s*(-?[\d.]+)", src, re.M)
    return float(hit.group(1)) if hit else None


def _dict_literal(src: str, attr: str) -> dict | None:
    """One of PadPilot's single-line settings dicts, as a dict.

    They are `self.axis = {"fwd": "Y", ...}` and friends - short, flat, on one
    line each. json.loads handles them once True/False are spelled the JSON way.
    """
    hit = re.search(rf"self\.{attr}\s*=\s*(\{{[^}}]*\}})", src)
    if not hit:
        return None
    text = hit.group(1).replace("True", "true").replace("False", "false")
    try:
        return json.loads(text)
    except ValueError:
        return None


def ceilings() -> dict[str, float]:
    """The hard stops. Well past anything trained on, on purpose."""
    src = _src(CTL_SRC)
    out = {}
    for key, _label, _unit, _const in COMMANDS:
        value = _number(src, f"CEIL_{key.upper()}")
        out[key] = value if value is not None else 0.0
    return out


def trained() -> dict[str, tuple[float, float]]:
    """The range the walk policy was actually shown, from the task config.

    Reuses `plan.dials()` rather than re-reading walk_env_cfg.py: that function
    already does it, and a third copy of the regex is a third thing to fix.
    """
    by_key = {d["key"]: d for d in plan.dials().get("command", [])}
    out = {}
    for key, _label, _unit, const in COMMANDS:
        dial = by_key.get(const)
        out[key] = (dial["lo"], dial["hi"]) if dial else (0.0, 0.0)
    return out


def defaults() -> dict:
    """What PadPilot starts with when there is no settings file."""
    src = _src(PAD_SRC)
    fallback_full = {k: hi for k, (_lo, hi) in trained().items()}
    return {
        "axis": _dict_literal(src, "axis") or {"fwd": "Y", "side": "X", "turn": "Z"},
        "invert": _dict_literal(src, "invert")
        or {"fwd": True, "side": True, "turn": True},
        "button": _dict_literal(src, "button")
        or {"stop": 2, "reset": 1, "unlock": 6},
        # PadPilot sets full to the top of the trained range, so that is the
        # default here too rather than a number written down twice.
        "full": fallback_full,
        "deadzone": _number(src, "DEADZONE") or 0.12,
    }


# --- the wish list ----------------------------------------------------------


def _slug(name: str) -> str:
    """A stable id from the name, so removing one is not an index into a list."""
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def _saved_buttons() -> dict[str, int]:
    """Control name to button number, as the owner set them."""
    if not BUTTONS.is_file():
        return {}
    try:
        got = json.loads(BUTTONS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(got, dict):
        return {}
    known = {name for name, _label in CONTROLS}
    return {name: int(n) for name, n in got.items()
            if name in known and isinstance(n, int) and 0 <= n <= BUTTON_MAX}


def button_map() -> tuple[dict[int, str], dict[str, int], dict[str, str]]:
    """number->name, name->number, and where each number came from.

    The owner's file wins over the measured table. If they say home is 2, then 2
    is home and the measured `A` for 2 is dropped - two names for one number is
    the one thing this cannot allow, because the drawing would light up twice.
    """
    mine = _saved_buttons()
    by_name = dict(mine)
    source = {name: "yours" for name in mine}
    for number, name in BUTTON_NAMES.items():
        if name in by_name or number in mine.values():
            continue
        by_name[name] = number
        source[name] = "measured"
    by_number = {n: name for name, n in by_name.items()}
    return by_number, by_name, source


def set_button(body: dict) -> dict:
    """Give a control a button number, or take it away again."""
    control = str(body.get("control", ""))
    labels = dict(CONTROLS)
    if control not in labels:
        raise ValueError(f"'{control}' is not a control on this pad.")

    raw = body.get("number")
    mine = _saved_buttons()
    if raw is None or raw == "":
        # Back to whatever the measured table says, which for most of these is
        # nothing at all.
        mine.pop(control, None)
    else:
        try:
            number = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"'{raw}' is not a button number.") from None
        if not 0 <= number <= BUTTON_MAX:
            raise ValueError(f"Button {number} is out of range. joyGetPosEx "
                             f"reports 0 to {BUTTON_MAX}.")
        _by_number, by_name, _src = button_map()
        clash = next((n for n, v in by_name.items()
                      if v == number and n != control), "")
        if clash and clash not in mine:
            raise ValueError(
                f"Button {number} is already {clash}. Windows sends one number "
                f"per press, so it cannot be {labels[control]} as well - give "
                f"{clash} a different number first, or pick another.")
        if clash:
            raise ValueError(f"Button {number} is already {clash}. Change that "
                             f"one first.")
        mine[control] = number

    tmp = BUTTONS.with_suffix(".json.tmp")
    try:
        BUTTONS.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(mine, indent=2), encoding="utf-8")
        os.replace(tmp, BUTTONS)
    except OSError as exc:
        raise ValueError(f"Could not write {BUTTONS.name}: {exc}") from exc
    return read()


def _place(ident: str) -> tuple[str, str, str]:
    """A place id to its label, the shape it marks, and its short label."""
    for pid, label, control, short in PLACES:
        if pid == ident:
            return label, control, short
    return "", "", ""


def wanted() -> list[dict]:
    if not WANTED.is_file():
        return []
    try:
        got = json.loads(WANTED.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(got, list):
        return []
    out = []
    for row in got:
        if not isinstance(row, dict) or not row.get("name"):
            continue
        where = str(row.get("where", ""))
        label, control, short = _place(where)
        out.append({"id": _slug(str(row["name"])),
                    "name": str(row["name"])[:NAME_MAX],
                    "does": str(row.get("does", ""))[:DOES_MAX],
                    "kind": row["kind"] if row.get("kind") in KINDS else "action",
                    "where": where if label or not where else "",
                    "where_label": label,
                    "where_control": control,
                    "where_short": short,
                    # Never true here, and it is spelled out rather than left
                    # absent: this list is what has NOT been written.
                    "built": False})
    return out


def places() -> list[dict]:
    """Every place, and what is already sitting on it."""
    taken = {}
    for key, _label, _unit, _const in COMMANDS:
        taken[_TAKEN_BY_COMMAND.get(key, "")] = key
    for row in wanted():
        if row["where"] and row["where"] not in taken:
            taken[row["where"]] = row["name"]
    return [{"id": pid, "label": label, "control": control, "short": short,
             "taken_by": taken.get(pid, "")}
            for pid, label, control, short in PLACES]


def _write_wanted(rows: list[dict]) -> None:
    tmp = WANTED.with_suffix(".json.tmp")
    try:
        WANTED.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        os.replace(tmp, WANTED)
    except OSError as exc:
        raise ValueError(f"Could not write {WANTED.name}: {exc}") from exc


def add_wanted(body: dict) -> dict:
    name = str(body.get("name", "")).strip()
    does = str(body.get("does", "")).strip()
    kind = body.get("kind")

    if not name:
        raise ValueError("Give it a name - one or two words for what it is.")
    if len(name) > NAME_MAX:
        raise ValueError(f"That name is {len(name)} characters. Keep it under "
                         f"{NAME_MAX} so it fits the drawing.")
    if not does:
        raise ValueError("Say what it should do, in a line. A name on its own "
                         "will not mean anything in a month.")
    if len(does) > DOES_MAX:
        raise ValueError(f"That is {len(does)} characters. Keep it under "
                         f"{DOES_MAX} - it is a table row, not a spec.")
    if kind not in KINDS:
        raise ValueError("Pick whether a stick makes it or a button does it.")

    ident = _slug(name)
    if not ident:
        raise ValueError("That name has no letters or numbers in it.")
    built = {key for key, *_r in ACTIONS} | {key for key, *_r in COMMANDS}
    if ident in built:
        raise ValueError(f"'{name}' is already built - it is in the table "
                         f"above, not something to add.")
    rows = wanted()
    if any(r["id"] == ident for r in rows):
        raise ValueError(f"'{name}' is already on the list.")

    where = str(body.get("where", ""))
    if where and not _place(where)[0]:
        raise ValueError(f"'{where}' is not a place on this controller.")

    # `_bare` because wanted() hands back derived fields - id, where_label,
    # where_control, built - and writing those to disk would freeze a label that
    # is supposed to be worked out fresh on every read.
    _write_wanted(_bare(rows) + [{"name": name, "does": does, "kind": kind,
                                  "where": where}])
    return read()


def _bare(rows: list[dict]) -> list[dict]:
    """Back to what goes on disk - the derived fields are worked out on read."""
    return [{"name": r["name"], "does": r["does"], "kind": r["kind"],
             "where": r["where"]} for r in rows]


def set_where(body: dict) -> dict:
    """Say where a wanted thing should live once somebody builds it."""
    ident = str(body.get("id", ""))
    where = str(body.get("where", ""))
    if where and not _place(where)[0]:
        raise ValueError(f"'{where}' is not a place on this controller.")
    rows = wanted()
    if not any(r["id"] == ident for r in rows):
        raise ValueError("That one is not on the list any more.")
    for row in rows:
        if row["id"] == ident:
            row["where"] = where
    _write_wanted(_bare(rows))
    return read()


def remove_wanted(body: dict) -> dict:
    ident = str(body.get("id", ""))
    rows = wanted()
    kept = _bare([r for r in rows if r["id"] != ident])
    if len(kept) == len(rows):
        raise ValueError("That one is not on the list any more.")
    _write_wanted(kept)
    return read()


# --- the map itself ---------------------------------------------------------


def _saved() -> dict | None:
    if not SETTINGS.is_file():
        return None
    try:
        got = json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return got if isinstance(got, dict) else None


# What each part of the map holds. settings.json is written by the tkinter panel
# AND by this dashboard, and it can be hand-edited between the two, so a value
# read out of it is not known to be the right type.
#
# A string where a number belongs used to travel all the way to float() below and
# answer /api/controls with a 500 - which took away the one page that could have
# put it right. A value of the wrong type is now dropped and reported instead.
_KINDS = {"axis": "text", "invert": "yes or no", "button": "number", "full": "number"}


def _usable(kind: str, value) -> bool:
    if kind == "text":
        return isinstance(value, str)
    if kind == "yes or no":
        return isinstance(value, bool)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def read() -> dict:
    """The whole map, plus everything needed to judge and edit it."""
    base = defaults()
    saved = _saved()
    now = dict(base)
    dropped = []
    if saved:
        for key, kind in _KINDS.items():
            merged = dict(base[key])
            for name, value in (saved.get(key) or {}).items():
                if name not in merged:
                    continue
                if _usable(kind, value):
                    merged[name] = value
                else:
                    dropped.append(f"{key}.{name} in {SETTINGS.name} holds "
                                   f"{value!r}, which is not a {kind}. "
                                   f"The default is used instead.")
            now[key] = merged
        if isinstance(saved.get("deadzone"), (int, float)):
            now["deadzone"] = float(saved["deadzone"])

    cap = ceilings()
    span = trained()

    commands = []
    for key, label, unit, const in COMMANDS:
        lo, hi = span[key]
        full = float(now["full"].get(key, hi))
        commands.append({
            "key": key, "label": label, "unit": unit, "const": const,
            # Whether the robot can actually do this today. Answered here rather
            # than worked out by the page, so "can perform" has one definition:
            # it is in COMMANDS or ACTIONS, which means there is code for it.
            "built": True,
            "trained_lo": lo, "trained_hi": hi,
            "ceiling": cap[key], "full": full,
            # Full stick past the top of the trained range is legal and
            # sometimes wanted, but it means the robot's behaviour at full
            # deflection says nothing about the policy. Worth a word.
            "past_trained": full > hi + 1e-9,
            "axis": now["axis"].get(key, ""),
            "invert": bool(now["invert"].get(key, False)),
        })

    by_number, by_name, source = button_map()
    labels = dict(CONTROLS)
    actions = [{"key": key, "says": says, "does": does, "built": True,
                "button": int(now["button"].get(key, -1)),
                "control": by_number.get(int(now["button"].get(key, -1)), ""),
                "control_label": labels.get(
                    by_number.get(int(now["button"].get(key, -1)), ""), "")}
               for key, says, does in ACTIONS]

    # Every control, its number, and whether that number was measured off a real
    # pad or set by hand. The page needs all three: no number means no job, and
    # "measured" versus "yours" is the difference between a fact and a decision.
    on_control = {a["control"]: a for a in actions if a["control"]}
    controls = [{"name": name, "label": label,
                 "button": by_name.get(name, -1),
                 "source": source.get(name, ""),
                 "action": (on_control.get(name) or {}).get("key", ""),
                 "says": (on_control.get(name) or {}).get("says", "")}
                for name, label in CONTROLS]

    return {
        "saved": saved is not None,
        "path": "Playground/settings.json",
        "axes": list(AXES),
        "buttons": [{"n": n, "name": name} for n, name in sorted(by_number.items())],
        "controls": controls,
        "button_max": BUTTON_MAX,
        "deadzone": round(float(now["deadzone"]), 3),
        "deadzone_max": DEADZONE_MAX,
        "commands": commands,
        "actions": actions,
        "wanted": wanted(),
        "places": places(),
        "kinds": list(KINDS),
        "warnings": dropped + _warnings(commands, actions),
    }


def _warnings(commands: list[dict], actions: list[dict]) -> list[str]:
    """Things that are legal but are probably not what was meant."""
    out = []

    used = [c["axis"] for c in commands]
    if len(set(used)) != len(used):
        out.append("Two commands are reading the same stick axis, so one of "
                   "them will never do anything on its own.")

    numbers = [a["button"] for a in actions]
    if len(set(numbers)) != len(numbers):
        out.append("Two actions are on the same button. Pressing it does both.")

    for a in actions:
        if not a["control"]:
            out.append(f"{a['key']} is on button {a['button']}, and no control "
                       f"has that number - so nothing on the drawing presses it. "
                       f"Give a control that number, or move {a['key']}.")

    for c in commands:
        if c["past_trained"]:
            out.append(
                f"Full stick on {c['label']} is {c['full']:g} {c['unit']}, past "
                f"the {c['trained_hi']:g} it trained on. Out there, what the "
                f"robot does tells you nothing about the policy.")
    return out


# --- writing ----------------------------------------------------------------


def save(body: dict) -> dict:
    """Validate and write the map. Raises ValueError with something readable.

    Checked hard, on purpose: this file is loaded straight into the thing that
    drives the robot, and a bad value here shows up as a robot that behaves
    strangely rather than as an error message.
    """
    cap = ceilings()

    axis, invert, full = {}, {}, {}
    for key, label, unit, _const in COMMANDS:
        name = str((body.get("axis") or {}).get(key, ""))
        if name not in AXES:
            raise ValueError(f"{label}: '{name}' is not one of the six axes "
                             f"Windows reports ({', '.join(AXES)}).")
        axis[key] = name
        invert[key] = bool((body.get("invert") or {}).get(key, False))

        try:
            value = float((body.get("full") or {}).get(key))
        except (TypeError, ValueError):
            raise ValueError(f"{label}: full stick is not a number.") from None
        if value <= 0:
            raise ValueError(f"{label}: full stick must be more than 0, "
                             f"or the stick does nothing.")
        if value > cap[key] + 1e-9:
            raise ValueError(f"{label}: full stick is {value:g} {unit}, past the "
                             f"hard stop of {cap[key]:g}. Raise CEIL_"
                             f"{key.upper()} in Playground/control.py first.")
        full[key] = round(value, 3)

    if len(set(axis.values())) != 3:
        raise ValueError("Two commands are on the same axis. One stick "
                         "direction cannot drive two of the three numbers.")

    button = {}
    for key, *_rest in ACTIONS:
        raw = (body.get("button") or {}).get(key)
        try:
            number = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key}: '{raw}' is not a button number.") from None
        if not 0 <= number <= 15:
            raise ValueError(f"{key}: button {number} is out of range. "
                             f"joyGetPosEx reports 0 to 15.")
        button[key] = number
    if len(set(button.values())) != 3:
        raise ValueError("Two actions are on the same button. Pressing it "
                         "would do both at once.")

    try:
        deadzone = float(body.get("deadzone"))
    except (TypeError, ValueError):
        raise ValueError("The deadzone is not a number.") from None
    if not 0.0 <= deadzone <= DEADZONE_MAX:
        raise ValueError(f"The deadzone must be between 0 and {DEADZONE_MAX}. "
                         f"Above that, most of the stick's travel does nothing.")

    # Exactly the shape and the indent panel.py writes, so the two editors
    # produce identical files and a diff never shows a formatting change.
    payload = json.dumps({"axis": axis, "invert": invert, "button": button,
                          "full": full, "deadzone": round(deadzone, 3)}, indent=2)

    # Write beside it and rename. A half-written settings.json is a simulator
    # that will not start, and the rename is the one step that cannot tear.
    tmp = SETTINGS.with_suffix(".json.tmp")
    try:
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, SETTINGS)
    except OSError as exc:
        raise ValueError(f"Could not write {SETTINGS.name}: {exc}") from exc

    return read()
