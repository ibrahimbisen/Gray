"""Gray dashboard - data layer.

Reads everything the dashboard shows straight off disk and returns one dict:

    collect(repo_root=".") -> dict

Sources, all real files, nothing invented:
  gray/config/robot.yaml                    mass model and leg kinematics
  logs/rsl_rl/gray_residual/<run>/          TensorBoard event file + params/*.yaml
  progress/summary.csv                      measured walk scores per checkpoint
  Overview/, sim/models/                    photos, renders and clips

Anything unreadable is reported in the returned "errors" list rather than
swallowed, so the page can show the user what is missing instead of a blank.

Deliberately light: stdlib + yaml + tensorboard's EventAccumulator. No torch,
no MuJoCo, no mjlab - this gets polled every few seconds by the server and must
stay fast and free of side effects.

Run `python dashboard/collect.py` to print the JSON and eyeball it.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any
from urllib.parse import quote

import yaml

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# Kept with forward slashes: they are joined via os.path.join (which accepts
# them on Windows too) and quoted straight into user-facing error messages,
# where a backslash would just look like a typo.
EXPERIMENT_DIR = "logs/rsl_rl/gray_residual"
SUMMARY_CSV = "progress/summary.csv"
ROBOT_YAML = "gray/config/robot.yaml"

# A run counts as "live" if anything inside its directory was touched this
# recently. The trainer writes the event file every iteration (~2.4 s at 16384
# robots), so 5 minutes is generous enough to survive a slow checkpoint write.
LIVE_WINDOW_S = 300.0

# Fallback only. The real value is read from params/agent.yaml (max_iterations).
DEFAULT_TOTAL_ITERATIONS = 3000

# Series sent to the browser are capped at this many points. 3000 iterations of
# 28 tags is fine to plot but wasteful to ship in full on every poll.
MAX_SERIES_POINTS = 400

# The Phase 2 hand-written gait, measured and recorded in docs/PROJECT_NOTES.md
# and docs/DEVLOG.md. These are used ONLY until progress/summary.csv exists;
# once it does, its "baseline" row wins (see _read_walks).
BASELINE_FALLBACK = {
    "distance_mm": 675.4,
    "speed_mms": 56.3,
    "drift_mm": 33.8,
    "upright_min": 0.976,
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _media_url(rel_path: str) -> str:
    """Repo-relative path -> URL the server serves it at.

    Paths contain spaces ("Overview/Screenshot 2022-05-28 114828.png"), so they
    must be percent-encoded; "/" stays literal so the server can walk it.
    """
    return "/media/" + quote(rel_path.replace("\\", "/"), safe="/")


def _newest_mtime(directory: str) -> float:
    """Most recent mtime of any file under `directory` (0.0 if none)."""
    newest = 0.0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                continue
    return newest


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that shrugs at tags it does not know.

    params/agent.yaml is dumped by the trainer and contains `!!python/tuple`,
    which yaml.safe_load refuses outright. We only ever want plain scalars out
    of these files, so unknown tags are collapsed to the underlying node.
    """


def _construct_unknown(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


# Empty prefix matches every tag, but only after PyYAML has failed to find an
# exact constructor - so the standard str/int/list handling is untouched.
_TolerantLoader.add_multi_constructor("", _construct_unknown)


def _load_yaml_tolerant(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_TolerantLoader)


def _deep_find(node: Any, key: str) -> Any:
    """First value stored under `key` anywhere in a nested dict/list."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _deep_find(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _deep_find(value, key)
            if found is not None:
                return found
    return None


def _to_float(text: Any, default: float | None = None) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _downsample(points: list, limit: int = MAX_SERIES_POINTS) -> list:
    """Uniform stride, and the final point is always kept.

    Losing the last point would make a live chart look frozen one iteration
    behind, which is exactly the number the user is watching.
    """
    n = len(points)
    if n <= limit:
        return points
    stride = -(-n // limit)  # ceil division
    out = points[::stride]
    if out[-1] is not points[-1]:
        out.append(points[-1])
    return out


# ---------------------------------------------------------------------------
# goal - the plain-English explanation of what is being trained and why
# ---------------------------------------------------------------------------


def _build_goal(baseline: dict) -> dict:
    return {
        "headline": (
            "Teach Gray to walk further and straighter in 12 seconds than the "
            "hand-written gait already manages, without falling over or "
            "stamping its feet."
        ),
        "explain": [
            "A walking gait for Gray already exists. It was written by hand: the "
            "feet follow a fixed looping path, three feet on the ground at all "
            "times. It works, and it is the score to beat.",
            "The AI is not learning to walk from scratch. It watches the robot "
            "and adds a small correction on top of that existing gait - at most "
            "about 11 degrees at any joint. Think of it as a trim adjustment on "
            "a machine that already runs, not a new machine.",
            "It learns by practising. 16,384 copies of the robot walk at once on "
            "one graphics card, each for 12 seconds. Every copy is scored, the "
            "good behaviour is reinforced, and the whole thing repeats. One "
            "round of that is one 'iteration' on the charts below.",
            "Every copy is built slightly differently on purpose - heavier "
            "parts, a battery sitting off-centre, a more slippery floor, servos "
            "that respond sluggishly. Something that only works on one exact set "
            "of numbers will score badly on average, so the AI is pushed towards "
            "a gait that tolerates the real robot not matching the drawing.",
            "The scores below are not percentages. They are running totals a "
            "12-second attempt earns, so they can be any size, and the penalty "
            "lines are negative on purpose.",
        ],
        "baseline": baseline,
        "targets": [
            {
                "name": "Distance covered",
                "target": f"beat {baseline['distance_mm']:.0f} mm in 12 s",
                "why": "The headline number. The hand-written gait crawls; the "
                       "AI should get more ground out of the same 12 seconds.",
            },
            {
                "name": "Walking speed",
                "target": f"beat {baseline['speed_mms']:.1f} mm/s",
                "why": "The same result read as a rate. It is asked for speeds "
                       "between -40 and +80 mm/s so it cannot just memorise one.",
            },
            {
                "name": "Straightness",
                "target": f"drift under {baseline['drift_mm']:.0f} mm",
                "why": "Sideways and turning are both commanded to zero, so any "
                       "sideways wander is error. Less is better.",
            },
            {
                "name": "Stays upright",
                "target": f"no falls, uprightness above {baseline['upright_min']:.3f}",
                "why": "1.000 means the body is perfectly level. An attempt is "
                       "abandoned the moment the robot tips past 60 degrees.",
            },
            {
                "name": "Gentle feet",
                "target": "softer foot impacts than the hand-written gait",
                "why": "The parts are printed resin and chip when hammered. Feet "
                       "should be placed, not dropped.",
            },
            {
                "name": "Survives reality",
                "target": "still works across the randomised range below",
                "why": "The point of the whole exercise: a gait that only works "
                       "on the perfect simulated robot is worthless on the bench.",
            },
        ],
        # "good" splits what it is PAID to do from what it is PENALISED for -
        # the sign of the weight is the only thing that decides this.
        "rewards": [
            {
                "name": "Distance covered",
                "weight": 2.0,
                "plain": "Points for actually getting somewhere, always counted "
                         "forwards. Added after the first attempt, where the "
                         "robot worked out that standing perfectly still was the "
                         "safest way to score.",
                "good": True,
            },
            {
                "name": "Hits the asked-for speed",
                "weight": 1.5,
                "plain": "Points for walking at the speed it was told to. Full "
                         "marks within about 70 mm/s of the target, tailing off "
                         "either side.",
                "good": True,
            },
            {
                "name": "Does not turn",
                "weight": 1.0,
                "plain": "Points for not rotating. It is always asked to turn at "
                         "zero, so this is what keeps it tracking straight.",
                "good": True,
            },
            {
                "name": "Keeps the body level",
                "weight": 0.5,
                "plain": "Points for staying upright rather than leaning, "
                         "rolling or pitching as it moves.",
                "good": True,
            },
            {
                "name": "Soft landings",
                "weight": -1e-4,
                "plain": "Marked down for slamming a foot into the ground. The "
                         "printed resin parts are brittle and chip.",
                "good": False,
            },
            {
                "name": "Smooth joint motion",
                "weight": -5e-7,
                "plain": "A very small mark-down for snatching a joint from one "
                         "speed to another.",
                "good": False,
            },
            {
                "name": "Steady corrections",
                "weight": -0.005,
                "plain": "Marked down for changing its correction jerkily from "
                         "one step to the next.",
                "good": False,
            },
            {
                "name": "Slow-changing corrections",
                "weight": -0.05,
                "plain": "Marked down for changing its correction quickly at "
                         "all. With the line above, this is what stops the "
                         "servos buzzing and cooking themselves.",
                "good": False,
            },
            {
                "name": "Easy on the servos",
                "weight": -1e-4,
                "plain": "Marked down for pulling hard on the servos. Saves the "
                         "gear trains and the battery.",
                "good": False,
            },
            {
                "name": "Off the end stops",
                "weight": -1.0,
                "plain": "Marked down for driving a joint into the end of its "
                         "travel, where the servo stalls and heats up.",
                "good": False,
            },
        ],
        "randomization": [
            {
                "name": "Part density",
                "range": "plus or minus 40%",
                "why": "Masses are estimated from mesh volume, not weighed - the "
                       "robot is in pieces. This covers being wrong about it.",
            },
            {
                "name": "Body centre of mass",
                "range": "plus or minus 2 cm",
                "why": "The battery, Pi and wiring never sit exactly where the "
                       "CAD says they do.",
            },
            {
                "name": "Ground friction",
                "range": "0.4 to 1.2",
                "why": "Tile, carpet and a bench top all grip differently. The "
                       "gait must not rely on one surface.",
            },
            {
                "name": "Servo rotor inertia",
                "range": "0.35x to 3.0x",
                "why": "How hard the servo is to accelerate is not known, and "
                       "the push-rod linkage changes what the joint feels.",
            },
            {
                "name": "Servo strength and damping",
                "range": "plus or minus 30-40%",
                "why": "Hobby servos are not matched parts. Each one behaves a "
                       "little differently, and drifts as it warms up.",
            },
            {
                "name": "Servo zero offset",
                "range": "plus or minus 0.03 rad (about 1.7 degrees)",
                "why": "A servo horn can only be fitted to the nearest spline, "
                       "so every joint carries a small built-in error.",
            },
            {
                "name": "Command delay",
                "range": "10 to 40 ms",
                "why": "The lag between deciding a move and the servo receiving "
                       "it over the Pi's I2C bus.",
            },
        ],
        "constraints": [
            {
                "name": "Command rate",
                "value": "50 times a second",
                "why": "The PCA9685 driver board holds each servo pulse for "
                       "20 ms. That is a hardware ceiling, not a design choice.",
            },
            {
                "name": "Size of the correction",
                "value": "at most 0.2 rad (about 11 degrees) per joint",
                "why": "Hard-limited, so the AI can nudge the existing gait but "
                       "can never command something violent.",
            },
            {
                "name": "No joint feedback",
                "value": "commands out, nothing back",
                "why": "The servos accept a target angle and report nothing. So "
                       "the AI is deliberately never shown measured joint "
                       "angles - it must work from body motion alone, because "
                       "that is all the real robot will ever have.",
            },
            {
                "name": "Length of one attempt",
                "value": "12 seconds",
                "why": "Ends early if the robot tips past 60 degrees, so a fall "
                       "costs it the rest of its scoring time.",
            },
            {
                "name": "Speeds it is asked for",
                "value": "-40 to +80 mm/s",
                "why": "Sideways and turning are both fixed at zero: this is a "
                       "robot being taught to walk straight.",
            },
            {
                "name": "Practice scale",
                "value": "16,384 robots at once, 3000 rounds",
                "why": "All on a single RTX 4070 Ti. Every robot in the batch "
                       "has slightly different physics.",
            },
        ],
    }


# ---------------------------------------------------------------------------
# training - the live TensorBoard feed
# ---------------------------------------------------------------------------


def _empty_training() -> dict:
    return {
        "running": False,
        "run_name": "",
        "iteration": 0,
        "total_iterations": DEFAULT_TOTAL_ITERATIONS,
        "num_envs": 0,
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
        "series": {},
        "latest": {},
    }


def _read_run_params(run_dir: str, errors: list) -> tuple[int, int]:
    """(total_iterations, num_envs) from params/*.yaml."""
    total = DEFAULT_TOTAL_ITERATIONS
    num_envs = 0
    params_dir = os.path.join(run_dir, "params")

    agent_path = os.path.join(params_dir, "agent.yaml")
    if os.path.isfile(agent_path):
        try:
            value = _deep_find(_load_yaml_tolerant(agent_path), "max_iterations")
            if isinstance(value, (int, float)) and value > 0:
                total = int(value)
        except Exception as exc:
            errors.append(f"Could not read {agent_path}: {exc}")

    env_path = os.path.join(params_dir, "env.yaml")
    if os.path.isfile(env_path):
        try:
            value = _deep_find(_load_yaml_tolerant(env_path), "num_envs")
            if isinstance(value, (int, float)):
                num_envs = int(value)
        except Exception as exc:
            # env.yaml is 1000+ lines of trainer config; if the parser chokes,
            # the one number we want is still trivially greppable.
            errors.append(f"Could not parse {env_path} ({exc}); scanned it as text instead.")
            try:
                with open(env_path, "r", encoding="utf-8", errors="replace") as fh:
                    match = re.search(r"num_envs:\s*(\d+)", fh.read())
                if match:
                    num_envs = int(match.group(1))
            except OSError:
                pass

    return total, num_envs


def _collect_training(root: str, errors: list) -> dict:
    exp_dir = os.path.join(root, EXPERIMENT_DIR)
    if not os.path.isdir(exp_dir):
        errors.append(
            f"No training runs found - {EXPERIMENT_DIR} does not exist yet."
        )
        return _empty_training()

    runs = [
        os.path.join(exp_dir, name)
        for name in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, name))
    ]
    if not runs:
        errors.append(f"No training runs inside {EXPERIMENT_DIR} yet.")
        return _empty_training()

    # Newest by content, not by directory name - a resumed or renamed run still
    # sorts correctly this way.
    run_dir = max(runs, key=_newest_mtime)
    newest_touch = _newest_mtime(run_dir)

    out = _empty_training()
    out["run_name"] = os.path.basename(run_dir)
    out["running"] = (time.time() - newest_touch) < LIVE_WINDOW_S
    out["total_iterations"], out["num_envs"] = _read_run_params(run_dir, errors)

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception as exc:
        errors.append(f"TensorBoard is not importable, so no training charts: {exc}")
        return out

    try:
        # size_guidance 0 means "keep every point" - the default silently
        # thins the history to ~1000 samples, which would lie about the curve.
        acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags()["scalars"]
    except Exception as exc:
        errors.append(f"Could not read the training log in {out['run_name']}: {exc}")
        return out

    if not tags:
        errors.append(
            f"Run {out['run_name']} has started but has not logged anything yet."
        )
        return out

    first_wall = None
    last_wall = None
    iteration = 0
    reference: list[tuple[int, float]] = []  # (step, wall_time) for the ETA

    for tag in sorted(tags):
        try:
            events = acc.Scalars(tag)
        except Exception as exc:
            errors.append(f"Could not read the '{tag}' chart: {exc}")
            continue
        if not events:
            continue

        pairs = [[int(e.step), round(float(e.value), 6)] for e in events]
        out["series"][tag] = _downsample(pairs)
        out["latest"][tag] = round(float(events[-1].value), 6)

        walls = [e.wall_time for e in events]
        first_wall = walls[0] if first_wall is None else min(first_wall, walls[0])
        last_wall = walls[-1] if last_wall is None else max(last_wall, walls[-1])

        # Tags ending in "/time" are the same data replotted against elapsed
        # seconds, so their "step" is a clock reading, not an iteration number.
        if not tag.endswith("/time"):
            iteration = max(iteration, int(events[-1].step))
            if len(events) > len(reference):
                reference = [(int(e.step), e.wall_time) for e in events]

    out["iteration"] = iteration
    if first_wall is not None and last_wall is not None:
        out["elapsed_seconds"] = round(last_wall - first_wall, 1)

    # Seconds-per-iteration over the tail of the run, not the whole thing: the
    # first iterations include warm-up and would flatter the estimate.
    # Only meaningful while the run is live; the page should hide it otherwise.
    if len(reference) >= 2 and iteration < out["total_iterations"]:
        tail = reference[-20:]
        d_step = tail[-1][0] - tail[0][0]
        d_wall = tail[-1][1] - tail[0][1]
        if d_step > 0 and d_wall > 0:
            out["eta_seconds"] = round(
                (d_wall / d_step) * (out["total_iterations"] - iteration), 1
            )

    return out


# ---------------------------------------------------------------------------
# walks - measured scores from progress/summary.csv
# ---------------------------------------------------------------------------


def _walk_row(raw: dict, root: str) -> dict:
    video = (raw.get("video") or "").strip().replace("\\", "/")
    video_url = None
    if video:
        rel = video.lstrip("./")
        if os.path.isfile(os.path.join(root, rel)):
            video_url = _media_url(rel)

    return {
        "distance_mm": _to_float(raw.get("distance_mm"), 0.0),
        "speed_mms": _to_float(raw.get("speed_mms"), 0.0),
        "drift_mm": _to_float(raw.get("drift_mm"), 0.0),
        "upright_min": _to_float(raw.get("upright_min"), 0.0),
        "height_mm": _to_float(raw.get("height_mm"), 0.0),
        "fell": int(_to_float(raw.get("fell"), 0) or 0),
        "video": video_url,
    }


def _read_walks(root: str, errors: list) -> tuple[list, dict | None]:
    path = os.path.join(root, SUMMARY_CSV)
    if not os.path.isfile(path):
        errors.append(
            "No walk scores yet - progress/summary.csv appears after the first "
            "run of scripts/make_progress_videos.py."
        )
        return [], None

    walks: list[dict] = []
    baseline: dict | None = None
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for raw in csv.DictReader(fh):
                tag = (raw.get("iteration") or "").strip()
                row = _walk_row(raw, root)
                if tag.lower() == "baseline":
                    row["iteration"] = None
                    baseline = row
                    continue
                number = _to_float(tag)
                if number is None:
                    errors.append(f"Skipped a summary.csv row with iteration '{tag}'.")
                    continue
                row["iteration"] = int(number)
                walks.append(row)
    except Exception as exc:
        errors.append(f"Could not read {SUMMARY_CSV}: {exc}")
        return [], None

    walks.sort(key=lambda r: r["iteration"])
    return walks, baseline


# ---------------------------------------------------------------------------
# robot - the physical machine, straight out of robot.yaml
# ---------------------------------------------------------------------------


def _empty_robot() -> dict:
    return {
        "total_mass_kg": 0.0,
        "servo": {},
        "servo_placement": {},
        "links": [],
        "legs": [],
        "facts": [],
    }


def _collect_robot(root: str, errors: list) -> dict:
    path = os.path.join(root, ROBOT_YAML)
    if not os.path.isfile(path):
        errors.append(f"Robot spec missing - {ROBOT_YAML} was not found.")
        return _empty_robot()

    try:
        with open(path, "r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh)
    except Exception as exc:
        errors.append(f"Could not read {ROBOT_YAML}: {exc}")
        return _empty_robot()

    out = _empty_robot()
    out["total_mass_kg"] = float(spec.get("total_mass_kg") or 0.0)
    out["servo"] = spec.get("servo") or {}
    out["servo_placement"] = spec.get("servo_placement") or {}

    for name, link in sorted((spec.get("links") or {}).items()):
        out["links"].append({
            "name": name,
            "mass_kg": float(link.get("mass_kg") or 0.0),
            "resin_g": float(link.get("resin_g") or 0.0),
            "baked_servo_g": float(link.get("baked_servo_g") or 0.0),
            "extra_g": float(link.get("extra_g") or 0.0),
            "servos_in_mesh": int(link.get("servos_in_mesh") or 0),
        })

    reach_mm = 0.0
    for name, leg in sorted((spec.get("legs") or {}).items()):
        thigh = float(leg.get("thigh_len") or 0.0)
        shank = float(leg.get("shank_len") or 0.0)
        mount = [float(v) for v in (leg.get("mount_pos") or [0.0, 0.0, 0.0])]
        reach = (thigh + shank) * 1000.0
        reach_mm = max(reach_mm, reach)
        out["legs"].append({
            "name": name,
            "thigh_len_mm": round(thigh * 1000.0, 1),
            "shank_len_mm": round(shank * 1000.0, 1),
            "reach_mm": round(reach, 1),
            "mount_pos_mm": [round(v * 1000.0, 1) for v in mount],
        })

    servo = out["servo"]
    placement = out["servo_placement"]
    material = spec.get("material") or {}
    base = (spec.get("links") or {}).get("base_link") or {}
    electronics_g = float(base.get("extra_g") or 0.0)
    transmission = spec.get("transmission") or {}

    facts: list[dict] = []

    if servo:
        facts.append({
            "label": "Servos",
            "value": f"{servo.get('count', 12)}x {servo.get('model', 'DS3218MG')}",
            "note": (
                f"Three per leg: one to swing the leg out sideways, one at the "
                f"shoulder, one at the knee. Rated {servo.get('effort_nm')} N.m "
                f"and {servo.get('velocity_rad_s')} rad/s, "
                f"{servo.get('range_deg')} degrees of travel, "
                f"{servo.get('mass_g')} g each."
            ),
        })
        facts.append({
            "label": "No position feedback",
            "value": "you can tell a servo where to go, never ask where it is",
            "note": (
                "The single biggest constraint on this project. These servos "
                "take a target angle and report nothing back. That is exactly "
                "why the AI is never shown measured joint angles in training - "
                "it has to work from body motion and its own commands, because "
                "that is all the real robot will ever be able to give it."
            ),
        })
        facts.append({
            "label": "Command rate",
            "value": f"{servo.get('control_hz', 50)} times a second",
            "note": (
                "The PCA9685 driver board holds each servo pulse for 20 ms. "
                "Nothing can be commanded faster, so the entire controller is "
                "built around 50 updates a second."
            ),
        })

    if transmission.get("knee"):
        facts.append({
            "label": "Knee drive",
            "value": "push-rod linkage, not direct drive",
            "note": (
                "The knee servo sits on the thigh and pushes a ball-jointed "
                "rod. Servo angle and knee angle are therefore not the same "
                "number. Harmless in simulation - a linkage driving a hinge is "
                "still a hinge - but it must be measured on the real robot "
                "before any trained gait is deployed. Logged as a blocker; "
                f"calibrated: {transmission.get('calibrated')}."
            ),
        })

    if out["total_mass_kg"]:
        facts.append({
            "label": "Total mass",
            "value": f"{out['total_mass_kg']:.3f} kg",
            "note": (
                "Estimated from mesh volume and datasheet masses - the robot is "
                "in pieces and could not be weighed. The first digital twin had "
                "this at 1.254 kg, which was wrong; it was corrected before any "
                "training started."
            ),
        })

    if placement:
        facts.append({
            "label": "Where the servos sit",
            "value": (
                f"{placement.get('base_link', 0)} in the body, "
                f"{placement.get('hip', 0)} at the shoulders, "
                f"{placement.get('thigh', 0)} on the thighs, "
                f"{placement.get('shank', 0)} on the shanks"
            ),
            "note": (
                "Nothing hangs off the shank, which keeps the swinging end of "
                "each leg light."
            ),
        })

    if reach_mm:
        facts.append({
            "label": "Leg reach",
            "value": f"about {reach_mm:.0f} mm fully extended",
            "note": "Thigh plus shank, measured off the CAD model.",
        })

    if electronics_g:
        facts.append({
            "label": "Electronics in the body",
            "value": f"{electronics_g:.0f} g of bought parts",
            "note": (
                "Raspberry Pi 4B, PCA9685 servo driver, IMU, TFmini-S lidar, "
                "camera, a 5000 mAh battery and hardware. More than a quarter "
                "of the whole robot's weight sits in the trunk."
            ),
        })

    if material:
        facts.append({
            "label": "Material",
            "value": (
                f"SLA resin at {material.get('resin_density_g_cm3')} g/cm3, "
                f"{int(float(material.get('fill_fraction', 0)) * 100)}% infill"
            ),
            "note": (
                "Brittle. This is why one of the training penalties is for "
                "hard foot impacts."
            ),
        })

    out["facts"] = facts
    return out


# ---------------------------------------------------------------------------
# media - photos, renders and clips, each checked to exist
# ---------------------------------------------------------------------------

# Captions are written from what is actually in each frame plus the numbers in
# robot.yaml. Ordered roughly build-order within each group.
_MEDIA_SPEC: dict[str, list[tuple[str, str]]] = {
    "cad": [
        ("Overview/Screenshot 2022-05-28 114828.png",
         "The 2022 CAD model. This exact geometry is what the simulation runs "
         "on: 13 parts, three joints per leg. Rebuilding it as a physics model "
         "was the first job of the 2026 work."),
        ("Overview/Screenshot 2022-05-28 114849.png",
         "The body with the computer hardware laid out - Raspberry Pi, servo "
         "driver, sensors and battery. Those bought parts add 447 g to the "
         "trunk, more than a quarter of the robot's 1.625 kg."),
        ("Overview/IMG_5251.jpg",
         "The hand controller being designed: two thumbsticks, a Raspberry Pi "
         "Zero and a Pico, a screen, 18650 cells and a radio, laid out on the "
         "carrier board before assembly."),
    ],
    "build": [
        ("Overview/photo_2022-09-18_12-46-47.jpg",
         "The bare body frame. Four DS3218MG servos bolt straight into the "
         "black chassis rails - these are the ones that swing each leg out "
         "sideways, and they are the only four mounted in the body itself."),
        ("Overview/photo_2022-09-18_12-46-50.jpg",
         "Close-up of the same frame. The splined metal output horn on each "
         "servo is visible - a horn can only be fitted to the nearest spline, "
         "which is why training randomises a small zero-offset on every joint."),
        ("Overview/photo_2022-09-18_12-46-52.jpg",
         "The frame from above before the legs went on: four sideways-swing "
         "servos, threaded steel cross-rods, and the servo wiring already "
         "loomed and pinned out."),
        ("Overview/photo_2022-09-18_12-46-35.jpg",
         "All four legs finished and off the robot. Each carries its own "
         "shoulder and knee servo, cable-tied down, with a rubber foot."),
        ("Overview/photo_2022-09-18_12-46-38.jpg",
         "The legs, showing the drive that dominates this project: the knee "
         "servo sits on the thigh and pushes a steel rod with blue ball joints "
         "down to the shank. Servo angle is not knee angle - that has to be "
         "measured before any trained gait can run on the real machine."),
        ("Overview/photo_2022-09-18_12-46-41.jpg",
         "A single leg close up. Thigh about 141 mm, shank about 170 mm, so "
         "roughly 312 mm of reach when straight. The push-rod and its ball "
         "joints run down the outside of the thigh."),
        ("Overview/photo_2022-09-18_12-46-43.jpg",
         "Legs waiting for assembly. The honeycomb cut-outs are weight saving - "
         "each printed leg segment is only about 29 g of resin."),
        ("Overview/photo_2022-09-18_12-46-15.jpg",
         "Frame and legs together for the first time. No electronics yet: at "
         "this point the robot was a complete mechanism with nothing to drive "
         "it."),
        ("Overview/photo_2022-09-18_12-46-55.jpg",
         "The assembled robot on the bench. Twelve servos, all wired, all "
         "waiting on software that would not be written for another four years."),
        ("Overview/photo_2022-09-18_12-47-01.jpg",
         "Straight down on the finished mechanism. The four sideways-swing "
         "servos sit inboard along the spine; the shoulder and knee servos ride "
         "out on the legs."),
        ("Overview/photo_2022-09-18_12-47-05.jpg",
         "The main body and legs assembled, on a cutting mat for scale. This is "
         "the machine the simulation is a copy of."),
        ("Overview/photo_2022-09-18_12-46-26.jpg",
         "The computer stack on its printed tray: Raspberry Pi 4B underneath, "
         "the IMU that senses tilt in the middle, and the PCA9685 board on the "
         "right that fans one signal out to all twelve servos."),
        ("Overview/photo_2022-09-18_12-46-30.jpg",
         "The same stack from the side. The red and black pair going off to the "
         "right is servo power, kept separate from the Pi's own supply."),
        ("Overview/photo_2022-09-18_12-46-32.jpg",
         "The tray flipped over, showing the Raspberry Pi 4B and the ribbon of "
         "signal wires running up to the servo driver."),
        ("Overview/RObot_from_angle.JPG",
         "The robot standing with electronics fitted and the nose shell on. The "
         "push-rods and ball joints running down each thigh are clear here."),
        ("Overview/D50167F7-8F12-43D7-A90E-EBB300F50C6A.JPG",
         "Looking straight down on the finished robot: Pi, servo driver and the "
         "full twelve-servo loom packed into the trunk between the four "
         "sideways-swing servos."),
    ],
    "sim": [
        ("sim/models/preview_standing.png",
         "The simulated robot standing. Getting this far meant repairing the "
         "exported model - it originally loaded lying on its side - and "
         "correcting the mass from 1.254 kg to the real 1.625 kg."),
        ("sim/models/preview_walking.png",
         "The same model mid-stride on the hand-written gait. This is the "
         "675 mm-in-12-seconds baseline the AI has to beat."),
    ],
    "gifs": [
        ("Overview/Test.gif",
         "The most recent test on the real hardware."),
        ("Overview/Gifs/1.gif",
         "Early simulation work from 2022, before the project was shelved (1 of 3)."),
        ("Overview/Gifs/2.gif",
         "Early simulation work from 2022, before the project was shelved (2 of 3)."),
        ("Overview/Gifs/3.gif",
         "Early simulation work from 2022, before the project was shelved (3 of 3)."),
    ],
}


def _collect_media(root: str, errors: list) -> dict:
    media: dict[str, list] = {}
    missing = 0
    for group, entries in _MEDIA_SPEC.items():
        found = []
        for rel, caption in entries:
            if os.path.isfile(os.path.join(root, rel.replace("/", os.sep))):
                found.append({"src": _media_url(rel), "caption": caption})
            else:
                missing += 1  # skip silently, but say how many at the end
        media[group] = found
    if missing:
        errors.append(f"{missing} image or clip file(s) listed for the page are missing.")
    return media


# ---------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------


def _build_timeline() -> list:
    return [
        {
            "when": "2021 - 2022",
            "title": "Designed, printed and built",
            "detail": "Twelve servos, a printed resin frame, a Raspberry Pi and "
                      "a hand controller. The mechanical build was finished. "
                      "The software to make it walk never was.",
            "kind": "milestone",
        },
        {
            "when": "November 2022",
            "title": "Work stopped",
            "detail": "The collaborator handling the software left. The robot "
                      "was complete and could not take a step. It sat that way "
                      "for nearly four years.",
            "kind": "problem",
        },
        {
            "when": "1 August 2026",
            "title": "Digital twin built",
            "detail": "The CAD export was Y-up, so the robot loaded lying on "
                      "its side - fixed. The mass model was wrong at 1.254 kg "
                      "and was corrected to 1.625 kg. A working physics model "
                      "of Gray now exists.",
            "kind": "fix",
        },
        {
            "when": "1 August 2026",
            "title": "First walk: 675 mm in 12 seconds",
            "detail": "A hand-written gait, with the feet following fixed "
                      "looping paths and three feet on the ground at all times. "
                      "56 mm/s, 34 mm of sideways drift, no falls. This is the "
                      "score the AI has to beat.",
            "kind": "milestone",
        },
        {
            "when": "1 August 2026",
            "title": "Three rendering faults fixed",
            "detail": "The model was being drawn wrongly in three separate "
                      "ways. Corrected before training, so what is on screen is "
                      "what the physics is actually doing.",
            "kind": "fix",
        },
        {
            "when": "1 August 2026",
            "title": "First AI run failed - it learned to stand still",
            "detail": "Standing perfectly still scored well: it never falls, "
                      "never stamps, never veers off line. The robot went from "
                      "568 mm at the start to -39 mm by round 150 - walking "
                      "backwards - because that was the safer way to score.",
            "kind": "problem",
        },
        {
            "when": "1 August 2026",
            "title": "Scoring fixed, training restarted",
            "detail": "Added points for ground actually covered, always counted "
                      "forwards, so standing still now scores zero on the "
                      "largest term. That is the run shown on this page.",
            "kind": "fix",
        },
    ]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def collect(repo_root: str = ".") -> dict:
    """Read every source on disk and return the dashboard payload."""
    root = os.path.abspath(repo_root)
    errors: list[str] = []

    if not os.path.isdir(root):
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "goal": _build_goal(dict(BASELINE_FALLBACK)),
            "training": _empty_training(),
            "walks": [],
            "baseline_walk": None,
            "robot": _empty_robot(),
            "media": {key: [] for key in _MEDIA_SPEC},
            "timeline": _build_timeline(),
            "errors": [f"Repository folder not found: {root}"],
        }

    walks, baseline_walk = _read_walks(root, errors)

    # Prefer the measured baseline row once it exists; the constants are only a
    # stand-in for the window before the first evaluation has been run.
    if baseline_walk:
        baseline = {key: baseline_walk[key] for key in BASELINE_FALLBACK}
    else:
        baseline = dict(BASELINE_FALLBACK)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "goal": _build_goal(baseline),
        "training": _collect_training(root, errors),
        "walks": walks,
        "baseline_walk": baseline_walk,
        "robot": _collect_robot(root, errors),
        "media": _collect_media(root, errors),
        "timeline": _build_timeline(),
        "errors": errors,
    }


if __name__ == "__main__":
    # Default to the repo root (this file lives in <repo>/dashboard/) so the
    # script works from anywhere.
    where = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    print(json.dumps(collect(where), indent=2))
