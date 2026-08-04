"""A settings window that sits beside the simulator.

    Playground\\pilot.bat              window plus panel
    Playground\\pilot.bat --no-panel   window only

The MuJoCo window cannot hold a button or a slider - it draws text and nothing
else. So the settings live in a small separate window, built with tkinter, which
ships with Python and needs nothing installed.

What it can change while the robot is walking:

  * which policy is loaded - any run, any checkpoint, swapped without restarting
  * which stick and which button does what, and how big the deadzone is
  * how fast full stick is, and whether it may go past the trained range
  * pause, reset, slow motion

**Threads.** tkinter owns the MAIN thread and the simulation runs on a worker -
that way round, not the other. Tk built from a background thread while MuJoCo
holds a GLFW window kills the process outright on Windows: no exception, no
traceback, both windows simply never appear. It cost an hour to find, so it is
worth saying plainly.

The two sides only ever exchange plain numbers and strings on the pilot object,
plus one request the physics thread drains - the same handover the keyboard
already uses. No tk call is made from the physics thread, and no simulation call
is made from here: a policy swap is REQUESTED here and PERFORMED there, one step
later.

**Nothing here is required.** Close the panel and the simulator carries on;
closing the simulator closes the panel.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = Path(__file__).resolve().parent / "settings.json"
LOG_ROOT = ROOT / "logs" / "rsl_rl"

AXES = ("X", "Y", "Z", "R", "U", "V")


# --- what gets remembered between sessions ---------------------------------


def load_settings(pilot) -> None:
    """Put last session's pad map back, if there is one and it still fits."""
    if not SETTINGS.is_file() or not hasattr(pilot, "axis"):
        return
    try:
        saved = json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for key in ("axis", "invert", "button", "full"):
        for name, value in (saved.get(key) or {}).items():
            if name in getattr(pilot, key):
                getattr(pilot, key)[name] = value
    if isinstance(saved.get("deadzone"), (int, float)):
        pilot.deadzone = float(saved["deadzone"])


def save_settings(pilot) -> None:
    if not hasattr(pilot, "axis"):
        return
    try:
        SETTINGS.write_text(json.dumps({
            "axis": pilot.axis, "invert": pilot.invert,
            "button": pilot.button, "full": pilot.full,
            "deadzone": pilot.deadzone,
        }, indent=2), encoding="utf-8")
    except OSError:
        pass


# --- finding policies -------------------------------------------------------


def all_runs() -> list[Path]:
    """Every rsl_rl run directory that has at least one checkpoint, newest last."""
    if not LOG_ROOT.is_dir():
        return []
    out = [d for exp in sorted(LOG_ROOT.iterdir()) if exp.is_dir()
           for d in sorted(exp.iterdir())
           if d.is_dir() and any(d.glob("model_*.pt"))]
    return out


def checkpoints(run_dir: Path) -> list[str]:
    """Checkpoint names, newest first, sorted by NUMBER not by name.

    model_975 sorts after model_2999 alphabetically, which is how a run's 'last'
    checkpoint ends up being one from a third of the way through.
    """
    files = sorted(run_dir.glob("model_*.pt"),
                   key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0),
                   reverse=True)
    return [f.name for f in files]


# --- the window -------------------------------------------------------------


class Panel:
    """Built and run on the main thread. `run()` blocks until it is closed."""

    def __init__(self, pilot, viewer, run_dir: Path, checkpoint: str):
        self.pilot = pilot
        self.viewer = viewer
        self.run_dir = run_dir
        self.loaded = f"{run_dir.name} / {checkpoint}"
        self.root: tk.Tk | None = None
        # The panel opens while the simulator is still compiling its kernels,
        # so "the viewer is not running" means "not yet" for the first half
        # minute and "closed, follow it" ever after. Without this the panel
        # closes itself a fifth of a second after opening.
        self._seen_running = False

    def run(self) -> None:
        self.root = tk.Tk()
        self.root.title("Gray - playground settings")
        self.root.geometry("380x640")
        self.root.attributes("-topmost", True)
        pad = {"padx": 8, "pady": 3}

        frame = ttk.Frame(self.root, padding=8)
        frame.pack(fill="both", expand=True)

        # --- policy ---
        box = ttk.LabelFrame(frame, text="Policy", padding=6)
        box.pack(fill="x", **pad)

        runs = all_runs()
        names = [d.name for d in runs] or ["(no runs with checkpoints)"]
        self._runs = {d.name: d for d in runs}

        self.run_var = tk.StringVar(value=self.run_dir.name)
        run_pick = ttk.Combobox(box, textvariable=self.run_var, values=names,
                                state="readonly", width=40)
        run_pick.pack(fill="x", pady=2)
        run_pick.bind("<<ComboboxSelected>>", lambda _e: self._refill())

        self.ckpt_var = tk.StringVar()
        self.ckpt_pick = ttk.Combobox(box, textvariable=self.ckpt_var,
                                      state="readonly", width=40)
        self.ckpt_pick.pack(fill="x", pady=2)

        ttk.Button(box, text="Load this policy", command=self._load).pack(
            fill="x", pady=(4, 2))
        self.loaded_label = ttk.Label(box, text=f"running: {self.loaded}",
                                      wraplength=330, foreground="#444")
        self.loaded_label.pack(fill="x")
        self._refill()

        # --- gamepad ---
        if hasattr(self.pilot, "axis"):
            self._build_pad(frame, pad)

        # --- limits ---
        box = ttk.LabelFrame(frame, text="What full stick is worth", padding=6)
        box.pack(fill="x", **pad)
        self.full_vars = {}
        limits = (("fwd", "forward  m/s"), ("side", "sideways  m/s"),
                  ("turn", "turn  rad/s"))
        for key, label in limits:
            row = ttk.Frame(box)
            row.pack(fill="x")
            ttk.Label(row, text=label, width=14).pack(side="left")
            var = tk.DoubleVar(value=self._full(key))
            self.full_vars[key] = var
            value = ttk.Label(row, text=f"{var.get():.2f}", width=5)
            value.pack(side="right")
            ttk.Scale(row, from_=0.0, to=self._ceiling(key), variable=var,
                      command=lambda _v, k=key, lab=value: self._set_full(k, lab)
                      ).pack(side="left", fill="x", expand=True)
        ttk.Label(box, text="trained range is the default - past it, what the "
                            "robot does means nothing",
                  wraplength=330, foreground="#777").pack(fill="x", pady=(4, 0))

        # --- simulation ---
        box = ttk.LabelFrame(frame, text="Simulation", padding=6)
        box.pack(fill="x", **pad)
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Button(row, text="Reset", command=self.viewer.request_reset).pack(
            side="left", expand=True, fill="x")
        ttk.Button(row, text="Pause", command=self.viewer.request_toggle_pause
                   ).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Slower", command=self.viewer.request_speed_down
                   ).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Faster", command=self.viewer.request_speed_up
                   ).pack(side="left", expand=True, fill="x")

        # --- live ---
        box = ttk.LabelFrame(frame, text="Live", padding=6)
        box.pack(fill="x", **pad)
        self.live = ttk.Label(box, text="", font=("Consolas", 9), justify="left")
        self.live.pack(fill="x")

        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._tick()
        self.root.mainloop()

    # --- pad section ---

    def _build_pad(self, parent, pad) -> None:
        box = ttk.LabelFrame(parent, text="Gamepad", padding=6)
        box.pack(fill="x", **pad)

        self.axis_vars, self.invert_vars = {}, {}
        for key, label in (("fwd", "forward"), ("side", "sideways"),
                           ("turn", "turn")):
            row = ttk.Frame(box)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=label, width=9).pack(side="left")
            avar = tk.StringVar(value=self.pilot.axis[key])
            self.axis_vars[key] = avar
            ttk.Combobox(row, textvariable=avar, values=list(AXES), width=4,
                         state="readonly").pack(side="left")
            avar.trace_add("write",
                           lambda *_a, k=key: self.pilot.axis.__setitem__(
                               k, self.axis_vars[k].get()))
            ivar = tk.BooleanVar(value=self.pilot.invert[key])
            self.invert_vars[key] = ivar
            ttk.Checkbutton(row, text="flip", variable=ivar,
                            command=lambda k=key: self.pilot.invert.__setitem__(
                                k, self.invert_vars[k].get())).pack(side="left",
                                                                    padx=6)

        row = ttk.Frame(box)
        row.pack(fill="x", pady=(6, 1))
        ttk.Label(row, text="deadzone", width=9).pack(side="left")
        self.dead_var = tk.DoubleVar(value=self.pilot.deadzone)
        dead_label = ttk.Label(row, text=f"{self.pilot.deadzone:.2f}", width=5)
        dead_label.pack(side="right")
        ttk.Scale(row, from_=0.0, to=0.4, variable=self.dead_var,
                  command=lambda _v: self._set_dead(dead_label)).pack(
                      side="left", fill="x", expand=True)

        self.button_vars = {}
        for key, label in (("stop", "stop"), ("reset", "reset"),
                           ("unlock", "unlock")):
            row = ttk.Frame(box)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=label, width=9).pack(side="left")
            var = tk.IntVar(value=self.pilot.button[key])
            self.button_vars[key] = var
            ttk.Spinbox(row, from_=0, to=15, textvariable=var, width=4,
                        command=lambda k=key: self.pilot.button.__setitem__(
                            k, self.button_vars[k].get())).pack(side="left")
            ttk.Label(row, text="button number", foreground="#777").pack(
                side="left", padx=6)

        self.pressed = ttk.Label(box, text="pressed: -", foreground="#0a0")
        self.pressed.pack(fill="x", pady=(4, 0))
        ttk.Label(box, text="press a button on the pad and read its number here",
                  wraplength=330, foreground="#777").pack(fill="x")

    # --- helpers ---

    def _full(self, key: str) -> float:
        return (self.pilot.full[key] if hasattr(self.pilot, "full")
                else self.pilot.trained[key][1])

    def _ceiling(self, key: str) -> float:
        return (self.pilot.ceiling[key] if hasattr(self.pilot, "ceiling")
                else {"fwd": 1.0, "side": 0.5, "turn": 2.0}[key])

    def _set_full(self, key: str, label) -> None:
        value = round(self.full_vars[key].get(), 2)
        label.config(text=f"{value:.2f}")
        if hasattr(self.pilot, "full"):
            self.pilot.full[key] = value

    def _set_dead(self, label) -> None:
        self.pilot.deadzone = round(self.dead_var.get(), 2)
        label.config(text=f"{self.pilot.deadzone:.2f}")

    def _refill(self) -> None:
        run = self._runs.get(self.run_var.get())
        names = checkpoints(run) if run else []
        self.ckpt_pick["values"] = names
        if names:
            self.ckpt_var.set(names[0])

    def _load(self) -> None:
        run = self._runs.get(self.run_var.get())
        name = self.ckpt_var.get()
        if run is None or not name:
            return
        # Requested here, performed on the physics thread one step later.
        self.viewer.request_policy(run / name)
        self.loaded = f"{run.name} / {name}"

    def _tick(self) -> None:
        """Refresh the live numbers five times a second."""
        if self.root is None:
            return
        told = self.pilot.command()
        doing = self.viewer.doing()
        self.live.config(text=(
            f"told   {told[0]:+.2f} fwd {told[1]:+.2f} side {told[2]:+.2f} turn\n"
            f"doing  {doing[0]:+.2f} fwd {doing[1]:+.2f} side {doing[2]:+.2f} turn\n"
            f"drift  {self.viewer.off_line() * 1000:+.0f} mm     "
            f"falls {self.viewer.falls()}\n"
            f"{'' if self.pilot.in_range() else 'OUTSIDE THE TRAINED RANGE'}"))
        if hasattr(self.pilot, "buttons_down"):
            down = self.pilot.buttons_down
            self.pressed.config(
                text=f"pressed: {down if down else '-'}"
                     + ("" if self.pilot.connected else "   PAD NOT RESPONDING"))
        self.loaded_label.config(text=f"running: {self.viewer.loaded_name()}")
        if self.viewer.is_running():
            self._seen_running = True
        elif self._seen_running:
            self._close()
            return
        self.root.after(200, self._tick)

    def _close(self) -> None:
        save_settings(self.pilot)
        if self.root is not None:
            self.root.destroy()
            self.root = None
