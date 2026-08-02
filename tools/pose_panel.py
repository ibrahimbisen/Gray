"""The control panel for the pose editor: sliders, checkboxes and buttons.

Kept in its own file because it is pure interface. tools/pose_editor.py owns the
model, the physics and the collision checking; this owns knobs.

WHY A SECOND WINDOW. MuJoCo's viewer has a slider per actuator and nothing else - no
number entry, no grouping, no buttons - so a pose has to be dragged one joint at a
time and can never be landed on a round number. This panel sits beside it and drives
the same values, so the two stay in step whichever one is used.

BUILT ON TKINTER, which ships with Python. A nicer toolkit would mean another install
on a machine that is meant to be training, and this is a tool for setting two poses.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

LEGS = ("fl", "fr", "br", "bl")
LEG_LABEL = {"fl": "front left", "fr": "front right",
             "br": "back right", "bl": "back left"}
SEGMENTS = ("hip", "top", "bottom")
SEG_LABEL = {"hip": "hip  (out +)", "top": "thigh  (forward +)",
             "bottom": "knee  (foot up +)"}
GROUP_LABEL = {"hip": "All hips  -  out +",
               "top": "All thighs  -  forward +",
               "bottom": "All knees  -  foot up +"}

# Rotating the whole robot, as opposed to its joints. Named for what the robot does
# rather than which axis it is, because "rotate about X" tells you nothing while
# standing in front of the machine.
BODY_LABEL = {
    "roll": "Roll  -  right side down +",
    "pitch": "Pitch  -  nose down +",
    "yaw": "Yaw  -  turn left +",
}
BODY_KEYS = tuple(BODY_LABEL)

# Full turn either way. Unlike the joints there is no servo to run out of travel here:
# this is the robot's attitude in the world, not something it is commanded to do.
BODY_LIMIT_DEG = 180.0


class PosePanel:
    """Sliders and buttons. Talks to the editor through the callbacks it is given.

    `get_deg`/`set_deg` work in PHYSICAL degrees - out, forward, up - matching the
    editor's measured sign convention, so a positive number does the same visible
    thing on all four legs. The raw joint angles stay the editor's business.
    """

    # How often the simulation is advanced and the 3D view redrawn, in milliseconds.
    # Driven from Tk's own event loop rather than from a second thread: MuJoCo's
    # viewer copies mjData when it syncs, and touching that data from anywhere else at
    # the same time aborts with "attempting to copy mjData while stack is in use".
    # One thread owns the model, and this is it.
    TICK_MS = 16

    def __init__(self, limit_deg: float, *, set_deg, get_deg, set_invert,
                 get_invert, on_check, on_save, on_load, on_mode, on_reset,
                 get_status, on_tick=None, is_alive=None,
                 set_limp=None, get_limp=None):
        self.limit = limit_deg
        self.set_deg, self.get_deg = set_deg, get_deg
        self.set_invert, self.get_invert = set_invert, get_invert
        self.on_check, self.on_save, self.on_load = on_check, on_save, on_load
        self.on_mode, self.on_reset, self.get_status = on_mode, on_reset, get_status
        self.on_tick, self.is_alive = on_tick, is_alive
        self.set_limp, self.get_limp = set_limp, get_limp
        self._vars: dict[str, tk.DoubleVar] = {}
        self._entries: dict[str, tk.StringVar] = {}
        self._group: dict[str, tk.DoubleVar] = {}
        self._invert: dict[str, tk.BooleanVar] = {}
        self._limits: dict[str, float] = {}
        # Set while a slider is being pushed programmatically, so the callback it
        # triggers does not bounce straight back and fight the user's drag.
        self._quiet = False

    # -- one row: label, slider, number box, optional invert box -------------------

    def _row(self, parent, key: str, label: str, row: int,
             invert_key: str | None = None, limit: float | None = None) -> None:
        ttk.Label(parent, text=label, width=20, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(4, 6), pady=1)

        span = self.limit if limit is None else limit
        var = tk.DoubleVar(value=0.0)
        text = tk.StringVar(value="0.0")
        self._vars[key], self._entries[key] = var, text
        self._limits[key] = span

        def slid(_v=None):
            if self._quiet:
                return
            deg = round(var.get(), 1)
            text.set(f"{deg:.1f}")
            self.set_deg(key, deg)

        scale = ttk.Scale(parent, from_=-span, to=span,
                          variable=var, command=slid, length=250)
        scale.grid(row=row, column=1, sticky="we", padx=2)

        def typed(_e=None):
            try:
                deg = float(text.get())
            except ValueError:
                text.set(f"{var.get():.1f}")
                return
            deg = max(-span, min(span, deg))
            self._quiet = True
            var.set(deg)
            self._quiet = False
            text.set(f"{deg:.1f}")
            self.set_deg(key, deg)

        entry = ttk.Entry(parent, textvariable=text, width=7, justify="right")
        entry.grid(row=row, column=2, padx=(4, 2))
        entry.bind("<Return>", typed)
        entry.bind("<FocusOut>", typed)
        ttk.Label(parent, text="deg").grid(row=row, column=3, sticky="w")

        if invert_key is not None:
            flip = tk.BooleanVar(value=self.get_invert(invert_key))
            self._invert[invert_key] = flip
            ttk.Checkbutton(
                parent, variable=flip,
                command=lambda: self.set_invert(invert_key, flip.get()),
            ).grid(row=row, column=4, padx=(8, 4))

    # -- the window ----------------------------------------------------------------

    def run(self) -> None:
        root = tk.Tk()
        root.title("Gray - pose editor")
        root.geometry("640x900")

        top = ttk.Frame(root, padding=6)
        top.pack(fill="x")
        self.mode = tk.StringVar(value="pose")
        ttk.Radiobutton(top, text="Posing  (no physics, joints stay put)",
                        variable=self.mode, value="pose",
                        command=lambda: self._mode(False)).pack(anchor="w")
        ttk.Radiobutton(top, text="Physics  (gravity on - let it go)",
                        variable=self.mode, value="physics",
                        command=lambda: self._mode(True)).pack(anchor="w")

        self.limp = tk.BooleanVar(
            value=bool(self.get_limp()) if self.get_limp else False)
        ttk.Checkbutton(
            top, text="Limp  -  servos off, the legs just flop under gravity",
            variable=self.limp,
            command=lambda: self.set_limp and self.set_limp(self.limp.get()),
        ).pack(anchor="w", pady=(4, 0))

        grp = ttk.LabelFrame(root, text="Move all four legs together", padding=6)
        grp.pack(fill="x", padx=6, pady=4)
        grp.columnconfigure(1, weight=1)
        for i, seg in enumerate(SEGMENTS):
            self._row(grp, f"group_{seg}", GROUP_LABEL[seg], i)

        body = ttk.LabelFrame(
            root, text="Rotate the whole body  (posing mode only)", padding=6)
        body.pack(fill="x", padx=6, pady=4)
        body.columnconfigure(1, weight=1)
        for i, key in enumerate(BODY_KEYS):
            self._row(body, f"body_{key}", BODY_LABEL[key], i,
                      limit=BODY_LIMIT_DEG)
        ttk.Button(body, text="Lay it flat",
                   command=self._level_body).grid(row=len(BODY_KEYS), column=1,
                                                  sticky="w", pady=(4, 0))

        legs = ttk.LabelFrame(root, text="One leg at a time", padding=6)
        legs.pack(fill="both", expand=True, padx=6, pady=4)
        legs.columnconfigure(1, weight=1)
        r = 0
        for leg in LEGS:
            ttk.Label(legs, text=LEG_LABEL[leg].upper(),
                      font=("TkDefaultFont", 9, "bold")).grid(
                row=r, column=0, sticky="w", pady=(8, 2))
            ttk.Label(legs, text="reverse", foreground="#888").grid(
                row=r, column=4, pady=(8, 2))
            r += 1
            for seg in SEGMENTS:
                key = f"{leg}_{seg}"
                self._row(legs, key, "   " + SEG_LABEL[seg], r, invert_key=key)
                r += 1

        btns = ttk.Frame(root, padding=6)
        btns.pack(fill="x")
        for text, fn in (
            ("Check clipping", self._check),
            ("Reset all to 0", self._reset),
            ("Save as BELLY", lambda: self._save("belly")),
            ("Save as STANDING", lambda: self._save("standing")),
            ("Load belly", lambda: self._load("belly")),
            ("Load standing", lambda: self._load("standing")),
        ):
            ttk.Button(btns, text=text, command=fn).pack(side="left", padx=3)

        self.result = tk.StringVar(value="")
        self.result_label = ttk.Label(root, textvariable=self.result, anchor="w",
                                      justify="left", padding=(6, 2),
                                      wraplength=600)
        self.result_label.pack(fill="x")

        self.status = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.status, anchor="w",
                  justify="left", padding=6, foreground="#0a0").pack(fill="x")

        # THE SIMULATION IS DRIVEN FROM HERE. Everything that touches the model - the
        # physics step, the 3D redraw, reading the trunk height for the status line -
        # happens in this one callback, on Tk's thread. Splitting it across two
        # threads is what crashed the first version.
        counter = {"n": 0}

        def tick():
            if self.is_alive is not None and not self.is_alive():
                root.destroy()             # 3D window closed - close this one too
                return
            if self.on_tick is not None:
                self.on_tick()
            counter["n"] += 1
            if counter["n"] % 15 == 0:     # ~4 Hz, plenty for a text readout
                self.status.set(self.get_status())
            root.after(self.TICK_MS, tick)

        root.after(self.TICK_MS, tick)
        root.mainloop()

    # -- helpers -------------------------------------------------------------------

    def _save(self, name: str) -> None:
        self.on_save(name)
        self._say(f"Saved as {name.upper()}")

    def _check(self) -> None:
        msg = self.on_check()
        self._say(msg or "")

    def _reset(self) -> None:
        extra = self.on_reset() or {}
        self._zero()
        self._quiet = True
        for key, deg in extra.items():
            if key in self._vars:
                self._vars[key].set(deg)
                self._entries[key].set(f"{deg:.1f}")
        self._quiet = False
        self._say("Reset - joints at zero, body level, feet on the floor")

    def _say(self, msg: str) -> None:
        """Show a result on the panel. Red when something is wrong."""
        self.result.set(msg)
        bad = msg.upper().startswith("CLIPPING")
        self.result_label.configure(foreground="#c00" if bad else "#060")

    def _mode(self, physics: bool) -> None:
        """Switch mode, and adopt whatever the robot settled into.

        Leaving physics returns the joint angles and body attitude the solver
        actually produced. Pushing them into the sliders is what stops the next
        slider touch from snapping the robot back to its pre-physics pose.
        """
        settled = self.on_mode(physics)
        if not settled:
            return
        self._quiet = True
        for key, deg in settled.items():
            if key in self._vars:
                span = self._limits.get(key, self.limit)
                deg = max(-span, min(span, deg))
                self._vars[key].set(deg)
                self._entries[key].set(f"{deg:.1f}")
        self._quiet = False

    def _level_body(self) -> None:
        """Put the body back to level without touching the legs."""
        self._quiet = True
        for key in BODY_KEYS:
            name = f"body_{key}"
            self._vars[name].set(0.0)
            self._entries[name].set("0.0")
        self._quiet = False
        for key in BODY_KEYS:
            self.set_deg(f"body_{key}", 0.0)

    def _zero(self) -> None:
        self._quiet = True
        for key, var in self._vars.items():
            var.set(0.0)
            self._entries[key].set("0.0")
        self._quiet = False

    def _load(self, name: str) -> None:
        loaded = self.on_load(name)
        if not loaded:
            return
        self._quiet = True
        for key, deg in loaded.items():
            if key in self._vars:
                self._vars[key].set(deg)
                self._entries[key].set(f"{deg:.1f}")
        for key in self._group:
            self._group[key].set(0.0)
        self._quiet = False
