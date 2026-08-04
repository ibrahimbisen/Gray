"""The gamepad, read straight from Windows. No extra packages.

    Playground\\pad.bat            watch what the pad is doing, live
    Playground\\pilot.bat --pad    drive the robot with it

Windows has carried a joystick API since the nineties - `joyGetPosEx` in
winmm.dll - and any pad that shows up in Device Manager as a "HID-compliant game
controller" is readable through it with nothing installed. That matters more
than it sounds: the alternative is SDL, which means adding a package to an
environment where the venv's own python is already blocked by Application
Control.

**A stick is not a key.** The keyboard version of this has to make speeds sticky,
because a window only reports keys going down. A stick reports where it IS,
sixty times a second, so the command follows the thumb exactly - push half way,
walk half speed. That is how the robot will really be driven, and it is why the
Joy-Cons were always the plan.

**It is polled on the physics thread**, once per simulation step, not from the
window's event loop. So it keeps working when the MuJoCo window is not the
focused window - you can have the dashboard in front and still be driving.

Axis names are Windows', not the pad's: X and Y are the left stick, and the
right stick lands on some pair of Z, R, U, V depending on the pad. Run
`Playground\\pad.bat` and push each stick to see which is which; the map below is
one line to change.
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path

# Run directly for the probe, or imported by pilot.py - either way the repo root
# has to be on the path before the line below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Playground.control import CEIL_FWD, CEIL_SIDE, CEIL_TURN, Pilot  # noqa: E402

winmm = ctypes.WinDLL("winmm")

JOY_RETURNALL = 0x000000FF
CENTRE = 32767.5      # what winmm calls the middle of an axis
HALF = 32767.5

# Below this, a stick counts as centred. Every stick rests a little off centre
# and a robot that creeps forward when nobody is touching anything is a robot
# that looks broken.
DEADZONE = 0.12


class JOYINFOEX(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("dwXpos", wintypes.DWORD), ("dwYpos", wintypes.DWORD),
        ("dwZpos", wintypes.DWORD), ("dwRpos", wintypes.DWORD),
        ("dwUpos", wintypes.DWORD), ("dwVpos", wintypes.DWORD),
        ("dwButtons", wintypes.DWORD), ("dwButtonNumber", wintypes.DWORD),
        ("dwPOV", wintypes.DWORD), ("dwReserved1", wintypes.DWORD),
        ("dwReserved2", wintypes.DWORD),
    ]


class JOYCAPS(ctypes.Structure):
    _fields_ = [
        ("wMid", wintypes.WORD), ("wPid", wintypes.WORD),
        ("szPname", ctypes.c_wchar * 32),
        ("wXmin", wintypes.UINT), ("wXmax", wintypes.UINT),
        ("wYmin", wintypes.UINT), ("wYmax", wintypes.UINT),
        ("wZmin", wintypes.UINT), ("wZmax", wintypes.UINT),
        ("wNumButtons", wintypes.UINT),
        ("wPeriodMin", wintypes.UINT), ("wPeriodMax", wintypes.UINT),
        ("wRmin", wintypes.UINT), ("wRmax", wintypes.UINT),
        ("wUmin", wintypes.UINT), ("wUmax", wintypes.UINT),
        ("wVmin", wintypes.UINT), ("wVmax", wintypes.UINT),
        ("wCaps", wintypes.UINT), ("wMaxAxes", wintypes.UINT),
        ("wNumAxes", wintypes.UINT), ("wMaxButtons", wintypes.UINT),
        ("szRegKey", ctypes.c_wchar * 32),
        ("szOEMVxD", ctypes.c_wchar * 260),
    ]


AXES = ("X", "Y", "Z", "R", "U", "V")


def find_pad() -> int | None:
    """The first joystick id Windows will actually answer for."""
    for jid in range(16):
        info = JOYINFOEX()
        info.dwSize = ctypes.sizeof(info)
        info.dwFlags = JOY_RETURNALL
        if winmm.joyGetPosEx(jid, ctypes.byref(info)) == 0:
            return jid
    return None


def describe(jid: int) -> str:
    caps = JOYCAPS()
    if winmm.joyGetDevCapsW(jid, ctypes.byref(caps), ctypes.sizeof(caps)) != 0:
        return f"id {jid}"
    return (f"id {jid}: {caps.szPname} - {caps.wNumAxes} axes, "
            f"{caps.wNumButtons} buttons")


def read(jid: int) -> tuple[dict[str, float], int, int] | None:
    """Every axis as -1..+1, the button bits, and the d-pad angle."""
    info = JOYINFOEX()
    info.dwSize = ctypes.sizeof(info)
    info.dwFlags = JOY_RETURNALL
    if winmm.joyGetPosEx(jid, ctypes.byref(info)) != 0:
        return None
    raw = (info.dwXpos, info.dwYpos, info.dwZpos,
           info.dwRpos, info.dwUpos, info.dwVpos)
    return ({name: (v - CENTRE) / HALF for name, v in zip(AXES, raw)},
            info.dwButtons, info.dwPOV)


def _shape(v: float, deadzone: float) -> float:
    """Deadzone, then square the throw so small pushes are gentle.

    A linear stick makes a robot that lurches: the first millimetre of travel is
    already a tenth of full speed. Squaring keeps the fine end fine and leaves
    full deflection at full speed.
    """
    if abs(v) < deadzone:
        return 0.0
    scaled = (abs(v) - deadzone) / (1.0 - deadzone)
    return (scaled * scaled) * (1.0 if v > 0 else -1.0)


class PadPilot(Pilot):
    """A Pilot whose three numbers come from the sticks instead of the keys.

    `unlocked` is what the shoulder button does: normally full deflection means
    the fastest the policy was ever trained on, because that is the only part of
    the command space where its behaviour means anything. Holding the button
    scales the sticks to the hard ceilings instead, for when driving it off the
    edge on purpose is the point.
    """

    def __init__(self, trained, jid: int):
        super().__init__(trained)
        self.jid = jid
        self.connected = True
        self.unlocked = False

        # Everything below is settings, not constants - the panel edits them
        # while the robot is walking, and pilot.py loads whatever was saved last
        # time. The values are what was measured on the PowerA Switch pad:
        # holding the left stick away gave Y = -0.81, the right stick right gave
        # Z = +0.83, so all three are inverted and pushing a stick away means
        # forward.
        self.axis = {"fwd": "Y", "side": "X", "turn": "Z"}
        self.invert = {"fwd": True, "side": True, "turn": True}
        self.deadzone = DEADZONE
        # Switch button order, confirmed the same way: Y=0, B=1, A=2, X=3, L=4,
        # R=5, ZL=6, ZR=7.
        self.button = {"stop": 2, "reset": 1, "unlock": 6}
        # What full deflection is worth. The fastest the policy was ever trained
        # on, so a stick at its stop still asks for something meaningful.
        self.full = {"fwd": trained["fwd"][1],
                     "side": trained["side"][1],
                     "turn": trained["turn"][1]}
        self.ceiling = {"fwd": CEIL_FWD, "side": CEIL_SIDE, "turn": CEIL_TURN}

        self.buttons_down: list[int] = []   # what the panel shows you
        self._prev_buttons = 0
        self._events: list[str] = []

    def poll(self) -> None:
        got = read(self.jid)
        if got is None:
            self.connected = False
            self.stop()
            return
        axes, buttons, _pov = got
        self.connected = True
        self.buttons_down = [b for b in range(16) if buttons & (1 << b)]

        fresh = buttons & ~self._prev_buttons
        self._prev_buttons = buttons
        self.unlocked = bool(buttons & (1 << self.button["unlock"]))
        if fresh & (1 << self.button["reset"]):
            self._events.append("reset")

        if buttons & (1 << self.button["stop"]):
            self.stop()
            return

        def axis_value(name: str) -> float:
            raw = axes.get(self.axis[name], 0.0)
            shaped = _shape(raw, self.deadzone)
            if self.invert[name]:
                shaped = -shaped
            span = self.ceiling[name] if self.unlocked else self.full[name]
            # `+ 0.0` is not decoration: without it a centred stick reads
            # "-0.00" on screen.
            return round(shaped * span, 3) + 0.0

        self.vx = axis_value("fwd")
        self.vy = axis_value("side")
        self.yaw = axis_value("turn")

    def take_events(self) -> list[str]:
        out, self._events = self._events, []
        return out


PAD_KEYS = """\
  left stick        walk - forward, back, and crab sideways
  right stick       turn on the spot
  A                 stop
  B                 put it back on its feet
  ZL (hold)         drive it past the range it was trained on

  Full stick is the fastest it was ever trained on, so you cannot accidentally
  ask for something meaningless. Push half way, walk half speed.

  The sticks work whether or not the window is in front. The mouse still moves
  the camera, and ctrl+drag still shoves the robot.
"""


def probe() -> None:
    """Print what the pad is doing, live, until Ctrl-C."""
    jid = find_pad()
    if jid is None:
        raise SystemExit(
            "Windows is not reporting any joystick.\n"
            "Check it appears in Device Manager under 'HID-compliant game "
            "controller', and that nothing else has grabbed it exclusively.")
    print(describe(jid))
    print("\nPush each stick to its edges and press the buttons. Ctrl-C to stop.")
    print("The axis that moves when you push a stick is the one to map.\n")
    moved = {a: [1.0, -1.0] for a in AXES}
    try:
        while True:
            got = read(jid)
            if got is None:
                print("pad stopped responding")
                return
            axes, buttons, pov = got
            for a in AXES:
                moved[a][0] = min(moved[a][0], axes[a])
                moved[a][1] = max(moved[a][1], axes[a])
            live = "  ".join(f"{a}{axes[a]:+.2f}" for a in AXES)
            span = " ".join(f"{a}:{'MOVED' if moved[a][1] - moved[a][0] > 0.2 else '-'}"
                            for a in AXES)
            pressed = [b for b in range(16) if buttons & (1 << b)]
            print(f"\r{live}   buttons {pressed}   dpad {pov if pov != 65535 else '-'}"
                  f"   [{span}]      ", end="", flush=True)
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\n")


if __name__ == "__main__":
    probe()
