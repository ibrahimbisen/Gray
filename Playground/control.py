"""What the robot is being told to do, and what changes it.

A policy takes exactly three numbers: how fast to go forward, how fast to go
sideways, and how fast to turn. That is the whole interface - everything else it
does, it works out for itself. The Joy-Cons will produce the same three numbers,
and so will whatever drives the real robot, so this file is deliberately free of
anything to do with simulation. It holds three floats and clamps them.

**Commands are sticky.** A key sets a speed and the speed holds until it is
changed, like a throttle rather than an accelerator pedal. That is not a
preference: MuJoCo's viewer reports a key going DOWN and never reports it coming
back up, so "walk while W is held" cannot be written against it. Sticky is also
closer to how the robot will really be driven - a stick position is a speed, not
a nudge.

**This runs on the render thread.** The viewer calls `key_callback` from the
window's own thread while physics is running on another. Touching the simulation
from here would be a race, so nothing in this file knows the simulation exists.
The physics thread reads these floats once per step and that is the only
handover.

**Why the numpad and not WASD.** MuJoCo binds every letter from A to Z to one of
its own view toggles - W is wireframe, S is shadow, A is auto-connect, D is
static bodies, and so on for the whole alphabet (the list is `mujoco.mjVISSTRING`
and `mujoco.mjRNDSTRING`). Those are handled inside MuJoCo's C++ window, before
Python ever sees the key, so they cannot be switched off: WASD steered the robot
and scrambled the picture at the same time. The numpad is the one block MuJoCo
leaves alone, and it reads like a stick anyway - 5 is centre.
"""

from __future__ import annotations

# GLFW key codes for the numpad, which are what the MuJoCo viewer reports.
# Written out rather than imported from mjlab so this file can be used by a
# Joy-Con reader or the real robot without dragging the simulator in with it.
KEY_KP_0 = 320
KEY_KP_2 = 322
KEY_KP_4 = 324
KEY_KP_5 = 325
KEY_KP_6 = 326
KEY_KP_7 = 327
KEY_KP_8 = 328
KEY_KP_9 = 329

# How much one key press moves each number.
STEP_FWD = 0.05     # m/s
STEP_SIDE = 0.05    # m/s
STEP_TURN = 0.10    # rad/s

# The hard stops. Well past anything the policy trained on, on purpose - driving
# it somewhere it has never been is a legitimate thing to want to do, as long as
# the readout says that is what you are doing.
CEIL_FWD = 1.00     # m/s
CEIL_SIDE = 0.50    # m/s
CEIL_TURN = 2.00    # rad/s


def _clamp(v: float, ceiling: float) -> float:
    return round(max(-ceiling, min(ceiling, v)), 3)


class Pilot:
    """The command being sent, and the range the policy was trained on.

    `trained` is the range the policy actually saw during training, passed in
    from the task's own config rather than written down here - a number copied
    into a second file is a number that goes stale.
    """

    def __init__(self, trained: dict[str, tuple[float, float]]):
        self.trained = trained
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0

    # --- what the simulation reads ---

    def command(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.yaw)

    def in_range(self) -> bool:
        """True when this command is one the policy has actually been trained on.

        Outside it, nothing the robot does means anything. A policy that falls
        over at 0.9 m/s has not failed - it has been asked about a part of the
        command space it was never shown. The HUD says so in as many words.
        """
        def inside(v: float, key: str) -> bool:
            lo, hi = self.trained[key]
            return lo - 1e-9 <= v <= hi + 1e-9
        # Standing still is always fair: the walk task trains a share of robots
        # on a full stop, and zero is what the stand and push tasks are.
        if self.vx == 0.0 and self.vy == 0.0 and self.yaw == 0.0:
            return True
        return inside(self.vx, "fwd") and inside(self.vy, "side") \
            and inside(self.yaw, "turn")

    # --- what the keyboard writes ---

    def stop(self) -> None:
        self.vx = self.vy = self.yaw = 0.0

    def middle(self) -> None:
        """The middle of the trained range: straight ahead, at the usual speed."""
        lo, hi = self.trained["fwd"]
        self.vx = round((lo + hi) / 2.0, 3)
        self.vy = 0.0
        self.yaw = 0.0

    def key_callback(self, key: int) -> None:
        if key == KEY_KP_8:
            self.vx = _clamp(self.vx + STEP_FWD, CEIL_FWD)
        elif key == KEY_KP_2:
            self.vx = _clamp(self.vx - STEP_FWD, CEIL_FWD)
        elif key == KEY_KP_4:
            self.yaw = _clamp(self.yaw + STEP_TURN, CEIL_TURN)
        elif key == KEY_KP_6:
            self.yaw = _clamp(self.yaw - STEP_TURN, CEIL_TURN)
        elif key == KEY_KP_7:
            self.vy = _clamp(self.vy + STEP_SIDE, CEIL_SIDE)
        elif key == KEY_KP_9:
            self.vy = _clamp(self.vy - STEP_SIDE, CEIL_SIDE)
        elif key == KEY_KP_5:
            self.stop()
        elif key == KEY_KP_0:
            self.middle()


KEYS = """\
  Numpad. Every letter key belongs to MuJoCo's own view toggles, so driving
  lives here instead. 5 is centre, like a stick.

      7  crab left      8  faster        9  crab right
      4  turn left      5  STOP          6  turn right
                        2  slower
      0  straight ahead at the usual speed

  forward +/- 0.05 m/s   sideways +/- 0.05 m/s   turn +/- 0.10 rad/s

  mouse drag        move the camera     ctrl + drag on the robot: shove it
  space             pause               enter   put it back on its feet
  - / =             slower / faster     P       show the reward plots
"""
