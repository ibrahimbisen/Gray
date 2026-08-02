"""Getting Gray off the floor: resting -> sitting -> standing.

    from gray.standup import StandUp
    seq = StandUp()
    for t in seq.times(dt=0.02):          # 50 Hz
        angles = seq.at(t)                # 12 joint angles, radians, in JOINT_ORDER
        send_to_servos(angles)

NO LEARNING, AND THAT IS THE POINT. This started as a reinforcement-learning task -
"learn to get from lying flat to sitting" - and was built as a script first only to
have something for the learned version to beat. The script then beat the premise:

    scripted stand-up, 40 randomly built robots, the full training envelope
    (mass +/-40%, friction 0.4-1.2, servo strength 0.5-4.0x, armature 0.35-3.0x,
    servo horn offsets +/-0.03 rad, start pose jittered +/-0.02 rad)

        stood up successfully   40 / 40      100%
        final trunk height      167-180 mm, mean 175
        failures                0

Nothing was left for a policy to improve. Standing up is a move from one known pose to
another known pose, over about six seconds, with the robot's own weight holding it
against the floor the whole way. It does not need to react to anything, so there is
nothing for feedback to feed back.

That is NOT true of walking, where errors accumulate step after step - which is where
the learning effort belongs.

## WHY IT WORKS WITHOUT KNOWING WHERE THE LEGS ARE

DS3218MG servos report nothing back, so the robot cannot check its own progress. It
does not need to: it starts from a pose gravity puts it in, and every intermediate
target is one the servos can hold statically. The robot is never asked to be anywhere
it could not simply be left.

## THE POSES

Set by the owner by hand in tools/pose_editor.py and checked against the real CAD
meshes for self-collision. See reference/POSES.md for the measurements.

    resting    hips +55 out, thighs -21, knees +39     trunk  42 mm
    sitting    hips   0,     thighs -22, knees +39     trunk 110 mm settled
    standing   hips   0,     thighs   0, knees  +4     trunk 172 mm settled

Standing is 60% of the 312 mm leg, which leaves the knee bent with travel left to
absorb a footfall. Both settled heights are BELOW their commanded ones: everything
sags 14-19 mm under the robot's 1.625 kg, and these figures are the settled values.

## MOVING BETWEEN THEM

Smoothstep, not linear. A linear ramp starts and stops instantly, and twelve servos
all snatching at once is what threw the trunk 330 mm backwards in the residual task.
Smoothstep leaves and arrives at zero speed.

Stdlib and numpy only. This module runs on the Pi.
"""

from __future__ import annotations

import numpy as np

# Order the servo driver expects. Matches train/gray_robot.py JOINT_ORDER and the
# joint names in sim/models/gray.xml.
JOINT_ORDER = (
    "fl_hip", "fl_top", "fl_bottom",
    "fr_hip", "fr_top", "fr_bottom",
    "br_hip", "br_top", "br_bottom",
    "bl_hip", "bl_top", "bl_bottom",
)

# RAW joint angles in radians, which is what the servos take. Not the "out / forward /
# up" convention the pose editor shows: the legs are mirrored and the raw signs differ
# per leg, so these cannot be read as four copies of one leg.
#
# Taken from progress/poses.json, set by hand and checked for self-collision on all 66
# non-jointed part pairs against the real CAD meshes.
RESTING = {
    "fl_hip": 0.9599, "fl_top": 0.3665, "fl_bottom": -0.6807,
    "fr_hip": -0.9599, "fr_top": -0.3665, "fr_bottom": 0.6807,
    "br_hip": 0.9599, "br_top": -0.3665, "br_bottom": 0.6807,
    "bl_hip": -0.9599, "bl_top": 0.3665, "bl_bottom": -0.6807,
}
SITTING = {
    "fl_hip": 0.0, "fl_top": 0.3840, "fl_bottom": -0.6807,
    "fr_hip": 0.0, "fr_top": -0.3840, "fr_bottom": 0.6807,
    "br_hip": 0.0, "br_top": -0.3840, "br_bottom": 0.6807,
    "bl_hip": 0.0, "bl_top": 0.3840, "bl_bottom": -0.6807,
}
STANDING = {
    "fl_hip": 0.0, "fl_top": 0.0, "fl_bottom": -0.0698,
    "fr_hip": 0.0, "fr_top": 0.0, "fr_bottom": 0.0698,
    "br_hip": 0.0, "br_top": 0.0, "br_bottom": 0.0698,
    "bl_hip": 0.0, "bl_top": 0.0, "bl_bottom": -0.0698,
}

# Seconds per leg of the move. Slow enough that the servos are never asked to move
# faster than they can, and the robot is quasi-static throughout - it is never
# relying on momentum to get anywhere, so a slow servo produces a slow stand rather
# than a failed one.
SETTLE_S = 0.4          # sit still first, so the start pose is actually reached
RISE_S = 2.2            # resting -> sitting
PAUSE_S = 1.0           # let the sitting pose settle before pushing up
STAND_S = 2.2           # sitting -> standing
HOLD_S = 1.5            # settle at the top


def _smoothstep(f: float) -> float:
    """Ease in and out. Zero speed at both ends."""
    f = min(1.0, max(0.0, f))
    return f * f * (3.0 - 2.0 * f)


class StandUp:
    """The stand-up sequence as a function of time."""

    def __init__(self, rise_s: float = RISE_S, stand_s: float = STAND_S):
        self._rest = np.array([RESTING[j] for j in JOINT_ORDER])
        self._sit = np.array([SITTING[j] for j in JOINT_ORDER])
        self._stand = np.array([STANDING[j] for j in JOINT_ORDER])
        self._legs = (
            (SETTLE_S, self._rest, self._rest),
            (rise_s, self._rest, self._sit),
            (PAUSE_S, self._sit, self._sit),
            (stand_s, self._sit, self._stand),
            (HOLD_S, self._stand, self._stand),
        )
        self.duration = sum(d for d, _, _ in self._legs)

    def at(self, t: float) -> np.ndarray:
        """The 12 joint angles at time `t` seconds, in JOINT_ORDER."""
        if t <= 0.0:
            return self._rest.copy()
        for span, a, b in self._legs:
            if t <= span:
                return a + (b - a) * _smoothstep(t / span)
            t -= span
        return self._stand.copy()

    def times(self, dt: float = 0.02):
        """Every control tick of the sequence. 0.02 s is the PCA9685's 50 Hz."""
        n = int(round(self.duration / dt))
        return [i * dt for i in range(n + 1)]
