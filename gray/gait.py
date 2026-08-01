"""Bezier-curve gait generation for Gray.

This is the classical walking controller: no learning, no neural network. It decides
where each foot should be at every instant, and gray.kinematics turns that into joint
angles. It is also the baseline the RL policy will later learn corrections on top of,
following D2-GMBC (arXiv 2010.12070), so it has to be good on its own - a policy can
only refine a gait that already roughly works.

HOW A GAIT IS BUILT
-------------------
Every leg runs the same cycle, offset in time:

  stance  - foot is planted. It travels BACKWARD relative to the body, which is what
            actually pushes the robot forward. Duration = duty x period.
  swing   - foot is in the air, arcing forward to where it will next land.

`duty` is the fraction of the cycle a foot spends planted, and it is what separates
the gaits:

  CRAWL  duty 0.75, legs a quarter-cycle apart - three feet down at all times. Slow
         and very hard to tip over. Start here.
  TROT   duty 0.5, diagonal pairs move together - two feet down. Faster, and needs
         active balance, which is exactly what the RL layer will be good at.

WHY THE SWING SHAPE MATTERS HERE
--------------------------------
Gray's parts are SLA resin, which is more brittle than printed thermoplastic under
repeated impact - and walking is nothing but repeated impact. The swing curve is
therefore shaped to touch down with low vertical speed rather than to minimise swing
time: the control points bracket the endpoints so the foot eases onto the ground
instead of stamping. The same concern reappears as a reward term in training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gray.kinematics import LEGS, Leg

# Phase offset per leg, in cycles. Order is (fl, fr, br, bl).
GAIT_PATTERNS: dict[str, dict[str, float]] = {
    # Lateral-sequence crawl: FL -> BR -> FR -> BL, the most stable quadruped gait.
    "crawl": {"fl": 0.00, "fr": 0.50, "br": 0.25, "bl": 0.75},
    # Diagonal pairs move together.
    "trot": {"fl": 0.00, "fr": 0.50, "br": 0.00, "bl": 0.50},
    # Lateral pairs - included for completeness; tippy on a robot this leg-heavy.
    "pace": {"fl": 0.00, "fr": 0.50, "br": 0.50, "bl": 0.00},
}

DEFAULT_DUTY = {"crawl": 0.75, "trot": 0.50, "pace": 0.50}


def bezier(points: np.ndarray, s: float | np.ndarray) -> np.ndarray:
    """Evaluate a Bezier curve at s in [0, 1] by De Casteljau's algorithm.

    Used instead of the closed-form Bernstein sum because it stays numerically
    well-behaved with the 7 control points below and costs nothing at 50 Hz.
    """
    pts = np.asarray(points, dtype=float)
    s = np.asarray(s, dtype=float)
    while len(pts) > 1:
        pts = pts[:-1] + (pts[1:] - pts[:-1]) * s
    return pts[0]


@dataclass
class GaitParams:
    """Everything that shapes the walk. All lengths in metres, times in seconds."""

    # Defaults tuned by sweep in MuJoCo (see scripts/walk.py). This combination
    # walks 52.8 mm/s with 0.5 mm of lateral drift over 8 s. Faster settings exist -
    # 160 mm / 0.7 s reaches 71.6 mm/s - but they veer 65 mm in the same distance.
    # Straightness is worth more than speed in a baseline the RL layer refines: drift
    # is asymmetry the policy would otherwise have to spend capacity fighting.
    # Above ~2 strides/s the feet skid and the robot travels BACKWARD.
    pattern: str = "crawl"
    stance_height: float = 0.16   # trunk height above the feet when standing
    step_length: float = 0.12     # total fore-aft travel of a foot per cycle
    step_height: float = 0.035    # peak foot lift during swing
    period: float = 0.6           # one full gait cycle
    duty: float | None = None     # stance fraction; defaults per pattern
    stance_dip: float = 0.004     # push the planted foot slightly into the ground so
                                  # contact is never marginal on a level floor
    width_scale: float = 1.0      # >1 widens the stance for more roll stability

    def __post_init__(self) -> None:
        if self.pattern not in GAIT_PATTERNS:
            raise ValueError(f"unknown gait {self.pattern!r}; "
                             f"choose from {sorted(GAIT_PATTERNS)}")
        if self.duty is None:
            self.duty = DEFAULT_DUTY[self.pattern]
        if not 0.0 < self.duty < 1.0:
            raise ValueError("duty must be strictly between 0 and 1")

    @property
    def offsets(self) -> dict[str, float]:
        return GAIT_PATTERNS[self.pattern]


def swing_control_points(step_length: float, step_height: float) -> np.ndarray:
    """Control points for one swing arc, in the (fore-aft, vertical) stride plane.

    The foot starts at -L/2 (fully behind) and ends at +L/2 (fully ahead). The two
    points just inside each endpoint sit slightly BEYOND it horizontally while still
    near the ground, which gives a near-vertical lift-off and, more importantly, a
    near-vertical descent: the foot arrives with little horizontal speed to scuff and
    little vertical speed to slam.
    """
    L, h = step_length, step_height
    return np.array([
        [-0.50 * L, 0.00],          # lift off
        [-0.62 * L, 0.25 * h],      # rise steeply, drifting back slightly
        [-0.40 * L, 1.15 * h],      # up to cruise height
        [+0.00 * L, 1.25 * h],      # apex
        [+0.40 * L, 1.15 * h],
        [+0.62 * L, 0.25 * h],      # descend steeply
        [+0.50 * L, 0.00],          # touch down
    ])


def foot_offset(phase: float, p: GaitParams) -> np.ndarray:
    """Foot displacement from its neutral position, at a given phase in [0, 1).

    Returns (fore-aft, vertical). Positive fore-aft is forward, positive vertical is up.
    """
    phase = float(phase) % 1.0
    if phase < p.duty:
        # Stance: travel backward at constant speed, pressed into the ground.
        s = phase / p.duty
        return np.array([p.step_length * (0.5 - s), -p.stance_dip])
    # Swing: arc forward through the air.
    s = (phase - p.duty) / (1.0 - p.duty)
    return bezier(swing_control_points(p.step_length, p.step_height), s)


def neutral_targets(legs: dict[str, Leg], p: GaitParams) -> dict[str, np.ndarray]:
    """Where each foot rests when the robot simply stands.

    Feet sit directly below their hip mounts fore-aft, and take the natural lateral
    splay the leg has at zero abduction - so a neutral stance needs no abduction
    effort to hold, which matters when the servos have no feedback and hold position
    by brute stiffness.
    """
    out = {}
    for name, leg in legs.items():
        natural = leg.forward(np.zeros(3))
        out[name] = np.array([leg.mount_pos[0],
                              natural[1] * p.width_scale,
                              -p.stance_height])
    return out


@dataclass
class GaitGenerator:
    """Turns elapsed time into a set of joint angles."""

    legs: dict[str, Leg]
    params: GaitParams = field(default_factory=GaitParams)
    _last_q: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._neutral = neutral_targets(self.legs, self.params)
        # Seed each leg from a standing solve so the first stride starts on the same
        # IK branch it will stay on.
        for name, leg in self.legs.items():
            self._last_q[name] = leg.inverse(self._neutral[name])

    # ---- targets ------------------------------------------------------------------

    def foot_targets(self, t: float, speed: float = 1.0) -> dict[str, np.ndarray]:
        """Foot positions in the trunk frame at time t.

        `speed` scales stride length: 0 stands still, 1 is a full stride, negative
        walks backward. Scaling length rather than frequency keeps every leg's
        touchdown timing identical, which makes the gait far easier to reason about.
        """
        p = self.params
        scaled = GaitParams(**{**p.__dict__, "step_length": p.step_length * speed})
        out = {}
        for name, neutral in self._neutral.items():
            phase = t / p.period + p.offsets[name]
            dx, dz = foot_offset(phase, scaled)
            out[name] = neutral + np.array([dx, 0.0, dz])
        return out

    def joint_angles(self, t: float, speed: float = 1.0) -> dict[str, np.ndarray]:
        """Foot targets solved through IK, staying on a consistent branch."""
        angles = {}
        for name, target in self.foot_targets(t, speed).items():
            q = self.legs[name].inverse(target, reference=self._last_q[name])
            self._last_q[name] = q
            angles[name] = q
        return angles

    def stand(self) -> dict[str, np.ndarray]:
        """Joint angles for the neutral standing pose."""
        return {name: leg.inverse(self._neutral[name])
                for name, leg in self.legs.items()}

    # ---- introspection ---------------------------------------------------------------

    def support_count(self, t: float) -> int:
        """How many feet are planted right now - a quick stability sanity check."""
        p = self.params
        return sum(1 for name in LEGS
                   if (t / p.period + p.offsets[name]) % 1.0 < p.duty)
