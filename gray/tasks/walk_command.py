"""The velocity command, with a straight-line share that works in both directions.

mjlab's `rel_forward_envs` was doing something worth reading carefully:

    self.vel_command_b[fwd_ids, 0] = self.vel_command_b[fwd_ids, 0].abs().clamp(min=0.3)
    self.vel_command_b[fwd_ids, 1] = 0.0
    self.vel_command_b[fwd_ids, 2] = 0.0

Zeroing sideways and turn is exactly what is wanted - a straight line is a line
with nothing else on it, and the two penalties that hold a line only apply there.
The problem is the first line. It takes the ABSOLUTE VALUE and then clamps it up
to 0.3 m/s.

WHAT THAT COST, ON THE OLD BOX. WALK_SPEED was (0.15, 0.35) and
rel_forward_envs was 0.8. So four attempts in five had their speed forced to at
least 0.3, and three quarters of those landed exactly ON 0.3 because that is
where the clamp puts everything below it. The policy did not train across
0.15-0.35 m/s at all. It trained at 0.30-0.35, with a spike at 0.30 - and the
speed-tracking bar has been failing at 0.071 against 0.05.

WHAT IT WOULD COST ON THE NEW BOX. `abs()` erases the sign. With WALK_SPEED
widened to (-0.35, 0.35), 80% of commands would have been flipped back to
positive - so widening the range would have bought almost nothing, and the
measurement afterwards would have said backward still does not work.

This subclass keeps the good half and drops the clamp: sideways and turn are
zeroed, and the forward speed is left exactly as it was drawn, sign and all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class StraightLineVelocityCommand(UniformVelocityCommand):
    """A share of attempts get a straight line, forwards OR backwards.

    It also OWNS THE LINE. Where the robot was and which way it pointed the
    moment a straight command began is one fact, and until 4 Aug 2026 it was
    stored twice - `veering` kept a heading on the env object, `wandering` kept a
    position and a heading of its own, and `verify.py` re-derived a third from
    the episode start. Three copies of one number, three chances to disagree.

    It lives here because this class already knows the only thing that decides
    it: the instant a command becomes a straight line. Everything else reads it:

        veering            charges the angle off the line
        wandering          charges the distance off the line
        heading_error_obs  TELLS the policy the angle off the line

    That last one is why the move was necessary and not just tidy. An
    observation is computed by a different manager from a reward, and a reward
    term that lazily initialises shared state on first call would hand the
    policy whatever the ordering happened to produce.
    """

    cfg: StraightLineVelocityCommandCfg

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        z = torch.zeros(self.num_envs, device=self.device)
        # Where the line starts and which way it runs. Updated every step in
        # _update_command: held while a straight command is in force, and
        # re-pinned to the robot's own position and heading at every other
        # moment, so a legitimate turn is never charged as drift.
        self.line_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self.line_heading = z.clone()
        # Which way the LINE runs, in world terms, as opposed to which way the
        # robot faces. They were the same number until 4 Aug 2026 because a line
        # only ever meant "straight ahead". On a crab command they differ by 90
        # degrees, and on a diagonal by whatever was drawn - so the direction of
        # travel has to be pinned alongside the facing, or `wandering` measures
        # drift perpendicular to the wrong axis and charges a perfect crab step
        # for the whole distance it was told to cover.
        self.line_course = z.clone()
        self.heading_error = z.clone()      # true, what the rewards charge
        self.heading_error_sensed = z.clone()   # what the policy is told
        # HOW FAR OFF THE LINE, and which side. Added 5 Aug 2026, and it is the
        # other half of straightness rather than a refinement of it.
        #
        # Heading error and cross-track error are the same problem when the robot
        # walks FORWARD: it is off the line because it is pointing wrong, and
        # steering fixes both. They are different problems when it CRABS. There
        # the robot holds its heading and drifts fore and aft, so heading error
        # reads near zero while the distance off the line grows - and `wandering`
        # charges for that distance at up to -3.0 a step. The policy was being
        # fined for an error it had no input for.
        #
        # The evidence it was not a practice problem: raising pure-crab draws
        # about 50-fold moved crab drift 4.69 -> 4.32 deg, while a 41% CUT in
        # turn draws doubled turn error. Exposure moves what the robot can
        # already sense. This is what it could not.
        self.off_track = z.clone()          # true, metres, + is left of the line
        self.off_track_sensed = z.clone()   # what the policy is told
        # The gap between true and sensed is the point. In simulation both the
        # heading and the position are exact; on the real robot they are
        # integrated from the IMU and the joints, and a policy trained on a
        # perfect number will lean on a precision the hardware has not got.
        self._gyro_bias = z.clone()         # startup calibration error, per episode
        self._gyro_walk = z.clone()         # integration drift, accumulating
        self._track_walk = z.clone()        # the same, for the distance estimate

    def reset(self, env_ids=None):
        """Redraw the gyro's error at the start of each episode.

        Deliberately here and not in `_resample_command`: a command is redrawn
        every 5-10 s, but the sensor's calibration error is a property of the
        power-up, so it must last the whole episode.
        """
        out = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        n = self.num_envs if env_ids is None else len(env_ids)
        if n:
            b = self.cfg.gyro_bias_rad
            self._gyro_bias[ids] = torch.empty(
                n, device=self.device).uniform_(-b, b) if b > 0 else 0.0
            self._gyro_walk[ids] = 0.0
            self._track_walk[ids] = 0.0
        return out

    def _on_a_line(self) -> torch.Tensor:
        """Told to travel in a fixed direction: moving somewhere, not told to turn.

        THIS USED TO REQUIRE THE DIRECTION TO BE STRAIGHT AHEAD - forward speed
        above the gate AND sideways below it - and that was the crab-drift bug.
        A sideways command has a direction of travel like any other, but it
        failed the `abs(vx) > g` half of the old test, so `veering`, `wandering`
        and the `off_line` observation all switched off for it. The policy was
        asked to crab in a straight line while being neither shown its error nor
        charged for it, and came in 19 to 41 degrees off. That is the same fault
        forward walking had before it could sense its heading, and it was fixed
        the same way: by giving the policy the number and a reason to care.

        What is asked for now is the SPEED OF TRAVEL in any direction, which is
        what having a line to hold actually means. A pure turn still has no line
        - the robot is meant to rotate, so charging it for rotating would be
        charging it for obeying - and neither does a stop.
        """
        g = self.cfg.straight_gate
        speed = torch.linalg.vector_norm(self.vel_command_b[:, :2], dim=1)
        return (speed > g) & (torch.abs(self.vel_command_b[:, 2]) < g)

    def _update_command(self) -> None:
        super()._update_command()

        pos = (self.robot.data.root_link_pos_w[:, :2]
               - self._env.scene.env_origins[:, :2])
        heading = self.robot.data.heading_w

        # Hold the line while going straight, otherwise re-pin it to here and now.
        # `episode_length_buf > 1` keeps the first step of a fresh episode from
        # locking onto a pose the reset has not finished settling.
        hold = self._on_a_line() & (self._env.episode_length_buf > 1)
        self.line_pos = torch.where(hold.unsqueeze(-1), self.line_pos, pos)
        self.line_heading = torch.where(hold, self.line_heading, heading)

        # The course is the facing plus whatever the command asks for off it:
        # 0 straight ahead, +90 degrees for a pure crab to the left, pi for
        # backward. Pinned at the same instant as the facing and held with it,
        # so a robot that drifts round keeps being measured against the line it
        # was ORIGINALLY sent along rather than one that follows it round.
        course = heading + torch.atan2(self.vel_command_b[:, 1],
                                       self.vel_command_b[:, 0])
        self.line_course = torch.where(hold, self.line_course, course)

        err = heading - self.line_heading
        self.heading_error = torch.atan2(torch.sin(err), torch.cos(err))

        # A gyro's error is a fixed offset plus a slow wander, and it re-zeroes
        # whenever the line does, because that is when the real robot would take
        # its reference. Scaled by sqrt(dt) so the wander over a given number of
        # SECONDS does not change if the control rate ever does.
        walk = self.cfg.gyro_walk_rad_per_s
        if walk > 0:
            step = torch.randn(self.num_envs, device=self.device) * walk * (
                self._env.step_dt ** 0.5)
            self._gyro_walk = torch.where(hold, self._gyro_walk + step,
                                          torch.zeros_like(self._gyro_walk))
        self.heading_error_sensed = (
            self.heading_error + self._gyro_bias + self._gyro_walk)

        # Cross-track distance, in the line's own frame: how far the robot is off
        # the line it was sent along, positive to the left OF THE COURSE. Same
        # rotation `wandering` charges with and `verify.py` scores with, so all
        # three read one number - which was the whole reason the line moved onto
        # this class on 4 Aug.
        away = pos - self.line_pos
        self.off_track = (-away[:, 0] * torch.sin(self.line_course)
                          + away[:, 1] * torch.cos(self.line_course))

        # The real robot has no ruler. It gets this by integrating body-frame
        # velocity, from the IMU and the joints, over the seconds since the line
        # was pinned - the same bounded window that makes the heading input
        # deployable. So the error grows as a random walk from the moment the
        # line is set, and re-zeroes with it. No fixed bias term: a distance
        # estimate starts AT zero by construction, unlike a heading, which
        # inherits whatever the power-up calibration got wrong.
        drift = self.cfg.track_walk_m_per_s
        if drift > 0:
            step = torch.randn(self.num_envs, device=self.device) * drift * (
                self._env.step_dt ** 0.5)
            self._track_walk = torch.where(hold, self._track_walk + step,
                                           torch.zeros_like(self._track_walk))
        self.off_track_sensed = self.off_track + self._track_walk

    def _pin(self, ids: torch.Tensor, axis: int, lo: float, hi: float,
             floor: float) -> None:
        """Give these envs a command along ONE axis and nothing else.

        Draws the speed from (lo, hi) and nudges anything inside `floor` clear of
        it, because a "walk in a straight line" command at 0.01 m/s is a standing
        command wearing the wrong hat - the terms that hold a line would score it
        as drift. Then zeroes the other two axes, so the command is a pure
        direction with no turn and no second component on it.
        """
        if len(ids) == 0:
            return
        v = torch.empty(len(ids), device=self.device).uniform_(lo, hi)
        v = torch.where(v.abs() < floor, torch.sign(v + 1e-9) * floor, v)
        self.vel_command_b[ids] = 0.0
        self.vel_command_b[ids, axis] = v
        self.vel_command_w[ids] = self.vel_command_b[ids]
        self.is_forward_env[ids] = False

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return

        # ONE uniform draw splits both shares, so they cannot overwrite each
        # other: straight takes [0, s), crab takes [s, s + c). Two independent
        # draws would give some envs both, and whichever ran second would win
        # silently.
        roll = torch.rand(len(env_ids), device=self.device)
        s = self.cfg.rel_straight_envs
        straight = roll < s
        crab = (roll >= s) & (roll < s + self.cfg.rel_crab_envs)

        # Straight: undo the parent's forward-only handling for these ids, then
        # redo it without the abs() and the clamp. Cheaper and far less brittle
        # than copying the whole method to change two lines of it.
        lo, hi = self.cfg.ranges.lin_vel_x
        self._pin(env_ids[straight], 0, lo, hi, self.cfg.straight_min_speed)

        # Crab: the same thing sideways, added 5 Aug 2026.
        #
        # WHY IT HAS TO BE ITS OWN SHARE. The three velocities are drawn
        # INDEPENDENTLY, so a PURE crab - sideways, no forward, no turn - needs
        # |vx| under 0.05 out of +/-0.35 and |wz| under 0.05 out of +/-1.0 at the
        # same time. That is 14% x 5%, about one draw in 350 of what is left
        # after the straight and standing shares are taken.
        #
        # And a pure crab at the box edge is exactly what verify.py scores, so
        # the robot was graded on the one command it almost never saw. It failed
        # crab drift on all three gate seeds - 4.33, 4.79, 4.95 deg against a 4.0
        # bar - while passing the other ten every time, and walking FORWARD at
        # 2.9 to 3.4 deg with the same terms and the same test length. The gap is
        # exposure, not ability.
        #
        # Weight tuning had already run out by then: `wandering` at -3.0 was the
        # best of four corners, and -6.0 measured 4.98, no better. Nothing left to
        # buy with a bigger number.
        lo, hi = self.cfg.ranges.lin_vel_y
        self._pin(env_ids[crab], 1, lo, hi, self.cfg.crab_min_speed)

        # Spin: turning on the spot, added 6 Aug 2026 - the third payment on
        # the same lesson. The three velocities are drawn independently, so a
        # PURE spin - turn, no travel - needs |vx| AND |vy| under 0.05 at the
        # same time: about one draw in 80. The turn bar scores nothing else,
        # and scores it at 1.00 rad/s, the edge of the range. The robot was
        # never unable to turn - driven, it turns at about three quarters of
        # any rate it is asked - it had just never practised turning WITHOUT
        # travel. Exposure, not ability, exactly as with the crab above.
        spin = (roll >= s + self.cfg.rel_crab_envs) & \
            (roll < s + self.cfg.rel_crab_envs + self.cfg.rel_spin_envs)
        lo, hi = self.cfg.ranges.ang_vel_z
        self._pin(env_ids[spin], 2, lo, hi, self.cfg.spin_min_rate)


@dataclass
class StraightLineVelocityCommandCfg(UniformVelocityCommandCfg):
    """`rel_forward_envs` is left at 0. Use `rel_straight_envs` instead."""

    rel_straight_envs: float = 0.0
    # The share that gets a PURE SIDEWAYS command, for the same reason
    # `rel_straight_envs` exists: an independent draw almost never produces one.
    # See _resample_command - about 1 in 350, against a test that scores nothing
    # else. 0.0 here so no other task changes behaviour; the walk task sets it.
    rel_crab_envs: float = 0.0
    # Below this the command is a stop, not a direction, and the terms that hold
    # a line switch off anyway. Matches walk_env_cfg's MOVING with room to spare.
    straight_min_speed: float = 0.10
    # The same floor for the crab share. Lower than straight_min_speed because
    # the sideways range is +/-0.20 against forward's +/-0.35 - the same fraction
    # of the range, so a crab draw is not pushed to the edge more often than a
    # straight one is. Still well clear of the 0.05 gate.
    crab_min_speed: float = 0.06
    # The share that gets a PURE TURN command - the third exposure fix, same
    # shape as the two above. An independent draw produces a pure spin about
    # once in 80. 0.0 here so no other task changes behaviour; the walk task
    # carries the field and --spin-share on train.py sets it for a probe.
    rel_spin_envs: float = 0.0
    # The floor for the spin share, in rad/s. Below about 0.3 a turn command
    # is a stand with a wobble - the 0.05 gate reads it as movement but the
    # feet barely have to step to satisfy it.
    spin_min_rate: float = 0.30
    # What counts as "told to hold a line": SPEED OF TRAVEL above this in any
    # direction, and turn below it. Same 0.05 the reward gate used, in one place
    # now. It required the travel to be forward until 4 Aug 2026, which switched
    # every line-holding term off for a crab command - see `_on_a_line`.
    straight_gate: float = 0.05
    # How wrong the heading the POLICY reads is allowed to be, so it does not
    # learn to trust a number the hardware cannot produce. Gray reads heading by
    # integrating three trunk gyros; the reference re-zeroes every time a
    # straight command begins, so the integration window is seconds, not minutes.
    #   bias  a fixed offset for the whole episode - calibration at power-up
    #   walk  a random wander, in radians per root-second, so 0.008 is about
    #         2 degrees of accumulated drift over a 20 s episode
    # Both at 0 give a perfect heading, which is what the sim would hand over if
    # nobody asked. That is the setting that does not transfer.
    gyro_bias_rad: float = 0.009        # about half a degree
    gyro_walk_rad_per_s: float = 0.008
    # The same idea for the cross-track distance the policy READS. Integrated
    # body-frame velocity drifts faster than integrated gyro yaw, because it
    # compounds the attitude error as well as its own - so this is deliberately
    # coarser than the heading numbers above.
    #
    # 0.02 gives about 60 mm of accumulated error over a 10 s command, against a
    # `wandering` allowance of 50 mm and a bar the robot misses by roughly
    # 350 mm. So the estimate is good enough to steer by and nowhere near good
    # enough to cheat with, which is the setting that transfers.
    track_walk_m_per_s: float = 0.02

    def build(self, env: ManagerBasedRlEnv) -> StraightLineVelocityCommand:
        return StraightLineVelocityCommand(self, env)
