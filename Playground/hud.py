"""The viewer, with a steering wheel bolted on and a readout drawn on top.

mjlab's `NativeMujocoViewer` already does the hard part - it opens a MuJoCo
window, keeps the simulation running at real speed on any hardware, lets the
mouse move the camera and shove the robot, and plots every reward term live.
Two things are added here:

1. **The command.** Once per simulation step, the three numbers the keyboard is
   holding are written into the command the policy reads. That is all steering
   is. It is written before the observation is built, so the policy acts on the
   command in the same step it was given, with no lag.

2. **The readout.** What it was told, against what it actually did.

Why the readout is not just "the speed": a robot's trunk surges at every
footfall, and that ripple was measured at about four times its mean speed. The
raw number is unreadable at 50 Hz. It is smoothed over `FILTER_STEPS`, which is
the same filter `track_speed` in the walk task is scored through - so the number
on screen is the number the reward is paying against, not a second opinion.

The measurements are taken on the physics thread, once per step, and only
FORMATTED when the frame is drawn. Taking them per frame would run the smoothing
at the frame rate instead of the step rate, which silently shortens the filter.
"""

from __future__ import annotations

import math

import mujoco
import torch

from mjlab.viewer import NativeMujocoViewer

from gray.tasks.walk_env_cfg import FILTER_STEPS

# The tilt at which the robot counts as fallen: gravity less than half way down
# the trunk's own vertical. Same test verify.py and drive.py use.
FALLEN = 0.5


class PilotViewer(NativeMujocoViewer):
    def __init__(self, env, policy, pilot, command_name: str = "walk",
                 load_policy=None, loaded_name: str = "?", **kwargs):
        super().__init__(env, policy, **kwargs)
        self.pilot = pilot
        self.command_name = command_name

        # Swapping the policy while it runs. The panel REQUESTS one from its own
        # thread by setting a path; the swap itself happens on the physics
        # thread, between two steps, where touching the runner is safe.
        self._load_policy = load_policy
        self._loaded = loaded_name
        self._policy_request = None

        self._cmd = torch.zeros(3, device=env.unwrapped.device)
        self._smooth: list[float] | None = None   # forward, sideways, turn
        self._ref: tuple[float, float, float] | None = None  # x, y, heading
        self._off = 0.0        # metres off the line it was sent along
        self._turned = 0.0     # radians off the heading it was sent along
        self._upright = 1.0
        self._falls = 0
        self._was_fallen = False

    # --- steering -----------------------------------------------------------

    def _execute_step(self) -> bool:
        self._swap_policy()
        self._write_command()
        ok = super()._execute_step()
        if ok:
            self._measure()
        return ok

    def _swap_policy(self) -> None:
        path = self._policy_request
        if path is None or self._load_policy is None:
            return
        self._policy_request = None
        try:
            self.policy = self._load_policy(path)
        except Exception as exc:  # a bad checkpoint must not kill the window
            self._loaded = f"could not load {path.name}: {exc}"
            return
        self._loaded = f"{path.parent.name} / {path.name}"
        # A new policy starts with a clean sheet - carrying the last one's falls
        # and drift into it would be reading two robots as one.
        self._falls = 0
        self._smooth = None
        self._ref = None
        self.request_reset()

    def _write_command(self) -> None:
        # A gamepad is READ here, on the physics thread, once per step - not
        # from the window's event loop like the keyboard. That is why the sticks
        # keep working when the MuJoCo window is not the one in front.
        poll = getattr(self.pilot, "poll", None)
        if poll is not None:
            poll()
            for event in self.pilot.take_events():
                if event == "reset":
                    self.request_reset()

        term = self.env.unwrapped.command_manager.get_term(self.command_name)
        self._cmd[0], self._cmd[1], self._cmd[2] = self.pilot.command()
        term.vel_command_b[:] = self._cmd

    # --- measuring ----------------------------------------------------------

    def _measure(self) -> None:
        i = self.env_idx
        env = self.env.unwrapped
        d = env.scene["robot"].data

        # One transfer off the GPU per step rather than eight. At 50 Hz the
        # difference is small but it is all on the critical path.
        vx, vy, wz, px, py, head, upright, steps = torch.stack([
            d.root_link_lin_vel_b[i, 0],
            d.root_link_lin_vel_b[i, 1],
            d.root_link_ang_vel_b[i, 2],
            d.root_link_pos_w[i, 0] - env.scene.env_origins[i, 0],
            d.root_link_pos_w[i, 1] - env.scene.env_origins[i, 1],
            d.heading_w[i],
            -d.projected_gravity_b[i, 2],
            env.episode_length_buf[i].to(d.heading_w.dtype),
        ]).tolist()

        # A reset is a new episode: start every running measurement again rather
        # than carrying the last one's motion across.
        fresh = steps <= 1

        if fresh or self._smooth is None:
            self._smooth = [vx, vy, wz]
        else:
            a = 1.0 / FILTER_STEPS
            self._smooth = [s + (n - s) * a
                            for s, n in zip(self._smooth, (vx, vy, wz))]

        # The line it was sent along, locked the moment a straight command
        # starts: where it was AND which way it pointed. A heading with no
        # origin does not define a line. Re-locked whenever the command is not
        # straight, because a commanded turn is not drift.
        cx, cy, cyaw = self.pilot.command()
        straight = cx > 0.05 and abs(cy) < 0.05 and abs(cyaw) < 0.05
        if fresh or not straight or self._ref is None:
            self._ref = (px, py, head)
        rx, ry, rh = self._ref
        mx, my = px - rx, py - ry
        # Cross-track distance, the same rotation the 'wandering' reward and
        # verify.py both measure drift with.
        self._off = -mx * math.sin(rh) + my * math.cos(rh)
        self._turned = (head - rh + math.pi) % (2 * math.pi) - math.pi

        self._upright = upright
        fallen = upright < FALLEN
        if fresh:
            self._was_fallen = False
        elif fallen and not self._was_fallen:
            self._falls += 1
        self._was_fallen = fallen

    # --- what the panel reads and asks for -----------------------------------

    def request_policy(self, path) -> None:
        """Load this checkpoint at the next step. Safe from any thread."""
        self._policy_request = path

    def loaded_name(self) -> str:
        return self._loaded

    def doing(self) -> tuple[float, float, float]:
        return tuple(self._smooth or (0.0, 0.0, 0.0))

    def off_line(self) -> float:
        return self._off

    def falls(self) -> int:
        return self._falls

    # --- drawing ------------------------------------------------------------

    def _set_status_overlay(self, viewer) -> None:
        """Three text boxes: mjlab's status, ours, and the keys.

        mjlab's own version of this method is not called - it would send its box
        to the render thread and ours would need a second round trip every
        frame. Its block is rebuilt here instead so all three go over in one
        call. If mjlab's status text changes, this copy of it will not.
        """
        status = self.get_status()
        capped = " [CAPPED]" if status.capped else ""
        stock = (
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            "Env\nStep\nStatus\nSpeed\nTarget RT\nActual RT",
            f"{self.env_idx + 1}/{self.env.num_envs}\n"
            f"{status.step_count}\n"
            f"{'PAUSED' if status.paused else 'RUNNING'}{capped}\n"
            f"{status.speed_label}\n"
            f"{status.target_realtime:.2f}x\n"
            f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)",
        )

        cx, cy, cyaw = self.pilot.command()
        gx, gy, gyaw = self._smooth or (0.0, 0.0, 0.0)
        warn = "" if self.pilot.in_range() else "\nRANGE"

        labels = "told\ndoing\noff line\nfalls" + warn
        values = (
            f"{cx:+.2f} fwd   {cy:+.2f} side   {cyaw:+.2f} turn\n"
            f"{gx:+.2f} fwd   {gy:+.2f} side   {gyaw:+.2f} turn\n"
            f"{self._off * 1000:+.0f} mm      heading {math.degrees(self._turned):+.0f} deg\n"
            f"{self._falls}           upright {self._upright:.2f}"
            + ("" if not warn else
               "\nOUTSIDE WHAT IT TRAINED ON - a fall here proves nothing")
        )
        mine = (
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT.value,
            labels,
            values,
        )

        if hasattr(self.pilot, "poll"):
            lost = "" if self.pilot.connected else "  - NOT RESPONDING"
            legend = (
                "left stick\nright stick\nA\nB\nZL (hold)\nctrl+drag",
                f"walk and crab{lost}\nturn\nstop\nback on its feet\n"
                f"{'PAST TRAINED RANGE' if self.pilot.unlocked else 'past trained range'}"
                f"\nshove it",
            )
        else:
            legend = (
                "numpad 8 / 2\nnumpad 4 / 6\nnumpad 7 / 9\nnumpad 5\nnumpad 0\n"
                "ctrl+drag",
                "forward\nturn\nsideways\nstop\nstraight ahead\nshove it",
            )
        keys = (
            mujoco.mjtFontScale.mjFONTSCALE_100.value,
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT.value,
            *legend,
        )

        viewer.set_texts([stock, mine, keys])
