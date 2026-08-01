"""The Phase 2 gait, precomputed into a tensor the GPU can read.

WHY THIS EXISTS
---------------
Training runs thousands of robots in parallel on the GPU. `GaitGenerator.joint_angles`
(gray/gait.py) is per-leg NumPy that solves IK four times per call and carries state
between calls to keep the knee on one branch. Calling it once per robot per control
tick would mean thousands of Python-level IK solves every 20 ms, and the gait - not
the physics or the network - would become the entire cost of training.

The way out is a property of the Phase 2 gait: it is **open loop**. Look at
`GaitGenerator.foot_targets`: it reads `t` and `speed` and nothing else. It never
touches robot state. So the nominal joint vector is a pure function of
(phase, stride scale), and a pure function of two bounded inputs can simply be
tabulated once and looked up forever.

This is a precomputation, not an approximation of the method: the table is filled by
running the real `GaitGenerator`, unmodified. `gray/` stays the single code path
shared with the Raspberry Pi, exactly as docs/PROJECT_NOTES.md requires.

TWO DETAILS THAT MATTER
-----------------------
*Branch continuity.* `Leg.inverse` picks the solution nearest a reference pose, so
the joint angles at a given phase depend on how the leg got there. The table is
therefore filled by walking phase forward in order, for **two** full cycles, keeping
only the second - by then the branch tracking has settled into the same steady state
it reaches when actually walking, rather than whatever the initial stand seeded.

*Phase grid alignment.* At the 50 Hz control rate with the default 0.6 s period there
are exactly 30 control ticks per cycle. `N_PHASE` is a multiple of 30, so every phase
the controller actually asks for lands exactly on a grid point and the phase axis
costs no interpolation error at all. Keep it a multiple of 30 if you change it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gray.gait import GaitGenerator, GaitParams
from gray.kinematics import joint_vector, load_legs

# 300 = 10 x the 30 control ticks per 0.6 s cycle, so controller phases hit grid
# points exactly. Speed is the stride scale that walk.py calls --speed.
N_PHASE = 300
N_SPEED = 41          # -1.0 .. +1.0 in steps of 0.05
SPEED_MIN, SPEED_MAX = -1.0, 1.0


def build_table(params: GaitParams | None = None, n_phase: int = N_PHASE,
                n_speed: int = N_SPEED) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate joint angles over (stride scale, phase).

    Returns (speeds, table) where speeds is (S,) and table is (S, P, 12) in the
    joint order `gray.kinematics.joint_vector` produces.
    """
    params = params or GaitParams()
    legs = load_legs()
    speeds = np.linspace(SPEED_MIN, SPEED_MAX, n_speed)
    phases = np.arange(n_phase) / n_phase
    table = np.empty((n_speed, n_phase, 12), dtype=np.float64)

    for i, speed in enumerate(speeds):
        # Fresh generator per speed so branch state cannot leak between rows.
        gen = GaitGenerator(legs, params)
        for cycle in range(2):
            for j, phase in enumerate(phases):
                q = gen.joint_angles((cycle + phase) * params.period, float(speed))
                if cycle == 1:
                    table[i, j] = joint_vector(q)
    return speeds, table


@dataclass
class GaitTable:
    """GPU-side lookup. Call it with per-robot phase and stride scale."""

    speeds: "object"      # torch.Tensor (S,)
    table: "object"       # torch.Tensor (S, P, 12)

    @classmethod
    def build(cls, device, params: GaitParams | None = None, **kw) -> "GaitTable":
        import torch
        speeds, table = build_table(params, **kw)
        return cls(
            speeds=torch.as_tensor(speeds, dtype=torch.float32, device=device),
            table=torch.as_tensor(table, dtype=torch.float32, device=device),
        )

    def __call__(self, phase, speed):
        """phase (N,) in [0,1) and stride scale (N,) -> nominal joint angles (N, 12).

        Bilinear: wrapped along phase (the gait is periodic, so phase P-1 blends back
        into phase 0), clamped along speed.
        """
        import torch

        n_speed, n_phase = self.table.shape[0], self.table.shape[1]

        f = (phase % 1.0) * n_phase
        p0 = torch.floor(f).long() % n_phase
        p1 = (p0 + 1) % n_phase
        wp = (f - torch.floor(f)).unsqueeze(-1)

        g = (speed.clamp(SPEED_MIN, SPEED_MAX) - SPEED_MIN) \
            / (SPEED_MAX - SPEED_MIN) * (n_speed - 1)
        s0 = torch.floor(g).long().clamp(0, n_speed - 1)
        s1 = (s0 + 1).clamp(0, n_speed - 1)
        ws = (g - s0.to(g.dtype)).unsqueeze(-1)

        q00 = self.table[s0, p0]
        q01 = self.table[s0, p1]
        q10 = self.table[s1, p0]
        q11 = self.table[s1, p1]
        lo = q00 + (q01 - q00) * wp
        hi = q10 + (q11 - q10) * wp
        return lo + (hi - lo) * ws


def _walk_cycle(legs, params: GaitParams, speed: float, n: int) -> np.ndarray:
    """Ground truth: walk `n` phases forward for two cycles, return the second.

    Phases are stepped in order rather than jumped to, because `Leg.inverse` resolves
    against the previous pose - sampling a phase out of sequence can land on a
    different IK branch than walking to it would, which would make any comparison
    against the table meaningless.
    """
    gen = GaitGenerator(legs, params)
    out = np.empty((n, 12))
    for cycle in range(2):
        for j in range(n):
            q = gen.joint_angles((cycle + j / n) * params.period, speed)
            if cycle == 1:
                out[j] = joint_vector(q)
    return out


def _selfcheck() -> int:
    """Compare the table against the real GaitGenerator. Run this file to execute it."""
    import torch

    params = GaitParams()
    legs = load_legs()
    speeds, table = build_table(params)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt = GaitTable(
        speeds=torch.as_tensor(speeds, dtype=torch.float32, device=device),
        table=torch.as_tensor(table, dtype=torch.float32, device=device),
    )

    def look_up(ph, sp):
        return gt(torch.tensor(ph, dtype=torch.float32, device=device),
                  torch.tensor(sp, dtype=torch.float32, device=device)).cpu().numpy()

    # 1. The phases the controller actually visits (30 ticks per cycle at 50 Hz), at a
    #    tabulated speed. These land on grid points, so only float32 error should show.
    ticks = np.arange(N_PHASE) / N_PHASE
    exact = _walk_cycle(legs, params, 1.0, N_PHASE)[::N_PHASE // 30]
    aligned = float(np.abs(look_up(ticks[::N_PHASE // 30], np.ones(30)) - exact).max())

    # 2. Off-grid PHASE at a tabulated speed. This is where the gait's own step
    #    discontinuity shows up: foot_offset() drops the foot by `stance_dip` the
    #    instant stance begins and lifts it back the instant swing begins, so the
    #    function being tabulated genuinely jumps twice per cycle. Interpolating
    #    across a jump cannot be accurate at any table resolution - the error is
    #    reported split so it is clear that away from those two phases it is tiny.
    n = 601                      # coprime with 300: nothing lands on a grid point
    ph = np.arange(n) / n
    p = params
    # Every leg runs the same cycle shifted by its own offset, so each contributes two
    # discontinuities at different points in the cycle: one entering stance, one
    # entering swing. For the crawl that is all four quarter-phases, not just two.
    edges = {(p.duty - o) % 1.0 for o in p.offsets.values()} | \
            {(-o) % 1.0 for o in p.offsets.values()}
    at_edge = np.zeros(n, dtype=bool)
    for e in edges:
        at_edge |= np.abs((ph - e + 0.5) % 1.0 - 0.5) < 1.5 / N_PHASE
    err = np.abs(look_up(ph, np.ones(n)) - _walk_cycle(legs, params, 1.0, n)).max(axis=1)
    edge, smooth = float(err[at_edge].max()), float(err[~at_edge].max())

    # 3. Off-grid SPEED at aligned phases - the axis that is genuinely interpolated
    #    at run time, since commanded velocity is continuous.
    ticks30 = np.arange(30) / 30.0
    spd = 0.0
    for sp in (-0.73, -0.12, 0.37, 0.62, 0.88):
        exact_s = _walk_cycle(legs, params, sp, N_PHASE)[::N_PHASE // 30]
        spd = max(spd, float(np.abs(look_up(ticks30, np.full(30, sp)) - exact_s).max()))

    print(f"table              {table.shape} on {device}")
    print(f"aligned phases     {aligned:.3e} rad  ({np.degrees(aligned):.5f} deg)")
    print(f"off-grid speed     {spd:.3e} rad  ({np.degrees(spd):.5f} deg)")
    print(f"off-grid phase     {smooth:.3e} rad  ({np.degrees(smooth):.5f} deg)"
          f"   away from stance/swing edges")
    print(f"  at those edges   {edge:.3e} rad  ({np.degrees(edge):.5f} deg)"
          f"   <- the gait's own {p.stance_dip*1000:.0f} mm stance_dip step")

    # What must hold: the controller only ever asks for grid-aligned phases (30 ticks
    # per cycle at 50 Hz, N_PHASE a multiple of 30), so `aligned` is the number that
    # governs training. Speed is the one axis truly interpolated at run time. The
    # DS3218MG resolves roughly 0.1 deg, so both must sit well under that.
    ok = aligned < 1e-5 and spd < 2e-3 and smooth < 2e-3
    print("OK" if ok else "OUT OF TOLERANCE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
