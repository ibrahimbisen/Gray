"""Tests for Gray's leg kinematics.

The FK test is the important one: it checks our closed-form solution against MuJoCo's
own forward kinematics on the real model. If those agree, the geometry in robot.yaml
is right and every downstream gait is built on solid ground. If they ever diverge,
something upstream changed - re-run tools/extract_kinematics.py.

    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from gray.kinematics import LEGS, joint_vector, load_legs, split_joints

JOINT_LIMIT = 2.3562  # +/-135 deg, DS3218MG 270 deg range
MJCF = "sim/models/gray.xml"


@pytest.fixture(scope="module")
def legs():
    return load_legs()


@pytest.fixture(scope="module")
def mj():
    mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed (e.g. on the Pi)")
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    foot = {l: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_bottom_collision")
            for l in LEGS}
    return mujoco, model, data, foot


def test_all_four_legs_load(legs):
    assert set(legs) == set(LEGS)


def test_leg_dimensions_are_consistent(legs):
    """Printed parts are identical, so leg lengths must agree to CAD noise."""
    thigh = np.array([l.thigh_len for l in legs.values()])
    shank = np.array([l.shank_len for l in legs.values()])
    assert np.ptp(thigh) < 1e-3, "thigh lengths differ by more than 1 mm"
    assert np.ptp(shank) < 1e-3, "shank lengths differ by more than 1 mm"
    assert 0.13 < thigh.mean() < 0.15
    assert 0.16 < shank.mean() < 0.18


def test_lateral_offset_points_outward(legs):
    """Each foot should splay away from the body's centreline, not into it."""
    for name, leg in legs.items():
        foot_y = leg.forward(np.zeros(3))[1]
        hip_y = leg.mount_pos[1]
        assert abs(foot_y) > abs(hip_y), f"{name} foot is inboard of its hip"
        assert np.sign(foot_y) == np.sign(hip_y), f"{name} foot crossed the centreline"


def test_forward_matches_mujoco(legs, mj):
    """Closed-form FK vs MuJoCo's own kinematics on the real model."""
    mujoco, model, data, foot = mj
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        q = rng.uniform(-1.2, 1.2, 12)
        data.qpos[:3] = 0.0
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:] = q
        mujoco.mj_kinematics(model, data)
        for i, name in enumerate(LEGS):
            mine = legs[name].forward(q[3 * i: 3 * i + 3])
            worst = max(worst, float(np.linalg.norm(mine - data.geom_xpos[foot[name]])))
    assert worst < 1e-6, f"FK disagrees with MuJoCo by {worst*1000:.4f} mm"


def test_inverse_round_trips(legs):
    rng = np.random.default_rng(1)
    worst = 0.0
    for trial in range(2000):
        leg = legs[LEGS[trial % 4]]
        q = rng.uniform(-1.0, 1.0, 3)
        target = leg.forward(q)
        solved = leg.inverse(target, reference=q)
        worst = max(worst, float(np.linalg.norm(leg.forward(solved) - target)))
    assert worst < 1e-6, f"IK round-trip off by {worst*1000:.4f} mm"


def test_inverse_respects_joint_limits(legs):
    rng = np.random.default_rng(2)
    for trial in range(1000):
        leg = legs[LEGS[trial % 4]]
        q = rng.uniform(-1.0, 1.0, 3)
        solved = leg.inverse(leg.forward(q), reference=q)
        assert np.abs(solved).max() <= JOINT_LIMIT + 1e-9


def test_inverse_stays_on_one_branch(legs):
    """Walking a foot along a path must not make the knee flip mid-stride."""
    leg = legs["fl"]
    start = leg.forward(np.array([0.0, 0.3, -0.6]))
    prev = leg.inverse(start)
    for step in np.linspace(0.0, 0.05, 40):
        target = start + np.array([step, 0.0, 0.0])
        q = leg.inverse(target, reference=prev)
        assert np.abs(q - prev).max() < 0.25, "branch flip during a continuous path"
        prev = q


def test_unreachable_targets_raise(legs):
    leg = legs["fl"]
    with pytest.raises(ValueError):
        leg.inverse(leg.mount_pos + np.array([0.0, 0.0, -0.45]))   # beyond full reach
    with pytest.raises(ValueError):
        leg.inverse(leg.mount_pos.copy())                          # on the roll axis


def test_joint_vector_round_trips(legs):
    per_leg = {name: np.arange(3) + 10 * i for i, name in enumerate(LEGS)}
    packed = joint_vector(per_leg)
    assert packed.shape == (12,)
    for name, q in split_joints(packed).items():
        assert np.allclose(q, per_leg[name])
