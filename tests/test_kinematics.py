"""The SO-101 profile, its URDF kinematics, the pitch-holding IK and smooth moves."""

import numpy as np
import pytest

from rax.kinematics import MotionLimits, make_ik, make_kinematics, quintic_waypoints
from rax.robots.profiles import load_profile


@pytest.fixture(scope="module")
def so101():
    p = load_profile("so101")
    kin = make_kinematics(p.urdf_path, p.ee_frame, list(p.joint_names))
    return p, kin, make_ik(kin, p)


def test_profile_reads_its_limits_from_the_urdf(so101):
    p, _, _ = so101
    lo, hi = p.limits()
    assert len(lo) == len(hi) == len(p.joint_names) == 5
    assert np.all(lo < hi)
    assert np.all(lo <= p.home_deg) and np.all(np.asarray(p.home_deg) <= hi)


@pytest.mark.parametrize("target, pitch", [((0.22, 0.00, 0.05), 90.0),
                                           ((0.20, 0.10, 0.10), 60.0),
                                           ((0.26, -0.05, 0.16), 0.0)])
def test_ik_reaches_the_point_holding_the_pitch(so101, target, pitch):
    p, kin, ik = so101
    q, err = ik.solve(np.array(p.home_deg, float), np.array(target), pitch_deg=pitch,
                      roll_deg=p.home_deg[p.roll_joint])
    assert err < 0.003
    tip = np.asarray(kin.forward_kinematics(q))[:3, 3]
    assert np.linalg.norm(tip - target) < 0.003
    assert abs(sum(q[i] for i in p.pitch_chain) - pitch) < 1.0


def test_quintic_move_ends_exactly_on_target(so101):
    p, _, _ = so101
    q0 = np.array(p.home_deg, float)
    q1 = q0 + 10.0
    wps, duration = quintic_waypoints(q0, q1, MotionLimits.from_profile(p))
    assert duration > 0 and np.allclose(wps[-1], q1)
    assert np.all(np.abs(np.diff(np.array(wps), axis=0)).max(axis=0) < 1.0)   # no jumps
