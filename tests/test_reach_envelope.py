"""The arm's reach is a measured fact, and three places disagreed about it.

Before this test the same quantity appeared as three different numbers:

    profile reach_max_m        0.55   (a plausibility bound, used as if it were reach)
    jog loop clip              0.42   (hardcoded, the tightest, so it won)
    triangulation gate         0.42   (hardcoded again)
    what the arm actually does 0.47   (nobody had measured it)

The visible symptom was an operator jogging the arm straight out, watching it stop
5 cm short of its real limit with nothing logged, and concluding it would not
straighten. The mission meanwhile accepted map targets out to 55 cm and only
discovered they were unreachable after driving there.

Reach is also strongly PITCH-dependent — about 17 cm between a flat wrist and a
vertical one — so "how far can it reach" has no single answer and the steep grasp
pitches tried first reach markedly less than the profile's ceiling.

    pytest tests/test_reach_envelope.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from rax.manipulation.arms.ik_strategy import make_ik  # noqa: E402
from rax.manipulation.arms.kinematics import make_kinematics  # noqa: E402
from rax.robots.profiles import load_profile  # noqa: E402

TABLE_Z = 0.02
TOL_M = 0.003


@pytest.fixture(scope="module")
def rig():
    p = load_profile("so101")
    kin = make_kinematics(p.urdf_path, p.ee_frame, list(p.joint_names))
    return p, make_ik(kin, p)


def max_reach(ik, profile, pitch_deg, z_m=TABLE_Z, step=0.01):
    """Largest radius the bare solver reaches at this pitch. Scans rather than
    bisects: the reachable set is NOT contiguous — shallow pitches have an
    elbow-flip dead band close in, so a bisection anchored at a near radius
    concludes the whole pitch is unreachable."""
    seed = np.array(profile.home_deg, dtype=np.float64)
    best = 0.0
    for r in np.arange(0.15, 0.56, step):
        _q, e = ik.solve(seed, np.array([r, 0.0, z_m]), pitch_deg=pitch_deg, roll_deg=0.0)
        if e < TOL_M:
            best = float(r)
    return best


def test_the_profiles_grasp_reach_matches_what_the_solver_achieves(rig):
    p, ik = rig
    flattest = max(max_reach(ik, p, pitch) for pitch in (0.0, 10.0, 15.0, 20.0))
    assert abs(flattest - p.reach_grasp_max_m) <= 0.02, (
        f"profile claims {p.reach_grasp_max_m*100:.0f}cm, solver achieves "
        f"{flattest*100:.0f}cm — re-derive with workspace.analyze_workspace")


def test_a_flat_wrist_reaches_much_further_than_a_vertical_one(rig):
    """~17 cm of reach is spent going from a flat wrist to a vertical one."""
    p, ik = rig
    flat, steep = max_reach(ik, p, 10.0), max_reach(ik, p, 90.0)
    assert flat > steep + 0.10, f"flat {flat*100:.0f}cm vs steep {steep*100:.0f}cm"


def test_reach_falls_off_monotonically_as_the_wrist_steepens(rig):
    p, ik = rig
    pitches = [10.0, 30.0, 50.0, 70.0, 90.0]
    reaches = [max_reach(ik, p, pt) for pt in pitches]
    for a, b, pa, pb in zip(reaches, reaches[1:], pitches, pitches[1:]):
        assert a >= b - 0.005, f"reach rose from pitch {pa} ({a:.3f}) to {pb} ({b:.3f})"


def test_the_first_grasp_pitch_tried_reaches_far_less_than_the_ceiling(rig):
    """75 deg is DEFAULT_GRASP_PITCHES[0]; it reaches ~13cm short of the ceiling, so
    a target inside reach_grasp_max_m is not thereby reachable at the pitch tried
    first. plan_pitch has to keep walking the list — this pins that it must."""
    p, ik = rig
    assert max_reach(ik, p, 75.0) < p.reach_grasp_max_m - 0.08


def test_the_plausibility_bound_is_looser_than_the_grasp_bound(rig):
    """They are different questions: 'could a localization be real' vs 'can we get
    there'. Collapsing them either hides distant objects or drives at unreachable ones."""
    p, _ = rig
    assert p.reach_max_m > p.reach_grasp_max_m


def test_a_target_past_the_grasp_ceiling_really_does_not_solve(rig):
    p, ik = rig
    seed = np.array(p.home_deg, dtype=np.float64)
    r = p.reach_grasp_max_m + 0.04          # the 51cm case from the live abort
    for pitch in (0.0, 15.0, 45.0, 75.0, 90.0):
        _q, e = ik.solve(seed, np.array([r, 0.0, TABLE_Z]), pitch_deg=pitch, roll_deg=0.0)
        assert e >= TOL_M, f"{r*100:.0f}cm should not solve at pitch {pitch}"
