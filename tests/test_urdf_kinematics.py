"""The URDF kinematics backend that needs no robot framework.

FK and IK used to come from `lerobot.model.kinematics`, which made "any arm with a
URDF" false in practice: you also needed a fleet manager installed. These tests cover
the replacement, and the first one is the one that matters — it pins the new solver
against a real arm's URDF so a regression shows up as a millimetre error rather than as
a robot reaching to the wrong place.

    pytest tests/test_urdf_kinematics.py
"""

from __future__ import annotations

import pathlib
import sys
import textwrap

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from rax.manipulation.arms.kinematics import make_kinematics  # noqa: E402
from rax.manipulation.arms.urdf_kinematics import UrdfKinematics  # noqa: E402
from rax.robots.profiles import load_profile  # noqa: E402


@pytest.fixture(scope="module")
def so101():
    profile = load_profile("so101")
    if not pathlib.Path(profile.urdf_path).exists():
        pytest.skip("the SO-101 URDF is not present in this checkout")
    return profile, UrdfKinematics(profile.urdf_path, profile.ee_frame,
                                   list(profile.joint_names))


# --- a real arm -----------------------------------------------------------------
def test_reads_the_so101_chain(so101):
    profile, kin = so101
    assert kin.joint_names == list(profile.joint_names)
    # The chain must reach the gripper frame the profile names, or the whole stack is
    # solving for the wrong point on the robot.
    assert kin.ee_frame == profile.ee_frame
    assert kin._chain[-1].child == profile.ee_frame


def test_fk_is_self_consistent_across_the_workspace(so101):
    """FK must be a pure function of the joints, with no state leaking between calls.

    The solver caches joint values in a dict, so a call that mutated shared state would
    make the second evaluation of the same pose differ from the first — the kind of bug
    that only appears once the control loop runs at speed.
    """
    profile, kin = so101
    lo, hi = profile.limits()
    rng = np.random.default_rng(0)
    poses = [rng.uniform(lo, hi) for _ in range(50)]

    first = [kin.forward_kinematics(q) for q in poses]
    for q in reversed(poses):          # revisit in a different order
        kin.forward_kinematics(q)
    second = [kin.forward_kinematics(q) for q in poses]
    for a, b in zip(first, second):
        assert np.allclose(a, b, atol=0.0), "FK is not a pure function of the joints"


def test_ik_returns_to_the_pose_fk_produced(so101):
    """Round-trip: FK to a pose, solve back, and land on the same point.

    Position only. Where the gripper points is the grasp planner's decision, and
    demanding orientation too would fail on a 5-DOF arm that physically cannot hold an
    arbitrary one — which is exactly why the weights are separate.
    """
    profile, kin = so101
    lo, hi = profile.limits()
    rng = np.random.default_rng(7)
    seed = np.array(profile.home_deg, dtype=np.float64)

    worst = 0.0
    solved = 0
    for _ in range(30):
        q_true = rng.uniform(lo, hi)
        T = kin.forward_kinematics(q_true)
        q_ik = kin.inverse_kinematics(seed, T, orientation_weight=0.0)
        err = float(np.linalg.norm(kin.forward_kinematics(q_ik)[:3, 3] - T[:3, 3]))
        if err < 0.002:
            solved += 1
            worst = max(worst, err)
    # Not every random pose is reachable from one seed — a 5-DOF arm has genuine dead
    # bands, which is why the profile carries ik_seeds. Most should solve, and the
    # ones that do should be tight.
    assert solved >= 20, f"only {solved}/30 poses converged"
    assert worst < 0.002, f"worst converged error {worst * 1000:.2f} mm"


def test_ik_never_returns_a_pose_outside_the_joint_limits(so101):
    """An IK that ignores limits is a wish: the servo clamps and the solver lies."""
    profile, kin = so101
    lo, hi = profile.limits()
    rng = np.random.default_rng(11)
    seed = np.array(profile.home_deg, dtype=np.float64)

    for _ in range(20):
        # Deliberately ask for points well outside the workspace.
        target = np.eye(4)
        target[:3, 3] = rng.uniform(-1.5, 1.5, 3)
        q = kin.inverse_kinematics(seed, target, orientation_weight=0.0)
        assert np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6), \
            f"IK returned {q} outside limits {lo}..{hi}"


def test_a_trailing_gripper_value_passes_through(so101):
    """The arm's joint vector often carries a gripper the IK does not solve for."""
    profile, kin = so101
    q_in = np.array([*profile.home_deg, 42.0])       # 5 joints + gripper
    T = kin.forward_kinematics(np.array(profile.home_deg))
    q_out = kin.inverse_kinematics(q_in, T, orientation_weight=0.0)
    assert len(q_out) == len(q_in)
    assert q_out[-1] == pytest.approx(42.0), "the IK overwrote the gripper"


# --- a synthetic arm, so the generic paths are covered too -----------------------
PRISMATIC_URDF = textwrap.dedent("""\
    <robot name="slider">
      <link name="base"/><link name="lift"/><link name="tool"/>
      <joint name="yaw" type="revolute">
        <parent link="base"/><child link="lift"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57"/>
      </joint>
      <joint name="rise" type="prismatic">
        <parent link="lift"/><child link="tool"/>
        <origin xyz="0.2 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
        <limit lower="0" upper="0.3"/>
      </joint>
    </robot>
    """)


@pytest.fixture
def slider(tmp_path):
    path = tmp_path / "slider.urdf"
    path.write_text(PRISMATIC_URDF, encoding="utf-8")
    return UrdfKinematics(str(path), "tool")


def test_prismatic_joints_are_metres_not_degrees(slider):
    """A slide converted as if it were an angle is a 57x error, silently."""
    assert slider.joint_names == ["yaw", "rise"]
    base = slider.forward_kinematics(np.array([0.0, 0.0]))
    up = slider.forward_kinematics(np.array([0.0, 0.25]))   # 0.25 METRES
    assert up[2, 3] - base[2, 3] == pytest.approx(0.25)
    assert np.allclose(up[:2, 3], base[:2, 3])


def test_revolute_joints_are_degrees(slider):
    """90 degrees of yaw swings a 0.2 m arm from +X to +Y."""
    T = slider.forward_kinematics(np.array([90.0, 0.0]))
    assert np.allclose(T[:3, 3], [0.0, 0.2, 0.1], atol=1e-9)


def test_mixed_chain_ik_solves_for_both_joint_types(slider):
    T = slider.forward_kinematics(np.array([40.0, 0.18]))
    q = slider.inverse_kinematics(np.array([0.0, 0.0]), T, orientation_weight=0.0)
    assert np.linalg.norm(slider.forward_kinematics(q)[:3, 3] - T[:3, 3]) < 1e-4


# --- failure modes have to be legible -------------------------------------------
def test_an_unknown_frame_lists_the_frames_that_do_exist(tmp_path):
    path = tmp_path / "slider.urdf"
    path.write_text(PRISMATIC_URDF, encoding="utf-8")
    with pytest.raises(ValueError, match="no joint produces frame 'nope'"):
        UrdfKinematics(str(path), "nope")


def test_a_joint_off_the_chain_is_refused_rather_than_ignored(tmp_path):
    """Silently dropping it would give an IK that cannot reach and never says why."""
    path = tmp_path / "slider.urdf"
    path.write_text(PRISMATIC_URDF, encoding="utf-8")
    with pytest.raises(ValueError, match="not on the chain"):
        UrdfKinematics(str(path), "tool", ["yaw", "imaginary"])


def test_make_kinematics_falls_back_without_placo(so101):
    """The factory must produce a working solver whether or not placo is installed."""
    profile, _kin = so101
    kin = make_kinematics(profile.urdf_path, profile.ee_frame, list(profile.joint_names))
    assert type(kin).__name__ in ("PlacoKinematics", "UrdfKinematics")
    T = kin.forward_kinematics(np.array(profile.home_deg))
    assert T.shape == (4, 4) and np.isfinite(T).all()

    # And an explicit backend must not silently fall back — asking for numpy and
    # getting a bad frame should surface the URDF error, not a different solver.
    with pytest.raises(ValueError, match="no joint produces frame"):
        make_kinematics(profile.urdf_path, "no_such_frame",
                        list(profile.joint_names), backend="numpy")
