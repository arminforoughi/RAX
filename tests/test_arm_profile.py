"""Tests for the arm-profile layer and the shared mission state.

The point of a profile is that the algorithms stop assuming the SO-101's shape. These
tests check the two halves of that: the profile really does describe the arm the
monolith hardcoded (same limits, same joint roles), and it refuses to describe an
arm incoherently.

    python tests/test_arm_profile.py
    pytest tests/test_arm_profile.py
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.mission_state import DEFAULT_STATE, Abort, MissionState  # noqa: E402
from robots.profiles import ArmProfile, load_profile  # noqa: E402
from robots.profiles.urdf_limits import joint_limits_deg, read_urdf_limits  # noqa: E402

# The values the monolith hardcoded, repeated here so the test fails if either side
# drifts. stack_mission2.py:4661 for the limits, :4670 for the joint roles.
MONOLITH_J_LO = np.array([-110.0, -100.0, -96.8, -95.0, -157.2])
MONOLITH_J_HI = np.array([+110.0, +100.0, +96.8, +95.0, +162.8])


def test_so101_limits_match_the_hardcoded_table():
    """The URDF-read limits reproduce the hand-transcribed J_LO / J_HI.

    The transcription is rounded to one decimal (96.8 for 96.83), so compare at that
    resolution — but no looser, because these bounds are what stop the IK returning
    poses the servos silently clamp.
    """
    lo, hi = load_profile("so101").limits()
    assert np.allclose(lo, MONOLITH_J_LO, atol=0.05), f"lower: {lo} vs {MONOLITH_J_LO}"
    assert np.allclose(hi, MONOLITH_J_HI, atol=0.05), f"upper: {hi} vs {MONOLITH_J_HI}"


def test_so101_topology_matches_the_hardcoded_ik():
    """The declared topology reproduces what _ik_hold_pitch did with literal indices:
    drive joints 0,1,2; slave joint 3 to hold the pitch; hold joint 4 fixed."""
    p = load_profile("so101")
    assert p.ik == "pitch_hold"
    assert p.pan_joint == 0
    assert p.pitch_chain == (1, 2, 3)
    assert p.roll_joint == 4
    assert p.positioning_joints == (0, 1, 2)
    assert p.slaved_joint == 3
    assert p.n_joints == 5
    assert len(p.ik_seeds) == 5, "the elbow-flip dead band needs all five re-seeds"


def test_so101_named_poses_and_gripper():
    p = load_profile("so101")
    assert p.home_deg == (-14.1, -99.1, 90.8, 33.2, -4.7)
    assert p.view_deg == (5.0, 37.1, 48.1, -40.4, 90.0)
    # SURVEY_TILT was HOME[1:] — the wrist pose held while surveying.
    assert p.survey_tilt_deg == p.home_deg[1:]
    assert p.gripper.hand_uv == (440.0, 394.0)
    assert p.camera.eye_in_hand
    assert p.camera.use_depth is False, "the pick stack runs monocular"


def test_urdf_reader_covers_every_arm_joint():
    p = load_profile("so101")
    limits = read_urdf_limits(p.urdf_path)
    for name in p.joint_names:
        assert name in limits, f"{name} missing from {p.urdf_path}"
    # The gripper is actuated too, so the reader must see it even though it is not an
    # arm joint (the profile drives it separately).
    assert p.gripper.joint_name in limits


def test_urdf_reader_rejects_an_unknown_joint():
    p = load_profile("so101")
    try:
        joint_limits_deg(p.urdf_path, ["shoulder_pan", "no_such_joint"])
    except ValueError as e:
        assert "no_such_joint" in str(e)
    else:
        raise AssertionError("expected a ValueError for a joint the URDF lacks")


def _bad(profile: ArmProfile, expect: str):
    try:
        profile.validate()
    except ValueError as e:
        assert expect in str(e), f"wrong complaint: {e}"
    else:
        raise AssertionError(f"expected validate() to reject: {expect}")


def test_validate_rejects_incoherent_profiles():
    """A profile that lies about the arm is worse than no profile — it makes the IK
    drive the wrong joint. Catch it at load, not at the servo."""
    p = load_profile("so101")
    _bad(dataclasses.replace(p, pan_joint=9), "pan_joint")
    _bad(dataclasses.replace(p, pitch_chain=(1, 1, 3)), "repeats")
    _bad(dataclasses.replace(p, pitch_chain=(1,)), "pitch_chain of 2+")
    _bad(dataclasses.replace(p, pan_joint=None), "needs a pan_joint")
    _bad(dataclasses.replace(p, pitch_chain=(0, 1, 2)), "also in pitch_chain")
    _bad(dataclasses.replace(p, home_deg=(1.0, 2.0)), "home_deg")
    _bad(dataclasses.replace(p, ik_seeds=((None, 1.0),)), "ik_seed")


def test_pose_ik_profile_needs_no_pitch_chain():
    """A generic 6-DOF arm declares ik='pose' and is valid with no pitch chain at all
    — that is the whole point of having two strategies."""
    p = ArmProfile(
        name="generic6",
        urdf="does/not/matter.urdf",
        ee_frame="tool0",
        joint_names=tuple(f"j{i}" for i in range(6)),
        ik="pose",
        limits_deg=(np.full(6, -180.0), np.full(6, 180.0)),
    )
    p.validate()
    assert p.positioning_joints == ()
    assert p.slaved_joint is None


# --- mission state ------------------------------------------------------------------
def test_mission_state_keys_match_the_monolith():
    """/status serializes this dict straight to the UI, so the cold key set is a
    contract with ui/admin.html and the guest page."""
    expected = {
        "phase", "detail", "joints", "gripper", "p_red", "p_green",
        "t0", "running", "loop_hz", "dist_mm",
    }
    assert set(DEFAULT_STATE) == expected
    assert set(MissionState(echo=lambda _: None).snapshot()) == expected
    assert DEFAULT_STATE["phase"] == "IDLE"


def test_set_phase_logs_and_clears_range():
    lines: list[str] = []
    st = MissionState(echo=lines.append)
    st["dist_mm"] = 123.0
    st.set_phase("PICK", "approach 1/3")
    assert st["phase"] == "PICK" and st["detail"] == "approach 1/3"
    assert st["dist_mm"] is None, "the range read-out belongs to the phase that ended"
    assert lines == ["[PICK] approach 1/3"]
    st.set_phase("IDLE")
    assert lines[-1] == "[IDLE]", "no detail => no trailing space"


def test_checkpoint_only_aborts_a_running_mission():
    st = MissionState(echo=lambda _: None)
    st.request_stop()
    st.checkpoint()          # not running: a stale stop must not abort anything
    st.begin_run()
    assert not st.stop_flag.is_set(), "begin_run clears a stale stop request"
    st.checkpoint()
    st.request_stop()
    try:
        st.checkpoint()
    except Abort:
        pass
    else:
        raise AssertionError("expected Abort once running and stopped")
    st.end_run()
    assert st.running is False


def test_log_is_bounded_and_newest_first():
    st = MissionState(log_maxlen=3, echo=lambda _: None)
    for i in range(5):
        st.say(f"line {i}")
    lines = st.log_lines()
    assert len(lines) == 3
    assert "line 4" in lines[0] and "line 2" in lines[-1]


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
