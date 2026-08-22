"""The sixth rig: a camera on the hand, with no depth at all.

Mount and sensor are independent, so the stack has to work across all six combinations
of ``{eye_in_hand, fixed} x {stereo, rgbd, mono}``. ``test_head_camera.py`` covers the
fixed column. This covers the one that was broken — and broken *silently*, which is the
part worth a regression test: the gaze engine read range off the point cloud, a mono rig
never produces one, and the approach ran on a hardcoded 0.25 m placeholder while
reporting healthy progress.

The check that matters is :func:`test_range_is_measured_not_the_placeholder`. Everything
else here could pass with the bug still present.

    pytest tests/test_wrist_mono.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from rax.manipulation.arms.gaze_engine import GazeConfig, GazeEngine  # noqa: E402
from rax.manipulation.arms.kinematics import CartesianKinematics  # noqa: E402
from rax.manipulation.arms.mock_arm import WORLD_UP, MockArmBody, MockObject  # noqa: E402
from rax.manipulation.arms.rig import Rig, geometry_for  # noqa: E402
from rax.models.detection import make_detector, make_mask_tracker  # noqa: E402
from rax.perception.camera_interface import NoDepthSource  # noqa: E402
from rax.perception.cameras.synthetic import SyntheticCamera  # noqa: E402
from rax.perception.depth_cloud import CloudTracker, PointCloudStream  # noqa: E402
from rax.robots.profiles import load_profile  # noqa: E402

#: In the CartesianKinematics base frame (OpenCV: X right, Y down, Z forward).
TRUE_CENTRE = np.array([0.05, 0.0, 0.42])
TRUE_RADIUS = 0.035
QUERY = "red object"


def _rig(kind: str = "mono"):
    """Profile + cameraless arm + wrist camera, joined through the Rig seam."""
    profile = load_profile("wrist_mono")
    objects = [MockObject(TRUE_CENTRE.copy(), TRUE_RADIUS, (225, 40, 40), QUERY)]
    kin = CartesianKinematics()
    arm = MockArmBody(objects=objects, q0=np.array(profile.home_deg, dtype=np.float64))
    geom = geometry_for(profile, kin)

    # The view moves with the hand, so the camera's pose must be read from the arm's
    # CURRENT joints on every frame. A test that rendered from a stale pose would pass
    # while the real rig failed.
    camera = SyntheticCamera(
        lambda: geom.T_base_cam(arm.get_state().joints_deg),
        kind=kind, objects=objects,
        width=profile.camera.width, height=profile.camera.height,
        fx=profile.camera.intrinsics_fallback[0])

    stereo = None
    if kind == "stereo":
        from rax.models.depth.sgbm_stereo import SgbmStereo

        stereo = SgbmStereo(max_disp_px=128)
    return profile, Rig(arm, camera, geom, stereo=stereo), kin


def _engine(rig, kin, profile, **cfg_kw):
    cloud = CloudTracker(make_detector("color_blob"), make_mask_tracker("ellipse"),
                         rig.depth, QUERY, detect_every=1, stream=PointCloudStream())
    cfg = GazeConfig(T_ee_cam=np.eye(4), world_up=WORLD_UP, **cfg_kw)
    return GazeEngine(rig, kin, cloud, cfg, cartesian=True), cloud


# --- the rig itself --------------------------------------------------------------
def test_the_profile_is_the_wrist_plus_mono_combination():
    profile = load_profile("wrist_mono")
    assert profile.camera.mount == "eye_in_hand"
    assert profile.camera.kind == "mono"
    geom = geometry_for(profile, CartesianKinematics())
    assert geom.pose.moves_with_arm

    # This profile's T_ee_cam is the identity — the camera sits exactly ON the EE
    # origin — so the fingertip is at zero depth and has no pixel. Returning None
    # rather than a projection through a near-zero divide is the correct answer.
    assert geom.tip_pixel(np.zeros(profile.n_joints)) is None


def test_an_offset_wrist_mount_can_predict_its_fingertip_pixel():
    """With a real (offset) mount the tip DOES have a constant pixel.

    That predicted pixel and the measured one (``GripperProfile.hand_uv``) are the same
    number computed two ways, so the gap between them reads out hand-eye error. It only
    exists once the camera is somewhere other than the fingertip itself.
    """
    from dataclasses import replace

    profile = load_profile("wrist_mono")
    # 6 cm back along the optical axis and 1 cm up: a plausible wrist bracket. The
    # ratio of those two is what decides where the tip lands vertically — mount the
    # camera high and close and the fingertip falls off the bottom of the frame, which
    # is a real mounting mistake this arithmetic will catch.
    offset = replace(profile, camera=replace(profile.camera, extrinsics="0,-0.01,-0.06,0,0,0"))
    geom = geometry_for(offset, CartesianKinematics())
    uv = geom.tip_pixel(np.zeros(profile.n_joints))
    assert uv is not None
    assert 0 <= uv[0] < offset.camera.width and 0 <= uv[1] < offset.camera.height


def test_the_view_moves_when_the_arm_does():
    """The defining property of eye-in-hand, and the opposite of the head-camera rig."""
    _profile, rig, _kin = _rig()
    T0 = rig.get_observation().T_base_cam.copy()
    rig.send_joint_targets(np.array([0.05, 0.02, -0.08, 0.0, 0.0, 0.0]))
    assert not np.allclose(rig.get_observation().T_base_cam, T0)


def test_a_wrist_mono_rig_has_no_depth_source():
    _profile, rig, _kin = _rig()
    assert isinstance(rig.depth, NoDepthSource)
    assert rig.depth.available is False
    obs = rig.get_observation()
    assert not obs.frame.has_stereo and not obs.frame.has_depth


# --- the regression this file exists for -----------------------------------------
def test_range_is_measured_not_the_placeholder():
    """Range must come from the image, not from the 0.25 m constant in the servo.

    This is the assertion that fails if the apparent-size fallback is removed. The
    object sits at a known distance; the engine is never told it.
    """
    profile, rig, kin = _rig()
    engine, cloud = _engine(rig, kin, profile)

    for _ in range(6):
        engine.step(0.05)

    assert engine.range_m is not None, "the engine never obtained a range"
    assert abs(engine.range_m - 0.25) > 1e-6, \
        "range is still the hardcoded placeholder, not a measurement"
    # Apparent-size ranging off a size prior is coarse — the prior is a generic
    # class width, not this ball's — so the bar is 'the right order of magnitude and
    # the right side of the object', not millimetres.
    true_range = float(TRUE_CENTRE[2])
    assert 0.5 * true_range < engine.range_m < 2.0 * true_range, \
        f"ranged {engine.range_m:.3f} m against a true {true_range:.3f} m"


def test_the_fallback_can_be_switched_off():
    """Turning it off must produce no range at all, not a quietly worse one."""
    profile, rig, kin = _rig()
    engine, _cloud = _engine(rig, kin, profile, apparent_size_fallback=False)
    for _ in range(6):
        engine.step(0.05)
    assert engine.range_m is None


def test_an_elongated_box_is_refused_rather_than_mis_ranged():
    """Bbox width is a bad proxy for a long thin object, and a wrong range closes early."""
    profile, rig, kin = _rig()
    engine, _cloud = _engine(rig, kin, profile)
    obs = rig.get_observation()

    class _Track:
        label, box, has_cloud = QUERY, (100.0, 200.0, 400.0, 230.0), False   # 10:1

    assert engine._depth_from_apparent_size(_Track(), obs) is None

    class _Square(_Track):
        box = (100.0, 200.0, 160.0, 260.0)

    assert engine._depth_from_apparent_size(_Square(), obs) is not None


def test_the_arm_advances_out_of_search_and_closes_the_distance():
    """End to end: the wrist webcam is enough to acquire an object and approach it."""
    profile, rig, kin = _rig()
    engine, cloud = _engine(rig, kin, profile)

    start = float(np.linalg.norm(rig.arm.tool_xyz() - TRUE_CENTRE))
    states = []
    for _ in range(120):
        states.append(engine.step(0.05))

    assert cloud.focus_track() is not None, "never acquired the object"
    assert any(s != "SEARCH" for s in states), f"stuck in SEARCH: {set(states)}"
    end = float(np.linalg.norm(rig.arm.tool_xyz() - TRUE_CENTRE))
    assert end < start, f"the approach moved away: {start:.3f} m -> {end:.3f} m"


# --- and the stereo wrist rig still behaves ---------------------------------------
def test_a_stereo_wrist_rig_prefers_the_cloud_over_apparent_size():
    """The fallback must not displace real depth where real depth exists."""
    profile, rig, kin = _rig("stereo")
    assert rig.depth.available is True
    engine, cloud = _engine(rig, kin, profile)
    for _ in range(6):
        engine.step(0.05)

    focus = cloud.focus_track()
    assert focus is not None
    if focus.has_cloud:
        # With a cloud present the measured range wins, and it should be much better
        # than the prior-based estimate: within a couple of centimetres of truth.
        assert engine.range_m == pytest.approx(float(TRUE_CENTRE[2]), abs=0.05)
