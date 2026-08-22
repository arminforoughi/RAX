"""End-to-end proof for the rig this repo is published for: an arm and a head camera.

``test_second_arm.py`` proved a different *arm* plugs in. This proves a different
*camera* does — including the one with no depth at all, which is the configuration most
people asking "can I use this?" actually have.

Three rigs are built from one scene and one arm, differing only in the sensor:

    mono    a webcam on a mast. No depth. Range comes from the table plane.
    rgbd    a RealSense-class sensor. Depth read off the device.
    stereo  a baseline pair. Depth from a matcher.

The arm is :class:`~manipulation.arms.mock_arm.MockArmBody`, which has no camera of its
own — the shape a real driver has — and :class:`~manipulation.arms.rig.Rig` joins it to
each sensor. If a change welds the stack back to stereo, the mono cases here fail; if it
welds it back to a wrist camera, the fixed-mount pose fails. That is the whole point.

    pytest tests/test_head_camera.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rax.manipulation.approach import ApproachConfig, approach_target, stage_step  # noqa: E402
from rax.manipulation.arms.ik_strategy import make_ik  # noqa: E402
from rax.manipulation.arms.kinematics import CartesianKinematics  # noqa: E402
from rax.manipulation.arms.mock_arm import MockArmBody, MockObject  # noqa: E402
from rax.manipulation.arms.rig import Rig, geometry_for  # noqa: E402
from rax.perception.camera_interface import (  # noqa: E402
    NoDepthSource,
    SensorDepthSource,
    StereoDepthSource,
    make_depth_source,
)
from rax.perception.cameras.synthetic import SyntheticCamera  # noqa: E402
from rax.perception.locate import PlaneRayLocalizer  # noqa: E402
from rax.perception.object_priors import PRIORS  # noqa: E402
from rax.robots.profiles import load_profile  # noqa: E402

#: Where the cup really is. Nothing downstream is told this; it renders the view and
#: scores the answer.
TRUE_XY = np.array([0.26, -0.07])
TRUE_LABEL = "cup"
TABLE_Z = 0.0


def _scene():
    """One cup-sized object standing on the table at ``TRUE_XY``."""
    meta = PRIORS.meta(TRUE_LABEL)
    radius = 0.5 * float(np.sqrt(meta["w_m"] * meta["d_m"]))
    centre = np.array([TRUE_XY[0], TRUE_XY[1], TABLE_Z + radius])
    return [MockObject(centre, radius, (220, 60, 60), TRUE_LABEL)], radius


def _rig(kind: str):
    """Profile + arm + camera + geometry, assembled the way a server would."""
    profile = load_profile("head_mono")
    objects, radius = _scene()
    arm = MockArmBody(objects=objects, q0=np.array(profile.home_deg, dtype=np.float64))
    geom = geometry_for(profile)          # fixed mount: no kinematics needed
    fb = profile.camera.intrinsics_fallback
    camera = SyntheticCamera(
        lambda: geom.T_base_cam(), kind=kind, objects=objects,
        width=profile.camera.width, height=profile.camera.height, fx=fb[0])
    stereo = None
    if kind == "stereo":
        from rax.models.depth.sgbm_stereo import SgbmStereo

        stereo = SgbmStereo(max_disp_px=128)
    return profile, Rig(arm, camera, geom, stereo=stereo), radius


def _bbox_of(frame) -> tuple[float, float, float, float]:
    """The detector's job, done by colour so the test needs no model weights."""
    rgb = np.asarray(frame.rgb)
    on = (rgb[:, :, 0] > 120) & (rgb[:, :, 1] < 110) & (rgb[:, :, 2] < 110)
    ys, xs = np.nonzero(on)
    assert xs.size > 20, "the object is not visible in the rendered frame"
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


# --- the profile itself ---------------------------------------------------------
def test_head_mono_profile_is_the_mono_fixed_branch():
    profile = load_profile("head_mono")
    assert profile.camera.kind == "mono"
    assert profile.camera.mount == "fixed"
    geom = geometry_for(profile)
    assert not geom.pose.moves_with_arm
    # A fixed camera has no constant fingertip pixel, and must say so rather than
    # returning a plausible-looking wrong one.
    assert geom.tip_pixel(np.zeros(profile.n_joints)) is None


def test_eye_in_hand_profile_refuses_to_build_without_kinematics():
    """The failure has to be loud: FK is not optional for a wrist camera."""
    with pytest.raises(ValueError, match="kinematics"):
        geometry_for(load_profile("so101"))


# --- the seam picks the right depth strategy ------------------------------------
@pytest.mark.parametrize("kind,expected", [
    ("mono", NoDepthSource), ("rgbd", SensorDepthSource), ("stereo", StereoDepthSource)])
def test_each_camera_kind_gets_its_depth_strategy(kind, expected):
    _profile, rig, _r = _rig(kind)
    assert isinstance(rig.depth, expected)
    assert rig.depth.available is (kind != "mono")


def test_a_stereo_camera_without_a_matcher_is_a_loud_error():
    with pytest.raises(ValueError, match="StereoDepth"):
        make_depth_source("stereo", stereo=None)


# --- the rig produces coherent observations -------------------------------------
@pytest.mark.parametrize("kind", ["mono", "rgbd", "stereo"])
def test_rig_joins_a_cameraless_arm_to_any_camera(kind):
    profile, rig, _r = _rig(kind)
    obs = rig.get_observation()

    assert rig.joint_names == list(profile.joint_names)
    assert obs.joints_deg.shape == (profile.n_joints,)
    assert obs.left.shape[:2] == (profile.camera.height, profile.camera.width)
    assert obs.frame.has_stereo is (kind == "stereo")
    assert obs.frame.has_depth is (kind == "rgbd")
    # The camera is bolted to the world, so moving the arm must NOT move the view.
    T0 = obs.T_base_cam.copy()
    rig.send_joint_targets(np.array([0.30, 0.10, 0.25, 0.0, 0.0, 0.0]))
    assert np.allclose(rig.get_observation().T_base_cam, T0)


def test_intrinsics_come_from_the_device_not_the_config():
    """A camera that delivers a different size than configured must win the argument."""
    profile, rig, _r = _rig("mono")
    rig.camera = SyntheticCamera(lambda: rig.geometry.T_base_cam(), kind="mono",
                                 objects=rig.arm.objects, width=800, height=600, fx=640.0)
    rig._synced = False
    obs = rig.get_observation()
    assert (obs.intrinsics.width, obs.intrinsics.height) == (800, 600)
    assert rig.geometry.fx == pytest.approx(640.0)


# --- localization with no depth at all ------------------------------------------
def test_mono_locates_the_object_off_the_table_plane():
    """The claim under test: a plain webcam is enough to find something to grasp.

    No depth map, no baseline, no learned model — the sightline through the object's
    base pixel is intersected with the measured table. This is the estimator a mono rig
    lives on, and it has to be good to centimetres for a grasp to succeed.
    """
    profile, rig, radius = _rig("mono")
    obs = rig.get_observation()
    loc = PlaneRayLocalizer(rig.geometry, PRIORS, z_plane=TABLE_Z,
                            reach_m=(profile.reach_min_m, profile.reach_max_m))
    fix = loc.locate(_bbox_of(obs.frame), obs.T_base_cam, label=TRUE_LABEL)

    assert fix.ok, f"mono localization failed: {fix}"
    err = float(np.linalg.norm(fix.xy - TRUE_XY))
    assert err < 0.03, f"located {np.round(fix.xy, 3)} vs true {TRUE_XY} — {err*1000:.0f} mm out"


def test_rgbd_depth_lands_on_the_object():
    """The sensor branch: read depth off the device and back-project it."""
    _profile, rig, radius = _rig("rgbd")
    obs = rig.get_observation()
    depth = rig.depth.depth_meters(obs.frame)
    assert depth is not None and np.isfinite(depth).any()

    x1, y1, x2, y2 = _bbox_of(obs.frame)
    uv = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
    win = depth[int(y1):int(y2), int(x1):int(x2)]
    z = float(np.nanmedian(win))
    p = rig.geometry.backproject(uv, z, obs.T_base_cam)
    # The sensor sees the near face of the object, so the recovered point sits about
    # one radius in front of the true centre — expected, not an error.
    err = float(np.linalg.norm(p[:2] - TRUE_XY))
    assert err < radius + 0.02, f"back-projected {np.round(p, 3)} vs true {TRUE_XY}"


def test_mono_cloud_tracker_stands_down_instead_of_crashing():
    """No depth must degrade, not explode: the cloud path returns nothing and says why."""
    from rax.perception.depth_cloud.cloud_tracker import CloudTracker
    from rax.perception.depth_cloud.object_cloud import ObjectTrack

    _profile, rig, _r = _rig("mono")
    obs = rig.get_observation()
    tracker = CloudTracker.__new__(CloudTracker)   # bypass detector/mask-tracker deps
    tracker.depth = rig.depth
    tracker._warned_no_depth = False
    tracker.tracks = {1: ObjectTrack(tag=1, label=TRUE_LABEL, box=_bbox_of(obs.frame))}
    tracker.focus_tag = 1
    tracker._need_focus_init = False
    tracker._rr = 0
    tracker._focus_locked = False
    tracker._tick = 0
    tracker.roi_pad = 8
    tracker.max_points = 500
    # Deliberately None: a rig with no depth must stand down BEFORE it pays for a
    # mask, so touching the mask tracker at all is the failure this pins.
    tracker.mask_tracker = None

    assert tracker._update_clouds(obs) == []
    assert tracker._warned_no_depth, "a rig with no depth should say so once"


# --- and it can still drive the arm ---------------------------------------------
def test_approach_converges_from_a_mono_head_camera_fix():
    """Pixels -> metres -> joint targets, with a webcam as the only sensor."""
    profile, rig, _r = _rig("mono")
    kin = CartesianKinematics()
    ik = make_ik(kin, profile)
    cfg = ApproachConfig()

    obs = rig.get_observation()
    loc = PlaneRayLocalizer(rig.geometry, PRIORS, z_plane=TABLE_Z,
                            reach_m=(profile.reach_min_m, profile.reach_max_m))
    fix = loc.locate(_bbox_of(obs.frame), obs.T_base_cam, label=TRUE_LABEL)
    assert fix.ok
    target = approach_target(fix.xy, back_m=cfg.back_m, right_trim_m=cfg.right_trim_m)

    q = np.array(profile.home_deg, dtype=np.float64)
    start = float(np.linalg.norm(np.asarray(kin.forward_kinematics(q))[:2, 3] - target))
    for stage in range(cfg.steps):
        tip = np.asarray(kin.forward_kinematics(q))[:3, 3]
        waypoint, remaining = stage_step(tip[:2], target, stage=stage, total=cfg.steps,
                                         first_frac=cfg.first_step_frac,
                                         max_first_m=cfg.max_first_step_m)
        if waypoint is None or remaining < cfg.arrived_m:
            break
        q, e = ik.solve(q, np.array([waypoint[0], waypoint[1], cfg.hover_z_m]))
        assert e < 0.005, f"stage {stage}: IK missed by {e*1000:.1f} mm"
        rig.send_joint_targets(q)

    final = float(np.linalg.norm(np.asarray(kin.forward_kinematics(q))[:2, 3] - target))
    assert final < start, "the approach moved away from the target"
    assert final < cfg.arrived_m, f"ended {final*1000:.0f} mm from the hover target"


def test_all_three_rigs_agree_on_where_the_object_is():
    """The sensor must change the precision, not the answer.

    Same scene, same arm, same geometry, three different physics for obtaining range.
    If the three disagree by more than the object's own size, one of the branches is
    computing something different rather than something noisier.
    """
    fixes = {}
    for kind in ("mono", "rgbd", "stereo"):
        profile, rig, radius = _rig(kind)
        obs = rig.get_observation()
        reach = (profile.reach_min_m, profile.reach_max_m)
        # Mono has only the plane; the others could use depth, but the plane estimator
        # is what all three share, so it is what makes them comparable.
        loc = PlaneRayLocalizer(rig.geometry, PRIORS, z_plane=TABLE_Z, reach_m=reach)
        fix = loc.locate(_bbox_of(obs.frame), obs.T_base_cam, label=TRUE_LABEL)
        assert fix.ok, f"{kind} failed to locate"
        fixes[kind] = fix.xy

    spread = max(float(np.linalg.norm(a - b))
                 for a in fixes.values() for b in fixes.values())
    assert spread < 0.02, f"the three rigs disagree by {spread*1000:.0f} mm: {fixes}"


def main() -> int:
    return pytest.main([__file__, "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
