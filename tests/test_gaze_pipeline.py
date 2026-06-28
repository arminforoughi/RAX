"""Unit checks for the gaze + per-object point-cloud pipeline (no hardware/weights).

Run with ``pytest tests/test_gaze_pipeline.py``. Covers the load-bearing math and
the CloudTracker's update budget — the pieces the plan calls out for verification.
"""

from __future__ import annotations

import numpy as np

from manipulation.arms.se3 import look_at_R, make_pose, vantage_dir
from models.depth.stereo import StereoIntrinsics
from perception.depth_cloud.backproject import backproject_masked, box_iou
from perception.depth_cloud.object_cloud import ObjectTrack


INTR = StereoIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, baseline_m=0.06, width=640, height=480)


def test_backproject_round_trip():
    # A constant-depth patch at the image centre back-projects to (0, 0, z).
    depth = np.full((480, 640), np.nan, np.float32)
    depth[230:250, 310:330] = 0.5
    mask = np.zeros((480, 640), bool)
    mask[230:250, 310:330] = True
    pts = backproject_masked(depth, mask, INTR, np.eye(4))
    assert pts.shape[0] > 0
    c = pts.mean(axis=0)
    assert abs(c[0]) < 0.02 and abs(c[1]) < 0.02
    assert abs(c[2] - 0.5) < 1e-3


def test_backproject_offset_pixel_sign():
    # A patch to the right of centre yields +x; below centre yields +y (OpenCV).
    depth = np.full((480, 640), np.nan, np.float32)
    depth[240:244, 440:444] = 1.0
    mask = depth > 0
    c = backproject_masked(depth, mask, INTR, np.eye(4)).mean(axis=0)
    assert c[0] > 0.1 and abs(c[2] - 1.0) < 1e-3


def test_camera_pose_applies():
    # Translating the camera shifts the cloud by the same amount.
    depth = np.full((480, 640), np.nan, np.float32)
    depth[238:242, 318:322] = 0.4
    mask = depth > 0
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    c = backproject_masked(depth, mask, INTR, T).mean(axis=0)
    assert np.allclose(c, [1.0, 2.0, 3.4], atol=2e-2)


def test_look_at_R_aligns_optical_axis():
    # Camera at origin (T_ee_cam = I); look_at should point cam +Z at the target dir.
    target_dir = np.array([1.0, 0.0, 1.0])
    R = look_at_R(np.eye(3), np.eye(4), target_dir)
    z_axis = R[:, 2]
    assert np.dot(z_axis, target_dir / np.linalg.norm(target_dir)) > 0.999


def test_vantage_dir_top_and_side():
    up = np.array([0.0, 0.0, 1.0])
    # Top-down: straight up the up-axis.
    assert np.allclose(vantage_dir(up, 0, 90), [0, 0, 1], atol=1e-9)
    # Side: purely horizontal (no up component), unit length.
    side = vantage_dir(up, 0, 0)
    assert abs(side[2]) < 1e-9 and abs(np.linalg.norm(side) - 1.0) < 1e-9
    # Azimuth rotates the side direction around up (orthogonal to az=0).
    assert abs(vantage_dir(up, 0, 0) @ vantage_dir(up, 90, 0)) < 1e-9
    # Works for a non-Z up axis (the mock uses -Y up).
    assert np.allclose(vantage_dir(np.array([0.0, -1.0, 0.0]), 0, 90), [0, -1, 0], atol=1e-9)


def test_object_track_top_and_extent():
    # A 0.1 m tall vertical column; up = +Z. Top point sits ~0.05 above centroid.
    z = np.linspace(0.0, 0.1, 50)
    pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    tr = ObjectTrack(tag=1, label="x", box=(0, 0, 1, 1))
    tr.set_cloud(pts)
    up = np.array([0.0, 0.0, 1.0])
    assert abs(tr.extent_along(up) - 0.1) < 1e-6
    top = tr.top_point(up)
    assert top[2] > tr.centroid[2]
    assert abs(top[2] - 0.095) < 0.02


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0.1 < box_iou((0, 0, 10, 10), (5, 0, 15, 10)) < 0.5


def test_ray_point_base_on_center_ray():
    from manipulation.arms.gaze_engine import _ray_point_base

    T = np.eye(4)
    p = _ray_point_base(320.0, 240.0, 0.5, T, INTR)
    assert abs(p[0]) < 1e-6 and abs(p[1]) < 1e-6
    assert abs(p[2] - 0.5) < 1e-6


def _engine(cfg=None):
    from manipulation.arms.gaze_engine import GazeConfig, GazeEngine
    from manipulation.arms.kinematics import CartesianKinematics

    class _Arm:
        joint_names = []

    return GazeEngine(_Arm(), CartesianKinematics(), None, cfg or GazeConfig())  # type: ignore[arg-type]


def test_gaze_deltas_and_depth_filter():
    from manipulation.arms.gaze_engine import GazeConfig
    from manipulation.arms.arm_interface import Observation

    eng = _engine(GazeConfig(pan_sign=1.0, tilt_sign=1.0, aim_v_offset_px=90.0))
    eng._aim_v_now = 90.0  # aim fully ramped to the gripper line (bottom-centre)
    obs = Observation(
        left=np.zeros((480, 640, 3), np.uint8),
        right=np.zeros((480, 640, 3), np.uint8),
        joints_deg=np.zeros(5),
        gripper_pct=100.0,
        intrinsics=INTR,
        T_base_cam=np.eye(4),
    )
    # Object right of aim -> positive pan delta (same sign convention as atan2(du)).
    d_pan, d_tilt = eng._gaze_deltas(obs, INTR.cx + 40, INTR.cy + 90)[2:]
    assert d_pan > 0
    assert abs(d_tilt) < 0.01

    eng._filter_depth(0.25)
    assert eng._depth_filt is not None and abs(eng._depth_filt - 0.25) < 1e-6
    # Garbage depth rejected
    assert eng._filter_depth(4.0) == eng._depth_filt
    # Sudden far jump rejected
    assert eng._filter_depth(0.50) == eng._depth_filt
    assert eng._filter_depth(0.20) is not None and eng._depth_filt < 0.25


def test_mock_reaches_grasp():
    from manipulation.arms.gaze_engine import GazeConfig, GazeEngine
    from manipulation.arms.mock_arm import WORLD_UP, MockArm
    from manipulation.arms.kinematics import CartesianKinematics
    from models.depth import make_stereo
    from models.detection import make_detector, make_mask_tracker
    from perception.depth_cloud import CloudTracker

    arm = MockArm()
    cloud = CloudTracker(
        make_detector("color_blob"), make_mask_tracker("ellipse"),
        make_stereo("sgbm", max_disp_px=192), "red object", detect_every=5,
    )
    cfg = GazeConfig(
        world_up=WORLD_UP, grasp_range_m=0.12, aim_v_offset_px=0.0,
        center_tol_px=80.0, center_dv_tol_px=120.0, reach_dv_tol_px=120.0,
        center_hold_frames=2, approach_max_step_m=0.006,
    )
    eng = GazeEngine(arm, CartesianKinematics(), cloud, cfg, cartesian=True)
    final = "SEARCH"
    for _ in range(800):
        final = eng.step(1.0 / cfg.loop_hz)
        if final in ("GRASP", "PLACE", "DONE"):
            break
    assert final in ("GRASP", "PLACE", "DONE")


def test_cloud_tracker_schedule_focus_every_tick_others_round_robin():
    from perception.depth_cloud.cloud_tracker import CloudTracker

    ct = CloudTracker.__new__(CloudTracker)  # bypass heavy __init__
    ct.focus_tag = 1
    ct._rr = 0
    ct._focus_locked = False
    ct.tracks = {tag: ObjectTrack(tag=tag, label="x", box=(0, 0, 1, 1)) for tag in (1, 2, 3, 4)}

    schedules = [ct._scheduled_tags() for _ in range(6)]
    # Focus (tag 1) is in every schedule.
    assert all(1 in s for s in schedules)
    # The non-focus tags (2,3,4) are visited round-robin, one per tick.
    others = [s[1] for s in schedules]
    assert others[:3] == [2, 3, 4]
    assert others[3:6] == [2, 3, 4]
