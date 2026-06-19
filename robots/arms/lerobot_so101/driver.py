"""``So101Arm`` — the real-hardware :class:`ArmInterface` for an SO-101 + OAK-D.

Wraps lerobot's ``make_robot_from_config`` (SO-101 follower) and drives the OAK-D
in ``export_stereo_rectified`` mode so we get the **raw rectified left/right pair**
(then run RAFT-Stereo / FoundationStereo ourselves) rather than the firmware depth.
The camera pose in base frame is FK(joints) composed with the eye-in-hand extrinsic.

Everything here is lerobot/SDK-specific and imported lazily, so the perception
stack and the mock harness never need the robot installed.
"""

from __future__ import annotations

import logging

import numpy as np

from manipulation.arms.arm_interface import Observation
from manipulation.arms.kinematics import PlacoKinematics
from models.depth.stereo import StereoIntrinsics

logger = logging.getLogger(__name__)

ARM_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _resolve_urdf(urdf: str) -> str:
    """Find the URDF: use it if it exists, else look under the lerobot repo root."""
    import pathlib

    p = pathlib.Path(urdf).expanduser()
    if p.exists():
        return str(p)
    try:
        import lerobot

        root = pathlib.Path(lerobot.__file__).resolve().parents[2]  # editable repo root
        cand = root / urdf
        if cand.exists():
            return str(cand)
    except Exception:
        pass
    return urdf  # let lerobot raise its (helpful) "download the URDF" error


def _parse_tf(tf_str: str) -> np.ndarray:
    """``"x,y,z,rx,ry,rz"`` (rotvec radians) -> 4x4 homogeneous transform."""
    from scipy.spatial.transform import Rotation

    p = [float(v) for v in tf_str.split(",")]
    if len(p) != 6:
        raise ValueError(f"expected 6 values, got {len(p)}: {tf_str}")
    T = np.eye(4)
    T[:3, 3] = p[:3]
    if any(abs(v) > 1e-8 for v in p[3:]):
        T[:3, :3] = Rotation.from_rotvec(p[3:]).as_matrix()
    return T


class So101Arm:
    def __init__(
        self,
        *,
        port: str,
        urdf: str = "SO101/so101_new_calib.urdf",
        ee_frame: str = "gripper_frame_link",
        camera_key: str = "front",
        gripper_camera_tf: str = "0.04,0,0.02,0,-0.35,0",
        # Native OAK-D mono resolution. read_stereo_rectified() returns the mono
        # frames at native res, and get_stereo_intrinsics() scales K to the config
        # size, so these MUST match the rectified frame size or back-projection breaks.
        width: int = 1280,
        height: int = 800,
        fps: int = 30,
    ):
        from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower import SOFollowerRobotConfig

        cam_cfg = OAKDCameraConfig(
            fps=fps, width=width, height=height, use_depth=False, export_stereo_rectified=True
        )
        robot_cfg = SOFollowerRobotConfig(port=port, cameras={camera_key: cam_cfg})
        self.robot = make_robot_from_config(robot_cfg)
        self.robot.connect()

        self.camera_key = camera_key
        self._cam = self.robot.cameras[camera_key]
        si = self._cam.get_stereo_intrinsics()  # {fx, fy, cx, cy, baseline_m}
        # depthai's getBaselineDistance() returns CENTIMETRES; lerobot passes it
        # through mislabelled as metres. A stereo baseline is never >1 m, so a
        # value that large is cm — convert. (OAK-D is ~7.5 cm = 0.075 m.)
        baseline_m = float(si["baseline_m"])
        if baseline_m > 1.0:
            baseline_m /= 100.0
        # The rectified frame size is whatever the camera actually returns.
        left0, _ = self._cam.read_stereo_rectified()
        h0, w0 = np.asarray(left0).shape[:2]
        self.intr = StereoIntrinsics(
            fx=si["fx"], fy=si["fy"], cx=si["cx"], cy=si["cy"],
            baseline_m=baseline_m, width=w0, height=h0,
        )
        self.T_ee_cam = _parse_tf(gripper_camera_tf)
        self.kin = PlacoKinematics(_resolve_urdf(urdf), ee_frame, ARM_MOTORS)
        self.joint_names = ARM_MOTORS

    def get_observation(self) -> Observation:
        obs = self.robot.get_observation()
        q = np.array([float(obs[f"{m}.pos"]) for m in ARM_MOTORS], dtype=np.float64)
        gripper = float(obs.get("gripper.pos", 0.0))
        left, right = self._cam.read_stereo_rectified()
        T_base_cam = self.kin.forward_kinematics(q) @ self.T_ee_cam
        return Observation(
            left=np.asarray(left), right=np.asarray(right), joints_deg=q,
            gripper_pct=gripper, intrinsics=self.intr, T_base_cam=T_base_cam,
        )

    def send_joint_targets(self, q_deg: np.ndarray) -> None:
        q = np.asarray(q_deg, dtype=np.float64).reshape(-1)
        self.robot.send_action({f"{m}.pos": float(q[i]) for i, m in enumerate(ARM_MOTORS)})

    def set_gripper(self, pct: float) -> None:
        self.robot.send_action({"gripper.pos": float(np.clip(pct, 0.0, 100.0))})

    def read_gripper_current(self) -> float | None:
        try:
            return float(self.robot.bus.read("Present_Current", "gripper", normalize=False))
        except Exception:
            return None

    def disconnect(self) -> None:
        try:
            self.robot.disconnect()
        except Exception:
            pass
