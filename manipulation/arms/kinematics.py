"""``Kinematics`` — the FK/IK seam.

The gaze engine reasons in Cartesian space (move the camera/EE toward the object)
and needs to turn a desired EE pose into joint targets. That conversion is the
only place a robot model is required, so it lives behind this Protocol.

Backends:
    * :class:`PlacoKinematics` — wraps lerobot's placo ``RobotKinematics`` (real arms).
    * :class:`CartesianKinematics` — identity model where the 6 "joints" *are* the
      EE pose (x, y, z, rx, ry, rz). Lets the dev harness exercise the full
      APPROACH loop with no URDF / placo.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation


@runtime_checkable
class Kinematics(Protocol):
    joint_names: list[str]

    def forward_kinematics(self, q_deg: np.ndarray) -> np.ndarray:
        """Joint angles (deg) -> 4x4 EE pose in base frame."""
        ...

    def inverse_kinematics(
        self,
        q_current_deg: np.ndarray,
        T_target: np.ndarray,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.05,
    ) -> np.ndarray:
        """Desired EE pose -> joint angles (deg), seeded with ``q_current_deg``."""
        ...


def pose_to_xyzrpy(T: np.ndarray) -> np.ndarray:
    """4x4 -> (x, y, z, rx, ry, rz) with rotation as a rotation vector (deg)."""
    T = np.asarray(T, dtype=np.float64)
    rvec = Rotation.from_matrix(T[:3, :3]).as_rotvec(degrees=True)
    return np.concatenate([T[:3, 3], rvec])


def xyzrpy_to_pose(q: np.ndarray) -> np.ndarray:
    """(x, y, z, rx, ry, rz)[deg] -> 4x4 pose."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    T = np.eye(4)
    T[:3, 3] = q[:3]
    T[:3, :3] = Rotation.from_rotvec(q[3:6], degrees=True).as_matrix()
    return T


class CartesianKinematics:
    """Identity model: the 6 joints are literally the EE pose components.

    Used by the mock arm so the gaze engine's IK path runs without a robot model.
    """

    joint_names = ["x", "y", "z", "rx", "ry", "rz"]

    def forward_kinematics(self, q_deg: np.ndarray) -> np.ndarray:
        return xyzrpy_to_pose(q_deg)

    def inverse_kinematics(
        self,
        q_current_deg: np.ndarray,
        T_target: np.ndarray,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.05,
    ) -> np.ndarray:
        return pose_to_xyzrpy(T_target)


class PlacoKinematics:
    """Wraps lerobot's placo-based ``RobotKinematics`` for real arms (e.g. SO-101)."""

    def __init__(
        self,
        urdf_path: str,
        ee_frame: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        from lerobot.model.kinematics import RobotKinematics

        self._kin = RobotKinematics(urdf_path, ee_frame, joint_names)
        self.joint_names = list(self._kin.joint_names)

    def forward_kinematics(self, q_deg: np.ndarray) -> np.ndarray:
        return np.asarray(self._kin.forward_kinematics(np.asarray(q_deg)), dtype=np.float64)

    def inverse_kinematics(
        self,
        q_current_deg: np.ndarray,
        T_target: np.ndarray,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.05,
    ) -> np.ndarray:
        return np.asarray(
            self._kin.inverse_kinematics(
                np.asarray(q_current_deg),
                np.asarray(T_target, dtype=np.float64),
                position_weight=float(position_weight),
                orientation_weight=float(orientation_weight),
            ),
            dtype=np.float64,
        )
