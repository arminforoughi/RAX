"""``ArmInterface`` — the hardware seam for the gaze engine.

Everything above this line (gaze loop, cloud tracker, grasp logic) is hardware
agnostic. A concrete arm only has to: hand over a synced stereo pair + the camera
pose in base frame, report joint/gripper state, and accept joint + gripper
commands. The dev :class:`manipulation.arms.MockArm` and the
:class:`robots.arms.lerobot_so101.So101Arm` both satisfy this Protocol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from models.depth.stereo import StereoIntrinsics


@dataclass
class Observation:
    """One synchronized snapshot from the arm + its eye-in-hand stereo camera."""

    left: np.ndarray  # rectified left frame, HxWx3 RGB (or HxW mono)
    right: np.ndarray  # rectified right frame, same shape as ``left``
    joints_deg: np.ndarray  # arm joint positions, degrees
    gripper_pct: float  # 0 = closed, 100 = open
    intrinsics: StereoIntrinsics
    T_base_cam: np.ndarray  # 4x4 pose of the (left) camera in base frame
    t: float = field(default_factory=time.time)


@runtime_checkable
class ArmInterface(Protocol):
    """Minimal arm + eye-in-hand-camera contract."""

    joint_names: list[str]

    def get_observation(self) -> Observation:
        """Latest synced stereo pair + joint/gripper state + camera pose."""
        ...

    def send_joint_targets(self, q_deg: np.ndarray) -> None:
        """Command absolute joint positions (degrees)."""
        ...

    def set_gripper(self, pct: float) -> None:
        """Command gripper opening: 0 = closed, 100 = open."""
        ...

    def read_gripper_current(self) -> float | None:
        """Raw gripper motor current (counts) for contact sensing, or None."""
        ...
