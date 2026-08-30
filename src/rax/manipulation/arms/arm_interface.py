"""``ArmInterface`` — the hardware seam for the gaze engine.

Everything above this line (gaze loop, cloud tracker, grasp logic) is hardware
agnostic. A concrete arm only has to: report joint/gripper state, accept joint +
gripper commands, and supply the camera's pose in base frame. The dev
:class:`manipulation.arms.MockArm` and the
:class:`robots.arms.lerobot_so101.So101Arm` both satisfy this Protocol.

The *sensor* half of an observation lives behind its own seam —
:class:`perception.camera_interface.Frame` — so that a stereo rig, an RGB-D rig and a
plain webcam are three cameras rather than three arms. An :class:`Observation` is the
join of the two: arm state at time *t*, the frame taken at time *t*, and the camera pose
that relates them. The historical ``obs.left`` / ``obs.right`` / ``obs.intrinsics``
names are kept as views onto the frame, because those spellings read correctly at every
call site that already uses them and renaming them would be churn, not clarity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from rax.models.depth.stereo import StereoIntrinsics
from rax.perception.camera_interface import Frame


class Observation:
    """One synchronized snapshot: arm state + camera frame + the pose joining them.

    Construct it either from a :class:`~perception.camera_interface.Frame`::

        Observation(frame=cam.frame(), joints_deg=q, gripper_pct=g, T_base_cam=T)

    or from raw images, which builds the frame for you and is what every existing
    driver does::

        Observation(left=l, right=r, intrinsics=intr, joints_deg=q,
                    gripper_pct=g, T_base_cam=T)
    """

    __slots__ = ("frame", "joints_deg", "gripper_pct", "T_base_cam", "t")

    def __init__(
        self,
        *,
        joints_deg: np.ndarray,
        gripper_pct: float,
        T_base_cam: np.ndarray,
        frame: Frame | None = None,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
        depth_m: np.ndarray | None = None,
        intrinsics: StereoIntrinsics | None = None,
        t: float | None = None,
    ):
        if frame is None:
            if left is None or intrinsics is None:
                raise TypeError(
                    "Observation needs either frame=Frame(...) or "
                    "left=... together with intrinsics=...")
            frame = Frame(rgb=left, right=right, depth_m=depth_m,
                          intrinsics=intrinsics, t=t if t is not None else time.time())
        self.frame = frame
        self.joints_deg = joints_deg
        self.gripper_pct = float(gripper_pct)
        self.T_base_cam = T_base_cam
        self.t = float(t if t is not None else frame.t)

    # --- views onto the frame ------------------------------------------------
    @property
    def left(self) -> np.ndarray:
        """The image detection runs on. Named ``left`` for the stereo rig it grew up on."""
        return self.frame.rgb

    @property
    def rgb(self) -> np.ndarray:
        return self.frame.rgb

    @property
    def right(self) -> np.ndarray | None:
        return self.frame.right

    @property
    def depth_m(self) -> np.ndarray | None:
        return self.frame.depth_m

    @property
    def intrinsics(self) -> StereoIntrinsics:
        return self.frame.intrinsics

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        w, h = self.frame.size
        extra = ("stereo" if self.frame.has_stereo
                 else "depth" if self.frame.has_depth else "mono")
        return (f"Observation({w}x{h} {extra}, "
                f"q={np.round(np.asarray(self.joints_deg), 1).tolist()}, "
                f"grip={self.gripper_pct:.0f}%)")


@dataclass
class ArmState:
    """Just the arm half — joints, gripper, timestamp. No pixels.

    This is what :class:`ArmInterface.get_state` returns, and it is what an arm that
    does *not* own a camera has to provide. A rig is then assembled by
    :class:`manipulation.arms.rig.Rig`, which pairs it with any
    :class:`~perception.camera_interface.CameraInterface`.
    """

    joints_deg: np.ndarray
    gripper_pct: float
    t: float = field(default_factory=time.time)


@runtime_checkable
class ArmInterface(Protocol):
    """Minimal arm contract.

    ``get_observation`` is the original single-call form, kept because the SO-101
    driver's camera is genuinely part of its lerobot robot object and splitting it
    would buy nothing. Arms that do not own a camera implement ``get_state`` instead
    and are paired with one by :class:`manipulation.arms.rig.Rig`.
    """

    joint_names: list[str]

    def get_observation(self) -> Observation:
        """Latest synced frame + joint/gripper state + camera pose."""
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


@runtime_checkable
class CameralessArm(Protocol):
    """An arm that only moves — no sensor. Pair it with a camera via ``Rig``."""

    joint_names: list[str]

    def get_state(self) -> ArmState: ...
    def send_joint_targets(self, q_deg: np.ndarray) -> None: ...
    def set_gripper(self, pct: float) -> None: ...
    def read_gripper_current(self) -> float | None: ...
