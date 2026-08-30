"""``Kinematics`` — the FK/IK seam.

The gaze engine reasons in Cartesian space (move the camera/EE toward the object)
and needs to turn a desired EE pose into joint targets. That conversion is the
only place a robot model is required, so it lives behind this Protocol.

Backends, chosen by :func:`make_kinematics`:
    * :class:`PlacoKinematics` — the placo QP solver, when placo is installed. Best
      behaviour on redundant arms and near singularities.
    * :class:`~rax.manipulation.arms.urdf_kinematics.UrdfKinematics` — pure numpy,
      always available. Reads the URDF with the standard library and solves IK by
      damped least squares.
    * :class:`CartesianKinematics` — identity model where the 6 "joints" *are* the
      EE pose (x, y, z, rx, ry, rz). Lets the dev harness exercise the full
      APPROACH loop with no URDF at all.

No backend requires a robot framework. That is deliberate: the previous version
imported its solver out of lerobot, which meant anyone with a URDF and an arm that is
not an SO-101 installed a fleet manager to compute forward kinematics.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


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
    # no urdf_dir — rerun_viz detects absence and uses fallback line-strip path

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

    def get_link_transforms_chain(self, q_deg: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Two-link chain for Rerun: base origin → EE. Enables ground grid + EE axes."""
        T_ee = self.forward_kinematics(q_deg)
        return [("base", np.eye(4)), ("ee", T_ee)]


class PlacoKinematics:
    """placo's QP kinematics solver, driven directly.

    Preferred when available: it solves IK as a weighted task problem rather than by
    inverting a Jacobian, which behaves better on redundant arms and near
    singularities. placo publishes no wheels for some platforms, so it cannot be the
    only backend — see :func:`make_kinematics`.
    """

    def __init__(
        self,
        urdf_path: str,
        ee_frame: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        import placo  # noqa: PLC0415 - optional backend, imported where it is used

        self.urdf_dir = _placo_dir(urdf_path)
        self.ee_frame = ee_frame
        self.robot = placo.RobotWrapper(self.urdf_dir)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)          # the base is bolted down, not floating
        self.joint_names = (list(self.robot.joint_names()) if joint_names is None
                            else list(joint_names))
        self._task = self.solver.add_frame_task(ee_frame, np.eye(4))

    def _apply(self, q_deg: np.ndarray) -> None:
        q = np.deg2rad(np.asarray(q_deg, dtype=np.float64).reshape(-1))
        for i, name in enumerate(self.joint_names):
            if i < len(q):
                self.robot.set_joint(name, float(q[i]))
        self.robot.update_kinematics()

    def forward_kinematics(self, q_deg: np.ndarray) -> np.ndarray:
        self._apply(q_deg)
        return np.asarray(self.robot.get_T_world_frame(self.ee_frame), dtype=np.float64)

    def inverse_kinematics(
        self,
        q_current_deg: np.ndarray,
        T_target: np.ndarray,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.05,
    ) -> np.ndarray:
        self._apply(q_current_deg)
        self._task.T_world_frame = np.asarray(T_target, dtype=np.float64)
        self._task.configure(self.ee_frame, "soft",
                             float(position_weight), float(orientation_weight))
        self.solver.solve(True)
        self.robot.update_kinematics()

        solved = np.rad2deg([self.robot.get_joint(n) for n in self.joint_names])
        current = np.asarray(q_current_deg, dtype=np.float64).reshape(-1)
        out = current.copy()
        out[:len(solved)] = solved
        return out                # a trailing gripper value passes through untouched

    def get_link_transforms_chain(self, q_deg: np.ndarray) -> list[tuple[str, np.ndarray]]:
        self._apply(q_deg)
        return [("base", np.eye(4)),
                (self.ee_frame,
                 np.asarray(self.robot.get_T_world_frame(self.ee_frame), dtype=np.float64))]


def _placo_dir(urdf_path: str) -> str:
    """placo's RobotWrapper takes a DIRECTORY and appends ``robot.urdf`` itself."""
    import pathlib

    p = pathlib.Path(urdf_path).expanduser()
    if p.is_dir():
        return str(p)
    if (p.parent / "robot.urdf").is_file():
        return str(p.parent)
    raise ValueError(
        f"placo needs a directory containing robot.urdf; {p} has no sibling by that "
        f"name. Either add one, or omit placo and use the numpy backend, which reads "
        f"the .urdf file directly.")


def make_kinematics(
    urdf_path: str,
    ee_frame: str = "gripper_frame_link",
    joint_names: list[str] | None = None,
    *,
    backend: str = "auto",
) -> Kinematics:
    """Build a URDF kinematics solver: placo when available, numpy otherwise.

    ``backend`` is ``"auto" | "placo" | "numpy"``. ``auto`` prefers placo and falls
    back silently — the numpy solver is a supported backend rather than a degraded
    mode, so a warning would be noise on every Windows run.
    """
    from rax.manipulation.arms.urdf_kinematics import UrdfKinematics

    want = backend.lower()
    if want in ("auto", "placo"):
        try:
            return PlacoKinematics(urdf_path, ee_frame, joint_names)
        except Exception as exc:
            if want == "placo":
                raise
            logger.debug("[kinematics] placo unavailable (%s); using the numpy backend", exc)
    return UrdfKinematics(urdf_path, ee_frame, joint_names)
