"""``Rig`` — one arm plus one camera, joined into an :class:`ArmInterface`.

This is the file that makes the title claim true: *any* arm that can move and report
its joints, plus *any* camera that can hand over a frame, is a working pick rig. The
arm does not import the camera and the camera does not import the arm; ``Rig`` owns the
one fact that relates them, which is the mount — and the mount already has a
representation in :class:`perception.camera_geometry.CameraPose`.

Two mounts cover the rigs people actually build:

``eye_in_hand``
    Camera on the wrist. ``T_base_cam = FK(q) @ T_ee_cam``, so it is recomputed from
    the joints on every observation.

``fixed``
    Camera on a head mast, a tripod, or the torso of a mobile base — anywhere that is
    not the moving hand. ``T_base_cam`` is constant and the joints do not enter into it.

Whichever it is, the rest of the stack sees the same :class:`Observation`, which is why
the gaze engine, the cloud tracker and the object map needed no changes to support a
head camera.
"""

from __future__ import annotations

import numpy as np

from rax.manipulation.arms.arm_interface import ArmState, Observation
from rax.perception.camera_geometry import (
    CameraGeometry,
    EyeInHand,
    FixedCamera,
    intrinsics_from_dict,
    parse_tf,
)
from rax.perception.camera_interface import CameraInterface, DepthSource, make_depth_source

__all__ = ["Rig", "geometry_for"]


def geometry_for(profile, kin=None, *, intrinsics=None) -> CameraGeometry:
    """Build the :class:`CameraGeometry` a profile describes.

    ``kin`` is required only for an eye-in-hand mount, where the pose is a function of
    the joints. A fixed camera needs nothing but its extrinsics, which is what lets a
    head-camera rig be constructed (and tested) without a URDF.
    """
    cam = profile.camera
    if intrinsics is None:
        intrinsics = intrinsics_from_dict(
            # strict: a profile whose intrinsics_fallback is not exactly four numbers
            # is malformed, and should say so here rather than silently drop cy.
            dict(zip(("fx", "fy", "cx", "cy"), cam.intrinsics_fallback, strict=True)),
            width=cam.width, height=cam.height)
    if cam.eye_in_hand:
        if kin is None:
            raise ValueError(
                f"profile {profile.name!r} mounts the camera on the hand, so its pose "
                "is FK(q) @ T_ee_cam — pass the arm's kinematics")
        pose = EyeInHand(lambda q: kin.forward_kinematics(q), parse_tf(cam.extrinsics))
    else:
        pose = FixedCamera(parse_tf(cam.extrinsics))
    return CameraGeometry(intrinsics, pose)


class Rig:
    """A cameraless arm + a camera, presented as one :class:`ArmInterface`.

    The camera is the authority on intrinsics: whatever the profile guessed is
    overwritten by what the device reports on the first frame. A rig that back-projects
    with the configured resolution while the sensor delivers another one is a silent
    metric error, so the frame wins.
    """

    def __init__(
        self,
        arm,
        camera: CameraInterface,
        geometry: CameraGeometry,
        *,
        depth: DepthSource | None = None,
        stereo=None,
    ):
        self.arm = arm
        self.camera = camera
        self.geometry = geometry
        self.depth = depth if depth is not None else make_depth_source(camera, stereo)
        self.joint_names = list(getattr(arm, "joint_names", []))
        self._synced = False

    # --- ArmInterface --------------------------------------------------------
    def get_observation(self) -> Observation:
        state = self._state()
        frame = self.camera.frame()
        if not self._synced:
            self.geometry.set_intrinsics(frame.intrinsics)
            self._synced = True
        q = np.asarray(state.joints_deg, dtype=np.float64)
        T = self.geometry.T_base_cam(q if self.geometry.pose.moves_with_arm else None)
        return Observation(frame=frame, joints_deg=q, gripper_pct=state.gripper_pct,
                           T_base_cam=T, t=frame.t)

    def send_joint_targets(self, q_deg: np.ndarray) -> None:
        self.arm.send_joint_targets(q_deg)

    def set_gripper(self, pct: float) -> None:
        self.arm.set_gripper(pct)

    def read_gripper_current(self) -> float | None:
        read = getattr(self.arm, "read_gripper_current", None)
        return read() if read is not None else None

    def disconnect(self) -> None:
        for part in (self.camera, self.arm):
            fn = getattr(part, "close", None) or getattr(part, "disconnect", None)
            if fn is not None:
                try:
                    fn()
                except Exception:
                    pass

    # --- internals -----------------------------------------------------------
    def _state(self) -> ArmState:
        """Arm state from either contract, so a camera-owning driver still composes.

        An existing driver whose only method is ``get_observation`` can be dropped into
        a rig with a different camera: its own frame is discarded and its joints kept.
        Wasteful on a live device, invaluable when swapping a sensor without also
        rewriting a working arm driver.
        """
        get_state = getattr(self.arm, "get_state", None)
        if get_state is not None:
            return get_state()
        obs = self.arm.get_observation()
        return ArmState(joints_deg=obs.joints_deg, gripper_pct=obs.gripper_pct, t=obs.t)
