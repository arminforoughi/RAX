"""``MockArm`` — a synthetic stereo arm so the whole stack runs with no hardware.

It renders a real rectified stereo pair of a few coloured, textured balls sitting
in front of an eye-in-hand camera. Because each object is blitted into the right
image at the integer disparity ``fx*baseline/z``, SGBM (or RAFT/FoundationStereo)
recovers correct depth, back-projection yields a cloud near the true 3D centre,
and the gaze engine can actually drive TRACK -> APPROACH -> GRASP -> PLACE.

The "joints" are the EE pose (see :class:`CartesianKinematics`), and the camera
is the EE (``T_ee_cam = I``). Base frame is the OpenCV camera convention
(X right, Y down, Z forward), so world-up is ``(0, -1, 0)``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from manipulation.arms.arm_interface import Observation
from manipulation.arms.kinematics import CartesianKinematics, pose_to_xyzrpy
from models.depth.stereo import StereoIntrinsics

WORLD_UP = np.array([0.0, -1.0, 0.0])  # OpenCV base frame: -Y is up


@dataclass
class MockObject:
    center: np.ndarray  # base-frame XYZ (m)
    radius_m: float
    color: tuple[int, int, int]
    label: str
    _patch: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    def patch(self, r_px: int) -> tuple[np.ndarray, np.ndarray]:
        """A coloured, textured circular sprite of radius ``r_px`` (cached by size)."""
        if self._patch is not None and self._patch[0].shape[0] == 2 * r_px:
            return self._patch
        seed = int(hashlib.md5(self.label.encode()).hexdigest()[:8], 16)  # stable across runs
        rng = np.random.default_rng(seed)
        d = 2 * r_px
        yy, xx = np.mgrid[0:d, 0:d]
        mask = (xx - r_px) ** 2 + (yy - r_px) ** 2 <= r_px**2
        base = np.array(self.color, np.float32)
        noise = rng.normal(0, 18, size=(d, d, 1)).astype(np.float32)  # texture for matching
        patch = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
        self._patch = (patch, mask)
        return self._patch


def default_scene() -> list[MockObject]:
    """Red, green, and blue objects ~0.4 m in front of the camera.

    Three colours so demos like "grab the red one, drop it on the green one" have
    a real support target. The colour-blob fallback detector keys off the colour
    word in the query, so any noun works ("red cube", "red box", ...).
    """
    return [
        MockObject(np.array([0.07, 0.0, 0.40]), 0.030, (225, 40, 40), "red object"),
        MockObject(np.array([0.0, 0.05, 0.46]), 0.042, (40, 200, 70), "green object"),
        MockObject(np.array([-0.08, 0.0, 0.44]), 0.045, (40, 70, 225), "blue object"),
    ]


class MockArm:
    joint_names = CartesianKinematics.joint_names

    def __init__(
        self,
        objects: list[MockObject] | None = None,
        *,
        width: int = 640,
        height: int = 480,
        fx: float = 525.0,
        baseline_m: float = 0.06,
        grasp_radius_m: float = 0.10,
    ):
        self.objects = objects if objects is not None else default_scene()
        self.intr = StereoIntrinsics(
            fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
            baseline_m=baseline_m, width=width, height=height,
        )
        self.kin = CartesianKinematics()
        self._q = pose_to_xyzrpy(np.eye(4))  # camera at base origin, looking +Z
        self._gripper = 100.0
        self._grasped: MockObject | None = None
        self._grasp_offset_cam: np.ndarray | None = None
        self._grasp_radius = float(grasp_radius_m)
        self.T_ee_cam = np.eye(4)

    # -- ArmInterface -------------------------------------------------------

    def get_observation(self) -> Observation:
        T_base_cam = self.kin.forward_kinematics(self._q) @ self.T_ee_cam
        self._carry(T_base_cam)
        left, right = self._render(T_base_cam)
        return Observation(
            left=left, right=right, joints_deg=self._q.copy(), gripper_pct=self._gripper,
            intrinsics=self.intr, T_base_cam=T_base_cam,
        )

    def send_joint_targets(self, q_deg: np.ndarray) -> None:
        self._q = np.asarray(q_deg, dtype=np.float64).reshape(-1)[:6].copy()

    def set_gripper(self, pct: float) -> None:
        pct = float(np.clip(pct, 0.0, 100.0))
        T_base_cam = self.kin.forward_kinematics(self._q) @ self.T_ee_cam
        tool = T_base_cam[:3, 3]
        if pct < 50.0 and self._grasped is None:
            near = min(self.objects, key=lambda o: np.linalg.norm(o.center - tool))
            if np.linalg.norm(near.center - tool) <= self._grasp_radius:
                self._grasped = near
                self._grasp_offset_cam = T_base_cam[:3, :3].T @ (near.center - tool)
        elif pct >= 80.0 and self._grasped is not None:
            self._grasped = None  # released
            self._grasp_offset_cam = None
        self._gripper = pct

    def read_gripper_current(self) -> float | None:
        idle = 100.0
        load = 220.0 if (self._grasped is not None and self._gripper < 80.0) else 0.0
        return idle + load

    # -- rendering ----------------------------------------------------------

    def _carry(self, T_base_cam: np.ndarray) -> None:
        if self._grasped is not None and self._grasp_offset_cam is not None:
            self._grasped.center = T_base_cam[:3, 3] + T_base_cam[:3, :3] @ self._grasp_offset_cam

    def _render(self, T_base_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H, W = self.intr.height, self.intr.width
        left = np.zeros((H, W, 3), np.uint8)
        right = np.zeros((H, W, 3), np.uint8)
        R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]

        def proj(c):
            p = R.T @ (c - t)
            return p  # camera-frame XYZ

        for obj in sorted(self.objects, key=lambda o: -float(proj(o.center)[2])):
            p = proj(obj.center)
            z = float(p[2])
            if z <= 0.05:
                continue
            u = self.intr.fx * p[0] / z + self.intr.cx
            v = self.intr.fy * p[1] / z + self.intr.cy
            r_px = int(max(2, round(self.intr.fx * obj.radius_m / z)))
            disp = int(round(self.intr.fx * self.intr.baseline_m / z))
            patch, mask = obj.patch(r_px)
            _blit(left, patch, mask, int(round(u)), int(round(v)))
            _blit(right, patch, mask, int(round(u)) - disp, int(round(v)))
        return left, right


def _blit(img: np.ndarray, patch: np.ndarray, mask: np.ndarray, cx: int, cy: int) -> None:
    H, W = img.shape[:2]
    r = patch.shape[0] // 2
    x0, y0 = cx - r, cy - r
    px0 = max(0, -x0)
    py0 = max(0, -y0)
    px1 = patch.shape[1] - max(0, (x0 + patch.shape[1]) - W)
    py1 = patch.shape[0] - max(0, (y0 + patch.shape[0]) - H)
    if px1 <= px0 or py1 <= py0:
        return
    ix0, iy0 = max(0, x0), max(0, y0)
    sub = img[iy0:iy0 + (py1 - py0), ix0:ix0 + (px1 - px0)]
    m = mask[py0:py1, px0:px1]
    sub[m] = patch[py0:py1, px0:px1][m]
