"""``CameraGeometry`` — pixels <-> base-frame points, for a camera mounted anywhere.

Four operations underpin every localization in the stack: project a base-frame point to
a pixel, back-project a pixel at a known range, intersect a pixel's sightline with the
table plane, and ask where the fingertip should appear. All four need exactly two
things — the intrinsics and the camera's pose in base frame — and in the monolith both
were module globals (``fx``, ``fy``, ``cx0``, ``cy0``, ``T_ee_cam``), which is what made
the geometry impossible to reuse or test.

Here the pose comes from a :class:`CameraPose`, which is the seam that lets the same
geometry serve two very different rigs:

:class:`EyeInHand`
    The camera rides the gripper, so ``T_base_cam = FK(q) @ T_ee_cam`` and the view
    changes with every joint move. This is the FPV rig the pick stack was built around.

:class:`FixedCamera`
    The camera is bolted to the world and ``T_base_cam`` is constant. Localization and
    the object map work unchanged; :meth:`CameraGeometry.tip_pixel` returns None,
    because "where the fingertip appears" is only a fixed pixel when the camera moves
    with the hand.

Conventions: the camera frame is OpenCV's (+Z along the optical axis, +X image right,
+Y image down) and all lengths are metres.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np

from models.depth.stereo import StereoIntrinsics

__all__ = [
    "CameraGeometry", "CameraPose", "EyeInHand", "FixedCamera",
    "intrinsics_from_dict", "parse_tf", "tf_to_string",
]


@runtime_checkable
class CameraPose(Protocol):
    """Supplies the camera's 4x4 pose in base frame for a given joint configuration."""

    moves_with_arm: bool

    def T_base_cam(self, q_deg: np.ndarray | None = None) -> np.ndarray: ...


class EyeInHand:
    """Camera bolted to the end effector: ``T_base_cam = FK(q) @ T_ee_cam``."""

    moves_with_arm = True

    def __init__(self, fk: Callable[[np.ndarray], np.ndarray], T_ee_cam: np.ndarray):
        self.fk = fk
        self.T_ee_cam = np.asarray(T_ee_cam, dtype=np.float64)

    def T_base_cam(self, q_deg: np.ndarray | None = None) -> np.ndarray:
        if q_deg is None:
            raise ValueError("an eye-in-hand camera's pose depends on the joints")
        return np.asarray(self.fk(np.asarray(q_deg, dtype=np.float64))) @ self.T_ee_cam

    def T_base_ee(self, q_deg: np.ndarray) -> np.ndarray:
        return np.asarray(self.fk(np.asarray(q_deg, dtype=np.float64)), dtype=np.float64)


class FixedCamera:
    """Camera bolted to the world: one constant pose, whatever the arm does."""

    moves_with_arm = False

    def __init__(self, T_base_cam: np.ndarray):
        self._T = np.asarray(T_base_cam, dtype=np.float64)

    def T_base_cam(self, q_deg: np.ndarray | None = None) -> np.ndarray:
        return self._T


class CameraGeometry:
    """Pixel <-> base-frame conversions for one camera."""

    def __init__(self, intrinsics: StereoIntrinsics, pose: CameraPose):
        self.intr = intrinsics
        self.pose = pose

    # --- intrinsics ----------------------------------------------------------
    @property
    def fx(self) -> float: return float(self.intr.fx)

    @property
    def fy(self) -> float: return float(self.intr.fy)

    @property
    def cx(self) -> float: return float(self.intr.cx)

    @property
    def cy(self) -> float: return float(self.intr.cy)

    def set_intrinsics(self, intrinsics: StereoIntrinsics) -> None:
        """Swap in the intrinsics the camera reported once it is connected."""
        self.intr = intrinsics

    # --- pose ----------------------------------------------------------------
    def T_base_cam(self, q_deg: np.ndarray | None = None) -> np.ndarray:
        return self.pose.T_base_cam(q_deg)

    # --- the four operations -------------------------------------------------
    def ray(self, uv) -> np.ndarray:
        """Unit direction of a pixel's sightline, in CAMERA frame."""
        d = np.array([(float(uv[0]) - self.cx) / self.fx,
                      (float(uv[1]) - self.cy) / self.fy,
                      1.0], dtype=np.float64)
        return d / np.linalg.norm(d)

    def backproject(self, uv, z_m: float, T_base_cam: np.ndarray) -> np.ndarray:
        """Pixel at metric depth ``z_m`` -> base-frame point.

        ``z_m`` is depth along the optical axis, not range along the ray — that is what
        a stereo depth map reports, and mixing the two is a silent range error.
        """
        x = (float(uv[0]) - self.cx) / self.fx * z_m
        y = (float(uv[1]) - self.cy) / self.fy * z_m
        return (np.asarray(T_base_cam, dtype=np.float64) @ np.array([x, y, z_m, 1.0]))[:3]

    def point_at_range(self, uv, range_m: float, T_base_cam: np.ndarray) -> np.ndarray:
        """Pixel at a metric RANGE along its sightline -> base-frame point.

        The apparent-size estimators produce a range, not an axial depth; keeping the
        two conversions separate is what stops one being used for the other.
        """
        T = np.asarray(T_base_cam, dtype=np.float64)
        return T[:3, 3] + T[:3, :3] @ (self.ray(uv) * float(range_m))

    def project(self, p_base, T_base_cam) -> tuple[float, float] | None:
        """Base-frame point -> pixel, or None if it is behind the camera.

        The exact inverse of :meth:`backproject`, and the ground-truth test for the
        hand-eye transform.
        """
        pc = np.linalg.inv(np.asarray(T_base_cam, np.float64)) @ np.append(
            np.asarray(p_base, np.float64), 1.0)
        if pc[2] <= 1e-4:
            return None
        return (float(self.fx * pc[0] / pc[2] + self.cx),
                float(self.fy * pc[1] / pc[2] + self.cy))

    def ray_to_plane(self, uv, T_base_cam, z_plane: float) -> np.ndarray | None:
        """Intersect a pixel's sightline with the horizontal plane ``z = z_plane``.

        Inverse perspective mapping: the one extra fact available without depth is that
        the object sits on a known surface. Returns None when the ray is parallel to
        the plane or points away from it — a sightline grazing the horizon otherwise
        yields a "detection" metres away.
        """
        T = np.asarray(T_base_cam, dtype=np.float64)
        o = T[:3, 3]
        d = T[:3, :3] @ self.ray(uv)
        if abs(d[2]) < 1e-6:
            return None
        t = (float(z_plane) - o[2]) / d[2]
        if t <= 0:
            return None
        return o + t * d

    def tip_pixel(self, q_deg) -> tuple[float, float] | None:
        """Where the hand-eye transform SAYS the end-effector origin appears.

        Only meaningful for an eye-in-hand rig, where the fingertip has exactly one
        pixel, the same in every pose. That predicted pixel and the MEASURED one
        (``GripperProfile.hand_uv``) are the same number computed two ways, so the gap
        between them is a live read-out of hand-eye error. Returns None for a fixed
        camera, where the fingertip's pixel moves with the arm.
        """
        if not getattr(self.pose, "moves_with_arm", False):
            return None
        q = np.asarray(q_deg, dtype=np.float64)
        T_ee = self.pose.T_base_ee(q)
        return self.project(T_ee[:3, 3], self.pose.T_base_cam(q))

    def depth_from_width(self, width_px: float, real_width_m: float) -> float:
        """Pinhole AXIAL DEPTH from apparent size: ``fx * real_width / width_px``.

        Note what this is and is not. The pinhole relation ``w_px = fx * W / Z`` uses
        Z, the depth along the optical axis — NOT the range along the sightline. The
        two are equal only for an object on the optical axis, and differ by
        ``cos(off-axis angle)`` elsewhere.

        Feeding this value to :meth:`point_at_range` therefore places an off-axis
        object systematically too close, always inward, by that cosine. Measured with
        a camera 60 cm above the table: 1 mm at r=10 cm, 10 mm at 20 cm, 48 mm at
        35 cm, 90 mm at 45 cm. Pair it with :meth:`backproject` to place it correctly.
        """
        return float(self.fx * float(real_width_m) / max(float(width_px), 1e-6))

    #: Historical name. Kept because callers read it as "the distance to the object",
    #: which is exactly the confusion documented above.
    range_from_width = depth_from_width


# --- helpers ------------------------------------------------------------------------
def parse_tf(tf: str) -> np.ndarray:
    """``"x,y,z,rx,ry,rz"`` (rotation vector, radians) -> 4x4 transform."""
    from scipy.spatial.transform import Rotation

    p = [float(v) for v in str(tf).split(",")]
    if len(p) != 6:
        raise ValueError(f"expected 6 comma-separated values, got {len(p)}: {tf!r}")
    T = np.eye(4)
    T[:3, 3] = p[:3]
    if any(abs(v) > 1e-8 for v in p[3:]):
        T[:3, :3] = Rotation.from_rotvec(p[3:]).as_matrix()
    return T


def tf_to_string(T: np.ndarray, places: int = 4) -> str:
    """4x4 transform -> the ``"x,y,z,rx,ry,rz"`` form used by profiles and JSON."""
    from scipy.spatial.transform import Rotation

    T = np.asarray(T, dtype=np.float64)
    rv = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return ",".join(f"{v:.{places}f}" for v in [*T[:3, 3], *rv])


def intrinsics_from_dict(d, width: int = 0, height: int = 0,
                         baseline_m: float = 0.0) -> StereoIntrinsics:
    """``{fx, fy, cx, cy}`` (what the cameras report) -> :class:`StereoIntrinsics`."""
    return StereoIntrinsics(
        fx=float(d["fx"]), fy=float(d["fy"]), cx=float(d["cx"]), cy=float(d["cy"]),
        baseline_m=float(d.get("baseline_m", baseline_m)),
        width=int(d.get("width", width)), height=int(d.get("height", height)),
    )
