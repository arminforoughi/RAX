"""``SyntheticCamera`` — the same scene rendered as stereo, RGB-D, or mono.

The point of this file is falsifiability. Claiming "any camera works" is cheap; the
claim is only worth publishing if every branch is exercised on every commit, without
hardware. A synthetic camera that can present one identical scene through all three
sensor kinds turns "we support mono" into a test that fails when we stop supporting it.

It renders :class:`~manipulation.arms.mock_arm.MockObject` spheres — reusing the mock
arm's scene rather than inventing a second one, so the two harnesses cannot drift — from
an arbitrary ``T_base_cam``. That last part is what the mock arm could not do: its view
is always its own end effector, whereas a head camera watches the arm from somewhere
else entirely.

Depth is rendered analytically (the sphere's front surface, ``nan`` elsewhere), so the
RGB-D branch is exact and any error downstream is the stack's, not the renderer's.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

from rax.manipulation.arms.mock_arm import MockObject, _blit, default_scene
from rax.models.depth.stereo import StereoIntrinsics
from rax.perception.camera_interface import Frame

__all__ = ["SyntheticCamera"]


class SyntheticCamera:
    """One scene, presented as whichever sensor kind you ask for.

    ``pose`` is a callable returning the current 4x4 ``T_base_cam``. It is a callable
    rather than a matrix because an eye-in-hand rig's pose changes with every joint
    command, and a test that renders from a stale pose would pass while the real rig
    fails.
    """

    def __init__(
        self,
        pose: Callable[[], np.ndarray],
        *,
        kind: str = "stereo",
        objects: Sequence[MockObject] | None = None,
        width: int = 640,
        height: int = 480,
        fx: float = 525.0,
        baseline_m: float = 0.06,
    ):
        if kind not in ("stereo", "rgbd", "mono"):
            raise ValueError(f"kind must be stereo|rgbd|mono, got {kind!r}")
        self.kind = kind
        self.pose = pose
        self.objects = list(objects if objects is not None else default_scene())
        self._intr = StereoIntrinsics(
            fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
            # A mono or RGB-D rig has no baseline, and advertising one would let a
            # stereo matcher be built over a camera that cannot feed it.
            baseline_m=baseline_m if kind == "stereo" else 0.0,
            width=width, height=height)

    # --- CameraInterface -----------------------------------------------------
    def frame(self) -> Frame:
        T = np.asarray(self.pose(), dtype=np.float64)
        left, right, depth = self._render(T)
        return Frame(
            rgb=left,
            right=right if self.kind == "stereo" else None,
            depth_m=depth if self.kind == "rgbd" else None,
            intrinsics=self._intr, t=time.time())

    def intrinsics(self) -> StereoIntrinsics:
        return self._intr

    def close(self) -> None:
        return None

    # --- rendering -----------------------------------------------------------
    def _render(self, T_base_cam: np.ndarray):
        intr = self._intr
        H, W = intr.height, intr.width
        left = np.zeros((H, W, 3), np.uint8)
        right = np.zeros((H, W, 3), np.uint8) if self.kind == "stereo" else None
        depth = np.full((H, W), np.nan, np.float32) if self.kind == "rgbd" else None
        R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]

        def to_cam(c):
            return R.T @ (np.asarray(c, dtype=np.float64) - t)

        # Painter's algorithm: far objects first, so nearer ones overwrite them.
        for obj in sorted(self.objects, key=lambda o: -float(to_cam(o.center)[2])):
            p = to_cam(obj.center)
            z = float(p[2])
            if z <= 0.05:
                continue
            u = intr.fx * p[0] / z + intr.cx
            v = intr.fy * p[1] / z + intr.cy
            r_px = int(max(2, round(intr.fx * obj.radius_m / z)))
            patch, mask = obj.patch(r_px)
            _blit(left, patch, mask, int(round(u)), int(round(v)))
            if right is not None:
                disp = int(round(intr.fx * intr.baseline_m / z))
                _blit(right, patch, mask, int(round(u)) - disp, int(round(v)))
            if depth is not None:
                self._blit_depth(depth, int(round(u)), int(round(v)), r_px, z,
                                 obj.radius_m, intr.fx)
        return left, right, depth

    @staticmethod
    def _blit_depth(depth, cu: int, cv: int, r_px: int, z: float,
                    radius_m: float, fx: float) -> None:
        """Write the sphere's front surface, not its centre depth.

        A flat disc at ``z`` would make every object read as a fronto-parallel plane and
        would flatter the back-projection: the centroid of a disc is exact by
        construction. Bulging the surface by ``sqrt(r^2 - d^2)`` reproduces the real
        bias a sphere gives a depth sensor, which is what the stack has to survive.
        """
        H, W = depth.shape
        y0, y1 = max(0, cv - r_px), min(H, cv + r_px + 1)
        x0, x1 = max(0, cu - r_px), min(W, cu + r_px + 1)
        if y1 <= y0 or x1 <= x0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(xx - cu, yy - cv)
        inside = rr <= r_px
        if not np.any(inside):
            return
        # Pixel radius -> metres at this range, then the sphere's front surface.
        r_m = rr / max(fx, 1e-6) * z
        bulge = np.sqrt(np.clip(radius_m**2 - r_m**2, 0.0, None))
        patch = np.where(inside, z - bulge, np.nan).astype(np.float32)
        win = depth[y0:y1, x0:x1]
        # Nearest surface wins, matching what a real sensor reports through occlusion.
        depth[y0:y1, x0:x1] = np.fmin(win, np.where(inside, patch, np.nan))
