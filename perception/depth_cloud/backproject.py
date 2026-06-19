"""Back-projection: (depth map, mask, intrinsics, camera pose) -> base-frame cloud.

This is the bridge between the 2D perception stack and 3D manipulation. Given a
metric depth map (in the left camera frame), a boolean object mask, and the
camera pose in base frame, it produces an ``Nx3`` array of base-frame points —
exactly what :class:`perception.depth_cloud.object_cloud.ObjectTrack` stores.

Camera convention: OpenCV (Z forward, X right, Y down), matching lerobot.
"""

from __future__ import annotations

import numpy as np

from models.depth.stereo import ROI, StereoIntrinsics


def backproject_masked(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intr: StereoIntrinsics,
    T_base_cam: np.ndarray,
    *,
    roi: ROI | None = None,
    max_points: int = 4000,
    z_min_m: float = 0.05,
    z_max_m: float = 4.0,
) -> np.ndarray:
    """Project masked, valid-depth pixels into the base frame.

    Only pixels that are inside ``mask``, inside ``roi`` (if given), and carry a
    finite depth within ``[z_min_m, z_max_m]`` are kept. The result is randomly
    subsampled to ``max_points`` to bound downstream cost.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    h, w = depth.shape[:2]
    m = np.asarray(mask, dtype=bool)
    if m.shape != (h, w):
        raise ValueError(f"mask {m.shape} does not match depth {(h, w)}")

    valid = m & np.isfinite(depth) & (depth > z_min_m) & (depth < z_max_m)
    if roi is not None:
        x1, y1, x2, y2 = roi
        window = np.zeros_like(valid)
        window[y1:y2, x1:x2] = True
        valid &= window

    vs, us = np.nonzero(valid)
    if us.size == 0:
        return np.empty((0, 3), np.float64)
    if us.size > max_points:
        idx = np.random.choice(us.size, max_points, replace=False)
        us, vs = us[idx], vs[idx]

    z = depth[vs, us]
    x = (us - intr.cx) / intr.fx * z
    y = (vs - intr.cy) / intr.fy * z
    pts_cam = np.stack([x, y, z], axis=1)  # Nx3

    T = np.asarray(T_base_cam, dtype=np.float64)
    pts_base = pts_cam @ T[:3, :3].T + T[:3, 3]
    return pts_base


def box_mask(shape: tuple[int, int], box: tuple[float, float, float, float], *, inset: float = 0.12) -> np.ndarray:
    """Boolean mask filling a box shrunk by ``inset`` on each side.

    The geometric mask used for *non-focused* objects (the focused object gets a
    precise SAM2 mask). Shrinking inward keeps box-edge background out of the cloud.
    """
    h, w = shape
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * inset, (y2 - y1) * inset
    x1, y1, x2, y2 = x1 + dx, y1 + dy, x2 - dx, y2 - dy
    m = np.zeros((h, w), bool)
    xi1, yi1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
    xi2, yi2 = int(min(w, np.ceil(x2))), int(min(h, np.ceil(y2)))
    if xi2 > xi1 and yi2 > yi1:
        m[yi1:yi2, xi1:xi2] = True
    return m


def box_to_roi(box: tuple[float, float, float, float]) -> ROI:
    """Float detection box -> integer ROI tuple."""
    x1, y1, x2, y2 = box
    return (int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2)))


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two boxes (used for detection<->track association)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0
