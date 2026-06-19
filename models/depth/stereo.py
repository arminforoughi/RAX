"""Stereo depth — the ``StereoDepth`` seam.

A stereo estimator turns a rectified left/right pair into a dense depth map (in
metres, expressed in the *left* camera frame). Concrete backends live alongside
this file (``raft_stereo``, ``foundation_stereo``, ``sgbm_stereo``) and are
selected through :func:`models.depth.make_stereo`.

Every backend honours an optional ``roi`` so the caller can pay only for the
pixels it cares about — this is what lets :class:`perception.depth_cloud.CloudTracker`
update the *focused* object every frame and the rest round-robin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# A region of interest as integer pixel bounds (x1, y1, x2, y2), x2/y2 exclusive.
ROI = tuple[int, int, int, int]


@dataclass(frozen=True)
class StereoIntrinsics:
    """Pinhole intrinsics of the (left) rectified camera plus the stereo baseline.

    Depth from disparity is ``z = fx * baseline_m / disparity_px``; back-projection
    of a pixel uses ``fx, fy, cx, cy``. ``(width, height)`` size the full frame so
    ROI math and the mock renderer agree on the image plane.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        """3x3 camera matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@runtime_checkable
class StereoDepth(Protocol):
    """Disparity / depth from a rectified stereo pair.

    Implementations must be safe to call every tick; heavy models should cache
    their weights on first use. ``depth_meters`` returns ``nan`` for invalid
    pixels so downstream masking can drop them cleanly.
    """

    name: str

    def depth_meters(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        intr: StereoIntrinsics,
        roi: ROI | None = None,
    ) -> np.ndarray:
        """Dense depth (m) in the left frame. Shape matches ``left`` (full HxW);
        when ``roi`` is given only that window is filled, the rest is ``nan``.
        """
        ...


def clamp_roi(roi: ROI, width: int, height: int, *, pad: int = 0) -> ROI:
    """Pad and clamp an ROI to the image, guaranteeing a non-empty window."""
    x1, y1, x2, y2 = roi
    x1 = int(max(0, min(width - 1, x1 - pad)))
    y1 = int(max(0, min(height - 1, y1 - pad)))
    x2 = int(max(x1 + 1, min(width, x2 + pad)))
    y2 = int(max(y1 + 1, min(height, y2 + pad)))
    return x1, y1, x2, y2


def disparity_window(roi: ROI, max_disp_px: int, width: int, height: int) -> ROI:
    """Expand an ROI leftward by ``max_disp_px`` so a stereo crop stays correct.

    Cropping *both* left and right images to the **same** window preserves the
    true per-pixel disparity (both share an x-origin), but a left pixel at column
    ``u`` only finds its match (at ``u - d``) if the window reaches ``d`` columns
    further left. Extending the window's left edge by the max expected disparity
    guarantees every match inside the original ROI is in-crop.
    """
    x1, y1, x2, y2 = clamp_roi(roi, width, height)
    x0 = int(max(0, x1 - int(max_disp_px)))
    return x0, y1, x2, y2
