"""OpenCV StereoSGBM backend — the always-available fallback.

No GPU, no weights: this keeps the whole pipeline (and the dev harness) runnable
anywhere. It also exercises the exact same ROI-crop path as the learned backends,
so switching to RAFT-Stereo / FoundationStereo changes only quality, not plumbing.
"""

from __future__ import annotations

import numpy as np

from models.depth.stereo import ROI, StereoIntrinsics, disparity_window


def _gray(img: np.ndarray) -> np.ndarray:
    import cv2

    a = np.asarray(img)
    if a.ndim == 3:
        return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    return a.astype(np.uint8)


class SgbmStereo:
    name = "sgbm"

    def __init__(self, max_disp_px: int = 192, block_size: int = 5):
        import cv2

        self.max_disp = max(16, (int(max_disp_px) // 16) * 16)
        self._matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.max_disp,
            blockSize=int(block_size),
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def depth_meters(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        intr: StereoIntrinsics,
        roi: ROI | None = None,
    ) -> np.ndarray:
        h, w = np.asarray(left).shape[:2]
        depth = np.full((h, w), np.nan, dtype=np.float32)

        if roi is None:
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            x0, y0, x1, y1 = disparity_window(roi, self.max_disp, w, h)
            # SGBM needs the crop wider than the disparity search range; widen the
            # x-band (keep the y-band) if the ROI ended up too narrow near an edge.
            min_w = self.max_disp + 16
            if x1 - x0 < min_w:
                x0 = max(0, min(x0, w - min_w))
                x1 = min(w, x0 + min_w)

        lc = _gray(np.asarray(left)[y0:y1, x0:x1])
        rc = _gray(np.asarray(right)[y0:y1, x0:x1])
        disp = self._matcher.compute(lc, rc).astype(np.float32) / 16.0  # fixed-point -> px

        with np.errstate(divide="ignore", invalid="ignore"):
            z = intr.fx * intr.baseline_m / disp
        z[disp <= 0.0] = np.nan
        z[~np.isfinite(z)] = np.nan
        depth[y0:y1, x0:x1] = z
        return depth
