"""FoundationStereo backend — wraps lerobot's existing estimator.

lerobot already ships ``perception/foundation_stereo.FoundationStereoEstimator``
(loads the NVlabs model from a local clone + checkpoint). We reuse it verbatim
and add the ROI-crop path so it fits the :class:`StereoDepth` seam: crop both
images to a disparity-safe window, run the model on the crop, scatter the metric
depth back into a full-frame ``nan`` map.

Enable via ``FOUNDATION_STEREO_REPO`` / ``FOUNDATION_STEREO_CKPT`` (or the
explicit config). If unavailable, :func:`try_create` returns ``None`` and the
factory falls back.
"""

from __future__ import annotations

import logging

import numpy as np

from models.depth.stereo import ROI, StereoIntrinsics, disparity_window

logger = logging.getLogger(__name__)


class FoundationStereoBackend:
    name = "foundation"

    def __init__(self, estimator, max_disp_px: int = 256):
        self._est = estimator  # lerobot FoundationStereoEstimator
        self.max_disp = int(max_disp_px)

    @classmethod
    def try_create(
        cls,
        *,
        repo_dir: str = "",
        ckpt_path: str = "",
        valid_iters: int = 16,
        device: str = "",
        max_disp_px: int = 256,
    ) -> "FoundationStereoBackend | None":
        try:
            from lerobot.perception.foundation_stereo import (
                FoundationStereoConfig,
                FoundationStereoEstimator,
            )
        except Exception as e:
            logger.warning("[foundation-stereo] lerobot estimator unavailable: %s", e)
            return None
        est = FoundationStereoEstimator.try_create(
            FoundationStereoConfig(
                repo_dir=repo_dir, ckpt_path=ckpt_path, valid_iters=valid_iters, device=device
            )
        )
        if est is None:
            return None
        return cls(est, max_disp_px=max_disp_px)

    def depth_meters(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        intr: StereoIntrinsics,
        roi: ROI | None = None,
    ) -> np.ndarray:
        left = np.asarray(left)
        right = np.asarray(right)
        h, w = left.shape[:2]
        depth = np.full((h, w), np.nan, dtype=np.float32)

        if roi is None:
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            x0, y0, x1, y1 = disparity_window(roi, self.max_disp, w, h)

        crop_depth = self._est.depth_map_meters(
            left[y0:y1, x0:x1], right[y0:y1, x0:x1], fx=intr.fx, baseline_m=intr.baseline_m
        )
        depth[y0:y1, x0:x1] = crop_depth
        return depth
