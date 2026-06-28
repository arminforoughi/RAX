"""Mask tracking for the *focused* object — the ``MaskTracker`` seam.

YOLO-World finds every object cheaply, but a box ROI bleeds background into the
point cloud. For the one object the arm is acting on we want a tight mask, every
frame. SAM2 does this: seed it with the object's box once, then segment each new
frame. We feed the latest detection box as a prompt so the mask follows the
object even through partial occlusion — temporal tracking without re-detecting.

Backends:
    * :class:`Sam2Tracker` — Meta SAM2 image predictor, prompted per frame.
    * :class:`EllipseMaskTracker` — inscribed-ellipse fallback (no weights); the
      depth z-range filter in back-projection cleans up the remaining background.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]


@runtime_checkable
class MaskTracker(Protocol):
    name: str

    def init(self, rgb: np.ndarray, box: Box) -> np.ndarray:
        """Begin tracking the object in ``box``; return its boolean mask (HxW)."""
        ...

    def track(self, rgb: np.ndarray, box_hint: Box | None = None) -> np.ndarray:
        """Return the focused object's boolean mask for a new frame."""
        ...


def _ellipse_mask(shape: tuple[int, int], box: Box) -> np.ndarray:
    import cv2

    h, w = shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    mask = np.zeros((h, w), np.uint8)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    ax, ay = max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return mask.astype(bool)


class EllipseMaskTracker:
    """Geometry-only fallback: an inscribed ellipse in the latest box."""

    name = "ellipse"

    def __init__(self) -> None:
        self._box: Box | None = None
        self._shape: tuple[int, int] | None = None

    def init(self, rgb: np.ndarray, box: Box) -> np.ndarray:
        self._box = box
        self._shape = np.asarray(rgb).shape[:2]
        return _ellipse_mask(self._shape, box)

    def track(self, rgb: np.ndarray, box_hint: Box | None = None) -> np.ndarray:
        shape = np.asarray(rgb).shape[:2]
        box = box_hint if box_hint is not None else self._box
        if box is None:
            return np.zeros(shape, bool)
        self._box, self._shape = box, shape
        return _ellipse_mask(shape, box)


class Sam2Tracker:
    """Meta SAM2 image predictor, box-prompted every frame (temporal via box hint)."""

    name = "sam2"

    def __init__(self, predictor):
        self._predictor = predictor
        self._box: Box | None = None

    @classmethod
    def try_create(cls, model_cfg: str = "", checkpoint: str = "") -> "Sam2Tracker | None":
        import os

        cfg = model_cfg or os.environ.get("SAM2_MODEL_CFG", "")
        ckpt = checkpoint or os.environ.get("SAM2_CHECKPOINT", "")
        if not cfg or not ckpt:
            logger.warning("[sam2] set SAM2_MODEL_CFG / SAM2_CHECKPOINT to enable mask tracking")
            return None
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(cfg, ckpt)
            return cls(SAM2ImagePredictor(model))
        except Exception as e:
            logger.warning("[sam2] failed to load: %s", e)
            return None

    def _predict(self, rgb: np.ndarray, box: Box) -> np.ndarray:
        self._predictor.set_image(np.asarray(rgb))
        masks, scores, _ = self._predictor.predict(
            box=np.asarray(box, dtype=np.float32)[None], multimask_output=False
        )
        mask = np.asarray(masks[0]).astype(bool)
        self._box = box
        return mask

    def init(self, rgb: np.ndarray, box: Box) -> np.ndarray:
        return self._predict(rgb, box)

    def track(self, rgb: np.ndarray, box_hint: Box | None = None) -> np.ndarray:
        box = box_hint if box_hint is not None else self._box
        if box is None:
            return np.zeros(np.asarray(rgb).shape[:2], bool)
        return self._predict(rgb, box)


def make_mask_tracker(backend: str = "auto", *, model_cfg: str = "", checkpoint: str = "") -> MaskTracker:
    """Build a mask tracker. ``auto`` prefers SAM2, falls back to the ellipse mask."""
    if backend in ("sam2", "auto"):
        t = Sam2Tracker.try_create(model_cfg, checkpoint)
        if t is not None:
            return t
        if backend == "sam2":
            raise RuntimeError("SAM2 requested but unavailable")
    logger.info("[mask-tracker] using EllipseMaskTracker fallback (no SAM2 weights)")
    return EllipseMaskTracker()
