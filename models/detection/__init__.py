"""Object detection models: class + 2D bounding box per frame.

Public surface:
    PromptDetector / Detection      open-vocab box detection seam
    make_detector(backend=...)      YOLO-World, or ColorBlob fallback (no weights)
    MaskTracker                     per-frame mask of the focused object
    make_mask_tracker(backend=...)  SAM2, or ellipse-mask fallback (no weights)
"""

from __future__ import annotations

from models.detection.prompt_detector import (
    ColorBlobDetector,
    Detection,
    PromptDetector,
    YoloWorldDetector,
    make_detector,
)
from models.detection.sam2_tracker import (
    EllipseMaskTracker,
    MaskTracker,
    Sam2Tracker,
    make_mask_tracker,
)

__all__ = [
    "Detection",
    "PromptDetector",
    "YoloWorldDetector",
    "ColorBlobDetector",
    "make_detector",
    "MaskTracker",
    "Sam2Tracker",
    "EllipseMaskTracker",
    "make_mask_tracker",
]
