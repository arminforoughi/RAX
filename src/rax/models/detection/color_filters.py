"""Colour-consistency gate on open-vocabulary detections.

An open-vocabulary detector matches on *shape and context* far more than on colour:
asked for "red cube" it will happily return the green one, because the box is the
salient cube-shaped thing in frame. This gate is the cheap fix — if the label names a
colour, require the box to actually contain that colour.

Ported from ``lerobot.perception.detection_filters`` (Apache-2.0, HuggingFace Inc.)
so the gate travels with RAX instead of with a private lerobot checkout.

**Not the same table as** :mod:`rax.models.detection.prompt_detector`'s ``_HSV_RANGES``,
and deliberately so. That one has to *find* a blob, so its bands are tight enough to
segment cleanly. This one only has to *reject an impostor*, so its bands are wide and
include the achromatic names (black/white/grey) a segmenter cannot usefully chase.
Merging them would force one set of bounds to do both jobs badly.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

__all__ = ["COLOR_HSV_INTERVALS", "color_names_in_query", "bbox_color_match_fraction"]

#: Colour word -> HSV intervals, OpenCV convention (H 0-179). Hues that wrap around
#: the origin (red) or span a broad everyday meaning get several intervals.
COLOR_HSV_INTERVALS: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 70, 50), (12, 255, 255)), ((165, 70, 50), (180, 255, 255))],
    "orange": [((8, 100, 80), (22, 255, 255))],
    "yellow": [((20, 80, 80), (38, 255, 255))],
    "green": [((38, 50, 50), (88, 255, 255))],
    "cyan": [((80, 50, 50), (100, 255, 255))],
    "blue": [((100, 50, 50), (128, 255, 255))],
    "purple": [((128, 50, 50), (152, 255, 255))],
    "pink": [((145, 40, 80), (175, 255, 255))],
    "magenta": [((140, 50, 50), (165, 255, 255))],
    "brown": [((8, 80, 40), (25, 255, 180))],
    "black": [((0, 0, 0), (179, 255, 90))],
    "white": [((0, 0, 180), (179, 60, 255))],
    "gray": [((0, 0, 80), (179, 80, 200)), ((0, 0, 80), (179, 40, 200))],
    "grey": [((0, 0, 80), (179, 80, 200)), ((0, 0, 80), (179, 40, 200))],
}


def color_names_in_query(query: str) -> list[str]:
    """Colour words appearing as whole words in ``query``, in table order.

    Whole-word matching is what keeps "orange" the fruit from being read as a colour
    only when it stands alone — it does not, and cannot, disambiguate that case. The
    caller decides what a colourless label means; here, no match means no gate.
    """
    if not query or not query.strip():
        return []
    q = query.lower()
    found: list[str] = []
    for name in COLOR_HSV_INTERVALS:
        if re.search(r"\b" + re.escape(name) + r"\b", q) and name not in found:
            found.append(name)
    return found


def bbox_color_match_fraction(
    rgb: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    color_names: list[str],
) -> float:
    """Fraction of the box's pixels (0-1) matching any interval of ``color_names``.

    Returns 1.0 for an empty ``color_names`` — nothing was asked for, so nothing
    fails — and 0.0 for a degenerate or out-of-frame box.
    """
    if not color_names:
        return 1.0
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    combined = np.zeros(roi.shape[:2], dtype=bool)
    for name in color_names:
        for lo, hi in COLOR_HSV_INTERVALS.get(name, []):
            m = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
            combined |= m > 0
    n_pix = roi.shape[0] * roi.shape[1]
    return float(np.count_nonzero(combined)) / float(n_pix) if n_pix else 0.0
