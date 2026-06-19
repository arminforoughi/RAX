"""Open-vocabulary detection — the ``PromptDetector`` seam.

A prompt detector takes a natural-language query ("red box", "blue plate") and
returns 2D boxes for *every* matching object in the frame. It is intentionally
cheap and stateless: :class:`perception.depth_cloud.CloudTracker` runs it on a
throttle to discover and re-confirm the surrounding objects, then hands the
*focused* object off to SAM2 for per-frame mask tracking.

Backends:
    * :class:`YoloWorldDetector` — wraps ultralytics YOLO-World (open vocab).
    * :class:`ColorBlobDetector` — pure-OpenCV HSV fallback so the dev harness
      runs with no model weights; maps colour words in the query to HSV ranges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# Box as (x1, y1, x2, y2) float pixels.
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """One detected instance: a box, a confidence, and the label that matched."""

    box: Box
    confidence: float
    label: str

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@runtime_checkable
class PromptDetector(Protocol):
    """Detect all instances matching ``query`` in an RGB frame."""

    name: str

    def detect(self, rgb: np.ndarray, query: str) -> list[Detection]:
        """Return detections sorted by descending confidence."""
        ...


class YoloWorldDetector:
    """ultralytics YOLO-World wrapper (mirrors lerobot ``perception/yolo_world.py``)."""

    name = "yolo_world"

    def __init__(self, model_path: str = "yolov8s-worldv2.pt", min_confidence: float = 0.2):
        from ultralytics import YOLO  # heavy import, deferred to construction

        self._model = YOLO(model_path)
        self._min_conf = float(min_confidence)
        self._query: str | None = None

    @classmethod
    def try_create(cls, model_path: str, min_confidence: float) -> "YoloWorldDetector | None":
        try:
            return cls(model_path, min_confidence)
        except Exception as e:  # ImportError or missing weights
            logger.warning("[yolo-world] unavailable (%s); falling back", e)
            return None

    def detect(self, rgb: np.ndarray, query: str) -> list[Detection]:
        if query != self._query:
            # YOLO-World takes a class vocabulary; one class = the query.
            self._model.set_classes([query])
            self._query = query
        res = self._model.predict(np.asarray(rgb), verbose=False)[0]
        out: list[Detection] = []
        for b in res.boxes:
            conf = float(b.conf.item())
            if conf < self._min_conf:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append(Detection((x1, y1, x2, y2), conf, query))
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out


# --- Pure-OpenCV fallback -------------------------------------------------

# Colour word -> list of inclusive HSV ranges (OpenCV H in 0..179).
_HSV_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 90, 70), (10, 255, 255)), ((170, 90, 70), (179, 255, 255))],
    "orange": [((11, 110, 90), (22, 255, 255))],
    "yellow": [((23, 90, 90), (34, 255, 255))],
    "green": [((35, 70, 60), (85, 255, 255))],
    "blue": [((95, 90, 60), (130, 255, 255))],
    "purple": [((131, 70, 60), (160, 255, 255))],
}


class ColorBlobDetector:
    """HSV colour-blob detector — no ML. Resolves a colour word from the query."""

    name = "color_blob"

    def __init__(self, min_area_px: float = 200.0):
        self._min_area = float(min_area_px)

    def detect(self, rgb: np.ndarray, query: str) -> list[Detection]:
        import cv2

        color = next((w for w in query.lower().split() if w in _HSV_RANGES), None)
        if color is None:
            return []
        hsv = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in _HSV_RANGES[color]:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # Close holes from interior texture so each object stays a single contour.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[Detection] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self._min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # Confidence ~ fill ratio, a cheap quality proxy.
            conf = float(np.clip(area / max(1.0, w * h), 0.0, 1.0))
            out.append(Detection((float(x), float(y), float(x + w), float(y + h)), conf, query))
        out.sort(key=lambda d: d.area, reverse=True)
        return out


class ForegroundBlobDetector:
    """Grayscale-friendly detector: the prominent object(s) on a plain background.

    No model, no colour. Otsu-thresholds the frame into background/foreground (the
    object is the minority class), returns one box per blob, and drops blobs that
    touch the image border (so the eye-in-hand gripper in the corner is ignored).
    Ideal for tabletop picking with the OAK-D mono stream, where YOLO-World may not
    recognise a plain cube and colour cues are unavailable.
    """

    name = "foreground_blob"

    def __init__(
        self,
        min_area_px: float = 600.0,
        border_margin_px: int = 4,
        min_contrast: float = 22.0,
        max_fg_frac: float = 0.30,
    ):
        self._min_area = float(min_area_px)
        self._border = int(border_margin_px)
        # An object must stand out: mean grey of foreground vs background must differ
        # by at least this much, else the "blob" is just floor noise (Otsu always
        # splits a near-uniform frame and invents a box on blank table).
        self._min_contrast = float(min_contrast)
        self._max_fg_frac = float(max_fg_frac)  # too much foreground => not an object

    def detect(self, rgb: np.ndarray, query: str) -> list[Detection]:
        import cv2

        a = np.asarray(rgb)
        gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a.astype(np.uint8)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Foreground = whichever class is the minority (the object, not the table).
        fg = otsu if int((otsu > 0).sum()) <= gray.size // 2 else cv2.bitwise_not(otsu)
        fg_bool = fg > 0
        n_fg = int(fg_bool.sum())
        # No real object: empty, or the split is the whole frame, or the two classes
        # are nearly the same brightness (a blank, low-contrast floor).
        if n_fg == 0 or n_fg > self._max_fg_frac * gray.size:
            return []
        contrast = abs(float(gray[fg_bool].mean()) - float(gray[~fg_bool].mean()))
        if contrast < self._min_contrast:
            return []
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        h, w = gray.shape
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[Detection] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self._min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            m = self._border
            if x <= m or y <= m or x + bw >= w - m or y + bh >= h - m:
                continue  # touches border -> likely the gripper / frame edge
            conf = float(np.clip(area / max(1.0, bw * bh), 0.0, 1.0))
            out.append(Detection((float(x), float(y), float(x + bw), float(y + bh)), conf, query))
        out.sort(key=lambda d: d.area, reverse=True)
        return out


def make_detector(
    backend: str = "auto",
    *,
    model_path: str = "yolov8s-worldv2.pt",
    min_confidence: float = 0.2,
) -> PromptDetector:
    """Construct a detector.

    ``auto`` prefers YOLO-World, falls back to colour blobs. ``blob`` forces the
    grayscale foreground-blob detector (best for a clear object on a plain table,
    or a mono camera). ``color_blob`` forces the HSV colour detector.
    """
    if backend in ("blob", "foreground_blob"):
        return ForegroundBlobDetector()
    if backend == "color_blob":
        return ColorBlobDetector()
    if backend in ("yolo", "yolo_world", "auto"):
        det = YoloWorldDetector.try_create(model_path, min_confidence)
        if det is not None:
            return det
        if backend != "auto":
            raise RuntimeError("YOLO-World requested but unavailable")
    # Mono tabletop default: the foreground-blob detector beats the colour one when
    # there is no colour word (the real OAK-D stream is grayscale).
    logger.info("[detector] using ForegroundBlobDetector fallback (no YOLO weights)")
    return ForegroundBlobDetector()
