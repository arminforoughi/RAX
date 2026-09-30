"""What to pick: a detector, plus the few numbers that differ between objects.

A :class:`Target` finds its object in a wrist-camera frame and, optionally, says which
way it lies. Two are provided:

* :class:`ColourTarget`: HSV blobs (the tube caps). Fast, no model.
* :class:`PromptTarget`: an open-vocabulary detector (YOLO-World) and a text prompt,
  so "cup" or "screwdriver" works without training anything.

Adding an object means writing ``detect`` (and ``axis`` if the default is not good
enough), not touching the pick.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """One object in the image: its grasp pixel, its box and its label."""

    u: float
    v: float
    box: tuple[float, float, float, float]
    label: str = ""
    source: str = "detector"          # or "tracked": held by OpenCV between detections

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def edge_axis(bgr: np.ndarray, det: Detection, min_elongation: float = 2.0) -> float | None:
    """The object's long axis in the image (degrees, mod 180) from the edges in its box.

    Principal direction of the edge pixels. None for a round or square object, where
    there is no long axis and the grasp may come from any angle.
    """
    x0, y0, x1, y1 = (int(round(c)) for c in det.box)
    roi = bgr[max(0, y0):y1, max(0, x0):x1]
    if roi.size == 0:
        return None
    edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 60, 160)
    ys, xs = np.nonzero(edges)
    if len(xs) < 20:
        return None
    pts = np.stack([xs, ys], 1).astype(float)
    evals, evecs = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))
    if evals[1] < min_elongation ** 2 * max(evals[0], 1e-9):
        return None
    vx, vy = evecs[:, 1]
    return math.degrees(math.atan2(vy, vx)) % 180.0


@dataclass
class Target:
    """Base class. Override ``detect``; the numbers have sensible defaults."""

    name: str = "object"
    #: Fingertip height at the grasp, metres: about half the object's height.
    grasp_z: float = 0.010
    #: Jaw opening to approach with, percent.
    open_pct: float = 45.0
    #: Jaws stopping wider than this mean two were taken (None: do not check).
    max_grip_pct: float | None = None

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    def axis(self, bgr: np.ndarray, det: Detection) -> float | None:
        """Image angle of the object's long axis (mod 180), or None if it has none."""
        return edge_axis(bgr, det)


@dataclass
class ColourTarget(Target):
    """Objects found by colour: here, test-tube caps.

    ``colours`` limits which cap colours count. The axis is read off the tube BODY,
    not the cap: the body is the bright streak leaving the cap, found by contrast.
    """

    colours: tuple[str, ...] = ("green", "blue", "red")
    #: Pixels to ignore (x0, y0, x1, y1), e.g. a jaw that reads as a cap.
    ignore: tuple[tuple[float, float, float, float], ...] = ()
    min_contrast: float = 18.0

    def detect(self, bgr):
        from rax.perception.tube_caps import find_caps
        out = []
        for c in find_caps(bgr, colours=self.colours):
            if any(x0 <= c.x <= x1 and y0 <= c.y <= y1 for x0, y0, x1, y1 in self.ignore):
                continue
            out.append(Detection(c.x, c.y, tuple(float(b) for b in c.bbox), c.colour))
        return out

    def axis(self, bgr, det):
        return body_axis(bgr, det, self.min_contrast)


def body_axis(bgr: np.ndarray, det: Detection, min_contrast: float = 18.0) -> float | None:
    """Direction of a tube's body from its cap, in the image (degrees, mod 180), or None.

    Rays out from the cap, scored by how much brighter they are than the rays 36deg to
    either side, so a uniformly bright bench scores nothing and a pale tube on a dark
    mat scores high. The darker half of each ray is what counts, so a gap kills it.
    """
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (7, 7), 0).astype(float)
    H, W = g.shape
    x0, y0, x1, y1 = det.box
    r0 = int(0.45 * max(x1 - x0, y1 - y0))

    def ray(a):
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        v = [g[int(det.v + sa * r), int(det.u + ca * r)] for r in range(r0, r0 + 150, 3)
             if 0 <= int(det.v + sa * r) < H and 0 <= int(det.u + ca * r) < W]
        return float(np.mean(sorted(v)[:len(v) // 2])) if len(v) >= 15 else None

    rays = {a: ray(a) for a in range(0, 360, 3)}
    best = None
    for a, v in rays.items():
        side = [rays.get((a + d) % 360) for d in (-36, 36)]
        side = [s for s in side if s is not None]
        if v is None or not side:
            continue
        score = v - float(np.mean(side))
        if best is None or score > best[0]:
            best = (score, a)
    if best is None or best[0] < min_contrast:
        return None
    return float(best[1]) % 180.0


#: Colour word -> HSV intervals (OpenCV, H 0-179). Wide on purpose: this only has to
#: reject an impostor ("red cube" latching onto the green one), not segment anything.
COLOUR_HSV = {
    "red": [((0, 70, 50), (12, 255, 255)), ((165, 70, 50), (180, 255, 255))],
    "orange": [((8, 100, 80), (22, 255, 255))],
    "yellow": [((20, 80, 80), (38, 255, 255))],
    "green": [((38, 50, 50), (88, 255, 255))],
    "blue": [((100, 50, 50), (128, 255, 255))],
    "purple": [((128, 50, 50), (152, 255, 255))],
    "pink": [((145, 40, 80), (175, 255, 255))],
    "black": [((0, 0, 0), (179, 255, 90))],
    "white": [((0, 0, 180), (179, 60, 255))],
    "grey": [((0, 0, 80), (179, 80, 200))],
    "gray": [((0, 0, 80), (179, 80, 200))],
}


def colour_names(label: str) -> list[str]:
    """The colour words in ``label``, as whole words ("bored" does not name red)."""
    return [c for c in COLOUR_HSV if re.search(rf"\b{c}\b", label.lower())]


def colour_fraction(bgr: np.ndarray, box, label: str) -> float:
    """How much of ``box`` is the colour ``label`` names (1.0 if it names none)."""
    names = colour_names(label)
    if not names:
        return 1.0
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    roi = bgr[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hit = np.zeros(roi.shape[:2], bool)
    for c in names:
        for lo, hi in COLOUR_HSV[c]:
            hit |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0
    return float(hit.mean())


def colour_blob(bgr: np.ndarray, label: str, min_area: float = 900.0,
                min_solidity: float = 0.75, max_aspect: float = 3.0):
    """The largest solid blob of the colour ``label`` names, as a box, or None.

    Saturation at least 100 so bare wood (S 12-40) never counts; opened to drop
    specks; a blob must be big, solid and not a streak, as the old cube trackers
    required.
    """
    names = colour_names(label)
    if not names:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for c in names:
        for lo, hi in COLOUR_HSV[c]:
            lo = (lo[0], max(lo[1], 100), max(lo[2], 50))
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    best = None
    for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        solid = area / max(cv2.contourArea(cv2.convexHull(cnt)), 1.0)
        if solid < min_solidity or max(w, h) > max_aspect * min(w, h):
            continue
        if best is None or area > best[0]:
            best = (area, (float(x), float(y), float(x + w), float(y + h)))
    return None if best is None else best[1]


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class PromptTarget(Target):
    """Anything a text prompt can name, found by an open-vocabulary detector."""

    prompt: str = "cup"
    model_path: str = "yolov8s-worldv2.pt"
    min_confidence: float = 0.06    # the old server's floor: YOLO-World scores real objects low
    #: A box labelled with a colour must be at least this much that colour.
    min_colour: float = 0.15
    #: Same-label boxes overlapping more than this are one object.
    nms_iou: float = 0.55
    #: When the detector misses a label that names a colour ("red cube"), find the
    #: colour itself: open-vocabulary models score plain coloured blocks very low.
    colour_fallback: bool = True
    _detector: object = field(default=None, repr=False)
    _lock: object = field(default_factory=threading.Lock, repr=False)

    def detect(self, bgr):
        with self._lock:               # one model, called from the view and the pick
            return self._detect(bgr)

    def _detect(self, bgr):
        if self._detector is None:
            from ultralytics import YOLOWorld  # pip install "rax[detect]"
            self._detector = YOLOWorld(self.model_path)
            # "red cube, green cube" is two classes; each detection keeps its own name
            self._detector.set_classes([c.strip() for c in self.prompt.split(",") if c.strip()])
        found = []
        for r in self._detector.predict(bgr, conf=self.min_confidence, verbose=False):
            for box, c, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                    r.boxes.conf.tolist()):
                label = r.names[int(c)]
                if colour_fraction(bgr, box, label) >= self.min_colour:   # not an impostor
                    found.append((conf, label, tuple(box)))
        out = []
        for _conf, label, box in sorted(found, reverse=True):             # best first
            if all(d.label != label or box_iou(d.box, box) < self.nms_iou for d in out):
                x0, y0, x1, y1 = box
                out.append(Detection((x0 + x1) / 2, (y0 + y1) / 2, box, label))
        if self.colour_fallback:
            for label in (c.strip() for c in self.prompt.split(",") if c.strip()):
                if any(d.label == label for d in out):
                    continue
                box = colour_blob(bgr, label)
                if box is not None:
                    x0, y0, x1, y1 = box
                    out.append(Detection((x0 + x1) / 2, (y0 + y1) / 2, box, label,
                                         source="colour"))
        return out
