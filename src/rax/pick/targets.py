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


@dataclass
class PromptTarget(Target):
    """Anything a text prompt can name, found by an open-vocabulary detector."""

    prompt: str = "cup"
    model_path: str = "yolov8s-worldv2.pt"
    min_confidence: float = 0.06    # the old server's floor: YOLO-World scores real objects low
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
        out = []
        for r in self._detector.predict(bgr, conf=self.min_confidence, verbose=False):
            for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                out.append(Detection((x0 + x1) / 2, (y0 + y1) / 2, (x0, y0, x1, y1),
                                     r.names[int(c)]))
        return out
