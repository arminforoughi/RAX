"""Keep hold of an object between detections, with plain OpenCV.

An open-vocabulary detector comes and goes: the same cup scores 0.3 in one frame and
nothing in the next. :class:`StickyTarget` wraps any :class:`Target`. Every time the
detector finds the object it learns two cheap appearance features from the box:

* a hue-saturation histogram (what colour it is), and
* a grey template (what it looks like).

When the detector misses, it searches around the last box (the arm moves between
looks): template matching over a range of sizes gives the position, and the colour
histogram has to agree before the match is believed. A sure match refreshes the
template, so the look can change as the hand closes in; the colour never relearns. A lock is dropped after ``hold_s`` without a fresh detection, so a tracker can
bridge flicker but never replaces the detector for long.
"""

from __future__ import annotations

import threading
import time

import cv2

from .targets import Detection, Target


def _hist(bgr, box):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 40, 30), (180, 255, 255))     # ignore grey and dark
    h = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def _clip(box, w, h):
    x0, y0, x1, y1 = box
    return (max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1))


class StickyTarget(Target):
    """``inner``'s detections, plus an OpenCV track of the chosen one when it drops out."""

    def __init__(self, inner: Target, hold_s: float = 3.0, min_ncc: float = 0.5,
                 min_colour: float = 0.5, search_px: int = 120):
        self.inner = inner
        self.name, self.grasp_z, self.open_pct = inner.name, inner.grasp_z, inner.open_pct
        self.max_grip_pct = inner.max_grip_pct
        self.hold_s, self.min_ncc, self.min_colour = hold_s, min_ncc, min_colour
        self.search_px = search_px
        self._box = self._label = self._hist = self._tmpl = None
        self._t = 0.0
        self._lock = threading.Lock()            # the live view and the pick share it

    def __getattr__(self, name):                 # prompt, colours, ... of the inner target
        return getattr(self.__dict__["inner"], name)

    def axis(self, bgr, det):
        return self.inner.axis(bgr, det)

    def detect(self, bgr):
        with self._lock:
            return self._detect(bgr)

    def _detect(self, bgr):
        dets = self.inner.detect(bgr)
        if dets:
            self._learn(bgr, self._choose(dets))
            return dets
        d = self._track(bgr)
        return [d] if d is not None else []

    # ---- learning and tracking ------------------------------------------------------
    def _choose(self, dets):
        if self._box is None:
            return max(dets, key=lambda d: d.area)
        cx, cy = (self._box[0] + self._box[2]) / 2, (self._box[1] + self._box[3]) / 2
        return min(dets, key=lambda d: (d.u - cx) ** 2 + (d.v - cy) ** 2)

    def _learn(self, bgr, d):
        h, w = bgr.shape[:2]
        box = _clip(d.box, w, h)
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            return
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        self._box, self._label, self._t = box, d.label, time.time()
        self._hist = _hist(bgr, box)
        self._tmpl = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

    def _track(self, bgr):
        if self._box is None or time.time() - self._t > self.hold_s:
            return None
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = self._box
        # the arm moves between looks: search well around the box, at a range of sizes
        m = max(self.search_px, x1 - x0, y1 - y0)
        sx0, sy0 = int(max(0, x0 - m)), int(max(0, y0 - m))
        sx1, sy1 = int(min(w, x1 + m)), int(min(h, y1 + m))
        grey = cv2.cvtColor(bgr[sy0:sy1, sx0:sx1], cv2.COLOR_BGR2GRAY)
        best = None
        for s in (0.7, 0.85, 1.0, 1.18, 1.4):
            t = cv2.resize(self._tmpl, None, fx=s, fy=s)
            if t.shape[0] >= grey.shape[0] or t.shape[1] >= grey.shape[1] or min(t.shape) < 6:
                continue
            _, ncc, _, (px, py) = cv2.minMaxLoc(cv2.matchTemplate(grey, t, cv2.TM_CCOEFF_NORMED))
            if best is None or ncc > best[0]:
                best = (ncc, sx0 + px, sy0 + py, t.shape[1], t.shape[0])
        if best is None or best[0] < self.min_ncc:
            return None
        ncc, bx, by, tw, th = best
        box = _clip((bx, by, bx + tw, by + th), w, h)
        colour = cv2.compareHist(self._hist, _hist(bgr, box), cv2.HISTCMP_CORREL)
        if colour < self.min_colour:
            return None
        self._box = box
        if ncc > 0.8:                             # a sure match: follow its changing look
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            self._tmpl = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        return Detection((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box, self._label,
                         source="tracked")
