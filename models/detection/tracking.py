"""Following an object between detections.

A detector that runs on its own cycle leaves gaps, and during those gaps the last
reported box is simply stale — it does not move as the object or the camera moves, then
jumps when the next detection lands. On a wrist camera closing on a target that reads as
"the box cannot keep up".

Two trackers fill the gap, differing in what they key on:

:class:`AnchorTracker`
    For objects defined by colour. Strict HSV segmentation plus window continuity, and
    an anchor in the BASE frame: an EMA of back-projected fixes that predicts the pixel
    window from the current arm pose after a dropout. That prediction is the eye-in-hand
    insight — the camera's motion is known exactly, so where the object went is known
    too, and only needs a camera geometry to compute.

:class:`PixelTracker`
    For any label at all. Classic CamShift over a colour-histogram back-projection,
    tagged from a fresh detection box and then followed every frame in between.

Nothing here needs a robot. AnchorTracker takes a camera geometry so its prediction
works for either a wrist-mounted or a world-fixed camera.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from perception.measure import silhouette_mask

__all__ = ["Track", "AnchorTracker", "PixelTracker", "HSV_BANDS", "HSV_BANDS_SOFT",
           "PIXEL_TRACK_MAX_AGE_S", "PIXEL_TRACK_MIN_RESPONSE"]


HSV_BANDS = {
    # saturation floor 110 keeps the warm wood grain out of "red"
    "red": [((0, 110, 80), (9, 255, 255)), ((170, 110, 80), (179, 255, 255))],
    "green": [((38, 80, 60), (85, 255, 255))],
}
# Relaxed bands for the second pass INSIDE a predicted window only — the
# looming gripper shades the object (saturation/value drop) during approach.
HSV_BANDS_SOFT = {
    "red": [((0, 70, 45), (11, 255, 255)), ((168, 70, 45), (179, 255, 255))],
    "green": [((36, 55, 40), (88, 255, 255))],
}



class Track:
    __slots__ = ("uv", "bbox_xyxy", "area_px", "clipped", "t")

    def __init__(self, uv, bbox, area, clipped, t):
        self.uv, self.bbox_xyxy = uv, bbox
        self.area_px, self.clipped, self.t = area, clipped, t


class AnchorTracker:
    """Strict-HSV blob tracker with window continuity and a base-frame anchor.

    The anchor (EMA of back-projected fixes) predicts the pixel window through
    detector dropouts using the CURRENT FK pose — the eye-in-hand insight.
    """

    def __init__(self, color, geometry, min_area=900):
        self.color = color
        self.geom = geometry
        self.min_area = int(min_area)
        self.last: Track | None = None
        self.p_anchor: np.ndarray | None = None
        self.anchor_t = 0.0

    def reset(self):
        self.last = None
        self.p_anchor = None

    def _mask(self, rgb, soft=False):
        hsv = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2HSV)
        m = np.zeros(hsv.shape[:2], np.uint8)
        bands = (HSV_BANDS_SOFT if soft else HSV_BANDS)[self.color]
        for lo, hi in bands:
            m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def _largest(self, mask, ox=0, oy=0, shape=None):
        n, _l, stats, _c = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best = None
        for i in range(1, n):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a < self.min_area:
                continue
            if best is None or a > best[0]:
                x, y, w, h = (int(stats[i, j]) for j in
                              (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
                best = (a, (x + ox, y + oy, x + w + ox, y + h + oy))
        if best is None:
            return None
        a, (x1, y1, x2, y2) = best
        H, W = shape
        clipped = x1 <= 1 or y1 <= 1 or x2 >= W - 2 or y2 >= H - 2
        return Track(((x1 + x2) / 2.0, (y1 + y2) / 2.0), (x1, y1, x2, y2), a, clipped, time.time())

    def predict_uv(self, T_base_cam):
        if self.p_anchor is None:
            return None
        p_cam = T_base_cam[:3, :3].T @ (self.p_anchor - T_base_cam[:3, 3])
        if p_cam[2] < 0.02:
            return None
        g = self.geom
        return (g.cx + g.fx * p_cam[0] / p_cam[2],
                g.cy + g.fy * p_cam[1] / p_cam[2])

    def update_anchor(self, uv, z_m, T_base_cam, now):
        g = self.geom
        d = np.array([(uv[0] - g.cx) / g.fx, (uv[1] - g.cy) / g.fy, 1.0])
        p = T_base_cam[:3, 3] + T_base_cam[:3, :3] @ (d * z_m)  # z along optical axis
        if self.p_anchor is None:
            self.p_anchor = p
        else:
            self.p_anchor = 0.6 * self.p_anchor + 0.4 * p
        self.anchor_t = now

    def track(self, rgb, T_base_cam=None):
        H, W = rgb.shape[:2]
        windows = []
        centre = None
        if self.last is not None and time.time() - self.last.t < 1.5:
            centre = self.last.uv
        elif T_base_cam is not None:
            centre = self.predict_uv(T_base_cam)   # FK prediction after dropout
        if centre is not None:
            cxp, cyp = int(centre[0]), int(centre[1])
            r = 140
            windows.append((max(0, cxp - r), max(0, cyp - r), min(W, cxp + r), min(H, cyp + r)))
        windows.append(None)
        for w in windows:
            if w is None:
                tr = self._largest(self._mask(rgb), 0, 0, (H, W))
            else:
                x1, y1, x2, y2 = w
                tr = self._largest(self._mask(rgb[y1:y2, x1:x2]), x1, y1, (H, W))
                if tr is None:
                    # shaded/blurred object inside a trusted window: relax bands
                    tr = self._largest(self._mask(rgb[y1:y2, x1:x2], soft=True), x1, y1, (H, W))
            if tr is not None:
                self.last = tr
                return tr
        return None


# ---------------- generic per-object pixel tracker ----------------
# WHY THIS EXISTS. YOLO only reports every ~2.5 s (yolo_worker's cycle), and for
# any label other than the two cubes, find_label() was just handing back that SAME
# cached box, unmoved, for the whole 2.5 s — then it JUMPS to wherever the object
# is now. As the arm approaches and the camera moves, that reads as "the box can't
# keep up" / shaky, because it isn't tracking anything between detections at all.
#
# The fix is the same idea AnchorTracker already uses for the cubes (window
# continuity between confirmations) generalised to ANY appearance, via classic
# CamShift: colour-histogram back-projection + mean-shift. This box's contrib
# modules (CSRT/KCF/MOSSE) are not installed on this machine (checked: only
# cv2.TrackerMIL is present, and CamShift needs nothing beyond core OpenCV), so
# CamShift is also the pragmatic choice, not just the simple one.
#
# yolo_worker "tags" a tracker with a fresh ground-truth box every ~2.5 s;
# find_label() then "tracks" it every call in between — every control-tick, not
# every 2.5 s — so the box actually follows the object instead of teleporting.
PIXEL_TRACK_MAX_AGE_S = 8.0   # no fresh YOLO tag within this long -> stop trusting
                              # pure pixel tracking, it may have drifted onto
                              # something else entirely
PIXEL_TRACK_MIN_RESPONSE = 12.0  # mean back-projection value inside the tracked
                                 # window; below this the histogram is no longer
                                 # matching anything real (object left / occluded)


class PixelTracker:
    """CamShift tracker for one object instance, tagged from a YOLO box and then
    followed frame-to-frame by colour-histogram mean-shift."""

    def __init__(self, label):
        self.label = label
        self.hist = None
        self.window = None        # (x, y, w, h)
        self.tagged_t = 0.0
        self.last: Track | None = None   # publish() reads this; it must never call
                                          # track() itself, or CamShift runs twice
                                          # per frame from two independent call sites
        # The actual pixels driving the current track, for the FPV overlay — a
        # boolean crop plus its (x, y) origin, refreshed every track() call.
        # None until the first successful track.
        self.pixel_mask = None
        self.pixel_origin = (0, 0)

    def tag(self, rgb, xyxy):
        """(Re)acquire from a FRESH, trusted detection box.

        CHOOSE PIXELS SMARTLY rather than histogramming the whole rectangular
        box: a YOLO box is axis-aligned and a diagonal or round object often
        fills only half of it, so histogramming the full box mixes in
        background pixels from its corners — that is what let a track slide
        onto the table the moment the object rotated. Reuse silhouette_mask
        (Lab colour-distance from a ring just outside the box, the same one
        perception.measure uses to separate an object from the table) to find
        the actual object pixels; it stays correct on a dark object because
        Lab distance is not a saturation/value test.

        THE TWO MASKS HAVE DIFFERENT JOBS AND MUST NOT BE MERGED BY INTERSECTION.
        `sil` answers "is this pixel the object" (Lab colour-distance, works on a
        black pen same as a bright one). The saturation/value gate answers "is
        this pixel's HUE trustworthy enough to put in a hue histogram" — a black
        or white pixel has essentially RANDOM hue, and CamShift keys on hue.
        AND-ing them together was a real bug caught on a synthetic dark object:
        the silhouette correctly found the pen (V~30), the sv gate rejected it
        for being too dark, and the code fell back to the sv gate ALONE — which
        happily kept the bright wood BACKGROUND instead. So: sv gate narrows the
        HISTOGRAM only, never decides what the object is. And when an object is
        genuinely achromatic and the narrowed set is too small to build a useful
        histogram, widen it back to the full silhouette rather than drop to zero
        — a noisy hue signal on the right pixels beats a clean one on the wrong
        pixels, and it also avoids keying on "generic dark blob", which risks
        matching our own black gripper the moment it enters frame.
        """
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
        H, W = rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 6 or y2 - y1 < 6:
            return
        rgb_u8 = np.asarray(rgb, np.uint8)
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        roi = hsv[y1:y2, x1:x2]
        sv_gate = cv2.inRange(roi, np.array((0, 60, 32), np.uint8),
                              np.array((179, 255, 255), np.uint8)) > 0
        sil = silhouette_mask(rgb_u8, (x1, y1, x2, y2))
        if sil is not None:
            obj_mask = sil[y1:y2, x1:x2] > 0
        else:
            # no reliable silhouette (object same colour as the table, or the box
            # too small for a background ring) — the sv gate is the only signal
            # left, imperfect as it is
            obj_mask = sv_gate
        hist_mask = obj_mask & sv_gate
        if int(np.count_nonzero(hist_mask)) < 0.15 * max(1, int(np.count_nonzero(obj_mask))):
            hist_mask = obj_mask        # achromatic object: accept a noisy hue signal
        mask_u8 = (hist_mask.astype(np.uint8)) * 255
        hist = cv2.calcHist([roi], [0], mask_u8, [30], [0, 180])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        self.hist = hist
        self.window = (x1, y1, x2 - x1, y2 - y1)
        self.tagged_t = time.time()
        self.pixel_mask = obj_mask
        self.pixel_origin = (x1, y1)

    def track(self, rgb):
        """One CamShift step on the CURRENT frame. None if lost or never tagged."""
        if self.hist is None or self.window is None:
            return None
        if time.time() - self.tagged_t > PIXEL_TRACK_MAX_AGE_S:
            self.hist = None            # stale — force a re-tag before trusting this again
            return None
        H, W = rgb.shape[:2]
        wx, wy, ww, wh = self.window
        if ww < 4 or wh < 4 or wx >= W or wy >= H:
            return None
        hsv = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2HSV)
        backproj = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
        term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        try:
            _rot, window = cv2.CamShift(backproj, self.window, term)
        except cv2.error:
            return None
        x, y, w, h = window
        if w < 6 or h < 6:
            return None                 # collapsed — the object is not here
        x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        bp_roi = backproj[y1:y2, x1:x2]
        response = float(np.mean(bp_roi)) if bp_roi.size else 0.0
        if response < PIXEL_TRACK_MIN_RESPONSE:
            return None                 # window found nothing that looks like the target
        self.window = (x1, y1, x2 - x1, y2 - y1)
        # WHICH PIXELS, RIGHT NOW, are actually driving this track — for the FPV
        # overlay. Free: Otsu-threshold the back-projection crop CamShift just
        # used, no extra frame work. This is the honest answer to "what is being
        # tracked", since it moves and reshapes with the object every frame,
        # unlike re-showing the mask captured at tag() time.
        if bp_roi.size >= 16 and bp_roi.max() > 0:
            _t, m = cv2.threshold(bp_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.pixel_mask = m > 0
            self.pixel_origin = (x1, y1)
        else:
            self.pixel_mask = None
        clipped = x1 <= 1 or y1 <= 1 or x2 >= W - 2 or y2 >= H - 2
        tr = Track(((x1 + x2) / 2.0, (y1 + y2) / 2.0), (x1, y1, x2, y2),
                   int((x2 - x1) * (y2 - y1)), clipped, time.time())
        self.last = tr
        return tr
