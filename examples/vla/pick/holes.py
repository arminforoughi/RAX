"""Rack holes in the wrist view: which exist, and which are empty.

Ported from the lab checkout's `perception/holes.py`. `pick.py` uses ``find_holes(bgr)``,
``pick_free_hole(holes, prefer=(x, y))`` and ``draw(frame, holes, target=None)``.

ONLY EVER FROM THE WRIST CAMERA, and `pick.py`'s own docstring says why: from overhead a
rack is about 65x85px with 8px holes and Hough finds two of them, while from the
demonstrated release pose the rack fills the wrist frame and the holes are around 45px
across. This assumes the latter.

"FREE" IS DECIDED BY DARKNESS. An empty hole is a hole -- it looks into the rack's
shadow and reads near-black. One with a tube in it shows the tube's cap or its bright
plastic body. So the mean brightness inside a hole, compared with the rack surface
around it, is the whole test; it needs no colour model and it does not care which cap
colour is in the way.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["Hole", "find_holes", "pick_free_hole", "draw"]


@dataclass(frozen=True)
class Hole:
    x: float
    y: float
    r: float
    free: bool
    darkness: float           # 0..1, how much darker than the rack around it


#: Hough parameters for a rack that fills the wrist frame.
MIN_R, MAX_R = 14, 60
#: A hole is "free" when its interior is at least this much darker than its surround.
FREE_MARGIN = 0.16


def find_holes(bgr) -> list[Hole]:
    """Every rack hole visible, left to right."""
    if bgr is None or bgr.size == 0:
        return []
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grey = cv2.medianBlur(grey, 5)
    circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=int(MIN_R * 1.6), param1=110, param2=28,
                               minRadius=MIN_R, maxRadius=MAX_R)
    if circles is None:
        return []
    out: list[Hole] = []
    H, W = grey.shape[:2]
    g = grey.astype(np.float32) / 255.0
    for cx, cy, r in circles[0]:
        cx, cy, r = float(cx), float(cy), float(r)
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        inner = np.zeros(grey.shape, np.uint8)
        cv2.circle(inner, (int(cx), int(cy)), max(2, int(r * 0.62)), 255, -1)
        ring = np.zeros(grey.shape, np.uint8)
        cv2.circle(ring, (int(cx), int(cy)), int(r * 1.45), 255, -1)
        cv2.circle(ring, (int(cx), int(cy)), int(r * 1.05), 0, -1)
        if inner.sum() == 0 or ring.sum() == 0:
            continue
        i_mean = float(g[inner > 0].mean())
        r_mean = float(g[ring > 0].mean())
        darkness = r_mean - i_mean
        out.append(Hole(cx, cy, r, darkness >= FREE_MARGIN, darkness))
    out.sort(key=lambda h: h.x)
    return out


def pick_free_hole(holes, prefer=None) -> Hole | None:
    """The free hole to aim for: the one nearest ``prefer``, else the darkest.

    Nearest-to-the-grasp-point rather than first-found, because the carry is already
    roughly over the rack when this runs -- the shortest correction is the one least
    likely to shed the tube, which the jaws hold by friction alone.
    """
    free = [h for h in holes if h.free]
    if not free:
        return None
    if prefer is None:
        return max(free, key=lambda h: h.darkness)
    px, py = float(prefer[0]), float(prefer[1])
    return min(free, key=lambda h: (h.x - px) ** 2 + (h.y - py) ** 2)


def draw(frame, holes, target=None):
    """Annotate a copy: every hole, free ones filled, the target ringed."""
    vis = frame.copy()
    for h in holes:
        colour = (0, 220, 0) if h.free else (0, 0, 220)
        cv2.circle(vis, (int(h.x), int(h.y)), int(h.r), colour, 2)
        cv2.putText(vis, f"{h.darkness:.2f}", (int(h.x) - 18, int(h.y) - int(h.r) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
    if target is not None:
        cv2.circle(vis, (int(target.x), int(target.y)), int(target.r) + 7,
                   (0, 255, 255), 3)
        cv2.drawMarker(vis, (int(target.x), int(target.y)), (0, 255, 255),
                       cv2.MARKER_CROSS, 18, 2)
    return vis
