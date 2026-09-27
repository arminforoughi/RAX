"""Find tube caps by colour in the wrist view.

Ported from the lab checkout's `perception/caps2.py`, which is not on this machine. The
interface is the one `pick.py` calls: ``find_caps(bgr, restrict_to_mat=True)`` returning
objects with ``.colour``, ``.x``, ``.y``, ``.area``.

WHAT THE FRAMES ACTUALLY LOOK LIKE, measured on this rig's wrist camera: the tubes sit
on a near-black rubber mat on a pale bench, and the caps are strongly saturated green,
gold/amber and blue. The mat is the useful part -- it is dark and it is where the tubes
are, so restricting to it removes the bench, the racks and most of the room in one step
without needing to know where anything is.

THE JAWS WEAR BLUE TAPE, and `pick.py`'s own comments record what that cost: the tape
reads as a blue cap, the pairing picked the wrong blobs, and the display looked like it
was tracking the gripper while the servo had correctly locked the tube. Two defences
here -- the mat restriction drops anything off the mat, and callers additionally exclude
a radius around each fingertip (EXCLUDE_R in pick.py). Neither alone was enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["Cap", "find_caps", "CAP_HSV"]


@dataclass(frozen=True)
class Cap:
    colour: str
    x: float
    y: float
    area: float
    w: float = 0.0
    h: float = 0.0

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


#: HSV windows for the three cap colours, OpenCV's 0-179 hue scale. Gold is split from
#: green at H=35 and from red below H=10; blue is wide because the caps photograph from
#: cyan to navy depending on how the bench light falls on them.
CAP_HSV: dict[str, list[tuple[tuple, tuple]]] = {
    "green": [((38, 80, 40), (85, 255, 255))],
    "gold": [((12, 90, 70), (34, 255, 255))],
    "blue": [((86, 70, 40), (128, 255, 255))],
}

MIN_AREA = 120.0
# A cap at the grasp moment measured 1548px median across the 113 demonstrations, so
# anything several times that is not a cap. Measured here: the navy gripper with its
# white tape reads as one 18192px "blue cap". The caller excludes a radius around each
# fingertip as well -- neither guard is sufficient alone, which is why both exist.
MAX_AREA = 8000.0


def _mat_mask(bgr) -> np.ndarray | None:
    """The dark mat the tubes lie on, as a filled mask, or None if it is not in view.

    The mat is the largest dark, unsaturated region in the frame. Taking its convex hull
    rather than the raw blob matters: a tube lying across the mat splits it, and the
    hull closes it back up so a cap sitting on the split is not discarded.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv, np.array((0, 0, 0), np.uint8),
                       np.array((179, 120, 110), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _c = cv2.connectedComponentsWithStats(dark, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < 0.06 * bgr.shape[0] * bgr.shape[1]:
        return None
    blob = (lab == i).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    out = np.zeros(blob.shape, np.uint8)
    cv2.fillConvexPoly(out, hull, 255)
    # Pull the boundary back in. The hull is there to close the gaps a tube lying across
    # the mat cuts into it, but it also bulges past the mat's edge -- measured on this
    # rig it reached about 30px beyond and swallowed the tube rack standing alongside,
    # so tubes sitting IN THE RACK were being offered as pick targets.
    return cv2.erode(out, np.ones((25, 25), np.uint8))


def find_caps(bgr, restrict_to_mat: bool = True, min_area: float = MIN_AREA) -> list[Cap]:
    """Every cap visible, biggest first."""
    if bgr is None or bgr.size == 0:
        return []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mat = _mat_mask(bgr) if restrict_to_mat else None
    out: list[Cap] = []
    for colour, windows in CAP_HSV.items():
        m = None
        for lo, hi in windows:
            part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            m = part if m is None else (m | part)
        if mat is not None:
            m = cv2.bitwise_and(m, mat)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, _lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            a = float(st[i, cv2.CC_STAT_AREA])
            if a < min_area or a > MAX_AREA:
                continue
            w, h = float(st[i, cv2.CC_STAT_WIDTH]), float(st[i, cv2.CC_STAT_HEIGHT])
            # A cap is a blob, not a streak. Long thin hits are usually a highlight
            # along a tube's body or the edge of the mat catching the light.
            if max(w, h) > 5.0 * max(min(w, h), 1.0):
                continue
            out.append(Cap(colour, float(ce[i][0]), float(ce[i][1]), a, w, h))
    out.sort(key=lambda c: -c.area)
    return out
