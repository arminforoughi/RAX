"""Find tube caps by colour — with the windows MEASURED ON THIS RIG'S CAMERA.

WHY NOT `examples/vla/pick/caps2.py`. That file does the same job and its shape is right,
but every constant in it was measured somewhere else: the X250's wrist camera, looking at
tubes on a near-black rubber mat. Pointed at the SO-101's OAK-D over a pale wooden
turntable it returned FOURTEEN caps in a frame containing two, and `restrict_to_mat` made
no difference at all because there is no dark mat here to restrict to. Among the twelve
false positives were the gripper's own body, the turntable, and the frame edges.

It was not broken. It was calibrated for a different bench, which is the same class of
mistake as a hand-eye transform copied between arms — and the fix is the same one: measure
it here.

WHAT WAS MEASURED, on a live frame from this camera (640x480, OAK-D, wrist mount):

    blue cap         H  96- 99    S 209-233    V 187-231
    green cap        H  83- 87    S 190-235    V 119-148
    turntable wood   H   0-178    S  12- 40    V  74-173
    wood, lit        H 101-111    S  11- 16    V 147-165
    dark ring        H   0-177    S   0-173    V  32-136
    gripper body     H   6-173    S   8-188    V  30-103
    wooden block     H 101-109    S  48-201    V  59-174

TWO THINGS FALL OUT OF THAT TABLE, and both matter more than the hue windows.

FIRST, SATURATION IS THE DISCRIMINATOR, not hue. The caps are strongly coloured plastic
(S >= 190); everything that fooled the ported detector is washed out (wood, S ~ 12-40) or
dark (the gripper, V ~ 44). caps2 gates at S >= 70-80, which on a black mat is plenty and
on pale wood lets the entire bench through. Gating on S and V first, and only then asking
about hue, removes the twelve false positives without needing to know where the mat is.

SECOND, GREEN AND BLUE ARE ONLY ~10 DEGREES OF HUE APART HERE, and caps2 splits them at
H=85 — with this green cap measuring a median of 86. The split sat inside the measurement.
It is moved to 92, in the gap that was actually observed, so a green cap cannot be read as
blue by one unit of sensor noise.

GOLD IS NOT OFFERED. The X250's rig has gold caps and this one does not, and pale wood
under warm light lands squarely in gold's hue window with enough saturation to pass in
places. A colour nobody here has a cap for is all false positives and no true ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["Cap", "find_caps", "CAP_HSV", "MIN_SAT", "MIN_VAL"]


@dataclass(frozen=True)
class Cap:
    """One cap, in pixels. ``bbox`` is x1, y1, x2, y2."""

    colour: str
    x: float
    y: float
    area: float
    bbox: tuple[float, float, float, float]

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def w(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def h(self) -> float:
        return self.bbox[3] - self.bbox[1]


#: Hue windows, OpenCV's 0-179 scale. Split at 92 — inside the observed gap between a
#: green cap (83-87) and a blue one (96-99), not on top of either.
CAP_HSV: dict[str, tuple[int, int]] = {
    "green": (72, 92),
    "blue": (93, 112),
}

#: THE REAL GATE. Both measured caps sit above 190; the turntable is 12-40 and the
#: gripper's median is 142 but dark. 160 leaves headroom under the caps and well over
#: anything on this bench that is merely tinted.
MIN_SAT = 160
#: Rejects the gripper (V ~ 44) and shadow, without touching the green cap (V 119-148).
MIN_VAL = 85

#: A cap is a blob, not a streak. Measured here: the blue cap reads about 44x81 px at
#: 25 cm and the green about 39x46 — the elongation is the tube body's specular streak
#: bleeding into the cap, so the limit is generous, but a 198x96 smear of bench is not a
#: cap in any orientation.
MAX_ASPECT = 3.2
#: Area bounds in px at this camera's 640x480. The smallest real cap measured ~320 px;
#: the gripper's tape blob measured 3872.
MIN_AREA = 150.0
MAX_AREA = 3200.0


def find_caps(bgr, *, exclude: tuple = (), exclude_r: float = 0.0,
              colours: tuple = ("green", "blue"),
              min_area: float = MIN_AREA, max_area: float = MAX_AREA) -> list[Cap]:
    """Every cap in the frame, biggest first.

    ``exclude`` is a list of pixels — the fingertips — within ``exclude_r`` of which a
    blob is assumed to BE the gripper. Kept from the ported detector because the reason
    for it survives the move: pick.py lost whole runs to the jaws' own tape reading as a
    cap, and its comments record that neither the mat restriction nor the exclusion was
    sufficient alone. Here the saturation gate does most of the work and this is the
    backstop.
    """
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Saturation and value FIRST, hue second — see the module docstring.
    vivid = (S >= MIN_SAT) & (V >= MIN_VAL)

    out: list[Cap] = []
    for colour in colours:
        lo, hi = CAP_HSV[colour]
        mask = (vivid & (H >= lo) & (H <= hi)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _lbl, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, w, h, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                                stats[i, cv2.CC_STAT_AREA])
            if not (min_area <= area <= max_area):
                continue
            if max(w, h) / max(1.0, min(w, h)) > MAX_ASPECT:
                continue
            cx, cy = float(cent[i][0]), float(cent[i][1])
            if exclude_r > 0 and any(
                    math.hypot(cx - fx, cy - fy) <= exclude_r for fx, fy in exclude):
                continue
            out.append(Cap(colour=colour, x=cx, y=cy, area=float(area),
                           bbox=(float(x), float(y), float(x + w), float(y + h))))
    out.sort(key=lambda c: -c.area)
    return out


def draw(bgr, caps, *, aim=None):
    """Annotate a copy: a box and a label per cap. For the FPV overlay."""
    vis = bgr.copy()
    COL = {"green": (60, 220, 90), "blue": (235, 170, 60)}
    for c in caps:
        x1, y1, x2, y2 = (int(v) for v in c.bbox)
        col = COL.get(c.colour, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(vis, f"{c.colour} cap", (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        cv2.drawMarker(vis, (int(c.x), int(c.y)), col, cv2.MARKER_CROSS, 11, 1)
    if aim is not None:
        cv2.drawMarker(vis, (int(aim[0]), int(aim[1])), (0, 255, 255),
                       cv2.MARKER_CROSS, 18, 2)
    return vis
