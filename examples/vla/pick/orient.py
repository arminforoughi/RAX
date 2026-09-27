"""Which way a tube is lying, and how far that is from square to the jaws.

Ported from the lab checkout's `perception/orient.py`. `pick.py` uses two things:
``tube_axis(bgr, near)`` for the tube's direction, and ``twist_error(bgr, near)``
returning ``(error_degrees, axis)``.

WHY THE JAW LINE IS A CONSTANT HERE. The wrist camera sits PAST the tool joint, so the
camera and the fingers rotate together -- twisting the tool does not move the jaw line in
the image at all. `pick.py` found this the hard way: its first attempt measured the twist
gain on the jaw line and returned "unmeasurable" on every run. What rotates in the
picture is the rest of the world, so the tube is what gets measured and the jaw line is
taken from the fixed fingertip geometry.

A tube is 180-degree symmetric -- it has no head or tail -- so every angle here is folded
into [-90, 90). On a quantity that wraps, 1 and 179 differ by 2 degrees, not 178, and a
controller fed the larger number drives the wrist the wrong way through its whole range.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["Axis", "tube_axis", "twist_error", "jaw_angle_deg", "fold"]

_GEO = json.loads((Path(__file__).parent / "config" / "gripper_geometry.json").read_text())
_FL, _FR = tuple(_GEO["finger_left"]), tuple(_GEO["finger_right"])

#: Radius around the given point searched for the tube body, in pixels.
NEAR_R = 90


@dataclass(frozen=True)
class Axis:
    angle_deg: float          # direction of the tube's long axis, folded to [-90, 90)
    length: float             # long side of the fitted rectangle, px
    width: float              # short side, px
    centre: tuple[float, float]


def fold(deg: float, period: float = 180.0) -> float:
    """Fold an angle into [-period/2, +period/2)."""
    return ((float(deg) + period / 2.0) % period) - period / 2.0


def jaw_angle_deg() -> float:
    """Direction of the line between the fingertips, in the image. A fixed number.

    The fingers are bolted to the camera, so this cannot change without the mount
    changing -- which is exactly why it is read from the measured geometry rather than
    detected per frame.
    """
    return fold(math.degrees(math.atan2(_FR[1] - _FL[1], _FR[0] - _FL[0])))


def tube_axis(bgr, near=None, r: int = NEAR_R) -> Axis | None:
    """The tube's long axis near a point, or None if no elongated body is there.

    The cap is found by colour; the BODY is clear plastic on a dark mat and has no
    colour to speak of, so it is found by contrast instead: inside a window around the
    cap, the brighter pixels are the tube and the mat is what is left.
    """
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    if near is None:
        near = (w / 2.0, h / 2.0)
    x0, y0 = int(max(0, near[0] - r)), int(max(0, near[1] - r))
    x1, y1 = int(min(w, near[0] + r)), int(min(h, near[1] + r))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    win = bgr[y0:y1, x0:x1]
    grey = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
    # Otsu rather than a fixed level: the mat is near-black and the tube is bright, but
    # how bright depends entirely on where the bench light is that day.
    _t, m = cv2.threshold(cv2.GaussianBlur(grey, (5, 5), 0), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ce = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < 150:
        return None
    ys, xs = np.where(lab == i)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    (cx, cy), (bw, bh), ang = cv2.minAreaRect(pts)
    long_side, short_side = max(bw, bh), min(bw, bh)
    if short_side < 3.0 or long_side < 1.6 * short_side:
        return None                      # not elongated: no axis worth reporting
    if bw < bh:                          # minAreaRect's angle names the WIDTH's side
        ang += 90.0
    return Axis(fold(ang), float(long_side), float(short_side),
                (float(cx) + x0, float(cy) + y0))


def twist_error(bgr, near=None) -> tuple[float, Axis] | None:
    """How far the tube is from square to the jaws: ``(degrees, axis)`` or None.

    A parallel gripper has to close ACROSS the tube, so the wanted state is the tube's
    axis perpendicular to the jaw line. The sign is the correction the tool should turn
    through, folded, so the caller can divide by a measured gain and command it.
    """
    ax = tube_axis(bgr, near)
    if ax is None:
        return None
    return fold(ax.angle_deg - (jaw_angle_deg() + 90.0)), ax
