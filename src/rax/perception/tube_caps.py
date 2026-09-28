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

__all__ = ["Cap", "find_caps", "draw", "CAP_HSV", "MIN_SAT", "MIN_VAL",
           "Axis", "tube_axis", "twist_error", "fold"]


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
    # GOLD/AMBER CAPS. Hue 16-34 in OpenCV's 0-179 scale is yellow through orange.
    # This is the one band that shares hue with the bench itself -- bare wood runs
    # roughly 10-25 -- so it leans entirely on the saturation gate below: the wood
    # measures 12-40 and MIN_SAT is 160. If a gold cap is ever missed on a light
    # bench, measure it before widening the hue; dropping MIN_SAT would turn the whole
    # table into a cap.
    "gold": (16, 34),
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

#: Smallest blob worth calling a cap, px. The smallest real one measured ~320.
MIN_AREA = 150.0

#: Largest. AND IT HAS TO BE BIG, because a cap's apparent area is a function of RANGE
#: and the whole point of an approach is to reduce the range.
#:
#: This was 3200, measured when the cap read 40x48 px from 25 cm away. That number is a
#: measurement of one viewing distance, not of a cap, and using it as a gate meant the
#: detector threw the cap away exactly as the arm got close enough to grasp it: measured
#: mid-approach at 55x66 = 2679 px and still growing, so one more increment crossed the
#: limit and the box vanished. The arm then held, lost track, and closed over the tube's
#: BODY instead of its cap — which is what an operator sees as "it grabbed the tail".
#:
#: Nothing needs this bound to be tight. The gripper is rejected on VALUE (it sits at
#: V ~ 44 against a gate of 85), the bench on SATURATION (12-40 against 160), and a
#: smear on aspect. This only has to stop a whole frame of something saturated from
#: reading as one cap, so it is set at a fifth of the frame.
MAX_AREA = 60000.0


def find_caps(bgr, *, exclude: tuple = (), exclude_r: float = 0.0,
              colours: tuple = ("green", "blue", "gold"),
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
    COL = {"green": (60, 220, 90), "blue": (235, 170, 60),
           "gold": (60, 200, 235)}
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


# ---------------------------------------------------------------------------------
# which way the tube is lying
# ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Axis:
    """A tube's long axis in the image. ``angle_deg`` is folded to [-90, 90)."""

    angle_deg: float
    length: float            # long side of the fitted rectangle, px
    width: float             # short side, px
    centre: tuple[float, float]
    elongation: float

    @property
    def is_confident(self) -> bool:
        """A tube seen end-on is barely elongated and its angle means little."""
        return self.elongation >= 2.0


def fold(deg: float, period: float = 180.0) -> float:
    """Fold an angle into [-period/2, +period/2).

    A tube is 180-degree symmetric — it has no head or tail — so 1 and 179 differ by 2
    degrees, not 178. A controller handed the larger number drives the wrist the wrong
    way through its whole range.
    """
    return ((float(deg) + period / 2.0) % period) - period / 2.0


def tube_axis(bgr, near, *, r: int = 110, min_area: int = 200,
              min_elongation: float = 1.6, border: int = 2) -> Axis | None:
    """The tube's long axis near a cap, or None if nothing elongated is there.

    TWO FIXES OVER THE PORTED `orient.tube_axis`, both forced by this bench. That version
    returned None for BOTH tubes at every window size here, and neither cause was subtle
    once the components were printed.

    FIRST, THE TUBE IS NOT ALWAYS THE BRIGHT THING. The port thresholds once, keeping
    pixels ABOVE Otsu's level, which is right when tubes sit on a near-black rubber mat.
    On this rig's pale wooden turntable the background is brighter than the tube, so the
    tube lands on the dark side of the split and never appears. Both polarities are tried
    here and the better candidate wins.

    SECOND, "LARGEST" SELECTS THE BENCH. The port takes the biggest component. Measured
    in a 220x220 window around the blue cap: the wood came to 35950 px at 220x220 with an
    elongation of 1.00, and the tube to 2459 px at 42x85, elongation 2.02. Largest picks
    the wood, which is then rejected for not being elongated, and the function returns
    None while the tube sits there unexamined. Most-elongated picks the tube.

    AND ONE TRAP IN THE FIX ITSELF, worth recording because the obvious repair is wrong:
    rejecting any component that TOUCHES the window edge does remove the background, and
    also removes the blue tube, which is longer than the window and legitimately runs off
    one side. The background is the thing that SURROUNDS — three or four edges, or most of
    the area — not the thing that reaches one edge.

    Measured on a live frame after the fix: blue +80deg (elongation 2.5-3.5) and green
    +58deg (2.2-6.7), both stable across window radii of 80, 100 and 130 px, and both
    drawn lying along the real tube bodies.
    """
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return None
    h, w = bgr.shape[:2]
    x0, y0 = int(max(0, near[0] - r)), int(max(0, near[1] - r))
    x1, y1 = int(min(w, near[0] + r)), int(min(h, near[1] + r))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    ww, wh = x1 - x0, y1 - y0
    grey = cv2.GaussianBlur(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (5, 5), 0)

    best: Axis | None = None
    for extra in (0, cv2.THRESH_BINARY_INV - cv2.THRESH_BINARY):
        _t, m = cv2.threshold(grey, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU + extra)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ce = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
            cw, ch = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            area = int(st[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            touch = ((x <= border) + (y <= border)
                     + (x + cw >= ww - border) + (y + ch >= wh - border))
            if touch >= 3 or area > 0.45 * ww * wh:
                continue                       # the surrounding bench, not a tube
            ys, xs = np.where(lab == i)
            (cx, cy), (bw, bh), ang = cv2.minAreaRect(
                np.stack([xs, ys], 1).astype(np.float32))
            long_side, short_side = max(bw, bh), min(bw, bh)
            if short_side < 3.0 or long_side < min_elongation * short_side:
                continue
            if bw < bh:                        # minAreaRect's angle names the WIDTH's side
                ang += 90.0
            cand = Axis(fold(ang), float(long_side), float(short_side),
                        (float(cx) + x0, float(cy) + y0),
                        float(long_side) / max(float(short_side), 1e-6))
            if best is None or cand.elongation > best.elongation:
                best = cand
    return best


def twist_error(bgr, near, jaw_axis_deg: float, *, r: int = 110):
    """How far the tube is from square to the jaws: ``(degrees, Axis)`` or None.

    A parallel gripper has to close ACROSS a tube, so the wanted state is the tube's axis
    PERPENDICULAR to the jaw line. The returned angle is the correction to turn through,
    folded, so a caller can divide it by a measured gain and command it.

    ``jaw_axis_deg`` is passed in rather than read from a config file, because it differs
    per arm and per mount — the SO-101 measures it live with /caljaw into a JawFrame,
    while the X250 reads two fixed fingertips out of gripper_geometry.json. This module
    has no business knowing which arm it is looking at.
    """
    ax = tube_axis(bgr, near, r=r)
    if ax is None:
        return None
    return fold(ax.angle_deg - (float(jaw_axis_deg) + 90.0)), ax
