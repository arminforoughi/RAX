"""Turning a 2D detection into a base-frame position.

There is no single right way to do this, because the available facts change with the
rig and the object. This module puts the three that work behind one protocol so a
caller can ask for a position without knowing which one answered:

:class:`StereoLocalizer`
    Uses a measured depth. The only method whose range does not depend on an
    assumption about the object — but it needs working stereo, which this rig does not
    always have.

:class:`PlaneRayLocalizer`
    Solves range and size TOGETHER against the known table plane. Neither the object's
    size nor the plane height has to be assumed; both fall out of the solve, so it is
    self-correcting. Needs a decent hand-eye rotation, since it rides on the sightline.

:class:`ApparentSizeLocalizer`
    Range from how big the object looks, using a prior for its real width. Survives
    close range and a bad plane, but is only as good as the prior — and for elongated
    objects it needs the special handling documented on the class.

:func:`chain` tries them in order and takes the first credible answer. **Depth-absent
is the normal path on this rig, not a fallback**, so the chain must degrade without
complaint.

Every localizer returns :class:`Fix` — position, the range it believed, and which
method produced it, because "how do we know" matters when the arm is about to drive
at the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Fix", "Localizer", "StereoLocalizer", "PlaneRayLocalizer", "ApparentSizeLocalizer",
    "chain", "rotate_xy", "ELONGATED_ASPECT",
]

# Above this footprint aspect ratio, apparent-size ranging off the bbox WIDTH is
# meaningless and the long-axis branch is used instead.
ELONGATED_ASPECT = 2.2

# Plausible metric range for anything on a tabletop, in metres. Outside this the solve
# is broken rather than the object being distant.
RANGE_MIN_M = 0.05
RANGE_MAX_M = 1.20
RANGE_CLIP_M = (0.03, 1.50)

# Depth readings outside this band are not trusted as a metric range.
DEPTH_VALID_M = (0.03, 1.20)


@dataclass
class Fix:
    """One localization result."""

    xy: np.ndarray | None          # base-frame (x, y); None if the solve failed
    range_m: float                 # camera->object range the solve believed
    size_m: float                  # the object width it assumed or implied
    method: str                    # which localizer answered
    z_m: float | None = None       # object centre height, when the method knows it

    @property
    def ok(self) -> bool:
        return self.xy is not None

    @classmethod
    def failed(cls, size_m: float, method: str) -> "Fix":
        return cls(None, float("nan"), float(size_m), method)


@runtime_checkable
class Localizer(Protocol):
    name: str

    def locate(self, bbox, T_base_cam, *, label=None, uv=None, z_m=None) -> Fix: ...


def rotate_xy(xy, deg: float) -> np.ndarray:
    """Rotate a base-frame (x, y) about the origin — corrects a heading error in the
    hand-eye, which otherwise smears the whole map around the base."""
    th = math.radians(float(deg))
    c, s = math.cos(th), math.sin(th)
    return np.array([xy[0] * c - xy[1] * s, xy[0] * s + xy[1] * c], dtype=np.float64)


class _Base:
    """Shared workspace gating and bearing correction."""

    def __init__(self, geometry, priors, *, reach_m=(0.08, 0.55),
                 range_scale=1.0, bearing_offset_deg=0.0, gate_reach=True):
        self.geom = geometry
        self.priors = priors
        self.reach_m = reach_m
        # Whether to reject solves outside the arm's workspace. On for anything feeding
        # the map; off for callers that want the raw solve and judge it themselves.
        self.gate_reach = bool(gate_reach)
        # Live-tunable corrections. Read through the instance so a retune takes effect.
        self.range_scale = float(range_scale)
        self.bearing_offset_deg = float(bearing_offset_deg)

    def _finish(self, xy, rng, size, method, z_m=None) -> Fix:
        """Gate on the workspace, then apply the bearing correction.

        Nothing that matters is outside the arm's own reach, so an object localized
        far beyond it is a broken solve, not a distant object — and letting those in
        is what fills the map with ghosts strung out along the sightline.
        """
        r = float(np.hypot(xy[0], xy[1]))
        if self.gate_reach and not (self.reach_m[0] < r < self.reach_m[1]):
            return Fix.failed(size, method)
        return Fix(rotate_xy(xy, self.bearing_offset_deg), float(rng), float(size),
                   method, z_m)

    @staticmethod
    def _bbox_center(bbox, uv=None):
        if uv is not None:
            return float(uv[0]), float(uv[1])
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class StereoLocalizer(_Base):
    """Range straight from a measured depth."""

    name = "stereo"

    def locate(self, bbox, T_base_cam, *, label=None, uv=None, z_m=None) -> Fix:
        size = self.priors.size_m(label)
        if z_m is None or not (DEPTH_VALID_M[0] < z_m < DEPTH_VALID_M[1]):
            return Fix.failed(size, self.name)
        rng = float(np.clip(float(z_m) * self.range_scale, *RANGE_CLIP_M))
        u, v = self._bbox_center(bbox, uv)
        p = self.geom.point_at_range((u, v), rng, T_base_cam)
        return self._finish(p[:2], rng, size, self.name, z_m=float(p[2]))


class PlaneRayLocalizer(_Base):
    """Solve range and size TOGETHER against the table plane.

    Two facts are actually known: the object sits ON the table, so its centre is at
    ``z_plane + S/2``; and apparent size gives range, ``S = d * w_px / fx``.
    Substituting the second into the first along the sightline ``p(d) = o + d*dir``
    leaves one unknown::

        o_z + d*dir_z = z_plane + d*w/(2*fx)
        =>  d = (o_z - z_plane) / ( w/(2*fx) - dir_z )

    ``dir_z`` is negative when the camera looks down, so the denominator is positive
    and ``d`` is well defined.

    This replaced a range of ``fx * assumed_size / w`` with the size hardcoded. If the
    real object is bigger, every range comes out SHORT, and because the sightline
    points down-and-forward a short range lands the object too NEAR and too HIGH —
    reporting an object centre above the table, which is impossible for something
    resting on it. Here the size falls out of the solve instead of being assumed.
    """

    name = "plane"

    def __init__(self, *a, z_plane: float = 0.0, **kw):
        super().__init__(*a, **kw)
        self.z_plane = float(z_plane)

    def locate(self, bbox, T_base_cam, *, label=None, uv=None, z_m=None) -> Fix:
        size = self.priors.size_m(label)
        x1, y1, x2, y2 = bbox
        w = float(max(4.0, x2 - x1))       # horizontal extent ~ the object's width
        T = np.asarray(T_base_cam, dtype=np.float64)
        o = T[:3, 3]
        u, v = self._bbox_center(bbox, uv)
        dirv = T[:3, :3] @ self.geom.ray((u, v))
        den = w / (2.0 * self.geom.fx) - float(dirv[2])
        if den <= 1e-6:
            return Fix.failed(size, self.name)
        d = (float(o[2]) - self.z_plane) / den
        if not (0.03 < d < 0.80):
            return Fix.failed(size, self.name)
        p = o + d * dirv
        implied = d * w / self.geom.fx     # the object width this implies
        return self._finish(p[:2], d, implied, self.name, z_m=float(p[2]))


class ApparentSizeLocalizer(_Base):
    """Range from apparent size: ``range = fx * real_width / bbox_width``.

    APPARENT-SIZE RANGING IS ONLY VALID FOR OBJECTS THAT LOOK THE SAME FROM EVERY SIDE.
    It treats the bbox width as the object's real width. For a cube or a cup that holds
    at any angle. For an elongated object it is nonsense: measured, a 14 cm pen 40 cm
    away reports 150 cm lying across the view, 14 cm diagonal and 11 cm end-on — a 13x
    swing driven purely by an angle nobody measured. Each frame it rotates slightly,
    the range jumps, and the map grows another ghost somewhere new.

    So elongated objects are measured against their LONG axis instead. The prior's
    geometric mean (3.7 cm for a pen) compared to a bbox WIDTH is meaningless, but the
    bbox's LONGEST side always corresponds to the object's longest axis, foreshortened
    by the viewing angle. That over-estimates range when foreshortened, but it is
    bounded and roughly right instead of swinging with rotation.
    """

    name = "apparent"

    #: Whether to treat the size-derived distance as AXIAL DEPTH (correct) rather than
    #: as range along the sightline (what the original did, and the default here so the
    #: extraction stays behaviour-preserving).
    #:
    #: The pinhole relation gives depth, not range, so walking it along the ray places
    #: an off-axis object too close by cos(off-axis angle) — always inward, growing with
    #: the angle. Measured from 60 cm up: 10 mm at r=20 cm, 48 mm at 35 cm, 90 mm at
    #: 45 cm. This is live on the real rig, because the approach deliberately keeps the
    #: object off-centre, and an inward bias is what a radial push-out fudge corrects
    #: for. Turn this on and re-check the trims on hardware before trusting it.
    axial_depth = False

    def locate(self, bbox, T_base_cam, *, label=None, uv=None, z_m=None) -> Fix:
        size = self.priors.size_m(label)
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            return Fix.failed(size, self.name)

        pm = self.priors.meta(label)
        long_m, short_m = max(pm["w_m"], pm["d_m"]), min(pm["w_m"], pm["d_m"])
        aspect = long_m / max(short_m, 1e-4)
        elongated = aspect > ELONGATED_ASPECT and (
            z_m is None or not (DEPTH_VALID_M[0] < z_m < DEPTH_VALID_M[1]))

        if elongated:
            rng = self.geom.range_from_width(max(float(max(w, h)), 4.0), long_m)
            method = "long-axis"
        elif z_m is not None and DEPTH_VALID_M[0] < z_m < DEPTH_VALID_M[1]:
            rng, method = float(z_m), "apparent+depth"
        else:
            rng, method = self.geom.range_from_width(float(w), size), self.name

        if method != "apparent+depth" and not (RANGE_MIN_M < rng < RANGE_MAX_M):
            return Fix.failed(size, method)
        rng = float(np.clip(rng * self.range_scale, *RANGE_CLIP_M))
        u, v = self._bbox_center(bbox, uv)
        p = (self.geom.backproject((u, v), rng, T_base_cam) if self.axial_depth
             else self.geom.point_at_range((u, v), rng, T_base_cam))
        return self._finish(p[:2], rng, size, method)


def chain(localizers, bbox, T_base_cam, *, label=None, uv=None, z_m=None) -> Fix:
    """First credible answer wins; the last failure is returned if none succeed."""
    last = None
    for loc in localizers:
        fix = loc.locate(bbox, T_base_cam, label=label, uv=uv, z_m=z_m)
        if fix.ok:
            return fix
        last = fix
    return last if last is not None else Fix.failed(0.0, "none")
