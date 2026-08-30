"""Measuring an object's real size, height and orientation from ONE frame.

Stereo depth is off on this rig (it crashed the camera mid-run), so size and
orientation are measured MONOCULARLY, using the one extra fact available: everything
sits on a known plane.

That fact turns a picture into metric geometry:

* every pixel where the object MEETS THE TABLE (the bottom of its silhouette, column
  by column) back-projects onto the plane at a definite (x, y). Those points are the
  object's real FOOTPRINT, in metres, in the base frame.
* ``cv2.minAreaRect`` over that footprint gives width, depth and yaw directly.
* the top of the silhouette, intersected with the vertical line through the footprint,
  gives the height.

No object-size assumption enters any of this — the class prior is only the fallback and
a sanity check. Nothing here is arm-specific: it needs a camera geometry, a plane, and
somewhere to record why a measurement was rejected.

What one view CAN and CANNOT see, measured on the synthetic bench:

* the extent ACROSS the sightline is recovered to about a millimetre (5.1 cm cube ->
  5.1, 15 cm book -> 15.0, 4.5 cm remote -> 4.6, 8 cm cup -> 7.9).
* the extent ALONG the sightline is NOT observable — the object's far side is behind
  the object. It over-reads on tall things (8 cm cup -> 11.5) and under-reads when the
  long axis points at the camera (22 cm book -> 16.4).

So :meth:`ObjectMeasurer.measure` returns the across-view width plus the direction it
was taken along, and the map fuses readings from several bearings into a footprint.
One view can only ever give one caliper reading.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from rax.perception.object_priors import MAX_TABLE_OBJ_M

__all__ = ["ObjectMeasurer", "silhouette_mask", "contact_points", "solve_height",
           "classify_shape", "pixels_to_table"]


SEG_RING = 8               # px ring around the bbox sampled as table background
SEG_MIN_FRAC = 0.06        # mask must cover this fraction of the bbox to be usable
# Rays that graze the table are useless for ranging: near the horizon one pixel of
# segmentation noise slides the intersection by many centimetres. Require the ray
# to come down onto the plane at least this steeply — sin(incidence) >= this.
# About 13 deg off the table. Shallow, and shallow rays ARE where the range gets
# unreliable — but raising this to 0.35 (20 deg) rejected 100% of real rays on this
# rig (measure_stats: no_contact_line 69/69), because objects at r~40 cm sit near
# the top of the gripper camera's view and are genuinely seen near-grazing. The
# far-flung ghosts it was meant to stop are better caught by the two gates that say
# what is actually wrong with them — size_vs_prior and out_of_workspace — so this
# stays permissive and those do the rejecting.
MIN_TABLE_INCIDENCE = 0.22
# A TALL object's silhouette is WIDEST AT ITS TOP, not at its base — the top is
# nearer the camera, so perspective spreads it. That means the outermost columns of
# the silhouette are the object's near-vertical SIDE edges, and their bottom pixel
# is somewhere up the side wall, NOT on the table. Back-projecting those onto the
# table plane throws them far outward: measured on the synthetic bench, an 8 cm cup
# came out 11.5 cm across from exactly this. A genuine contact pixel sits on the
# base edge, where the silhouette's lower boundary runs roughly HORIZONTALLY; on a
# side edge it plunges. So reject columns where the lower boundary is steeper than
# this many pixels of drop per pixel across.
MAX_CONTACT_SLOPE = 2.5


def pixels_to_table(geom, uv, T_base_cam, z_plane, min_incidence=0.0):
    """Back-project an (N,2) array of pixels onto the horizontal plane z=z_plane.

    Returns (points (M,3), keep_mask (N,)). Rays that point up, that meet the plane
    behind the camera or absurdly far away, or that graze it more shallowly than
    min_incidence, are dropped.
    """
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    d = np.stack([(uv[:, 0] - geom.cx) / geom.fx, (uv[:, 1] - geom.cy) / geom.fy,
                  np.ones(len(uv))], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d @ np.asarray(T_base_cam[:3, :3], np.float64).T
    o = np.asarray(T_base_cam[:3, 3], np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z_plane - o[2]) / d[:, 2]
    keep = ((d[:, 2] < -max(1e-4, float(min_incidence))) & np.isfinite(t)
            & (t > 0.03) & (t < 1.50))
    return o + d[keep] * t[keep, None], keep


def silhouette_mask(rgb, bbox):
    """Separate the object from the table inside a YOLO box.

    Colour-distance segmentation, not GrabCut: a ring of pixels just OUTSIDE the
    box is the table, so any pixel inside the box far enough from that background
    colour (in Lab, which is roughly perceptually uniform) is object. Otsu picks
    the cut so it adapts to contrast instead of needing a tuned threshold. This
    costs ~1 ms against GrabCut's ~60 ms, which matters because sense_2d runs
    inside the scan sweep.

    Returns a uint8 mask in FULL-FRAME coordinates, or None.
    """
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    lab = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    # background colour = median of a ring just outside the box (that is table)
    rx1, ry1 = max(0, x1 - SEG_RING), max(0, y1 - SEG_RING)
    rx2, ry2 = min(W, x2 + SEG_RING), min(H, y2 + SEG_RING)
    ring = np.ones((ry2 - ry1, rx2 - rx1), bool)
    ring[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
    ring_px = lab[ry1:ry2, rx1:rx2][ring]
    if ring_px.shape[0] < 40:
        return None
    bg = np.median(ring_px, axis=0)

    roi = lab[y1:y2, x1:x2]
    dist = np.linalg.norm(roi - bg, axis=2)
    dmax = float(dist.max())
    if dmax < 8.0:                       # object is the same colour as the table
        return None
    d8 = np.clip(dist / dmax * 255.0, 0, 255).astype(np.uint8)
    _thr, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    # keep the component that actually covers the box centre — Otsu on a
    # background gradient can light up a corner of the ROI instead of the object
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    cx_r, cy_r = (x2 - x1) // 2, (y2 - y1) // 2
    inner = lbl[max(0, cy_r - 3):cy_r + 4, max(0, cx_r - 3):cx_r + 4]
    inner = inner[inner > 0]
    if inner.size:
        best = int(np.bincount(inner).argmax())
    else:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < SEG_MIN_FRAC * (x2 - x1) * (y2 - y1):
        return None
    full = np.zeros((H, W), np.uint8)
    full[y1:y2, x1:x2] = np.where(lbl == best, 255, 0).astype(np.uint8)
    return full


def contact_points(geom, mask, bbox, T_base_cam, z_plane):
    """Base-frame footprint points where the object meets the table, plus the
    matching top-of-silhouette pixel for each of those columns.

    For a convex object standing on a plane, the LOWEST object pixel in each image
    column is the point where that column's surface touches the table — so those
    pixels, and only those, can be back-projected onto z=z_plane honestly. Columns
    whose bottom pixel sits on the box's own bottom edge are dropped: the object is
    cut off there and its real contact line is outside the frame.

    Returns (contact_pts (M,3), top_uv (M,2)) column-for-column, so the height
    solve can pair each roof pixel with the floor pixel DIRECTLY BELOW IT rather
    than with the footprint centre — which is what a flat elongated object needs
    (the highest pixel of a lying remote is its far END, not its top face).

    Failures report WHICH gate rejected, via ``reason``. They used to collapse into a
    single "no_contact_line", which is why a 100% failure rate went undiagnosed: five
    quite different causes — a mask too narrow, a box running off the bottom of the
    frame, rays too shallow to trust — all looked identical from the outside.
    """
    H, W = mask.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    sub = mask[y1:y2, x1:x2] > 0
    cols = np.where(sub.any(axis=0))[0]
    if cols.size < 6:
        return None, None, "mask_too_narrow"
    bottom = (sub.shape[0] - 1) - np.argmax(sub[::-1, :], axis=0)
    top = np.argmax(sub, axis=0)
    # Clipping means the IMAGE ran out, not the box: a bounding box touches the
    # silhouette on all four sides by construction, so testing the bottom pixel
    # against the box's own floor discards the entire true contact line. (It did
    # exactly that — a cup kept 28 of 147 columns, all of them up on its far rim.)
    if y2 >= H - 2:
        return None, None, "object_runs_off_frame_bottom"
    # drop the side-edge columns (see MAX_CONTACT_SLOPE) — keep only the stretch of
    # the lower boundary that is genuinely lying along the object's base
    b = bottom[cols].astype(np.float64)
    slope = np.gradient(b, cols.astype(np.float64))
    flat = np.abs(slope) <= MAX_CONTACT_SLOPE
    if flat.sum() >= 6:
        cols, b = cols[flat], b[flat]
    else:
        b = bottom[cols].astype(np.float64)
    uv_bot = np.stack([cols + x1 + 0.5, b + y1 + 0.5], axis=1)
    pts, ok = pixels_to_table(geom, uv_bot, T_base_cam, z_plane, MIN_TABLE_INCIDENCE)
    if pts.shape[0] < 6:
        # The rays were too shallow to intersect the plane usefully. This is the gate
        # most likely to be wrong on a given rig: it depends entirely on where the
        # camera is pointing, and near the horizon one pixel of segmentation noise
        # slides the intersection by centimetres.
        return None, None, f"rays_too_shallow ({pts.shape[0]}/{len(uv_bot)} kept)"
    uv_top = np.stack([cols + x1 + 0.5, top[cols] + y1 + 0.5], axis=1)[ok]
    return pts, uv_top, ""


def solve_height(geom, uv_top, xy, half_along, T_base_cam, z_plane):
    """Height of the object above the table, from the top of its silhouette.

    The highest silhouette pixel is the object's FAR TOP edge — looking down at a
    box you see its top face, and its skyline is the far rim. That rim stands
    vertically above the FAR edge of the footprint, so the ray is walked to the
    vertical line there, not to the one through the centre. Anchoring on the centre
    is what made a 2.2 cm remote measure 8.4 cm: a long object's far edge is half
    its length away, and the ray keeps climbing over that distance.

    half_along is the footprint's half-extent along the horizontal viewing
    direction — i.e. how far the far edge sits behind the centre.
    """
    uv = np.asarray(uv_top, np.float64).reshape(-1, 2)
    if uv.shape[0] == 0:
        return None
    o = np.asarray(T_base_cam[:3, 3], np.float64)
    xy = np.asarray(xy, np.float64)
    view = xy - o[:2]
    n = float(np.linalg.norm(view))
    if n < 1e-6:
        return None
    far_xy = xy + view / n * float(half_along)     # the skyline stands over here

    hs = []
    for k in np.argsort(uv[:, 1])[:max(3, uv.shape[0] // 10)]:   # the highest pixels
        d = np.array([(uv[k, 0] - geom.cx) / geom.fx, (uv[k, 1] - geom.cy) / geom.fy, 1.0], np.float64)
        d /= np.linalg.norm(d)
        d = np.asarray(T_base_cam[:3, :3], np.float64) @ d
        denom = float(d[0] ** 2 + d[1] ** 2)
        if denom < 1e-9:
            continue
        t = float((far_xy - o[:2]) @ d[:2] / denom)
        if not (0.03 < t < 1.50):
            continue
        h = float(o[2] + t * d[2] - z_plane)
        if -0.005 < h < 0.60:
            hs.append(h)
    return float(np.median(hs)) if len(hs) >= 3 else None


def classify_shape(w_m, d_m, h_m, prior):
    """Name the solid from its measured proportions, keeping the class prior when
    the label is one we know (a 'cup' stays a cylinder even if the footprint arc
    came out slightly rectangular)."""
    if prior in ("cylinder", "sphere"):
        return prior
    lo, hi = min(w_m, d_m), max(w_m, d_m)
    if hi < 1e-4:
        return prior
    if hi / max(lo, 1e-4) > 2.5:
        return "cuboid"                       # clearly elongated: pen, knife, book
    if h_m > 1.6 * hi:
        return "cylinder"                     # tall and square-ish on the table
    if abs(h_m - hi) / hi < 0.30:
        return "cube"
    return "cuboid"


class ObjectMeasurer:
    """Monocular size/orientation measurement against a known table plane.

    ``stats`` records why measurements were rejected, so "everything says (prior)" is
    diagnosable instead of a mystery. It is surfaced in /status as measure_stats.
    """

    def __init__(self, geometry, priors, *, reach_m=(0.08, 0.55), table_z=0.0):
        self.geom = geometry
        self.priors = priors
        self.reach_m = reach_m
        self.table_z = float(table_z)
        self.stats: dict[str, int] = {}

    def _fail(self, why):
        self.stats[why] = self.stats.get(why, 0) + 1
        return None

    def viewpoint_incidence(self, T_base_cam, uv=None) -> float:
        """How steeply a ray meets the table: sin(incidence), 0 = grazing.

        ``uv`` selects the ray; without it this uses the image centre, which is a
        property of the CAMERA rather than of any object. That distinction cost me a
        wrong diagnosis: a camera mounted above the table can have a level optical axis
        (centre incidence ~0) while every pixel in the lower half of the frame — where
        table objects actually appear — comes down at 0.26+. Judging a viewpoint by its
        centre ray rejected 173 perfectly measurable frames.
        """
        T = np.asarray(T_base_cam, dtype=np.float64)
        ray = (np.array([0.0, 0.0, 1.0]) if uv is None
               else np.array([(float(uv[0]) - self.geom.cx) / self.geom.fx,
                              (float(uv[1]) - self.geom.cy) / self.geom.fy, 1.0]))
        d = T[:3, :3] @ (ray / np.linalg.norm(ray))
        return float(-d[2])

    def can_measure(self, T_base_cam, bbox=None) -> bool:
        """Whether a footprint measurement is geometrically possible for THIS object.

        The solve needs the object's contact line back-projected onto the table, which
        needs rays that come DOWN onto it steeply enough — near the horizon, one pixel
        of segmentation noise slides the intersection by centimetres.

        The ray that matters is the one to the object's BASE, not the optical axis. A
        camera mounted above the table routinely has a near-level axis while the lower
        frame, where table objects sit, is comfortably steep. Testing the axis instead
        rejects viewpoints that would have measured perfectly well.

        Without a bbox this cannot answer honestly, so it says yes and lets the
        per-pixel gate inside contact_points do the real work — that gate was always
        there and is the one with the object's actual pixels in hand.
        """
        if bbox is None:
            return True
        x1, y1, x2, y2 = bbox
        base_uv = ((x1 + x2) / 2.0, y2)          # the object's bottom edge
        return self.viewpoint_incidence(T_base_cam, base_uv) >= MIN_TABLE_INCIDENCE

    def measure(self, rgb, bbox, T_base_cam, label, z_plane=None):
        """Measure an object's position, footprint, height and yaw from ONE frame.

        Returns a dict {xy, across_m, u_deg, w_m, d_m, h_m, yaw_deg, shape, measured,
        rng_m} or None if the silhouette solve did not produce something believable —
        the caller then falls back to the class prior.

        'yaw_deg' is the direction of the footprint's MAJOR axis in the base frame,
        normalised to [-90, 90) because a rectangle has no front.
        """
        if self.geom.fx <= 0:
            return None
        if not self.can_measure(T_base_cam, bbox):
            # Carry the actual number. Twice now a plausible story about WHY this
            # rejects has been wrong; the value it rejected on is not guessable from
            # outside, so it goes in the label.
            _i = self.viewpoint_incidence(T_base_cam,
                                          ((bbox[0] + bbox[2]) / 2.0, bbox[3]))
            return self._fail(f"viewpoint_too_shallow (inc {_i:+.2f} < {MIN_TABLE_INCIDENCE},"
                              f" base row {bbox[3]:.0f}, fx {self.geom.fx:.0f})")
        z_plane = self.table_z if z_plane is None else float(z_plane)
        mask = silhouette_mask(rgb, bbox)
        if mask is None:
            return self._fail("no_silhouette")
        pts, uv_top, why = contact_points(self.geom, mask, bbox, T_base_cam, z_plane)
        if pts is None:
            return self._fail(f"no_contact_line: {why}")

        # WHAT ONE VIEW CAN AND CANNOT SEE. Measured on the synthetic bench:
        #   * the extent ACROSS the sightline is recovered to about a millimetre
        #     (5.1 cm cube -> 5.1, 15 cm book -> 15.0, 4.5 cm remote -> 4.6,
        #     8 cm cup -> 7.9) — the silhouette's width is that extent, full stop.
        #   * the extent ALONG the sightline is NOT observable: the object's far side
        #     is behind the object. It over-reads on tall things (8 cm cup -> 11.5)
        #     and under-reads when the long axis points at the camera (22 cm book ->
        #     16.4, the rest of it hidden).
        # So this returns the across-view width as the measurement and hands it to the
        # map with the direction it was taken along; the map fuses the widths gathered
        # from the different bearings of a scan sweep into the actual footprint (see
        # _fit_rect_from_support). One view can only ever give one caliper reading.
        xy_pts = pts[:, :2].astype(np.float32)
        (cx_f, cy_f), (a, b), ang = cv2.minAreaRect(xy_pts)
        if max(a, b) < 0.004 or max(a, b) > MAX_TABLE_OBJ_M * 2.2:
            return self._fail("rect_size")

        prior = self.priors.meta(label)
        cam_xy = np.asarray(T_base_cam[:3, 3], np.float64)[:2]
        view = cam_xy - np.array([cx_f, cy_f])
        n_view = float(np.linalg.norm(view))
        if n_view < 1e-6:
            return self._fail("degenerate_view")
        v = view / n_view                       # unit vector back toward the camera
        u = np.array([-v[1], v[0]])             # ACROSS the sightline: the caliper axis

        proj_u = xy_pts.astype(np.float64) @ u
        across = float(proj_u.max() - proj_u.min())
        if not (0.004 < across < MAX_TABLE_OBJ_M):
            return self._fail("across_range")
        # SANITY-CHECK THE MEASUREMENT AGAINST WHAT THE CLASS IS. A 5.1 cm cube coming
        # out 0.9 cm or 9.3 cm wide is a broken silhouette, not a surprising cube, and
        # letting those through is what scattered one-off ghosts across the map. The
        # band is deliberately wide (the prior is only a guess) — it rejects nonsense,
        # not disagreement. Outside it, the caller falls back to the prior path.
        p_size = float(math.sqrt(max(prior["w_m"], 1e-3) * max(prior["d_m"], 1e-3)))
        if not (0.35 * p_size < across < 2.8 * p_size):
            return self._fail("size_vs_prior")

        # The contact arc is the NEAR side of the footprint, so its centroid sits about
        # half a depth too close to the camera. We do not know the depth yet, so push
        # back by half the across-width (a circle's worth) — the map's multi-bearing
        # average then cancels most of what this leaves behind.
        xy = np.array([proj_u.mean() * u[0], proj_u.mean() * u[1]], np.float64) \
            + v * float((xy_pts.astype(np.float64) @ v).mean()) - v * (0.5 * across)
        if not (self.reach_m[0] < float(np.hypot(*xy)) < self.reach_m[1]):
            return self._fail("out_of_workspace")
        rng = float(np.linalg.norm(cam_xy - xy))

        # HEIGHT NEEDS TO KNOW HOW FAR BACK THE OBJECT'S FAR EDGE IS, and using the
        # across-width for that is wrong for anything elongated. A pen pointing along
        # the sightline measures across=1.8cm (correctly - that is its width), but it
        # extends ~7cm away from us, so anchoring the roof ray 0.9cm behind centre walks
        # it nowhere near far enough and the height over-reads: measured, a flat pen
        # came out 4.5cm tall and got classified as a standing cylinder.
        #
        # We do not know the along-view extent from one view, so use the best estimate
        # available - the class prior's long axis - and never less than the across
        # width. Errors here stay second-order because the roof ray is steep.
        prior_long = max(prior["w_m"], prior["d_m"])
        half_along = 0.5 * max(across, min(prior_long, MAX_TABLE_OBJ_M))
        h_m = solve_height(self.geom, uv_top, xy, half_along, T_base_cam, z_plane)
        if h_m is None or not (0.002 < h_m < 0.60):
            h_m = float(prior["h_m"])
        # And do not let a single view claim an object is TALL when its footprint was
        # never determined: h > footprint reads as "standing up", which for a pen seen
        # end-on is exactly the wrong conclusion.
        if h_m > 1.5 * max(across, 1e-3) and prior_long > 2.2 * min(prior["w_m"], prior["d_m"]):
            h_m = float(prior["h_m"])

        # SANITY-CHECK THE SOLID AGAINST WHAT THE CLASS IS, and fall back to the prior
        # rather than publish a shape that cannot be true. From ONE viewpoint an
        # elongated object pointing along the sightline has no measurable length -
        # `across` is its WIDTH, correctly measured, and the 14 cm of a pen is simply
        # invisible. Left alone that produced "pen: 2.6 x 2.6 x 3.6 cm", i.e. a stubby
        # thing STANDING UP, which is worse than admitting we do not know: the map drew
        # a confident orientation for an object whose shape it had not determined.
        # Recovering the real footprint needs several bearings (the support fusion in
        # world2d_update), which a Scan sweep provides and idle sensing does not.
        if h_m > 2.5 * max(prior["h_m"], 0.003):
            return self._fail("height_vs_prior")
        self.stats["ok"] = self.stats.get("ok", 0) + 1
        yaw = math.degrees(math.atan2(u[1], u[0]))
        return {"xy": xy, "across_m": across, "u_deg": yaw, "h_m": float(h_m),
                "w_m": across, "d_m": max(across, min(prior["w_m"], prior["d_m"])),
                "yaw_deg": float(((yaw + 90.0) % 180.0) - 90.0),
                "shape": prior["shape"], "measured": True, "rng_m": rng}
