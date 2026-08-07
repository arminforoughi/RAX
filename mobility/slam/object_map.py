"""``ObjectMap`` — a bird's-eye map of the objects on the table.

Not SLAM, and deliberately so: the camera rides the arm, so its pose is never
*estimated*, it is computed exactly from the servo angles. FK from encoders is exact
odometry, and mapping with known poses is not SLAM any more — it is pure mapping. So
there is no pose graph, no loop closure and no drift model here, only the fusion of
many observations of the same object into one entry.

The hard part turned out to be association, and two failures shaped this code:

**Merging only within a label was the bug.** An open vocabulary gives one object
several names — a single pen fired as pen, knife, scissors, toothbrush AND remote, so
it became five "objects" that could never combine no matter how close they sat
(measured: five entries within 6-9 cm). Two detections at the same place ARE the same
thing; the label is the least reliable part of an observation. So position decides and
the best-supported name wins.

**But merging across labels bluntly is also wrong.** It merged the red and green
cubes, because a bad hand-eye put them inside the merge radius, and the green cube
vanished under a better-supported "red cube". The distinction: if BOTH names are
things we were explicitly asked to look for, they are meant to be told apart and must
never merge. If only one is in the query, the other is the detector reaching for a
different word for the same thing.

Entries are per-INSTANCE, not per-label: two cups on the table are two entries.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from perception.object_priors import MAX_TABLE_OBJ_M

__all__ = ["ObjectMap", "MapEntry", "sup_bin", "fit_rect_from_support",
           "yaw_blend", "SUP_BINS", "SUP_MIN_BINS"]

# Detections of the same object within this distance are the same object. Tightened
# because objects were being merged too aggressively; increase it if one object grows
# duplicate ghosts.
DEFAULT_MERGE_M = 0.14
MAX_MERGE_M = 0.30

# An entry not re-observed for this long is STALE: the object was moved or taken away,
# and the map should stop asserting it is there. Without this the map keeps reporting a
# scene that no longer exists — and worse, a stale high-n entry sits in the way and
# swallows observations of whatever is now at that spot.
DEFAULT_TTL_S = 90.0

# A different label landing on an entry not confirmed under its OWN name for this long
# takes it over. Fresh disagreement is genuine ambiguity and the better-supported name
# should win; stale disagreement means the object changed.
LABEL_TAKEOVER_S = 4.0

# Weight given to a fresh observation, so the map converges rather than staying stuck
# on an early bad localization.
XY_BLEND_NEW = 0.45
SIZE_BLEND_NEW = 0.3

# --- support ring ---------------------------------------------------------------
# Each view measures the footprint's width along ONE direction (its across-view axis).
# That is the footprint's SUPPORT WIDTH along that direction, and a convex shape is
# determined by its support widths — so a scan sweep, seeing each object from a spread
# of bearings, measures the whole footprint between them.
SUP_BINS = 12              # direction bins over 180 deg (15 deg each)
SUP_MIN_BINS = 3           # fit a rectangle only once this many bearings are in
SUP_BLEND_NEW = 0.4


def sup_bin(u_deg) -> int:
    return int(((float(u_deg) % 180.0) / 180.0) * SUP_BINS) % SUP_BINS


def fit_rect_from_support(sup) -> tuple[float, float, float] | None:
    """Least-squares rectangle through the accumulated support widths.

    A rectangle with half-sides (a, b) at yaw phi has support width::

        s(theta) = 2a|cos(theta - phi)| + 2b|sin(theta - phi)|

    Sweep phi over 1 deg steps; for each, a and b fall out of a 2x2 linear solve. Keep
    the phi with the smallest residual. Returns ``(w_m, d_m, yaw_deg)`` with d_m the
    long side and yaw along it, or None when too few bearings have been seen.
    """
    obs = [(math.radians((k + 0.5) * 180.0 / SUP_BINS), s)
           for k, s in sorted(sup.items()) if s > 0]
    if len(obs) < SUP_MIN_BINS:
        return None
    th = np.array([o[0] for o in obs])
    s = np.array([o[1] for o in obs])
    best = None
    for phi_deg in range(0, 180):
        phi = math.radians(phi_deg)
        A = np.stack([np.abs(np.cos(th - phi)), np.abs(np.sin(th - phi))], axis=1)
        try:
            x, *_ = np.linalg.lstsq(A, s, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if x[0] <= 0 or x[1] <= 0:
            continue
        r = float(np.linalg.norm(A @ x - s))
        if best is None or r < best[0]:
            best = (r, float(x[0]), float(x[1]), phi_deg)
    if best is None:
        return None
    _r, e1, e2, phi_deg = best
    # e1 is the extent along phi, e2 across it; report the long side as d/yaw
    if e1 >= e2:
        d_m, w_m, yaw = e1, e2, phi_deg
    else:
        d_m, w_m, yaw = e2, e1, phi_deg + 90.0
    # A footprint outside tabletop scale means the support widths disagreed, not that
    # the object is that big — reject rather than write a confident wrong size.
    if not (0.004 < w_m < MAX_TABLE_OBJ_M and 0.004 < d_m < MAX_TABLE_OBJ_M):
        return None
    return float(w_m), float(d_m), float(((yaw + 90.0) % 180.0) - 90.0)


def yaw_blend(y_old, y_new, w_new) -> float:
    """Circular mean of two axis angles.

    A footprint rectangle has no front, so yaw lives mod 180 deg — averaging -89 and
    +89 naively gives 0, which is a right angle away from both. Average the doubled
    angle instead.
    """
    a = math.radians(2.0 * float(y_old))
    b = math.radians(2.0 * float(y_new))
    s = (1 - w_new) * math.sin(a) + w_new * math.sin(b)
    c = (1 - w_new) * math.cos(a) + w_new * math.cos(b)
    if abs(s) < 1e-9 and abs(c) < 1e-9:
        return float(y_new)
    return float(((math.degrees(math.atan2(s, c)) / 2.0 + 90.0) % 180.0) - 90.0)


#: One mapped object. A plain dict, because the HTTP layer serializes it straight to
#: the UI and several keys are set opportunistically by whichever subsystem saw it.
MapEntry = dict


class ObjectMap:
    """Thread-safe bird's-eye map of table objects, keyed by integer tag."""

    def __init__(self, *, merge_m: float = DEFAULT_MERGE_M, ttl_s: float = DEFAULT_TTL_S,
                 may_merge_labels=None, classify_shape=None, prior_shape=None, log=None):
        self.merge_m = float(merge_m)
        self.ttl_s = float(ttl_s)
        self.lock = threading.RLock()
        self._objs: dict[int, MapEntry] = {}
        self._next = 1
        # Injected because they depend on things the map does not own: the active
        # query, and how a measured footprint maps to a shape name.
        self._may_merge = may_merge_labels or (lambda a, b: True)
        self._classify = classify_shape
        self._prior_shape = prior_shape or (lambda label: "cube")
        self._log = log or (lambda msg: None)

    # --- introspection -----------------------------------------------------------
    def __len__(self) -> int:
        with self.lock:
            return len(self._objs)

    @property
    def objs(self) -> dict[int, MapEntry]:
        """The live dict. Hold :attr:`lock` while iterating."""
        return self._objs

    def get(self, tag) -> MapEntry | None:
        with self.lock:
            return self._objs.get(int(tag))

    def clear(self) -> None:
        with self.lock:
            self._objs.clear()
            self._next = 1

    def merge_radius(self, a, b=None) -> float:
        """How close two detections must be to count as one object.

        THE FLOOR IS SET BY LOCALIZATION NOISE, NOT BY OBJECT SIZE. Scaling this down
        to a fraction of the object's own footprint (3.5 cm for a cube) produced a
        16-ghost map: consecutive views of ONE cube land 5-15 cm apart, so every
        observation spawned a fresh tag. Object size may only ever WIDEN the radius —
        a laptop needs more than 14 cm — never narrow it below what the jitter demands.
        """
        r = self.merge_m
        for o in (a, b):
            if o is not None:
                r = max(r, 0.55 * max(o["w_m"], o["d_m"]))
        return float(np.clip(r, self.merge_m, MAX_MERGE_M))

    # --- the write path ----------------------------------------------------------
    def update(self, label, xy, *, stereo=None, w_m, d_m, h_m, shape, yaw,
               measured=False, across_m=None, u_deg=None) -> int:
        """Fold one observation of one object into the map; returns its tag.

        ``across_m`` / ``u_deg`` are one caliper reading of the footprint — its width
        along the across-view direction — which is the only footprint fact a single
        view actually establishes. They accumulate per direction bin, and once enough
        bearings are in, the footprint and yaw are re-fitted from all of them.
        """
        xy = np.asarray(xy, dtype=np.float64)
        obs = {"w_m": float(w_m), "d_m": float(d_m)}
        now = time.time()
        with self.lock:
            # Match on POSITION, not on the label: the same object arrives under
            # different names from an open vocabulary, and a new name must land on the
            # existing entry rather than spawn a rival ghost beside it.
            best, bd = None, None
            for t, o in self._objs.items():
                if not self._may_merge(o["label"], label):
                    continue
                dist = float(np.hypot(*(o["xy"] - xy)))
                if dist < self.merge_radius(o, obs) and (bd is None or dist < bd):
                    best, bd = t, dist

            if best is None:
                best = self._next
                self._next += 1
                self._objs[best] = {
                    "label": label, "xy": xy,
                    "w_m": float(w_m), "d_m": float(d_m), "h_m": float(h_m),
                    "shape": str(shape), "yaw": float(yaw),
                    "measured": bool(measured), "sup": {},
                    "n": 1, "stereo": stereo, "t": now,
                }
            else:
                self._absorb(best, label, xy, stereo, w_m, d_m, h_m, shape, yaw,
                             measured, now)

            o = self._objs[best]
            if across_m is not None and u_deg is not None:
                self._add_support(o, label, across_m, u_deg)
            self.consolidate()
        return best

    def _absorb(self, tag, label, xy, stereo, w_m, d_m, h_m, shape, yaw, measured, now):
        o = self._objs[tag]
        # A DIFFERENT label landing on a STALE entry means the thing at this spot
        # changed — the old name is not evidence any more, however many times it was
        # seen. Take the position over outright rather than let a stale n=1793
        # "red cube" swallow every observation of the green one now sitting there.
        #
        # Compare against when this entry was last confirmed UNDER ITS OWN NAME, not
        # when it was last touched: touch-time never goes stale, because every incoming
        # observation refreshed the leftover entry it was being merged into, so the
        # relabel that would have fixed it could never fire.
        seen_as_itself = o.get("label_t", o["t"])
        if o["label"] != label and (now - seen_as_itself) > LABEL_TAKEOVER_S:
            self._log(f"map: {o['label']}#{tag} not confirmed as '{o['label']}' for "
                      f"{now - seen_as_itself:.0f}s — relabelling as '{label}'")
            o["label"], o["aka"], o["n"] = label, [], 0
            o["measured"] = False
        if o["label"] == label:
            o["label_t"] = now

        o["xy"] = (1.0 - XY_BLEND_NEW) * o["xy"] + XY_BLEND_NEW * xy
        if measured and not o.get("measured"):
            o["w_m"], o["d_m"], o["h_m"] = float(w_m), float(d_m), float(h_m)
            o["yaw"], o["shape"], o["measured"] = float(yaw), str(shape), True
        elif measured or not o.get("measured"):
            o["h_m"] = (1 - SIZE_BLEND_NEW) * o["h_m"] + SIZE_BLEND_NEW * float(h_m)
            if not o["sup"]:            # no caliper readings yet — keep blending
                o["w_m"] = (1 - SIZE_BLEND_NEW) * o["w_m"] + SIZE_BLEND_NEW * float(w_m)
                o["d_m"] = (1 - SIZE_BLEND_NEW) * o["d_m"] + SIZE_BLEND_NEW * float(d_m)
                o["yaw"] = yaw_blend(o["yaw"], yaw, SIZE_BLEND_NEW)
        if o["label"] != label:
            o["aka"] = sorted(set(o.get("aka", ())) | {label} - {o["label"]})
        o["n"] += 1
        o["stereo"] = stereo
        o["t"] = now

    def _add_support(self, o, label, across_m, u_deg):
        k = sup_bin(u_deg)
        o["sup"][k] = ((1 - SUP_BLEND_NEW) * o["sup"][k] + SUP_BLEND_NEW * float(across_m)
                       if k in o["sup"] else float(across_m))
        fit = fit_rect_from_support(o["sup"])
        if fit is not None:
            o["w_m"], o["d_m"], o["yaw"] = fit
            if self._classify is not None:
                o["shape"] = self._classify(o["w_m"], o["d_m"], o["h_m"],
                                            self._prior_shape(label))

    def consolidate(self) -> None:
        """Collapse entries that are really ONE physical object. Caller holds the lock.

        Weighted by observation count, so the better-supported entry keeps its name and
        the other's evidence is folded in rather than discarded.
        """
        objs = self._objs
        changed = True
        while changed:
            changed = False
            items = list(objs.items())
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    ta, a = items[i]
                    tb, b = items[j]
                    if ta not in objs or tb not in objs:
                        continue
                    if not self._may_merge(a["label"], b["label"]):
                        continue
                    if float(np.hypot(*(a["xy"] - b["xy"]))) < self.merge_radius(a, b):
                        self._fuse(ta, tb)
                        changed = True
                        break
                if changed:
                    break

    def _fuse(self, ta, tb) -> None:
        objs = self._objs
        a, b = objs[ta], objs[tb]
        keep, drop = (ta, tb) if a["n"] >= b["n"] else (tb, ta)
        ko, do = objs[keep], objs[drop]
        wsum = ko["n"] + do["n"]
        ko["xy"] = (ko["n"] * ko["xy"] + do["n"] * do["xy"]) / wsum
        for k in ("w_m", "d_m", "h_m"):
            ko[k] = (ko["n"] * ko[k] + do["n"] * do[k]) / wsum
        ko["yaw"] = yaw_blend(ko["yaw"], do["yaw"], do["n"] / wsum)
        # a measurement beats a prior, whichever entry it came from
        if do.get("measured") and not ko.get("measured"):
            ko["shape"], ko["measured"] = do["shape"], True
        # two ghosts of one object each hold caliper readings from the bearings they
        # were seen from — pooling them is exactly the extra evidence the fit wants
        for bk, bv in do.get("sup", {}).items():
            ko["sup"][bk] = 0.5 * (ko["sup"][bk] + bv) if bk in ko["sup"] else bv
        fit = fit_rect_from_support(ko["sup"])
        if fit is not None:
            ko["w_m"], ko["d_m"], ko["yaw"] = fit
        if ko["label"] != do["label"]:
            alt = set(ko.get("aka", ())) | set(do.get("aka", ())) | {do["label"]}
            ko["aka"] = sorted(alt - {ko["label"]})
        ko["n"] = wsum
        ko["t"] = max(ko["t"], do["t"])
        del objs[drop]

    # --- the read path -----------------------------------------------------------
    def prune(self, now=None) -> list[int]:
        """Drop entries past their TTL. Returns the tags removed."""
        now = time.time() if now is None else now
        with self.lock:
            stale = [t for t, o in self._objs.items() if now - o["t"] > self.ttl_s]
            for t in stale:
                del self._objs[t]
        return stale
