"""Tests for the bird's-eye object map.

The map's difficulty is association, and two specific failures shaped it: merging only
within a label left one pen sitting on the table as five separate "objects", and
merging across labels bluntly made the green cube vanish under a better-supported
"red cube". Both are pinned here, because a refactor that quietly reintroduces either
produces a map that looks fine and is wrong.

    python tests/test_object_map.py
    pytest tests/test_object_map.py
"""

from __future__ import annotations

import pathlib
import random
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mobility.slam.object_map import (  # noqa: E402
    ObjectMap, fit_rect_from_support, sup_bin, yaw_blend)

CUBE = dict(w_m=0.05, d_m=0.05, h_m=0.05, shape="cube", yaw=0.0)


def _map(**kw) -> ObjectMap:
    return ObjectMap(log=lambda _m: None, **kw)


def test_one_object_seen_twice_is_one_entry():
    m = _map()
    m.update("cup", [0.25, 0.0], **CUBE)
    m.update("cup", [0.26, 0.01], **CUBE)
    assert len(m) == 1
    # The position is blended toward the newer observation, not replaced or averaged
    # evenly, so the map converges instead of staying stuck on a first bad fix.
    xy = m.objs[1]["xy"]
    assert 0.25 < xy[0] < 0.26 and m.objs[1]["n"] == 2


def test_aliases_of_one_object_collapse():
    """An open vocabulary gives one pen five names; they are all the same pen."""
    m = _map()
    for label in ("pen", "knife", "scissors", "toothbrush", "remote"):
        m.update(label, [0.25, 0.02], **CUBE)
    assert len(m) == 1, f"expected one object, got {[o['label'] for o in m.objs.values()]}"
    entry = next(iter(m.objs.values()))
    assert entry["n"] == 5
    assert len(entry["aka"]) == 4, "the other names are kept as aliases"


def test_two_queried_labels_never_merge():
    """If both names are things we were asked to tell apart, they must stay apart —
    even sitting inside the merge radius, which a bad hand-eye can cause."""
    queried = {"red cube", "green cube"}
    m = _map(may_merge_labels=lambda a, b: a == b or not (a in queried and b in queried))
    m.update("red cube", [0.25, 0.00], **CUBE)
    m.update("green cube", [0.26, 0.01], **CUBE)   # 1.4 cm apart: well inside the radius
    assert len(m) == 2
    assert {o["label"] for o in m.objs.values()} == queried


def test_stale_entry_is_relabelled_not_swallowed():
    """A different label on an entry not confirmed under its own name is a takeover.

    Otherwise a stale, heavily-observed 'red cube' absorbs every observation of the
    green one now actually sitting there.
    """
    m = _map()
    tag = m.update("red cube", [0.25, 0.0], **CUBE)
    for _ in range(50):
        m.update("red cube", [0.25, 0.0], **CUBE)
    assert m.objs[tag]["n"] > 50

    m.objs[tag]["label_t"] = time.time() - 60.0     # not seen as itself for a minute
    m.update("green cube", [0.25, 0.0], **CUBE)
    assert m.objs[tag]["label"] == "green cube"
    assert m.objs[tag]["n"] == 1, "the old name's evidence is discarded, not inherited"


def test_merge_radius_floor_is_noise_not_object_size():
    """Object size may only WIDEN the radius. Narrowing it to a fraction of a small
    object's footprint is what produced a 16-ghost map: consecutive views of one cube
    land 5-15 cm apart, so every observation spawned a fresh tag."""
    m = _map(merge_m=0.14)
    small = {"w_m": 0.02, "d_m": 0.02}
    big = {"w_m": 0.40, "d_m": 0.30}
    assert m.merge_radius(small) == 0.14, "a small object must not shrink the radius"
    assert m.merge_radius(big) > 0.14, "a large object may widen it"
    assert m.merge_radius(big) <= 0.30, "but not without bound"


def test_ttl_prunes_and_clear_resets():
    m = _map(ttl_s=1.0)
    m.update("cup", [0.25, 0.0], **CUBE)
    assert m.prune() == []
    m.objs[1]["t"] = time.time() - 5.0
    assert m.prune() == [1] and len(m) == 0
    m.update("cup", [0.25, 0.0], **CUBE)
    m.clear()
    assert len(m) == 0
    assert m.update("cup", [0.25, 0.0], **CUBE) == 1, "tags restart after a clear"


def test_yaw_blend_is_circular_mod_180():
    """A footprint rectangle has no front, so yaw lives mod 180. Averaging -89 and +89
    naively gives 0 — a right angle away from both."""
    got = yaw_blend(-89.0, 89.0, 0.5)
    assert min(abs(got - 90.0), abs(got + 90.0)) < 1e-6, got
    assert abs(yaw_blend(10.0, 20.0, 0.5) - 15.0) < 1e-6


def test_support_ring_recovers_a_rectangle():
    """Support widths from several bearings should reconstruct the footprint that
    generated them — that is the whole premise of measuring shape by sweeping."""
    import math

    from mobility.slam.object_map import SUP_BINS

    w_true, d_true, yaw_true = 0.06, 0.16, 30.0
    sup = {}
    # Sample at the bin CENTRES, which is where the fit places each reading. Sampling
    # at the edges instead biases the recovered yaw by half a bin (7.5 deg).
    for k in range(SUP_BINS):
        theta = (k + 0.5) * 180.0 / SUP_BINS
        rel = math.radians(theta - yaw_true)
        sup[k] = d_true * abs(math.cos(rel)) + w_true * abs(math.sin(rel))
    fit = fit_rect_from_support(sup)
    assert fit is not None
    w, d, yaw = fit
    assert abs(w - w_true) < 0.005 and abs(d - d_true) < 0.005, (w, d)
    assert min(abs(yaw - yaw_true), abs(abs(yaw - yaw_true) - 180)) < 3.0, yaw


def test_support_ring_needs_enough_bearings():
    assert fit_rect_from_support({sup_bin(0): 0.05}) is None
    assert fit_rect_from_support({}) is None


def test_support_ring_rejects_impossible_sizes():
    """Disagreeing support widths mean a bad solve, not a two-metre object."""
    sup = {sup_bin(t): 5.0 for t in (0, 60, 120)}
    assert fit_rect_from_support(sup) is None


def test_map_is_stable_under_a_random_observation_stream():
    """Association must not depend on arrival order producing runaway tags."""
    m = _map()
    rng = random.Random(3)
    for _ in range(300):
        label = rng.choice(["cup", "bottle", "book"])
        # three clusters, jittered by the localization noise the real map sees
        cx, cy = {"cup": (0.20, -0.10), "bottle": (0.30, 0.05), "book": (0.38, -0.18)}[label]
        m.update(label, [cx + rng.gauss(0, 0.02), cy + rng.gauss(0, 0.02)],
                 w_m=0.06, d_m=0.06, h_m=0.08, shape="cube", yaw=0.0)
    assert len(m) <= 4, f"{len(m)} entries for 3 objects — association is spawning ghosts"
    assert len(m) >= 1


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
