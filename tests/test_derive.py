"""Tests for parameters derived from geometry and measurement rather than dialled.

Each function here replaces a constant somebody tuned by hand on one object at one
range. So the tests do two things: check the derivation agrees with the hand-tuned
value *in the situation it was tuned for* — which is the evidence the old number was
sensible and the new one reproduces it — and check it does the right thing in the
situations the old number silently got wrong.

    python tests/test_derive.py
    pytest tests/test_derive.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manipulation.approach.derive import (  # noqa: E402
    align_tolerance_px, apparent_width_px, grasp_height, horizontal_fov_deg,
    hover_height, right_trim_for_visibility)
from mobility.slam.object_map import ObjectMap  # noqa: E402
from perception.camera_geometry import (  # noqa: E402
    CameraGeometry, FixedCamera, intrinsics_from_dict, parse_tf)

CUBE_H = 0.0508          # the 5 cm cube every constant was tuned on
HAND_GRASP_Z = 0.015     # PICK_GRASP_Z
HAND_ALIGN_PX = 40.0     # ALIGN_TOL_PX
HAND_TRIM_M = 0.05       # TARGET_RIGHT_TRIM_M


def _geom():
    return CameraGeometry(
        intrinsics_from_dict({"fx": 517.0, "fy": 517.0, "cx": 320.0, "cy": 240.0}, 640, 480),
        FixedCamera(parse_tf("0,0,0.60,3.14159265,0,0")))


# --- grasp height ------------------------------------------------------------------
def test_grasp_height_reproduces_the_hand_tuned_value_for_the_cube():
    """EXACTLY reproduces it, because the grasp fraction is recovered from that very
    pair. This is what makes the derivation safe to switch on: the object the constant
    was tuned on grasps identically, and every other object inherits the rule instead
    of the number."""
    z = grasp_height(CUBE_H)
    assert abs(z - HAND_GRASP_Z) < 1e-9, f"derived {z*100:.3f}cm vs hand {HAND_GRASP_Z*100:.3f}cm"


def test_grasp_height_scales_with_the_object():
    """The thing a fixed 1.5 cm cannot do. A tall object must not be gripped at its
    very bottom, and a flat one must not be gripped above its top."""
    flat = grasp_height(0.012)      # a remote
    cube = grasp_height(CUBE_H)
    tall = grasp_height(0.23)       # a bottle
    assert flat < cube < tall
    assert flat < 0.012, "gripping above a 1.2 cm object's top would close on air"
    assert tall > HAND_GRASP_Z * 3, "a bottle gripped at 1.5 cm is held at its very base"


def test_grasp_height_stays_inside_the_object():
    """Whatever the height, the grip point must be above the table and below the top."""
    for h in (0.004, 0.01, 0.02, 0.05, 0.1, 0.25, 0.4):
        z = grasp_height(h)
        assert 0.0 < z < h, f"h={h}: grip at {z}"


def test_grasp_height_handles_objects_too_thin_for_clearance():
    """Below twice the clearance there is no room for both margins; gripping at the
    middle is the honest answer rather than a clearance the geometry cannot give."""
    h = 0.006
    assert abs(grasp_height(h) - h / 2) < 1e-9


def test_grasp_height_follows_the_table_plane():
    """The table is measured and tilted; grasp height must ride on it, not on z=0."""
    assert abs(grasp_height(CUBE_H, table_z_m=-0.02) - (grasp_height(CUBE_H) - 0.02)) < 1e-9


def test_hover_clears_the_top_of_a_tall_object():
    """Hovering at grasp+standoff would drive a tall object's approach through its own
    body; the hover must clear its top."""
    h = 0.23
    assert hover_height(h, standoff_m=0.05) >= h + 0.05 - 1e-9
    # for a short object the standoff above the grip point is what dominates
    assert hover_height(0.02, standoff_m=0.05) >= 0.02 + 0.05 - 1e-9


# --- align tolerance ----------------------------------------------------------------
def test_align_tolerance_matches_the_hand_value_at_the_tuned_range():
    """40 px was tuned with the cube at roughly 20 cm; check the derivation agrees."""
    tol = align_tolerance_px(_geom(), CUBE_H, 0.20)
    assert abs(tol - HAND_ALIGN_PX) < 15.0, f"derived {tol:.0f}px vs hand {HAND_ALIGN_PX:.0f}px"


def test_align_tolerance_tightens_as_the_object_gets_further():
    """The failure a fixed pixel count hides: 40 px is a third of a near object and a
    whole far one, so a fixed tolerance demands different physical accuracy at
    different ranges."""
    g = _geom()
    near = align_tolerance_px(g, CUBE_H, 0.15)
    far = align_tolerance_px(g, CUBE_H, 0.45)
    assert near > far, f"near {near:.0f}px should exceed far {far:.0f}px"


def test_align_tolerance_is_a_constant_fraction_of_apparent_size():
    g = _geom()
    for r in (0.18, 0.25, 0.35):
        tol = align_tolerance_px(g, CUBE_H, r)
        frac = tol / apparent_width_px(g, CUBE_H, r)
        assert abs(frac - 0.33) < 0.02, f"r={r}: fraction {frac:.2f}"


def test_align_tolerance_is_bounded_both_ways():
    g = _geom()
    assert align_tolerance_px(g, 0.002, 0.5) >= 12.0, "a tiny object must not demand sub-pixel"
    assert align_tolerance_px(g, 0.40, 0.10) <= 60.0, "a big object must not licence a huge miss"


# --- visibility trim ----------------------------------------------------------------
def test_right_trim_is_near_the_hand_value_at_close_approach():
    trim = right_trim_for_visibility(_geom(), CUBE_H, 0.12)
    assert 0.01 < trim < 0.12
    assert abs(trim - HAND_TRIM_M) < 0.05, f"derived {trim*100:.1f}cm vs hand {HAND_TRIM_M*100:.0f}cm"


def test_right_trim_shrinks_as_the_gripper_gets_closer():
    """Closer means less of the table is in frame, so there is less room to sit aside.
    A fixed trim slides the object out of view exactly when centring needs it."""
    g = _geom()
    assert right_trim_for_visibility(g, CUBE_H, 0.08) < right_trim_for_visibility(g, CUBE_H, 0.20)


def test_right_trim_declines_to_guess_without_a_frame_width():
    """Better to sit on the line than invent an offset from an unknown field of view."""
    g = CameraGeometry(intrinsics_from_dict({"fx": 517., "fy": 517., "cx": 320., "cy": 240.}, 0, 0),
                       FixedCamera(np.eye(4)))
    assert right_trim_for_visibility(g, CUBE_H, 0.12) == 0.0
    assert horizontal_fov_deg(g) == 0.0


def test_horizontal_fov_is_sane():
    fov = horizontal_fov_deg(_geom())
    assert 50.0 < fov < 80.0, f"{fov:.1f}deg for a 640px/517fx camera"


# --- merge radius from measured scatter ---------------------------------------------
def test_map_measures_its_own_localization_scatter():
    """The map is the right instrument: every update's residual against the running
    estimate is one sample of localization noise, since the object did not move."""
    m = ObjectMap(log=lambda _m: None)
    rng = np.random.default_rng(4)
    true_xy = np.array([0.28, 0.05])
    sigma = 0.03
    assert m.observation_scatter() is None, "must not report a figure before it has data"
    for _ in range(120):
        m.update("cup", true_xy + rng.normal(0, sigma, 2), w_m=0.08, d_m=0.08,
                 h_m=0.10, shape="cylinder", yaw=0.0)
    got = m.observation_scatter()
    assert got is not None
    # residuals are measured against a running mean, so they read a bit under the raw
    # per-sample sigma; the point is that it lands in the right decade, not exactly.
    assert 0.3 * sigma < got < 2.0 * sigma, f"scatter {got:.4f} for sigma {sigma}"


def test_suggested_merge_radius_covers_the_measured_noise():
    """The floor must be set by noise, not object size — that lesson cost a 16-ghost
    map. So the suggestion has to exceed the scatter it measured."""
    m = ObjectMap(log=lambda _m: None)
    rng = np.random.default_rng(9)
    for _ in range(120):
        m.update("cup", np.array([0.28, 0.05]) + rng.normal(0, 0.02, 2),
                 w_m=0.08, d_m=0.08, h_m=0.10, shape="cylinder", yaw=0.0)
    s = m.observation_scatter()
    r = m.suggested_merge_radius(k=3.0)
    assert r is not None and r > s, f"radius {r} does not cover scatter {s}"
    assert 0.02 <= r <= 0.30


def test_suggestion_is_returned_not_applied():
    """Changing association on a running robot is a decision, not a side effect."""
    m = ObjectMap(merge_m=0.14, log=lambda _m: None)
    rng = np.random.default_rng(1)
    for _ in range(120):
        m.update("cup", np.array([0.28, 0.05]) + rng.normal(0, 0.005, 2),
                 w_m=0.08, d_m=0.08, h_m=0.10, shape="cylinder", yaw=0.0)
    assert m.suggested_merge_radius() is not None
    assert m.merge_m == 0.14, "the live radius must not move on its own"


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
