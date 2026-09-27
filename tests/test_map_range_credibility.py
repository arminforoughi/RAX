"""A map that cannot say which of two objects is nearer cannot be driven on.

Two cubes on the table. The operator could see green nearer than red, and the
detector's own overlay agreed — it labelled them green 20cm, red 26cm. The map had
red at 35cm and green at 42cm: both far too distant, and their ORDER swapped.

The overlay and the map are different computations. The overlay is apparent size,
``fx * real_width / pixel_width`` — no hand-eye transform, no table plane, no arm pose.
The map preferred measure_object's silhouette solve, which depends on all three, and on
this rig degrades at the shallow angles the arm surveys from. It degrades toward a
confident wrong number rather than toward noise, and sense_2d folds every frame in, so
the map averages them.

Apparent size makes an honest referee precisely because it shares no machinery with the
thing it is judging.

    pytest tests/test_map_range_credibility.py
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "src", REPO / "examples" / "mission_server"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

CUBE_M = 0.0508


@pytest.fixture(scope="module")
def S():
    try:
        import mission_server
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"mission_server did not import: {exc}")
    return mission_server


def _box(width_px, x1=100.0, y1=100.0):
    return (x1, y1, x1 + width_px, y1 + width_px)


def test_apparent_range_is_the_textbook_pinhole_relation(S):
    got = S.apparent_range_m(_box(100.0), "red cube")
    assert got == pytest.approx(S.GEOM.fx * CUBE_M / 100.0, rel=1e-6)


def test_a_bigger_box_means_a_nearer_object(S):
    near = S.apparent_range_m(_box(200.0), "red cube")
    far = S.apparent_range_m(_box(50.0), "red cube")
    assert near < far


def test_a_degenerate_box_has_no_apparent_range(S):
    assert S.apparent_range_m(_box(1.0), "red cube") is None


# --- the gate -----------------------------------------------------------------------

def test_a_solve_that_agrees_with_the_picture_is_kept(S):
    ref = S.apparent_range_m(_box(100.0), "red cube")
    assert S.measured_range_is_credible({"rng_m": ref}, _box(100.0), "red cube")


@pytest.mark.parametrize("factor", [0.5, 0.6, 1.5, 2.0])
def test_a_solve_that_grossly_disagrees_is_dropped(S, factor):
    ref = S.apparent_range_m(_box(100.0), "red cube")
    assert not S.measured_range_is_credible({"rng_m": ref * factor}, _box(100.0), "red cube")


@pytest.mark.parametrize("factor", [0.8, 0.95, 1.0, 1.2, 1.35])
def test_ordinary_disagreement_is_tolerated(S, factor):
    """The gate is for gross failure, not for demanding the two agree exactly."""
    ref = S.apparent_range_m(_box(100.0), "red cube")
    assert S.measured_range_is_credible({"rng_m": ref * factor}, _box(100.0), "red cube")


def test_the_observed_failure_would_have_been_caught(S):
    """The live numbers: a 100px green box is ~26cm by apparent size; the map had 42cm."""
    ref = S.apparent_range_m(_box(100.0), "green cube")
    assert 0.24 < ref < 0.29, f"sanity: {ref:.3f}m"
    assert not S.measured_range_is_credible({"rng_m": 0.419}, _box(100.0), "green cube")


def test_no_solve_is_not_credible(S):
    assert not S.measured_range_is_credible(None, _box(100.0), "red cube")


def test_a_solve_with_no_range_is_left_alone(S):
    """Missing the field is not evidence against it — do not silently discard."""
    assert S.measured_range_is_credible({}, _box(100.0), "red cube")


def test_an_unknown_label_has_no_referee_so_the_solve_stands(S):
    """Without a class prior there is nothing to judge against; refusing everything
    would empty the map of exactly the objects priors do not cover."""
    assert S.measured_range_is_credible({"rng_m": 9.0}, _box(100.0), "no such thing xyzzy")


def test_a_nonsense_range_is_dropped(S):
    assert not S.measured_range_is_credible({"rng_m": 0.0}, _box(100.0), "red cube")
