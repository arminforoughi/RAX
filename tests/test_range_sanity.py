"""A range that disagrees with the object's own size is not a reach problem.

From the rig: the arm aborted with "r=51cm is out of reach (best IK 43mm)" and the
operator reasonably read that as the arm failing to extend. It was not. The IK had
reached 46.7cm, which is this arm's real limit — measured independently at 47cm — so
the arm was already doing everything it could.

The target was the fault. The cube's own pixels put it at ~20cm; the map had it at
42cm and the mission asked for 51cm. The map gave the game away: it recorded the
5.08cm cube as 10.4cm across. Range and apparent size are the SAME measurement seen
from opposite ends — infer twice the range and you infer twice the size — so a doubled
size is a doubled range wearing a disguise, and "out of reach" is the symptom.

    pytest tests/test_range_sanity.py
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


def _seed_map(S, label, w_m, d_m, n=50):
    """Put one entry in the live map with a known measured footprint."""
    S.WORLD.clear()
    with S.w2d_lock:
        S.WORLD.objs[1] = {"label": label, "xy": (0.30, 0.05), "w_m": w_m, "d_m": d_m,
                           "h_m": 0.05, "n": n, "t": 0.0}


def test_the_measured_size_is_read_back_from_the_map(S):
    _seed_map(S, "red cube", 0.104, 0.104)
    got = S._mapped_size_m("red")
    assert got is not None and abs(got - 0.104) < 1e-6, got


def test_a_doubled_size_is_what_a_doubled_range_looks_like(S):
    """The live case: a 5.08cm cube recorded as 10.4cm across."""
    _seed_map(S, "red cube", 0.104, 0.104)
    measured = S._mapped_size_m("red")
    prior = S.PRIORS.size_m("red cube")
    ratio = measured / prior
    assert 1.9 < ratio < 2.2, f"expected ~2x, got {ratio:.2f}"
    assert not (0.6 < ratio < 1.7), "this must fall outside the trusted band"


def test_a_correctly_sized_object_is_inside_the_trusted_band(S):
    _seed_map(S, "red cube", CUBE_M, CUBE_M)
    ratio = S._mapped_size_m("red") / S.PRIORS.size_m("red cube")
    assert 0.6 < ratio < 1.7, f"a correctly measured cube must be trusted, got {ratio:.2f}"


def test_an_unmeasured_entry_reports_nothing_rather_than_guessing(S):
    S.WORLD.clear()
    with S.w2d_lock:
        S.WORLD.objs[1] = {"label": "red cube", "xy": (0.3, 0.0), "w_m": 0.0,
                           "d_m": 0.0, "h_m": 0.0, "n": 3, "t": 0.0}
    assert S._mapped_size_m("red") is None


def test_a_label_that_is_not_on_the_map_reports_nothing(S):
    S.WORLD.clear()
    assert S._mapped_size_m("red") is None


def test_the_best_supported_entry_wins(S):
    """Several entries for one object: trust the one with the most confirmations."""
    S.WORLD.clear()
    with S.w2d_lock:
        S.WORLD.objs[1] = {"label": "red cube", "xy": (0.3, 0.0), "w_m": 0.30,
                           "d_m": 0.30, "h_m": 0.05, "n": 2, "t": 0.0}
        S.WORLD.objs[2] = {"label": "red cube", "xy": (0.3, 0.0), "w_m": CUBE_M,
                           "d_m": CUBE_M, "h_m": 0.05, "n": 90, "t": 0.0}
    assert abs(S._mapped_size_m("red") - CUBE_M) < 1e-6
