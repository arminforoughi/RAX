"""An object you successfully picked must not vanish from the map forever.

Observed: both cubes plainly in the camera view and boxed by the detector, both listed
in the 2D map, and the task refusing with

    step 1: no 'red cube' on the 2D map after a scan - check it is on the table and in view

The map was right and the lookup was wrong. `picked` is SET on a successful grasp with
a loose label test ("red" marks the entry labelled "red cube", because _target_finder
hands the mission a bare colour) and was RETIRED with an exact one
(`o["label"] == carry_label`, i.e. "red cube" == "red", never true). So the flag went on
and never came off, and _find_map_tag skips picked entries.

Every object the arm had ever picked became permanently invisible to tasks, and the
only cure was clearing the map.

    pytest tests/test_map_picked_flag.py
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "src", REPO / "examples" / "mission_server"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


@pytest.fixture(scope="module")
def S():
    try:
        import mission_server
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"mission_server did not import: {exc}")
    return mission_server


@pytest.fixture(autouse=True)
def clean_map(S):
    S.WORLD.clear()
    yield
    S.WORLD.clear()


def _put(S, tag, label, n=100, picked=False):
    with S.w2d_lock:
        S.WORLD.objs[tag] = {"label": label, "xy": (0.30, 0.05), "w_m": 0.05,
                             "d_m": 0.05, "h_m": 0.05, "n": n, "t": 0.0,
                             "picked": picked}


# --- the matcher, loose in both directions ------------------------------------------

@pytest.mark.parametrize("want,label", [
    ("red", "red cube"),          # what the mission carries vs what the detector wrote
    ("red cube", "red"),          # and the reverse
    ("red cube", "red cube"),
    ("green", "green cube"),
])
def test_labels_that_mean_the_same_object_match(S, want, label):
    assert S.map_label_matches(want, label)


@pytest.mark.parametrize("want,label", [
    ("red", "green cube"), ("red cube", "blue cube"), ("", "red cube"), ("red", ""),
])
def test_labels_that_mean_different_objects_do_not(S, want, label):
    assert not S.map_label_matches(want, label)


# --- the defect itself ---------------------------------------------------------------

def test_a_picked_entry_is_retired_by_the_bare_colour_the_mission_carries(S):
    """The exact mismatch: carry_label is "red", the map says "red cube"."""
    _put(S, 1, "red cube", picked=True)
    assert S.clear_picked_flags("red") == 1
    with S.w2d_lock:
        assert not S.WORLD.objs[1]["picked"]


def test_a_picked_object_is_findable_again_once_the_jaws_are_empty(S):
    """End to end: this is the state the operator was stuck in."""
    _put(S, 1, "red cube", picked=True)
    assert S._find_map_tag("red cube") is None, "precondition: picked entries are skipped"
    S._set_carry(False)
    assert S._find_map_tag("red cube") == 1, "an empty gripper means nothing is picked"


def test_clearing_is_scoped_to_the_label_when_one_is_given(S):
    _put(S, 1, "red cube", picked=True)
    _put(S, 2, "green cube", picked=True)
    assert S.clear_picked_flags("red") == 1
    with S.w2d_lock:
        assert not S.WORLD.objs[1]["picked"] and S.WORLD.objs[2]["picked"]


def test_clearing_with_no_label_clears_everything(S):
    _put(S, 1, "red cube", picked=True)
    _put(S, 2, "green cube", picked=True)
    assert S.clear_picked_flags() == 2


def test_an_unpicked_map_is_left_alone(S):
    _put(S, 1, "red cube")
    assert S.clear_picked_flags() == 0
    assert S._find_map_tag("red") == 1


def test_a_missed_pick_does_not_poison_the_map(S):
    """A grasp that closed on air ends in _set_carry(False) like any other."""
    _put(S, 1, "red cube", picked=True)
    S._set_carry(False)
    assert S._find_map_tag("red") == 1


def test_the_best_supported_entry_still_wins_among_matches(S):
    _put(S, 1, "red cube", n=5)
    _put(S, 2, "red cube", n=900)
    assert S._find_map_tag("red") == 2


# --- a late verdict must still be able to stop the place ----------------------------
# The grasp check runs off-thread. On a real run it landed FIVE SECONDS after place_at
# had already read carry["held"] at entry:
#
#   17:34:55  CONTACT (idle current 665)          <- the current sensor, wrong
#   17:34:57  place red on green cube #2          <- committed, held read once
#   17:35:02  camera: "the red object is clearly visible on the surface below the
#             gripper"                            <- right, and too late
#   17:35:12  [DONE] placed red on green cube #2  <- nothing was ever in the jaws
#
# The map then recorded a 10.2cm stack. The cubes were side by side on the table.

def test_place_at_re_reads_the_carry_before_the_descent(S):
    """Pins the second check. Without it a late 'empty' verdict changes nothing."""
    import inspect
    src = inspect.getsource(S.place_at)
    head, _, tail = src.partition("# ---- 3b.")
    assert tail, "place_at must re-check the carry before committing to the descent"
    # and it must come BEFORE the descent, not after
    assert "# ---- 4. descend" in tail
    assert 'carry["held"]' in tail.split("# ---- 4. descend")[0]


def test_the_entry_check_is_still_there_too(S):
    """The late check supplements the early one; it does not replace it."""
    import inspect
    head = inspect.getsource(S.place_at).partition("# ---- 3b.")[0]
    assert "nothing in the jaws" in head


# --- the ghost must not win on popularity -------------------------------------------
# One physical cube, two map entries, because observations from different arm poses get
# rotated to different places by a wrong hand-eye transform — far enough apart not to
# merge. Measured on a map cleared seconds earlier and rebuilt from a single pose:
#
#   'green cube'  base r=37.1cm   18.6cm from the tip   n=40   <- the real one
#   'green cube'  base r=52.0cm   34.3cm from the tip   n=98   <- the ghost, better supported
#
# Choosing by observation count picked the ghost. The arm drove at 52cm, hit its 47cm
# limit, and reported "out of reach" — accurate about entirely the wrong thing.

def _two_entries(S, near_r, far_r, near_n, far_n):
    S.WORLD.clear()
    with S.w2d_lock:
        S.WORLD.objs[1] = {"label": "green cube", "xy": (near_r, 0.0), "w_m": 0.05,
                           "d_m": 0.05, "h_m": 0.05, "n": near_n, "t": 0.0}
        S.WORLD.objs[2] = {"label": "green cube", "xy": (far_r, 0.0), "w_m": 0.05,
                           "d_m": 0.05, "h_m": 0.05, "n": far_n, "t": 0.0}


def test_the_camera_breaks_the_tie_against_the_popular_ghost(S, monkeypatch):
    """The live numbers: a real cube at 0.371 and a ghost at 0.520 with 2.5x the support."""
    _two_entries(S, 0.371, 0.520, near_n=40, far_n=98)
    tip = __import__("numpy").array([0.221, 0.0, 0.077])
    monkeypatch.setattr(S, "_live_range_and_tip", lambda _l: (0.186, tip))
    assert S._find_map_tag("green") == 1, "must pick the entry the camera agrees with"


def test_without_a_view_it_still_falls_back_to_support(S, monkeypatch):
    """No picture, no arbitration — the old rule is the right one."""
    _two_entries(S, 0.371, 0.520, near_n=40, far_n=98)
    monkeypatch.setattr(S, "_live_range_and_tip", lambda _l: (None, None))
    assert S._find_map_tag("green") == 2


def test_a_view_that_matches_nothing_does_not_get_to_choose(S, monkeypatch):
    """If even the closest entry is wildly out, the camera is arbitrating noise."""
    _two_entries(S, 0.371, 0.520, near_n=40, far_n=98)
    tip = __import__("numpy").array([0.221, 0.0, 0.077])
    monkeypatch.setattr(S, "_live_range_and_tip", lambda _l: (1.40, tip))
    assert S._find_map_tag("green") == 2, "should fall back, not pick the least-bad"


def test_the_camera_can_also_confirm_the_popular_entry(S, monkeypatch):
    """The guard is not biased toward near entries — it is biased toward agreement."""
    _two_entries(S, 0.371, 0.520, near_n=98, far_n=40)
    tip = __import__("numpy").array([0.221, 0.0, 0.077])
    monkeypatch.setattr(S, "_live_range_and_tip", lambda _l: (0.30, tip))
    assert S._find_map_tag("green") == 2, "the far entry matches the picture here"


def test_a_single_entry_needs_no_arbitration(S, monkeypatch):
    S.WORLD.clear()
    with S.w2d_lock:
        S.WORLD.objs[7] = {"label": "green cube", "xy": (0.371, 0.0), "w_m": 0.05,
                           "d_m": 0.05, "h_m": 0.05, "n": 3, "t": 0.0}
    monkeypatch.setattr(S, "_live_range_and_tip", lambda _l: (None, None))
    assert S._find_map_tag("green") == 7


def test_picked_entries_are_still_skipped(S, monkeypatch):
    _two_entries(S, 0.371, 0.520, near_n=40, far_n=98)
    with S.w2d_lock:
        S.WORLD.objs[1]["picked"] = True
    tip = __import__("numpy").array([0.221, 0.0, 0.077])
    monkeypatch.setattr(S, "_live_range_and_tip", lambda _l: (0.186, tip))
    assert S._find_map_tag("green") == 2, "a picked entry is out of the running"
