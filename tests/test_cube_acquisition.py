"""Both cube colours must acquire the same way.

``find_green`` was a bare one-liner — ``return green_tracker.track(rgb, T)`` — while
``find_red`` had three ways in: continuity, a detector seed, and a full-frame strict
mask, plus an area gate so it could not latch a speck of glare. Nothing chose that
asymmetry; green simply never grew the paths red did.

It bit during mount calibration, which calls ``tracker.reset()`` before its pose sweep.
With the tracker cleared, green had no way back: the cube sat plainly in frame, the
status showed no green fix at all, and the sweep gathered 6 usable views instead of 10.

    pytest tests/test_cube_acquisition.py
"""

from __future__ import annotations

import pathlib
import sys
import time

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


class FakeTracker:
    """Records how it was asked, and answers with whatever the test planted."""

    def __init__(self, *, windowed=None, fullframe=None):
        self.last = None
        self.p_anchor = None
        self.windowed = windowed          # returned when a fix/anchor exists
        self.fullframe = fullframe        # returned for the T=None full-frame call
        self.calls = []

    def track(self, rgb, T_base_cam=None):
        self.calls.append("full" if T_base_cam is None else "window")
        if T_base_cam is None:
            return self.fullframe
        return self.windowed


class FakeTrack:
    def __init__(self, area_px, uv=(320.0, 240.0)):
        self.area_px = area_px
        self.uv = uv
        self.clipped = False


class FakeDetect:
    def __init__(self, boxes=None, age_s=0.0):
        self.boxes = boxes or []
        self.age_s = age_s
        self.asked = []

    def instances(self, label):
        self.asked.append(label)
        return list(self.boxes), time.time() - self.age_s


@pytest.mark.parametrize("colour", ["red", "green"])
def test_both_colours_seed_from_the_detector_when_the_tracker_is_empty(S, colour, monkeypatch):
    """The path green never had. A cleared tracker must still get back on the cube."""
    tr = FakeTracker(windowed=FakeTrack(9000))
    det = FakeDetect(boxes=[{"xyxy": (100.0, 100.0, 200.0, 200.0)}])
    monkeypatch.setattr(S, f"{colour}_tracker", tr)
    monkeypatch.setattr(S, "DETECT", det)
    got = getattr(S, f"find_{colour}")(object(), object())
    assert got is not None, f"find_{colour} failed to acquire from the detector"
    assert det.asked == [f"{colour} cube"], det.asked
    assert tr.last is not None, "the detector box should seed the tracker"


@pytest.mark.parametrize("colour", ["red", "green"])
def test_both_colours_refuse_to_latch_a_speck(S, colour, monkeypatch):
    """A tiny strict-HSV blob is glare, not a cube. Latching one anchors the tracker
    to noise and every later frame then happily 'tracks' it."""
    tr = FakeTracker(fullframe=FakeTrack(40))
    monkeypatch.setattr(S, f"{colour}_tracker", tr)
    monkeypatch.setattr(S, "DETECT", FakeDetect())
    assert getattr(S, f"find_{colour}")(object(), None) is None
    assert tr.last is None, "a rejected speck must not stay latched"


@pytest.mark.parametrize("colour", ["red", "green"])
def test_both_colours_accept_a_full_frame_blob_that_is_big_enough(S, colour, monkeypatch):
    tr = FakeTracker(fullframe=FakeTrack(S.CUBE_ACQUIRE_MIN_AREA_PX + 1))
    monkeypatch.setattr(S, f"{colour}_tracker", tr)
    monkeypatch.setattr(S, "DETECT", FakeDetect())
    assert getattr(S, f"find_{colour}")(object(), None) is not None


@pytest.mark.parametrize("colour", ["red", "green"])
def test_both_colours_prefer_continuity_over_re_acquiring(S, colour, monkeypatch):
    """With a live fix, neither should pay for a detector lookup or a full-frame scan."""
    tr = FakeTracker(windowed=FakeTrack(9000))
    tr.last = FakeTrack(9000)
    det = FakeDetect(boxes=[{"xyxy": (0.0, 0.0, 10.0, 10.0)}])
    monkeypatch.setattr(S, f"{colour}_tracker", tr)
    monkeypatch.setattr(S, "DETECT", det)
    assert getattr(S, f"find_{colour}")(object(), object()) is not None
    assert tr.calls == ["window"], tr.calls
    assert det.asked == [], "should not have consulted the detector"


@pytest.mark.parametrize("colour", ["red", "green"])
def test_a_stale_detector_box_is_not_treated_as_evidence(S, colour, monkeypatch):
    """The detector runs at ~2.5s; a box far older than that says nothing about now."""
    tr = FakeTracker(fullframe=None)
    det = FakeDetect(boxes=[{"xyxy": (100.0, 100.0, 200.0, 200.0)}],
                     age_s=S.CUBE_DET_MAX_AGE_S + 5.0)
    monkeypatch.setattr(S, f"{colour}_tracker", tr)
    monkeypatch.setattr(S, "DETECT", det)
    assert getattr(S, f"find_{colour}")(object(), None) is None
    assert "full" in tr.calls, "should still have tried the full-frame mask"


@pytest.mark.parametrize("colour", ["red", "green"])
def test_acquisition_works_before_the_detector_thread_exists(S, colour, monkeypatch):
    """DETECT is None until main() builds it; acquisition must not crash on that."""
    tr = FakeTracker(fullframe=FakeTrack(9000))
    monkeypatch.setattr(S, f"{colour}_tracker", tr)
    monkeypatch.setattr(S, "DETECT", None)
    assert getattr(S, f"find_{colour}")(object(), None) is not None


def test_the_two_finders_share_one_implementation(S):
    """The asymmetry came from two hand-written copies drifting apart."""
    import inspect
    red = inspect.getsource(S.find_red).split("return", 1)[1]
    green = inspect.getsource(S.find_green).split("return", 1)[1]
    assert "_find_cube" in red and "_find_cube" in green
