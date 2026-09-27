"""The one vision-model call that STEERS, and the guards that keep it honest.

Every other call in this layer is a classification the caller branches on; a wrong one
picks a wrong branch and the log says so. This one is a measurement the arm moves by,
and a wrong one does not fail loudly — it drives somewhere else and the run looks
normal right up until the jaws close on air. So the parser is where the safety lives,
and all of it is pure and tested without a network.

What the model is NEVER asked: how far away anything is. Range is where it is least
reliable, and where the rig's own geometry is already correct given a right pixel.

    pytest tests/test_vlm_locate.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "examples" / "mission_server"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from gemini_vision import BOX_SCALE, Fix, parse_fix  # noqa: E402

W, H = 640, 480


def reply(**kw):
    d = {"found": True, "box": [200, 100, 400, 300], "confidence": 0.9, "reason": "ok"}
    d.update(kw)
    return json.dumps(d)


# --- the happy path, and the pixel conversion ---------------------------------------

def test_a_good_box_becomes_a_centre_pixel_in_frame_coordinates():
    f = parse_fix(reply(box=[200, 100, 400, 300]), W, H)
    assert f.ok
    # box is [ymin, xmin, ymax, xmax] on a 0-1000 grid
    assert f.bbox_xyxy == pytest.approx((64.0, 96.0, 192.0, 192.0))
    assert f.uv == pytest.approx((128.0, 144.0))


def test_the_conversion_follows_the_frame_it_was_given():
    """Same reply, different frame size, different pixels — no hidden constants."""
    a = parse_fix(reply(), 640, 480)
    b = parse_fix(reply(), 1280, 800)
    assert b.uv[0] == pytest.approx(a.uv[0] * 2)
    assert b.uv[1] == pytest.approx(a.uv[1] * 800 / 480)


def test_a_fix_carries_no_distance_at_all():
    """Pins the boundary: this type must never grow a range field."""
    f = parse_fix(reply(), W, H)
    for banned in ("range_m", "range", "distance", "depth", "z_m"):
        assert not hasattr(f, banned), f"Fix must not carry {banned}"


# --- refusals: each one is a way the arm would otherwise be sent somewhere wrong ----

def test_prose_instead_of_json_is_refused():
    assert not parse_fix("I can see a red cube on the left", W, H).ok


def test_json_that_is_not_an_object_is_refused():
    assert not parse_fix("[1,2,3,4]", W, H).ok


def test_found_false_is_a_refusal_and_keeps_the_models_reason():
    f = parse_fix(json.dumps({"found": False, "reason": "only a blue tray"}), W, H)
    assert not f.ok and "blue tray" in f.reason


@pytest.mark.parametrize("box", [[1, 2, 3], [], [1, 2, 3, 4, 5], "nope", None])
def test_a_box_that_is_not_four_numbers_is_refused(box):
    assert not parse_fix(reply(box=box), W, H).ok


def test_a_non_numeric_box_is_refused():
    assert not parse_fix(reply(box=["a", "b", "c", "d"]), W, H).ok


def test_an_inverted_box_is_refused():
    """Reversed corners have a plausible-looking centre, which is the danger."""
    assert not parse_fix(reply(box=[400, 300, 200, 100]), W, H).ok


def test_a_box_outside_the_grid_is_refused():
    assert not parse_fix(reply(box=[0, 0, 5000, 5000]), W, H).ok


def test_a_sliver_is_refused():
    """The shape a hallucinated box takes when there is nothing to point at."""
    assert not parse_fix(reply(box=[200, 100, 203, 300]), W, H).ok


def test_a_box_covering_the_whole_frame_is_refused():
    """'Somewhere in here' carries no more information than not answering."""
    assert not parse_fix(reply(box=[0, 0, 1000, 1000]), W, H).ok


def test_low_confidence_is_refused_and_the_threshold_is_the_callers():
    assert not parse_fix(reply(confidence=0.2), W, H).ok
    assert parse_fix(reply(confidence=0.2), W, H, min_confidence=0.1).ok


def test_a_missing_confidence_is_treated_as_no_confidence():
    d = json.loads(reply()); del d["confidence"]
    assert not parse_fix(json.dumps(d), W, H).ok


# --- clipping: still steerable, but never rangeable ---------------------------------

@pytest.mark.parametrize("box,edge", [
    ([200, 0, 400, 300], "left"),
    ([0, 100, 400, 300], "top"),
    ([200, 100, 1000, 300], "bottom"),
    ([200, 100, 400, 1000], "right"),
])
def test_a_box_touching_an_edge_is_flagged_clipped(box, edge):
    f = parse_fix(reply(box=box), W, H)
    assert f.ok, f"{edge} box should still yield a steerable fix"
    assert f.clipped, f"{edge} box must be flagged clipped"


def test_a_box_clear_of_every_edge_is_not_flagged():
    assert not parse_fix(reply(box=[200, 100, 400, 300]), W, H).clipped


def test_a_refused_fix_says_why():
    for bad in ("junk", reply(box=[0, 0, 1000, 1000]), reply(confidence=0.0)):
        f = parse_fix(bad, W, H)
        assert not f.ok and f.reason, "a refusal must carry a reason"
        assert "no fix" in f.describe()


# --- where the assist is wired, and where it deliberately is not --------------------
# Tried at the centring hover FIRST, and it did not help: of the eight frames the servo
# saved after reporting "object not in view", seven contain no cube at all — the model
# correctly answers "there is no green cube visible in the image, only a blurry surface
# and part of the gripper". That is the camera being aimed somewhere else, and no
# detector fixes a pose error. So it sits upstream, at the survey, while the object is
# still in frame.

import pytest as _pytest  # noqa: E402


@_pytest.fixture(scope="module")
def S():
    for extra in (REPO / "src",):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    try:
        import mission_server
    except Exception as exc:  # pragma: no cover - environment dependent
        _pytest.skip(f"mission_server did not import: {exc}")
    return mission_server


def test_the_assist_is_budgeted_per_locate_not_per_frame(S):
    """A survey reads many frames; a network round trip on each would put the model
    in the control path."""
    S.vlm_budget_reset(2)
    assert S._vlm_left[0] == 2


def test_it_declines_once_the_budget_is_spent(S, monkeypatch):
    monkeypatch.setattr(S, "GEMINI", object())     # non-None, must still not be called
    S._vlm_left[0] = 0
    assert S.vlm_track(object(), "red") is None


def test_it_declines_when_there_is_no_model(S):
    S.vlm_budget_reset(2)
    orig = S.GEMINI
    try:
        S.GEMINI = None
        assert S.vlm_track(object(), "red") is None
    finally:
        S.GEMINI = orig


def test_it_declines_when_switched_off(S, monkeypatch):
    monkeypatch.setattr(S, "VLM_ASSIST", False)
    S.vlm_budget_reset(2)
    assert S._vlm_left[0] == 0
    assert S.vlm_track(object(), "red") is None


def test_a_bare_colour_is_expanded_before_asking(S, monkeypatch):
    """'red' is a poor thing to ask a vision model to box; 'red cube' is not."""
    asked = []

    class FakeGemini:
        def locate_object(self, rgb, label):
            asked.append(label)
            return Fix(False, reason="stub")

    monkeypatch.setattr(S, "GEMINI", FakeGemini())
    monkeypatch.setattr(S, "VLM_ASSIST", True)
    S.vlm_budget_reset(2)
    S.vlm_track(object(), "red")
    assert asked == ["red cube"], asked


def test_a_good_fix_becomes_a_track_the_servo_can_use(S, monkeypatch):
    class FakeGemini:
        def locate_object(self, rgb, label):
            return Fix(True, uv=(320.0, 240.0), bbox_xyxy=(300.0, 220.0, 340.0, 260.0),
                       clipped=False, confidence=0.95, reason="the cube")

    monkeypatch.setattr(S, "GEMINI", FakeGemini())
    monkeypatch.setattr(S, "VLM_ASSIST", True)
    S.vlm_budget_reset(2)
    tr = S.vlm_track(object(), "red cube")
    assert tr is not None
    assert tr.uv == (320.0, 240.0)
    assert tr.bbox_xyxy == (300.0, 220.0, 340.0, 260.0)
    assert tr.area_px == 1600 and tr.clipped is False


def test_a_model_exception_never_reaches_the_mission(S, monkeypatch):
    class Boom:
        def locate_object(self, rgb, label):
            raise RuntimeError("network died mid-pick")

    monkeypatch.setattr(S, "GEMINI", Boom())
    monkeypatch.setattr(S, "VLM_ASSIST", True)
    S.vlm_budget_reset(2)
    assert S.vlm_track(object(), "red cube") is None


def test_the_empty_verdict_is_trusted_only_one_way(S):
    """An 'empty' verdict makes the robot do LESS; 'holding' would make it do more on
    the model's word alone. Fail-safe is not symmetric."""
    assert S.GRASP_EMPTY_TRUST >= 0.8


# --- the frames must reach the model the colour they actually are -------------------
# cv2.imencode writes its input as if it were BGR. This server's frames are RGB
# throughout (publish does rgb[:, :, ::-1] before encoding; the trackers use
# COLOR_RGB2HSV), but the vision layer handed its RGB array straight to imencode — so
# red and blue were exchanged in every picture the model was shown, green untouched.
#
# Nothing raised. A colour-swapped frame is a perfectly plausible picture, just not the
# one in front of the camera. It surfaced only as the model insisting a red cube was
# blue — three times — and once as a confident 100% "the jaws are empty" that cleared a
# good carry, because it had been asked about a red cube and shown a blue one.

def _decode(jpeg_bytes):
    import cv2
    import numpy as np
    bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)      # back to RGB to compare


def _solid_rgb(r, g, b):
    import numpy as np
    a = np.zeros((32, 32, 3), np.uint8)
    a[:, :, 0], a[:, :, 1], a[:, :, 2] = r, g, b
    return a


@pytest.mark.parametrize("name,rgb,dominant", [
    ("red", (220, 20, 20), 0),
    ("green", (20, 200, 20), 1),
    ("blue", (20, 20, 220), 2),
])
def test_a_frame_survives_encoding_the_colour_it_started(name, rgb, dominant):
    from gemini_vision import GeminiVision
    out = _decode(GeminiVision._jpeg(_solid_rgb(*rgb)))
    mean = out.reshape(-1, 3).mean(axis=0)
    assert int(mean.argmax()) == dominant, (
        f"a {name} frame decoded as channel {int(mean.argmax())} dominant: {mean}")


def test_red_does_not_come_out_blue():
    """The exact defect, stated as its symptom."""
    from gemini_vision import GeminiVision
    out = _decode(GeminiVision._jpeg(_solid_rgb(220, 20, 20)))
    mean = out.reshape(-1, 3).mean(axis=0)
    assert mean[0] > 2 * mean[2], f"red channel {mean[0]:.0f} vs blue {mean[2]:.0f}"


def test_a_greyscale_frame_is_passed_through_unharmed():
    """The swap must be conditional on there being three channels to swap."""
    import numpy as np

    from gemini_vision import GeminiVision
    assert GeminiVision._jpeg(np.full((32, 32), 128, np.uint8))


# --- the assist has to be on a path the pick actually takes -------------------------
# It was first wired into locate_from_survey, which only runs when USE_SURVEY_LOCATE is
# on — and that is [False]. So it sat in code the pick never executes and had never
# once fired on hardware, while the operator kept hitting the exact failure it was
# built for: "'red cube' is not on the 2D map", with the cube in plain view.

def test_the_assist_is_reachable_from_run_mission(S):
    """Guards the mistake: wired somewhere real, not just somewhere plausible."""
    import inspect
    src = inspect.getsource(S.run_mission)
    assert "vlm_track" in src, "run_mission must be able to fall back to the model"
    assert "vlm_budget_reset" in src, "and must reset the budget when it does"


def test_it_is_the_last_resort_not_the_first_choice(S):
    """The map is consulted first; the model only answers when the map has nothing."""
    import inspect
    src = inspect.getsource(S.run_mission)
    assert src.index("_mapped_xy(label)") < src.index("vlm_track"), (
        "the model must be tried AFTER the map, not instead of it")


def test_survey_locate_is_still_off_by_default(S):
    """Pins why the original wiring was dead — if this ever flips, revisit that path."""
    assert S.USE_SURVEY_LOCATE[0] is False


def test_the_model_is_never_asked_for_a_distance(S):
    """The pixel goes through the rig's own geometry. Range is where a model is least
    reliable and where the geometry is already correct given a right pixel."""
    import inspect
    src = inspect.getsource(S.run_mission)
    tail = src.split("vlm_track")[1][:600]
    assert "obj_xy_2d" in tail, "the pixel must be converted by the existing localizer"
