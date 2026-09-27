"""The parts of the vision-model layer that must work without a network.

A model behind an HTTP call is the least reliable thing in this stack: it times out, it
returns 503 under load, it answers off-schema. Both happened while this was being built
— a read timeout on one frame and "high demand" 503s on another model — so what is
pinned here is not the model's judgement but the SERVER's behaviour when the model is
unavailable or wrong. Every one of these paths must end in ``ok=False`` and a caller
that carries on exactly as it did before.

    pytest tests/test_gemini_vision.py
"""

from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "examples" / "mission_server"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from gemini_vision import (  # noqa: E402
    MIN_TIMEOUT_S,
    MISS_REASONS,
    GeminiVision,
    Verdict,
    load_api_key,
    parse_verdict,
)

MISS = set(MISS_REASONS)


# --- parsing -------------------------------------------------------------------------
def test_a_well_formed_answer_parses():
    v = parse_verdict('{"answer":"clipped_bottom","reason":"at the bottom edge",'
                      '"confidence":0.9}', MISS)
    assert v.ok and v.answer == "clipped_bottom" and v.confidence == 0.9
    assert v.reason == "at the bottom edge"


def test_prose_instead_of_json_is_not_an_answer():
    v = parse_verdict("The cube seems to be at the bottom.", MISS)
    assert not v.ok and "did not return JSON" in v.reason


def test_an_answer_outside_the_closed_set_is_refused():
    """A caller branches on this token. An unexpected one falls through every branch
    silently, which is worse than a call that visibly failed."""
    v = parse_verdict('{"answer":"probably_fine","reason":"x","confidence":1}', MISS)
    assert not v.ok and "not one of" in v.reason


def test_json_that_is_not_an_object_is_refused():
    assert not parse_verdict('["clipped_bottom"]', MISS).ok


def test_empty_and_none_are_refused_rather_than_raising():
    for bad in ("", None, "   "):
        assert not parse_verdict(bad, MISS).ok


def test_a_missing_or_junk_confidence_does_not_raise():
    v = parse_verdict('{"answer":"visible","reason":"r"}', MISS)
    assert v.ok and v.confidence == 0.0
    v = parse_verdict('{"answer":"visible","reason":"r","confidence":"high"}', MISS)
    assert v.ok and v.confidence == 0.0


def test_confidence_is_clamped_to_a_probability():
    assert parse_verdict('{"answer":"visible","reason":"","confidence":8}', MISS).confidence == 1.0
    assert parse_verdict('{"answer":"visible","reason":"","confidence":-3}', MISS).confidence == 0.0


def test_a_long_reason_is_truncated_not_dropped():
    v = parse_verdict('{"answer":"visible","reason":"%s","confidence":1}' % ("x" * 900),
                      MISS)
    assert v.ok and 0 < len(v.reason) <= 300


def test_allowed_none_accepts_any_token():
    assert parse_verdict('{"answer":"whatever","reason":"","confidence":1}', None).ok


# --- the unavailable path ------------------------------------------------------------
def test_no_key_means_every_call_reports_unavailable_and_never_raises():
    """The whole contract: without a key the server must keep working."""
    g = GeminiVision(api_key="   ")
    import numpy as np
    frame = np.zeros((8, 8, 3), np.uint8)
    for v in (g.explain_miss(frame, "red cube"),
              g.verify_grasp(frame, "red cube"),
              g.same_object(frame, frame, "red cube")):
        assert isinstance(v, Verdict) and not v.ok
        assert "GOOGLE_API_KEY" in v.reason


def test_an_unavailable_verdict_describes_itself_without_pretending():
    assert Verdict(False, reason="no network").describe().startswith("gemini: unavailable")


def test_ok_is_separate_from_the_answer():
    """"I could not tell" and "I did not run" are different facts. Collapsing them is
    how a silent outage becomes a silent behaviour change."""
    unsure = parse_verdict('{"answer":"unsure","reason":"blurry","confidence":0.2}',
                           {"holding", "empty", "unsure"})
    assert unsure.ok and unsure.answer == "unsure"
    assert not Verdict(False, reason="timeout").ok


def test_the_transport_deadline_cannot_be_set_below_the_api_minimum():
    """The API rejects anything under 10s outright ("Manually set deadline 6s is too
    short"), so a caller asking for 2s would break every call rather than tighten it."""
    assert GeminiVision(timeout_s=2.0, api_key="x").timeout_s == MIN_TIMEOUT_S


def test_load_api_key_reads_env_local_when_the_environment_has_none(tmp_path, monkeypatch):
    """The server does not use python-dotenv, so a key in .env.local is invisible to it
    unless this reads the file itself."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    (tmp_path / ".env.local").write_text('GOOGLE_API_KEY="abc123"\nOTHER=1\n')
    assert load_api_key(tmp_path) == "abc123"


def test_load_api_key_prefers_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "from-env")
    (tmp_path / ".env.local").write_text("GOOGLE_API_KEY=from-file\n")
    assert load_api_key(tmp_path) == "from-env"


def test_load_api_key_is_empty_when_there_is_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert load_api_key(tmp_path) == ""


# --- the closed answer set -----------------------------------------------------------
def test_every_miss_reason_says_what_to_do_about_it():
    """The set is the caller's branch table, so an answer with no action is a gap."""
    assert MISS_REASONS
    for name, meaning in MISS_REASONS.items():
        assert name.islower() and " " not in name
        assert len(meaning) > 20, name


def test_the_clip_answers_cover_every_edge():
    """explain_miss must be able to name the edge, because which edge it is decides
    whether the object can still be ranged (see tracking.table_ray_is_usable)."""
    for edge in ("bottom", "top", "side"):
        assert f"clipped_{edge}" in MISS_REASONS
