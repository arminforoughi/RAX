"""The layer above the servo, which both arms now share instead of each having a copy.

THE PROPERTIES WORTH PINNING here are the ones that were learned the expensive way on
hardware, because those are the ones a well-meaning refactor quietly undoes:

  * the camera override is ONE-WAY. A confident camera "empty" clears a carry; a camera
    "holding" never authorises one. Making it symmetric looks like a tidy-up and lets a
    remote model drive the arm through a place on nothing but its own say-so.
  * a grasp check may be SLOWER than the grasp, and is allowed to change its mind inside
    the attempt that can still retry. Dropping the settle window is invisible in tests
    that do not model the delay, and cost a live run.
  * a marginal reading is not a confident one. pick.py recorded a real hold at 31.98
    against a 31.9 threshold.
  * recording an episode must NEVER fail a run. It is diagnostic.
"""

from __future__ import annotations

import json

import pytest

from rax.manipulation.attempt import with_retries
from rax.manipulation.episodes import EpisodeLog
from rax.manipulation.grip import (
    GRASP_EMPTY_TRUST, CameraVerdict, CurrentRise, GripReading, PositionThreshold,
    reconcile, settled)

#: The X250's two measured populations, from 113 demonstrations.
X250 = PositionThreshold(holding=31.9, empty=30.2)


# ---------------------------------------------------------------------------------
class TestTheTwoSensorsGiveTheSameShapeOfAnswer:
    def test_position_above_the_line_is_a_hold(self):
        r = X250.verdict(34.0)
        assert r.held is True and r.kind == "position" and r.margin > 0

    def test_position_on_air_is_not(self):
        r = X250.verdict(30.2)
        assert r.held is False and r.margin < 0

    def test_the_real_edge_case_reads_as_held_but_marginal(self):
        # pick.py logged this exact number: a tube gripped by its edge at 31.98.
        r = X250.verdict(31.98)
        assert r.held is True
        assert r.marginal, "a hold 0.08 past the line should invite a second opinion"

    def test_a_confident_hold_is_not_marginal(self):
        assert not X250.verdict(40.0).marginal

    def test_AN_OPEN_GRIPPER_IS_NOT_SCORED_AS_A_GRASP(self):
        """The trap this guard exists for, observed in the first end-to-end place.

        "Above holding" means "something stopped the jaws" only if they were CLOSING. The
        place-open position is 50.0, which is far above the 31.9 holding line, so a reading
        taken after releasing a tube reported a firm grasp of the air it had just let go
        of — and the run reported DONE because of it.
        """
        guarded = PositionThreshold(holding=31.9, empty=30.2, open_above=42.0)
        r = guarded.verdict(50.0)
        assert r.held is False, "an open gripper must never read as holding"
        assert "open" in r.detail and "not a grasp verdict" in r.detail

    def test_the_guard_leaves_a_real_hold_alone(self):
        guarded = PositionThreshold(holding=31.9, empty=30.2, open_above=42.0)
        assert guarded.verdict(31.98).held is True     # the logged edge case
        assert guarded.verdict(34.0).held is True
        assert guarded.verdict(30.2).held is False     # air

    def test_without_the_guard_the_old_wrong_answer_comes_back(self):
        # Documents the difference, so `open_above` cannot be deleted as decoration.
        assert PositionThreshold(holding=31.9, empty=30.2).verdict(50.0).held is True

    def test_a_guarded_open_reading_reports_how_far_open_it_is(self):
        # The only thing the reading actually tells us, so it is what margin means here.
        r = PositionThreshold(holding=31.9, empty=30.2, open_above=42.0).verdict(50.0)
        assert r.margin == pytest.approx(42.0 - 50.0)

    def test_the_current_sensor_needs_no_such_guard(self):
        # The instructive asymmetry: current only rises when the motor is WORKING, so an
        # open gripper reads idle and scores empty by itself.
        assert CurrentRise(idle=100.0, delta=8.0).verdict(100.0).held is False

    def test_current_rise_is_measured_against_idle_not_zero(self):
        # The same absolute current means opposite things at different idle draws.
        assert CurrentRise(idle=100.0, delta=8.0).verdict(110.0).held is True
        assert CurrentRise(idle=120.0, delta=8.0).verdict(110.0).held is True   # |rise|
        assert CurrentRise(idle=105.0, delta=8.0).verdict(110.0).held is False

    def test_both_sensors_report_their_own_units_in_the_detail(self):
        assert "settled" in X250.verdict(33.0).detail
        assert "current" in CurrentRise(idle=100.0, delta=8.0).verdict(110.0).detail


class TestSettleWaitsForMotionToStop:
    def test_it_returns_the_value_once_it_stops_changing(self):
        seq = iter([45.0, 40.0, 35.0, 31.0, 30.2, 30.2, 30.2])
        got = settled(lambda: next(seq), tol=0.05, timeout=2.0, dt=0.001)
        assert got == pytest.approx(30.2)

    def test_a_mid_motion_read_would_have_lied(self):
        # The point of settling: the jaws pass THROUGH the held band on their way shut,
        # so an early read reports a grasp that has not happened.
        assert X250.verdict(45.0).held is True        # what an early read sees
        assert X250.verdict(30.2).held is False       # what it actually was

    def test_a_still_creeping_gripper_returns_its_last_reading(self):
        vals = iter([50.0 - i for i in range(500)])
        got = settled(lambda: next(vals), tol=0.05, timeout=0.05, dt=0.001)
        assert got < 50.0      # it returned something, rather than hanging or raising


# ---------------------------------------------------------------------------------
class TestTheCameraOverrideIsOneWay:
    """The asymmetry is the design. See the module docstring."""

    def test_no_camera_leaves_the_mechanical_verdict_alone(self):
        d = reconcile(X250.verdict(34.0), None)
        assert d.held is True and not d.overridden

    def test_an_unavailable_camera_leaves_it_alone(self):
        d = reconcile(X250.verdict(34.0), CameraVerdict(ok=False, reason="no key"))
        assert d.held is True and not d.overridden
        assert "unavailable" in d.detail

    def test_an_unsure_camera_is_not_evidence(self):
        d = reconcile(X250.verdict(34.0),
                      CameraVerdict(ok=True, answer="unsure", confidence=0.9))
        assert d.held is True and not d.overridden

    def test_a_confident_camera_empty_CLEARS_a_mechanical_hold(self):
        d = reconcile(X250.verdict(34.0),
                      CameraVerdict(ok=True, answer="empty", confidence=0.95,
                                    reason="the jaws are closed on nothing"))
        assert d.held is False, "this is the override that stops a place on empty jaws"
        assert d.overridden is True
        assert "OVERRULES" in d.detail

    def test_an_UNCONFIDENT_camera_empty_does_not(self):
        d = reconcile(X250.verdict(34.0),
                      CameraVerdict(ok=True, answer="empty",
                                    confidence=GRASP_EMPTY_TRUST - 0.01))
        assert d.held is True and not d.overridden
        assert d.advisory_disagreement is True

    def test_a_camera_HOLDING_never_authorises_a_carry(self):
        # The reverse direction, and it must stay advisory however sure the model is:
        # otherwise a hallucination drives the arm through a place with nothing in it.
        d = reconcile(X250.verdict(30.2),
                      CameraVerdict(ok=True, answer="holding", confidence=0.99))
        assert d.held is False, "a camera must not authorise a carry on its own say-so"
        assert d.overridden is False
        assert d.advisory_disagreement is True
        assert "threshold" in d.detail    # but it does say the threshold may be too high

    def test_agreement_is_reported_as_agreement(self):
        d = reconcile(X250.verdict(34.0),
                      CameraVerdict(ok=True, answer="holding", confidence=0.9))
        assert d.held is True and not d.advisory_disagreement
        assert "agrees" in d.detail

    def test_the_override_works_for_the_current_sensor_too(self):
        # Both arms share the blind spot, so both share the override.
        mech = CurrentRise(idle=100.0, delta=8.0).verdict(112.0)
        assert mech.held is True
        d = reconcile(mech, CameraVerdict(ok=True, answer="empty", confidence=0.9))
        assert d.held is False and d.overridden


# ---------------------------------------------------------------------------------
class TestRetries:
    def test_it_stops_at_the_first_confirmed_success(self):
        calls = []
        out = with_retries(lambda n: calls.append(n) or "tried",
                           lambda: (True, "held"), tries=3)
        assert out.ok and out.tries == 1 and calls == [1]

    def test_it_keeps_going_after_a_miss(self):
        seen = []
        verdicts = iter([(False, "empty"), (False, "empty"), (True, "held")])
        out = with_retries(lambda n: seen.append(n) or "tried",
                           lambda: next(verdicts), tries=3)
        assert out.ok and out.tries == 3 and seen == [1, 2, 3]

    def test_it_gives_up_and_says_so(self):
        out = with_retries(lambda n: "tried", lambda: (False, "empty"), tries=2)
        assert not out.ok and out.tries == 2 and "empty" in out.detail

    def test_a_raising_action_is_a_failed_attempt_not_a_crash(self):
        def boom(n):
            raise RuntimeError("target not in view")
        out = with_retries(boom, lambda: (False, ""), tries=2)
        assert not out.ok and "target not in view" in out.detail

    def test_A_VERIFY_DETAIL_DOES_NOT_BURY_WHY_THE_ACTION_FAILED(self):
        """The reporting bug that cost a debugging round.

        The descent was dying of a TypeError mid-way and the recorded detail read
        "gripper current 0.0 vs idle 0.8" — true, and entirely beside the point. When the
        action RAISED, its message is the only thing that explains the attempt.
        """
        def boom(n):
            raise TypeError("'NoneType' object is not subscriptable")
        out = with_retries(boom, lambda: (False, "gripper reads empty"), tries=1)
        assert "NoneType" in out.detail, "the crash must survive into the record"
        assert "gripper reads empty" in out.detail, "the reading is still worth keeping"

    def test_a_clean_miss_still_reports_the_reading_alone(self):
        # When nothing raised, the verify detail IS the explanation and should not be
        # cluttered with the action's ordinary return value.
        out = with_retries(lambda n: "closed", lambda: (False, "jaws empty"), tries=1)
        assert out.detail == "jaws empty"

    def test_between_runs_before_retries_only(self):
        got = []
        verdicts = iter([(False, ""), (True, "")])
        with_retries(lambda n: "", lambda: next(verdicts), tries=2,
                     between=got.append)
        assert got == [2], "between() must not run before the first attempt"

    def test_checkpoint_can_abort_the_whole_loop(self):
        class Stop(Exception):
            pass

        def stop():
            raise Stop

        with pytest.raises(Stop):
            with_retries(lambda n: "", lambda: (False, ""), tries=5, checkpoint=stop)

    def test_A_LATE_OVERRULE_LANDS_INSIDE_THE_ATTEMPT_THAT_CAN_RETRY(self):
        """The bug this parameter exists for. See the module docstring.

        The verifier says held, then changes its mind — as the off-thread camera check
        really does, 1.5-5s after the mechanical verdict. Without the settle window the
        loop returns success and the overrule arrives after everything that could act on
        it has returned.
        """
        seen = []
        polls = {"n": 0}

        def verify():
            """Attempt 1: 'contact', then a beat later 'actually, empty'. Attempt 2 holds.

            Written as a counter rather than a fixed list because the settle window polls
            until the deadline, so the number of calls is a timing detail, not something
            a test should have to predict.
            """
            polls["n"] += 1
            if seen[-1] == 1:
                return (True, "contact") if polls["n"] == 1 else (False, "the jaws are empty")
            return (True, "held")

        out = with_retries(lambda n: seen.append(n) or "closed", verify,
                           tries=2, settle_s=0.02, poll_s=0.001)
        assert seen == [1, 2], "the first success should have been overturned and retried"
        assert out.ok and out.tries == 2

    def test_without_a_settle_window_the_first_answer_stands(self):
        # Documents the difference, so the parameter cannot be removed as a no-op.
        answers = iter([(True, "contact"), (False, "empty")])
        out = with_retries(lambda n: "closed", lambda: next(answers), tries=2)
        assert out.ok and out.tries == 1

    def test_an_overturned_run_that_never_succeeds_says_it_was_overturned(self):
        answers = iter([(True, "contact"), (False, "empty"),
                        (True, "contact"), (False, "empty")])
        out = with_retries(lambda n: "closed", lambda: next(answers), tries=2,
                           settle_s=0.02, poll_s=0.001)
        assert not out.ok and out.overturned


# ---------------------------------------------------------------------------------
class TestEpisodesAreRecordedAndNeverFatal:
    def test_a_run_is_written_as_one_json_line(self, tmp_path):
        log = EpisodeLog(tmp_path / "eps.jsonl")
        ep = log.start("pick", "test tube", arm="x250")
        log.attempt(ep, 1, False, "missed by 2cm")
        log.attempt(ep, 2, True, "picked")
        log.end(ep, True, "picked on the second try")

        lines = (tmp_path / "eps.jsonl").read_text().splitlines()
        assert len(lines) == 1
        r = json.loads(lines[0])
        assert r["kind"] == "pick" and r["label"] == "test tube" and r["arm"] == "x250"
        assert r["ok"] is True and r["tries"] == 2
        assert [a["ok"] for a in r["attempts"]] == [False, True]

    def test_every_attempt_is_kept_not_just_the_last(self):
        # "How many tries did it take" is the question these are for.
        log = EpisodeLog("/nonexistent/should/not/matter")
        ep = log.start("pick", "tube")
        for n in range(1, 4):
            log.attempt(ep, n, n == 3, f"try {n}")
        assert len(ep.attempts) == 3

    def test_a_failing_view_callback_does_not_fail_the_run(self, tmp_path):
        def broken(when, ep_id):
            raise OSError("camera gone")

        log = EpisodeLog(tmp_path / "eps.jsonl", views=broken)
        ep = log.start("pick", "tube")          # must not raise
        log.end(ep, True, "picked")             # must not raise
        assert json.loads((tmp_path / "eps.jsonl").read_text())["ok"] is True

    def test_an_unwritable_path_does_not_fail_the_run(self, tmp_path):
        # A pick that worked must not be reported as failed because a disk was full.
        blocker = tmp_path / "file"
        blocker.write_text("not a directory")
        log = EpisodeLog(blocker / "eps.jsonl")
        ep = log.start("pick", "tube")
        assert log.end(ep, True, "picked").ok is True

    def test_a_failing_probe_is_skipped_not_fatal(self, tmp_path):
        def broken():
            raise RuntimeError("no FK")
        log = EpisodeLog(tmp_path / "eps.jsonl", probe=broken)
        ep = log.start("pick", "tube")
        log.attempt(ep, 1, True, "picked")
        assert ep.attempts[0]["ok"] is True

    def test_the_probe_decorates_each_attempt(self, tmp_path):
        log = EpisodeLog(tmp_path / "eps.jsonl", probe=lambda: {"tip_cm": [1.0, 2.0, 3.0]})
        ep = log.start("pick", "tube")
        log.attempt(ep, 1, True, "picked")
        assert ep.attempts[0]["tip_cm"] == [1.0, 2.0, 3.0]

    def test_a_truncated_last_line_does_not_break_reading(self, tmp_path):
        # Normal: the writer can be stopped mid-line. A reader that raises here makes the
        # log unreadable exactly when something has gone wrong.
        p = tmp_path / "eps.jsonl"
        p.write_text('{"id":1,"kind":"pick","ok":true,"tries":1}\n{"id":2,"kin')
        assert len(EpisodeLog(p).records()) == 1

    def test_tally_counts_what_a_change_gets_judged_on(self, tmp_path):
        p = tmp_path / "eps.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in [
            {"id": 1, "kind": "pick", "ok": True, "tries": 1},
            {"id": 2, "kind": "pick", "ok": False, "tries": 3},
            {"id": 3, "kind": "pick", "ok": True, "tries": 2},
            {"id": 4, "kind": "place", "ok": True, "tries": 1},
        ]))
        t = EpisodeLog(p).tally("pick")
        assert t["runs"] == 3 and t["ok"] == 2
        assert t["rate"] == pytest.approx(2 / 3)
        assert t["mean_tries"] == pytest.approx(2.0)

    def test_an_absent_log_tallies_to_nothing_rather_than_raising(self, tmp_path):
        assert EpisodeLog(tmp_path / "never.jsonl").tally()["runs"] == 0
