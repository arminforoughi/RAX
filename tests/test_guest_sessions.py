"""Tests for timed, queued access to one shared robot.

This policy ran in production for months and had never been tested, because it was
tangled up with Flask request objects and arm motion. Extracted, it is pure logic with
an injectable clock — so an evening of visitor behaviour runs in a millisecond.

Each test names the behaviour someone actually exhibits: not handing the robot back,
closing the tab, or coming straight back for another go.

    python tests/test_guest_sessions.py
    pytest tests/test_guest_sessions.py
"""

from __future__ import annotations

import pathlib
import sys

# Guest queueing is a property of the *demo server*, not of the pick stack, so it lives
# beside the server rather than in the published package. The test stays here because it
# is pure logic and the suite runs everything in one place.
REPO = pathlib.Path(__file__).resolve().parents[1]
MISSION_SERVER_DIR = REPO / "examples" / "mission_server"
if str(MISSION_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION_SERVER_DIR))

from guest_sessions import GuestConfig, GuestSessions  # noqa: E402


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def _sessions(**kw):
    clock = Clock()
    ended = []
    g = GuestSessions(GuestConfig(**kw), now=clock,
                      on_turn_end=lambda: ended.append(clock.t))
    return g, clock, ended


def test_first_visitor_gets_the_robot_immediately():
    g, _c, _e = _sessions()
    g.see("a", "1.1.1.1")
    st = g.tick("a")
    assert st.playing and st.position == 0 and st.token
    assert st.left == g.cfg.minutes * 60.0


def test_second_visitor_queues_behind_the_first():
    g, _c, _e = _sessions()
    for v in ("a", "b"):
        g.see(v)
    g.tick("a")
    st = g.tick("b")
    assert not st.playing and st.position == 1 and st.token is None
    assert st.eta > 0, "a waiter should be told how long they have to wait"


def test_a_turn_expires_and_the_next_visitor_is_promoted():
    """Nobody hands the robot back voluntarily."""
    g, clock, ended = _sessions(minutes=5.0)
    g.see("a"); g.see("b")
    g.tick("a"); g.tick("b")
    clock.advance(5 * 60 + 1)
    st = g.tick("b")
    assert st.playing, "b should have been promoted once a's turn expired"
    assert ended, "the turn-end callback must fire so the arm gets parked"
    assert not g.tick("a").playing


def test_expiry_is_lazy_not_timed():
    """Checked when someone asks, so the queue cannot drift out of sync with a
    background thread that died."""
    g, clock, _e = _sessions(minutes=1.0)
    g.see("a")
    g.tick("a")
    clock.advance(120)
    active, left, _tok = g.state()          # nobody ticked; reading is what expires it
    assert not active and left == 0.0


def test_waiters_who_stop_polling_drop_out():
    """A closed tab must not hold a place, or real visitors queue behind ghosts."""
    g, clock, _e = _sessions(queue_ttl_s=25.0)
    for v in ("a", "b", "c"):
        g.see(v)
    g.tick("a"); g.tick("b"); g.tick("c")
    assert g.tick("c").position == 2
    for _ in range(6):                       # b stops polling; a and c keep going
        clock.advance(5)
        g.see("a"); g.tick("a")
        g.see("c"); g.tick("c")
    assert "b" not in g.queue
    assert g.tick("c").position == 1, "c should have moved up when b dropped"


def test_cooldown_blocks_an_immediate_second_turn():
    """One enthusiast must not be able to hold the robot all evening."""
    g, clock, _e = _sessions(minutes=1.0, cooldown_min=10.0)
    g.see("a"); g.see("b")
    g.tick("a"); g.tick("b")
    clock.advance(61)
    assert g.tick("b").playing
    st = g.tick("a")
    assert not st.playing and st.cooldown > 0


def test_a_cooling_visitor_yields_but_is_not_starved():
    """Yielding to newcomers is the point; being locked out is not. With nobody else
    waiting, the cooling visitor still gets a turn."""
    g, clock, _e = _sessions(minutes=1.0, cooldown_min=10.0)
    g.see("a")
    g.tick("a")
    clock.advance(61)
    g.tick("a")                              # turn expires, a re-queues while cooling
    assert g.cooldown_left("a") > 0
    st = g.tick("a")
    assert st.playing, "with an empty queue, a cooling visitor should still get a turn"


def test_a_newcomer_jumps_ahead_of_someone_cooling():
    """Needs someone else holding the robot, otherwise the cooling visitor is simply
    promoted again and there is no queue order to observe."""
    g, clock, _e = _sessions(minutes=1.0, cooldown_min=10.0)
    g.see("a"); g.see("b")
    g.tick("a"); g.tick("b")                 # a plays, b waits
    clock.advance(61)
    g.see("b"); g.tick("b")                  # a's turn ends, b takes over; a is cooling
    assert g.holds_turn("b")
    g.see("a"); g.tick("a")                  # a re-queues while cooling
    g.see("c"); g.tick("c")                  # c has never played
    assert g.cooldown_left("a") > 0 and g.cooldown_left("c") == 0
    assert g.queue.index("c") < g.queue.index("a"), (
        f"newcomer should be ahead of a cooling visitor: {g.queue}")


def test_only_the_holder_is_recognised_as_the_caller():
    g, _c, _e = _sessions()
    g.see("a"); g.see("b")
    g.tick("a"); g.tick("b")
    assert g.holds_turn("a")
    assert not g.holds_turn("b")
    assert not g.holds_turn(None)
    assert not g.holds_turn("never-seen")


def test_a_token_is_never_handed_to_the_wrong_visitor():
    """The token authorises driving the arm, so it must only ever reach its owner."""
    g, _c, _e = _sessions()
    g.see("a"); g.see("b")
    a = g.tick("a")
    b = g.tick("b")
    assert a.token and b.token is None
    assert a.token != g.tick("b").token


def test_operator_can_end_a_turn_early():
    g, _c, ended = _sessions()
    g.see("a")
    g.tick("a")
    assert g.end_current()
    assert ended, "kicking someone off must still park the arm"
    assert not g.end_current(), "ending an idle robot is a no-op, not an error"


def test_roster_reports_who_is_where():
    g, _c, _e = _sessions()
    g.see("a", "1.1.1.1"); g.see("b", "2.2.2.2")
    g.tick("a"); g.tick("b")
    roster = {r["vid"]: r for r in g.roster()}
    assert roster["a"]["playing"] and not roster["b"]["playing"]
    assert roster["b"]["queued"]
    assert roster["a"]["ip"] == "1.1.1.1"


def test_repeat_visitor_is_seen_not_duplicated():
    g, _c, _e = _sessions()
    assert g.see("a", "1.1.1.1") is True
    assert g.see("a", "1.1.1.1") is False
    assert len(g.visitors) == 1


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
