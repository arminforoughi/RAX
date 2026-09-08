"""Tests for the final centring servo, against a simulated arm.

This is control logic, so golden values cannot prove it — what matters is whether the
loop converges, and whether it does something sane when it cannot. The simulated arm
below has a deliberately awkward property: its pixel gains are INVERTED relative to the
naive expectation, which is exactly what a hand-eye transform carrying roll or yaw error
does on a real rig. A servo with hand-modelled signs drives the wrong way and stalls; one
that measures its gains by probing does not care.

    python tests/test_visual_center.py
    pytest tests/test_visual_center.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rax.manipulation.approach import ApproachConfig  # noqa: E402
from rax.manipulation.approach.visual_center import (  # noqa: E402
    PAN_PROBE_DEG,
    CenteringOps,
    center_on_object,
)

AIM = (440.0, 394.0)


class _Track:
    def __init__(self, uv):
        self.uv = uv


class FakeArm:
    """A tip that moves, and an object whose pixel is a linear function of the pose.

    Signs are inverted on purpose (see the module docstring). ``lost_after`` makes the
    object disappear mid-loop, and ``blocked`` makes every reach unreachable.
    """

    def __init__(self, *, du_dpan=-6.0, dv_dr=-900.0, pan=0.0, radius=0.30,
                 err_u=120.0, err_v=80.0, lost_after=None, blocked=False,
                 range_m=0.30):
        self.du_dpan, self.dv_dr = du_dpan, dv_dr
        self.pan, self.radius = pan, radius
        self.err_u, self.err_v = err_u, err_v      # current pixel error from AIM
        self.pan0, self.radius0 = pan, radius
        self.lost_after, self.blocked = lost_after, blocked
        self._range = range_m
        self.log: list[str] = []
        self.tracks = 0
        self.moves = 0

    # --- the CenteringOps surface ------------------------------------------
    def joints(self):
        return np.array([self.pan, 0.0, 0.0, 0.0, 0.0])

    def tip(self, q=None):
        return np.array([self.radius, 0.0, 0.05])

    def track(self, tries=3):
        self.tracks += 1
        if self.lost_after is not None and self.tracks > self.lost_after:
            return None
        u = AIM[0] + self.err_u + (self.pan - self.pan0) * self.du_dpan
        v = AIM[1] + self.err_v + (self.radius - self.radius0) * self.dv_dr
        return _Track((u, v))

    def range_m(self, track):
        return self._range

    def move_pan(self, delta_deg, *, settle, step):
        self.moves += 1
        self.pan += float(delta_deg)

    def move_tip(self, p_base, *, settle, step):
        if self.blocked:
            return False
        self.moves += 1
        self.radius = float(np.hypot(p_base[0], p_base[1]))
        return True

    def say(self, msg):
        self.log.append(msg)

    def checkpoint(self):
        pass

    # --- for assertions ----------------------------------------------------
    def pixel_error(self):
        t = self.track()
        self.tracks -= 1        # do not let the assertion perturb lost_after
        return abs(t.uv[0] - AIM[0]), abs(t.uv[1] - AIM[1])


def test_ops_protocol_is_satisfied():
    assert isinstance(FakeArm(), CenteringOps)


def test_servo_converges_despite_inverted_gains():
    """The whole point of probing: signs that would break a hand-modelled loop."""
    arm = FakeArm()
    before = arm.pixel_error()
    res = center_on_object(arm, AIM, ApproachConfig())
    after = arm.pixel_error()
    assert res.centered, f"did not converge: {res.reason}; log={arm.log}"
    assert after[0] < before[0] and after[1] < before[1]
    cfg = ApproachConfig()
    assert after[0] < cfg.align_tol_px and after[1] < cfg.align_tol_px * 1.3


def test_servo_measures_its_gains_rather_than_assuming_them():
    """Flip the sign of both gains and it must still converge, unchanged."""
    for du, dv in ((-6.0, -900.0), (6.0, 900.0), (-6.0, 900.0), (6.0, -900.0)):
        arm = FakeArm(du_dpan=du, dv_dr=dv)
        res = center_on_object(arm, AIM, ApproachConfig())
        assert res.centered, f"gains ({du}, {dv}) failed: {res.reason}"


def test_already_centred_costs_no_moves():
    arm = FakeArm(err_u=2.0, err_v=2.0)
    res = center_on_object(arm, AIM, ApproachConfig())
    assert res.centered and res.iterations == 1
    # Only the two probe moves plus their returns; no corrective motion.
    assert arm.moves <= 4


def test_a_useless_pan_gain_is_refused_not_divided_by():
    """If rotating the base barely moves the object, the gain is noise — dividing by
    it produces a wild step. Report and stop instead."""
    arm = FakeArm(du_dpan=0.2)
    res = center_on_object(arm, AIM, ApproachConfig())
    assert not res.centered and res.reason == "pan_probe_failed"
    assert any("barely moves" in m for m in arm.log)
    assert res.xy is not None, "it should still report where the tip ended up"


def test_object_not_in_view_is_reported():
    arm = FakeArm(lost_after=0)
    res = center_on_object(arm, AIM, ApproachConfig())
    assert not res.centered and res.reason == "not_in_view" and res.xy is None


def test_losing_the_object_mid_loop_stops_cleanly():
    """The object usually disappears because it is under the jaws — that is not a
    failure to retry, it is a reason to stop and let the descent happen."""
    # Three tracks are consumed before the loop starts: the initial fix and one per
    # probe. Losing it on the fourth puts the disappearance inside the servo loop.
    arm = FakeArm(lost_after=4)
    res = center_on_object(arm, AIM, ApproachConfig())
    assert not res.centered and res.reason == "lost", f"{res.reason}; log={arm.log}"
    assert any("stopping" in m for m in arm.log)


def test_unreachable_radial_moves_do_not_spin_forever():
    arm = FakeArm(blocked=True, err_u=0.0, err_v=200.0)
    res = center_on_object(arm, AIM, ApproachConfig())
    assert not res.centered
    assert res.iterations <= ApproachConfig().align_iters


def test_close_range_moves_more_gently():
    """Speed scales with range: the base carries the arm's inertia, so a near object
    must be approached with smaller steps."""
    far = FakeArm(range_m=0.40)
    near = FakeArm(range_m=0.05)
    center_on_object(far, AIM, ApproachConfig())
    center_on_object(near, AIM, ApproachConfig())
    # With a tighter per-iteration cap, the close arm cannot swing as far in one step.
    assert abs(near.pan - near.pan0) <= 4.5 * ApproachConfig().align_iters + 1e-9


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


def test_the_result_field_is_named_centered():
    """Pin the attribute name the callers branch on.

    mission_server's _center_on_cube decides whether to trust the servo's position or
    fall back to the mapped one. It was written against a guessed `res.ok`, which does
    not exist — so on the first real failure it raised AttributeError inside the pick
    and aborted a run that would otherwise have recovered. The library's own tests all
    passed, because the bug was in a caller with no coverage.

    A rename here is a silent break there, so the name is the contract.
    """
    res = center_on_object(FakeArm(du_dpan=0.2), AIM, ApproachConfig())
    assert hasattr(res, "centered"), "callers branch on .centered"
    assert not hasattr(res, "ok"), "there is no .ok — do not reintroduce the ambiguity"
    # The other two fields callers read.
    assert hasattr(res, "xy") and hasattr(res, "reason")


# --- online gain adaptation ----------------------------------------------------------
# The probe linearizes over 3 deg and the loop then sizes every correction from that one
# number. On the real rig they are not the same gain: the probe reported 12.8 px/deg
# where the corrective steps realized about 4.
#
# WHERE THIS DOES AND DOES NOT MATTER, because it is easy to over-claim. While the
# computed step exceeds max_pan it is clipped, and a clipped step is the same step
# whatever the gain says — at the 268 px that opens a real pick, 268/12.8 and 268/4 both
# clip to 4.5 deg. The gain only starts to govern once the computed step drops UNDER the
# clip, which is the endgame: the last few iterations closing on the tolerance. A gain
# that over-reports there makes every remaining step a fraction of what it should be, so
# the loop creeps and stalls just outside tolerance instead of finishing.
#
# So these fixtures deliberately work in the unclipped regime. What fixes the clipped
# opening is the iteration budget and the trim decay, pinned separately below.


class OverReportingProbeArm(FakeArm):
    """An arm whose probe flatters the gain that its corrective moves actually realize.

    The probe is identified by its exact magnitude (``PAN_PROBE_DEG``), which is a
    fixture shortcut for what a real rig does through geometry: the response measured
    over 3 deg around the start pose is simply not the response over the next 20.
    """

    def __init__(self, *, probe_boost=12.0, **kw):
        super().__init__(**kw)
        self.probe_boost = probe_boost

    def move_pan(self, delta_deg, *, settle, step):
        self.moves += 1
        d = float(delta_deg)
        is_probe = abs(abs(d) - PAN_PROBE_DEG) < 1e-9
        self.pan += d * (self.probe_boost if is_probe else 1.0)


def test_a_gain_that_over_reports_no_longer_stalls_the_endgame():
    """Steps below the clip, sized from a 12x-optimistic probe, must still converge."""
    arm = OverReportingProbeArm(err_u=200.0, err_v=0.0, dv_dr=0.0, range_m=0.30)
    res = center_on_object(arm, AIM, ApproachConfig())
    assert res.centered, f"did not converge: {res.reason}; log={arm.log}"
    assert any("gains now" in m for m in arm.log), "it never revised its gain"


def test_without_adaptation_the_same_arm_creeps_and_stalls():
    """Pins that the fixture is discriminating, not merely passing."""
    import rax.manipulation.approach.visual_center as vc
    saved = vc.GAIN_DAMPING
    vc.GAIN_DAMPING = 0.0                       # == the servo before this change
    try:
        arm = OverReportingProbeArm(err_u=200.0, err_v=0.0, dv_dr=0.0, range_m=0.30)
        res = center_on_object(arm, AIM, ApproachConfig())
        assert not res.centered, "the fixture is too easy — it converges either way"
    finally:
        vc.GAIN_DAMPING = saved


def test_adaptation_does_not_disturb_an_already_correct_gain():
    """A rig where the probe is right must behave exactly as before."""
    arm = FakeArm(err_u=120.0, err_v=80.0)
    res = center_on_object(arm, AIM, ApproachConfig())
    assert res.centered and res.reason == "converged"


def test_a_gain_sign_flip_is_not_believed():
    """One bad frame must not invert a gain the probe established."""
    from rax.manipulation.approach.visual_center import _update_gain
    assert _update_gain(-12.0, +40.0, 4.0, 1.0) == -12.0     # opposite-sign response
    assert _update_gain(-12.0, -40.0, 0.2, 1.0) == -12.0     # move too small to inform


def test_one_update_cannot_move_a_gain_arbitrarily_far():
    from rax.manipulation.approach.visual_center import GAIN_MAX_RATIO, _update_gain
    old = -12.0
    got = _update_gain(old, -12000.0, 1.0, 1.0)
    assert abs(got) <= abs(old) * GAIN_MAX_RATIO + 1e-9


def test_the_iteration_budget_exceeds_the_error_it_inherits():
    """THE fix for the logged failure, and the one that is easy to lose in a refactor.

    Every pick in mission_server_stdout.log reported max_iters with the object still
    ~180 px off, then grasped the uncorrected mapped position. 250 px at the measured
    12.8 px/deg is ~20 deg of pan; the close-range cap is 4.5 deg per iteration, so
    3 iterations could spend 13.5 deg against a 20 deg problem. The loop was not slow,
    it was structurally unable to finish, and the counter that ended it read like
    "needed more time" rather than "was never given enough".
    """
    cfg = ApproachConfig()
    assert cfg.align_iters * 4.5 > 250.0 / 12.8


# --- the probe divides by what the base ACHIEVED, not what it was asked for ---------
# A real pick logged "center: base rotation barely moves the object" and fell back to
# the mapped position — losing the one step that corrects localization error — because
# the pan joint was against its limit. The probe had divided a near-zero pixel delta by
# the full 3 deg it *requested*, and concluded the object was unservoable.


class LimitedPanArm(FakeArm):
    """A FakeArm whose pan joint clips at ``pan_hi`` and reports what it achieved."""

    def __init__(self, *, pan_hi=1.0, pan_lo=-90.0, travel=1.0, **kw):
        super().__init__(**kw)
        self.pan_hi, self.pan_lo, self.travel = pan_hi, pan_lo, travel

    def move_pan(self, delta_deg, *, settle, step):
        self.moves += 1
        before = self.pan
        # `travel` models servo under-travel: it reaches only this fraction of the ask.
        want = before + float(delta_deg) * self.travel
        self.pan = float(np.clip(want, self.pan_lo, self.pan_hi))
        return self.pan - before


def test_a_pan_probe_blocked_by_a_joint_limit_retries_the_other_way():
    """On its upper limit the base cannot turn +3, but it can turn -3 — and with the
    object needing a negative correction that is enough to centre.

    Before the fix this returned pan_probe_failed: the +3 probe clipped to nothing, the
    pixel delta was ~0, and dividing by the COMMANDED 3 deg looked exactly like an
    object that rotation cannot move.
    """
    arm = LimitedPanArm(pan=1.0, pan_hi=1.0, err_u=-120.0)
    res = center_on_object(arm, AIM, ApproachConfig())
    assert res.centered, f"should have centred; log={arm.log}"
    assert not any("cannot turn here" in m for m in arm.log)


def test_a_base_pinned_both_ways_is_reported_not_divided_by():
    arm = LimitedPanArm(pan=0.0, pan_lo=0.0, pan_hi=0.0)   # welded shut
    res = center_on_object(arm, AIM, ApproachConfig())
    assert not res.centered and res.reason == "pan_probe_failed"
    assert any("cannot turn here" in m for m in arm.log), arm.log


def test_the_gain_is_measured_against_achieved_pan_not_commanded_pan():
    """Under-travel must not shrink the measured gain.

    The arm reaches half of every commanded pan. Dividing the pixel delta by the
    commanded 3 deg would report half the true gain; dividing by the achieved 1.5 deg
    reports it correctly.
    """
    arm = LimitedPanArm(travel=0.5, du_dpan=-6.0)
    center_on_object(arm, AIM, ApproachConfig())
    reported = [m for m in arm.log if "du/dpan=" in m]
    assert reported, arm.log
    got = float(reported[0].split("du/dpan=")[1].split("px/deg")[0])
    assert abs(got - (-6.0)) < 0.5, f"gain should reflect achieved pan, got {got}"


def test_the_probe_returns_the_base_to_where_it_started():
    """Undoing the ACHIEVED pan, not the commanded pan, keeps the probe non-destructive."""
    arm = LimitedPanArm(pan=0.0, pan_hi=90.0)   # free to move, nothing clipped
    start = arm.pan
    _ = _probe_pan_for_test(arm)
    assert abs(arm.pan - start) < 1e-9, f"probe left the base at {arm.pan}, not {start}"


def _probe_pan_for_test(arm):
    from rax.manipulation.approach.visual_center import _probe_pan

    uv0 = np.array(arm.track().uv, dtype=np.float64)
    return _probe_pan(arm, uv0, 0.05, 1.0)


# --- a probe that throws the object out of frame must try the other way -------------
# Measured on the rig: du/dpan ~34 px/deg, so the 3 deg probe swings the object ~100 px.
# Started near a frame edge that ejects it, and the servo then reported "object gone"
# having itself caused the loss — a self-inflicted failure, and the arm fell back to
# the uncorrected map estimate and closed on air.

class EdgeOfFrameArm(FakeArm):
    """An object near a real frame edge, with the rig's measured pan gain.

    Models an actual image boundary rather than "the last move went the wrong way":
    the object has a pixel column, the frame is FRAME_W wide, and it is invisible when
    its column falls outside. With du_dpan = -34 px/deg (measured on the rig) a 3 deg
    probe swings it ~100 px, so from 620 px one probe direction ejects it off the right
    edge and the other brings it safely inward. That is the real situation.
    """

    FRAME_W = 640.0

    def __init__(self, *, du_dpan=110.0, err_u=150.0, **kw):
        super().__init__(du_dpan=du_dpan, err_u=err_u, **kw)

    def track(self, tries=3):
        tr = super().track(tries)
        if tr is None:
            return None
        return None if not (0.0 <= tr.uv[0] <= self.FRAME_W) else tr


def test_a_probe_that_loses_the_object_retries_the_other_direction():
    """Object at u=590 in a 640px frame. The +3 probe throws it to 920 (off the right
    edge); the -3 probe brings it to 260, safely inward."""
    arm = EdgeOfFrameArm()
    res = center_on_object(arm, AIM, ApproachConfig())
    assert any("other way" in m for m in arm.log), f"no retry happened: {arm.log}"
    assert res.centered, f"should have recovered; log={arm.log}"


def test_the_retry_works_whichever_edge_the_object_is_near():
    """Mirror image: object near the LEFT edge, so the ejecting direction flips."""
    arm = EdgeOfFrameArm(du_dpan=-110.0, err_u=-150.0)   # u=290, +3 probe -> -40
    res = center_on_object(arm, AIM, ApproachConfig())
    assert any("other way" in m for m in arm.log), f"no retry happened: {arm.log}"
    assert res.centered, f"log={arm.log}"


def test_an_object_never_visible_is_reported_not_retried_forever():
    arm = FakeArm(lost_after=1)
    res = center_on_object(arm, AIM, ApproachConfig())
    assert not res.centered
    assert res.reason in ("pan_probe_failed", "not_in_view", "lost"), res.reason


def test_a_centred_object_far_from_any_edge_needs_no_retry():
    """The retry must not fire on the ordinary case — it costs two extra moves."""
    arm = EdgeOfFrameArm(du_dpan=-34.0, err_u=40.0)      # u=480, nowhere near an edge
    center_on_object(arm, AIM, ApproachConfig())
    assert not any("other way" in m for m in arm.log), arm.log
