"""The FPV servo: the safety invariant, and closing the loop on a simulated arm.

Two things are being pinned here.

THE INVARIANT. A tick with no detection can never return a command that moves the arm,
from any state the loop can reach. This is the bug the module exists to make
unrepresentable: the arm descended on a black box during a pick for a green cube,
because the centring step reported "the object was never in view" and the caller fell
back to a remembered map position.

THE CONTROL LAW. ``SimArm`` below is not decoration — it reproduces the specific
geometry that defeated the hand-tuned predecessor: the wrist is not a pure aiming joint,
and inside about 20cm its effect on the image REVERSES SIGN. Any servo with a fixed
vertical gain diverges there (measured on the real arm: -70 -> -86 -> -111px while
correcting). ``test_it_converges_through_a_gain_reversal`` is the regression test for
that, and it is the reason the Jacobian is measured and then corrected as the arm moves
rather than tuned once.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rax.manipulation.approach.jacobian import (
    ImageJacobian, JacobianError, measure_jacobian)
from rax.manipulation.approach.servo_arm import Axis, Sighting, features
from rax.manipulation.approach.visual_servo import (
    ABORT, ARRIVED, HOLD, MOVE, ServoConfig, ServoState, begin, step)

AIM = (440.0, 275.0)   # (aim column, a row comfortably inside the view band)
CFG = ServoConfig(aim_u=AIM[0])


def sighting(u, v, h, w=None, clipped_bottom=False, frame=(640, 480)):
    w = h if w is None else w
    return Sighting(bbox=(u - w / 2, v - h / 2, u + w / 2, v + h / 2),
                    frame_w=frame[0], frame_h=frame[1], clipped_bottom=clipped_bottom)


# --------------------------------------------------------------------------------
# a simulated arm with the real arm's nasty property
# --------------------------------------------------------------------------------

class SimArm:
    """pan / pitch / reach / lift, with cross-coupling and a pitch gain that reverses.

    The vertical response to PITCH is a rotation term (constant, negative) plus a
    translation term that grows as the range closes, because the camera sits a lever arm
    off the pitch axis. Below ~20cm the translation term wins and the sign flips — which
    is what the SO-101 does, and what a fixed gain cannot survive.

    LIFT is here because of what building this sim exposed. With pitch as the only axis
    touching the vertical, its gain passes through ZERO on its way to reversing, and at
    that range the image row is not controllable by anything — no servo, however clever,
    can fix it, and the honest answer is to stop and say the aim is unreachable. That is
    not a modelling artefact: it is exactly why the real arm stalled 45-63px short with
    the fingertip row unreachable. The cure is an actuator whose vertical effect does
    not vanish, i.e. moving the tip UP AND DOWN rather than rotating the wrist. The sim
    carries one so the tests exercise the geometry the arm should actually be given.
    """

    LEVER = 0.10

    def __init__(self, *, reversal=True, obj_bearing=14.0, obj_elev=-9.0,
                 obj_r=0.42, start=(0.0, 0.0, 0.05, 0.0), noise=0.0, seed=0):
        self.axes = (
            Axis("pan", 0, probe=5.0, max_step=6.0, lo=-100.0, hi=100.0),
            Axis("pitch", 1, probe=5.0, max_step=5.0, lo=-95.0, hi=95.0),
            Axis("reach", 2, probe=0.02, max_step=0.03, lo=0.05, hi=0.40),
            Axis("lift", 3, probe=0.02, max_step=0.03, lo=-0.15, hi=0.25),
        )
        self.q = np.array(start, dtype=np.float64)
        self.obj_bearing, self.obj_elev, self.obj_r = obj_bearing, obj_elev, obj_r
        self.k_rev = 18.0 if reversal else 0.0
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.moves = 0
        self.visible = True

    # --- the seam -----------------------------------------------------------------
    def actuators(self):
        return self.q.copy()

    def apply(self, q):
        q = np.asarray(q, dtype=np.float64)
        for ax in self.axes:
            if not (ax.lo - 1e-9 <= q[ax.index] <= ax.hi + 1e-9):
                return False
        self.q = q.copy()
        self.moves += 1
        return True

    def sense(self):
        if not self.visible:
            return None
        u, v, h = self._uvh()
        if self.noise:
            u += self.rng.normal(0, self.noise)
            v += self.rng.normal(0, self.noise)
        return sighting(u, v, h)

    # --- the "physics" ------------------------------------------------------------
    def range_m(self):
        return max(self.obj_r - self.q[2], 0.03)

    def _uvh(self):
        pan, pitch, _reach, lift = self.q
        rng = self.range_m()
        u = 320.0 + (self.obj_bearing - pan) * 9.0 + pitch * 0.8      # cross-coupled
        dv_rot = (self.obj_elev - pitch) * 9.0
        dv_trans = self.k_rev * self.LEVER / rng * pitch
        v = 240.0 + dv_rot + dv_trans - 900.0 * lift
        h = 16.0 / rng
        return u, v, h


def converge(arm, cfg, jac=None, max_ticks=None):
    """Run the loop the way the server does. Returns (kind, ticks, last_command)."""
    jac = jac if jac is not None else measure_jacobian(arm)
    s = begin(jac)
    last = None
    for _ in range(max_ticks or cfg.max_ticks + 5):
        seen = arm.sense()
        s, cmd = step(s, seen, cfg)
        last = cmd
        if cmd.kind in (ARRIVED, ABORT):
            return cmd.kind, s.ticks, cmd
        if cmd.kind == MOVE:
            q = arm.actuators()
            for ax, d in zip(arm.axes, cmd.dq):
                q[ax.index] = float(np.clip(q[ax.index] + d, ax.lo, ax.hi))
            arm.apply(q)
    return "ranout", s.ticks, last


def _any_jac():
    return measure_jacobian(SimArm())


# --------------------------------------------------------------------------------
# THE INVARIANT
# --------------------------------------------------------------------------------

def test_a_blind_tick_can_never_move_the_arm():
    """No detection => no motion. From every state the loop can be in."""
    jac = _any_jac()
    lock = sighting(AIM[0], AIM[1], 60)
    states = [
        begin(jac),
        ServoState(jac=jac, lock=lock, blind=0, ticks=1),
        ServoState(jac=jac, lock=lock, blind=CFG.max_blind_ticks, ticks=9),
        ServoState(jac=jac, lock=lock, blind=3, ticks=40,
                   last_dq=np.array([1.0, 1.0, 0.01, 0.0]),
                   last_feat=features(lock)),
        ServoState(jac=jac, lock=sighting(10, 10, 20), blind=0,
                   ticks=CFG.max_ticks - 1),
    ]
    for s in states:
        _, cmd = step(s, None, CFG)
        assert not cmd.moves, f"blind tick returned motion from {s}"
        assert cmd.kind in (HOLD, ABORT)
        assert not np.any(np.abs(cmd.dq) > 0)


def test_the_state_carries_no_fallback_position():
    """Nothing in the state a lost detection could steer on.

    Structural, not stylistic: the previous approach failed *because* it had a
    ``cube_xy`` to fall back to. ``jac`` is a model of motion, not a place.
    """
    names = {f.name for f in dataclasses.fields(ServoState)}
    assert names == {"jac", "lock", "blind", "ticks", "last_dq", "last_feat",
                     "best_err", "since_improve"}
    for banned in ("xy", "target", "point", "map", "estimate", "p_obj", "cube", "pos"):
        assert not any(banned in n for n in names), f"state gained a fallback: {banned}"


def test_a_short_dropout_holds_then_recovers():
    s = begin(_any_jac())
    s, _ = step(s, sighting(300, 200, 50), CFG)
    for i in range(CFG.max_blind_ticks):
        s, cmd = step(s, None, CFG)
        assert cmd.kind == HOLD and s.blind == i + 1
    s, cmd = step(s, sighting(300, 200, 50), CFG)
    assert cmd.kind != ABORT and s.blind == 0


def test_a_sustained_dropout_aborts_and_says_it_is_not_using_memory():
    s = begin(_any_jac())
    s, _ = step(s, sighting(300, 200, 50), CFG)
    for _ in range(CFG.max_blind_ticks):
        s, cmd = step(s, None, CFG)
    s, cmd = step(s, None, CFG)
    assert cmd.kind == ABORT
    assert "lost sight" in cmd.reason and "remembered" in cmd.reason


# --------------------------------------------------------------------------------
# identity, measured against what the arm's own motion explains
# --------------------------------------------------------------------------------

def test_motion_the_jacobian_explains_is_not_a_new_object():
    """The servo's own successful correction must not read as a teleport — that
    aborted a converging approach on the first live run."""
    arm = SimArm()
    kind, _ticks, cmd = converge(arm, CFG)
    assert kind != ABORT or "identity" not in cmd.reason, cmd.reason


def test_a_genuine_teleport_is_still_caught():
    s = begin(_any_jac())
    s, cmd = step(s, sighting(300, 200, 50), CFG)
    assert cmd.kind == MOVE
    s, cmd = step(s, sighting(700, 500, 50), CFG)
    assert cmd.kind == ABORT and "changed identity" in cmd.reason
    assert "different object" in cmd.reason


def test_a_box_that_jumps_in_size_is_a_different_object():
    s = begin(_any_jac())
    s, _ = step(s, sighting(300, 200, 40), CFG)
    s, cmd = step(s, sighting(302, 201, 160), CFG)
    assert cmd.kind == ABORT and "changed identity" in cmd.reason


# --------------------------------------------------------------------------------
# the Jacobian itself
# --------------------------------------------------------------------------------

def test_the_jacobian_matches_a_finite_difference_of_the_sim():
    arm = SimArm()
    jac = measure_jacobian(arm)
    q0 = arm.actuators()
    for k, ax in enumerate(arm.axes):
        q = q0.copy()
        q[ax.index] += ax.probe * 0.5
        arm.apply(q)
        f1 = features(arm.sense())
        arm.apply(q0)
        f0 = features(arm.sense())
        got = (f1 - f0) / (ax.probe * 0.5)
        assert np.allclose(got[:2], jac.J[:2, k], atol=1.5), (
            f"{ax.name}: measured {jac.J[:2, k]} vs local {got[:2]}")


def test_measuring_without_the_target_in_view_raises_rather_than_guessing():
    arm = SimArm()
    arm.visible = False
    with pytest.raises(JacobianError, match="not in view"):
        measure_jacobian(arm)


def test_probing_returns_the_arm_to_where_it_started():
    arm = SimArm()
    q0 = arm.actuators()
    measure_jacobian(arm)
    assert np.allclose(arm.actuators(), q0)


def test_a_broyden_update_moves_the_model_toward_the_truth():
    arm = SimArm()
    jac = measure_jacobian(arm)
    wrong = ImageJacobian(J=jac.J * -0.5, axes=jac.axes,
                          probe_response_px=jac.probe_response_px)
    dq = np.array([2.0, 0.0, 0.0, 0.0])
    truth = jac.predict(dq)
    before = float(np.linalg.norm(wrong.predict(dq) - truth))
    after = float(np.linalg.norm(wrong.updated(dq, truth).predict(dq) - truth))
    assert after < before


# --------------------------------------------------------------------------------
# closing the loop
# --------------------------------------------------------------------------------

def test_it_converges_on_a_well_behaved_arm():
    arm = SimArm(reversal=False)
    kind, ticks, cmd = converge(arm, CFG)
    assert kind == ARRIVED, f"{kind} after {ticks}: {cmd.reason}"


def test_it_converges_through_a_gain_reversal():
    """THE REGRESSION TEST FOR THE REAL FAILURE.

    Inside ~20cm the pitch axis's effect on the image reverses sign. The hand-tuned
    predecessor diverged there, the vertical error growing while it corrected
    (-70 -> -86 -> -111px). A measured Jacobian corrected from the arm's own motion has
    to follow the sign through and still arrive.
    """
    arm = SimArm(reversal=True)
    kind, ticks, cmd = converge(arm, CFG)
    assert kind == ARRIVED, f"{kind} after {ticks}: {cmd.reason}"
    assert arm.range_m() < 0.20, "should have closed through the sign change"


def test_it_converges_with_a_noisy_detector():
    arm = SimArm(reversal=True, noise=2.0, seed=7)
    kind, ticks, cmd = converge(arm, CFG)
    assert kind == ARRIVED, f"{kind} after {ticks}: {cmd.reason}"


def test_it_converges_from_several_starting_offsets():
    for bearing in (-22.0, -8.0, 6.0, 20.0):
        arm = SimArm(reversal=True, obj_bearing=bearing)
        kind, ticks, cmd = converge(arm, CFG)
        assert kind == ARRIVED, f"bearing {bearing}: {kind} after {ticks}: {cmd.reason}"


def test_a_stale_jacobian_measured_elsewhere_still_gets_there():
    """Broyden has to rescue a model measured at the wrong pose — otherwise the
    Jacobian would need re-probing every few centimetres."""
    ref = measure_jacobian(SimArm(reversal=True, start=(0.0, 0.0, 0.35, 0.0)))
    arm = SimArm(reversal=True, start=(0.0, 0.0, 0.20, 0.0))
    kind, ticks, cmd = converge(arm, CFG, jac=ref)
    assert kind == ARRIVED, f"{kind} after {ticks}: {cmd.reason}"


# --------------------------------------------------------------------------------
# arrival
# --------------------------------------------------------------------------------

def test_arrival_needs_both_alignment_and_size():
    jac = _any_jac()
    _, cmd = step(begin(jac), sighting(AIM[0], AIM[1], CFG.target_height_px), CFG)
    assert cmd.kind == ARRIVED
    _, cmd = step(begin(jac), sighting(AIM[0], AIM[1], 30), CFG)
    assert cmd.kind == MOVE, "on the pixel but far away is not arrival"
    _, cmd = step(begin(jac), sighting(AIM[0] + 180, AIM[1], CFG.target_height_px), CFG)
    assert cmd.kind == MOVE, "big but not aimed is not arrival"


def test_a_bottom_clipped_box_never_counts_as_arrival():
    """Cut off at the bottom, the box is shorter than the object, so its height
    under-reads the range exactly when the gripper is closest."""
    _, cmd = step(begin(_any_jac()),
                  sighting(AIM[0], AIM[1], CFG.target_height_px + 40,
                           clipped_bottom=True), CFG)
    assert cmd.kind != ARRIVED


def test_an_unreachable_aim_stops_and_says_so_rather_than_grinding():
    """A servo that cannot get there must report what is left, not oscillate.

    This is how the fingertip row was found to be unreachable in the first place.
    """
    s = begin(_any_jac())
    stuck = sighting(AIM[0] + 150, AIM[1], 40)
    kinds = []
    for _ in range(CFG.stall_ticks + 6):
        s, cmd = step(s, stuck, CFG)     # the picture never changes: nothing is working
        kinds.append(cmd.kind)
        if cmd.kind == ABORT:
            assert "stopped improving" in cmd.reason
            assert "may not be reachable" in cmd.reason
            assert not cmd.moves
            return
    raise AssertionError(f"never gave up: {kinds}")


def test_it_gives_up_rather_than_servoing_forever():
    s = ServoState(jac=_any_jac(), lock=sighting(AIM[0], AIM[1], 60),
                   ticks=CFG.max_ticks)
    _, cmd = step(s, sighting(AIM[0] + 100, AIM[1], 60), CFG)
    assert cmd.kind == ABORT and "without converging" in cmd.reason


def test_the_sim_really_does_reverse_its_pitch_gain():
    """Guard the regression test's premise: if the sim stops reversing, the test above
    stops testing anything."""
    far = SimArm(reversal=True, start=(0.0, 0.0, 0.05, 0.0))
    near = SimArm(reversal=True, start=(0.0, 0.0, 0.30, 0.0))
    def dv_dpitch(arm):
        j = measure_jacobian(arm)
        return j.J[1, 1]
    a, b = dv_dpitch(far), dv_dpitch(near)
    assert a * b < 0, f"pitch gain did not change sign: {a:.2f} then {b:.2f}"


def test_a_stall_relaxes_the_size_tolerance_too():
    """Measured: the servo closed the size gap to 0.13 against a 0.12 tolerance and
    then stalled on the joint deadband. Refusing a good approach by 0.01 wastes it, and
    the grasp descends onto a position measured at the handoff, not onto the box."""
    cfg = ServoConfig(aim_u=AIM[0])
    just_short = sighting(AIM[0], AIM[1],
                          cfg.target_height_px * 0.88)     # ~0.13 in log-size
    s = begin(_any_jac())
    for _ in range(cfg.stall_ticks + 6):
        s, cmd = step(s, just_short, cfg)
        if cmd.kind == ARRIVED:
            assert "deadband" in cmd.reason
            return
        assert cmd.kind != ABORT, cmd.reason
    raise AssertionError("a stall just short of the size tolerance should be taken")


# --------------------------------------------------------------------------------
# it has to actually settle, not ring — and never call a ring "arrived"
# --------------------------------------------------------------------------------

def test_the_approach_settles_instead_of_oscillating():
    """Measured on the real arm with no loop gain: the column error went
    -114, -92, -44, -8, +19, +18, +4, -24, -36, -24, -2, +24, +22, +2, -40, -49,
    with pan alternating +5.00/-5.00 at the clamp. It never converged, and the stall
    rule then declared it arrived 34px off. Count the sign changes."""
    arm = SimArm(reversal=True)
    jac = measure_jacobian(arm)
    s = begin(jac)
    errs = []
    for _ in range(CFG.max_ticks):
        seen = arm.sense()
        s, cmd = step(s, seen, CFG)
        if cmd.kind in (ARRIVED, ABORT):
            break
        errs.append(CFG.aim_u - seen.cx)
        q = arm.actuators()
        for ax, d in zip(arm.axes, cmd.dq):
            q[ax.index] = float(np.clip(q[ax.index] + d, ax.lo, ax.hi))
        arm.apply(q)
    tail = errs[3:]
    flips = sum(1 for a, b in zip(tail, tail[1:]) if a * b < 0)
    assert flips <= 3, f"column error is ringing ({flips} sign changes): {tail}"


def test_a_ringing_error_is_never_accepted_as_arrival():
    """A stall relaxes the SIZE tolerance, never the aim: "stopped improving" fires on
    an oscillation too, and accepting one hands the grasp a target centimetres off."""
    cfg = ServoConfig(aim_u=AIM[0])
    s = begin(_any_jac())
    swing = cfg.tol_px * 1.6            # inside the old 2x relaxation, outside tol
    for i in range(cfg.stall_ticks + 8):
        u = AIM[0] + (swing if i % 2 else -swing)
        s, cmd = step(s, sighting(u, AIM[1], cfg.target_height_px), cfg)
        assert cmd.kind != ARRIVED, (
            f"accepted a ringing {swing:.0f}px error as arrival: {cmd.reason}")
        if cmd.kind == ABORT:
            return
    raise AssertionError("a ringing servo should abort, not run forever")


def test_the_loop_gain_actually_damps_the_step():
    cfg_hot = ServoConfig(aim_u=AIM[0], step_gain=1.0)
    cfg_cool = ServoConfig(aim_u=AIM[0], step_gain=0.45)
    jac = _any_jac()
    # A SMALL error on purpose: once the raw step exceeds the per-axis limits, the
    # direction-preserving clamp scales both gains down to the same magnitude, so the
    # gain is only observable while the step still fits.
    seen = sighting(AIM[0] - 30, AIM[1], CFG.target_height_px * 0.9)
    _, hot = step(begin(jac), seen, cfg_hot)
    _, cool = step(begin(jac), seen, cfg_cool)
    assert np.linalg.norm(cool.dq) < np.linalg.norm(hot.dq)


def test_the_descent_drives_the_row_onto_the_fingertip():
    """With aim_v set the row becomes a target; without it the row is free.

    The descent that only drove column and size never lowered the gripper at all —
    measured, size 0.77 -> 0.74 with the tip parked at z=6.9cm.
    """
    free = ServoConfig(aim_u=AIM[0])                       # approach: row is a band
    targeted = ServoConfig(aim_u=AIM[0], aim_v=394.0)      # descent: row is a target
    seen = sighting(AIM[0], 200.0, 60)                     # inside the view band
    _, cmd_free = step(begin(_any_jac()), seen, free)
    _, cmd_aim = step(begin(_any_jac()), seen, targeted)
    assert "+0)px" in cmd_free.reason, "a row inside the band must contribute no error"
    assert "+194)px" in cmd_aim.reason, "a targeted row must pull toward the fingertip"
    assert np.linalg.norm(cmd_aim.dq) > np.linalg.norm(cmd_free.dq)


def test_closing_range_does_not_disturb_the_aim():
    """The null-space secondary must move toward the object WITHOUT moving the box.

    Weighted against each other the two jobs fight: heavy size weight retracts the arm,
    zero size weight aligns perfectly and stops 4x too far away (measured: aim (-3,+10)px
    with size error 1.46).
    """
    jac = _any_jac()
    err = np.array([0.0, 0.0, 0.8])          # perfectly aimed, far too far away
    dq = jac.solve_prioritised(err, size_gain=0.5)
    moved = jac.predict(dq)
    assert abs(moved[0]) < 1.0 and abs(moved[1]) < 1.0, (
        f"closing range shifted the box by {moved[:2]} px — it must not")
    assert moved[2] > 0.0, "it must actually get closer"


def test_the_prioritised_solve_still_fixes_a_real_aim_error():
    jac = _any_jac()
    err = np.array([60.0, -40.0, 0.5])
    moved = jac.predict(jac.solve_prioritised(err, size_gain=0.5))
    assert moved[0] > 0 and moved[1] < 0, "the aim must move toward the target"


def test_the_step_clamp_preserves_direction():
    """Clamping each axis separately destroys a solve that relies on two axes
    cancelling — and on a real arm two axes usually do nearly the same thing.

    Measured through the overhead camera: shoulder_pan (3.53,3.21) px/deg and
    shoulder_lift (4.15,3.26). Independent clipping sent the arm the wrong way and the
    error grew 180px -> 300px.
    """
    from rax.manipulation.approach.visual_servo import _clamp_dq
    axes = (Axis("a", 0, probe=4.0, max_step=5.0),
            Axis("b", 1, probe=4.0, max_step=3.0))
    big = np.array([40.0, -30.0])
    out = _clamp_dq(big, axes)
    assert np.allclose(out / np.linalg.norm(out), big / np.linalg.norm(big)), (
        "clamping must not rotate the step")
    assert abs(out[0]) <= 5.0 + 1e-9 and abs(out[1]) <= 3.0 + 1e-9
    # a step already inside the limits is untouched
    small = np.array([1.0, -0.5])
    assert np.allclose(_clamp_dq(small, axes), small)
