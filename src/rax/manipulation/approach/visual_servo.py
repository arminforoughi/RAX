"""FPV visual servo: follow what the camera sees NOW, and stop when it stops seeing it.

WHY THIS EXISTS, and what it refuses to do.

The approach this replaces was MAP-FIRST. It localized the object once, wrote a 3D point
into a world map, drove at that point in staged hops, and treated each fresh look as an
optional correction to a belief it already held. Every failure had the same shape —
**the belief outranked the picture**: a re-measure that disagreed by more than a bound
was discarded as "a different object"; an accepted correction was applied at a fraction
and had its range thrown away; a stage that failed to re-detect drove on regardless; and
when the final centring reported the object had NEVER BEEN IN VIEW, the caller fell back
to the mapped position and closed the jaws there. That is how the arm came to descend on
a black box during a pick for a green cube, with a saved frame proving the camera was
looking at no green whatsoever.

THE RULE: **NO SIGHTING, NO MOTION.** There is no stored 3D target to fall back to
because there is no stored 3D target at all. The box in the current frame is the only
thing that steers, and losing it ends the approach instead of consulting memory. That is
structural, not a check — see ``test_the_state_carries_no_fallback_position``.

HOW IT STEERS. One error vector and one linear solve:

    e  = (aim_column - u,  aim_row - v,  log target_size - log size)
    dq = J+ e          J measured by moving each axis and watching   (jacobian.py)

Note what is NOT in that: no camera intrinsics, no hand-eye transform, no URDF, no
inverse kinematics, no table plane, no joint names. The servo does not know which
actuator is a shoulder; it learns what each one does to the picture. Any arm that can
report and command its actuators, with any camera attached to it, can be driven by this
— see ``servo_arm.py`` for the whole seam.

WHAT SOLVING ALL THREE TOGETHER FIXED. The hand-tuned predecessor ran two independent
scalar gains — one joint for horizontal, one for vertical — plus a separate "advance"
phase. Two measured failures came straight out of that split, and neither is
representable here:

  * Centring and closing range undid each other. Each advance knocked the aim off, the
    gaze spent eight ticks recovering it, and the box height sat at 82px across three
    advances — the range never closed at all. Rows of one solve cannot fight; they are
    minimised together.
  * Inside ~20cm the vertical loop went unstable, the error GROWING while the correction
    was applied (-70 -> -86 -> -111px at increasing wrist pitch). The wrist is not a
    pure aiming joint: the camera sits ~10cm off its axis, so close in, pitching down
    swings the camera backward and upward more than the rotation gains in aim, and the
    effective gain reverses. No fixed scalar gain survives that. A Jacobian corrected
    from the arm's own motion every tick tracks it — see ``ImageJacobian.updated``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from rax.manipulation.approach.jacobian import ImageJacobian
from rax.manipulation.approach.servo_arm import (
    Axis, Sighting, feature_error, features)

__all__ = [
    "ServoConfig", "ServoState", "Command", "Sighting",
    "MOVE", "ARRIVED", "ABORT", "HOLD",
    "begin", "step",
]

#: What to do with this tick. ``HOLD`` is a real outcome, not a placeholder: it is what
#: a blind tick returns, and the whole point is that the arm does not move on it.
MOVE = "move"
ARRIVED = "arrived"
ABORT = "abort"
HOLD = "hold"


@dataclass(frozen=True)
class ServoConfig:
    """Where the target should end up, and when to give up.

    THERE ARE NO GAINS HERE. Everything that depends on how this particular arm is
    built lives in the measured Jacobian, which is the point: these fields describe the
    GOAL, not the robot, so they transfer between arms and the robot-specific half is
    measured rather than typed in.
    """

    #: Where the target should sit in the image at HANDOFF — and note this is NOT the
    #: fingertip pixel, deliberately.
    #:
    #: It is tempting to aim at the fingertip, since that is where the object must end
    #: up to be grasped. But a table object only projects onto the fingertip when the
    #: gripper is already ON it, and the servo hands off BEFORE contact, so the two
    #: conditions are contradictory and asking for both produces an oscillation that
    #: never converges. Measured on this rig against the fingertip row (v=394): the
    #: servo held a steady -45 to -63px, the object bottoming out near v=275, and spent
    #: its whole tick budget pitching down and recovering at 103-108px of a 110px
    #: target. The predecessor hit the same wall from the other side, logging "object
    #: not in view" from its centring step on EVERY pick.
    #:
    #: So this is where the object ACTUALLY ends up when the arm is as close as it can
    #: get — measurable in one approach, and reachable by construction. The last
    #: centimetres are the grasp's job, from a live measurement taken here.
    aim_u: float = 440.0
    #: The rows the object must stay between. A CONSTRAINT, not a target —
    #: see feature_error. Wide, because anywhere in the frame is fine as long
    #: as the detector keeps hold of it; the band only stops the approach from
    #: walking the object out of the picture.
    view_band: tuple[float, float] = (90.0, 430.0)
    #: Set for the FINAL DESCENT only: the row the object must be driven ONTO — the
    #: real fingertip row. Unreachable while travelling (an object only projects there
    #: when the gripper is nearly on it), which is why it is None for the approach and a
    #: target for the descent. See feature_error.
    aim_v: float | None = None
    #: How tall the target's box is at handoff. This is the RANGE gate expressed in the
    #: thing actually measured — pixels — rather than a distance inferred by dividing
    #: by a guessed object size. MEASURED: the approach plateaus around 108px for a 5cm
    #: cube on this arm, so 100 fires with margin.
    target_height_px: float = 100.0

    #: Aligned enough to hand off, in pixels...
    tol_px: float = 22.0
    #: ...and in log-size, so 0.12 is "within about 12% of the handoff range".
    tol_log_size: float = 0.12

    #: THE LOOP GAIN: what fraction of the computed correction to actually apply.
    #:
    #: This is not the same thing as `damping`, and conflating them cost a live run.
    #: Levenberg damping regularises the SOLVE near a singular pose; it does not close
    #: the loop gently. With the Marquardt term alone the servo takes ~89% of the full
    #: Newton step every tick, and against a real arm — settling time, backlash, an
    #: approximate Jacobian — that overshoots and rings. Measured: the column error went
    #: -114, -92, -44, -8, +19, +18, +4, -24, -36, -24, -2, +24, +22, +2, -40, -49,
    #: with pan alternating +5.00 / -5.00 at the clamp, and it never converged.
    #:
    #: The predecessor had this factor (0.5, folded into its measured scalar gains) and
    #: was stable. Replacing the gains with a least-squares solve quietly dropped it.
    step_gain: float = 0.45
    #: Levenberg damping on the solve — stability near a singular pose, not a gain.
    damping: float = 0.35
    #: Relative weight of the three feature rows. Pixels run to hundreds; log-size is a
    #: ratio where 0.1 is already 10% of range. Without this the size row is
    #: numerically invisible and the servo centres perfectly while never closing in,
    #: which is precisely how the two-phase predecessor failed.
    weights: tuple[float, float, float] = (1.0, 1.0, 60.0)
    #: When set, alignment is solved FIRST and range is closed only in the joint
    #: freedom left over, instead of the two being weighted against each other. Used by
    #: the final descent, where they otherwise fight. See solve_prioritised.
    prioritise_aim: bool = False
    #: Secondary-task gain for that null-space range closing.
    size_gain: float = 0.5
    #: How much of the Broyden correction to accept per tick.
    broyden_rate: float = 0.5

    #: Consecutive blind ticks tolerated. Ticks are ~0.3s once the arm actually settles,
    #: so this is a patience in seconds in disguise: a dropout is normal, a sustained
    #: one means the object is gone and there is nothing honest left to steer on.
    max_blind_ticks: int = 8
    #: Motion the Jacobian does NOT explain, beyond which the lock is a different
    #: object. Measured against the PREDICTION, so the servo's own successful
    #: correction is never mistaken for the target teleporting — which is exactly what
    #: aborted a converging approach on the first live run.
    max_unexplained_px: float = 130.0
    #: A box that changes size by more than this in one tick is not what we were
    #: following.
    max_scale_ratio: float = 2.5

    #: Ticks with no improvement before the servo is declared finished. The joints have
    #: a deadband — measured, sub-degree commands move nothing at all — so "converged"
    #: and "stopped improving" are the same event on real hardware.
    stall_ticks: int = 8
    #: An error must shrink by this much to count as improvement, so detector jitter
    #: cannot look like progress forever.
    improve_px: float = 1.5
    #: Total ticks before giving up.
    max_ticks: int = 140


@dataclass(frozen=True)
class ServoState:
    """What the loop remembers.

    Note what is NOT here: any 3D point, any map tag, any "original estimate". There is
    nothing a lost detection could fall back onto, which is what makes the rule in the
    module docstring structural rather than a check.

    ``jac`` is a model of MOTION, not a position: it says what happens if the arm moves.
    It cannot be steered at, and with no sighting it is never consulted.
    """

    jac: ImageJacobian
    lock: Sighting | None = None
    blind: int = 0
    ticks: int = 0
    #: The last commanded step and the features it was issued from — the two things
    #: needed to correct the Jacobian and to predict where the box should have gone.
    last_dq: np.ndarray | None = None
    last_feat: np.ndarray | None = None
    best_err: float = float("inf")
    since_improve: int = 0


@dataclass(frozen=True)
class Command:
    """What to do with this tick, and why. ``dq`` is per-axis, in the arm's own units."""

    kind: str
    dq: np.ndarray = field(default_factory=lambda: np.zeros(0))
    reason: str = ""

    @property
    def moves(self) -> bool:
        return self.kind == MOVE and bool(np.any(np.abs(self.dq) > 0))


def begin(jac: ImageJacobian) -> ServoState:
    return ServoState(jac=jac)


def _clamp_dq(dq: np.ndarray, axes: tuple[Axis, ...]) -> np.ndarray:
    """Shrink the step until it fits, PRESERVING ITS DIRECTION.

    This clamped every axis independently, and that is wrong whenever two axes do
    similar things to the picture — which on a real arm is most of the time. Measured on
    the SO-101 through the overhead camera, shoulder_pan moved the gripper (3.53, 3.21)
    px/deg and shoulder_lift (4.15, 3.26): nearly collinear. The least-squares answer for
    a direction they do not span is a large positive of one and a large negative of the
    other, which very nearly cancel. Clip them separately and the cancellation is
    destroyed, so the arm sets off along whichever survived the clip — a direction nobody
    solved for. Live, the first tick moved the gripper AWAY from the target and the error
    grew 180px -> 300px.

    Scaling the whole vector keeps the direction the solve actually asked for and only
    shortens it. With this the same loop converged 182px -> 28px.
    """
    dq = np.asarray(dq, dtype=np.float64)
    over = 0.0
    for d, ax in zip(dq, axes):
        if ax.max_step > 0:
            over = max(over, abs(float(d)) / ax.max_step)
    return dq / over if over > 1.0 else dq


def step(state: ServoState, seen: Sighting | None,
         cfg: ServoConfig) -> tuple[ServoState, Command]:
    """Advance the servo one tick.

    ``seen`` is None when the detector found nothing THIS FRAME. That is the only input
    that matters for the safety property: it can produce HOLD or ABORT, and it can never
    produce motion.
    """
    state = replace(state, ticks=state.ticks + 1)

    if state.ticks > cfg.max_ticks:
        return state, Command(ABORT, reason=(
            f"gave up after {cfg.max_ticks} ticks without converging"))

    # ---- BLIND: the branch the old approach got wrong ------------------------------
    # No "drive on the last estimate", because there is no estimate.
    if seen is None:
        state = replace(state, blind=state.blind + 1, last_dq=None, last_feat=None)
        if state.blind > cfg.max_blind_ticks:
            return state, Command(ABORT, reason=(
                f"lost sight of the target for {state.blind} consecutive frames — "
                f"stopping. NOT falling back to a remembered position: there is none"))
        return state, Command(HOLD, reason=(
            f"nothing detected this frame ({state.blind}/{cfg.max_blind_ticks}) — "
            f"holding station, not moving blind"))

    feat = features(seen)

    # ---- IDENTITY, and learning from the move just made ----------------------------
    if (state.lock is not None and state.last_dq is not None
            and state.last_feat is not None):
        predicted = state.jac.predict(state.last_dq)
        actual = feat - state.last_feat
        unexplained = float(np.hypot(actual[0] - predicted[0], actual[1] - predicted[1]))
        if unexplained > cfg.max_unexplained_px:
            return state, Command(ABORT, reason=(
                f"target changed identity: the box moved {unexplained:.0f}px more than "
                f"the arm's own motion explains (limit {cfg.max_unexplained_px:.0f}) — "
                f"that is a different object, not a fast one"))
        p, n = max(state.lock.height, 1.0), max(seen.height, 1.0)
        ratio = max(p, n) / min(p, n)
        if ratio > cfg.max_scale_ratio:
            return state, Command(ABORT, reason=(
                f"target changed identity: the box changed size {ratio:.1f}x in one "
                f"tick (limit {cfg.max_scale_ratio:.1f})"))
        # That move told us how wrong the model was along it. Correct it, for free.
        state = replace(state, jac=state.jac.updated(state.last_dq, actual,
                                                     rate=cfg.broyden_rate))

    state = replace(state, lock=seen, blind=0)

    # ---- the error, and whether we are done ---------------------------------------
    err = feature_error(seen, cfg.aim_u, cfg.view_band, cfg.target_height_px,
                        aim_v=cfg.aim_v)
    px_err = float(np.hypot(err[0], err[1]))
    size_err = float(err[2])

    # A box cut off at the bottom is SHORTER than the object, so its height under-reads
    # the range exactly when the gripper is closest. Never arrive on it.
    big_enough = size_err <= cfg.tol_log_size and not seen.clipped_bottom
    if px_err <= cfg.tol_px and big_enough:
        return state, Command(ARRIVED, reason=(
            f"on the aim column ({px_err:.0f}px) at handoff size "
            f"({seen.height:.0f}px) — handing off"))

    # PROGRESS IS MEASURED ON THE WHOLE ERROR, not just the pixels.
    #
    # This tracked px_err alone, and on the first live run of the Jacobian servo that
    # was the only thing standing between it and a completed pick: the pixel error
    # converged to 5px in five ticks and then sat there, correctly, while the size error
    # was still closing (0.39 -> 0.34). Watching pixels only, the servo saw eight ticks
    # of "no improvement" and aborted 5px from the aim, blaming an unreachable aim
    # point for what was actually steady progress on the other axis.
    #
    # The thing to watch is the thing the solve minimises: the weighted norm of all
    # three rows. The weights are the same ones the solve uses, so a tick that trades
    # pixels for range reads as progress exactly when the solver thinks it is.
    w = np.asarray(cfg.weights, dtype=np.float64)
    total_err = float(np.linalg.norm(w * err))
    improved = total_err < state.best_err - cfg.improve_px
    state = replace(state,
                    best_err=min(total_err, state.best_err),
                    since_improve=0 if improved else state.since_improve + 1)

    if state.since_improve >= cfg.stall_ticks:
        # The joints have a deadband; on real hardware "stopped improving" IS
        # "converged". Take it when it is good enough, and say what is left when not.
        # A STALL RELAXES THE SIZE TOLERANCE, AND ONLY THE SIZE TOLERANCE.
        #
        # The joints have a deadband, so the last fraction of any error is not
        # commandable, and refusing a good approach because the size gap is 0.13 against
        # a 0.12 tolerance wastes it. Handing off slightly short is safe: the grasp
        # descends onto a position measured HERE, so a few percent of extra range costs
        # nothing.
        #
        # THE AIM TOLERANCE IS NOT RELAXED, because relaxing it is how a bad pick gets
        # dressed up as a good one. "Stopped improving" also fires when the error is
        # OSCILLATING rather than settling — an oscillation never beats its own best —
        # and with the aim relaxed to 2x, the servo accepted a ringing 34px error as
        # "arrived" and handed the grasp a target ~3cm off. It closed on air, and from
        # the outside it looked like the arm had simply gone to the wrong place.
        if (px_err <= cfg.tol_px and size_err <= cfg.tol_log_size * 2.0
                and not seen.clipped_bottom):
            return state, Command(ARRIVED, reason=(
                f"stopped improving {state.since_improve} ticks ago at {px_err:.0f}px "
                f"and {size_err:+.2f} log-size — taking it; the joints have a deadband "
                f"and this arm will not do better"))
        if px_err <= cfg.tol_px and size_err > cfg.tol_log_size:
            return state, Command(ABORT, reason=(
                f"aimed to {px_err:.0f}px but stopped closing range, still "
                f"{size_err:+.2f} in log-size (about {100 * (2.718 ** size_err - 1):.0f}% "
                f"too far). The axes that change range cannot do so from here without "
                f"giving up the aim — stopping rather than grasping short"))
        return state, Command(ABORT, reason=(
            f"stopped improving {state.since_improve} ticks ago, still {px_err:.0f}px "
            f"off the aim and {size_err:+.2f} in log-size. The aim may not be reachable "
            f"from here — stopping rather than grasping at a misalignment it can see"))

    raw = (state.jac.solve_prioritised(err, damping=cfg.damping,
                                       size_gain=cfg.size_gain)
           if cfg.prioritise_aim else
           state.jac.solve(err, damping=cfg.damping, weights=cfg.weights))
    dq = _clamp_dq(cfg.step_gain * raw, state.jac.axes)
    if not np.any(np.abs(dq) > 1e-9):
        return state, Command(ABORT, reason=(
            "the solve asks for no motion while still off target — from this pose the "
            "Jacobian says no axis moves the picture the way it needs to go"))

    state = replace(state, last_dq=dq, last_feat=feat)
    moves = ", ".join(f"{ax.name} {d:+.2f}"
                      for ax, d in zip(state.jac.axes, dq) if abs(d) > 1e-3)
    return state, Command(MOVE, dq=dq, reason=(
        f"off by ({err[0]:+.0f},{err[1]:+.0f})px, size {size_err:+.2f} -> {moves}"))
