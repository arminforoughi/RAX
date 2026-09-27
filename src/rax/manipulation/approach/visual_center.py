"""Final centring by eye: put the object under the jaws before descending.

The staged approach gets the gripper close using the mapped position, but that position
carries whatever error the localization had. The last correction has to come from the
picture, because the picture is the only thing that sees both the object and the jaws at
once.

**Decoupled single-DOF servos, not a Cartesian Jacobian.**

    horizontal pixel error  ->  rotate the base
    vertical pixel error    ->  reach radially in / out

Each axis is one joint and monotonic, so its sign is trivial to measure and the loop is
stable. A 2x2 Cartesian Jacobian was tried first and oscillated — 109 -> 96 -> 119 px —
because base rotation, reach, and an auto-swept wrist pitch all mixed into it. Pitch is
held fixed here.

**The gains are measured, not modelled.** Before servoing, the arm makes one small probe
move on each axis and measures how many pixels the object shifted. That is the honest
way to get a sign through an unknown mount: a hand-eye transform with roll or yaw error
can invert the raw pixel signs, so a hand-derived gain can drive the loop the wrong way
and stall. Probing costs two small moves and removes the whole class of problem.

**And they keep being measured.** A probe is a local linearization: it is taken over
3 deg and then used to size every correction after it. Over that range base rotation is
not linear, and the probe overestimated the gain 3x on a real rig. Each corrective move
is itself a probe with a known command and an observed response, so the gains are
re-estimated from it as the loop runs (see :func:`_update_gain`). The probe now only has
to be right about the SIGN and the order of magnitude; the loop finds the rest. This
governs the endgame, where the steps are small enough not to be clipped — see the note
on the adaptation constants for what it does not fix.

The arm is reached through :class:`CenteringOps`, a deliberately small interface, so
this works for any arm that can rotate its base, reach radially, and report where its
tip is.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["CenteringOps", "CenteringResult", "center_on_object"]

# Small, slow probe moves so the measurement is clean and the arm does not jerk.
PAN_PROBE_DEG = 3.0
RAD_PROBE_M = 0.015

# Below this range the arm is "close" and everything slows down.
CLOSE_RANGE_M = 0.12

# A base rotation that barely moves the object in the image cannot be servoed on —
# the gain is noise and dividing by it produces a wild step.
MIN_DU_DPAN = 3.0

# Below this many degrees the base did not meaningfully turn, so the probe learned
# nothing about the gain and must not divide by it. Distinct from MIN_DU_DPAN: that
# one rejects "the base turned but the object did not move", this one rejects "the
# base never turned at all" - a different fault with a different remedy.
MIN_PROBE_PAN_DEG = 0.75

# --- online gain adaptation ------------------------------------------------------
# The probe measures each gain ONCE, with a 3 deg / 1.5 cm perturbation, and the loop
# then divides by that number for every correction after it. Measured on a real pick:
# the probe reported 12.8 px/deg where the corrective steps realized about 4. Base
# rotation at 15 cm range is simply not linear over 20 deg.
#
# Every corrective move is itself a probe: it commanded a known amount and the next
# frame says what happened. Folding that back in costs nothing, needs no extra motion,
# and converges the gain in about two iterations. This is the same idea as the probe —
# measure, do not model — carried into the loop instead of stopping at its threshold.
#
# WHAT THIS DOES NOT FIX, so nobody re-derives it from the log and over-credits it:
# while the computed step exceeds max_pan it is CLIPPED, and a clipped step is the same
# step whatever the gain says — at the 268 px that opened a real pick, 268/12.8 and
# 268/4 both clip to 4.5 deg. The gain governs only once the computed step falls under
# the clip, i.e. the endgame closing on the tolerance, where an over-reporting gain
# makes every remaining step a fraction of what it should be and the loop creeps and
# stalls just outside tolerance. The clipped OPENING is fixed by cfg.align_iters and by
# decaying the approach trim (see geometry.stage_trim), not by this.
#
# The update is damped and clamped rather than taken whole: one detection outlier must
# not be able to rewrite the gain, and a gain that flips sign mid-loop is far more
# likely to be a bad read than a real inversion (the probe already established the sign
# through whatever the hand-eye is doing).
GAIN_DAMPING = 0.6          # fraction of the way to the newly measured gain
GAIN_MAX_RATIO = 4.0        # one update may not change a gain by more than this factor
MIN_PAN_FOR_GAIN_DEG = 1.0  # below this the pixel delta is mostly detector noise
MIN_RAD_FOR_GAIN_M = 0.004


def _update_gain(old, delta_px, delta_cmd, min_cmd, say=None, what=""):
    """Fold one commanded move and its observed pixel response into a gain estimate.

    Returns the updated gain, or ``old`` when this move cannot inform it. Attribution
    is decoupled — all of ``delta_px`` is credited to ``delta_cmd`` — which is exactly
    the assumption the control law already makes, so the estimate stays consistent with
    the thing it is steering. Cross-coupling shows up as noise, and the damping is what
    absorbs it.
    """
    if old is None or abs(delta_cmd) < min_cmd:
        return old
    measured = delta_px / delta_cmd
    if measured == 0.0 or (measured > 0.0) != (old > 0.0):
        # A sign flip is not believable from one frame; keep the probed sign.
        if say:
            say(f"center: ignoring a {what} gain of {measured:.1f} — sign flipped")
        return old
    lo, hi = sorted((old / GAIN_MAX_RATIO, old * GAIN_MAX_RATIO))
    measured = float(np.clip(measured, lo, hi))
    return old + (measured - old) * GAIN_DAMPING


@runtime_checkable
class CenteringOps(Protocol):
    """What the servo needs from the arm. Everything else it works out itself."""

    def joints(self) -> np.ndarray:
        """Current joint angles (deg)."""

    def tip(self, q=None) -> np.ndarray:
        """Fingertip position in base frame; current pose when q is None."""

    def track(self, tries: int = 3):
        """Find the object now. Returns something with a ``.uv``, or None."""

    def range_m(self, track) -> float:
        """Rough camera-to-object range, used only to pick a speed."""

    def move_pan(self, delta_deg: float, *, settle: float, step: float):
        """Rotate the base by delta_deg, settle, and return the degrees ACHIEVED.

        The achieved amount differs from the requested one when the joint clips at a
        limit or the servo under-travels; the probe divides by it, so reporting it is
        what keeps the measured gain honest. Return None if this arm cannot read its
        own joints back, and the caller falls back to the commanded amount.
        """

    def move_tip(self, p_base, *, settle: float, step: float) -> bool:
        """Move the fingertip to a base-frame point, holding pitch and roll.
        Returns False if it is unreachable."""

    def say(self, msg: str) -> None: ...

    def checkpoint(self) -> None:
        """Raise if the operator asked to stop."""


class CenteringResult:
    """Where the tip ended up, and whether the servo actually converged."""

    __slots__ = ("xy", "centered", "reason", "iterations")

    def __init__(self, xy, centered: bool, reason: str = "", iterations: int = 0):
        self.xy = xy
        self.centered = centered
        self.reason = reason
        self.iterations = iterations

    def __repr__(self):
        return (f"CenteringResult(xy={self.xy}, centered={self.centered}, "
                f"reason={self.reason!r}, iters={self.iterations})")


def _probe_pan(ops, uv0, settle, step):
    """Pixels of horizontal object motion per degree of base rotation, or None.

    Divides by the pan the base ACTUALLY turned, not the pan it was asked to turn.
    Those differ whenever the joint clips at a limit or the servo under-travels under
    load, and the difference is not benign: a base that moved 0.5 deg instead of 3
    reports a gain six times too small, which either trips the MIN_DU_DPAN gate (the
    servo gives up on an object it could have centred) or, worse, survives it and makes
    every later correction six times too large.

    The probe is also the most dangerous move in the loop, because it is the only one
    made BLIND -- the gain it is measuring is the very thing that says how far 3 deg
    will shove the object across the frame. Measured on this rig at ~34 px/deg, the
    3 deg probe moves the object ~100 px; started near a frame edge, that ejects it and
    the servo reports "object gone" having itself thrown the object out of view. So a
    probe that loses the object is retried the other way before giving up: the opposite
    direction moves it back across the frame rather than further out.

    ``move_pan`` returns the achieved degrees; an ops layer that cannot measure its own
    joints returns None and we fall back to the commanded amount, which is old behaviour.
    """
    def attempt(direction):
        """Probe one way. -> (gain | None, moved_deg, saw_object). Leaves the base
        where it started."""
        moved = ops.move_pan(direction * PAN_PROBE_DEG, settle=settle, step=step)
        moved = direction * PAN_PROBE_DEG if moved is None else float(moved)
        if abs(moved) < MIN_PROBE_PAN_DEG:
            ops.move_pan(-moved, settle=settle, step=step)
            return None, moved, True          # blocked, not lost
        tr = ops.track(tries=4)
        # Undo what actually happened, not what was requested: commanding -3 after a
        # clipped +0.5 walks the base a little further off on every probe.
        ops.move_pan(-moved, settle=settle, step=step)
        if tr is None:
            return None, moved, False
        gain = (float(tr.uv[0]) - uv0[0]) / moved
        return (None if abs(gain) < MIN_DU_DPAN else gain), moved, True

    gain, moved, saw = attempt(+1.0)
    if gain is not None:
        return gain
    if not saw:
        ops.say("center: the probe pushed the object out of view - trying the other way")
    elif abs(moved) < MIN_PROBE_PAN_DEG:
        pass                                   # blocked; the retry below is the point
    else:
        return None                            # it moved, it was seen, it just does not respond
    gain, moved2, saw2 = attempt(-1.0)
    if gain is not None:
        return gain
    if not saw2:
        ops.say("center: the object is not visible from either probe direction")
    elif abs(moved2) < MIN_PROBE_PAN_DEG:
        ops.say(f"center: the base cannot turn here (asked {PAN_PROBE_DEG:.1f}deg either "
                f"way, moved {abs(moved):.2f}/{abs(moved2):.2f}deg)")
    return None


def _probe_radial(ops, uv0, p0, settle, step):
    """Pixels of vertical object motion per metre of radial reach, or None."""
    r0 = float(np.hypot(p0[0], p0[1]))
    u = np.array([p0[0], p0[1]]) / max(r0, 1e-6)
    out = np.array([p0[0] + u[0] * RAD_PROBE_M, p0[1] + u[1] * RAD_PROBE_M, p0[2]])
    if not ops.move_tip(out, settle=settle, step=step):
        return None
    tr = ops.track(tries=4)
    ops.move_tip(np.asarray(p0, dtype=np.float64), settle=settle, step=step)
    if tr is None:
        return None
    return (float(tr.uv[1]) - uv0[1]) / RAD_PROBE_M


def center_on_object(ops: CenteringOps, target_uv, cfg, *, tolerance_px=None
                     ) -> CenteringResult:
    """Servo the arm until the object sits on ``target_uv`` in the image.

    ``target_uv`` is where the object should appear when the jaws are around it —
    normally the measured fingertip pixel plus the configured aim trims.

    ``tolerance_px`` overrides the configured fixed tolerance. Prefer passing one
    derived from the object's apparent size (see approach.derive.align_tolerance_px):
    a fixed pixel count is a third of a near object and a whole far one, so it demands
    quite different physical accuracy depending on where the object happens to be.
    """
    tol = float(cfg.align_tol_px if tolerance_px is None else tolerance_px)
    # The vertical axis is servoed through radial reach, which is coarser than base
    # rotation, so it gets a slightly looser tolerance rather than chasing forever.
    tol_v = tol * 1.3

    tr = ops.track(tries=5)
    if tr is None:
        ops.say("center: object not in view")
        return CenteringResult(None, False, "not_in_view")
    uv0 = np.array(tr.uv, dtype=np.float64)
    q0 = np.asarray(ops.joints(), dtype=np.float64)
    p0 = np.asarray(ops.tip(q0), dtype=np.float64)

    # Speed scales with range so the arm slows as it closes. The base carries the whole
    # arm's inertia and is what visibly jerks, so it gets the tightest limits.
    close = ops.range_m(tr) < CLOSE_RANGE_M
    speed = 0.9 if close else 1.6
    step = 1.8 * speed
    settle = 0.12 if close else 0.08
    max_pan = 4.5 if close else 8.0
    max_rad = 0.020 if close else 0.040

    du_dpan = _probe_pan(ops, uv0, settle, step)
    if du_dpan is None:
        ops.say("center: base rotation barely moves the object")
        return CenteringResult(np.asarray(ops.tip())[:2], False, "pan_probe_failed")

    dv_dr = _probe_radial(ops, uv0, p0, settle, step)
    ops.say(f"center: du/dpan={du_dpan:.1f}px/deg" +
            (f", dv/dr={dv_dr:.0f}px/m" if dv_dr else ", (no vertical probe)"))

    centered, reason, it = False, "max_iters", 0
    # What the previous iteration commanded, so this one can measure what it achieved.
    prev_uv, prev_dpan, prev_dr = None, 0.0, 0.0
    for it in range(int(cfg.align_iters)):
        ops.checkpoint()
        tr = ops.track(tries=3)
        if tr is None:
            ops.say("center: object gone (likely under the jaws) - stopping")
            reason = "lost"
            break
        du = float(tr.uv[0]) - target_uv[0]
        dv = float(tr.uv[1]) - target_uv[1]

        # Learn from the last move before sizing the next one.
        if prev_uv is not None:
            before = du_dpan, dv_dr
            du_dpan = _update_gain(du_dpan, float(tr.uv[0]) - prev_uv[0], prev_dpan,
                                   MIN_PAN_FOR_GAIN_DEG, ops.say, "pan")
            dv_dr = _update_gain(dv_dr, float(tr.uv[1]) - prev_uv[1], prev_dr,
                                 MIN_RAD_FOR_GAIN_M, ops.say, "radial")
            if (du_dpan, dv_dr) != before:
                ops.say(f"center: gains now du/dpan={du_dpan:.1f}px/deg" +
                        (f", dv/dr={dv_dr:.0f}px/m" if dv_dr else ""))
            if du_dpan is not None and abs(du_dpan) < MIN_DU_DPAN:
                # The loop has learned that the base no longer moves the object — the
                # same condition the probe refuses to start on.
                ops.say("center: base rotation no longer moves the object - stopping")
                reason = "pan_gain_collapsed"
                break

        ops.say(f"center {it}: {abs(du):.0f}px {'right' if du > 0 else 'left'}, "
                f"{abs(dv):.0f}px {'below' if dv > 0 else 'above'} the jaws")
        if abs(du) < tol and abs(dv) < tol_v:
            ops.say("centered on the object")
            centered, reason = True, "converged"
            break

        prev_uv, prev_dpan, prev_dr = (float(tr.uv[0]), float(tr.uv[1])), 0.0, 0.0
        moved = False
        if abs(du) >= tol:                                  # horizontal: rotate the base
            dpan = float(np.clip(-du / du_dpan, -max_pan, max_pan))
            ops.move_pan(dpan, settle=settle, step=step)
            prev_dpan = dpan
            moved = True
        if dv_dr and abs(dv) >= tol_v:                      # vertical: reach radially
            p = np.asarray(ops.tip(), dtype=np.float64)
            r = float(np.hypot(p[0], p[1]))
            u = np.array([p[0], p[1]]) / max(r, 1e-6)
            dr = float(np.clip(-dv / dv_dr, -max_rad, max_rad))
            if ops.move_tip(np.array([p[0] + u[0] * dr, p[1] + u[1] * dr, p[2]]),
                            settle=settle, step=step):
                prev_dr = dr
                moved = True
        if not moved:
            reason = "stuck"
            break

    return CenteringResult(np.asarray(ops.tip(), dtype=np.float64)[:2],
                           centered, reason, it + 1)
