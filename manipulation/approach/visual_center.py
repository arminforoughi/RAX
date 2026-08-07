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

    def move_pan(self, delta_deg: float, *, settle: float, step: float) -> None:
        """Rotate the base by delta_deg and settle."""

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
    """Pixels of horizontal object motion per degree of base rotation, or None."""
    ops.move_pan(+PAN_PROBE_DEG, settle=settle, step=step)
    tr = ops.track(tries=4)
    ops.move_pan(-PAN_PROBE_DEG, settle=settle, step=step)
    if tr is None:
        return None
    gain = (float(tr.uv[0]) - uv0[0]) / PAN_PROBE_DEG
    return None if abs(gain) < MIN_DU_DPAN else gain


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


def center_on_object(ops: CenteringOps, target_uv, cfg) -> CenteringResult:
    """Servo the arm until the object sits on ``target_uv`` in the image.

    ``target_uv`` is where the object should appear when the jaws are around it —
    normally the measured fingertip pixel plus the configured aim trims.
    """
    tol = float(cfg.align_tol_px)
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
    for it in range(int(cfg.align_iters)):
        ops.checkpoint()
        tr = ops.track(tries=3)
        if tr is None:
            ops.say("center: object gone (likely under the jaws) - stopping")
            reason = "lost"
            break
        du = float(tr.uv[0]) - target_uv[0]
        dv = float(tr.uv[1]) - target_uv[1]
        ops.say(f"center {it}: {abs(du):.0f}px {'right' if du > 0 else 'left'}, "
                f"{abs(dv):.0f}px {'below' if dv > 0 else 'above'} the jaws")
        if abs(du) < tol and abs(dv) < tol_v:
            ops.say("centered on the object")
            centered, reason = True, "converged"
            break

        moved = False
        if abs(du) >= tol:                                  # horizontal: rotate the base
            dpan = float(np.clip(-du / du_dpan, -max_pan, max_pan))
            ops.move_pan(dpan, settle=settle, step=step)
            moved = True
        if dv_dr and abs(dv) >= tol_v:                      # vertical: reach radially
            p = np.asarray(ops.tip(), dtype=np.float64)
            r = float(np.hypot(p[0], p[1]))
            u = np.array([p[0], p[1]]) / max(r, 1e-6)
            dr = float(np.clip(-dv / dv_dr, -max_rad, max_rad))
            if ops.move_tip(np.array([p[0] + u[0] * dr, p[1] + u[1] * dr, p[2]]),
                            settle=settle, step=step):
                moved = True
        if not moved:
            reason = "stuck"
            break

    return CenteringResult(np.asarray(ops.tip(), dtype=np.float64)[:2],
                           centered, reason, it + 1)
