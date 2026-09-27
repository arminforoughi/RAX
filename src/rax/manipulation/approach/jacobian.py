"""The image Jacobian: what each actuator does to the picture, MEASURED, not derived.

THE ONE IDEA. A visual servo needs to answer exactly one question — "if I move this
actuator a little, which way does the target slide in the image, and does it get
bigger?" Classical image-based servoing computes that from camera intrinsics, the
hand-eye transform and the robot's kinematics. All three are calibrations, all three
were wrong on this rig at some point, and none of them is known at all for an arm you
have just plugged in.

You can also just move the actuator and look. That is this module. What comes back is a
3xN matrix — rows (column, row, log-size), one column per axis — in units of feature per
actuator unit. It requires no intrinsics, no extrinsics, no URDF and no joint semantics,
which is precisely why it ports to any arm with a camera bolted to it.

WHY IT ALSO FIXES A BUG THIS RIG ACTUALLY HAD. The hand-tuned predecessor used two
independent scalar gains, one joint per image axis, and measured them once in a
mid-range pose. Inside ~20cm the vertical loop went unstable — error growing while the
correction was applied, -70 -> -86 -> -111px at increasing wrist pitch. The cause is
that the wrist is not a pure aiming joint: the camera sits ~10cm off its axis, so at
close range pitching down swings the camera backward and upward by MORE than the
rotation gains in aim, and the effective gain changes sign. Two facts follow, and both
are structural rather than tuning:

  * The Jacobian is a LOCAL model. It is only true near the pose it was measured at, so
    it has to keep up — see :meth:`ImageJacobian.updated`, which corrects it from the
    motion the servo has just made, for free, every tick.
  * Cross-coupling is not a nuisance to be trimmed out, it is most of the signal. A
    least-squares solve over all axes at once uses it; two independent scalar gains
    cannot represent it at all, which is why they fought each other.

Solving all three features together also collapses the old two-phase "gaze, then
advance" structure into one step. Centring and closing range stop being separate
behaviours that undo each other's work and become three rows of one linear solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rax.manipulation.approach.servo_arm import Axis, ServoArm, Sighting, features

__all__ = ["ImageJacobian", "JacobianError", "measure_jacobian"]


class JacobianError(RuntimeError):
    """Raised when the Jacobian cannot be measured — never guessed around."""


@dataclass(frozen=True)
class ImageJacobian:
    """d(feature)/d(actuator), 3 rows by N axes, in feature units per actuator unit."""

    J: np.ndarray
    axes: tuple[Axis, ...]
    #: Per-axis pixel motion seen while probing. A near-zero entry means that axis does
    #: nothing visible from here, and the solve should not lean on it.
    probe_response_px: np.ndarray

    def __post_init__(self):
        if self.J.shape != (3, len(self.axes)):
            raise ValueError(f"J is {self.J.shape}, expected (3, {len(self.axes)})")

    def solve(self, err: np.ndarray, *, damping: float = 0.35,
              weights: tuple[float, float, float] = (1.0, 1.0, 60.0)) -> np.ndarray:
        """Actuator step that best reduces ``err``. Damped least squares.

        ``damping`` is a Levenberg term, not a gain: it trades exactness for stability
        near a singular pose, where some direction in the image is unreachable by any
        combination of axes and an exact inverse would demand an enormous move to chase
        it. It is also what keeps a single noisy detection from throwing the arm.

        ``weights`` puts the three feature rows on comparable footing. Columns and rows
        are in pixels, tens to hundreds of them; log-size is a pure ratio where 0.1 is
        already a 10% change in range. Without the weight the size row is numerically
        invisible and the servo centres beautifully while never closing in — which is
        the exact failure the two-phase predecessor showed, with the box height frozen
        at 82px across three advances.
        """
        W = np.diag(np.asarray(weights, dtype=np.float64))
        A = W @ self.J
        b = W @ np.asarray(err, dtype=np.float64)
        H = A.T @ A

        # PER-AXIS DAMPING (Marquardt's diag(H), not a scalar times I). This is not a
        # refinement, it is required for correctness the moment axes have different
        # units — and on any real arm they do: degrees for a joint, metres for a reach.
        #
        # With a single scalar term taken from the trace, the largest-gain axis sets the
        # damping for every axis. Measured in the sim: a lift axis worth 900px/m put
        # trace(H)/n at ~2e5, so the pan axis, worth 9px/deg, was damped by 25000
        # against its own 81 and moved 0.11deg per tick where it needed 35. The servo
        # sat 306px off the aim and reported it had "stopped improving" — true, and
        # entirely self-inflicted.
        #
        # Scaling each axis by its own sensitivity makes the solve invariant to the
        # units the arm happens to report in, which is what lets an arbitrary arm be
        # plugged in without retuning `damping`.
        d = np.diag(H).copy()
        floor = max(float(d.max()), 1e-12) * 1e-6      # keep a dead axis invertible
        d = np.maximum(d, floor)
        dq = np.linalg.solve(H + float(damping) ** 2 * np.diag(d), A.T @ b)
        return np.asarray(dq, dtype=np.float64)

    def solve_prioritised(self, err: np.ndarray, *, damping: float = 0.35,
                          size_gain: float = 0.5) -> np.ndarray:
        """Line the box up FIRST; close the range only in the freedom left over.

        The two jobs kept fighting, and weighting them against each other was the wrong
        answer both ways round. With size weighted heavily the solve pulled the arm BACK
        — retracting is a cheap way to trade alignment for range. With size weighted at
        zero the alignment became perfect and the gripper stopped 4x too far away,
        because aligning a box with the fingertip puts them on the SAME RAY, not at the
        same point. Measured on the rig: aim error (-3,+10)px with a size error of 1.46.

        They are not equals, so they should not be weighted. Alignment is the task;
        closing in is what to do with the slack. This arm has four joints and alignment
        uses two of them, so there are two spare degrees of freedom, and moving along
        those cannot disturb the aim to first order — which is precisely "move toward
        the object without looking away from it".

            dq = J_align+ e_align  +  (I - J_align+ J_align) (gain * e_size * J_size)
                 ^ put the box on the fingertip   ^ ...then close, in the null space
        """
        Ja = self.J[:2, :]
        ea = np.asarray(err, dtype=np.float64)[:2]
        n = Ja.shape[1]

        Ha = Ja.T @ Ja
        d = np.diag(Ha).copy()
        d = np.maximum(d, max(float(d.max()), 1e-12) * 1e-6)
        dq = np.linalg.solve(Ha + float(damping) ** 2 * np.diag(d), Ja.T @ ea)

        # Null-space projector of the alignment task: everything that moves the joints
        # WITHOUT moving the box in the image.
        null = np.eye(n) - np.linalg.pinv(Ja) @ Ja
        js = self.J[2, :]
        dq = dq + float(size_gain) * float(err[2]) * (null @ js)
        return np.asarray(dq, dtype=np.float64)

    def predict(self, dq: np.ndarray) -> np.ndarray:
        """How the features should change if we apply ``dq``."""
        return self.J @ np.asarray(dq, dtype=np.float64)

    def updated(self, dq: np.ndarray, d_feat: np.ndarray, *,
                rate: float = 0.5) -> "ImageJacobian":
        """Broyden rank-1 correction from a move the servo has already made.

        The Jacobian is a local model, and the servo walks away from where it was
        measured — on this arm far enough that the vertical gain reverses sign by the
        time the gripper is close. Re-probing every few centimetres would cost more
        motion than the approach itself. Instead: we commanded ``dq`` and the features
        moved ``d_feat``, so we know exactly how wrong the model was along ``dq``, and
        we correct it in that direction only. Free, one tick of lag, and it tracks a
        sign change instead of diverging through it.

        ``rate`` damps the correction so one badly-detected frame cannot rewrite the
        model. Directions the servo has not moved in are left alone, which is correct:
        nothing was learned about them.
        """
        dq = np.asarray(dq, dtype=np.float64)
        denom = float(dq @ dq)
        if denom < 1e-12:
            return self
        resid = np.asarray(d_feat, dtype=np.float64) - self.J @ dq
        J = self.J + float(rate) * np.outer(resid, dq) / denom
        return ImageJacobian(J=J, axes=self.axes,
                             probe_response_px=self.probe_response_px)

    def describe(self) -> str:
        rows = []
        for k, ax in enumerate(self.axes):
            rows.append(f"{ax.name}: du={self.J[0, k]:+7.2f} dv={self.J[1, k]:+7.2f} "
                        f"dlogh={self.J[2, k]:+7.4f} per unit "
                        f"({self.probe_response_px[k]:.0f}px probe)")
        return " | ".join(rows)


def measure_jacobian(arm: ServoArm, *, settle_reads: int = 2,
                     min_response_px: float = 3.0,
                     log=lambda _m: None) -> ImageJacobian:
    """Measure the Jacobian by moving each axis and watching the target.

    CENTRAL DIFFERENCES, not one-sided. Probing only in the + direction folds any drift
    or backlash straight into the gain, and cannot tell "this axis does nothing" from
    "this axis moved the arm and something else moved the picture back". Probing both
    ways costs one extra move per axis and cancels both.

    The arm is returned to its starting actuators after every probe and again at the
    end, so a failed measurement leaves the robot where it found it.

    Raises :class:`JacobianError` rather than returning a guess: an approach driven by a
    Jacobian nobody could measure is exactly the class of thing this whole rewrite
    exists to stop.
    """
    q0 = np.asarray(arm.actuators(), dtype=np.float64).copy()

    def look() -> Sighting:
        best = None
        for _ in range(max(1, settle_reads)):
            s = arm.sense()
            if s is not None:
                best = s
        if best is None:
            raise JacobianError(
                "the target is not in view, so there is nothing to measure the "
                "Jacobian against. Put it in the camera's view first.")
        return best

    f0 = features(look())
    cols, responses = [], []
    for ax in arm.axes:
        plus = np.clip(q0[ax.index] + ax.probe, ax.lo, ax.hi)
        minus = np.clip(q0[ax.index] - ax.probe, ax.lo, ax.hi)
        span = plus - minus
        if abs(span) < 1e-9:
            raise JacobianError(f"axis '{ax.name}' has no room to probe at "
                                f"{q0[ax.index]:.1f} (limits {ax.lo:.1f}..{ax.hi:.1f})")
        qp, qm = q0.copy(), q0.copy()
        qp[ax.index], qm[ax.index] = plus, minus

        if not arm.apply(qp):
            raise JacobianError(f"could not move '{ax.name}' to {plus:.1f} to probe it")
        fp = features(look())
        if not arm.apply(qm):
            arm.apply(q0)
            raise JacobianError(f"could not move '{ax.name}' to {minus:.1f} to probe it")
        fm = features(look())
        arm.apply(q0)

        col = (fp - fm) / span
        resp = float(np.hypot(fp[0] - fm[0], fp[1] - fm[1]))
        cols.append(col)
        responses.append(resp)
        log(f"jacobian: {ax.name} +-{ax.probe:g} moved the target {resp:.0f}px "
            f"-> du/dq={col[0]:+.2f} dv/dq={col[1]:+.2f} dlogh/dq={col[2]:+.4f}")

    arm.apply(q0)
    J = np.column_stack(cols) if cols else np.zeros((3, 0))
    responses = np.asarray(responses, dtype=np.float64)
    if J.shape[1] == 0:
        raise JacobianError("no axes to measure")
    if not np.any(responses >= min_response_px):
        raise JacobianError(
            f"no axis moved the target more than {min_response_px:.0f}px "
            f"(best {responses.max():.1f}px). The camera may not be seeing what it "
            f"thinks it is, or the probe steps are too small to measure.")
    dead = [ax.name for ax, r in zip(arm.axes, responses) if r < min_response_px]
    if dead:
        log(f"jacobian: {', '.join(dead)} moved the picture almost not at all — the "
            f"solve will simply not use them from this pose")
    # f0 is measured but deliberately unused in the result: it is the pose the model is
    # valid AROUND, and keeping it would invite someone to treat it as a target.
    del f0
    return ImageJacobian(J=J, axes=tuple(arm.axes), probe_response_px=responses)
