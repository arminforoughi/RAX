"""``IkStrategy`` — turning a target point + a desired tool angle into joint angles.

The pick stack does not need full 6-DOF pose IK. It needs "put the fingertip HERE, with
the hand pitched THIS far over, and do not twist the wrist" — and it needs to be told
when that is impossible, rather than being handed a half-solved pose.

Two strategies satisfy that, chosen by :attr:`ArmProfile.ik`:

:class:`PitchHoldIK`
    For arms whose pitch joints share a parallel axis, so the tool's world pitch is
    exactly their sum. The last joint of the chain is then slaved *algebraically*
    instead of solved, which is why the held pitch never drifts, and the remaining
    joints are solved by damped least squares. This is the SO-101's proven solver,
    generalized: the arithmetic is unchanged, only the joint indices come from the
    profile instead of being literals.

:class:`PoseIK`
    For arms with a real 6-DOF wrist. Builds the target orientation from the requested
    pitch and roll and hands the whole problem to the ``Kinematics`` backend.

Both return ``(q_deg, residual_m)`` and **callers must check the residual**: a target
the arm cannot reach is a fact to report, not a pose to drive to.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["IkStrategy", "PitchHoldIK", "PoseIK", "make_ik", "DEFAULT_GRASP_PITCHES"]

# Candidate grasp pitches, tried in this order: steep first (best grip on a table
# object), then progressively shallower as fallbacks. Reach depends strongly on the
# angle — measured on the SO-101 from its URDF: 90deg -> 30.8cm, 70 -> 36.4, 60 -> 39.8,
# 50 -> 42.7, 40 -> 45.2, 30 -> 46.2, 20 -> 47.0, 10 -> 47.5, 0 -> 47.8. Stopping the
# list at 55 capped the arm at ~41cm, so anything further out was declared unreachable
# and the approach died short. The shallow entries let the arm actually GET THERE; the
# loop still returns the steepest angle that solves, so near objects are unaffected.
DEFAULT_GRASP_PITCHES = (75.0, 80.0, 70.0, 85.0, 90.0, 65.0, 60.0, 55.0,
                         50.0, 45.0, 40.0, 35.0, 30.0, 25.0, 20.0, 15.0,
                         10.0, 5.0, 0.0)

# Fingertip parks this far above the object, then descends.
DEFAULT_STANDOFF_M = 0.05


@runtime_checkable
class IkStrategy(Protocol):
    def solve(self, q_seed, p_target, *, pitch_deg=None, roll_deg=None
              ) -> tuple[np.ndarray, float]:
        """Target point -> (joint angles deg, position residual m)."""
        ...

    def plan_pitch(self, p_target, q_seed) -> tuple[float | None, float]:
        """Steepest reachable grasp pitch, or (None, best_residual)."""
        ...


class _BaseIK:
    """Shared pitch planning: try each candidate angle and prove the arm can hold it."""

    def __init__(self, kin, profile, *, grasp_pitches=DEFAULT_GRASP_PITCHES,
                 standoff_m: float = DEFAULT_STANDOFF_M, workspace=None):
        self.kin = kin
        self.profile = profile
        self.lo, self.hi = profile.limits()
        self.grasp_pitches = tuple(grasp_pitches)
        self.standoff_m = float(standoff_m)
        # An optional WorkspaceMap, derived from this arm's own kinematics. When
        # present it supplies the candidate angles per target instead of the hardcoded
        # list, so the search is over what this arm can actually do rather than over
        # what someone measured on a different one. See manipulation/arms/workspace.py.
        self.workspace = workspace

    def _fk_pos(self, q) -> np.ndarray:
        return np.asarray(self.kin.forward_kinematics(q), dtype=np.float64)[:3, 3]

    def plan_pitch(self, p_target, q_seed) -> tuple[float | None, float]:
        """Choose the grasp pitch, and PROVE the arm can get there.

        Both the standoff pose ABOVE the object and the object pose itself must solve —
        an angle that reaches one but not the other cannot complete a descent.

        The geometry that gets ignored otherwise: the fingertip is well out in front of
        the wrist, so putting it on an object at r=15cm, z=2cm with the hand HORIZONTAL
        demands the wrist sit inside the robot's own base column. You grab an object off
        a table from ABOVE — point the hand down. Steeper is kinematically safer too:
        on the SO-101, pitch 40-60 at r=10-15cm is a genuine elbow-flip dead band where
        no seed converges.
        """
        p_obj = np.asarray(p_target, dtype=np.float64)
        p_above = p_obj + np.array([0.0, 0.0, self.standoff_m])
        candidates = self.grasp_pitches
        if self.workspace is not None:
            # Derived candidates: already filtered to what is reachable here, and
            # ordered steepest-first. Fall back if the target is off the probed grid.
            derived = self.workspace.pitch_candidates(p_obj)
            if derived:
                candidates = derived
        best = None
        for pitch in candidates:
            _, e_hi = self.solve(q_seed, p_above, pitch_deg=pitch)
            _, e_lo = self.solve(q_seed, p_obj, pitch_deg=pitch)
            worst = max(float(e_hi), float(e_lo))
            # At the workspace edge (shallow pitch / far reach) the residual can be a few
            # mm larger and still be a valid pose. A sliding tolerance keeps the arm's
            # real reach; the visual centering pass then fine-tunes.
            tol = 0.025 if pitch <= 15.0 else 0.004
            if worst <= tol:
                return pitch, worst
            if best is None or worst < best[1]:
                best = (pitch, worst)
        # Last resort: if nothing solved tightly but the best residual is still usable,
        # return it rather than declaring the object unreachable at the edge of reach.
        if best is not None and best[1] <= 0.030:
            return best
        return None, (best[1] if best else float("inf"))


class PitchHoldIK(_BaseIK):
    """Position IK on the positioning joints, with the last pitch joint slaved.

    THIS SOLVER USED TO SILENTLY NOT CONVERGE, and that was the "the arm grabs at air /
    just moves out" bug (fixed 2026-07-13). It ran a FIXED 10 iterations with a +-4
    deg/iter clamp -- a total travel budget of 40 deg -- while a perfectly ordinary
    reach like tip -> (0.15, 0, 0.02) needs 80-160 deg of elbow. Measured residual for
    that exact target with the old code: 107 mm at pitch 0, 93 mm at pitch 20, 35 mm at
    pitch 60 -- for a point THIS code hits to 0.2 mm. It returned a half-solved pose,
    the motion layer faithfully drove to it, the next hop re-seeded from there, and the
    fingertip crept outward and UPWARD forever.

    Why it only ran 10 iterations: FK costs ~0.8 ms and a fresh numeric Jacobian is 3
    more FK, so 10 iters was already 32 ms -- near the 70 ms jog tick. The fix is to
    stop rebuilding J every step: over a <=3 cm step it barely rotates, so it is reused
    for 8 iterations. That buys convergence AND is faster than before.

    Two invariants that must survive any edit here:
      * CLAMP TO THE JOINT LIMITS EVERY ITERATION, AND SCORE THE CLAMPED POSE. An IK
        that does not know the limits returns elbow=+162 deg on a joint that stops at
        +96.8; the servo silently clamps, the arm parks at the stop, and the solver
        reports a 0.2 mm residual on a pose the robot cannot hold.
      * If the slaved joint saturates, the pitch was NOT held, so the pose is reported
        unreachable rather than as a success at a different angle.
    """

    def __init__(self, kin, profile, **kw):
        super().__init__(kin, profile, **kw)
        if profile.ik != "pitch_hold":
            raise ValueError(f"profile {profile.name!r} does not declare ik='pitch_hold'")
        self.pos = list(profile.positioning_joints)   # driven numerically
        self.chain_head = list(profile.pitch_chain[:-1])  # contribute to the pitch sum
        self.slaved = profile.slaved_joint            # solved algebraically
        self.roll = profile.roll_joint
        self.seeds = profile.ik_seeds

    def _slave(self, q, pitch_deg) -> float:
        """The slaved joint's angle that makes the chain sum to the requested pitch.
        Exact — no convergence needed — so the angle never drifts.

        Subtracted one joint at a time rather than via a sum, so the floating-point
        rounding matches the original `pitch - j1 - j2` exactly. That keeps this a
        bit-identical replacement, which is what makes the parity harness a sharp test
        rather than one with a tolerance to hide behind.
        """
        v = float(pitch_deg)
        for i in self.chain_head:
            v -= float(q[i])
        return float(np.clip(v, self.lo[self.slaved], self.hi[self.slaved]))

    #: Public name for the slaving rule; callers that hold a joint vector can use it
    #: to keep the tool angle while moving the other pitch joints themselves.
    slave = _slave

    def solve(self, q_seed, p_target, *, pitch_deg=None, roll_deg=None,
              iters: int = 80, tol: float = 2e-3, _retry: bool = True
              ) -> tuple[np.ndarray, float]:
        pitch = 0.0 if pitch_deg is None else float(pitch_deg)
        p_tgt = np.asarray(p_target, dtype=np.float64)
        q = np.array(q_seed, dtype=np.float64)
        if self.roll is not None and roll_deg is not None:
            q[self.roll] = float(np.clip(roll_deg, self.lo[self.roll], self.hi[self.roll]))
        q[self.slaved] = self._slave(q, pitch)

        pos = self.pos
        J = None
        for it in range(iters):
            p_now = self._fk_pos(q)
            err = p_tgt - p_now
            if np.linalg.norm(err) < 3e-4:
                break
            if J is None or it % 8 == 0:
                J = np.empty((3, len(pos)))
                for c, ji in enumerate(pos):
                    dq = q.copy()
                    dq[ji] = float(np.clip(dq[ji] + 0.5, self.lo[ji], self.hi[ji]))
                    if ji in self.chain_head:          # keep the pitch held while
                        dq[self.slaved] = self._slave(dq, pitch)   # perturbing
                    J[:, c] = (self._fk_pos(dq) - p_now) / 0.5
            dth = np.clip(J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), err), -8.0, 8.0)
            q[pos] = np.clip(q[pos] + dth, self.lo[pos], self.hi[pos])  # STAY INSIDE
            q[self.slaved] = self._slave(q, pitch)

        e = float(np.linalg.norm(p_tgt - self._fk_pos(q)))
        if abs(float(np.sum(q[list(self.profile.pitch_chain)])) - pitch) > 2.0:
            e = max(e, 0.05)      # pitch could not be held here: treat as unreachable

        if e > tol and _retry and self.seeds:
            # Wrong IK branch. There are genuine dead bands where no single seed
            # converges (the SO-101's elbow flip at r=10-15cm with a shallow pitch), so
            # re-seed from the profile's alternatives and keep the best.
            for seed in self.seeds:
                alt = np.array([q_seed[i] if v is None else v
                                for i, v in enumerate(seed)], dtype=np.float64)
                q2, e2 = self.solve(alt, p_tgt, pitch_deg=pitch, roll_deg=roll_deg,
                                    iters=iters, tol=tol, _retry=False)
                if e2 < e:
                    q, e = q2, e2
                if e <= tol:
                    break
        return q, e


class PoseIK(_BaseIK):
    """Full-pose IK for arms with a real wrist, via the ``Kinematics`` backend.

    The requested pitch and roll become a target orientation: the tool's approach axis
    is tilted ``pitch`` degrees below horizontal, in the vertical plane containing the
    target's bearing from the base, then rolled about itself. An arm whose tool frame
    does not point along +Z should subclass and override :meth:`target_rotation`.
    """

    #: Which end-effector axis points out of the tool. +Z matches the usual convention.
    approach_axis = np.array([0.0, 0.0, 1.0])

    def target_rotation(self, p_target, pitch_deg: float, roll_deg: float) -> np.ndarray:
        bearing = math.atan2(float(p_target[1]), float(p_target[0]))
        pitch = math.radians(float(pitch_deg))
        # Unit approach direction: pitch=0 reaches horizontally, pitch=90 straight down.
        d = np.array([math.cos(bearing) * math.cos(pitch),
                      math.sin(bearing) * math.cos(pitch),
                      -math.sin(pitch)], dtype=np.float64)
        z = d / np.linalg.norm(d)
        # Any perpendicular gives a valid frame; roll then rotates about the tool axis.
        ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
        x = np.cross(ref, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.column_stack([x, y, z])
        if roll_deg:
            from scipy.spatial.transform import Rotation
            R = R @ Rotation.from_rotvec(math.radians(float(roll_deg)) * np.array([0, 0, 1.0])).as_matrix()
        return R

    def solve(self, q_seed, p_target, *, pitch_deg=None, roll_deg=None,
              orientation_weight: float = 0.05) -> tuple[np.ndarray, float]:
        p_tgt = np.asarray(p_target, dtype=np.float64)
        q_seed = np.asarray(q_seed, dtype=np.float64)
        T = np.asarray(self.kin.forward_kinematics(q_seed), dtype=np.float64).copy()
        T[:3, 3] = p_tgt
        if pitch_deg is not None:
            T[:3, :3] = self.target_rotation(p_tgt, pitch_deg, roll_deg or 0.0)
            ow = orientation_weight
        else:
            ow = 0.0
        q = np.asarray(self.kin.inverse_kinematics(
            q_seed, T, position_weight=1.0, orientation_weight=ow), dtype=np.float64)
        q = np.clip(q, self.lo, self.hi)     # score the pose the robot can actually hold
        return q, float(np.linalg.norm(p_tgt - self._fk_pos(q)))


def make_ik(kin, profile, **kw) -> IkStrategy:
    """Build the IK strategy the profile asks for."""
    if profile.ik == "pitch_hold":
        return PitchHoldIK(kin, profile, **kw)
    if profile.ik == "pose":
        return PoseIK(kin, profile, **kw)
    raise ValueError(f"unknown ik strategy {profile.ik!r} on profile {profile.name!r}")
