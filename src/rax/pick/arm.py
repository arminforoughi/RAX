"""What the pick needs from a robot: joints, a model, a wrist camera and a gripper.

Anything that implements :class:`Arm` can run :func:`rax.pick.pick`. There is no
training and no per-robot tuning in the pick itself: every sign and scale it depends on
is either read off the arm's own kinematic model or measured with a test move.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np


class Arm(Protocol):
    """A serial arm with a camera on the wrist. Angles are degrees, lengths metres."""

    pan: int                       # index of the base (yaw) joint
    roll: int                      # index of the wrist roll joint
    pitch_chain: tuple[int, ...]   # joints whose angles sum to the hand's pitch
    lo: np.ndarray                 # joint limits
    hi: np.ndarray
    home: np.ndarray               # a raised pose with the table in view

    jaw_uv: tuple[float, float]    # where the grip centre appears in the wrist image
    jaw_axis_deg: float            # image angle of the line the jaws close along
    roll_gain: float               # image degrees turned per degree of wrist roll

    def joints(self) -> np.ndarray: ...
    def move(self, q: np.ndarray, speed: float = 1.0, settle: float = 0.2) -> None: ...
    def tip(self, q: np.ndarray) -> np.ndarray: ...
    def ik(self, seed: np.ndarray, p: np.ndarray, pitch: float,
           roll: float) -> tuple[np.ndarray, float]: ...

    def frame(self) -> np.ndarray | None: ...                      # fresh BGR image
    def cast(self, uv, q: np.ndarray) -> tuple[float, float] | None: ...  # pixel -> table xy
    def project(self, p, q: np.ndarray) -> tuple[float, float] | None: ...

    def grip(self, pct: float) -> None: ...
    def release(self) -> None: ...                     # open fully
    def close(self, from_pct: float) -> bool: ...     # True if the current rose on contact
    def grip_pos(self) -> float: ...                   # settled jaw opening, percent

    def checkpoint(self) -> None: ...                  # raises if the operator pressed stop
    def log(self, msg: str) -> None: ...
    def phase(self, name: str, note: str = "") -> None: ...


def fold(deg: float, period: float = 180.0) -> float:
    """Wrap an angle into [-period/2, period/2)."""
    return ((float(deg) + period / 2.0) % period) - period / 2.0


def pitch_of(arm: Arm, q) -> float:
    """The hand's pitch: 0 is level, 90 is pointing straight down."""
    return float(sum(q[i] for i in arm.pitch_chain))


def bearing_of(p) -> float:
    """Bearing of a base-frame point, radians (+ is to the robot's left)."""
    return math.atan2(float(p[1]), float(p[0]))


def clamp(arm: Arm, q) -> np.ndarray:
    return np.clip(np.asarray(q, float), arm.lo, arm.hi)


def pan_for_bearing(arm: Arm, q, bearing: float) -> float:
    """The base angle that points the fingertip along ``bearing``, read off the model.

    Searched, not assumed: on the SO-101 the base angle runs opposite to the bearing
    (base -33 faces +26deg), and an assumed sign turns the arm the wrong way.
    """
    best, best_err = float(q[arm.pan]), None
    for pan in np.arange(arm.lo[arm.pan], arm.hi[arm.pan], 1.0):
        qq = np.asarray(q, float).copy()
        qq[arm.pan] = pan
        err = abs(fold(math.degrees(bearing_of(arm.tip(qq)) - bearing), 360.0))
        if best_err is None or err < best_err:
            best, best_err = float(pan), err
    return best


def with_pan(arm: Arm, q, pan: float) -> np.ndarray:
    q = np.asarray(q, float).copy()
    q[arm.pan] = float(np.clip(pan, arm.lo[arm.pan], arm.hi[arm.pan]))
    return q


#: Extra IK starting poses (shoulder, elbow, wrist pitch). One seed is why grasps came
#: in at 70deg: the model reaches 90 at grasp height out to 30cm, but not from wherever
#: the solver happened to start.
_SEEDS = ((-30, 40, 70), (0, -10, 90), (20, -40, 95), (-10, 20, 80))


def solve(arm: Arm, p, pitch: float, roll: float, seed=None, tol: float = 0.01,
          max_jump: float | None = None):
    """Joints that put the fingertip at ``p`` holding ``pitch`` and ``roll``, or None.

    Tries several seeds and keeps the solution nearest the current pose, so the arm
    never flips its elbow to get somewhere.
    """
    q0 = arm.joints() if seed is None else np.asarray(seed, float)
    seeds = [q0]
    for s in _SEEDS:
        sd = q0.copy()
        for i, v in zip(arm.pitch_chain, s):
            sd[i] = v
        seeds.append(sd)
    best = None
    for sd in seeds:
        q, err = arm.ik(sd, np.asarray(p, float), float(pitch), float(roll))
        if err > tol:
            continue
        q = np.asarray(q, float)
        jump = max(abs(q[i] - q0[i]) for i in arm.pitch_chain)
        if max_jump is not None and jump > max_jump:
            continue
        if best is None or np.abs(q - q0).sum() < np.abs(best - q0).sum():
            best = q
    if best is not None:
        best[arm.roll] = roll
    return best


def steepest(arm: Arm, p, pitch_hi: float, pitch_lo: float, roll: float, tol=0.004):
    """The steepest pitch in [pitch_lo, pitch_hi] that reaches ``p``: (q, pitch) or (None, None)."""
    for pitch in np.arange(pitch_hi, pitch_lo - 0.1, -2.5):
        q = solve(arm, p, float(pitch), roll, tol=tol)
        if q is not None:
            return q, float(pitch)
    return None, None


def move_to(arm: Arm, p, pitch: float, roll: float, what: str, speed=1.0, settle=0.25,
            tol=0.01) -> np.ndarray:
    """Put the fingertip at ``p`` holding the hand's angle, or raise."""
    q = solve(arm, p, pitch, roll, tol=tol)
    if q is None:
        raise RuntimeError(f"cannot reach {what} at ({p[0]*100:+.1f},{p[1]*100:+.1f},"
                           f"{p[2]*100:+.1f})cm, pitch {pitch:.0f}")
    arm.move(q, speed=speed, settle=settle)
    return q


def straight_down(arm: Arm, z: float, pitch: float, roll: float,
                  step_m: float = 0.02) -> float:
    """Lower the fingertip to height ``z`` at fixed x, y and angle. Returns the height reached.

    One IK-solved waypoint every ``step_m``, so the fingers follow a straight vertical
    line (a single joint-space move would arc sideways into the object) without the
    stop-and-settle of many tiny moves: only the last waypoint settles.
    """
    p0 = arm.tip(arm.joints())
    z0 = float(p0[2])
    steps = max(1, math.ceil(abs(z0 - z) / step_m))
    for k in range(1, steps + 1):
        arm.checkpoint()
        zk = z0 + (z - z0) * k / steps
        q = solve(arm, (p0[0], p0[1], zk), pitch, roll, tol=0.005)
        if q is None:
            arm.log(f"        z={zk*100:+.1f}cm not reachable at {pitch:.0f}deg — stopping")
            break
        arm.move(q, speed=0.9, settle=0.12 if k == steps else 0.0)
    return float(arm.tip(arm.joints())[2])
