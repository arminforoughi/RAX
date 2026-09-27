"""The seam a visual servo needs from a robot — and it is much smaller than an arm.

WHY THIS IS NOT ``ArmInterface``. That protocol asks for joint state, gripper state and
**the camera's pose in the base frame**, which means a URDF, forward kinematics, and a
hand-eye transform. Every one of those is a calibration you have to get right before the
robot can move at all, and on this rig every one of them has been wrong at some point:
the hand-eye TF was out by 370px for two days, ``pan = bearing`` aimed the base 46deg
the wrong way, and the fingertip pixel the grasp aimed at could not physically be
reached. A stack that needs all of that to work cannot be plugged into a new arm in an
afternoon, because none of it is known for the new arm either.

So this seam asks for none of it:

    axes        which actuators the servo may move, and how far
    actuators() where they are now
    apply(q)    go there, AND SETTLE
    sense()     find the target in the current frame, in PIXELS

No kinematics, no intrinsics, no extrinsics, no table plane, no joint semantics. The
servo does not know which axis is a shoulder and which is a wrist; it learns what each
one does to the picture by moving it and watching (see ``jacobian.py``). Any arm that
can report and command its actuators, with any camera rigidly attached to it, satisfies
this — including arms with no URDF at all.

``apply`` MUST NOT RETURN UNTIL THE ARM HAS ARRIVED. Measured on the SO-101: a
non-blocking version let the loop read the next frame ~50ms later, compute the same
correction from the same unchanged picture, and overwrite a command the arm had not
begun to execute — 60 ticks in 3 seconds, all issuing the same 0.9deg correction against
a pixel error frozen at -20px. A servo has to let its own output happen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["Axis", "Sighting", "ServoArm", "features", "feature_error"]


@dataclass(frozen=True)
class Axis:
    """One actuator the servo is allowed to move.

    ``probe`` is how far to move it when measuring what it does to the picture. It wants
    to be big enough that the image moves well clear of detector noise, and small enough
    that the local linear model still holds and the target stays in frame. On the SO-101
    5 degrees moved the box ~54px, which is a good ratio against ~2px of jitter.
    """

    name: str
    index: int                  # position in the arm's actuator vector
    probe: float                # step used to measure the Jacobian column
    max_step: float             # largest correction allowed in one servo tick
    lo: float = -math.inf       # travel limits, in the actuator's own units
    hi: float = math.inf

    def clamp_step(self, d: float) -> float:
        return -self.max_step if d < -self.max_step else (
            self.max_step if d > self.max_step else float(d))


@dataclass(frozen=True)
class Sighting:
    """One detection, in pixels. ``None`` in place of a Sighting means "nothing seen".

    Pixels and nothing else: no range, no 3D point, no class confidence. Whatever the
    detector is — YOLO, SAM, a colour blob, a fiducial — this is all the servo wants
    from it, and it is all any of them can supply without a calibration.

    ``clipped_bottom`` is carried because the size feature reads the box HEIGHT: a box
    cut off by the frame edge is shorter than the object, so it reads as further away
    exactly when the gripper is closest.
    """

    bbox: tuple[float, float, float, float]      # x1, y1, x2, y2
    frame_w: int
    frame_h: int
    clipped_bottom: bool = False

    @property
    def cx(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.bbox[1] + self.bbox[3])

    @property
    def width(self) -> float:
        return abs(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return abs(self.bbox[3] - self.bbox[1])


@runtime_checkable
class ServoArm(Protocol):
    """Any arm with a rigidly attached camera and a target detector."""

    #: The actuators the servo may move, in the order its Jacobian columns use.
    axes: tuple[Axis, ...]

    def actuators(self) -> np.ndarray:
        """Current value of every actuator, in the arm's own units."""

    def apply(self, q: np.ndarray) -> bool:
        """Command the full actuator vector and BLOCK until the arm has settled.

        Returns False if the move could not be made (a limit, no IK solution for an
        arm that needs one) — the servo treats that as a reason to stop, never as a
        reason to try somewhere else.
        """

    def sense(self) -> Sighting | None:
        """The target in the current frame, or None if it is not there."""


def features(seen: Sighting) -> np.ndarray:
    """The three numbers the servo controls: image column, image row, log size.

    LOG SIZE, not size. Apparent size goes as 1/range, so equal fractions of range are
    equal steps in log — which makes the size row of the Jacobian a single constant
    over the whole approach instead of a number that quadruples as you close in. It also
    makes the row scale-free, so the same servo works for a 2cm cube and a 20cm bottle
    without retuning.
    """
    return np.array([seen.cx, seen.cy, math.log(max(seen.height, 1.0))], dtype=np.float64)


def feature_error(seen: Sighting, aim_u: float, view_band: tuple[float, float],
                  target_height_px: float, aim_v: float | None = None) -> np.ndarray:
    """Where we want the target minus where it is, in feature space.

    THE ROW IS A CONSTRAINT, NOT A TARGET, and getting that wrong cost a whole
    approach. Pinning the object to a fixed image row over-constrains the arm: measured
    on the SO-101, ``shoulder_lift`` and ``wrist_flex`` move the row almost identically
    (-9.32 and -9.50 px/deg) while changing apparent size differently (+0.0096 and
    +0.0132 per deg), so the combination that holds the row FIXED nets only 0.0034 of
    log-size per degree. Closing a 0.34 gap that way needs 101 degrees of coordinated
    motion. The servo aimed to 15px and then correctly reported that it could not close
    range without giving up the aim — it was right, and the aim was the problem.

    Nothing actually requires a particular row. What matters is that the object stays
    IN VIEW while the arm closes in, and the grasp point comes from a measurement taken
    at the end, not from where the box sat. So the row contributes zero error anywhere
    inside ``view_band`` and only pushes back when the object nears a frame edge, which
    leaves the size row free to do its job.

    The COLUMN stays a target: bearing has to be right, it is achievable at any range,
    and nothing else supplies it.

    ``aim_v`` PUTS THE ROW BACK, and the final descent needs it. Once the gripper is
    nearly on the object the fingertip row stops being unreachable and starts being the
    whole point — the object has to come DOWN onto the jaws, and nothing else asks it
    to. Leaving the row free through the descent is why an otherwise working descent
    never descended: it drove column and size only, and the size target alone gave it no
    reason to lower the gripper (measured, size 0.77 -> 0.74 with the tip parked at
    z=6.9cm). Free row while travelling, targeted row on arrival.
    """
    v = seen.cy
    if aim_v is not None:
        e_v = float(aim_v) - v
    else:
        lo, hi = float(view_band[0]), float(view_band[1])
        e_v = (lo - v) if v < lo else ((hi - v) if v > hi else 0.0)
    return np.array([
        float(aim_u) - seen.cx,
        e_v,
        math.log(max(float(target_height_px), 1.0)) - math.log(max(seen.height, 1.0)),
    ], dtype=np.float64)
