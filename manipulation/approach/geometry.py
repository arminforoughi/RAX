"""Where to aim during an approach, and how far to move each stage.

Pure functions of position and configuration — no robot, no camera, no clock — which
is what makes the approach's geometry checkable without hardware. The staging policy
they encode is the answer to a specific failure:

**One long move to the target is a dive.** If the mapped position is a little off,
nothing notices until the gripper is already there. Instead the distance is covered in
stages: each closes part of the remaining gap, then the caller looks again and refines
the target. Errors get corrected while there is still room to correct them.

Going 100% in one go was tried and was worse — it arrives with no margin left, so any
residual localization error lands as a miss. 90% then small re-centring leaves room to
fix things while still close.
"""

from __future__ import annotations

import numpy as np

__all__ = ["shift_right", "approach_target", "stage_step", "push_out_radial"]


def shift_right(xy, distance_m: float) -> np.ndarray:
    """Move a point sideways, to the right as seen from the base looking outward.

    The perpendicular is taken in the plane, so this stays correct at any bearing. If
    the gripper ends up on the object's LEFT instead, the base frame's y-axis is
    inverted relative to what this assumes.
    """
    p = np.asarray(xy, dtype=np.float64)
    r = float(np.hypot(p[0], p[1]))
    if r < 1e-6 or distance_m == 0.0:
        return p.copy()
    u = p / r
    return p + np.array([u[1], -u[0]]) * float(distance_m)


def approach_target(object_xy, *, back_m: float = 0.0, right_trim_m: float = 0.0
                    ) -> np.ndarray:
    """The hover position for an approach: short of the object, and to its right.

    ``back_m`` keeps the object in view instead of under the gripper; ``right_trim_m``
    keeps it on one known side of the frame so the centring step knows which way to
    correct. Refinements update the OBJECT position, never this offset.
    """
    p = np.asarray(object_xy, dtype=np.float64)
    r = float(np.hypot(p[0], p[1]))
    backed = p * ((r - float(back_m)) / r) if r > 1e-6 else p.copy()
    return shift_right(backed, right_trim_m)


def stage_step(tip_xy, target_xy, *, stage: int, total: int,
               first_frac: float = 0.9, max_first_m: float = 0.12
               ) -> tuple[np.ndarray | None, float]:
    """Where to move on this stage, and how far away the target still is.

    Returns ``(waypoint | None, remaining_m)``. ``None`` means the hover has been
    reached and the staging is done — the caller should stop rather than issue a
    zero-length move.
    """
    tip = np.asarray(tip_xy, dtype=np.float64)[:2]
    tgt = np.asarray(target_xy, dtype=np.float64)[:2]
    delta = tgt - tip
    dist = float(np.linalg.norm(delta))
    if dist <= 1e-6:
        return None, dist
    last = stage >= total - 1
    length = dist if last else min(dist * float(first_frac), float(max_first_m))
    return tip + delta / dist * length, dist


def push_out_radial(p, push_m: float) -> np.ndarray:
    """Move a base-frame point radially outward, away from the base's z-axis.

    A correction of last resort for a camera that sits behind the fingertips: a
    too-steep sightline hits the table too soon and reads every object too near. It
    must be applied to EVERY localization, initial and refined, or a raw re-measure
    drags the target back inward and undoes it.

    Default it to zero. Once the hand-eye transform is properly calibrated that offset
    lives in the transform's translation, and this fudge double-counts — which pushes
    the object out of reach.
    """
    p = np.asarray(p, dtype=np.float64).copy()
    r = float(np.hypot(p[0], p[1]))
    if r > 1e-3 and push_m != 0.0:
        p[0] += p[0] / r * float(push_m)
        p[1] += p[1] / r * float(push_m)
    return p
