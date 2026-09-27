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

__all__ = ["shift_right", "approach_target", "stage_step", "push_out_radial",
           "cap_reach", "stage_trim"]


def stage_trim(right_trim_m: float, *, stage: int, total: int,
               final_frac: float = 0.3, final_m: float | None = None) -> float:
    """How much lateral trim this stage should hold, decaying over the approach.

    The trim keeps the object to one side of the frame so it does not vanish under the
    gripper mid-transit. That is worth the most on the first hop -- far out, moving
    fast, the object small in frame -- and less on the last one, where every extra
    centimetre is pixel error the centring servo then has to undo.

    ``final_m`` is where the decay LANDS, in metres, and it is what you want: the
    lateral offset the grasp itself needs (approach.derive.grasp_bias_m). Decaying to a
    FRACTION instead -- which is what ``final_frac`` does, and all this used to do --
    walks the gripper onto the object's centre line and leaves the centring servo to
    shove it back out sideways at the hover, with the jaws already beside the object.
    Observed on the rig as "it goes from middle, then goes to right, which makes it
    push the object away". Landing the decay on the grasp offset means the approach and
    the servo agree, and the gripper never crosses the object at all.

    Linear from the full trim at stage 0 to the floor at the last stage. With one
    stage there is no later hop to decay toward, so that stage is the grasp: it takes
    ``final_m`` when given, and the full trim otherwise.
    """
    start = float(right_trim_m)
    end = float(final_m) if final_m is not None else start * float(final_frac)
    if total <= 1:
        return end if final_m is not None else start
    f = min(max(float(stage) / float(total - 1), 0.0), 1.0)
    return start + (end - start) * f


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
               first_frac: float = 0.9, max_first_m: float = 0.12,
               commit: bool = True) -> tuple[np.ndarray | None, float]:
    """Where to move on this stage, and how far away the target still is.

    Returns ``(waypoint | None, remaining_m)``. ``None`` means the hover has been
    reached and the staging is done — the caller should stop rather than issue a
    zero-length move.

    ``commit`` is what lets the LAST stage spend the whole remaining distance. Pass
    False while nothing has confirmed the target since it was first estimated: the
    full-distance step is only safe once a fresh sighting has agreed with where the
    object is, and unconfirmed it is the dive this staging exists to avoid. Held back,
    every stage keeps its margin and the arm stops just short instead of just past.
    """
    tip = np.asarray(tip_xy, dtype=np.float64)[:2]
    tgt = np.asarray(target_xy, dtype=np.float64)[:2]
    delta = tgt - tip
    dist = float(np.linalg.norm(delta))
    if dist <= 1e-6:
        return None, dist
    last = stage >= total - 1 and commit
    length = dist if last else min(dist * float(first_frac), float(max_first_m))
    return tip + delta / dist * length, dist


def cap_reach(xy, max_r_m: float) -> np.ndarray:
    """Pull a fix back to ``max_r_m`` from the base, keeping its BEARING.

    The two halves of a close-up re-measure are not equally trustworthy. The bearing is
    where the object sits in the frame, and it is solid. The range is how WIDE its box
    is — and a box narrowed by an occluding gripper, a clipped edge or a shadow reads
    too FAR away. That error has a sign: occlusion only ever removes pixels, so the bias
    is outward, never inward.

    Uncapped, a staged approach compounds it. Measured on a real pick, the target went
    r = 41.0 -> 44.9 -> 46.5 cm across three stages while the cube sat at 43.3, each
    stage stepping further out on the last stage's over-estimate, until the arm was past
    the object with it out of frame.

    Keeping the bearing and capping the radius takes the trustworthy half of the
    measurement and drops the other. The cap belongs on the TOTAL, measured from the
    original estimate, or the creep just accumulates one permitted step at a time.
    """
    p = np.asarray(xy, dtype=np.float64)[:2].astype(np.float64)
    r = float(np.hypot(p[0], p[1]))
    if r <= float(max_r_m) or r < 1e-6:
        return p.copy()
    return p * (float(max_r_m) / r)


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
