"""Deriving approach parameters from the object and the camera, not from a dial.

Three constants in this stack are really answers to geometry questions that nobody
computed:

``PICK_GRASP_Z = 0.015``
    "Close the jaws 1.5 cm above the table" — correct for the 5 cm cube it was tuned
    on, wrong for everything else. On a 1 cm-thick remote it closes above the object;
    on a 23 cm bottle it grips the very bottom, which is the least stable place to
    hold one. The object's measured height answers this, and the map already has it.

``ALIGN_TOL_PX = 40``
    "Within 40 px is centred." But 40 px means different things at different ranges: a
    5 cm cube spans ~130 px at 20 cm and ~65 px at 40 cm, so a fixed tolerance is half
    an object at one distance and a whole object at another. What matters is the error
    as a fraction of the object's apparent size.

``TARGET_RIGHT_TRIM_M = 0.05``
    "Sit 5 cm to the object's right so it stays in view." Whether it stays in view
    depends on the camera's field of view and how close the gripper gets — both known.

None of these needs a robot to compute; they are functions of numbers the system
already measures. That makes them transfer to another arm, another camera, and another
object without being re-dialled.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "grasp_height", "hover_height", "align_tolerance_px", "apparent_width_px",
    "right_trim_for_visibility", "horizontal_fov_deg",
]

#: Keep at least this much daylight between the jaw tips and the table, and between the
#: grip point and the top of the object. Below this the jaws foul the surface on one
#: side or slide off on the other.
DEFAULT_CLEARANCE_M = 0.008

#: Grip this far up the object's height. Below the centre of mass so the object cannot
#: pivot out of the jaws, but clear of the table.
#:
#: The value is not a guess — it is recovered from the one grasp that was empirically
#: tuned on this rig: 1.5 cm on a 5.08 cm cube, i.e. 0.295. Anchoring here means the
#: object the constant was tuned on grasps exactly as before, and everything else
#: inherits the same *rule* instead of the same *number*. If a future rig re-tunes the
#: reference grasp, recompute this from it rather than nudging it.
REFERENCE_GRASP_M = 0.015          # PICK_GRASP_Z, tuned by hand
REFERENCE_OBJECT_H_M = 0.0508      # ...on the 5.08 cm cube
DEFAULT_GRASP_FRACTION = REFERENCE_GRASP_M / REFERENCE_OBJECT_H_M

#: "Centred" means the error is under this fraction of the object's apparent width.
#: A third is tight enough that the jaws straddle the object and loose enough that the
#: servo is not chasing detector jitter.
DEFAULT_ALIGN_FRACTION = 0.33

#: Tolerance is clamped into this band whatever the object's size implies — a very
#: small object must not demand sub-pixel centring, a very large one must not let the
#: gripper arrive a whole jaw-width off.
ALIGN_TOL_BOUNDS_PX = (12.0, 60.0)


def grasp_height(object_h_m: float, *, table_z_m: float = 0.0,
                 fraction: float = DEFAULT_GRASP_FRACTION,
                 clearance_m: float = DEFAULT_CLEARANCE_M) -> float:
    """Base-frame z at which to close the jaws on an object of this height.

    Grips below the object's mid-height so it cannot pivot out of the jaws as they
    close, while staying clear of the table. For an object too thin to have room for
    both, it grips at the middle and accepts the compromise — that is the honest answer
    for a flat object, rather than a clearance the geometry cannot provide.
    """
    h = max(float(object_h_m), 1e-4)
    if h <= 2.0 * clearance_m:
        return float(table_z_m + 0.5 * h)
    z = fraction * h
    z = min(max(z, clearance_m), h - clearance_m)
    return float(table_z_m + z)


def hover_height(object_h_m: float, *, table_z_m: float = 0.0,
                 standoff_m: float = 0.05, **kw) -> float:
    """Where to park before descending: clear of the object's top by the standoff.

    Measured from the object's TOP, not from the table, so a tall object is not
    approached through its own body.
    """
    return float(max(grasp_height(object_h_m, table_z_m=table_z_m, **kw) + standoff_m,
                     table_z_m + float(object_h_m) + standoff_m))


def apparent_width_px(geometry, object_size_m: float, range_m: float) -> float:
    """How many pixels across an object of this size looks at this range."""
    r = max(float(range_m), 1e-3)
    return float(geometry.fx * float(object_size_m) / r)


def align_tolerance_px(geometry, object_size_m: float, range_m: float, *,
                       fraction: float = DEFAULT_ALIGN_FRACTION,
                       bounds=ALIGN_TOL_BOUNDS_PX) -> float:
    """Pixel error that counts as centred, as a fraction of the object's apparent size.

    Scales with range the way the picture does, so the same physical accuracy is
    demanded whether the object is near or far — which a fixed pixel count cannot do.
    """
    tol = fraction * apparent_width_px(geometry, object_size_m, range_m)
    return float(min(max(tol, bounds[0]), bounds[1]))


def horizontal_fov_deg(geometry) -> float:
    """The camera's horizontal field of view, from its intrinsics."""
    w = float(getattr(geometry.intr, "width", 0) or 0)
    if w <= 0:
        return 0.0
    return float(2.0 * math.degrees(math.atan(0.5 * w / geometry.fx)))


def right_trim_for_visibility(geometry, object_size_m: float, closest_range_m: float, *,
                              margin_frac: float = 0.25, max_trim_m: float = 0.12) -> float:
    """Lateral offset that keeps the object inside the frame at closest approach.

    The gripper approaches until the object is ``closest_range_m`` away. At that range
    the half-frame covers a known width; sitting off to one side by less than that,
    minus room for the object itself and a margin, keeps it visible for the centring
    step instead of sliding out of view exactly when it is needed most.

    Returns 0 when the camera reports no frame width — better to sit on the line than
    to invent an offset from an unknown field of view.
    """
    w = float(getattr(geometry.intr, "width", 0) or 0)
    if w <= 0:
        return 0.0
    r = max(float(closest_range_m), 1e-3)
    half_frame_m = 0.5 * w * r / geometry.fx          # metres visible either side
    usable = half_frame_m - 0.5 * float(object_size_m)
    trim = usable * (1.0 - float(margin_frac))
    return float(min(max(trim, 0.0), max_trim_m))
