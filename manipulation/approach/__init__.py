"""Approach — closing on a located object and grasping it.

The pieces here are deliberately independent of how the object was found and of which
arm is doing the reaching: they take a target position, a configuration, and the
kinematic and camera seams defined in ``manipulation.arms`` and ``perception``.

Contents:
    config.py     ApproachConfig — every tunable, with the knob-name API the UI and
                  the autotuner drive by name.
    geometry.py   Where to hover and how to stage the distance. Pure functions.
    visual_center.py  Final centring by eye, with gains measured by probing rather
                  than modelled — a hand-eye error can invert the raw pixel signs.
"""

from manipulation.approach.config import KNOBS, ApproachConfig, Knob
from manipulation.approach.geometry import (
    approach_target, push_out_radial, shift_right, stage_step)
from manipulation.approach.visual_center import (
    CenteringOps, CenteringResult, center_on_object)

__all__ = [
    "ApproachConfig", "Knob", "KNOBS",
    "approach_target", "shift_right", "push_out_radial", "stage_step",
    "center_on_object", "CenteringOps", "CenteringResult",
]
