"""Where the gripper is in the wrist image.

Ported from the lab checkout's `perception/jaws.py`. `pick.py` imports ``grasp_point``
and ``find_jaws`` for its display.

THESE ARE CONSTANTS, AND THAT IS THE POINT. `gripper_geometry.json` records them as
measured across a whole session, with the grasp point varying by +-1.8px horizontally,
and its comment names detecting them live as "the source of most of the erratic
behaviour": the white finger tape reads as a blue cap, so a live detector would
occasionally lock the robot onto its own gripper. The camera is bolted to the wrist, so
the fingers cannot move in the image -- there is nothing here to detect.

`find_jaws` therefore returns the measured pair. It exists so the display code has
something to call, not because anything is being searched for.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["grasp_point", "find_jaws", "fingers", "exclude_radius"]

_GEO = json.loads((Path(__file__).parent / "config" / "gripper_geometry.json").read_text())


def grasp_point(frame=None) -> tuple[float, float]:
    """The pixel a tube has to be driven onto to sit between the fingers."""
    return (float(_GEO["grasp_point"][0]), float(_GEO["grasp_point"][1]))


def fingers() -> list[tuple[float, float]]:
    return [tuple(float(v) for v in _GEO["finger_left"]),
            tuple(float(v) for v in _GEO["finger_right"])]


def exclude_radius() -> float:
    """How close to a fingertip a colour blob has to be before it is assumed to BE the
    fingertip. 70px on this mount, and it is what keeps the blue finger tape out of the
    cap list."""
    return float(_GEO["exclude_radius"])


def find_jaws(frame=None):
    """The fingertip pair. Measured, not detected -- see the module docstring."""
    return tuple(fingers())
