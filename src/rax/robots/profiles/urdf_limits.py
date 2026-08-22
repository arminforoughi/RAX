"""Read joint limits straight out of a URDF.

``stack_mission2.py:4661`` transcribes the SO-101's limits into a pair of literal
arrays, and the comment above them explains why they have to be there at all: an IK
that does not know the limits "is not an IK, it is a wish" — it returns
``elbow_flex=+162`` on a joint that stops at ``+96.8``, the servo silently clamps, and
the solver reports a 0.2 mm residual on a pose the robot cannot hold.

Transcribing them works for exactly one robot. Reading them from the URDF the arm
already ships with is what lets a new arm plug in, so that is what this does — with
stdlib ``xml.etree`` rather than a URDF library, because neither placo nor yourdfpy is
installed here and the ``<limit>`` tag needs no help to parse.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

# A joint of type "continuous" spins freely and carries no <limit> element. Give it a
# full turn either way rather than infinity, so downstream clipping stays finite.
CONTINUOUS_LIMIT_DEG = 180.0

_UNBOUNDED_TYPES = {"continuous"}
_FIXED_TYPES = {"fixed", "floating", "planar"}


def read_urdf_limits(urdf_path: str) -> dict[str, tuple[float, float]]:
    """Every actuated joint in the URDF -> its ``(lower, upper)`` limit in degrees."""
    root = ET.parse(str(urdf_path)).getroot()
    out: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        jtype = (joint.get("type") or "").strip().lower()
        if not name or jtype in _FIXED_TYPES:
            continue
        if jtype in _UNBOUNDED_TYPES:
            out[name] = (-CONTINUOUS_LIMIT_DEG, +CONTINUOUS_LIMIT_DEG)
            continue
        node = joint.find("limit")
        if node is None:
            continue
        lower, upper = node.get("lower"), node.get("upper")
        if lower is None or upper is None:
            continue
        # URDF angles are radians. Prismatic joints are metres and are left alone —
        # this stack is all-revolute, so flag rather than silently mis-convert.
        if jtype == "prismatic":
            out[name] = (float(lower), float(upper))
        else:
            out[name] = (math.degrees(float(lower)), math.degrees(float(upper)))
    return out


def joint_limits_deg(urdf_path: str, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` arrays in ``joint_names`` order, degrees.

    Raises if the URDF does not describe every joint asked for — a partial limit set
    is more dangerous than none, because the missing joint is the one that will drive
    into its stop.
    """
    limits = read_urdf_limits(urdf_path)
    missing = [n for n in joint_names if n not in limits]
    if missing:
        raise ValueError(
            f"{urdf_path} has no usable <limit> for {missing}; "
            f"it defines {sorted(limits)}"
        )
    lo = np.array([limits[n][0] for n in joint_names], dtype=np.float64)
    hi = np.array([limits[n][1] for n in joint_names], dtype=np.float64)
    bad = [n for n, l, h in zip(joint_names, lo, hi) if not l < h]
    if bad:
        raise ValueError(f"{urdf_path}: non-increasing limits on {bad}")
    return lo, hi
