"""SLAM: build/maintain a map and localize for obstacle-aware navigation.

What exists today is the tabletop object map. It is deliberately NOT SLAM: an
eye-in-hand camera's pose is computed exactly from the servo angles rather than
estimated, and mapping with known poses is just mapping. The hard part it does solve is
association — deciding when two detections are the same physical object.

Public surface:
    ObjectMap              bird's-eye map of table objects, keyed by tag
    fit_rect_from_support  footprint rectangle from multi-bearing caliper readings
    yaw_blend              circular mean of two axis angles (mod 180)
"""

from __future__ import annotations

from rax.mobility.slam.object_map import (
    MapEntry,
    ObjectMap,
    fit_rect_from_support,
    sup_bin,
    yaw_blend,
)

__all__ = ["ObjectMap", "MapEntry", "fit_rect_from_support", "sup_bin", "yaw_blend"]
