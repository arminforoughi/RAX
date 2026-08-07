"""Perception — connections to vision; turns sensors into world understanding.

Nothing here knows about a specific robot. Each piece takes the seams it needs
(a camera geometry, a set of class priors, a table plane) and is usable by any arm.

Public surface:
    CameraGeometry / CameraPose    pixels <-> base frame, for a wrist or fixed camera
    EyeInHand / FixedCamera        the two camera mounts
    Plane / fit_plane              the measured table surface
    ObjectPriors / PRIORS          per-class size and shape priors
    Localizer / Fix                detection -> base-frame position, three strategies
    ObjectMeasurer                 monocular size, height and yaw from one frame
    fit_reprojection / fit_consistency   hand-eye calibration from arm motion

Subpackages:
    vision/       Camera frame pipeline: capture -> detection -> labeled scene.
    depth_cloud/  Fuse depth + detections into a point cloud and recover the
                  3D position of each object (x, y, depth) for manipulation
                  and navigation.
"""

from __future__ import annotations

from perception.camera_geometry import (
    CameraGeometry, CameraPose, EyeInHand, FixedCamera, intrinsics_from_dict,
    parse_tf, tf_to_string)
from perception.locate import (
    ApparentSizeLocalizer, Fix, Localizer, PlaneRayLocalizer, StereoLocalizer, chain)
from perception.handeye import (
    HandEyeFit, HandEyeSample, fit_consistency, fit_reprojection,
    load_hand_eye, save_hand_eye)
from perception.measure import ObjectMeasurer
from perception.object_priors import PRIORS, ObjectPriors
from perception.table_plane import Plane, PlaneFit, fit_plane

__all__ = [
    "CameraGeometry", "CameraPose", "EyeInHand", "FixedCamera",
    "intrinsics_from_dict", "parse_tf", "tf_to_string",
    "Plane", "PlaneFit", "fit_plane",
    "ObjectPriors", "PRIORS",
    "Localizer", "Fix", "ApparentSizeLocalizer", "PlaneRayLocalizer",
    "StereoLocalizer", "chain",
    "ObjectMeasurer",
    "HandEyeSample", "HandEyeFit", "fit_reprojection", "fit_consistency",
    "load_hand_eye", "save_hand_eye",
]
