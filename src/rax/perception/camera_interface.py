"""``CameraInterface`` — the sensor seam, sitting beside the arm seam.

``rax.manipulation.arms.arm_interface`` made the *arm* swappable. This file does the same
for the *camera*, and the two are deliberately separate: an SO-101 with an OAK-D on the
wrist and an SO-101 with a RealSense bolted to a head mast are the same arm and a
different sensor, so a rig should be describable without writing a new driver for each
combination.

The seam is one method. A camera hands over a :class:`Frame` — always an RGB image and
its intrinsics, plus *whatever else that rig happens to have*:

===================  ==================  ==========================================
Rig                  Frame carries       Depth comes from
===================  ==================  ==========================================
stereo (OAK-D, ...)  ``rgb`` + ``right`` :class:`StereoDepthSource` — a matcher runs
RGB-D (RealSense)    ``rgb`` + ``depth`` :class:`SensorDepthSource` — read it off
mono (any webcam)    ``rgb``             :class:`NoDepthSource` — see below
===================  ==================  ==========================================

A mono camera is not a degraded case to be apologised for. Two of the three range
strategies in :mod:`rax.perception.locate` never needed depth: ``PlaneRayLocalizer``
intersects the sightline with the measured table, and ``ApparentSizeLocalizer`` divides
a known class size by the apparent one. Those work on any camera that can see. What a
depth map buys is the third strategy and a denser cloud — an upgrade, not a
prerequisite. :class:`NoDepthSource` therefore returns ``None`` rather than raising, and
callers treat ``None`` as "use another strategy", which is what they already did on
frames where stereo matching failed.

Pose is deliberately NOT here. Where the camera is in base frame is a property of the
*mount*, not of the sensor, and it already has a home in
:class:`rax.perception.camera_geometry.CameraGeometry` — ``EyeInHand`` resolves it through
FK, ``FixedCamera`` holds it constant. A camera that reported its own pose would have to
know about the arm, which is the coupling this file exists to remove.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from rax.models.depth.stereo import ROI, StereoDepth, StereoIntrinsics, clamp_roi

__all__ = [
    "Frame", "CameraInterface", "DepthSource",
    "StereoDepthSource", "SensorDepthSource", "NoDepthSource", "make_depth_source",
]


@dataclass
class Frame:
    """One snapshot from a camera, whatever kind of camera it is.

    ``rgb`` is the only field every rig has, and it is the image detection runs on.
    ``right`` and ``depth_m`` are the optional extras that decide which
    :class:`DepthSource` can be built over this camera.
    """

    rgb: np.ndarray                      # HxWx3 RGB (or HxW mono), rectified
    intrinsics: StereoIntrinsics         # of ``rgb``; baseline_m is 0 for non-stereo
    right: np.ndarray | None = None      # rectified right frame, stereo rigs only
    depth_m: np.ndarray | None = None    # HxW metric depth (nan = invalid), RGB-D only
    t: float = field(default_factory=time.time)

    @property
    def has_stereo(self) -> bool:
        return self.right is not None and self.intrinsics.baseline_m > 0.0

    @property
    def has_depth(self) -> bool:
        return self.depth_m is not None

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` of ``rgb`` — from the pixels, not from the config.

        A camera that silently delivers a different resolution than it was asked for is
        a back-projection error, not a cosmetic one, so the frame is the authority.
        """
        h, w = np.asarray(self.rgb).shape[:2]
        return int(w), int(h)


@runtime_checkable
class CameraInterface(Protocol):
    """Minimal camera contract: hand over a frame, say what your optics are."""

    #: "stereo" | "rgbd" | "mono" — what extras :meth:`frame` populates.
    kind: str

    def frame(self) -> Frame:
        """The latest snapshot. Blocking; must be safe to call every tick."""
        ...

    def intrinsics(self) -> StereoIntrinsics:
        """The camera's own intrinsics, once connected."""
        ...

    def close(self) -> None:
        """Release the device. Idempotent."""
        ...


# --- depth strategies ---------------------------------------------------------------
@runtime_checkable
class DepthSource(Protocol):
    """Metric depth for a frame, however this rig happens to obtain it.

    ``depth_meters`` returns a full-size HxW map with ``nan`` for unknown pixels, or
    ``None`` when this rig cannot produce depth at all. The two are different answers:
    ``nan`` means "this pixel did not match", ``None`` means "do not wait for depth
    from me, use another strategy".
    """

    name: str
    available: bool

    def depth_meters(self, frame: Frame, *, roi: ROI | None = None) -> np.ndarray | None: ...


class StereoDepthSource:
    """Depth by running a stereo matcher over the frame's left/right pair."""

    available = True

    def __init__(self, stereo: StereoDepth):
        self.stereo = stereo
        self.name = f"stereo:{getattr(stereo, 'name', type(stereo).__name__)}"

    def depth_meters(self, frame: Frame, *, roi: ROI | None = None) -> np.ndarray | None:
        if not frame.has_stereo:
            return None
        return self.stereo.depth_meters(
            frame.rgb, frame.right, intr=frame.intrinsics, roi=roi)


class SensorDepthSource:
    """Depth read straight off an RGB-D sensor.

    The ROI is honoured by masking rather than by cropping: the sensor already paid for
    every pixel, so a crop would save nothing, and returning a full-size map keeps the
    contract identical to the stereo path — which is the point of having the seam.
    """

    name = "sensor"
    available = True

    def depth_meters(self, frame: Frame, *, roi: ROI | None = None) -> np.ndarray | None:
        if frame.depth_m is None:
            return None
        d = np.asarray(frame.depth_m, dtype=np.float32)
        if roi is None:
            return d
        w, h = frame.size
        x1, y1, x2, y2 = clamp_roi(roi, w, h)
        out = np.full_like(d, np.nan)
        out[y1:y2, x1:x2] = d[y1:y2, x1:x2]
        return out


class NoDepthSource:
    """A camera with no depth of any kind. Always ``None``, never an exception.

    Returning ``None`` (rather than refusing to construct the rig) is what makes a plain
    webcam a supported configuration: every caller already handles a depth miss, because
    stereo matching fails on textureless frames too.
    """

    name = "none"
    available = False

    def depth_meters(self, frame: Frame, *, roi: ROI | None = None) -> np.ndarray | None:
        return None


def make_depth_source(camera: CameraInterface | str,
                      stereo: StereoDepth | None = None) -> DepthSource:
    """Pick the depth strategy that matches a camera's kind.

    Accepts a camera or a bare kind string, so a rig can be resolved before any device
    is opened — which is how the headless tests exercise every branch.
    """
    kind = camera if isinstance(camera, str) else getattr(camera, "kind", "mono")
    if kind == "stereo":
        if stereo is None:
            raise ValueError(
                "a stereo camera needs a StereoDepth backend; "
                "pass one from rax.models.depth.make_stereo()")
        return StereoDepthSource(stereo)
    if kind == "rgbd":
        return SensorDepthSource()
    return NoDepthSource()
