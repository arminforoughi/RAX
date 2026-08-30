"""``RgbdCamera`` — RealSense, Femto, Kinect, or anything that hands over metric depth.

An RGB-D sensor is the easiest rig to support and the easiest to get subtly wrong, and
both reasons are about units and alignment:

**Units.** Depth arrives as uint16 in device units, not metres. RealSense defaults to
1 mm (scale 0.001), Kinect v2 to 1 mm, some Femto modes to 0.1 mm. Getting this wrong by
10x does not crash anything — it produces a robot that reaches confidently to the wrong
place. ``depth_scale`` is therefore a required-by-conscience argument with a documented
default, and zero is mapped to ``nan`` because 0 means "no return", not "touching the
lens".

**Alignment.** Colour and depth come from physically different lenses. If depth is not
aligned into the colour frame, pixel *(u,v)* in the RGB image does not name the same
world point in the depth image, and every back-projection is offset by the lens
separation. Every SDK ships an alignment step (``rs.align`` for RealSense); this class
asserts the two frames are the same size, which catches the common case of forgetting it.

The class takes a ``read`` callable rather than importing ``pyrealsense2``, so the SDK
stays an optional dependency and the adapter is testable with a synthetic reader.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from rax.models.depth.stereo import StereoIntrinsics
from rax.perception.camera_interface import Frame

__all__ = ["RgbdCamera", "realsense_reader"]

#: ``() -> (rgb HxWx3 uint8, depth HxW uint16|float)``
RgbdRead = Callable[[], tuple[np.ndarray, np.ndarray]]


class RgbdCamera:
    """A colour + aligned metric depth sensor."""

    kind = "rgbd"

    def __init__(
        self,
        read: RgbdRead,
        intrinsics: StereoIntrinsics,
        *,
        depth_scale: float = 0.001,   # RealSense/Kinect default: uint16 millimetres
        max_range_m: float = 6.0,
        close: Callable[[], None] | None = None,
    ):
        self._read = read
        self._intr = intrinsics
        self._scale = float(depth_scale)
        self._max_range = float(max_range_m)
        self._close = close

    # --- CameraInterface -----------------------------------------------------
    def frame(self) -> Frame:
        rgb, depth_raw = self._read()
        rgb = np.asarray(rgb)
        depth = np.asarray(depth_raw)
        if depth.shape[:2] != rgb.shape[:2]:
            raise ValueError(
                f"depth {depth.shape[:2]} and colour {rgb.shape[:2]} differ in size — "
                "align depth into the colour frame in your SDK (e.g. rs.align) before "
                "handing it over, or every pixel names a different world point in each")
        return Frame(rgb=rgb, depth_m=self._to_metres(depth),
                     intrinsics=self._intr, t=time.time())

    def intrinsics(self) -> StereoIntrinsics:
        return self._intr

    def close(self) -> None:
        if self._close is not None:
            try:
                self._close()
            except Exception:
                pass

    # --- internals -----------------------------------------------------------
    def _to_metres(self, depth: np.ndarray) -> np.ndarray:
        """Device units -> metres, with no-return and out-of-range pixels as ``nan``.

        ``nan`` rather than 0 because downstream masks are written as ``isfinite``: a
        zero would be read as an object touching the lens and would drag every centroid
        and median range toward the camera.
        """
        d = depth.astype(np.float32)
        if np.issubdtype(depth.dtype, np.integer):
            d *= self._scale
        invalid = (d <= 0.0) | (d > self._max_range)
        d[invalid] = np.nan
        return d


def realsense_reader(width: int = 640, height: int = 480, fps: int = 30):
    """Open a RealSense and return ``(RgbdCamera-ready read, intrinsics, depth_scale)``.

    Imported lazily and never at module scope, so ``pyrealsense2`` is an extra
    (``pip install rax[realsense]``) and this file imports fine without it.
    """
    import pyrealsense2 as rs  # noqa: PLC0415 - optional dependency, by design

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(cfg)

    align = rs.align(rs.stream.color)  # depth into the colour frame; see module docstring
    scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    ci = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intr = StereoIntrinsics(fx=ci.fx, fy=ci.fy, cx=ci.ppx, cy=ci.ppy,
                            baseline_m=0.0, width=ci.width, height=ci.height)

    def read():
        frames = align.process(pipeline.wait_for_frames())
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        return color, depth

    return read, intr, scale, pipeline.stop
