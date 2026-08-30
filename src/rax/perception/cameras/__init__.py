"""Camera adapters — the concrete sensors behind :class:`perception.CameraInterface`.

One factory, :func:`make_camera`, turns the ``camera`` block of an
:class:`~robots.profiles.ArmProfile` into a live device. Adding support for a sensor
this repo has never seen means writing one class with three methods and registering it
here — not touching the gaze engine, the localizer, or the object map.

    ============  =========================  =========================================
    ``kind``      class                      needs
    ============  =========================  =========================================
    ``stereo``    :class:`StereoCamera`      a rectified pair + baseline (OAK-D, ZED)
    ``rgbd``      :class:`RgbdCamera`        aligned metric depth (RealSense, Femto)
    ``mono``      :class:`MonoCamera`        anything ``cv2.VideoCapture`` opens
    ``synthetic`` :class:`SyntheticCamera`   nothing; renders any of the three
    ============  =========================  =========================================

Every concrete backend is imported lazily inside the factory, so this package imports
on a machine with no camera SDKs at all — which is the machine the tests run on.
"""

from __future__ import annotations

from rax.perception.camera_interface import CameraInterface, Frame
from rax.perception.cameras.mono import MonoCamera, intrinsics_from_fov
from rax.perception.cameras.rgbd import RgbdCamera
from rax.perception.cameras.stereo import StereoCamera, normalize_baseline
from rax.perception.cameras.synthetic import SyntheticCamera

__all__ = [
    "CameraInterface", "Frame",
    "MonoCamera", "RgbdCamera", "StereoCamera", "SyntheticCamera",
    "intrinsics_from_fov", "normalize_baseline", "make_camera",
]


def make_camera(profile, *, source: int | str | None = None, **kw) -> CameraInterface:
    """Open the camera an :class:`~robots.profiles.CameraProfile` describes.

    ``profile`` is an ``ArmProfile`` or a ``CameraProfile``; passing the arm is the
    common case, since a rig is normally named by its arm.
    """
    cam = getattr(profile, "camera", profile)
    kind = getattr(cam, "kind", "stereo")
    w, h, fps = int(cam.width), int(cam.height), int(cam.fps)

    if kind == "stereo":
        from rax.perception.cameras.stereo import oakd_reader

        read, intr, close = oakd_reader(width=w, height=h, fps=fps, **kw)
        return StereoCamera(read, intr, close=close)

    if kind == "rgbd":
        from rax.perception.cameras.rgbd import realsense_reader

        read, intr, scale, close = realsense_reader(width=w, height=h, fps=fps, **kw)
        return RgbdCamera(read, intr, depth_scale=scale, close=close)

    if kind == "mono":
        fx, fy, cx, cy = cam.intrinsics_fallback
        from rax.models.depth.stereo import StereoIntrinsics

        # A profile that carries real calibrated intrinsics should use them; the
        # sentinel default means "nobody has calibrated this camera", and the FOV
        # guess in MonoCamera is the honest answer there.
        intr = (StereoIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, baseline_m=0.0,
                                 width=w, height=h)
                if cam.intrinsics_fallback != (517.0, 517.0, 329.5, 231.4) else None)
        src = source if source is not None else kw.pop("index", 0)
        return MonoCamera(src, width=w, height=h, fps=fps, intrinsics=intr, **kw)

    raise ValueError(
        f"unknown camera kind {kind!r} for profile "
        f"{getattr(profile, 'name', '?')!r}; expected stereo|rgbd|mono")
