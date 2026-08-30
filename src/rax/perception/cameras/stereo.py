"""``StereoCamera`` — a rectified pair from any baseline rig.

The stack grew up on an OAK-D, so this is the best-tested path. The adapter takes a
``read`` callable returning ``(left, right)`` rather than importing ``depthai``, which
keeps the OAK-D SDK an extra and lets a ZED, a synchronised pair of webcams, or a
recorded log satisfy the same interface.

Two facts about rectified pairs that are worth stating once, because getting either
wrong silently scales every distance:

* **Rectified, not raw.** ``z = fx * baseline / disparity`` assumes epipolar lines are
  image rows. Feeding raw pairs produces disparities that vary with vertical offset and
  a depth map that is wrong in a pose-dependent way. Every SDK exposes a rectified
  output; use it.
* **Baseline in metres.** depthai's ``getBaselineDistance()`` returns *centimetres*, and
  at least one wrapper passes it through labelled as metres. A stereo baseline is never
  more than a metre, so :func:`normalize_baseline` treats a larger value as centimetres
  and says so. This exact bug cost real debugging time on the SO-101 rig.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np

from rax.models.depth.stereo import StereoIntrinsics
from rax.perception.camera_interface import Frame

logger = logging.getLogger(__name__)

__all__ = ["StereoCamera", "normalize_baseline", "oakd_reader"]

#: ``() -> (left HxWx3|HxW, right same shape)``, both rectified
StereoRead = Callable[[], tuple[np.ndarray, np.ndarray]]


def normalize_baseline(baseline: float) -> float:
    """Coerce a stereo baseline to metres, correcting the classic cm mislabelling."""
    b = float(baseline)
    if b > 1.0:
        logger.warning(
            "[stereo] baseline of %.3f is larger than any real stereo rig — reading it "
            "as centimetres and converting to %.4f m", b, b / 100.0)
        return b / 100.0
    return b


class StereoCamera:
    """A rectified left/right pair plus the intrinsics of the left camera."""

    kind = "stereo"

    def __init__(
        self,
        read: StereoRead,
        intrinsics: StereoIntrinsics,
        *,
        close: Callable[[], None] | None = None,
    ):
        if intrinsics.baseline_m <= 0.0:
            raise ValueError(
                "a stereo camera needs a non-zero baseline_m; without it disparity "
                "cannot be converted to depth")
        self._read = read
        self._intr = intrinsics
        self._close = close

    # --- CameraInterface -----------------------------------------------------
    def frame(self) -> Frame:
        left, right = self._read()
        return Frame(rgb=np.asarray(left), right=np.asarray(right),
                     intrinsics=self._intr, t=time.time())

    def intrinsics(self) -> StereoIntrinsics:
        return self._intr

    def close(self) -> None:
        if self._close is not None:
            try:
                self._close()
            except Exception:
                pass


def oakd_reader(width: int = 1280, height: int = 800, fps: int = 30):
    """Open an OAK-D through lerobot and return ``(read, intrinsics, close)``.

    Lazily imported: ``lerobot`` and ``depthai`` are extras, and this module must stay
    importable on a machine that has neither.
    """
    from lerobot.cameras.oakd.camera_oakd import OAKDCamera  # noqa: PLC0415
    from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig  # noqa: PLC0415

    cam = OAKDCamera(OAKDCameraConfig(
        fps=fps, width=width, height=height, use_depth=False,
        export_stereo_rectified=True))
    cam.connect()

    si = cam.get_stereo_intrinsics()  # {fx, fy, cx, cy, baseline_m}
    left0, _ = cam.read_stereo_rectified()
    h0, w0 = np.asarray(left0).shape[:2]
    intr = StereoIntrinsics(
        fx=si["fx"], fy=si["fy"], cx=si["cx"], cy=si["cy"],
        baseline_m=normalize_baseline(si["baseline_m"]),
        # The rectified frame size is whatever the device actually returns, which is
        # not always what it was configured with. K is reported for the configured
        # size, so a mismatch here means the intrinsics need scaling, not trusting.
        width=w0, height=h0)
    return cam.read_stereo_rectified, intr, cam.disconnect
