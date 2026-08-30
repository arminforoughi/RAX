"""``MonoCamera`` — any camera OpenCV can open, which is very nearly all of them.

A USB webcam, a laptop lid camera, a GoPro over UVC, an RTSP stream from a robot's head
— if ``cv2.VideoCapture`` accepts it, this class turns it into a
:class:`~perception.camera_interface.CameraInterface`. That matters more than it
sounds: it is the difference between "this repo needs a $400 depth camera" and "point it
at whatever you already own".

Intrinsics are the one thing a webcam cannot tell you. The options, best first:

1. Calibrate it (``cv2.calibrateCamera`` on a chessboard) and pass ``fx, fy, cx, cy``.
2. Pass a horizontal field of view; ``fx = (w/2) / tan(hfov/2)``.
3. Accept the default 60 degree guess, which is right to within ~15% for most webcams.

Option 3 is deliberately allowed. A wrong focal length scales every distance by a
constant, and :mod:`perception.selfcal` exists precisely to measure and remove a
constant range scale from the robot's own motion — so an uncalibrated camera converges
to a calibrated one after a self-cal run rather than being a dead end.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from rax.models.depth.stereo import StereoIntrinsics
from rax.perception.camera_interface import Frame

__all__ = ["MonoCamera", "intrinsics_from_fov"]


def intrinsics_from_fov(width: int, height: int, hfov_deg: float = 60.0) -> StereoIntrinsics:
    """Pinhole intrinsics from an image size and a horizontal field of view.

    Square pixels and a centred principal point are assumed — both are true enough on
    consumer cameras that the error is far below the range error this stack already
    corrects for.
    """
    fx = (width / 2.0) / math.tan(math.radians(float(hfov_deg)) / 2.0)
    return StereoIntrinsics(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
                            baseline_m=0.0, width=int(width), height=int(height))


class MonoCamera:
    """One colour camera, no depth.

    Frames are pulled on a background thread and the newest one is kept. A robot
    control loop that blocks on ``VideoCapture.read()`` inherits the camera's latency
    *and* its buffering: OpenCV queues frames, so a loop slower than the sensor reads
    progressively staler images while believing they are current. Draining in the
    background and serving the latest keeps the age of a frame bounded by the sensor
    period rather than by the caller's.
    """

    kind = "mono"

    def __init__(
        self,
        source: int | str = 0,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        hfov_deg: float = 60.0,
        intrinsics: StereoIntrinsics | None = None,
        backend: int | None = None,
    ):
        import cv2

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(source if backend is None else source, *([] if backend is None else [backend]))
        if not self.cap.isOpened():
            raise RuntimeError(
                f"could not open camera {source!r}. On Linux check /dev/video*; on "
                "Windows try a different index, or pass backend=cv2.CAP_DSHOW")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))
        # A 1-frame driver buffer is the other half of the staleness fix above.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        ok, first = self.cap.read()
        if not ok or first is None:
            self.cap.release()
            raise RuntimeError(f"camera {source!r} opened but delivered no frames")
        h, w = first.shape[:2]
        # The frame is the authority on size — a camera that quietly ignores a
        # resolution request would otherwise poison every back-projection.
        self._intr = intrinsics or intrinsics_from_fov(w, h, hfov_deg)
        if (self._intr.width, self._intr.height) != (w, h):
            self._intr = StereoIntrinsics(
                fx=self._intr.fx * w / max(self._intr.width, 1),
                fy=self._intr.fy * h / max(self._intr.height, 1),
                cx=self._intr.cx * w / max(self._intr.width, 1),
                cy=self._intr.cy * h / max(self._intr.height, 1),
                baseline_m=0.0, width=w, height=h)

        self._lock = threading.Lock()
        self._latest = (self._to_rgb(first), time.time())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    # --- CameraInterface -----------------------------------------------------
    def frame(self) -> Frame:
        with self._lock:
            rgb, t = self._latest
        return Frame(rgb=rgb, intrinsics=self._intr, t=t)

    def intrinsics(self) -> StereoIntrinsics:
        return self._intr

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass

    # --- internals -----------------------------------------------------------
    def _to_rgb(self, bgr: np.ndarray) -> np.ndarray:
        return self._cv2.cvtColor(np.asarray(bgr), self._cv2.COLOR_BGR2RGB)

    def _pump(self) -> None:
        while not self._stop.is_set():
            ok, bgr = self.cap.read()
            if not ok or bgr is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = (self._to_rgb(bgr), time.time())
