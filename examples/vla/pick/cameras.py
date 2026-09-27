"""Finding and pinning the cameras.

Ported from the lab checkout's `perception/cameras.py`. `pick.py` calls
``find_by_motion(port)`` when an index is not given, and ``force_mode(indices)`` before
handing the devices to the robot.

READ pick.py's OWN WARNING BEFORE USING find_by_motion: the motion probe identifies the
wrist camera reliably (a 45x margin) but "leaves one device in a mode the robot's connect
then refuses", so a production run should pass --wrist-cam and --top-cam explicitly. On
this rig those are **wrist=1, top=2**, established by opening each index and looking at
the frame: index 1 shows the mat with the jaws in shot, index 2 shows the whole bench
including the arm, index 0 is a third camera pointed at clutter. The README's defaults
(0 and 1) are the Mac's numbering.

THE PROBE MOVES THE ARM, which is why it is not the default here. It also requires a
calibration to command a pose at all, so on an uncalibrated arm it reports what it can
and declines rather than guessing.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["find_by_motion", "force_mode", "open_index", "describe"]

WIDTH, HEIGHT = 640, 480


def open_index(i: int, width: int = WIDTH, height: int = HEIGHT):
    """Open one index, preferring DirectShow on Windows.

    The default backend opens these UVC devices but frequently negotiates a mode that
    then delivers no frames, which reads downstream as a dead camera rather than a
    backend problem.
    """
    cap = cv2.VideoCapture(i, getattr(cv2, "CAP_DSHOW", 0))
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def force_mode(indices, width: int = WIDTH, height: int = HEIGHT) -> dict:
    """Pin each index to 640x480 and report what it actually came back as.

    `pick.py` calls this because the probe "just released these devices, and one of them
    comes back in 1920 mode, which makes the robot's connect fail outright". Pinning is
    cheap insurance whether or not the probe ran.
    """
    modes = {}
    for i in indices:
        if i is None or int(i) < 0:
            continue
        cap = open_index(int(i), width, height)
        if cap is None:
            modes[int(i)] = None
            continue
        ok, frame = cap.read()
        modes[int(i)] = (frame.shape[1], frame.shape[0]) if ok and frame is not None else None
        cap.release()
    return modes


def describe(max_index: int = 6) -> dict:
    """``{index: (w, h, mean_brightness)}`` for every index that opens. Touches no arm."""
    out = {}
    for i in range(max_index):
        cap = open_index(i)
        if cap is None:
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            out[i] = (frame.shape[1], frame.shape[0], float(frame.mean()))
        cap.release()
    return out


def find_by_motion(port: str, max_index: int = 6, nudge: float = 6.0):
    """``(top, wrist, diffs)`` -- which camera moves when the arm does. MOVES THE ARM.

    The wrist camera is carried by the arm, so when the arm moves its whole image
    changes; an overhead camera sees only the arm itself move through a mostly static
    scene. Scoring by CHANGE rather than by content is deliberate and `pick.py` records
    why: content scoring once put the operator's FaceTime camera in the wrist slot and a
    whole run executed against a picture of his face.
    """
    from x250_driver import X250Follower, X250FollowerConfig

    robot = X250Follower(X250FollowerConfig(port=port, cameras={}))
    robot.connect()
    try:
        if not robot.is_calibrated:
            raise RuntimeError(
                "the motion probe needs to command a pose, and this arm has no "
                "calibration — pass --wrist-cam and --top-cam explicitly instead "
                "(on this rig: --wrist-cam 1 --top-cam 2)")
        caps = {i: open_index(i) for i in range(max_index)}
        caps = {i: c for i, c in caps.items() if c is not None}
        before = {}
        for i, c in caps.items():
            ok, f = c.read()
            if ok and f is not None:
                before[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)

        pose = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
        robot.send_action({**pose, "base.pos": pose["base.pos"] + nudge})
        time.sleep(1.2)

        diffs = {}
        for i, c in caps.items():
            ok, f = c.read()
            if ok and f is not None and i in before:
                after = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
                diffs[i] = float(np.abs(after - before[i]).mean())
        robot.send_action(pose)
        time.sleep(0.8)
        for c in caps.values():
            c.release()

        if not diffs:
            return None, None, {}
        order = sorted(diffs, key=lambda i: -diffs[i])
        wrist = order[0]
        top = order[1] if len(order) > 1 else None
        return top, wrist, diffs
    finally:
        robot.disconnect()


# --------------------------------------------------------------------------------
# Choosing the cameras: look, then pass the numbers in.
# --------------------------------------------------------------------------------
# WHY THERE IS NO AUTOMATIC WRIST DETECTOR HERE. There was, twice, and both versions
# were wrong in the dangerous direction -- they NAMED a camera confidently and named the
# wrong one. The first scored brightness at the fingertip positions and ranked the
# overhead bench view (0.93) above the wrist (0.24), because a pale bench is bright
# everywhere. The second scored "lower-middle darker than the rest, with some white in
# it" and passed a camera pointed at a tripod once its exposure drifted. The measured
# reason the obvious test fails: the fingertip positions in gripper_geometry.json are
# the tips themselves, which are DARK (21 and 24 of 255 in the wrist view) -- the white
# tape is beside them, not on them.
#
# An index is not an identity on this rig anyway: these Sonix cameras all report the
# same hardcoded serial SN0001, so Windows cannot tell two of them apart and they swap
# indices and drop off. The reliable procedure is to look at a frame from each index and
# pass the numbers explicitly, which is also what pick.py's README advises.

def preview(indices=range(6), out_dir="."):
    """Save one frame per index and report what each looks like. Moves nothing.

    Run this, look at the files, and pass the index whose frame shows the mat with the
    gripper in the bottom of the picture as --wrist-cam.
    """
    from pathlib import Path as _P

    out = {}
    for i in indices:
        cap = open_index(int(i))
        if cap is None:
            logger.info("index %d: unavailable", i)
            continue
        frame = None
        for _ in range(4):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
            time.sleep(0.05)
        cap.release()
        time.sleep(0.3)
        if frame is None:
            logger.info("index %d: opened but gave no frame", i)
            continue
        path = str(_P(out_dir) / f"camera_{i}.jpg")
        cv2.imwrite(path, frame)
        out[int(i)] = path
        logger.info("index %d: %dx%d mean %.0f -> %s",
                    i, frame.shape[1], frame.shape[0], frame.mean(), path)
    return out
