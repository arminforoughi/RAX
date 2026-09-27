"""The operator's window: what the controller is looking at, while it looks at it.

Ported from the lab checkout's `perception/ui.py`. `pick.py` calls ``show(...)`` every
tick with whatever it has and ``close()`` at the end, and wraps both in a bare
``except Exception`` -- so this must never raise, and a headless machine must simply get
no window rather than a crashed pick.

IT DRAWS THE CONTROLLER'S VIEW, NOT THE DETECTOR'S. `pick.py` passes an already-filtered
cap list for a specific reason recorded in its comments: the jaws' own tape reads as a
blue cap, and showing the raw detections "made the display look like it was tracking the
gripper even when the servo had correctly locked the tube". A debug window that disagrees
with the controller is worse than none, because it gets believed.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["show", "close"]

WINDOW = "X250 pick"
_opened = [False]

_COLOUR = {"green": (80, 220, 80), "gold": (60, 200, 245), "blue": (240, 170, 60)}


def _panel(frame, w=640, h=480):
    if frame is None:
        return np.zeros((h, w, 3), np.uint8)
    return cv2.resize(frame, (w, h)) if frame.shape[1] != w or frame.shape[0] != h else frame.copy()


def show(wrist, top=None, caps=(), jaws_pair=None, grasp=None, chosen=None, target=None,
         phase="", gripper=None, err=None, pose=None, note="", axis=None, twist=None):
    """Draw one frame. Never raises."""
    try:
        w = _panel(wrist)

        # the grid the operator reasons in: cap in a cell, jaws in a cell, make them match
        for c in range(1, 10):
            x = int(c * w.shape[1] / 10)
            cv2.line(w, (x, 0), (x, w.shape[0]), (45, 45, 45), 1)
        for r in range(1, 8):
            y = int(r * w.shape[0] / 8)
            cv2.line(w, (0, y), (w.shape[1], y), (45, 45, 45), 1)

        if jaws_pair:
            for fx, fy in jaws_pair:
                cv2.circle(w, (int(fx), int(fy)), 8, (200, 200, 200), 2)
            cv2.line(w, tuple(int(v) for v in jaws_pair[0]),
                     tuple(int(v) for v in jaws_pair[1]), (200, 200, 200), 1)
        if grasp:
            cv2.drawMarker(w, (int(grasp[0]), int(grasp[1])), (0, 255, 255),
                           cv2.MARKER_CROSS, 22, 2)
        for c in caps:
            col = _COLOUR.get(getattr(c, "colour", ""), (180, 180, 180))
            cv2.circle(w, (int(c.x), int(c.y)), 11, col, 2)
        if chosen is not None:
            cx, cy = int(chosen[0]), int(chosen[1])
            cv2.circle(w, (cx, cy), 17, (0, 255, 0), 3)
            if grasp:
                cv2.line(w, (cx, cy), (int(grasp[0]), int(grasp[1])), (0, 255, 0), 2)
        if target is not None:
            cv2.drawMarker(w, (int(target[0]), int(target[1])), (255, 120, 255),
                           cv2.MARKER_TILTED_CROSS, 18, 2)
        if axis is not None and getattr(axis, "centre", None):
            import math

            a = math.radians(axis.angle_deg)
            cx, cy = axis.centre
            dx, dy = math.cos(a) * axis.length / 2, math.sin(a) * axis.length / 2
            cv2.line(w, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)),
                     (255, 255, 0), 2)

        lines = [f"{phase}"]
        if gripper is not None:
            lines.append(f"grip {gripper:.1f}")
        if err is not None:
            lines.append(f"err {err}")
        if twist is not None:
            lines.append(f"twist {twist:+.0f}d")
        if pose:
            lines.append(" ".join(f"{k[:2]}{v:+.0f}" for k, v in pose.items()))
        if note:
            lines.append(str(note))
        y = 22
        for t in lines:
            cv2.putText(w, str(t)[:78], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(w, str(t)[:78], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 20

        canvas = np.hstack([w, _panel(top)]) if top is not None else w
        cv2.imshow(WINDOW, canvas)
        cv2.waitKey(1)
        _opened[0] = True
    except Exception:
        pass


def close():
    try:
        if _opened[0]:
            cv2.destroyWindow(WINDOW)
            cv2.waitKey(1)
            _opened[0] = False
    except Exception:
        pass
