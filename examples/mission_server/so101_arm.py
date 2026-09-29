"""The real SO-101, as a :class:`rax.pick.arm.Arm`, through the mission server's own primitives.

The mission server owns the serial bus and the camera, so this runs in its process and
calls into it (module ``ms``) instead of opening either device a second time.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from rax.manipulation.grip import settled
from rax.pick.arm import bearing_of, pan_for_bearing, with_pan

#: The jaw line in the wrist image, measured off a live frame (fingertips at about
#: (154,382) and (400,376)). Re-measure if the camera or the jaws are remounted.
JAW_AXIS_DEG = 179.0


class So101Arm:
    def __init__(self, ms, log=None, phase=None):
        self.ms = ms
        a = ms.ARM
        self.pan, self.roll, self.pitch_chain = a.pan_joint, a.roll_joint, tuple(a.pitch_chain)
        self.lo, self.hi = np.asarray(ms.J_LO, float), np.asarray(ms.J_HI, float)
        self.home = np.asarray(ms.HOME, float)
        self.jaw_axis_deg = JAW_AXIS_DEG
        self._log = log or ms.say
        self._phase = phase or (lambda n, note="": ms.say(f"[{n}] {note}"))
        #: While set, the base never faces further right than this bearing (deg): the
        #: racks are over there and picking has no business turning toward them.
        self.min_bearing_deg: float | None = None

    @property
    def jaw_uv(self):
        return self.ms.jaw_frame().centre_uv

    @property
    def roll_gain(self):
        return float(self.ms.jaw_frame().roll_gain) or 1.0

    # ---- motion and model ----
    def joints(self):
        return self.ms.observe(False)[0].astype(float)

    def move(self, q, speed=1.0, settle=0.2):
        q = np.asarray(q, float)
        if self.min_bearing_deg is not None:
            if math.degrees(bearing_of(self.tip(q))) < self.min_bearing_deg:
                q = with_pan(self, q, pan_for_bearing(
                    self, q, math.radians(self.min_bearing_deg)))
                self._log(f"        guardrail: held at {self.min_bearing_deg:+.0f}deg")
        fast = self.min_bearing_deg is None       # the pick keeps its own, proven pace
        self.ms.goto_smooth(self.ms._clamp_joints(q), settle=settle * (0.75 if fast else 1.0),
                            step=2.0 * speed * (1.35 if fast else 1.0))

    def tip(self, q):
        return np.asarray(self.ms._tip(q), float)

    def ik(self, seed, p, pitch, roll):
        return self.ms._ik_hold_pitch(np.asarray(seed, float), np.asarray(p, float),
                                      pitch, roll, ret_err=True)

    # ---- camera ----
    def frame(self):
        self.ms.observe(True)          # a fresh frame (and the overlay for the operator)
        rgb = self.ms.latest_rgb[0]
        return None if rgb is None else cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)

    def cast(self, uv, q):
        try:
            pt = self.ms.ray_to_table((float(uv[0]), float(uv[1])), self.ms.T_cam_of(q))
        except Exception:
            return None
        return None if pt is None else (float(pt[0]), float(pt[1]))

    def project(self, p, q):
        try:
            uv = self.ms.project_base(np.asarray(p, float), self.ms.T_cam_of(q))
        except Exception:
            return None
        return None if uv is None else (float(uv[0]), float(uv[1]))

    # ---- gripper ----
    def grip(self, pct):
        self.ms.send_joints(self.ms.observe(False)[0], gripper=float(pct))
        time.sleep(0.25)

    def release(self):
        self.grip(self.ms.ARM.gripper.place_open_pct)
        self.ms._set_carry(False)

    def close(self, from_pct):
        # a current rise while the jaws are still above 60 is the motor starting, not contact
        contact, _idle = self.ms.close_with_current(step=3.0, delay=0.09,
                                                    ignore_above_pct=60.0, from_pct=from_pct)
        return bool(contact)

    def grip_pos(self):
        return settled(lambda: float(self.ms.state.get("gripper") or 0.0),
                       tol=0.4, timeout=1.5, dt=0.06)

    # ---- operator ----
    def checkpoint(self):
        self.ms.checkpoint()

    def log(self, msg):
        self._log(msg)

    def phase(self, name, note=""):
        self._phase(name, note)
