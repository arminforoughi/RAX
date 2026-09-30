"""The X250 (Dynamixel protocol 2.0) with a USB wrist camera.

Joint units: the servos report ticks; the arm's calibration (lerobot's format, at
``x250_units.CALIB``) maps ticks to normalised units, and :mod:`x250_units` maps those to
URDF degrees. The gripper is kept in its normalised 0..100.

The gripper verdict is by POSITION: across 113 demonstrations a held tube stopped the
jaws above 31.9 and a close on nothing settled at 30.2, with no overlap.
"""

from __future__ import annotations

import json

import numpy as np

from rax.robots.base import Stopped, WristCameraArm
from rax.robots.dynamixel import DynamixelBus
from rax.robots.x250_units import CALIB, RANGE_0_100, load_units

__all__ = ["X250", "Stopped"]

#: Servo ids, measured by broadcast ping.
MOTOR_IDS = {"base": 2, "shoulder_2": 3, "elbow": 4, "wrist": 5, "tool": 6, "gripper": 7}


class X250(WristCameraArm):
    profile_name = "x250"
    jaw_axis_deg = 0.0          # NOT MEASURED on this camera: measure before trusting TWIST
    roll_gain = 1.0
    grip_levels = {"air": 30.2, "blocked": 31.2, "jammed": 50.0}
    close_by_current = False

    def __init__(self, port: str, handeye_file: str | None = None, log=print,
                 camera_index: int = 0):
        super().__init__(port, handeye_file, log)
        self.camera_index = camera_index
        self.bus = None
        self.cap = None
        self.units = load_units()
        self.calib = json.loads(CALIB.read_text())

    # ---- units ------------------------------------------------------------------
    def _norm(self, motor, ticks):
        c = self.calib[motor]
        frac = (float(ticks) - c["range_min"]) / max(c["range_max"] - c["range_min"], 1.0)
        return frac * 100.0 if motor in RANGE_0_100 else frac * 200.0 - 100.0

    def _ticks(self, motor, norm):
        c = self.calib[motor]
        frac = norm / 100.0 if motor in RANGE_0_100 else (norm + 100.0) / 200.0
        frac = min(1.0, max(0.0, frac))
        return int(round(c["range_min"] + frac * (c["range_max"] - c["range_min"])))

    # ---- hardware ---------------------------------------------------------------
    def _connect(self) -> None:
        import cv2
        self.bus = DynamixelBus(self.port, self.p.bus.baud).open()
        found = self.bus.ping()
        missing = [m for m, i in MOTOR_IDS.items() if i not in found]
        if missing:
            self.bus.close()
            raise RuntimeError(f"no answer on {self.port} from {missing}")
        present = self.bus.read_positions(MOTOR_IDS.values())
        for i in MOTOR_IDS.values():           # hold where it is, then energise: no snap
            self.bus.write(i, "goal_position", present[i])
            self.bus.write(i, "torque_enable", 1)
        self.cap = cv2.VideoCapture(self.camera_index, getattr(cv2, "CAP_DSHOW", 0))
        c = self.p.camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)

    def _disconnect(self) -> None:
        if self.bus is not None:
            for i in MOTOR_IDS.values():
                try:
                    self.bus.write(i, "torque_enable", 0)
                except Exception:
                    pass
            self.bus.close()
        if self.cap is not None:
            self.cap.release()

    def _torque(self, on: bool) -> None:
        present = self.bus.read_positions(MOTOR_IDS.values())
        for i in MOTOR_IDS.values():
            if on:
                self.bus.write(i, "goal_position", present[i])
            self.bus.write(i, "torque_enable", 1 if on else 0)

    def _read(self):
        import cv2
        raw = self.bus.read_positions(MOTOR_IDS.values())
        norm = {m: self._norm(m, raw[i]) for m, i in MOTOR_IDS.items()}
        q = np.array([self.units.to_deg(m, norm[m]) for m in self.motors])
        ok, bgr = self.cap.read() if self.cap is not None else (False, None)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None
        return q, norm["gripper"], rgb

    def _write(self, q_deg, gripper_pct) -> None:
        for m, deg in zip(self.motors, q_deg):
            self.bus.write(MOTOR_IDS[m], "goal_position",
                           self._ticks(m, self.units.to_norm(m, float(deg))))
        self.bus.write(MOTOR_IDS["gripper"], "goal_position",
                       self._ticks("gripper", float(gripper_pct)))
