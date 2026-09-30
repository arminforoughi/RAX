"""Any arm with a URDF, a wrist camera and a gripper, as a :class:`rax.pick.arm.Arm`.

:class:`WristCameraArm` does everything that is the same on every robot: kinematics and
IK from the URDF, the camera model, smooth moves, the gripper close, the stop button.
A robot subclasses it and supplies five methods that talk to its hardware:

    _connect()  _disconnect()  _read() -> (joints_deg, gripper_pct, rgb)
    _write(joints_deg, gripper_pct)   _gripper_current() -> float | None

See :mod:`rax.robots.so101` (lerobot, Feetech + OAK-D) and :mod:`rax.robots.x250`
(Dynamixel + a USB camera).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time

import numpy as np

from rax.kinematics import MotionLimits, make_ik, make_kinematics, quintic_waypoints
from rax.perception.camera_geometry import (
    CameraGeometry,
    EyeInHand,
    intrinsics_from_dict,
    parse_tf,
)
from rax.robots.profiles import load_profile


class Stopped(Exception):
    """The operator pressed stop."""


def settled(read, tol=0.4, timeout=1.5, dt=0.06) -> float:
    """Read a value once it has stopped changing (mid-close the jaws pass every value)."""
    last = float(read())
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(dt)
        v = float(read())
        if abs(v - last) <= tol:
            return v
        last = v
    return last


class WristCameraArm:
    """The robot-independent half of a driver. Subclasses talk to the hardware."""

    #: Profile name (see rax.robots.profiles).
    profile_name = ""
    #: Image angle of the line the jaws close along, degrees.
    jaw_axis_deg = 0.0
    #: Image degrees the scene turns per degree of wrist roll.
    roll_gain = 1.0
    #: Where the jaws stop, in this gripper's own percent: shut on air, stopped by
    #: something (holding), stopped too early (a false contact), wide enough for two.
    grip_levels = {"air": 1.2, "blocked": 3.5, "jammed": 36.0, "two": 16.0}
    #: Close by current (stop when it rises) or by position only.
    close_by_current = True
    #: What this arm's pick was tuned to: PickConfig fields (see PickConfig.for_arm).
    pick_tuning: dict = {}

    def __init__(self, port: str, handeye_file: str | None = None, log=print):
        self.p = load_profile(self.profile_name)
        self.port = port
        self.log_fn = log
        self.phase_fn = lambda name, note="": log(f"[{name}] {note}")
        self.pan, self.roll = self.p.pan_joint, self.p.roll_joint
        self.pitch_chain = tuple(self.p.pitch_chain)
        self.lo, self.hi = self.p.limits()
        self.home = np.array(self.p.home_deg, float)
        self.motors = list(self.p.joint_names)
        self.jaw_uv = tuple(self.p.gripper.hand_uv)
        self.tip_uv = tuple(self.p.gripper.hand_uv)   # where the fingertip frame appears
        self.table_z = float(self.p.table_z_m)

        self.kin = make_kinematics(self.p.urdf_path, self.p.ee_frame, self.motors)
        self._ik = make_ik(self.kin, self.p)
        tf = self.p.camera.extrinsics
        if handeye_file and os.path.exists(handeye_file):
            with open(handeye_file, encoding="utf-8") as f:
                tf = json.load(f)["tf"]
        c = self.p.camera
        fallback = c.intrinsics_fallback or (c.width, c.width, c.width / 2, c.height / 2)
        self.geom = CameraGeometry(
            intrinsics_from_dict(dict(zip(("fx", "fy", "cx", "cy"), fallback)),
                                 width=c.width, height=c.height),
            EyeInHand(lambda q: self.kin.forward_kinematics(q), parse_tf(tf)))
        #: False when there is no hand-eye transform: the pick then works from the
        #: picture alone and does not trust a cast onto the table.
        self.has_handeye = any(abs(float(v)) > 1e-9 for v in tf.split(","))
        self._limits = MotionLimits.from_profile(self.p)

        self.io_lock = threading.RLock()
        self.stop_flag = threading.Event()
        self.rgb = None                    # the latest frame, RGB
        self.q = self.home.copy()
        self.gripper_pct = 0.0
        self.min_bearing_deg: float | None = None   # a guardrail, set by the caller
        self.pace = 1.0                    # speed multiplier on every move
        self.relaxed = False

    # ---- hardware: a subclass implements these ------------------------------------
    def _connect(self) -> None:
        raise NotImplementedError

    def _disconnect(self) -> None:
        raise NotImplementedError

    def _read(self):
        """-> (joints in degrees, gripper percent, RGB frame)."""
        raise NotImplementedError

    def _write(self, q_deg, gripper_pct) -> None:
        raise NotImplementedError

    def _gripper_current(self):
        return None

    def _torque(self, on: bool) -> None:
        """Energise or release every servo, holding the present pose (no snap)."""
        raise NotImplementedError

    # ---- lifecycle --------------------------------------------------------------
    def connect(self) -> None:
        self._connect()
        self.observe()

    def disconnect(self) -> None:
        self.stop_flag.set()
        time.sleep(0.5)                   # let readers fall out before the camera goes
        try:
            self._disconnect()
        except Exception as e:
            self.log_fn(f"disconnect: {e}")

    # ---- raw I/O ------------------------------------------------------------------
    def observe(self, check_stop: bool = True):
        """Read the joints, the gripper and a frame."""
        if check_stop:
            self.checkpoint()
        with self.io_lock:
            q, g, rgb = self._read()
        self.q = np.asarray(q, float)
        self.gripper_pct = float(g)
        if rgb is not None:
            self.rgb = np.asarray(rgb)
        return self.q.copy()

    def relax(self) -> None:
        """Fold home, then cut torque so nothing is held and nothing heats up.
        The next motion wakes the arm."""
        self.move(self.home, speed=1.0, settle=0.4)
        with self.io_lock:
            self._torque(False)
        self.relaxed = True
        self.log("arm relaxed: torque off")

    def send(self, q, gripper=None) -> None:
        if self.relaxed:
            with self.io_lock:
                self._torque(True)
            self.relaxed = False
            self.log("arm awake")
        with self.io_lock:
            self._write(np.asarray(q, float), self.gripper_pct if gripper is None else gripper)

    # ---- rax.pick.Arm ----------------------------------------------------------------
    def joints(self):
        return self.observe()

    def move(self, q, speed=1.0, settle=0.2):
        """A smooth (quintic) move. While ``min_bearing_deg`` is set, the base never
        turns the fingertip further right than that bearing."""
        from rax.pick.arm import bearing_of, pan_for_bearing, with_pan
        q = np.clip(np.asarray(q, float), self.lo, self.hi)
        if self.min_bearing_deg is not None and \
                math.degrees(bearing_of(self.tip(q))) < self.min_bearing_deg:
            q = with_pan(self, q, pan_for_bearing(self, q, math.radians(self.min_bearing_deg)))
            self.log(f"        guardrail: held at {self.min_bearing_deg:+.0f}deg")
        q0 = self.observe()
        grip = self.gripper_pct
        limits = self._limits.scaled(speed * self.pace)
        waypoints, _ = quintic_waypoints(q0, q, limits)
        t0 = time.time()
        for k, qk in enumerate(waypoints):
            self.checkpoint()
            self.send(qk, gripper=grip)
            wait = t0 + (k + 1) * limits.dt_s - time.time()
            if wait > 0:
                time.sleep(wait)
        time.sleep(settle / self.pace)

    def tip(self, q):
        return np.asarray(self.kin.forward_kinematics(np.asarray(q, float)))[:3, 3]

    def ik(self, seed, p, pitch, roll):
        return self._ik.solve(np.asarray(seed, float), np.asarray(p, float),
                              pitch_deg=pitch, roll_deg=roll)

    def frame(self):
        import cv2
        self.observe()
        return None if self.rgb is None else cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)

    def cast(self, uv, q):
        if not self.has_handeye:
            return None
        pt = self.geom.ray_to_plane(uv, self.geom.T_base_cam(np.asarray(q, float)), self.table_z)
        return None if pt is None else (float(pt[0]), float(pt[1]))

    def project(self, p, q):
        if not self.has_handeye:
            return None
        return self.geom.project(np.asarray(p, float), self.geom.T_base_cam(np.asarray(q, float)))

    def grip(self, pct):
        self.send(self.observe(), gripper=float(pct))
        time.sleep(0.25)

    def release(self):
        self.grip(self.p.gripper.place_open_pct)

    def close(self, from_pct):
        """Close in small steps; with ``close_by_current``, stop when the current rises.
        True on a current-sensed contact. A rise while the jaws are still above 60% open
        is the motor starting, not an object, and is ignored."""
        g = self.p.gripper
        idle = [c for c in (self._gripper_current() for _ in range(5)) if c is not None]
        i_idle = float(np.mean(idle)) if idle else 0.0
        pct = float(from_pct)
        step = g.close_step_pct if not self.close_by_current else 3.0
        while pct > g.closed_pct:
            self.checkpoint()
            pct = pct - step if self.close_by_current else max(g.closed_pct, pct - step)
            q = self.observe()
            self.send(q, gripper=pct)
            time.sleep(0.09)
            if not self.close_by_current:
                continue
            c = self._gripper_current()
            if pct <= 60.0 and c is not None and abs(c - i_idle) >= g.contact_current_delta:
                self.send(q, gripper=max(0.0, pct - g.squeeze_extra_pct))
                time.sleep(0.18)
                return True
        return False

    def grip_pos(self):
        return settled(lambda: (self.observe(), self.gripper_pct)[1],
                       tol=0.4, timeout=1.5, dt=0.06)

    def checkpoint(self):
        if self.stop_flag.is_set():
            raise Stopped("stopped by user")

    def log(self, msg):
        self.log_fn(msg)

    def phase(self, name, note=""):
        self.phase_fn(name, note)
