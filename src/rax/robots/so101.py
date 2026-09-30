"""The SO-101 with its wrist OAK-D, as a :class:`rax.pick.arm.Arm`.

Talks to the servos and the camera through lerobot (imported at connect time only).
The lessons from running this rig for weeks are kept, and only those:

* the gripper servo answers too slowly for lerobot's handshake ping, so it is skipped;
* a servo that latched its overload flag is cleared with a raw torque cycle first;
* a bus that stops answering mid-run is closed and reopened, not given up on;
* the camera must be released on exit, or the OAK-D stays booted with no owner and
  the next start fails with "No available devices" until it is unplugged.
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


class Stopped(Exception):
    """The operator pressed stop."""


class So101:
    """One SO-101 arm, its wrist camera and its gripper."""

    #: The line the jaws close along, in the wrist image: fingertips measured at about
    #: (154,382) and (400,376) on a live frame. Re-measure if the camera is remounted.
    jaw_axis_deg = 179.0
    roll_gain = 1.0          # image degrees per wrist-roll degree, measured +0.96/+1.01
    #: The moving fingertip in the wrist image (the fixed one is the profile's hand_uv).
    MOVING_TIP_UV = (210.0, 476.0)

    def __init__(self, port: str, handeye_file: str | None = None, log=print):
        self.p = load_profile("so101")
        self.port = port
        self.log_fn = log
        self.phase_fn = lambda name, note="": log(f"[{name}] {note}")
        self.pan, self.roll = self.p.pan_joint, self.p.roll_joint
        self.pitch_chain = tuple(self.p.pitch_chain)
        self.lo, self.hi = self.p.limits()
        self.home = np.array(self.p.home_deg, float)
        self.motors = list(self.p.joint_names)
        fu, fv = self.p.gripper.hand_uv
        mu, mv = self.MOVING_TIP_UV
        self.jaw_uv = ((fu + mu) / 2.0, (fv + mv) / 2.0)
        self.table_z = float(self.p.table_z_m)

        self.kin = make_kinematics(self.p.urdf_path, self.p.ee_frame, self.motors)
        self._ik = make_ik(self.kin, self.p)
        tf = self.p.camera.extrinsics
        if handeye_file and os.path.exists(handeye_file):
            with open(handeye_file, encoding="utf-8") as f:
                tf = json.load(f)["tf"]
        self.geom = CameraGeometry(
            intrinsics_from_dict(dict(zip(("fx", "fy", "cx", "cy"),
                                          self.p.camera.intrinsics_fallback)),
                                 width=self.p.camera.width, height=self.p.camera.height),
            EyeInHand(lambda q: self.kin.forward_kinematics(q), parse_tf(tf)))
        self._limits = MotionLimits.from_profile(self.p)

        self.bus_lock = threading.RLock()
        self.stop_flag = threading.Event()
        self.robot = None
        self.cam = None
        self.rgb = None                    # the latest frame, RGB
        self.q = self.home.copy()
        self.gripper_pct = 0.0
        self.min_bearing_deg: float | None = None   # a guardrail, set by the caller
        self.pace = 1.0                    # speed multiplier on every move

    # ---- connection -------------------------------------------------------------
    def connect(self) -> None:
        from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig
        from lerobot.motors.feetech.feetech import FeetechMotorsBus
        from lerobot.robots.so_follower import SO101FollowerConfig
        from lerobot.robots.utils import make_robot_from_config

        FeetechMotorsBus._handshake = lambda self: None      # the gripper pings too slowly
        self._clear_overload()
        c = self.p.camera
        for attempt in range(6):
            self.robot = make_robot_from_config(SO101FollowerConfig(
                port=self.port, id="so101_follower",
                cameras={"front": OAKDCameraConfig(fps=c.fps, width=c.width,
                                                   height=c.height, use_depth=False)}))
            try:
                self.robot.connect()
                break
            except (RuntimeError, ConnectionError, OSError) as e:
                self.log_fn(f"connect attempt {attempt + 1}/6 failed: {e}")
                try:
                    self.robot.bus.port_handler.closePort()
                except Exception:
                    pass
                if attempt == 5:
                    raise
                time.sleep(2.0)
                self._clear_overload()
        self.cam = self.robot.cameras["front"]
        try:
            intr = dict(self.cam.get_depth_intrinsics())
            self.geom.set_intrinsics(intrinsics_from_dict(intr, width=c.width, height=c.height))
        except Exception:
            pass                          # keep the profile's intrinsics
        self.observe()

    def disconnect(self) -> None:
        self.stop_flag.set()
        time.sleep(0.5)                   # let readers fall out before the camera goes
        try:
            if self.robot is not None:
                self.robot.disconnect()
        except Exception as e:
            self.log_fn(f"disconnect: {e}")

    def _clear_overload(self) -> None:
        """Torque-cycle every servo: a latched overload flag otherwise kills the connect."""
        bus = self.p.bus
        try:
            import scservo_sdk as scs
            ph = scs.PortHandler(self.port)
            if not ph.openPort():
                return
            ph.setBaudRate(bus.baud)
            pk = scs.PacketHandler(0)
            for mid in bus.motor_ids:
                pk.write1ByteTxRx(ph, mid, bus.torque_register, 0)
                time.sleep(0.12)
                pk.write1ByteTxRx(ph, mid, bus.torque_register, 1)
                time.sleep(0.06)
            ph.closePort()
        except Exception as e:
            self.log_fn(f"overload clear skipped: {e}")

    def _reopen_bus(self) -> None:
        ph = self.robot.bus.port_handler
        try:
            ph.closePort()
        except Exception:
            pass
        time.sleep(0.6)
        ph.openPort()
        time.sleep(0.3)
        self.log_fn("        servo bus stopped answering — reopened the port")

    def _reconnect_camera(self) -> None:
        self.log_fn("        wrist camera stopped — reconnecting it")
        try:
            self.cam.disconnect()
        except Exception:
            pass
        time.sleep(2.0)                   # let the device finish rebooting
        self.cam.connect()

    # ---- raw I/O ------------------------------------------------------------------
    def observe(self, check_stop: bool = True):
        """Read the joints, the gripper and a frame. Retries a flaky bus."""
        if check_stop:
            self.checkpoint()
        for attempt in range(12):
            try:
                with self.bus_lock:
                    obs = self.robot.get_observation()
                break
            except ConnectionError:
                if attempt == 11:
                    raise
                if attempt == 5:
                    with self.bus_lock:
                        self._reopen_bus()
                else:
                    time.sleep(0.08)
            except RuntimeError as e:
                # The OAK-D crashes now and then (X_LINK_ERROR) and reboots on its own;
                # its read thread is then dead. Reconnect it rather than stay blind.
                if "OAKD" not in str(e) or attempt == 11:
                    raise
                with self.bus_lock:
                    self._reconnect_camera()
        self.q = np.array([float(obs[f"{m}.pos"]) for m in self.motors])
        self.gripper_pct = float(obs.get("gripper.pos", 0.0))
        self.rgb = np.asarray(obs["front"])
        return self.q.copy()

    def send(self, q, gripper=None) -> None:
        act = {f"{m}.pos": float(v) for m, v in zip(self.motors, q)}
        if gripper is not None:
            act["gripper.pos"] = float(gripper)
        for attempt in range(3):
            try:
                with self.bus_lock:
                    self.robot.send_action(act)
                return
            except ConnectionError:
                if attempt == 2:
                    raise
                with self.bus_lock:
                    self._reopen_bus()

    def gripper_current(self):
        try:
            with self.bus_lock:
                return float(self.robot.bus.read("Present_Current", "gripper", normalize=False))
        except Exception:
            return None

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
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)

    def cast(self, uv, q):
        pt = self.geom.ray_to_plane(uv, self.geom.T_base_cam(np.asarray(q, float)), self.table_z)
        return None if pt is None else (float(pt[0]), float(pt[1]))

    def project(self, p, q):
        return self.geom.project(np.asarray(p, float), self.geom.T_base_cam(np.asarray(q, float)))

    def grip(self, pct):
        self.send(self.observe(), gripper=float(pct))
        time.sleep(0.25)

    def release(self):
        self.grip(self.p.gripper.place_open_pct)

    def close(self, from_pct):
        """Close in small steps, stopping when the current rises. True on contact.

        A rise while the jaws are still above 60% open is the motor starting, not an
        object, and is ignored."""
        g = self.p.gripper
        idle = [c for c in (self.gripper_current() for _ in range(5)) if c is not None]
        i_idle = float(np.mean(idle)) if idle else 0.0
        pct = float(from_pct)
        while pct > g.closed_pct:
            self.checkpoint()
            pct -= 3.0
            q = self.observe()
            self.send(q, gripper=pct)
            time.sleep(0.09)
            c = self.gripper_current()
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
