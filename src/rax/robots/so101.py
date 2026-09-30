"""The SO-101 with its wrist OAK-D, through lerobot (imported at connect time only).

The lessons from running this rig for weeks are kept, and only those:

* the gripper servo answers too slowly for lerobot's handshake ping, so it is skipped;
* a servo that latched its overload flag is cleared with a raw torque cycle first;
* a bus that stops answering mid-run is closed and reopened, not given up on;
* the OAK-D crashes now and then (X_LINK_ERROR) and is reconnected;
* the camera must be released on exit, or the OAK-D stays booted with no owner and
  the next start fails with "No available devices" until it is unplugged.
"""

from __future__ import annotations

import time

import numpy as np

from rax.perception.camera_geometry import intrinsics_from_dict
from rax.robots.base import Stopped, WristCameraArm

__all__ = ["So101", "Stopped"]


class So101(WristCameraArm):
    profile_name = "so101"
    #: Fingertips measured at about (154,382) and (400,376) on a live frame.
    jaw_axis_deg = 179.0
    roll_gain = 1.0                       # measured +0.96 / +1.01
    grip_levels = {"air": 1.2, "blocked": 3.5, "jammed": 36.0, "two": 16.0}
    close_by_current = True
    #: The moving fingertip in the wrist image (the fixed one is the profile's hand_uv).
    MOVING_TIP_UV = (210.0, 476.0)

    def __init__(self, port: str, handeye_file: str | None = None, log=print):
        super().__init__(port, handeye_file, log)
        fu, fv = self.p.gripper.hand_uv
        mu, mv = self.MOVING_TIP_UV
        self.jaw_uv = ((fu + mu) / 2.0, (fv + mv) / 2.0)
        self.robot = None
        self.cam = None

    def _connect(self) -> None:
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

    def _disconnect(self) -> None:
        if self.robot is not None:
            self.robot.disconnect()

    def _read(self):
        for attempt in range(12):
            try:
                obs = self.robot.get_observation()
                break
            except ConnectionError:
                if attempt == 11:
                    raise
                if attempt == 5:
                    self._reopen_bus()
                else:
                    time.sleep(0.08)
            except RuntimeError as e:
                if "OAKD" not in str(e) or attempt == 11:
                    raise
                self._reconnect_camera()
        q = np.array([float(obs[f"{m}.pos"]) for m in self.motors])
        return q, float(obs.get("gripper.pos", 0.0)), obs["front"]

    def _write(self, q_deg, gripper_pct) -> None:
        act = {f"{m}.pos": float(v) for m, v in zip(self.motors, q_deg)}
        act["gripper.pos"] = float(gripper_pct)
        for attempt in range(3):
            try:
                self.robot.send_action(act)
                return
            except ConnectionError:
                if attempt == 2:
                    raise
                self._reopen_bus()

    def _torque(self, on: bool) -> None:
        if on:
            obs = self.robot.get_observation()
            hold = {k: v for k, v in obs.items() if k.endswith(".pos")}
            self.robot.bus.enable_torque()
            self.robot.send_action(hold)          # hold where it is, do not snap
        else:
            self.robot.bus.disable_torque()

    def _gripper_current(self):
        try:
            with self.io_lock:
                return float(self.robot.bus.read("Present_Current", "gripper", normalize=False))
        except Exception:
            return None

    # ---- recovery ---------------------------------------------------------------
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
