#!/usr/bin/env python3
"""
Robot Client — runs on the Booster K1 robot.
Streams camera + depth + audio to a remote server via WebSocket.
Receives and executes control commands.

Usage:
    python3 robot_client.py eth0 --server ws://YOUR_PC_IP:9090
"""

import os
import sys
import asyncio
import threading
import time
import argparse
import json
import struct
import zlib

import numpy as np
import cv2
try:
    import pyaudio
except ImportError:
    import pyaudio_compat as pyaudio
import websockets

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from booster_robotics_sdk_python import (
    B1LocoClient, ChannelFactory, RobotMode, B1HandIndex, B1HandAction,
    Position, Orientation, Posture,
)

SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHUNK = 1024

# Binary message type prefixes (robot -> server stream)
MSG_VIDEO = 0x01
MSG_DEPTH = 0x02
MSG_AUDIO_IN = 0x03

# Optional low-level joint API (fight guard / jabs use real elbow/shoulder angles).
_LOWLEVEL = None
try:
    import booster_robotics_sdk_python as _br_sdk
    _b1 = getattr(_br_sdk, 'b1', None)
    _LOWLEVEL = {
        'B1LowStateSubscriber': getattr(_br_sdk, 'B1LowStateSubscriber', None),
        'B1LowCmdPublisher': getattr(_br_sdk, 'B1LowCmdPublisher', None),
        'LowCmd': getattr(_br_sdk, 'LowCmd', None),
        'MotorCmd': getattr(_br_sdk, 'MotorCmd', None),
        'CmdType': getattr(_br_sdk, 'CmdType', None),
        'JointIndex': getattr(_br_sdk, 'JointIndex', None),
        'kTopicJointCtrl': getattr(_br_sdk, 'kTopicJointCtrl', None) or (
            getattr(_b1, 'kTopicJointCtrl', None) if _b1 else None
        ),
        'kJointCnt': getattr(_br_sdk, 'kJointCnt', None),
    }
    if not all((_LOWLEVEL['B1LowStateSubscriber'], _LOWLEVEL['B1LowCmdPublisher'],
                _LOWLEVEL['LowCmd'], _LOWLEVEL['MotorCmd'], _LOWLEVEL['JointIndex'])):
        _LOWLEVEL = None
except ImportError:
    _LOWLEVEL = None


def _ji(joint_index_enum, name):
    """Safe JointIndex lookup (int value for motor_cmd index)."""
    if joint_index_enum is None:
        return None
    j = getattr(joint_index_enum, name, None)
    if j is None:
        return None
    try:
        return int(j)
    except (TypeError, ValueError):
        return j


def _joint_index_enum():
    """JointIndex for arm mapping (SDK may expose it even when DDS lowcmd is unused)."""
    if _LOWLEVEL and _LOWLEVEL.get('JointIndex'):
        return _LOWLEVEL['JointIndex']
    try:
        import booster_robotics_sdk_python as _br
        return getattr(_br, 'JointIndex', None)
    except ImportError:
        return None


class FightLowCmdController:
    """LowCmd fight control: ROS2 joint_ctrl + /low_state (Booster deploy), else DDS LowCmd."""

    def __init__(self, loco_client: B1LocoClient, sdk_lock: threading.Lock, ros_node=None):
        self.client = loco_client
        self.sdk_lock = sdk_lock
        self._ros_node = ros_node
        self._use_ros = False
        self._q = None
        self._q_lock = threading.Lock()
        self._pub = None
        self._sub = None
        self._n = 0
        self._hold = threading.Event()
        self._hold_thread = None
        self._hz = 50.0
        self._kp_default = 55.0
        self._kd_default = 2.2
        self._kp_arm = 72.0
        self._kd_arm = 2.8
        self._guard_ov = {}
        self._overlay = {}
        self._overlay_lock = threading.Lock()

    @staticmethod
    def _booster_ros_msgs_ok():
        try:
            from booster_interface.msg import LowCmd  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def available(self):
        # Always try if we have the streaming node (it will import booster_interface on demand).
        if self._ros_node is not None:
            return True
        return self._booster_ros_msgs_ok() or _LOWLEVEL is not None

    def set_ros_node(self, node):
        self._ros_node = node

    def _on_state(self, msg):
        try:
            serial = msg.motor_state_serial
            with self._q_lock:
                self._q = [float(m.q) for m in serial]
                self._n = len(self._q)
        except Exception:
            pass

    def _get_base_q(self):
        if self._use_ros and self._ros_node is not None:
            q = self._ros_node.booster_joint_snapshot()
            if q and len(q) == self._n:
                return q
            return None
        with self._q_lock:
            return list(self._q) if self._q else None

    def _ensure_io(self):
        if self._n > 0 and (self._use_ros or self._pub is not None):
            return True

        if self._ros_node is not None:
            if self._ros_node.ensure_booster_fight_bridge():
                t0 = time.time()
                while time.time() - t0 < 3.0:
                    n = self._ros_node.booster_num_joints()
                    if n > 0:
                        break
                    time.sleep(0.02)
                n = self._ros_node.booster_num_joints()
                if n <= 0:
                    print('[FightLowCmd] /low_state not received (check Booster ROS2 bridge)')
                else:
                    self._ros_node.booster_init_lowcmd_motors(n)
                    if self._ros_node.booster_wait_joint_ctrl_subscriber(25.0):
                        self._n = n
                        self._use_ros = True
                        print('[FightLowCmd] ROS2 joint_ctrl + /low_state (same as booster_deploy / teleop stack)')
                        return True
                    print('[FightLowCmd] No subscriber on joint_ctrl yet (motion stack listening?)')
                    self._n = 0

        if _LOWLEVEL is None:
            return False
        L = _LOWLEVEL
        topic = L['kTopicJointCtrl'] or 'rt/low_cmd'
        try:
            self._sub = L['B1LowStateSubscriber'](self._on_state)
            self._sub.InitChannel()
        except Exception as e:
            print(f"[FightLowCmd] DDS LowState subscriber failed: {e}")
            return False
        topics = []
        for t in (topic, 'rt/low_cmd', 'rt/joint_ctrl'):
            if t and t not in topics:
                topics.append(t)
        self._pub = None
        last_err = None
        for tp in topics:
            try:
                pub = L['B1LowCmdPublisher'](tp)
                pub.InitChannel()
                self._pub = pub
                break
            except Exception as e:
                last_err = e
        if self._pub is None:
            try:
                self._pub = L['B1LowCmdPublisher']()
                self._pub.InitChannel()
            except Exception as e:
                print(f"[FightLowCmd] DDS LowCmd publisher failed ({last_err or e})")
                return False
        t0 = time.time()
        while self._n == 0 and time.time() - t0 < 2.5:
            time.sleep(0.02)
        if self._n > 0:
            print('[FightLowCmd] DDS LowCmd publisher (fallback)')
        return self._n > 0

    def _cmd_type(self):
        L = _LOWLEVEL
        ct = L['CmdType']
        if ct is None:
            return None
        return getattr(ct, 'SERIAL', None) or getattr(ct, 'PARALLEL', None)

    def _write_lowcmd(self, q_targets, arm_idx_set):
        """Publish one frame: position hold on all joints to q_targets."""
        if self._use_ros and self._ros_node is not None:
            mcs = self._ros_node._bf_motor_cmd
            for i in range(self._n):
                q = q_targets[i]
                kp = self._kp_arm if i in arm_idx_set else self._kp_default
                kd = self._kd_arm if i in arm_idx_set else self._kd_default
                mcs[i].q = q
                mcs[i].dq = 0.0
                mcs[i].tau = 0.0
                mcs[i].kp = kp
                mcs[i].kd = kd
                mcs[i].weight = 1.0
            self._ros_node.booster_publish_lowcmd()
            return True

        L = _LOWLEVEL
        LowCmd, MotorCmd = L['LowCmd'], L['MotorCmd']
        msg = LowCmd()
        ct = self._cmd_type()
        if ct is not None:
            try:
                msg.cmd_type = ct
            except Exception:
                try:
                    msg.cmd_type(ct)
                except Exception:
                    pass
        motors = []
        for i in range(self._n):
            mc = MotorCmd()
            q = q_targets[i]
            kp = self._kp_arm if i in arm_idx_set else self._kp_default
            kd = self._kd_arm if i in arm_idx_set else self._kd_default
            for attr, val in (('q', q), ('dq', 0.0), ('tau', 0.0), ('kp', kp), ('kd', kd), ('weight', 1.0)):
                try:
                    setattr(mc, attr, val)
                except Exception:
                    try:
                        getattr(mc, attr)(val)
                    except Exception:
                        pass
            motors.append(mc)
        try:
            msg.motor_cmd = motors
        except Exception:
            try:
                for m in motors:
                    msg.motor_cmd.append(m)
            except Exception:
                return False
        try:
            if hasattr(self._pub, 'Write'):
                self._pub.Write(msg)
            elif hasattr(self._pub, 'write'):
                self._pub.write(msg)
            elif hasattr(self._pub, 'Publish'):
                self._pub.Publish(msg)
            else:
                return False
        except Exception as e:
            print(f"[FightLowCmd] Write failed: {e}")
            return False
        return True

    def _blend(self, a, b, alpha):
        return [a[i] + (b[i] - a[i]) * alpha for i in range(len(a))]

    def _arm_indices(self):
        J = _joint_index_enum()
        idx = []
        for name in (
            'kLeftShoulderPitch', 'kLeftShoulderRoll', 'kLeftElbowPitch', 'kLeftElbowYaw',
            'kRightShoulderPitch', 'kRightShoulderRoll', 'kRightElbowPitch', 'kRightElbowYaw',
        ):
            v = _ji(J, name)
            if v is not None and v < self._n:
                idx.append(v)
        return set(idx)

    def guard_targets(self):
        """Joint radians: arms up in guard (shoulders forward/up, elbows clearly bent)."""
        J = _joint_index_enum()
        t = {}
        # Tunable on-robot; signs follow Booster B1-style layout.
        t[_ji(J, 'kLeftShoulderPitch')] = 0.82
        t[_ji(J, 'kLeftShoulderRoll')] = 0.58
        t[_ji(J, 'kLeftElbowPitch')] = 1.72
        t[_ji(J, 'kLeftElbowYaw')] = -0.08
        t[_ji(J, 'kRightShoulderPitch')] = 0.82
        t[_ji(J, 'kRightShoulderRoll')] = -0.58
        t[_ji(J, 'kRightElbowPitch')] = 1.72
        t[_ji(J, 'kRightElbowYaw')] = 0.08
        return {k: v for k, v in t.items() if k is not None}

    def punch_delta(self, hand):
        J = _joint_index_enum()
        if hand == 'left':
            return {
                _ji(J, 'kLeftShoulderPitch'): 0.42,
                _ji(J, 'kLeftElbowPitch'): -0.82,
            }
        return {
            _ji(J, 'kRightShoulderPitch'): 0.42,
            _ji(J, 'kRightElbowPitch'): -0.82,
        }

    def punch_peak_overrides(self, hand):
        """Absolute arm targets = guard + punch delta (elbow extends, shoulder drives forward)."""
        out = dict(self._guard_ov)
        for k, dv in self.punch_delta(hand).items():
            if k is not None and k in out:
                out[k] = out[k] + dv
        return out

    def set_overlay(self, d):
        with self._overlay_lock:
            self._overlay = dict(d) if d else {}

    def clear_overlay(self):
        with self._overlay_lock:
            self._overlay = {}

    def _apply_overrides(self, base_q, overrides):
        out = list(base_q)
        for idx, val in overrides.items():
            if idx is not None and 0 <= idx < len(out):
                out[idx] = val
        return out

    def run_hold_loop(self, arm_idx_set):
        """Hold loop: legs/head track measured q; arms use guard + optional punch overlay."""
        period = 1.0 / self._hz
        while self._hold.is_set():
            base = self._get_base_q()
            if not base or len(base) != self._n:
                time.sleep(period)
                continue
            with self._overlay_lock:
                ov = {**self._guard_ov, **self._overlay}
            tgt = self._apply_overrides(base, ov)
            self._write_lowcmd(tgt, arm_idx_set)
            time.sleep(period)

    def interpolate_to(self, target_overrides, arm_idx_set, duration_s=0.65):
        start = self._get_base_q()
        if not start or len(start) != self._n:
            return False
        end = self._apply_overrides(start, target_overrides)
        steps = max(1, int(duration_s * self._hz))
        for s in range(1, steps + 1):
            if not self._hold.is_set():
                break
            a = s / steps
            blended = self._blend(start, end, a)
            self._write_lowcmd(blended, arm_idx_set)
            time.sleep(1.0 / self._hz)
        return True

    def enter_custom_and_stream(self, guard_overrides, arm_idx_set):
        Custom = (
            getattr(RobotMode, 'kCustom', None)
            or getattr(RobotMode, 'k_Manual', None)
            or getattr(RobotMode, 'kDevelop', None)
        )
        if Custom is None:
            print('[FightLowCmd] No Custom/manual RobotMode — cannot use low-level fight')
            return False
        try:
            with self.sdk_lock:
                self.client.SwitchHandEndEffectorControlMode(False)
        except Exception:
            pass

        # Booster deploy order: seed LowCmd from measured q, publish, sleep, then Custom.
        if self._use_ros:
            init = self._get_base_q()
            if not init or len(init) != self._n:
                print('[FightLowCmd] No joint snapshot for Custom startup')
                return False
            mcs = self._ros_node._bf_motor_cmd
            for i in range(self._n):
                mcs[i].q = init[i]
                kp = self._kp_arm if i in arm_idx_set else self._kp_default
                kd = self._kd_arm if i in arm_idx_set else self._kd_default
                mcs[i].dq = 0.0
                mcs[i].tau = 0.0
                mcs[i].kp = kp
                mcs[i].kd = kd
                mcs[i].weight = 1.0
            self._ros_node.booster_publish_lowcmd()
            time.sleep(0.1)
            try:
                with self.sdk_lock:
                    self.client.ChangeMode(Custom)
                time.sleep(0.05)
            except Exception as e:
                print(f"[FightLowCmd] ChangeMode(Custom) failed: {e}")
                return False
        else:
            try:
                with self.sdk_lock:
                    self.client.ChangeMode(Custom)
                time.sleep(0.05)
            except Exception as e:
                print(f"[FightLowCmd] ChangeMode(Custom) failed: {e}")
                return False

        self._guard_ov = dict(guard_overrides)
        with self._overlay_lock:
            self._overlay = {}
        self._hold.set()
        if not self.interpolate_to(guard_overrides, arm_idx_set, duration_s=0.7):
            self._hold.clear()
            self._guard_ov = {}
            try:
                with self.sdk_lock:
                    self.client.ChangeMode(RobotMode.kPrepare)
                time.sleep(0.25)
                with self.sdk_lock:
                    self.client.ChangeMode(RobotMode.kWalking)
            except Exception:
                pass
            print('[FightLowCmd] No LowState yet — cannot hold Custom pose')
            return False

        def _loop():
            self.run_hold_loop(arm_idx_set)

        self._hold_thread = threading.Thread(target=_loop, daemon=True)
        self._hold_thread.start()
        return True

    def stop_and_walk(self):
        self._hold.clear()
        if self._hold_thread and self._hold_thread.is_alive():
            self._hold_thread.join(timeout=1.5)
        self._hold_thread = None
        self._guard_ov = {}
        with self._overlay_lock:
            self._overlay = {}
        try:
            with self.sdk_lock:
                self.client.ChangeMode(RobotMode.kPrepare)
            time.sleep(0.35)
            with self.sdk_lock:
                self.client.ChangeMode(RobotMode.kWalking)
            time.sleep(0.15)
        except Exception as e:
            print(f"[FightLowCmd] return to Walking failed: {e}")


# ── ROS2 Camera Streamer ────────────────────────────────────────────────────


class CameraStreamer(Node):
    """ROS2 node: subscribes to camera + depth, buffers latest frames."""

    def __init__(self):
        super().__init__('robot_stream_client')
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._frame_jpeg = None
        self._depth_compressed = None
        self._new_frame = threading.Event()
        self._new_depth = threading.Event()
        self._raw_frame = None

        self.create_subscription(Image, '/image_left_raw', self._on_image, 10)
        self.create_subscription(CompressedImage, '/booster_video_stream', self._on_compressed, 10)
        self.create_subscription(Image, '/StereoNetNode/stereonet_depth', self._on_depth, 10)

    def _on_image(self, msg):
        try:
            if msg.encoding == 'nv12':
                h, w = msg.height, msg.width
                yuv = np.frombuffer(msg.data, dtype=np.uint8).reshape((int(h * 1.5), w))
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._encode_frame(frame)
        except Exception as e:
            self.get_logger().error(f'Image error: {e}')

    def _on_compressed(self, msg):
        if self._raw_frame is not None:
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self._encode_frame(frame)
        except Exception as e:
            self.get_logger().error(f'Compressed image error: {e}')

    def _encode_frame(self, frame):
        h, w = frame.shape[:2]
        if max(h, w) > 640:
            s = 640 / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with self._lock:
            self._raw_frame = frame
            self._frame_jpeg = jpeg.tobytes()
        self._new_frame.set()

    def _on_depth(self, msg):
        try:
            if msg.encoding == 'mono16':
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
            else:
                depth = self.bridge.imgmsg_to_cv2(msg)
            h, w = depth.shape
            small = cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
            sh, sw = small.shape
            header = struct.pack('<HH', sw, sh)
            compressed = zlib.compress(small.tobytes(), level=1)
            with self._lock:
                self._depth_compressed = header + compressed
            self._new_depth.set()
        except Exception as e:
            self.get_logger().error(f'Depth error: {e}')

    def take_frame(self):
        self._new_frame.clear()
        with self._lock:
            return self._frame_jpeg

    def take_depth(self):
        self._new_depth.clear()
        with self._lock:
            return self._depth_compressed

    # ── Booster fight / teleop-style lowcmd (same stack as booster_deploy & typical teleop) ──
    # Ref: github.com/BoosterRobotics/booster_deploy — booster_robot_controller.py:
    # subscribe /low_state, publish LowCmd on "joint_ctrl", CMD_TYPE_SERIAL, wait for
    # subscriber, seed q/kp/kd, publish, sleep 0.1s, ChangeMode(kCustom).

    def ensure_booster_fight_bridge(self):
        """Create /low_state subscription and joint_ctrl publisher if booster_interface is present."""
        if getattr(self, '_booster_fight_ready', False):
            return True
        try:
            from booster_interface.msg import LowState, LowCmd, MotorCmd
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        except ImportError:
            return False

        self._bf_q_lock = threading.Lock()
        self._bf_joint_q = None
        self._bf_num_joints = 0

        def _on_low_state(msg):
            try:
                q = [float(m.q) for m in msg.motor_state_serial]
                with self._bf_q_lock:
                    self._bf_joint_q = q
                    self._bf_num_joints = len(q)
            except Exception:
                pass

        qos_be = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        qos_rel = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(LowState, '/low_state', _on_low_state, qos_be)
        self._bf_lowcmd_pub = self.create_publisher(LowCmd, 'joint_ctrl', qos_rel)
        self._bf_low_cmd = LowCmd()
        self._bf_low_cmd.cmd_type = LowCmd.CMD_TYPE_SERIAL
        self._bf_MotorCmd = MotorCmd
        self._booster_fight_ready = True
        return True

    def booster_joint_snapshot(self):
        with self._bf_q_lock:
            if not self._bf_joint_q:
                return None
            return list(self._bf_joint_q)

    def booster_num_joints(self):
        with self._bf_q_lock:
            return self._bf_num_joints

    def booster_init_lowcmd_motors(self, n):
        """Resize motor_cmd array like booster_deploy create_low_cmd_publisher."""
        MotorCmd = self._bf_MotorCmd
        seq = self._bf_low_cmd.motor_cmd
        try:
            seq.clear()
        except AttributeError:
            while len(seq) > 0:
                seq.pop()
        for _ in range(n):
            mc = MotorCmd()
            mc.q = 0.0
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = 0.0
            mc.kd = 0.0
            mc.weight = 0.0
            seq.append(mc)
        self._bf_motor_cmd = self._bf_low_cmd.motor_cmd

    def booster_wait_joint_ctrl_subscriber(self, timeout_s=30.0):
        """Controller on robot must subscribe to joint_ctrl before we stream."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                if self._bf_lowcmd_pub.get_subscription_count() > 0:
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return False

    def booster_publish_lowcmd(self):
        self._bf_lowcmd_pub.publish(self._bf_low_cmd)


# ── Robot Command Executor ──────────────────────────────────────────────────


class RobotExecutor:
    """Receives JSON commands from server and executes them on the robot SDK."""

    def __init__(self, client: B1LocoClient):
        self.client = client
        self.lock = threading.Lock()
        self.head_pitch = 0.0
        self.head_yaw = 0.0
        self.right_arm_pos = [0.35, -0.25, 0.1]
        self.left_arm_pos = [0.35, 0.25, 0.1]
        self._gesture_cancel = threading.Event()
        self._fight_active = False
        self._fight_punch_lock = threading.Lock()
        self._fight_low = FightLowCmdController(client, self.lock, ros_node=None)
        self._fight_use_lowlevel = False

    def attach_ros_camera(self, camera_node):
        """Wire the same ROS2 node that spins for camera (needed for /low_state + joint_ctrl)."""
        self._fight_low.set_ros_node(camera_node)

    def handle(self, msg):
        cmd = msg.get('cmd')
        if not cmd:
            return
        try:
            handler = getattr(self, f'_cmd_{cmd}', None)
            if handler:
                handler(msg)
            else:
                print(f"[Exec] Unknown command: {cmd}")
        except Exception as e:
            print(f"[Exec] Error in {cmd}: {e}")

    # ── Low-level commands (called at high frequency by server tracking loops)

    def _cmd_move(self, m):
        with self.lock:
            self.client.Move(m.get('x', 0), m.get('y', 0), m.get('yaw', 0))

    def _cmd_rotate_head(self, m):
        p = max(-0.5, min(1.0, m.get('pitch', 0)))
        y = max(-0.785, min(0.785, m.get('yaw', 0)))
        self.head_pitch, self.head_yaw = p, y
        with self.lock:
            self.client.RotateHead(p, y)

    # ── Gesture commands

    def _cmd_stop_gesture(self, _):
        """Stop any ongoing gesture and reset arms/head to neutral."""
        self._cancel_gesture_and_reset()

    def _cmd_wave(self, _):
        def _do():
            self._cancel_gesture_and_reset()
            with self.lock:
                self.client.WaveHand(B1HandAction.kHandOpen)
        threading.Thread(target=_do, daemon=True).start()

    def _cmd_handshake(self, _):
        def _do():
            self._cancel_gesture_and_reset()
            with self.lock:
                self.client.Handshake(B1HandAction.kHandOpen)
        threading.Thread(target=_do, daemon=True).start()

    def _cmd_nod(self, _):
        def _do():
            self._cancel_gesture_and_reset()
            for _ in range(3):
                if self._gesture_cancel.is_set():
                    return
                self._set_head(0.3, self.head_yaw)
                self._sleep_cancelable(0.25)
                self._set_head(-0.1, self.head_yaw)
                self._sleep_cancelable(0.25)
            self._set_head(0.0, 0.0)
        threading.Thread(target=_do, daemon=True).start()

    def _cmd_head_shake(self, _):
        def _do():
            self._cancel_gesture_and_reset()
            for _ in range(3):
                if self._gesture_cancel.is_set():
                    return
                self._set_head(self.head_pitch, 0.4)
                self._sleep_cancelable(0.2)
                self._set_head(self.head_pitch, -0.4)
                self._sleep_cancelable(0.2)
            self._set_head(0.0, 0.0)
        threading.Thread(target=_do, daemon=True).start()

    # ── Dance commands

    def _cmd_dance(self, m):
        name = (m.get('name') or 'robot').lower()
        threading.Thread(target=self._run_dance, args=(name,), daemon=True).start()

    def _cmd_dab(self, _):
        def _do():
            self._cancel_gesture_and_reset()
            self._dab()
        threading.Thread(target=_do, daemon=True).start()

    def _cmd_flex(self, _):
        def _do():
            self._cancel_gesture_and_reset()
            self._flex()
        threading.Thread(target=_do, daemon=True).start()

    def _cmd_get_up(self, _):
        def _do():
            with self.lock:
                self.client.GetUp()
        threading.Thread(target=_do, daemon=True).start()

    def _cmd_shoot(self, _):
        """Powerful kicking motion (soccer shoot). Uses whole-body kick (dance_id 5)."""
        def _do():
            self._cancel_gesture_and_reset()
            try:
                from booster_robotics_sdk_python import B1LocoApiId
                with self.lock:
                    self.client.SendApiRequest(
                        B1LocoApiId(2029),  # kWholeBodyDance
                        json.dumps({'dance_id': 5})  # boxing kick - whole-body kick motion
                    )
            except (ImportError, AttributeError):
                with self.lock:
                    self.client.SendApiRequest(2029, json.dumps({'dance_id': 5}))
        threading.Thread(target=_do, daemon=True).start()

    def _cmd_visual_kick(self, m):
        """Side-foot kick. start=True to kick, start=False to stop."""
        start = m.get('start', True)
        def _do():
            try:
                from booster_robotics_sdk_python import B1LocoApiId
                param = json.dumps({'start': start})
                with self.lock:
                    self.client.SendApiRequest(B1LocoApiId(2038), param)
            except (ImportError, AttributeError):
                param = json.dumps({'start': start})
                with self.lock:
                    self.client.SendApiRequest(2038, param)
        threading.Thread(target=_do, daemon=True).start()

    def _cmd_soccer_combo(self, _):
        """Shoot (power kick) then celebrate."""
        def _do():
            self._cancel_gesture_and_reset()
            try:
                from booster_robotics_sdk_python import B1LocoApiId
                with self.lock:
                    self.client.SendApiRequest(
                        B1LocoApiId(2029),
                        json.dumps({'dance_id': 5})  # whole-body kick
                    )
                time.sleep(2.5)  # let kick complete
                if self._gesture_cancel.is_set():
                    return
                with self.lock:
                    self.client.SendApiRequest(
                        B1LocoApiId.kDance,
                        json.dumps({'dance_id': 6})  # celebrate/cheer
                    )
            except (ImportError, AttributeError):
                with self.lock:
                    self.client.SendApiRequest(2029, json.dumps({'dance_id': 5}))
                time.sleep(2.5)
                if self._gesture_cancel.is_set():
                    return
                with self.lock:
                    self.client.SendApiRequest(2030, json.dumps({'dance_id': 6}))
        threading.Thread(target=_do, daemon=True).start()

    # ── Fight mode (guard + jab punches for /fight page and voice)

    def _cmd_fight_mode_on(self, _):
        def _do():
            self._gesture_cancel.set()
            time.sleep(0.08)
            self._gesture_cancel.clear()
            self._fight_use_lowlevel = False
            fl = self._fight_low
            if fl._ensure_io():
                arm_idx = fl._arm_indices()
                g = fl.guard_targets()
                if len(arm_idx) >= 4 and g and fl.enter_custom_and_stream(g, arm_idx):
                    self._fight_use_lowlevel = True
                    self._fight_active = True
                    print('[Fight] Joint-space LowCmd guard (Custom mode).')
                    return
                print('[Fight] LowCmd guard/Custom stream failed; using MoveHandEndEffectorV2 fallback.')
            else:
                _hint_ros = not fl._booster_ros_msgs_ok()
                _hint_dds = _LOWLEVEL is None
                if _hint_ros or _hint_dds:
                    print(
                        '[Fight] No low-level joint pipe (need ROS2 `booster_interface` on PYTHONPATH '
                        '— source e.g. `/opt/booster/BoosterRos2Interface/install/setup.bash` — '
                        'and a subscriber on `joint_ctrl`; or a full SDK with B1LowCmdPublisher). '
                        'Using MoveHandEndEffectorV2 fallback.'
                    )
                else:
                    print(
                        '[Fight] Low-level init failed (see [FightLowCmd] lines above). '
                        'Using MoveHandEndEffectorV2 fallback.'
                    )
            self._fight_active = True
            try:
                with self.lock:
                    self.client.SwitchHandEndEffectorControlMode(True)
            except AttributeError:
                pass
            gl = [0.30, 0.12, 0.27]
            gr = [0.30, -0.12, 0.27]
            self._move_hand_ee(gl[0], gl[1], gl[2], 'left', 750)
            self._move_hand_ee(gr[0], gr[1], gr[2], 'right', 750)
            self.left_arm_pos = list(gl)
            self.right_arm_pos = list(gr)
        threading.Thread(target=_do, daemon=True).start()

    def _cmd_fight_mode_off(self, _):
        def _do():
            self._fight_active = False
            if self._fight_use_lowlevel:
                self._fight_low.stop_and_walk()
                self._fight_use_lowlevel = False
                return
            self._arm_to_side('left')
            self._arm_to_side('right')
        threading.Thread(target=_do, daemon=True).start()

    def _cmd_punch_left(self, _):
        self._run_fight_punch('left')

    def _cmd_punch_right(self, _):
        self._run_fight_punch('right')

    # ── Arm commands (for server-driven choreography if needed)

    def _cmd_arm_to_side(self, m):
        self._arm_to_side(m.get('hand', 'right'))

    def _cmd_arm_move_inc(self, m):
        self._arm_inc(m.get('direction', 'up'), m.get('hand', 'right'))

    def _cmd_change_mode(self, m):
        mode_str = m.get('mode', 'walking')
        mode_map = {
            'prepare': RobotMode.kPrepare,
            'walking': RobotMode.kWalking,
        }
        mode = mode_map.get(mode_str, RobotMode.kWalking)
        with self.lock:
            self.client.ChangeMode(mode)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _set_head(self, pitch, yaw):
        pitch = max(-0.5, min(1.0, pitch))
        yaw = max(-0.785, min(0.785, yaw))
        self.head_pitch, self.head_yaw = pitch, yaw
        with self.lock:
            self.client.RotateHead(pitch, yaw)

    def _sleep_cancelable(self, duration):
        """Sleep for duration, return early if gesture cancelled."""
        start = time.time()
        while time.time() - start < duration:
            if self._gesture_cancel.is_set():
                return
            time.sleep(0.05)

    def _cancel_gesture_and_reset(self):
        """Stop any running gesture and reset arms/head to neutral."""
        self._fight_active = False
        if self._fight_use_lowlevel:
            self._fight_low.stop_and_walk()
            self._fight_use_lowlevel = False
        self._gesture_cancel.set()
        time.sleep(0.08)  # let previous gesture thread notice and exit
        self._gesture_cancel.clear()
        self._arm_to_side('right')
        self._arm_to_side('left')
        self._set_head(0.0, 0.0)

    def _arm_to_side(self, hand):
        is_left = hand == 'left'
        y_sign = 1 if is_left else -1
        hand_idx = B1HandIndex.kLeftHand if is_left else B1HandIndex.kRightHand
        posture = Posture()
        posture.position = Position(0.35, y_sign * 0.25, 0.1)
        posture.orientation = Orientation(-y_sign * 1.57, -1.57, 0.0)
        with self.lock:
            self.client.MoveHandEndEffectorV2(posture, 800, hand_idx)
        if is_left:
            self.left_arm_pos = [0.35, 0.25, 0.1]
        else:
            self.right_arm_pos = [0.35, -0.25, 0.1]

    def _arm_inc(self, direction, hand):
        STEP = 0.03
        is_left = hand == 'left'
        pos = self.left_arm_pos if is_left else self.right_arm_pos
        hand_idx = B1HandIndex.kLeftHand if is_left else B1HandIndex.kRightHand
        y_sign = 1 if is_left else -1

        if direction == 'up':
            pos[2] = min(pos[2] + STEP, 0.35)
        elif direction == 'down':
            pos[2] = max(pos[2] - STEP, -0.10)
        elif direction == 'forward':
            pos[0] = min(pos[0] + STEP, 0.40)
        elif direction == 'back':
            pos[0] = max(pos[0] - STEP, 0.20)
        elif direction == 'out':
            pos[1] += STEP * y_sign
        elif direction == 'in':
            pos[1] -= STEP * y_sign

        posture = Posture()
        posture.position = Position(pos[0], pos[1], pos[2])
        posture.orientation = Orientation(-y_sign * 1.57, -1.57, 0.0)
        with self.lock:
            self.client.MoveHandEndEffectorV2(posture, 300, hand_idx)

    def _move_hand_ee(self, x, y, z, hand, duration_ms):
        is_left = hand == 'left'
        y_sign = 1 if is_left else -1
        hand_idx = B1HandIndex.kLeftHand if is_left else B1HandIndex.kRightHand
        posture = Posture()
        posture.position = Position(x, y, z)
        posture.orientation = Orientation(-y_sign * 1.57, -1.57, 0.0)
        with self.lock:
            self.client.MoveHandEndEffectorV2(posture, int(duration_ms), hand_idx)

    def _run_fight_punch(self, hand):
        def _do():
            if not self._fight_active:
                return
            if not self._fight_punch_lock.acquire(blocking=False):
                return
            try:
                if self._fight_use_lowlevel:
                    peak = self._fight_low.punch_peak_overrides(hand)
                    self._fight_low.set_overlay(peak)
                    time.sleep(0.14)
                    self._fight_low.clear_overlay()
                    return
                is_left = hand == 'left'
                pos = self.left_arm_pos if is_left else self.right_arm_pos
                gx, gy, gz = pos[0], pos[1], pos[2]
                hx = min(gx + 0.11, 0.42)
                hy = gy + (-0.025 if is_left else 0.025)
                hz = max(gz - 0.06, 0.12)
                self._move_hand_ee(hx, hy, hz, hand, 115)
                time.sleep(0.13)
                self._move_hand_ee(gx, gy, gz, hand, 210)
            finally:
                self._fight_punch_lock.release()

        threading.Thread(target=_do, daemon=True).start()

    # ── Dance routines (run locally for timing precision) ───────────────────

    def _run_dance(self, name):
        self._cancel_gesture_and_reset()
        try:
            from booster_robotics_sdk_python import B1LocoApiId

            sdk_wholebody = {
                'arabic': 0, 'salsa': 0,
                'michael jackson': 1, 'michael': 1, 'mj': 1,
                'michael2': 2, 'moonwalk': 4,
                'kick': 5, 'boxing': 5,
                'roundhouse': 6, 'karate': 6,
            }
            sdk_upper = {
                'newyear': 0, 'new year': 0, 'nezha': 1, 'future': 2,
                'dab': 3, 'ultraman': 4, 'respect': 5,
                'cheer': 6, 'celebrate': 6, 'luckycat': 7, 'lucky cat': 7,
            }

            if name in sdk_wholebody:
                self.client.SendApiRequest(
                    B1LocoApiId(2029),
                    json.dumps({'dance_id': sdk_wholebody[name]})
                )
                return
            if name in sdk_upper:
                self.client.SendApiRequest(
                    B1LocoApiId.kDance,
                    json.dumps({'dance_id': sdk_upper[name]})
                )
                return
        except ImportError:
            pass

        custom = {
            'macarena': self._dance_macarena,
            'twist': self._dance_twist,
            'bow': self._dance_bow,
            'chicken': self._dance_chicken,
            'disco': self._dance_disco,
        }
        if name in custom:
            custom[name]()
        else:
            self._dance_default()

    def _dance_default(self):
        D = 0.2
        for _ in range(2):
            if self._gesture_cancel.is_set(): return
            self._set_head(0.0, -0.5)
            self._arm_to_side('right')
            self._sleep_cancelable(0.5)
            for _ in range(5):
                if self._gesture_cancel.is_set(): return
                self._arm_inc('up', 'right'); self._sleep_cancelable(D)
            self._set_head(0.0, 0.5)
            self._arm_to_side('left')
            self._sleep_cancelable(0.5)
            for _ in range(5):
                if self._gesture_cancel.is_set(): return
                self._arm_inc('up', 'left'); self._sleep_cancelable(D)
        for _ in range(4):
            if self._gesture_cancel.is_set(): return
            self._arm_inc('up', 'right'); self._arm_inc('up', 'left'); self._sleep_cancelable(D)
        self._set_head(-0.2, 0.0)
        self._sleep_cancelable(1.5)
        if self._gesture_cancel.is_set(): return
        self._set_head(0.0, 0.0)
        self._arm_to_side('right'); self._arm_to_side('left')

    def _dance_macarena(self):
        D = 0.25
        self._arm_to_side('right'); self._arm_to_side('left'); self._sleep_cancelable(0.5)
        for _ in range(5):
            if self._gesture_cancel.is_set(): return
            self._arm_inc('forward', 'right'); self._arm_inc('forward', 'left'); self._sleep_cancelable(D)
        self._sleep_cancelable(0.3)
        for _ in range(5):
            if self._gesture_cancel.is_set(): return
            self._arm_inc('up', 'right'); self._arm_inc('up', 'left'); self._sleep_cancelable(D)
        self._sleep_cancelable(0.3)
        for _ in range(4):
            if self._gesture_cancel.is_set(): return
            self._arm_inc('in', 'right'); self._arm_inc('in', 'left')
            self._arm_inc('down', 'right'); self._arm_inc('down', 'left'); self._sleep_cancelable(D)
        self._sleep_cancelable(0.3)
        if self._gesture_cancel.is_set(): return
        with self.lock:
            self.client.Move(0, 0, 0.5)
        self._sleep_cancelable(1.0)
        with self.lock:
            self.client.Move(0, 0, 0)
        self._arm_to_side('right'); self._arm_to_side('left')

    def _dance_twist(self):
        D = 0.2
        self._arm_to_side('right'); self._arm_to_side('left'); self._sleep_cancelable(0.5)
        for _ in range(5):
            if self._gesture_cancel.is_set(): return
            self._arm_inc('up', 'right'); self._arm_inc('up', 'left'); self._sleep_cancelable(D)
        for _ in range(3):
            if self._gesture_cancel.is_set(): return
            with self.lock: self.client.Move(0, 0, 0.4)
            self._set_head(0.0, 0.4); self._sleep_cancelable(0.6)
            with self.lock: self.client.Move(0, 0, -0.4)
            self._set_head(0.0, -0.4); self._sleep_cancelable(0.6)
        with self.lock: self.client.Move(0, 0, 0)
        self._set_head(0.0, 0.0); self._sleep_cancelable(0.3)
        self._arm_to_side('right'); self._arm_to_side('left')

    def _dance_bow(self):
        self._set_head(0.8, 0.0); self._sleep_cancelable(2.0)
        if self._gesture_cancel.is_set(): return
        self._set_head(0.0, 0.0)

    def _dance_chicken(self):
        D = 0.15
        self._arm_to_side('right'); self._arm_to_side('left'); self._sleep_cancelable(0.5)
        for _ in range(5):
            if self._gesture_cancel.is_set(): return
            for _ in range(3):
                self._arm_inc('out', 'right'); self._arm_inc('out', 'left'); self._sleep_cancelable(D)
            self._set_head(0.3, 0.0)
            for _ in range(3):
                self._arm_inc('in', 'right'); self._arm_inc('in', 'left'); self._sleep_cancelable(D)
            self._set_head(-0.1, 0.0)
        self._set_head(0.0, 0.0); self._sleep_cancelable(0.3)
        self._arm_to_side('right'); self._arm_to_side('left')

    def _dance_disco(self):
        D = 0.2
        self._arm_to_side('right'); self._arm_to_side('left'); self._sleep_cancelable(0.5)
        for _ in range(3):
            if self._gesture_cancel.is_set(): return
            for _ in range(6):
                self._arm_inc('up', 'right'); self._arm_inc('out', 'right'); self._sleep_cancelable(D)
            self._set_head(-0.2, -0.3)
            with self.lock: self.client.Move(0, -0.2, 0)
            self._sleep_cancelable(0.5)
            self._arm_to_side('right')
            for _ in range(6):
                if self._gesture_cancel.is_set(): return
                self._arm_inc('up', 'left'); self._arm_inc('out', 'left'); self._sleep_cancelable(D)
            self._set_head(-0.2, 0.3)
            with self.lock: self.client.Move(0, 0.2, 0)
            self._sleep_cancelable(0.5)
            self._arm_to_side('left')
        with self.lock: self.client.Move(0, 0, 0)
        self._set_head(0.0, 0.0)

    def _dab(self):
        D = 0.25
        self._arm_to_side('right'); self._arm_to_side('left')
        self._sleep_cancelable(0.6)
        for _ in range(5):
            if self._gesture_cancel.is_set(): return
            self._arm_inc('back', 'right'); self._sleep_cancelable(D)
        for _ in range(5):
            if self._gesture_cancel.is_set(): return
            self._arm_inc('in', 'right'); self._sleep_cancelable(D)
        for _ in range(6):
            if self._gesture_cancel.is_set(): return
            self._arm_inc('up', 'right'); self._sleep_cancelable(D)
        for _ in range(7):
            if self._gesture_cancel.is_set(): return
            self._arm_inc('up', 'left'); self._sleep_cancelable(D)
        for _ in range(4):
            if self._gesture_cancel.is_set(): return
            self._arm_inc('out', 'left'); self._sleep_cancelable(D)
        self._set_head(0.5, 0.5); self._sleep_cancelable(2.5)
        if self._gesture_cancel.is_set(): return
        self._set_head(0.0, 0.0); self._sleep_cancelable(0.3)
        self._arm_to_side('right'); self._arm_to_side('left')

    def _flex(self):
        D = 0.25
        self._arm_to_side('right'); self._arm_to_side('left')
        self._sleep_cancelable(0.6)
        for _ in range(7):
            if self._gesture_cancel.is_set(): return
            self._arm_inc('up', 'right'); self._arm_inc('up', 'left'); self._sleep_cancelable(D)
        for _ in range(3):
            if self._gesture_cancel.is_set(): return
            self._arm_inc('out', 'right'); self._arm_inc('out', 'left'); self._sleep_cancelable(D)
        self._set_head(-0.3, 0.0); self._sleep_cancelable(2.0)
        if self._gesture_cancel.is_set(): return
        self._set_head(0.0, 0.0); self._sleep_cancelable(0.3)
        self._arm_to_side('right'); self._arm_to_side('left')


# ── WebSocket Client ────────────────────────────────────────────────────────


def _amplify_audio(data, gain):
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    samples *= gain
    np.clip(samples, -32768, 32767, out=samples)
    return samples.astype(np.int16).tobytes()


async def run_client(args, camera: CameraStreamer, executor: RobotExecutor):
    pya = pyaudio.PyAudio()

    # Auto-detect iFlytek mic
    mic_device = args.mic_device
    if mic_device is None:
        for i in range(pya.get_device_count()):
            info = pya.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0 and 'xfm' in info['name'].lower():
                mic_device = i
                print(f"Auto-detected mic: [{i}] {info['name']}")
                break

    uri = args.server
    print(f"Connecting to server: {uri}")
    print(f"websockets version: {websockets.__version__}")

    while True:
        try:
            async with websockets.connect(
                uri,
                max_size=10 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=60,
                open_timeout=10,
            ) as ws:
                print("Connected to server!")
                tasks = [
                    asyncio.create_task(_stream_video(ws, camera, args.fps)),
                    asyncio.create_task(_stream_depth(ws, camera, args.depth_fps)),
                    asyncio.create_task(_stream_audio(ws, pya, mic_device, args.mic_gain)),
                    asyncio.create_task(_receive_commands(ws, executor, pya)),
                ]
                try:
                    await asyncio.gather(*tasks)
                except websockets.ConnectionClosed:
                    print("Connection lost")
                finally:
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"Connection error ({type(e).__name__}: {e}), retrying in 3s...")
        await asyncio.sleep(3)


async def _stream_video(ws, camera: CameraStreamer, fps):
    interval = 1.0 / fps
    try:
        while True:
            jpeg = camera.take_frame()
            if jpeg:
                await ws.send(bytes([MSG_VIDEO]) + jpeg)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def _stream_depth(ws, camera: CameraStreamer, fps):
    interval = 1.0 / fps
    try:
        while True:
            data = camera.take_depth()
            if data:
                await ws.send(bytes([MSG_DEPTH]) + data)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def _stream_audio(ws, pya, mic_device, mic_gain):
    kwargs = dict(
        format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=SEND_SAMPLE_RATE,
        input=True, frames_per_buffer=AUDIO_CHUNK,
    )
    if mic_device is not None:
        kwargs['input_device_index'] = mic_device
    stream = pya.open(**kwargs)
    apply_gain = mic_gain > 1.01
    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await loop.run_in_executor(
                None, lambda: stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            )
            if apply_gain:
                data = _amplify_audio(data, mic_gain)
            await ws.send(bytes([MSG_AUDIO_IN]) + data)
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()


async def _receive_commands(ws, executor: RobotExecutor, pya):
    speaker = pya.open(
        format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=RECV_SAMPLE_RATE,
        output=True, frames_per_buffer=AUDIO_CHUNK,
    )
    loop = asyncio.get_event_loop()
    try:
        async for message in ws:
            if isinstance(message, bytes) and len(message) > 0:
                msg_type = message[0]
                payload = message[1:]
                if msg_type == 0x10:  # audio playback from Gemini
                    await loop.run_in_executor(None, speaker.write, payload)
            elif isinstance(message, str):
                try:
                    msg = json.loads(message)
                    executor.handle(msg)
                except json.JSONDecodeError:
                    pass
    except asyncio.CancelledError:
        pass
    finally:
        speaker.stop_stream()
        speaker.close()


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='K1 Robot Client — streams to remote server')
    parser.add_argument('interface', help='Network interface for robot SDK (e.g. eth0)')
    parser.add_argument('--server', default='ws://localhost:9090',
                        help='WebSocket server URL (e.g. ws://192.168.1.100:9090)')
    parser.add_argument('--fps', type=int, default=10, help='Video stream FPS')
    parser.add_argument('--depth-fps', type=int, default=5, help='Depth stream FPS')
    parser.add_argument('--mic-gain', type=float, default=3.0, help='Mic gain multiplier')
    parser.add_argument('--mic-device', type=int, default=None, help='PyAudio mic device index')
    args = parser.parse_args()

    print("=" * 60)
    print("K1 Robot Client")
    print(f"  Streaming to: {args.server}")
    print(f"  Video: {args.fps} fps | Depth: {args.depth_fps} fps")
    print("=" * 60)

    # Robot SDK
    print(f"Connecting to robot via {args.interface}...")
    ChannelFactory.Instance().Init(0, args.interface)
    loco_client = B1LocoClient()
    loco_client.Init()
    time.sleep(1.0)

    print("Switching to walking mode...")
    loco_client.ChangeMode(RobotMode.kPrepare)
    time.sleep(2.0)
    loco_client.ChangeMode(RobotMode.kWalking)
    time.sleep(1.0)
    print("Robot ready")

    executor = RobotExecutor(loco_client)

    # ROS2
    rclpy.init()
    camera = CameraStreamer()
    executor.attach_ros_camera(camera)
    ros_thread = threading.Thread(target=rclpy.spin, args=(camera,), daemon=True)
    ros_thread.start()

    print("Waiting for camera frame...")
    deadline = time.time() + 10
    while camera._raw_frame is None and time.time() < deadline:
        time.sleep(0.1)
    if camera._raw_frame is not None:
        print("Camera ready!")
    else:
        print("Warning: no frame after 10s — continuing anyway")

    try:
        asyncio.run(run_client(args, camera, executor))
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        with executor.lock:
            executor.client.Move(0, 0, 0)
        camera.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
