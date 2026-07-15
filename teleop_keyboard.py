#!/usr/bin/env python3
"""
Keyboard teleoperation for SO-101 arm with live Rerun visualization.

Controls
--------
  W / S      EE +Y / -Y   (depth: forward / back)
  A / D      EE -X / +X   (lateral: left / right)
  O / P      EE +Z / -Z   (height: up / down)
  Q / E      wrist roll  +/- (direct joint)
  Space      toggle gripper open / closed
  R          return to home (pose at startup)
  Esc / ^C   quit

Usage
-----
  python teleop_keyboard.py                                  # mock, no hardware
  python teleop_keyboard.py --backend so101 --port COM3
  python teleop_keyboard.py --backend so101 --port COM3 --rerun

Smarter input options
---------------------
  --gamepad   Use a USB gamepad (Xbox/PS) via pygame instead of the keyboard.
              Left stick = XY, right stick = Z+wrist, triggers = gripper.
              More ergonomic; install with:  pip install pygame

  Leader arm  The SO-101 is designed for leader/follower teleop — use a second
              arm as a physical puppet controller (see lerobot record docs).
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from typing import FrozenSet, Set

import numpy as np
from scipy.spatial.transform import Rotation

# ── Key-state tracker ─────────────────────────────────────────────────────────

_held: Set[str] = set()
_events: list[str] = []
_lock = threading.Lock()
_quit = threading.Event()


def _start_pynput() -> bool:
    try:
        from pynput import keyboard as _kb
    except ImportError:
        return False

    def _press(key):
        try:
            k = key.char.lower() if getattr(key, "char", None) else key.name.lower()
        except Exception:
            k = str(key).replace("Key.", "").lower()
        with _lock:
            _held.add(k)
            if k in (" ", "r"):
                _events.append(k)
        if k == "esc":
            _quit.set()

    def _release(key):
        try:
            k = key.char.lower() if getattr(key, "char", None) else key.name.lower()
        except Exception:
            k = str(key).replace("Key.", "").lower()
        with _lock:
            _held.discard(k)

    lst = _kb.Listener(on_press=_press, on_release=_release)
    lst.daemon = True
    lst.start()
    return True


def _held_keys() -> FrozenSet[str]:
    with _lock:
        return frozenset(_held)


def _pop_events() -> list[str]:
    with _lock:
        evts = list(_events)
        _events.clear()
    return evts


def _msvcrt_poll() -> tuple[FrozenSet[str], list[str]]:
    """Non-blocking key read for Windows msvcrt fallback."""
    import msvcrt

    keys: Set[str] = set()
    events: list[str] = []
    while msvcrt.kbhit():
        ch = msvcrt.getwch().lower()
        if ch == "\x1b":
            _quit.set()
        if ch in (" ", "r"):
            events.append(ch)
        keys.add(ch)
    return frozenset(keys), events


# ── Gamepad input (optional) ──────────────────────────────────────────────────

class GamepadReader:
    """Reads a USB gamepad via pygame.  Left stick → XY, right stick Y → Z,
    right stick X → wrist roll, right trigger → close, left trigger → open."""

    DEADZONE = 0.12

    def __init__(self) -> None:
        import pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("no gamepad detected")
        self._js = pygame.joystick.Joystick(0)
        self._js.init()
        print(f"[teleop] gamepad: {self._js.get_name()}")

    def read(self) -> tuple[float, float, float, float, float]:
        """Return (dx, dy, dz, d_wrist, gripper_axis) in [-1, 1]."""
        import pygame

        pygame.event.pump()

        def _axis(i: int) -> float:
            v = self._js.get_axis(i)
            return v if abs(v) > self.DEADZONE else 0.0

        # Standard Xbox / PS mapping (axis indices may vary by controller)
        lx = _axis(0)   # left stick X  → robot X
        ly = -_axis(1)  # left stick Y  → robot Y (invert: push forward = +Y)
        rz = -_axis(4)  # right stick Y → robot Z
        rw = _axis(3)   # right stick X → wrist roll
        # Triggers: axis 2 = left (−1 rest→+1 pressed), axis 5 = right
        grip = _axis(5) - _axis(2)  # positive = close, negative = open
        return lx, ly, rz, rw, grip


# ── Rerun visualization ───────────────────────────────────────────────────────

def _open_usb_webcam(indices=(0, 1, 2, 3)):
    import cv2
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[rerun] USB webcam opened at index {idx}")
                return cap
            cap.release()
    print("[rerun] no USB webcam found")
    return None


class TeleopViz:
    def __init__(self) -> None:
        import rerun as rr

        self.rr = rr
        rr.init("rax_teleop", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._frame = 0
        self._webcam = _open_usb_webcam()

    @classmethod
    def try_create(cls) -> "TeleopViz | None":
        try:
            return cls()
        except Exception as e:
            print(f"[rerun] unavailable: {e}")
            return None

    def log(
        self,
        obs,
        kin,
        *,
        target_T: "np.ndarray | None" = None,
        gripper_open: bool = True,
        status: str = "",
    ) -> None:
        import cv2

        rr = self.rr
        self._frame += 1
        rr.set_time("frame", sequence=self._frame)

        # Camera image with HUD
        img = np.ascontiguousarray(obs.left)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = img.copy()
        cx = int(round(obs.intrinsics.cx))
        cy = int(round(obs.intrinsics.cy))
        cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
        grip_lbl = "OPEN" if gripper_open else "CLOSED"
        cv2.putText(
            img,
            f"TELEOP  grip:{grip_lbl}  {status}",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 0),
            2,
        )
        rr.log("camera/image", rr.Image(img))

        if self._webcam is not None and self._webcam.isOpened():
            ok, frame = self._webcam.read()
            if ok:
                import cv2 as _cv2
                rr.log("webcam/image", rr.Image(_cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)))

        # Camera position in world
        T_cam = np.asarray(obs.T_base_cam, dtype=float)
        eye = T_cam[:3, 3]
        z_cam = T_cam[:3, 2] / max(float(np.linalg.norm(T_cam[:3, 2])), 1e-9)
        rr.log("world/camera", rr.Points3D([eye], colors=[80, 160, 255], radii=0.012))
        rr.log(
            "world/camera/axis",
            rr.LineStrips3D(
                [np.stack([eye, eye + 0.20 * z_cam])],
                radii=0.003,
                colors=[[255, 255, 80]],
            ),
        )

        # EE frame (RGB axes)
        T_ee = kin.forward_kinematics(obs.joints_deg)
        p_ee = T_ee[:3, 3]
        rr.log("world/ee", rr.Points3D([p_ee], colors=[255, 120, 60], radii=0.016))
        for i, col in enumerate([[220, 60, 60], [60, 220, 60], [60, 60, 220]]):
            ax = T_ee[:3, i] * 0.08
            rr.log(
                f"world/ee/ax{i}",
                rr.LineStrips3D([np.stack([p_ee, p_ee + ax])], radii=0.002, colors=[col]),
            )

        # Target (where IK is commanded)
        if target_T is not None:
            p_t = np.asarray(target_T, dtype=float)[:3, 3]
            rr.log("world/target", rr.Points3D([p_t], colors=[255, 0, 255], radii=0.013))
            rr.log(
                "world/target_line",
                rr.LineStrips3D(
                    [np.stack([p_ee, p_t])], radii=0.002, colors=[[200, 0, 200, 160]]
                ),
            )

        # Arm skeleton if kinematics exposes link chain
        try:
            lk = getattr(kin, "_kin", kin)
            chain = lk.get_link_transforms_chain(obs.joints_deg)
            if chain:
                pts = np.stack([T[:3, 3] for _, T in chain])
                rr.log(
                    "world/arm_chain",
                    rr.LineStrips3D([pts], radii=0.005, colors=[[100, 140, 255]]),
                )
        except Exception:
            pass

        # lerobot sim3d: URDF mesh + ground plane
        try:
            from lerobot.utils.manipulation_sim3d import log_manipulation_sim3d

            lk = getattr(kin, "_kin", kin)
            log_manipulation_sim3d(
                frame_sequence=self._frame,
                kinematics=lk,
                joint_deg=np.asarray(obs.joints_deg, dtype=float),
                object_centers_base=None,
                object_half_sizes_base=None,
                object_labels=None,
                focus_object_index=None,
                plan_summary=f"teleop {status}",
                ground_plane_z_m=0.0,
            )
        except Exception:
            pass

        # Joint angle scalars
        for name, val in zip(kin.joint_names, obs.joints_deg):
            try:
                rr.log(f"joints/{name}", rr.Scalars(float(val)))
            except Exception:
                pass

        try:
            rr.flush()
        except Exception:
            pass


# ── Arm factory ───────────────────────────────────────────────────────────────

def _build_arm(args):
    if args.backend == "mock":
        from manipulation.arms.kinematics import CartesianKinematics
        from manipulation.arms.mock_arm import MockArm

        return MockArm(), CartesianKinematics()

    if args.backend == "so101":
        from robots.arms.lerobot_so101.driver import So101Arm

        arm = So101Arm(
            port=args.port,
            urdf=args.urdf,
            ee_frame=args.ee_frame,
            gripper_camera_tf=args.gripper_camera_tf,
        )
        return arm, arm.kin

    raise ValueError(f"unknown backend: {args.backend!r}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backend", choices=["mock", "so101"], default="mock")
    p.add_argument("--port", default="COM4", help="SO-101 serial port")
    p.add_argument("--urdf", default="SO101/so101_new_calib.urdf")
    p.add_argument("--ee-frame", default="gripper_frame_link")
    p.add_argument("--gripper-camera-tf", default="0.04,0,0.07,0,-0.5,0")
    p.add_argument(
        "--step-mm", type=float, default=3.0,
        help="translation step per control tick (mm)",
    )
    p.add_argument(
        "--step-deg", type=float, default=4.0,
        help="wrist roll step per control tick (°)",
    )
    p.add_argument("--hz", type=float, default=50.0, help="control loop rate")
    p.add_argument("--ik-pos-weight", type=float, default=4.0)
    p.add_argument("--ik-ori-weight", type=float, default=0.4)
    p.add_argument("--rerun", action="store_true", help="open Rerun 3D viewer")
    p.add_argument("--gamepad", action="store_true", help="use USB gamepad instead of keyboard")
    args = p.parse_args()

    print(__doc__)
    print(
        f"[teleop] backend={args.backend}  hz={args.hz}"
        f"  step={args.step_mm}mm / {args.step_deg}°  rerun={args.rerun}"
    )

    arm, kin = _build_arm(args)
    viz = TeleopViz.try_create() if args.rerun else None

    # Input device setup
    gamepad: GamepadReader | None = None
    use_pynput = False
    if args.gamepad:
        try:
            gamepad = GamepadReader()
        except Exception as e:
            print(f"[teleop] gamepad init failed: {e}; falling back to keyboard")
    if gamepad is None:
        use_pynput = _start_pynput()
        if use_pynput:
            print("[teleop] keyboard listener ready (pynput — hold keys for smooth motion)")
        else:
            print("[teleop] pynput not found — using msvcrt (one step per keypress)")
            print("         pip install pynput   for smooth hold-key motion")

    # Read initial state and record home
    print("[teleop] reading initial arm state...")
    has_fast_read = hasattr(arm, "read_joints")
    if has_fast_read:
        q_init, grip_init = arm.read_joints()
    else:
        obs0 = arm.get_observation()
        q_init, grip_init = obs0.joints_deg, obs0.gripper_pct
    home_joints = q_init.copy()
    gripper_open = grip_init > 50.0
    prev_grip_axis = 0.0
    q_current = q_init.copy()
    print(f"[teleop] home joints (°): {np.round(home_joints, 1)}")
    print("[teleop] ready — WASD/OP move XYZ, Q/E wrist, Space gripper, R home, Esc quit\n")

    dt = 1.0 / args.hz
    step_m = args.step_mm * 1e-3
    step_deg = args.step_deg
    target_T: np.ndarray | None = None
    status = "idle"

    # Identify wrist-roll joint index (direct control for Q/E)
    jnames = list(kin.joint_names)
    wrist_roll_idx: int | None = None
    for _wn in ("wrist_roll", "rz"):
        if _wn in jnames:
            wrist_roll_idx = jnames.index(_wn)
            break

    # ── Viz background thread ─────────────────────────────────────────────────
    # Camera reads (~33 ms) and Rerun logging run in a separate thread so they
    # never stall the control loop.  The control loop pushes (q, target_T,
    # gripper_open, status) into a single-slot queue; the viz thread drains it.
    _viz_q: queue.Queue = queue.Queue(maxsize=1)

    def _viz_worker():
        while True:
            item = _viz_q.get()
            if item is None:
                break
            vq, v_target_T, v_gripper_open, v_status = item
            try:
                obs = arm.get_observation() if not has_fast_read else None
                if has_fast_read:
                    left, right = arm._cam.read_stereo_rectified()
                    T_base_cam = kin.forward_kinematics(vq) @ arm.T_ee_cam
                    from manipulation.arms.arm_interface import Observation as _Obs
                    obs = _Obs(
                        left=np.asarray(left), right=np.asarray(right),
                        joints_deg=vq, gripper_pct=100.0 if v_gripper_open else 0.0,
                        intrinsics=arm.intr, T_base_cam=T_base_cam,
                    )
                viz.log(obs, kin, target_T=v_target_T,
                        gripper_open=v_gripper_open, status=v_status)
            except Exception as e:
                print(f"[viz] {e}")

    if viz is not None:
        _vt = threading.Thread(target=_viz_worker, daemon=True)
        _vt.start()

    def _push_viz(vq, v_target_T, v_gripper_open, v_status):
        if viz is None:
            return
        try:
            _viz_q.put_nowait((vq.copy(), v_target_T, v_gripper_open, v_status))
        except queue.Full:
            pass  # viz is behind — drop frame, never block control loop

    try:
        while not _quit.is_set():
            t0 = time.time()

            # ── fast joint read (no camera) ──
            if has_fast_read:
                q_current, grip_pct = arm.read_joints()
                if not gripper_open and grip_pct > 50.0:
                    gripper_open = True
                elif gripper_open and grip_pct < 50.0:
                    gripper_open = False
            else:
                obs_now = arm.get_observation()
                q_current = obs_now.joints_deg

            # ── gather input ──
            if gamepad is not None:
                lx, ly, rz_pad, rw, grip_axis = gamepad.read()
                dx = lx * step_m
                dy = ly * step_m
                dz = rz_pad * step_m
                drz = rw * step_deg
                events: list[str] = []
                if grip_axis > 0.5 and prev_grip_axis <= 0.5:
                    events.append(" ")
                elif grip_axis < -0.5 and prev_grip_axis >= -0.5:
                    events.append(" ")
                prev_grip_axis = grip_axis
            elif use_pynput:
                keys = _held_keys()
                events = _pop_events()
                dx = (("d" in keys) - ("a" in keys)) * step_m
                dy = (("w" in keys) - ("s" in keys)) * step_m
                dz = (("o" in keys) - ("p" in keys)) * step_m
                drz = (("q" in keys) - ("e" in keys)) * step_deg
            else:
                keys, events = _msvcrt_poll()
                dx = (("d" in keys) - ("a" in keys)) * step_m
                dy = (("w" in keys) - ("s" in keys)) * step_m
                dz = (("o" in keys) - ("p" in keys)) * step_m
                drz = (("q" in keys) - ("e" in keys)) * step_deg

            # ── one-shot events ──
            for ev in events:
                if ev == " ":
                    gripper_open = not gripper_open
                    arm.set_gripper(100.0 if gripper_open else 0.0)
                    print(f"[teleop] gripper {'OPEN' if gripper_open else 'CLOSED'}")
                elif ev == "r":
                    print("[teleop] returning to home...")
                    arm.send_joint_targets(home_joints)
                    q_current = home_joints.copy()
                    target_T = None

            # ── Cartesian XYZ via IK ──
            has_cart = dx != 0 or dy != 0 or dz != 0
            has_wrist = drz != 0

            if has_cart or has_wrist:
                if has_cart:
                    T_ee = kin.forward_kinematics(q_current)
                    T_target = T_ee.copy()
                    T_target[0, 3] += dx
                    T_target[1, 3] += dy
                    T_target[2, 3] += dz
                    q_new = kin.inverse_kinematics(
                        q_current, T_target,
                        position_weight=args.ik_pos_weight,
                        orientation_weight=args.ik_ori_weight,
                    )
                    target_T = T_target
                else:
                    q_new = q_current.copy()
                    target_T = None

                if has_wrist and wrist_roll_idx is not None:
                    q_new[wrist_roll_idx] += drz

                arm.send_joint_targets(q_new)
                p = kin.forward_kinematics(q_new)[:3, 3]
                status = f"({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})m"
            else:
                target_T = None
                status = "idle"

            # ── push to viz thread (non-blocking) ──
            _push_viz(q_current, target_T, gripper_open, status)

            # ── Rate limit ──
            elapsed = time.time() - t0
            slack = dt - elapsed
            if slack > 0:
                time.sleep(slack)

    except KeyboardInterrupt:
        pass
    finally:
        if viz is not None:
            _viz_q.put(None)
        print("\n[teleop] stopped")


if __name__ == "__main__":
    main()
