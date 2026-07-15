#!/usr/bin/env python
"""
Persistent lerobot gaze robot daemon.

Robot stays connected forever. Loads YOLO + kinematics ONCE at startup.
Accepts commands via stdin; the full state machine (SEARCH→APPROACH→GRASP)
restarts in ~100 ms between commands with zero reload penalty.

stdin commands:
  PICK <query>       — find and grasp the object matching <query>
  DROP               — open gripper (release held object)
  HOME               — move arm to home/standby position
  STOP               — abort current task

stdout protocol (parsed by livekit_gaze_agent.py):
  LOADING ...        — startup progress
  READY              — init done, robot connected, YOLO loaded
  STARTING PICK ...  — beginning a pick run
  STATE <kw>         — state transition (SEARCH / APPROACH / GRASP / …)
  DONE               — pick completed (gripper closed on object)
  DROPPED            — gripper opened
  FAILED <reason>    — pick failed
  STOPPED            — task aborted
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
import logging

# ── Paths ──────────────────────────────────────────────────────────────────
LEROBOT_SRC = "/mnt/c/Users/labot/Documents/lerobot/src"
LEROBOT_DIR = "/mnt/c/Users/labot/Documents/lerobot"
CONFIG_PATH = os.environ.get(
    "LERO_CONFIG",
    "/mnt/c/Users/labot/Documents/lerobot/configs/gaze_engine_zmq.yaml",
)

if LEROBOT_SRC not in sys.path:
    sys.path.insert(0, LEROBOT_SRC)

os.chdir(LEROBOT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="[lero-daemon] %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _out(msg: str) -> None:
    print(msg, flush=True)


# ── Load config ─────────────────────────────────────────────────────────────
_out("LOADING CONFIG...")
try:
    import draccus
    from lerobot.manipulation.visual_servo.gaze_engine import GazeEngineConfig

    _orig_argv = sys.argv[:]
    sys.argv = ["lerobot-gaze-engine", "--config", CONFIG_PATH]
    cfg: GazeEngineConfig = draccus.parse(config_class=GazeEngineConfig)
    sys.argv = _orig_argv
    _out(f"CONFIG query={cfg.query!r}")
except Exception as e:
    _out(f"FAILED CONFIG: {e}")
    sys.exit(1)

# ── Pre-load: YOLO in background, robot + kinematics in parallel ───────────
_out("LOADING YOLO...")

import numpy as np
from lerobot.perception.yolo_world import YoloWorldDetector
from lerobot.manipulation.visual_servo.gaze_engine import _search_vocab_queries, ARM_MOTORS

_search_vocab = _search_vocab_queries(cfg)
_det_imgsz = int(getattr(cfg, "detector_imgsz", 320)) or None

_yolo_box: list = []
_yolo_err: list = []


def _load_yolo() -> None:
    try:
        d = YoloWorldDetector(
            cfg.model_path,
            conf=float(getattr(cfg, "detector_raw_conf", 0.10)),
            imgsz=_det_imgsz,
        )
        d.set_query(_search_vocab[0])
        d.predict_rgb(np.zeros((64, 64, 3), dtype=np.uint8))  # warmup
        _yolo_box.append(d)
    except Exception as exc:
        _yolo_err.append(exc)


yolo_thread = threading.Thread(target=_load_yolo, daemon=True, name="yolo-load")
yolo_thread.start()

_out("LOADING ROBOT...")
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots import make_robot_from_config

try:
    _kin = RobotKinematics(cfg.urdf, cfg.ee_frame, ARM_MOTORS)
except Exception as e:
    _out(f"FAILED KINEMATICS: {e}")
    sys.exit(1)

try:
    _robot = make_robot_from_config(cfg.robot)
    _robot.connect()
    _out("ROBOT connected")
except Exception as e:
    _out(f"FAILED ROBOT: {e}")
    sys.exit(1)

yolo_thread.join()
if _yolo_err:
    _out(f"FAILED YOLO: {_yolo_err[0]}")
    sys.exit(1)

_detector: YoloWorldDetector = _yolo_box[0]

# ── Monkey-patch: make run_gaze_engine free of reload cost ─────────────────
# Every run_gaze_engine call will:
#   1. Import RobotKinematics / YoloWorldDetector → gets pre-loaded objects
#   2. Call make_robot_from_config → gets pre-loaded robot
#   3. Call robot.connect()        → no-op
#   4. Call _try_init_rerun        → idempotent (init once, skip after)
#   5. Call _startup_search_probe  → no-op (instant restart, already positioned)

import lerobot.model.kinematics as _kin_mod
import lerobot.perception.yolo_world as _det_mod
import lerobot.manipulation.visual_servo.gaze_engine as _ge_mod

_kin_mod.RobotKinematics = lambda *a, **kw: _kin
_det_mod.YoloWorldDetector = lambda *a, **kw: _detector
_ge_mod.make_robot_from_config = lambda *a, **kw: _robot
_robot.connect = lambda: None

# Rerun: init once, skip on subsequent calls
_rerun_inited = [False]
_orig_try_init_rerun = _ge_mod._try_init_rerun


def _idempotent_rerun(cfg):
    if _rerun_inited[0]:
        return True
    result = _orig_try_init_rerun(cfg)
    _rerun_inited[0] = True
    return result


_ge_mod._try_init_rerun = _idempotent_rerun

# Startup probe: skip — robot is already in position, restart is instant
_ge_mod._startup_search_probe = lambda **kw: (None, None)

_out("READY")

# ── Gripper open position (for DROP) ───────────────────────────────────────
GRIPPER_OPEN_DEG  = float(os.environ.get("GRIPPER_OPEN_DEG", "100.0"))
GRIPPER_CLOSE_DEG = float(os.environ.get("GRIPPER_CLOSE_DEG", "5.0"))
GRIPPER_MOTOR     = "gripper"

# ── Manual jog (Cartesian teleop from the phone UI) ─────────────────────────
# Same geometry as teleop_cam.py: forward/right are in the CAMERA frame, up is
# world +z. The camera pitch is locked at the current pose so the first-person
# view stays level (no droop) while the arm translates.
from lerobot.manipulation.visual_servo.gaze_engine import parse_tf_string
from lerobot.manipulation.visual_servo.cvs_engine import build_look_at_R

JOG_Z_FLOOR = float(os.environ.get("JOG_Z_FLOOR", "-0.02"))


def _build_T_ee_cam() -> np.ndarray:
    """Gripper→camera transform, matching gaze_engine's construction exactly."""
    T = np.asarray(parse_tf_string(cfg.gripper_camera_tf), dtype=np.float64)
    if bool(getattr(cfg, "invert_gripper_camera_tf", False)):
        T = np.linalg.inv(T)
    trim = float(getattr(cfg, "gripper_camera_pitch_trim_deg", 0.0))
    if abs(trim) > 1e-6:
        from scipy.spatial.transform import Rotation as _R
        T[:3, :3] = T[:3, :3] @ _R.from_euler("x", trim, degrees=True).as_matrix()
    return T


_T_ee_cam = _build_T_ee_cam()

try:
    _obs0 = _robot.get_observation()
    _q0 = np.array([float(_obs0[f"{m}.pos"]) for m in ARM_MOTORS], dtype=np.float64)
    _z0 = (np.asarray(_kin.forward_kinematics(_q0)) @ _T_ee_cam)[:3, 2]
    _PITCH_LOCK = float(np.arctan2(-_z0[2], np.hypot(_z0[0], _z0[1])))
    logger.info("jog camera pitch locked at %+.1f deg below horizon", np.degrees(_PITCH_LOCK))
except Exception as _e:  # noqa: BLE001
    _PITCH_LOCK = 0.5
    logger.warning("jog pitch-lock init failed (%s); defaulting", _e)


def _do_move(forward: float, right: float, up: float) -> None:
    """One Cartesian jog step (metres), holding the locked camera pitch."""
    obs = _robot.get_observation()
    q = np.array([float(obs[f"{m}.pos"]) for m in ARM_MOTORS], dtype=np.float64)
    T_ee = np.asarray(_kin.forward_kinematics(q), dtype=np.float64)
    R_cam = (T_ee @ _T_ee_cam)[:3, :3]
    d_base = R_cam @ np.array([right, 0.0, forward], dtype=np.float64)
    d_base[2] += up

    z_cur = R_cam[:, 2]
    az = np.arctan2(z_cur[1], z_cur[0])
    z_des = np.array([
        np.cos(_PITCH_LOCK) * np.cos(az),
        np.cos(_PITCH_LOCK) * np.sin(az),
        -np.sin(_PITCH_LOCK),
    ])
    R_goal = build_look_at_R(T_ee[:3, :3], _T_ee_cam, z_des)

    T_goal = T_ee.copy()
    T_goal[:3, :3] = R_goal
    T_goal[:3, 3] = T_ee[:3, 3] + d_base
    T_goal[2, 3] = max(T_goal[2, 3], JOG_Z_FLOOR)
    q_new = _kin.inverse_kinematics(q, T_goal, position_weight=1.0, orientation_weight=1.0)
    _robot.send_action({f"{m}.pos": float(q_new[i]) for i, m in enumerate(ARM_MOTORS)})


def _do_grip(open_it: bool) -> None:
    obs = _robot.get_observation()
    action = {k: obs[k] for k in obs if str(k).endswith(".pos")}
    gkey = next((k for k in action if GRIPPER_MOTOR in k), None)
    if gkey:
        action[gkey] = GRIPPER_OPEN_DEG if open_it else GRIPPER_CLOSE_DEG
        _robot.send_action(action)


# ── Command loop ────────────────────────────────────────────────────────────
from lerobot.manipulation.visual_servo.gaze_engine import run_gaze_engine

_run_thread: threading.Thread | None = None
_stop_requested = threading.Event()


def _raise_keyboard_interrupt_in(thread: threading.Thread) -> None:
    import ctypes
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread.ident),
        ctypes.py_object(KeyboardInterrupt),
    )


def _do_pick(query: str) -> None:
    _out(f"STARTING PICK {query}")
    cfg.query = query
    _stop_requested.clear()
    try:
        run_gaze_engine(cfg)
        _out("DONE")
    except KeyboardInterrupt:
        _out("STOPPED")
    except Exception as exc:
        _out(f"FAILED {exc}")


def _do_drop() -> None:
    _out("STARTING DROP")
    try:
        obs = _robot.get_observation()
        action = dict(obs)  # copy current joint positions
        # Find the gripper motor key
        gripper_key = next(
            (k for k in action if GRIPPER_MOTOR in k and "pos" in k), None
        )
        if gripper_key:
            action[gripper_key] = GRIPPER_OPEN_DEG
            _robot.send_action(action)
            time.sleep(0.5)
        _out("DROPPED")
    except Exception as exc:
        _out(f"FAILED DROP: {exc}")


def _do_home() -> None:
    """Send arm to a safe standby pose (all joints near zero)."""
    _out("STARTING HOME")
    try:
        obs = _robot.get_observation()
        action = {k: 0.0 for k in obs if ".pos" in k}
        _robot.send_action(action)
        time.sleep(1.0)
        _out("HOME")
    except Exception as exc:
        _out(f"FAILED HOME: {exc}")


for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue

    upper = line.upper()

    # Abort running task
    if upper == "STOP":
        if _run_thread is not None and _run_thread.is_alive():
            _stop_requested.set()
            _raise_keyboard_interrupt_in(_run_thread)
        else:
            _out("STOPPED")
        continue

    # ── Manual jog / gripper (fast path, silent — no protocol reply) ──
    # These only run between autonomous tasks; while a PICK holds the loop the
    # lines simply queue. Errors go to stderr so they never look like FAILED.
    if upper.startswith("MOVE"):
        if _run_thread is None or not _run_thread.is_alive():
            try:
                _, sf, sr, su = line.split()
                _do_move(float(sf), float(sr), float(su))
            except Exception as exc:  # noqa: BLE001
                logger.warning("MOVE failed: %s", exc)
        continue

    if upper.startswith("GRIP"):
        if _run_thread is None or not _run_thread.is_alive():
            try:
                _do_grip("OPEN" in upper)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GRIP failed: %s", exc)
        continue

    # Wait for any active task before starting next
    if _run_thread is not None and _run_thread.is_alive():
        _out("BUSY — finishing current task first")
        _run_thread.join()

    if upper.startswith("PICK"):
        query = line[4:].strip() or cfg.query
        _run_thread = threading.Thread(target=_do_pick, args=(query,), daemon=True)
        _run_thread.start()
        _run_thread.join()

    elif upper == "DROP":
        _run_thread = threading.Thread(target=_do_drop, daemon=True)
        _run_thread.start()
        _run_thread.join()

    elif upper == "HOME":
        _run_thread = threading.Thread(target=_do_home, daemon=True)
        _run_thread.start()
        _run_thread.join()

    else:
        # Bare query text → treat as PICK
        _run_thread = threading.Thread(target=_do_pick, args=(line,), daemon=True)
        _run_thread.start()
        _run_thread.join()
