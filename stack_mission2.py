# Stack mission v2 — smooth continuous control + robust tracking + dual-camera UI.
#
# What changed vs v1 (user feedback):
#  * MOTION: no more stop-and-go waypoints during servo. A single ~14 Hz control
#    loop streams joint targets each tick: desired EE velocity (camera frame) is
#    EMA-filtered (acceleration limiting), converted via DLS IK, and joint deltas
#    are rate-clamped. Straight-line approach law: forward speed backs off as
#    lateral pixel error grows, so corrections never zigzag.
#  * TRACKING: strict HSV (saturation-gated, wood-grain immune) + window
#    continuity + a base-frame ANCHOR maintained from FK: when the blob drops
#    out, the anchor reprojects through the current pose to predict the search
#    window. YOLO runs in a PARALLEL thread (acquire + revalidate ~2.5 s) and
#    never sits in the control loop.
#  * UI: second (room) camera proxied from camserv on :5000; overlays only show
#    FRESH tracks (age < 0.7 s) so stale boxes don't wander; anchor shown as a
#    circle.
import json, math, os, subprocess, sys, tempfile, threading, time
from collections import deque

import cv2
import numpy as np
import requests as pyrequests
from flask import Flask, Response, jsonify, request

sys.path.insert(0, r"C:\Users\labot\Documents\lerobot\src")

from lerobot.model.kinematics import RobotKinematics
from lerobot.perception.yolo_world import YoloWorldDetector
from lerobot.manipulation.visual_servo.gaze_engine import ARM_MOTORS, parse_tf_string
from lerobot.manipulation.yolo_track.motion_primitives import send_joint_target_smoothly
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig

# The gripper servo (ID 6) replies too slowly for lerobot's handshake ping
# timeout (raw pings see it 100%; lerobot's missed 48/48). Skip the existence
# assert: sync WRITES need no ACK, and every read in this script already
# tolerates a miss (gripper_current returns None, observe retries).
from lerobot.motors.feetech.feetech import FeetechMotorsBus as _FTBus

_FTBus._handshake = lambda self: None

LEROBOT = r"C:\Users\labot\Documents\lerobot"
OUT = os.path.join(tempfile.gettempdir(), "rax_stack_mission")  # debug-image dumps
os.makedirs(OUT, exist_ok=True)
TF = "-0.0503,0.0906,-0.1730,-0.2921,1.0770,-2.1688"
# [shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll]
# pan  -6.7 -> +5.0 : turn the whole robot a few degrees LEFT, so the view swings RIGHT
# roll -28.4 -> 90  : twist the wrist 90deg so the jaws are square to the cube
GRASP_ROLL = 90.0
# Display-only: degrees added to wrist_roll before drawing the URDF, because the
# URDF's roll zero is rotated from the servo's zero. If the rendered gripper is now
# twisted the OTHER way, flip this sign.
WRIST_RENDER_OFFSET = 90.0
# Same story for the base: the arm sits physically straight while shoulder_pan
# reads about -14deg, so the drawn robot looks swung round. Display only.
BASE_RENDER_OFFSET = 14.1
VIEW = np.array([5.0, 37.1, 48.1, -40.4, GRASP_ROLL])
# New home: captured from the physically-correct folded pose (2026-07-20).
HOME = np.array([-14.1, -99.1, 90.8, 33.2, -4.7])
HAND_UV = (440.0, 394.0)   # measured via /caltip against the real black fingertip
HAND_AREA_MIN = 9000.0
# MEASURED FROM THE URDF + JAW MESH (2026-07-13), do not guess this:
#   moving_jaw_so101_v1.stl, expressed in gripper_frame_link coords, spans
#   X -38.1..-15.8, Y -23.9..+24.1, Z -84.7..+7.3 mm.
# So the jaws hinge 75-85 mm BEHIND gripper_frame_link and the fingertips reach
# only +7 mm past it: `gripper_frame_link` IS the fingertip / grasp centre (it is
# a TCP frame, 98 mm out past gripper_link, the wrist). `kin.forward_kinematics`
# therefore already returns the FINGERTIP, and `T_ee[:3,3]` is the right thing to
# gate the approach on. lookat_engine's `gripper_tip_offset_m = 0.10` does NOT
# transfer to this FK -- applying it pushed the "tip" 10 cm out into empty air
# (TIP->cube read LARGER than grip->cube, which is what exposed the error).
GRIP_TIP_OFFSET_M = 0.007
# The camera sits BEHIND the fingertips on the gripper — measured on the real mount at
# ~10 cm. The shipped hand-eye TF puts it at 20.2 cm, i.e. wrong by 2x in translation on
# top of being ~370 px wrong in rotation. Used to seed/bound calibrate_handeye.
CAM_TIP_M = 0.10
FINAL_INCH_M = 0.006   # we arrive already touching the cube; don't shove it
PLACE_DZ = 0.055
PORT = 8484
CAMSURV = ("http://127.0.0.1:5000", "camsurv123")

# control loop tuning
LOOP_DT = 0.07            # ~14 Hz
V_LAT_MAX = 0.016         # m/s lateral
V_FWD_MAX = 0.030         # m/s approach
ACC_ALPHA = 0.12          # velocity EMA (lower = smoother, more damped)
PAN_ACC_LIMIT = 0.8       # deg/s per tick max change in pan rate — ramps up gently
PAN_RAMP = 0.04           # soft transition factor for pan enable (smooth start)
K_LAT = 0.45              # lateral P gain (0.9 caused left-right hunting)
DEADBAND_PX = 18          # no lateral correction inside this — kills the limit cycle
JOINT_RATE_MAX = 25.0     # deg/s per joint hard clamp (lower = less jerk at stop)

# gaze-engine approach: point at the target (direct-joint pixel P-control, no
# IK, so it can't swing) then step straight down the line of sight, decreasing
# the radius, toward the object's back-projected 3D point (radial-to-object —
# descends ONTO the cube instead of hovering above it).
Z_TABLE = 0.02            # base-frame height of the cube CENTRE. THE sightline is
                          # intersected with THIS plane to localize the cube, so it
                          # must match the real cube-centre height. A 3 cm cube on
                          # the table sits ~1.5–2 cm up; the old 0.05 was ~3 cm too
                          # high, which put the cube too CLOSE + too HIGH (floating
                          # inside the robot in the 3D view) and made the grasp
                          # close ~3 cm ABOVE the cube (contact=False). >>> If the
                          # grasp stops short/high, raise this a few mm; if it
                          # drives into the table, lower it. <<<
TARGET_SIZE_M = 0.03      # cube edge — pinhole range from bbox size (survives close range)
GAZE_KP_PAN = 0.18        # shoulder_pan P-gain on horizontal pixel error (fraction/tick)
GAZE_KP_TILT = 0.30       # wrist_flex  P-gain on vertical pixel error
GAZE_MAX_PAN_DEG = 1.3    # per-tick clamp on the pan gaze step (was 2.2 — too whippy)
GAZE_MAX_TILT_DEG = 2.0   # per-tick clamp on the tilt gaze step
GAZE_DEADBAND_PX = 18.0   # no gaze correction inside this — kills the limit cycle
GAZE_LP = 0.6             # low-pass on the pixel error feeding pan/tilt (0=off)
APPROACH_KP = 0.6         # P-gain on remaining range → forward step
APPROACH_VMAX = 0.030     # m/s cap on the forward step
APPROACH_CENTER_PX = 42.0 # only advance while horizontally centred (pan error small)
GRASP_TRIGGER_M = 0.075   # hand off to the grasp once camera→object range is this small
OBJ_FLOOR_Z = -0.030      # lowest the gripper frame may descend onto the table cube

state = {
    "phase": "IDLE", "detail": "", "joints": [], "gripper": None,
    "p_red": None, "p_green": None, "t0": None, "running": False, "loop_hz": 0.0,
    "dist_mm": None,  # live camera→target range during approach/place
}
log = deque(maxlen=140)
frame_jpeg = [None]
lock = threading.Lock()
bus_lock = threading.RLock()   # Feetech bus is not thread-safe
stop_flag = threading.Event()
mission_thread = [None]
latest_rgb = [None]            # for the YOLO thread


def say(msg):
    log.appendleft(f"{time.strftime('%H:%M:%S')}  {msg}")
    print(msg, flush=True)


def set_phase(phase, detail=""):
    with lock:
        state["phase"], state["detail"] = phase, detail
        state["dist_mm"] = None
    say(f"[{phase}] {detail}" if detail else f"[{phase}]")


class Abort(Exception):
    pass


def checkpoint():
    with lock:
        running = state["running"]
    if stop_flag.is_set() and running:
        raise Abort("stopped by user")


robot = None
kin = None
cam = None
T_ee_cam = parse_tf_string(TF)
fx = fy = cx0 = cy0 = 0.0

# ---------------- strict HSV tracking + FK anchor ----------------
HSV_BANDS = {
    # saturation floor 110 keeps the warm wood grain out of "red"
    "red": [((0, 110, 80), (9, 255, 255)), ((170, 110, 80), (179, 255, 255))],
    "green": [((38, 80, 60), (85, 255, 255))],
}
# Relaxed bands for the second pass INSIDE a predicted window only — the
# looming gripper shades the object (saturation/value drop) during approach.
HSV_BANDS_SOFT = {
    "red": [((0, 70, 45), (11, 255, 255)), ((168, 70, 45), (179, 255, 255))],
    "green": [((36, 55, 40), (88, 255, 255))],
}


class Track:
    __slots__ = ("uv", "bbox_xyxy", "area_px", "clipped", "t")

    def __init__(self, uv, bbox, area, clipped, t):
        self.uv, self.bbox_xyxy = uv, bbox
        self.area_px, self.clipped, self.t = area, clipped, t


class AnchorTracker:
    """Strict-HSV blob tracker with window continuity and a base-frame anchor.

    The anchor (EMA of back-projected fixes) predicts the pixel window through
    detector dropouts using the CURRENT FK pose — the eye-in-hand insight.
    """

    def __init__(self, color, min_area=900):
        self.color = color
        self.min_area = int(min_area)
        self.last: Track | None = None
        self.p_anchor: np.ndarray | None = None
        self.anchor_t = 0.0

    def reset(self):
        self.last = None
        self.p_anchor = None

    def _mask(self, rgb, soft=False):
        hsv = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2HSV)
        m = np.zeros(hsv.shape[:2], np.uint8)
        bands = (HSV_BANDS_SOFT if soft else HSV_BANDS)[self.color]
        for lo, hi in bands:
            m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def _largest(self, mask, ox=0, oy=0, shape=None):
        n, _l, stats, _c = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best = None
        for i in range(1, n):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a < self.min_area:
                continue
            if best is None or a > best[0]:
                x, y, w, h = (int(stats[i, j]) for j in
                              (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
                best = (a, (x + ox, y + oy, x + w + ox, y + h + oy))
        if best is None:
            return None
        a, (x1, y1, x2, y2) = best
        H, W = shape
        clipped = x1 <= 1 or y1 <= 1 or x2 >= W - 2 or y2 >= H - 2
        return Track(((x1 + x2) / 2.0, (y1 + y2) / 2.0), (x1, y1, x2, y2), a, clipped, time.time())

    def predict_uv(self, T_base_cam):
        if self.p_anchor is None:
            return None
        p_cam = T_base_cam[:3, :3].T @ (self.p_anchor - T_base_cam[:3, 3])
        if p_cam[2] < 0.02:
            return None
        return (cx0 + fx * p_cam[0] / p_cam[2], cy0 + fy * p_cam[1] / p_cam[2])

    def update_anchor(self, uv, z_m, T_base_cam, now):
        d = np.array([(uv[0] - cx0) / fx, (uv[1] - cy0) / fy, 1.0])
        p = T_base_cam[:3, 3] + T_base_cam[:3, :3] @ (d * z_m)  # z along optical axis
        if self.p_anchor is None:
            self.p_anchor = p
        else:
            self.p_anchor = 0.6 * self.p_anchor + 0.4 * p
        self.anchor_t = now

    def track(self, rgb, T_base_cam=None):
        H, W = rgb.shape[:2]
        windows = []
        centre = None
        if self.last is not None and time.time() - self.last.t < 1.5:
            centre = self.last.uv
        elif T_base_cam is not None:
            centre = self.predict_uv(T_base_cam)   # FK prediction after dropout
        if centre is not None:
            cxp, cyp = int(centre[0]), int(centre[1])
            r = 140
            windows.append((max(0, cxp - r), max(0, cyp - r), min(W, cxp + r), min(H, cyp + r)))
        windows.append(None)
        for w in windows:
            if w is None:
                tr = self._largest(self._mask(rgb), 0, 0, (H, W))
            else:
                x1, y1, x2, y2 = w
                tr = self._largest(self._mask(rgb[y1:y2, x1:x2]), x1, y1, (H, W))
                if tr is None:
                    # shaded/blurred object inside a trusted window: relax bands
                    tr = self._largest(self._mask(rgb[y1:y2, x1:x2], soft=True), x1, y1, (H, W))
            if tr is not None:
                self.last = tr
                return tr
        return None


red_tracker = AnchorTracker("red")
green_tracker = AnchorTracker("green")

# ---------------- parallel YOLO validation ----------------
detector = None
yolo_red = {"t": 0.0, "box": None}
pending_query = [None]      # set by /setquery, applied inside the YOLO thread


def yolo_worker():
    while True:
        time.sleep(2.5)
        # apply a pending query change here — the model must only be mutated on
        # the thread that runs it (set_classes from Flask crashes it).
        if pending_query[0] is not None and detector is not None:
            q = pending_query[0]; pending_query[0] = None
            try:
                detector.set_query(q)
                with lock:
                    state["query"] = q
                say(f"detection query set: {q}")
            except Exception as e:
                say(f"query change failed: {e}")
        with lock:
            rgb = latest_rgb[0]
        if rgb is None or detector is None:
            continue
        try:
            dets = detector.predict_rgb(np.ascontiguousarray(rgb))
        except Exception:
            continue
        with lock:
            yolo_red["t"] = time.time()
            yolo_red["box"] = dets[0].xyxy if dets else None


def find_red(rgb, T_base_cam=None):
    """Acquire via YOLO (parallel thread) OR a big strict-HSV blob (the S>=110
    gate already excludes wood grain, and YOLO misses edge-clipped slivers);
    afterwards window/anchor continuity tracks."""
    if red_tracker.last is not None or red_tracker.p_anchor is not None:
        tr = red_tracker.track(rgb, T_base_cam)
        if tr is not None:
            return tr
    with lock:
        box, t = yolo_red["box"], yolo_red["t"]
    if box is not None and time.time() - t < 4.0:
        x1, y1, x2, y2 = box
        red_tracker.last = Track(((x1 + x2) / 2, (y1 + y2) / 2), tuple(box),
                                 int((x2 - x1) * (y2 - y1)), False, time.time())
        return red_tracker.track(rgb, T_base_cam)
    tr = red_tracker.track(rgb, None)   # full-frame strict mask
    if tr is not None and tr.area_px >= 2500:
        return tr
    red_tracker.last = None             # too small: don't latch a speck
    return None


def find_green(rgb, T_base_cam=None):
    return green_tracker.track(rgb, T_base_cam)


# ---------------- robot I/O ----------------
def publish(rgb, joints=None):
    img = np.ascontiguousarray(rgb[:, :, ::-1])
    now = time.time()
    # CYAN CROSS = HAND_UV, the fingertips as MEASURED in the image (/caltip).
    cv2.drawMarker(img, (int(HAND_UV[0]), int(HAND_UV[1])), (60, 200, 255), cv2.MARKER_CROSS, 26, 2)
    # MAGENTA CIRCLE = the same fingertips as PREDICTED by FK + the hand-eye TF.
    # These two must land on top of each other (see tip_pixel). The gap between
    # them IS the hand-eye error, in pixels, live. Run Calib to close it.
    if joints is not None and kin is not None and fx > 0:
        try:
            uv = tip_pixel(joints)
        except Exception:
            uv = None
        if uv is not None:
            u, v = int(round(uv[0])), int(round(uv[1]))
            gap = math.hypot(uv[0] - HAND_UV[0], uv[1] - HAND_UV[1])
            col = (255, 0, 255) if gap > 40 else (0, 255, 0)
            if -200 < u < img.shape[1] + 200 and -200 < v < img.shape[0] + 200:
                cv2.circle(img, (u, v), 9, col, 2)
                cv2.line(img, (u, v), (int(HAND_UV[0]), int(HAND_UV[1])), col, 1)
            cv2.putText(img, f"hand-eye err {gap:.0f}px", (10, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    # YELLOW BOX = the LOCKED 3D cube point, projected BACK into the image.
    # This is the only honest check on the localization. The logs cannot tell you it is
    # wrong -- they are computed FROM the same broken transform, so they will cheerfully
    # print "tip->cube=27mm" while the gripper is nowhere near the cube. But a 3D point
    # reprojected onto the picture either lands on the red cube you can see, or it does
    # not. If this box is not sitting on the cube, the LOCALIZATION is wrong and nothing
    # downstream can save it. (Same trick as the pink line, applied to the target.)
    if joints is not None and kin is not None and fx > 0:
        with lock:
            p3 = state.get("obj3d")
        if p3 is not None:
            try:
                T_ee = np.asarray(kin.forward_kinematics(np.asarray(joints, np.float64)))
                uvc = project_base(np.asarray(p3, np.float64), T_ee @ T_ee_cam)
            except Exception:
                uvc = None
            if uvc is not None:
                u, v = int(round(uvc[0])), int(round(uvc[1]))
                if -300 < u < img.shape[1] + 300 and -300 < v < img.shape[0] + 300:
                    cv2.rectangle(img, (u - 16, v - 16), (u + 16, v + 16), (0, 235, 255), 2)
                    cv2.drawMarker(img, (u, v), (0, 235, 255), cv2.MARKER_TILTED_CROSS, 12, 1)
                    cv2.putText(img, "3D lock", (u - 18, v - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 235, 255), 1)
    for tk, color, name in ((red_tracker, (0, 0, 255), "red"), (green_tracker, (0, 200, 0), "green")):
        tr = tk.last
        if tr is not None and now - tr.t < 0.7:   # fresh only — no wandering stale boxes
            x1, y1, x2, y2 = (int(v) for v in tr.bbox_xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # label with the DISTANCE, from apparent size: range = f * edge / width.
            # Needs only the lens focal length and the cube's real size, so it stays
            # honest regardless of the camera-mount numbers.
            w_px = float(max(4, x2 - x1))
            rng_cm = fx * CUBE_EDGE_M / w_px * 100.0
            cv2.putText(img, f"{name} {rng_cm:.0f}cm", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    with lock:
        cv2.putText(img, state["phase"], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 255, 120), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if ok:
        with lock:
            frame_jpeg[0] = buf.tobytes()


def observe(overlay=True):
    checkpoint()
    obs = None
    for attempt in range(8):
        try:
            with bus_lock:
                obs = robot.get_observation()
            break
        except ConnectionError:
            if attempt == 7:
                raise
            time.sleep(0.08)
    joints = np.array([float(obs[f"{m}.pos"]) for m in ARM_MOTORS])
    rgb = np.asarray(obs["front"])
    with lock:
        state["joints"] = [round(float(v), 1) for v in joints]
        state["gripper"] = round(float(obs.get("gripper.pos", -1)), 1)
        latest_rgb[0] = rgb
    if overlay:
        publish(rgb, joints)
    return joints, rgb, obs


def send_joints(q, gripper=None):
    act = {f"{m}.pos": float(v) for m, v in zip(ARM_MOTORS, q)}
    if gripper is not None:
        act["gripper.pos"] = float(gripper)
    with bus_lock:
        robot.send_action(act)


# Per-joint velocity / acceleration limits for smooth transit moves (deg/s, deg/s^2).
# The base carries the most inertia and causes the visible jump, so it gets the
# gentlest limits. Wrist joints can move faster.
_GOTO_VMAX = np.array([38.0, 55.0, 55.0, 75.0, 90.0])
_GOTO_AMAX = np.array([75.0, 110.0, 110.0, 150.0, 180.0])
_GOTO_DT = 0.02  # 50 Hz command rate


def goto_smooth(target, settle=0.15, step=2.0):
    """Transit move with a quintic S-curve profile: zero start/end velocity and
    acceleration. This removes the base jump at motion start/stop.

    The old send_joint_target_smoothly moved at a fixed step per tick, which is
    just a velocity cap — it still commanded abrupt starts and stops. Here the
    velocity ramps up and down smoothly, so the camera/gripper "head" glides.
    """
    joints, _r, obs = observe(overlay=False)
    gp = float(obs.get("gripper.pos", 50.0))
    q0 = np.asarray(joints, dtype=np.float64)
    q1 = np.asarray(target, dtype=np.float64)
    delta = q1 - q0

    # step=2.0 was the old default degrees/tick; use it as a speed scale.
    speed = float(step) / 2.0
    vmax = _GOTO_VMAX * speed
    amax = _GOTO_AMAX * speed

    # Quintic p(u) = 10u^3 - 15u^4 + 6u^5  ->  max vel 1.875/T, max acc 5.78/T^2
    abs_d = np.abs(delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        T_v = np.where(abs_d > 0.001, 1.875 * abs_d / np.maximum(vmax, 1e-6), 0.0)
        T_a = np.where(abs_d > 0.001, np.sqrt(5.78 * abs_d / np.maximum(amax, 1e-6)), 0.0)
    T = float(np.max(np.maximum(T_v, T_a)))
    T = max(T, 0.12)  # always at least 120 ms for tiny moves

    n = int(np.ceil(T / _GOTO_DT))
    t0 = time.time()
    for k in range(n + 1):
        t = min(k * _GOTO_DT, T)
        u = t / T
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        q_cmd = q0 + delta * s
        send_joints(q_cmd, gripper=gp)
        # sleep to maintain 50 Hz, accounting for command overhead
        to_sleep = t0 + (k + 1) * _GOTO_DT - time.time()
        if to_sleep > 0:
            time.sleep(to_sleep)

    time.sleep(settle)


def T_cam_of(joints):
    return np.asarray(kin.forward_kinematics(joints)) @ T_ee_cam


def read_depth_m(uv, win=7):
    """Median stereo depth (metres) in a small window around pixel uv, or None.
    The OAK-D returns uint16 millimetres aligned to the RGB frame."""
    try:
        with bus_lock:
            depth = cam.read_depth()          # (H,W) uint16 mm
    except Exception:
        return None
    if depth is None:
        return None
    h, w = depth.shape[:2]
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= u < w and 0 <= v < h):
        return None
    x0, x1 = max(0, u - win), min(w, u + win + 1)
    y0, y1 = max(0, v - win), min(h, v + win + 1)
    patch = depth[y0:y1, x0:x1].astype(np.float32)
    vals = patch[(patch > 80) & (patch < 2000)]   # 8cm–2m valid band
    if vals.size < 8:
        return None
    return float(np.median(vals)) / 1000.0


def locate_3d(uv, z_m, T_base_cam):
    """Back-project pixel uv at metric depth z_m through the camera intrinsics,
    then transform by the camera pose -> object point in the BASE frame.
    This is single-shot metric localization (no multi-vantage triangulation)."""
    x = (uv[0] - cx0) / fx * z_m
    y = (uv[1] - cy0) / fy * z_m
    p_cam = np.array([x, y, z_m, 1.0])
    return (T_base_cam @ p_cam)[:3]


def project_base(p_base, T_base_cam):
    """Base-frame point -> pixel. The inverse of locate_3d; the ground truth test
    for the hand-eye TF."""
    pc = np.linalg.inv(np.asarray(T_base_cam, np.float64)) @ np.append(
        np.asarray(p_base, np.float64), 1.0)
    if pc[2] <= 1e-4:
        return None                      # behind the camera
    return (float(fx * pc[0] / pc[2] + cx0), float(fy * pc[1] / pc[2] + cy0))


def tip_pixel(joints):
    """Where the hand-eye TF SAYS the fingertip appears in the FPV.

    The camera is bolted to the gripper, so the fingertip (= the ee frame origin,
    see GRIP_TIP_OFFSET_M) has exactly ONE pixel, the same in every pose. We also
    MEASURED that pixel directly, with /caltip: it is HAND_UV. So these two numbers
    are the same number computed two ways, and they MUST agree.

    They did not. The TF put the fingertip at ~(335, 71) -- the top of the frame --
    while the fingers really sit at HAND_UV=(440, 394), low and right. 330 px apart
    in a 640x480 image, and in the wrong HALF: no intrinsics fix that (you would need
    cy ~ 563 in a 480-tall frame). The TF's camera is pitched ~40 deg too far DOWN, so
    every sightline we back-project is TOO STEEP, so it hits the table TOO SOON, so
    every cube is reported NEARER than it is -- the user's "it should be further out",
    arrived at independently. calibrate_handeye() re-fits the TF to kill this.
    """
    T = np.asarray(kin.forward_kinematics(np.asarray(joints, np.float64)))
    return project_base(T[:3, 3], T @ T_ee_cam)


def locate_object(finder, tracker, label, tries=6):
    """Detect the object, read its stereo depth, and return its precise base-
    frame 3D coordinate — 'the G-code of the object'. Averages a few reads."""
    pts = []
    n_det = n_depth = 0
    for _ in range(tries):
        joints, rgb, _ = observe()
        T = T_cam_of(joints)
        tr = finder(rgb, T)
        if tr is None:
            time.sleep(0.05)
            continue
        n_det += 1
        z = read_depth_m(tr.uv)
        if z is None:
            time.sleep(0.05)
            continue
        n_depth += 1
        pts.append(locate_3d(tr.uv, z, T))
    if len(pts) < 3:
        say(f"{label}: locate failed — detections={n_det}/{tries} depth_ok={n_depth}")
        return None
    p = np.median(np.array(pts), axis=0)
    say(f"{label} located @ ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) m  "
        f"[stereo depth, {len(pts)} reads]")
    return p


def ik_to_point(p, q_seed):
    T = np.asarray(kin.forward_kinematics(q_seed)).copy()
    T[:3, 3] = p
    q = kin.inverse_kinematics(q_seed, T, position_weight=1.0, orientation_weight=0.0)
    err = float(np.linalg.norm(np.asarray(kin.forward_kinematics(q))[:3, 3] - p))
    return q, err


def gripper_current():
    try:
        with bus_lock:
            return float(robot.bus.read("Present_Current", "gripper", normalize=False))
    except Exception:
        return None


# ---------------- continuous servo core ----------------
def servo_loop(step_fn, done_fn, timeout_s=60.0, label="servo", stall_ok=None):
    """~14 Hz streaming loop. step_fn(joints, rgb, dt) -> v_cam (3,) desired EE
    velocity in CAMERA axes or None (hold). Joint deltas are IK'd + rate-clamped
    + velocity-EMA'd -> smooth, straight trajectories instead of waypoint hops.
    stall_ok(joints)->bool: if the arm physically stalls and this returns True,
    the stall is treated as ARRIVAL (fingers reached the object) and the loop
    returns success instead of retreating."""
    v_f = np.zeros(3)
    pan_f = 0.0   # smoothed shoulder-pan rate (deg/s) for horizontal centering
    pan_rate = 0.0  # acceleration-limited pan command that ramps up from zero
    t_end = time.time() + timeout_s
    n, t_hz = 0, time.time()
    q_cmd = None  # persistent commanded trajectory — integrates velocity so
    # per-tick deltas accumulate instead of being re-anchored to the measured
    # position (sub-deadband goals near the current position are ignored by
    # the Feetech servos, which stalled the arm dead).
    hist = []  # (t, joints) for physical-stall detection
    stall_cool = 0.0
    while time.time() < t_end:
        t0 = time.time()
        joints, rgb, _ = observe()
        if q_cmd is None:
            q_cmd = joints.copy()
        if done_fn(joints, rgb):
            return True
        now = time.time()
        hist.append((now, joints.copy()))
        while hist and now - hist[0][0] > 1.6:
            hist.pop(0)
        # physical stall: velocity commanded but the arm isn't moving
        # (fingertips on the table, joint limit). Retreat 12 mm to unjam,
        # re-sync the command to reality, and let the servo resume.
        if (now > stall_cool and now - hist[0][0] > 1.3
                and float(np.max(np.abs(joints - hist[0][1]))) < 0.25
                and float(np.linalg.norm(v_f)) > 0.004):
            # arm has stalled. If we're advancing onto an aligned object, this
            # stall means the fingers have reached it — grab, don't retreat.
            if stall_ok is not None and stall_ok(joints):
                say(f"{label}: stalled on target — fingers at object, grasping")
                return True
            off = q_cmd - joints
            say(f"{label}: blocked q={np.round(joints,1).tolist()} "
                f"cmd_off={np.round(off,1).tolist()} v={np.round(v_f*1e3,0).tolist()}mm/s — retreating")
            T_ee = np.asarray(kin.forward_kinematics(joints))
            p_up = T_ee[:3, 3] + np.array([0.0, 0.0, 0.012])
            q_up, _e = ik_to_point(p_up, joints)
            send_joints(q_up)
            q_cmd = np.asarray(q_up, dtype=np.float64).copy()
            v_f = np.zeros(3)
            pan_rate = 0.0
            hist.clear()
            stall_cool = now + 2.0
            time.sleep(0.18)
            continue
        v_des = step_fn(joints, rgb, LOOP_DT)
        pan_des = 0.0
        if v_des is None:
            v_f *= 0.6
            pan_f *= 0.6
        else:
            v_des = np.asarray(v_des, dtype=np.float64)
            pan_des = float(v_des[3]) if v_des.size >= 4 else 0.0
            v_f = (1 - ACC_ALPHA) * v_f + ACC_ALPHA * v_des[:3]
            pan_f = (1 - ACC_ALPHA) * pan_f + ACC_ALPHA * pan_des
        moved = False
        if np.linalg.norm(v_f) > 1e-5:
            T_ee = np.asarray(kin.forward_kinematics(q_cmd))
            d_base = (T_ee @ T_ee_cam)[:3, :3] @ (v_f * LOOP_DT)
            # deck mode: measured FK z=-0.022 is physical table contact (servo
            # sag included). Near it, forbid further down-motion but KEEP the
            # horizontal slide — that's how the cube gets between the fingers,
            # instead of the push-stall-retreat loop.
            z_meas = float(np.asarray(kin.forward_kinematics(joints))[2, 3])
            if z_meas < -0.012:
                d_base[2] = max(d_base[2], 0.0)
            T_goal = T_ee.copy()
            T_goal[:3, 3] = T_ee[:3, 3] + d_base
            # hard floor: never command the gripper FRAME below the base plane —
            # the black fingertips extend below it and reach the table first.
            T_goal[2, 3] = max(T_goal[2, 3], 0.0)
            q_t = kin.inverse_kinematics(q_cmd, T_goal, position_weight=1.0, orientation_weight=0.0)
            dq = np.clip(q_t - q_cmd, -JOINT_RATE_MAX * LOOP_DT, JOINT_RATE_MAX * LOOP_DT)
            q_cmd = q_cmd + dq
            moved = True
        # horizontal centering via BASE ROTATION (shoulder_pan) — singularity-
        # free, unlike sideways wrist translation which fails at extended reach.
        pan_target = pan_f
        if abs(pan_target) > 0.02:
            d_pan = float(np.clip(pan_target - pan_rate, -PAN_ACC_LIMIT, PAN_ACC_LIMIT))
            pan_rate += d_pan
            q_cmd[0] += float(np.clip(pan_rate, -25.0, 25.0)) * LOOP_DT
            moved = True
        else:
            pan_rate *= 0.85   # decay when idle — no hard stop
        if moved:
            # anti-windup: gentle pull instead of hard clamp — blends 70%
            # measured + 30% commanded so the arm never snaps.
            q_cmd = 0.7 * joints + 0.3 * q_cmd
            send_joints(q_cmd)
        n += 1
        if n % 20 == 0:
            with lock:
                state["loop_hz"] = round(20.0 / max(1e-3, time.time() - t_hz), 1)
            t_hz = time.time()
        time.sleep(max(0.0, LOOP_DT - (time.time() - t0)))
    raise Abort(f"{label}: timed out")


def center_smooth(finder, label, timeout_s=30.0):
    """Continuous pan/wrist centering (rotational, no translation)."""
    set_phase(f"CENTER {label}")
    t_end = time.time() + timeout_s
    misses = 0
    while time.time() < t_end:
        t0 = time.time()
        joints, rgb, _ = observe()
        tr = finder(rgb, T_cam_of(joints))
        if tr is None:
            misses += 1
            if misses > 12:
                raise Abort(f"{label}: lost while centering")
            time.sleep(LOOP_DT)
            continue
        misses = 0
        du, dv = tr.uv[0] - cx0, tr.uv[1] - cy0
        if abs(du) < 45 and abs(dv) < 55:
            say(f"{label} centered (du={du:+.0f} dv={dv:+.0f})")
            return
        q = joints.copy()
        q[0] += float(np.clip(+0.35 * math.degrees(math.atan2(du, fx)), -2.2, 2.2))
        q[3] += float(np.clip(+0.35 * math.degrees(math.atan2(dv, fy)), -2.2, 2.2))
        send_joints(q)
        time.sleep(max(0.0, LOOP_DT - (time.time() - t0)))
    raise Abort(f"{label}: centering timed out")


def search(finder, label):
    sweep = [VIEW + np.array([p, dl, de, dw, 0.0])
             for (dl, de, dw) in ((0, 0, 0), (-15, -10, 15), (-25, -15, 25))
             for p in (0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0)]
    for i, q in enumerate(sweep):
        set_phase(f"SEARCH {label}", f"sweep {i + 1}/{len(sweep)}")
        goto_smooth(q, settle=0.35)
        for _ in range(2):
            joints, rgb, _ = observe()
            tr = finder(rgb, T_cam_of(joints))
            if tr is not None:
                say(f"{label} found: uv=({tr.uv[0]:.0f},{tr.uv[1]:.0f}) area={tr.area_px}")
                return
            time.sleep(0.12)
    raise Abort(f"{label} cube not found in sweep")


def triangulate(finder, tracker, label):
    set_phase(f"TRIANGULATE {label}")
    rays = []
    base = observe()[0]
    for dq in (np.zeros(5), np.array([+8, 0, 0, +3, 0]), np.array([-8, -5, +4, +3, 0]),
               np.array([0, -8, +6, +4, 0]), np.array([+6, +5, -4, -3, 0])):
        if len(rays) >= 4:
            break
        goto_smooth(base + dq, settle=0.45)
        tr = None
        for _ in range(2):
            joints, rgb, _ = observe()
            tr = finder(rgb, T_cam_of(joints))
            if tr is not None and not tr.clipped:
                break
            time.sleep(0.15)
        if tr is None or tr.clipped:
            say(f"{label} vantage {dq}: unusable — skipped")
            continue
        T = T_cam_of(joints)
        d_cam = np.array([(tr.uv[0] - cx0) / fx, (tr.uv[1] - cy0) / fy, 1.0])
        d = T[:3, :3] @ (d_cam / np.linalg.norm(d_cam))
        rays.append((T[:3, 3], d))
    goto_smooth(base, settle=0.30)
    if len(rays) < 3:
        raise Abort(f"{label}: only {len(rays)} usable vantages")
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    p = np.linalg.solve(A, b)
    gaps = [float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (p - o))) for o, d in rays]
    say(f"{label} at ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) gaps mm {[round(g * 1e3, 1) for g in gaps]}")
    if max(gaps) > 0.05 or not (0.08 < np.hypot(p[0], p[1]) < 0.42) or not (-0.12 < p[2] < 0.15):
        raise Abort(f"{label}: triangulation implausible")
    tracker.p_anchor = p.copy()
    tracker.anchor_t = time.time()
    return p


def close_with_current(step=5.0, delay=0.05):
    """Close the gripper in small increments, watching the servo current, and
    stop the instant it rises (torque change = fingers on the object). Smaller
    step / longer delay = the slow, gentle close the user asked for."""
    idle = [c for c in (gripper_current() for _ in range(5)) if c is not None]
    i_idle = float(np.mean(idle)) if idle else 0.0
    pct = 95.0
    while pct > 2.0:
        checkpoint()
        pct -= step
        joints = observe(overlay=True)[0]
        send_joints(joints, gripper=pct)
        time.sleep(delay)
        c = gripper_current()
        if c is not None and abs(c - i_idle) >= 8.0:
            # firmer squeeze — ΔI=1.8 holds slipped the cube during transit
            send_joints(joints, gripper=max(0.0, pct - 14.0))
            time.sleep(0.18)
            return True, i_idle
    # Closed on air: DO NOT stay stalled shut (that's what tripped the servo's
    # overload protection earlier) — relax to a neutral opening.
    send_joints(observe(overlay=False)[0], gripper=40.0)
    time.sleep(0.18)
    return False, i_idle


def ee_move_rel(d_base, step=2.2, settle=0.22):
    joints = observe()[0]
    T = np.asarray(kin.forward_kinematics(joints))
    p = T[:3, 3] + np.asarray(d_base)
    # floor in MEASURED space: table contact is z=-0.022 (FK of real joints,
    # sag included). Clamping at 0.0 was commanding the hand 2 cm back UP
    # right before the close — that's why the gripper kept missing.
    p[2] = max(p[2], -0.020)
    q, err = ik_to_point(p, joints)
    if err > 0.02:
        raise Abort(f"relative move unreachable (err {err * 1e3:.0f} mm)")
    goto_smooth(q, settle=settle, step=step)


# ---------------- polar bearing-lock localize + approach ----------------
# The table plane the sightline is intersected with. Starts at the Z_TABLE guess
# and is MEASURED at every LOCATE (see locate_on_table) — do not trust the guess.
TABLE_Z = [Z_TABLE]

# RADIAL LOCALIZATION CORRECTION (metres). The camera sits ~10 cm BEHIND the
# fingertips, so the back-projected sightline reads every object TOO CLOSE to the
# base (a too-steep ray hits the table too soon). The user measured the miss at a
# consistent ~10 cm too near, and the arm dived straight down at a point basically
# under itself instead of reaching out to the cube. So AFTER localizing, push the
# point radially OUTWARD (away from the base) by this much. Live-tunable via /pushout
# (a number box in the UI) so it can be dialled against the 3D view without a restart.
# DEFAULT 0: once the hand-eye TF is CALIBRATED, the 10 cm camera-behind-gripper
# offset lives in the TF translation, so this fudge double-counts and pushes the
# cube OUT OF REACH (seen live: raw r=34.6cm + 10cm = 44.6cm -> "cannot reach").
PUSH_OUT = [0.0]


def push_out_radial(p):
    """Move a base-frame point radially OUTWARD (away from base z-axis) by PUSH_OUT[0].
    Applied to EVERY localization (initial lock AND every approach refine) so the
    correction is consistent — otherwise a raw re-measure would drag the target back
    inward and undo the push. See PUSH_OUT."""
    p = np.asarray(p, np.float64).copy()
    r = float(np.hypot(p[0], p[1]))
    push = float(PUSH_OUT[0])
    if r > 1e-3 and push != 0.0:
        p[0] += p[0] / r * push
        p[1] += p[1] / r * push
    return p


def ray_to_table(uv, T_base_cam, z_plane=None):
    """Intersect the pixel's back-projected sightline with the table plane. The
    plane height is TABLE_Z[0], which locate_on_table MEASURES from the bbox
    pinhole range rather than assuming."""
    o = T_base_cam[:3, 3]
    z = TABLE_Z[0] if z_plane is None else float(z_plane)
    d_cam = np.array([(uv[0] - cx0) / fx, (uv[1] - cy0) / fy, 1.0])
    d = T_base_cam[:3, :3] @ (d_cam / np.linalg.norm(d_cam))
    if abs(d[2]) < 1e-6:
        return None
    t = (z - o[2]) / d[2]
    if t <= 0:
        return None
    return o + t * d


TABLE_Z0 = 0.0     # the table IS the robot's own base plane (the user's premise:
                   # "everything is on the same table and height as the base")


# ================= 2D BEV object map (top-down, base X-Y plane) =================
# A simple bird's-eye map of the table. Each detection's sightline is cast onto
# the table plane (inverse perspective mapping) through the CAD-calibrated camera
# -> object (x, y). One running-mean entry per object; click a dot in the UI to
# fly the gripper on top. Each entry also carries the STEREO range next to the
# IPM range, so the two can be compared (if they disagree, the hand-eye distance
# is off; if they agree, the object really is that far).
W2D_MERGE = 0.14          # detections of the same label within this = same object.
                          # Tightened because objects were being merged too
                          # aggressively in the compressed map. Increase if you get
                          # duplicate ghosts for one cube.
W2D = {"objs": {}, "next": 1}
w2d_lock = threading.Lock()


def _rotate_xy(xy, deg):
    """Rotate a base-frame (x, y) point about the origin by deg degrees."""
    th = math.radians(float(deg))
    c, s = math.cos(th), math.sin(th)
    return np.array([xy[0] * c - xy[1] * s, xy[0] * s + xy[1] * c], dtype=np.float64)


def obj_xy_2d(bbox, T_cam, z_m=None):
    """Where the cube is, base-frame (x, y).

    RANGE COMES FROM STEREO DEPTH when it is available and sane; otherwise it
    falls back to apparent size:

        range = focal * real_cube_edge / bbox_width_px

    The apparent-size method needs only the lens and the cube's real size, but
    it is sensitive to CUBE_EDGE_M being wrong. Stereo depth is independent of
    the object's size, so it anchors the range and kills the random jumps.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 4:
        return None, float("nan"), CUBE_EDGE_M

    # Prefer stereo depth if the caller passed a valid metric range.
    if z_m is not None and 0.03 < z_m < 1.20:
        rng = float(z_m)
    else:
        rng = float(fx * CUBE_EDGE_M / float(w))
        if not (0.05 < rng < 1.20):
            return None, float("nan"), CUBE_EDGE_M

    rng *= float(RANGE_SCALE[0])
    rng = float(np.clip(rng, 0.03, 1.50))

    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0
    d = np.array([(u - cx0) / fx, (v - cy0) / fy, 1.0], dtype=np.float64)
    d /= np.linalg.norm(d)
    p = T_cam[:3, 3] + T_cam[:3, :3] @ (d * rng)
    xy = p[:2]
    if not (0.05 < float(np.hypot(*xy)) < 0.95):
        return None, float("nan"), CUBE_EDGE_M
    # Correct heading/yaw error in the hand-eye by rotating the bearing.
    xy = _rotate_xy(xy, MAP_BEARING_OFFSET_DEG[0])
    return xy, float(rng), CUBE_EDGE_M


def _consolidate_2d():
    """Merge same-label objects whose centres are within W2D_MERGE — collapses the
    duplicate 'ghost' tags that a jittery localization spawns for ONE physical cube
    (the overlapping-predictions the user saw). Weighted by observation count.
    Caller holds w2d_lock."""
    objs = W2D["objs"]
    changed = True
    while changed:
        changed = False
        items = list(objs.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                ta, a = items[i]
                tb, b = items[j]
                if ta not in objs or tb not in objs or a["label"] != b["label"]:
                    continue
                if float(np.hypot(*(a["xy"] - b["xy"]))) < W2D_MERGE:
                    keep, drop = (ta, tb) if a["n"] >= b["n"] else (tb, ta)
                    ko, do = objs[keep], objs[drop]
                    wsum = ko["n"] + do["n"]
                    ko["xy"] = (ko["n"] * ko["xy"] + do["n"] * do["xy"]) / wsum
                    ko["size"] = max(ko.get("size", 0.03), do.get("size", 0.03))
                    ko["n"] = wsum
                    ko["t"] = max(ko["t"], do["t"])
                    del objs[drop]
                    changed = True
                    break
            if changed:
                break


def world2d_update(label, xy, stereo, size=0.03):
    with w2d_lock:
        best, bd = None, W2D_MERGE
        for t, o in W2D["objs"].items():
            if o["label"] == label and float(np.hypot(*(o["xy"] - xy))) < bd:
                best, bd = t, float(np.hypot(*(o["xy"] - xy)))
        if best is None:
            t = W2D["next"]; W2D["next"] += 1
            W2D["objs"][t] = {"label": label, "xy": np.asarray(xy, float),
                              "size": float(size), "n": 1, "stereo": stereo, "t": time.time()}
        else:
            o = W2D["objs"][best]
            # Give fresh observations more weight so the map converges faster and
            # does not stay stuck on an early bad localization.
            o["xy"] = 0.55 * o["xy"] + 0.45 * np.asarray(xy, float)
            o["size"] = 0.8 * o.get("size", size) + 0.2 * float(size)
            o["n"] += 1
            o["stereo"] = stereo
            o["t"] = time.time()
        _consolidate_2d()


def sense_2d(joints=None, rgb=None):
    if joints is None:
        joints, rgb, _ = observe()
    T = T_cam_of(joints)
    for finder, label in ((find_red, "red cube"), (find_green, "green cube")):
        tr = finder(rgb, T)
        if tr is not None and not tr.clipped:
            # Average a few stereo depth reads for a cleaner range estimate.
            zs = [z for z in (read_depth_m(tr.uv) for _ in range(3)) if z is not None]
            z = float(np.median(zs)) if zs else None
            xy, st, sz = obj_xy_2d(tr.bbox_xyxy, T, z_m=z)
            if xy is not None:
                world2d_update(label, xy, st, sz)


def world2d_snapshot():
    now = time.time()
    with w2d_lock:
        return [{"tag": t, "label": o["label"],
                 "x": round(float(o["xy"][0]), 3), "y": round(float(o["xy"][1]), 3),
                 "size": round(float(o.get("size", 0.03)), 3),
                 "r_cm": round(float(np.hypot(*o["xy"])) * 100, 1),
                 "ang": round(math.degrees(math.atan2(o["xy"][1], o["xy"][0]))),
                 "stereo_cm": (round(o["stereo"] * 100) if o["stereo"] == o["stereo"] else None),
                 "n": o["n"], "age": round(now - o["t"], 1)}
                for t, o in W2D["objs"].items()]


def scan_2d():
    """Pan the base smoothly across the front arc, sensing into the 2D map."""
    set_phase("SCAN2D", "sweeping the table into the 2D map")
    start = float(observe()[0][0])
    for target in (start - 45.0, start + 45.0, start):
        target = float(np.clip(target, -110.0, 110.0))
        cur = float(observe()[0][0])
        n = 0
        while abs(cur - target) > 1.0:
            checkpoint()
            cur += float(np.clip(target - cur, -0.6, 0.6))     # smooth fine step
            q = observe(overlay=True)[0].astype(np.float64)
            q[0] = cur
            send_joints(q)
            if n % 4 == 0:
                sense_2d()
            n += 1
            time.sleep(0.033)
    set_phase("IDLE", f"2D map: {len(world2d_snapshot())} object(s)")


def goto_2d(tag):
    """Fly the gripper on top of a mapped object (hover ~6 cm above its (x,y))."""
    with w2d_lock:
        o = W2D["objs"].get(tag)
    if o is None:
        raise Abort(f"tag {tag} not in the 2D map")
    xy, label = o["xy"], o["label"]
    set_phase("GOTO2D", f"{label} #{tag} @ ({xy[0]*100:.0f},{xy[1]*100:.0f})cm")
    j = observe()[0].astype(np.float64)
    pitch, e = plan_grasp_pitch(np.array([xy[0], xy[1], 0.02]), j)
    if pitch is None:
        raise Abort(f"{label} #{tag}: out of reach (best IK {e*1e3:.0f}mm, "
                    f"r={np.hypot(*xy)*100:.0f}cm)")
    q, err = _ik_hold_pitch(observe()[0], np.array([xy[0], xy[1], 0.06]), pitch,
                            float(j[4]), ret_err=True)
    if err > 0.008:
        raise Abort(f"hover pose unreachable (IK {err*1e3:.0f}mm)")
    goto_smooth(q, settle=0.25)
    with lock:
        state["obj3d"] = [float(xy[0]), float(xy[1]), 0.02]
        state["obj3d_label"] = label.split()[0]
    set_phase("GOTO2D", f"on top of {label} #{tag}")


def solve_on_table(tr, T_base_cam):
    """Range to the cube WITHOUT assuming its size or guessing a plane height.

    Two facts we actually know:
      (a) the cube sits ON THE TABLE, i.e. its centre is at TABLE_Z0 + S/2,
      (b) apparent size gives range: S = d * w_px / fx   (pinhole).
    Substituting (b) into (a) along the sightline p(d) = o + d*dir leaves ONE
    unknown:
          o_z + d*dir_z = TABLE_Z0 + d*w/(2*fx)
      =>  d = (o_z - TABLE_Z0) / ( w/(2*fx) - dir_z )
    dir_z < 0 (the camera looks down), so the denominator is positive and d is
    well defined. Returns (point_base, implied_cube_edge_m).

    Why this replaces the old code: range used to be `fx * TARGET_SIZE_M / w`
    with TARGET_SIZE_M HARD-CODED to 0.03. If the real cube is bigger, every
    range comes out SHORT, and because the sightline points down-forward a short
    range lands the cube too NEAR and too HIGH — the last run reported the cube
    centre at z=+4.0cm, which is impossible for a cube resting on the table. That
    is exactly the "it should be further out" error. Here the size falls out of
    the solve instead of being assumed, so it is self-correcting.
    """
    x1, y1, x2, y2 = tr.bbox_xyxy
    w = float(max(4.0, x2 - x1))          # horizontal extent ~ the cube edge
    o = T_base_cam[:3, 3]
    d_cam = np.array([(tr.uv[0] - cx0) / fx, (tr.uv[1] - cy0) / fy, 1.0])
    dirv = T_base_cam[:3, :3] @ (d_cam / np.linalg.norm(d_cam))
    den = w / (2.0 * fx) - float(dirv[2])
    if den <= 1e-6:
        return None, None
    d = (float(o[2]) - TABLE_Z0) / den
    if not (0.03 < d < 0.80):
        return None, None
    p = o + d * dirv
    S = d * w / fx                        # the cube edge this implies
    return p, float(S)


def locate_on_table(finder, tracker, label):
    """Locate the cube by solving range + size together against the table plane.
    See solve_on_table. Median over several reads; the implied cube edge is logged
    as a sanity check (a sane red cube is ~3-5 cm — if this prints something wild,
    the hand-eye TF is the suspect, not the detector)."""
    set_phase(f"LOCATE {label}", "solving range + size against the table")
    pts, sizes = [], []
    for _ in range(9):
        checkpoint()
        joints, rgb, _ = observe()
        T = T_cam_of(joints)
        tr = finder(rgb, T)
        if tr is None:
            time.sleep(0.05)
            continue
        p, S = solve_on_table(tr, T)
        if p is not None and 0.06 < float(np.hypot(p[0], p[1])) < 0.45:
            pts.append(p)
            sizes.append(S)
        time.sleep(0.03)
    if len(pts) < 3:
        raise Abort(f"{label}: could not localize ({len(pts)} good reads)")
    p = np.median(np.array(pts), axis=0)
    S = float(np.median(sizes))
    TABLE_Z[0] = TABLE_Z0 + 0.5 * S       # keep the hop refiner on the same plane

    r_raw = float(np.hypot(p[0], p[1]))
    p = push_out_radial(p)                  # camera-behind-gripper correction (see PUSH_OUT)
    r_final = float(np.hypot(p[0], p[1]))
    say(f"{label} located: r={r_final*100:.1f}cm "
        f"ang={math.degrees(math.atan2(p[1], p[0])):+.0f}deg z={p[2]*100:.1f}cm "
        f"| raw r={r_raw*100:.1f}cm + pushed out {PUSH_OUT[0]*100:.0f}cm "
        f"| cube edge solved={S*100:.1f}cm")
    tracker.p_anchor = p.copy()
    tracker.anchor_t = time.time()
    return p


# ---------------- hand-eye self-calibration ----------------
TF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handeye_tf.json")


def load_tf_override():
    """A TF we FITTED beats the TF we were handed. Written by calibrate_handeye()."""
    global T_ee_cam
    try:
        with open(TF_FILE) as f:
            d = json.load(f)
        T_ee_cam = parse_tf_string(d["tf"])
        return d
    except FileNotFoundError:
        return None
    except Exception as e:
        say(f"hand-eye: ignoring bad {os.path.basename(TF_FILE)} ({e})")
        return None


def calibrate_mount_multiview(finder, label="red", n_pan=5):
    """Pin the camera mount by MULTI-VIEW CONSISTENCY.

    A stationary cube on the table must map to the SAME (x, y) no matter which pose
    the arm views it from. That is the honest objective, and it pins the mount
    orientation properly.

    Why the previous attempts failed:
      * Fitting the cube's blob reprojection has a ~55-80px noise floor, so the fit
        wandered and the acceptance gate threw away good solutions.
      * Constraining only the FINGERTIP pixel gives 2 equations for 3 rotation
        angles -- roll about the tip ray is unobservable, so a single-pose fix held
        near that pose and drifted everywhere else (map went from r=34cm to r=53cm).
    Here every extra viewpoint adds constraints, and the fingertip term is kept as
    an anchor so the solution cannot slide off into a mirrored/degenerate pose.
    """
    global T_ee_cam
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    set_phase("CALIB", "multi-view mount calibration: sampling the cube")
    q0 = observe()[0].astype(np.float64)
    samples = []
    for dpan in np.linspace(-16.0, 16.0, n_pan):
        for dlift in (0.0, -7.0):
            checkpoint()
            q = q0.copy()
            q[0] = float(np.clip(q[0] + dpan, -100, 100))
            q[1] = float(np.clip(q[1] + dlift, -95, 95))
            try:
                goto_smooth(q, settle=0.18)
            except Exception:
                continue
            j, rgb, _ = observe()
            tr = finder(rgb, T_cam_of(j))
            if tr is None or tr.clipped:
                continue
            T_ee = np.asarray(kin.forward_kinematics(np.asarray(j, np.float64)))
            samples.append((T_ee, np.array(tr.uv, np.float64)))
    goto_smooth(q0, settle=0.25)
    say(f"multi-view: {len(samples)} usable views of the {label} cube")
    if len(samples) < 5:
        raise Abort(f"only {len(samples)} views - need at least 5. "
                    f"Keep the cube visible while the arm pans.")

    t0 = T_ee_cam[:3, 3].copy()

    def unpack(x):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
        T[:3, 3] = x[3:6]
        return T

    def table_pts(T_cam_ee):
        pts = []
        for T_ee, uv in samples:
            T = T_ee @ T_cam_ee
            o = T[:3, 3]
            d = T[:3, :3] @ np.array([(uv[0]-cx0)/fx, (uv[1]-cy0)/fy, 1.0])
            if d[2] >= -1e-3:
                return None
            t = (TABLE_Z0 - o[2]) / d[2]
            if not (0.02 < t < 2.0):
                return None
            pts.append((o + t*d)[:2])
        return np.array(pts) if pts else None

    def resid(x):
        T = unpack(x)
        pts = table_pts(T)
        if pts is None:
            return np.full(2*len(samples) + 2, 10.0)
        spread = (pts - pts.mean(axis=0)).ravel() * 40.0      # metres -> weighted
        # fingertip anchor: it must still reproject to its measured pixel
        T_ee, _ = samples[0]
        tip = T_ee[:3, 3] + T_ee[:3, :3] @ np.array([0, 0, GRIP_TIP_OFFSET_M])
        Tbc = T_ee @ T
        pc = np.linalg.inv(Tbc) @ np.append(tip, 1.0)
        if pc[2] <= 1e-6:
            anchor = np.array([10.0, 10.0])
        else:
            u = fx*pc[0]/pc[2] + cx0
            v = fy*pc[1]/pc[2] + cy0
            anchor = np.array([u - HAND_UV[0], v - HAND_UV[1]]) * 0.02
        return np.concatenate([spread, anchor])

    x0 = np.concatenate([Rotation.from_matrix(T_ee_cam[:3, :3]).as_rotvec(), t0])
    lo = np.concatenate([x0[:3] - 1.2, t0 - 0.06])
    hi = np.concatenate([x0[:3] + 1.2, t0 + 0.06])
    sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac",
                        max_nfev=3000, ftol=1e-10, xtol=1e-10)

    def spread_cm(x):
        pts = table_pts(unpack(x))
        if pts is None:
            return 999.0
        return float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean() * 100)

    before, after = spread_cm(x0), spread_cm(sol.x)
    say(f"multi-view spread: {before:.1f}cm -> {after:.1f}cm "
        f"(how much the same cube moves between viewpoints)")
    if after > before or after > 4.0:
        raise Abort(f"multi-view calibration did not converge "
                    f"(spread {after:.1f}cm) - mount NOT changed")
    T_new = unpack(sol.x)
    rv = Rotation.from_matrix(T_new[:3, :3]).as_rotvec()
    tf_str = ",".join(f"{v:.4f}" for v in list(T_new[:3, 3]) + list(rv))
    T_ee_cam = T_new
    with open(TF_FILE, "w") as f:
        json.dump({"tf": tf_str, "rms_px": 0, "tip_px": 0,
                   "spread_cm": after, "views": len(samples),
                   "source": "multi-view consistency", "fitted": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, indent=2)
    say(f"mount CALIBRATED (multi-view) -> {tf_str}")
    set_phase("IDLE", f"mount calibrated - same cube now agrees to {after:.1f}cm across views")



def calibrate_handeye(finder, n_target=14):
    """Fit the gripper->camera transform FROM THE ROBOT'S OWN MOTION. No chessboard,
    no tape measure, ~30 s.

    WHY THIS EXISTS. The shipped TF is wrong by ~40 deg in camera pitch, and that one
    error produced most of the symptoms chased for two days. Proof, no fitting needed:
    the camera is bolted to the gripper, so the fingertip (the ee origin) projects to
    ONE fixed pixel in every pose. The TF puts it at ~(335, 71). We MEASURED it with
    /caltip at HAND_UV=(440, 394). 330 px apart, and in the wrong HALF of the frame --
    no intrinsics can reconcile that. A camera that thinks it is pitched further down
    than it is back-projects every sightline TOO STEEPLY, so the ray hits the table TOO
    SOON, so every object is reported NEARER than it is. That is exactly the standing
    complaint that the cube "should be further out", derived from the other end.

    TWO INDEPENDENT FACTS PIN THE TF DOWN:
      (A) THE FINGERTIP IS IN THE PICTURE.  project(FK_tip) must equal HAND_UV.
          2 equations. Nails the camera's POINTING DIRECTION -- the broken part.
      (B) A STATIC CUBE LOOKS THE SAME FROM EVERYWHERE.  Park the arm in N poses that
          all keep the cube in view; every sightline must pass through ONE point, and
          that point sits on the table. 2 equations per pose. Nails the camera's
          POSITION on the gripper, which (A) alone cannot see.

    Unknowns: TF translation (3) + TF rotvec (3) + the cube (3, z bounded to the table).
    Residuals: 2 + 2N. Seeded from the current TF, so a good TF stays put.
    """
    global T_ee_cam
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    set_phase("CALIB", "hand-eye: sampling the cube from several poses")
    q0 = observe()[0].astype(np.float64)
    tr0 = detect_now(finder, tries=25)
    if tr0 is None:
        raise Abort("hand-eye: no cube in view — put the cube in the gripper view first")

    # Poses that keep the cube in frame but move the camera in genuinely different
    # ways: pan rotates the camera about base z, wrist_flex rotates it about the
    # pitch axis, lift/elbow TRANSLATE it. Rotation observes the TF's rotation;
    # translation observes the TF's translation. We need both or the fit is degenerate.
    deltas = []
    for dpan in (-9.0, -4.5, 0.0, 4.5, 9.0):
        deltas.append(np.array([dpan, 0.0, 0.0, 0.0, 0.0]))
    for dwf in (-9.0, -4.0, 4.0, 9.0):
        deltas.append(np.array([0.0, 0.0, 0.0, dwf, 0.0]))
    for dl, de in ((-6.0, 6.0), (6.0, -6.0), (-4.0, 10.0), (4.0, -10.0), (-8.0, 4.0)):
        deltas.append(np.array([0.0, dl, de, 0.0, 0.0]))

    samples = []          # (joints, cube_uv)
    for d in deltas:
        checkpoint()
        q = q0 + d
        goto_smooth(q, settle=0.25, step=1.5)
        j = observe()[0].astype(np.float64)
        tr = detect_now(finder, tries=6)
        if tr is None:
            continue
        samples.append((j, np.array(tr.uv, np.float64)))
        if len(samples) >= n_target:
            break
    goto_smooth(q0, settle=0.3, step=1.5)

    if len(samples) < 6:
        raise Abort(f"hand-eye: only {len(samples)} usable views (need 6) — "
                    "keep the cube in the gripper view for the whole sweep")

    T_ee = [np.asarray(kin.forward_kinematics(j)) for j, _ in samples]
    uvs = [uv for _, uv in samples]
    # The fingertip anchor is pose-independent (the camera is rigid to the ee), so it
    # is ONE constraint however many poses we took. Weight it like sqrt(N) samples so
    # it is not drowned out by the noisier cube pixels.
    w_tip = math.sqrt(len(samples))

    def unpack(x):
        tf = np.eye(4)
        tf[:3, 3] = x[:3]
        tf[:3, :3] = Rotation.from_rotvec(x[3:6]).as_matrix()
        return tf, np.array(x[6:9])

    def resid(x):
        tf, p = unpack(x)
        r = []
        for T, uv in zip(T_ee, uvs):
            pu = project_base(p, T @ tf)
            r += [400.0, 400.0] if pu is None else [pu[0] - uv[0], pu[1] - uv[1]]
        pt = project_base(T_ee[0][:3, 3], T_ee[0] @ tf)      # the fingertip anchor
        r += ([400.0, 400.0] if pt is None else
              [w_tip * (pt[0] - HAND_UV[0]), w_tip * (pt[1] - HAND_UV[1])])
        return r

    p_seed = _measure_point(tr0, T_ee[0] @ T_ee_cam)
    if p_seed is None:
        p_seed = np.array([0.18, 0.0, 0.02])

    # SEED THE TRANSLATION AT THE MEASURED MOUNT DISTANCE, not at the old TF's value.
    # The old TF puts the camera 20.2 cm from the fingertip; the mount was measured at
    # ~10 cm. Starting a nonlinear fit 2x off in translation invites a bad local minimum,
    # so keep the old direction (the mount geometry is roughly right) and rescale it, and
    # bound |t| to something a camera bolted to this gripper can physically be.
    t_old = np.asarray(T_ee_cam[:3, 3], np.float64)
    t_seed = t_old / max(1e-6, np.linalg.norm(t_old)) * CAM_TIP_M
    x0 = np.concatenate([t_seed,
                         Rotation.from_matrix(T_ee_cam[:3, :3]).as_rotvec(),
                         np.asarray(p_seed, np.float64)])
    lo = np.array([-0.16, -0.16, -0.16, -4.0, -4.0, -4.0, -0.45, -0.45, 0.005])
    hi = np.array([0.16, 0.16, 0.16, 4.0, 4.0, 4.0, 0.45, 0.45, 0.050])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)

    def rms(x):
        r = np.array(resid(x))[: 2 * len(samples)]
        return float(np.sqrt((r ** 2).reshape(-1, 2).sum(1).mean()))

    def tipgap(x):
        tf, _ = unpack(x)
        pu = project_base(T_ee[0][:3, 3], T_ee[0] @ tf)
        return 999.0 if pu is None else math.hypot(pu[0] - HAND_UV[0], pu[1] - HAND_UV[1])

    say(f"hand-eye BEFORE: cube reprojection RMS={rms(x0):.0f}px  "
        f"fingertip off by {tipgap(x0):.0f}px  ({len(samples)} views)")

    sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac",
                        max_nfev=4000, ftol=1e-10, xtol=1e-10)
    tf, p = unpack(sol.x)
    say(f"hand-eye AFTER:  cube reprojection RMS={rms(sol.x):.0f}px  "
        f"fingertip off by {tipgap(sol.x):.0f}px")

    # Gate on the FINGERTIP gap — a HARD geometric constraint (the camera is bolted
    # to the gripper, so FK's fingertip must reproject to the measured HAND_UV), which
    # locks to ~0px on a good fit. Do NOT gate tightly on the cube RMS: a colour-blob
    # centroid has a ~50-80px noise floor that MORE poses do not lower, so a 25px bar
    # rejected a GOOD fit (tip 0px, rms 49px) and kept the broken CAD TF. Accept when
    # the fingertip nails it and the cube RMS is merely sane.
    if tipgap(sol.x) > 12.0 or rms(sol.x) > 100.0:
        raise Abort(f"hand-eye: fit did not converge (RMS {rms(sol.x):.0f}px, "
                    f"tip {tipgap(sol.x):.0f}px) — TF NOT changed. More pose spread needed.")

    rv = Rotation.from_matrix(tf[:3, :3]).as_rotvec()
    tf_str = ",".join(f"{v:.4f}" for v in list(tf[:3, 3]) + list(rv))
    with open(TF_FILE, "w") as f:
        json.dump({"tf": tf_str, "rms_px": rms(sol.x), "tip_px": tipgap(sol.x),
                   "views": len(samples), "fitted": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    T_ee_cam = tf
    say(f"hand-eye CALIBRATED -> {tf_str}")
    say(f"  (saved to {os.path.basename(TF_FILE)}; loaded automatically on every restart)")
    say(f"  cube now solves to r={np.hypot(p[0], p[1])*100:.1f}cm "
        f"ang={math.degrees(math.atan2(p[1], p[0])):+.0f}deg z={p[2]*100:.1f}cm")
    set_phase("CALIB", f"hand-eye fixed — reprojection {rms(sol.x):.0f}px")
    return tf


def bbox_range_m(tr):
    """Camera→object range from the bbox apparent size (pinhole, known cube edge).
    The only depth that survives inside the OAK-D stereo minimum range."""
    x1, y1, x2, y2 = tr.bbox_xyxy
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    if w >= h:
        d = fx * TARGET_SIZE_M / w
    else:
        d = fy * TARGET_SIZE_M / h
    return float(np.clip(d, 0.02, 0.60))


def _measure_point(tr, T_cam):
    """Best single-frame base-frame coordinate for a detection: the TABLE-RAY
    (sightline ∩ table plane). NOT stereo depth — stereo gave garbage here
    (z ≈ -1 m), and a bad far range washes out the lateral bearing so the target
    lands straight ahead instead of where the cube is. Table-ray pins z to the
    table, so the bearing is always right. Bbox pinhole is the last-ditch."""
    p_tab = ray_to_table(tr.uv, T_cam)
    if p_tab is not None and 0.05 < float(np.hypot(p_tab[0], p_tab[1])) < 0.45:
        return p_tab
    return locate_3d(tr.uv, bbox_range_m(tr), T_cam)


def tip_base(T_ee):
    """Fingertip position in base frame: the ee frame origin pushed out along the
    hand (+Z of gripper_frame_link). Mirrors lookat_engine._gripper_tip_base on
    the identical urdf + ee_frame. `T_ee[:3,3]` is the WRIST, not the fingers."""
    T = np.asarray(T_ee, np.float64)
    return T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, GRIP_TIP_OFFSET_M])


# ---- FIRST-PERSON APPROACH ------------------------------------------------
# The camera is the head of the snake: it rides on the gripper, so every move
# changes the view, and the closer we get the WORSE the detector behaves (it
# hallucinates, then stops recognising the cube at all once the cube fills the
# frame and slides under the fingers). So we do NOT servo on pixels all the way
# in. We look from a distance where the detector is honest, LOCK the cube's 3D
# point, and run the last leg on the lock -- the cube is static, so a remembered
# coordinate beats a close-range guess.
STANDOFF_H = 0.05        # m, fingertip parks this far ABOVE the cube, then descends
COMMIT_M = 0.10          # m, closer than this we STOP believing the detector
HOP_M = 0.030            # m, max Cartesian step per hop
# Candidate grasp pitches, tried in this order. Constrained by the REAL joint limits,
# which is what the last attempt got wrong. Measured achievable pitch at z=2cm:
#     r=10cm -> 85..95 only     r=15cm -> 65..95     r=20cm -> 45..95
# So a close cube can ONLY be grasped nearly straight down; there is no choice about
# it. We try steep-first and let the (now limit-aware) IK veto what it cannot hold.
# Steep first (best grip on a table cube), then progressively SHALLOWER as
# fallbacks. Reach depends strongly on this angle — measured from the URDF:
#   90deg -> 30.8cm | 70deg -> 36.4cm | 60deg -> 39.8cm
#   50deg -> 42.7cm | 40deg -> 45.2cm | 30deg -> 46.2cm
#   20deg -> 47.0cm | 10deg -> 47.5cm |  0deg -> 47.8cm
# Stopping the list at 55deg capped the arm at ~41cm, so anything further out was
# declared unreachable and the approach died short. The shallow entries let the
# arm actually GET THERE; the loop still returns the steepest angle that solves,
# so near cubes are unaffected. Real reach depends on the target height and pose;
# the arm can exceed 52 cm in favourable configurations, so the edge tolerance is
# relaxed rather than hard-capping the range.
GRASP_PITCH = (75.0, 80.0, 70.0, 85.0, 90.0, 65.0, 60.0, 55.0,
               50.0, 45.0, 40.0, 35.0, 30.0, 25.0, 20.0, 15.0,
               10.0, 5.0, 0.0)


def plan_grasp_pitch(p_obj, q_seed):
    """Choose the gripper pitch to grasp with, and PROVE the arm can get there.
    Returns (pitch_deg, worst_ik_residual_m), or (None, best_residual) if nothing works.

    THE GEOMETRY THAT WAS BEING IGNORED (measured 2026-07-13). The fingertip is 98 mm
    out in FRONT of the wrist. So putting the fingertip on a cube at r=15cm, z=2cm with
    the hand HORIZONTAL demands the WRIST sit at r=52mm, z=20mm -- inside the robot's
    own base column. Physically impossible:

        hand elevation  ->  where the WRIST must then be
              0 deg           r= 52 mm  z= 20 mm   <- inside the base. impossible.
            -40 deg           r= 75 mm  z= 83 mm
            -60 deg           r=101 mm  z=105 mm
            -90 deg           r=150 mm  z=118 mm   <- comfortable

    The old code asked for the impossible one; the old solver had no way to say "no";
    its half-solved answer crept the arm out and up. That IS the "it just grabs at air"
    run. You grab a cube off a table from ABOVE -- point the hand down.

    Steeper is kinematically safer too: pitch 40-60 at r=10-15cm is a genuine elbow-flip
    DEAD BAND where no seed converges (mapped 2026-07-13). The old code clamped pitch to
    [-10, 55] -- it aimed straight into the dead band. At 70-90 the whole 10-28cm working
    range solves to under 0.3 mm.
    """
    p_above = np.array([p_obj[0], p_obj[1], p_obj[2] + STANDOFF_H])
    best = None
    for pitch in GRASP_PITCH:
        _, e_hi = _ik_hold_pitch(q_seed, p_above, pitch, float(q_seed[4]), ret_err=True)
        _, e_lo = _ik_hold_pitch(q_seed, p_obj, pitch, float(q_seed[4]), ret_err=True)
        worst = max(float(e_hi), float(e_lo))
        # At the workspace edge (shallow pitch / far reach) the IK residual can be a
        # few mm larger and still be a valid pose. Use a sliding tolerance so we do
        # not throw away the arm's real reach; the visual centering pass then fine-tunes.
        tol = 0.025 if pitch <= 15.0 else 0.004
        if worst <= tol:
            return pitch, worst
        if best is None or worst < best[1]:
            best = (pitch, worst)
    # Last resort: if nothing solved tightly but the best residual is still usable,
    # return it rather than declaring the cube unreachable at the edge of reach.
    if best is not None and best[1] <= 0.030:
        return best
    return None, best[1]


def approach_over_the_top(finder, tracker, label):
    """LOCK the cube, then fly the fingertip to a waypoint directly ABOVE it.
    Returns (p_obj, grasp_pitch).

    Every Cartesian target is checked for reachability BEFORE the arm is told to move.
    An unreachable target is a fact to REPORT, not a pose to drive to -- driving to
    half-solved IK poses is exactly what made the arm wander outward for 30 hops while
    reporting it was a steady 93 mm from the cube.
    """
    set_phase(f"APPROACH {label}", "locking the cube, then over the top of it")
    send_joints(observe()[0], gripper=95.0)
    time.sleep(0.15)

    p_obj = np.asarray(locate_on_table(finder, tracker, label), np.float64)
    q0 = observe()[0].astype(np.float64)
    pitch, e = plan_grasp_pitch(p_obj, q0)
    if pitch is None:
        raise Abort(f"{label}: the arm cannot reach that cube at ANY grasp pitch "
                    f"(best IK residual {e * 1e3:.0f}mm; cube at "
                    f"r={np.hypot(p_obj[0], p_obj[1]) * 100:.0f}cm). Move it closer.")
    say(f"{label}: grasping with the hand {pitch:.0f}deg DOWN (IK residual "
        f"{e * 1e3:.1f}mm) — a horizontal hand would need the wrist inside the base, "
        f"which is why every previous run closed on air")

    pitch0 = float(q0[1] + q0[2] + q0[3])
    tip0 = np.asarray(kin.forward_kinematics(q0))[:3, 3]
    d0 = max(0.05, float(np.linalg.norm(p_obj - tip0)))

    for hop in range(24):
        checkpoint()
        q = observe()[0].astype(np.float64)
        tip = np.asarray(kin.forward_kinematics(q))[:3, 3]
        p_above = np.array([p_obj[0], p_obj[1], p_obj[2] + STANDOFF_H])
        d_way = float(np.linalg.norm(p_above - tip))
        d_cube = float(np.linalg.norm(p_obj - tip))
        r_obj = float(np.hypot(p_obj[0], p_obj[1]))
        ang = float(math.degrees(math.atan2(p_obj[1], p_obj[0])))
        with lock:
            state["dist_mm"] = round(d_cube * 1e3)
            state["target_polar"] = [round(r_obj * 100, 1), round(ang, 1),
                                     round(float(p_obj[2]) * 100, 1)]
            state["obj3d"] = [float(v) for v in p_obj]
            state["obj3d_label"] = label.lower()
        say(f"approach {hop}: tip->waypoint={d_way * 1e3:.0f}mm tip->cube={d_cube * 1e3:.0f}mm "
            f"| cube r={r_obj * 100:.1f}cm ang={ang:+.0f}deg z={p_obj[2] * 100:.1f}cm "
            f"| pitch={float(q[1] + q[2] + q[3]):.0f}deg")

        if d_way <= 0.012:
            say(f"{label}: at the waypoint, {STANDOFF_H * 100:.0f}cm above the cube")
            break
        # Don't back away from the cube to reach a waypoint we are already past. That
        # retreat is what looked like "goes forward 12cm then goes back". If we are
        # already above the cube and close, just descend.
        if tip[2] >= p_obj[2] + 0.015 and float(np.hypot(*(tip[:2] - p_obj[:2]))) <= 0.020:
            say(f"{label}: already over the cube — descending from here")
            break

        # Refine the lock ONLY while the detector is still honest. Inside COMMIT_M the
        # cube is half out of frame and sliding under the fingers -- its opinion there is
        # worthless, and believing it is what used to drag the target around.
        if d_cube > COMMIT_M:
            tr = detect_now(finder, tries=6)
            if tr is not None:
                p_new = _measure_point(tr, T_cam_of(observe()[0]))
                if p_new is not None:
                    p_new = push_out_radial(p_new)     # same correction as the lock, or
                    # the raw (too-close) re-measure would drag the target back inward
                    if float(np.linalg.norm(p_new - p_obj)) < 0.06:
                        p_obj = 0.65 * p_obj + 0.35 * np.asarray(p_new, np.float64)
                        p_obj[2] = float(np.clip(p_obj[2], 0.005, 0.050))
                        tracker.p_anchor = p_obj.copy()
                        tracker.anchor_t = time.time()

        # Ramp the pitch toward the grasp pitch as we close in rather than snapping the
        # camera down: keeps the cube in view longer, and keeps the motion smooth.
        prog = float(np.clip(1.0 - d_cube / d0, 0.0, 1.0))
        pitch_i = float(pitch0 + (pitch - pitch0) * min(1.0, 1.6 * prog))

        step = p_above - tip
        n = float(np.linalg.norm(step))
        if n < 1e-6:
            break
        p_tgt = tip + step / n * min(HOP_M, n)
        q_tgt, e = _ik_hold_pitch(q, p_tgt, pitch_i, float(q[4]), ret_err=True)
        if e > 0.006:      # try going straight to the grasp pitch instead of the ramp
            q_tgt, e = _ik_hold_pitch(q, p_tgt, pitch, float(q[4]), ret_err=True)
        if e > 0.006:
            raise Abort(f"{label}: waypoint unreachable (IK residual {e * 1e3:.0f}mm). "
                        f"Refusing to drive to a half-solved pose.")
        goto_smooth(q_tgt, settle=0.08, step=2.5)
    else:
        say(f"{label}: ran out of hops — descending from here")

    return p_obj.copy(), pitch


def descend_and_close(p_obj, pitch, label):
    """Straight down the last few cm onto the LOCKED point, hand pointing down.

    DELIBERATELY NO PIXELS HERE. The old descent exited on `bbox_bottom >= 477` -- the
    cube touching the bottom of the frame -- but a down-looking camera puts the cube at
    the image bottom while the hand is still ~2 cm ABOVE it, so the fingers closed on air
    every run (z_ee=+39mm with cube_z=+20mm). We already know where the cube is; go there.
    Descending vertically also means the fingers arrive around the cube's SIDES instead of
    shoving it away.
    """
    set_phase(f"GRASP {label}", "descending onto the cube")
    send_joints(observe()[0].astype(np.float64), gripper=95.0)   # fingers OPEN first
    time.sleep(0.15)

    p_grip = p_obj.copy()
    p_grip[2] = max(float(p_obj[2]), OBJ_FLOOR_Z)

    for i in range(14):
        checkpoint()
        q = observe()[0].astype(np.float64)
        tip = np.asarray(kin.forward_kinematics(q))[:3, 3]
        d = p_grip - tip
        n = float(np.linalg.norm(d))
        say(f"descend {i}: tip->cube={n * 1e3:.0f}mm  "
            f"tip=({tip[0] * 1e3:+.0f},{tip[1] * 1e3:+.0f},{tip[2] * 1e3:+.0f}) "
            f"cube=({p_grip[0] * 1e3:+.0f},{p_grip[1] * 1e3:+.0f},{p_grip[2] * 1e3:+.0f})mm")
        if n <= 0.008:
            say(f"{label}: fingertip on the cube ({n * 1e3:.0f}mm) — closing")
            return True
        p_tgt = tip + d / n * min(0.015, n)
        q_tgt, e = _ik_hold_pitch(q, p_tgt, pitch, float(q[4]), ret_err=True)
        if e > 0.005:
            say(f"{label}: descend blocked — IK residual {e * 1e3:.0f}mm. Refusing to "
                f"drive to a half-solved pose (that is what made the arm wander).")
            return False
        goto_smooth(q_tgt, settle=0.06, step=1.8)
    say(f"{label}: descend ran out of steps")
    return False


def detect_now(finder, tries=12):
    """Look at the CURRENT gripper view without moving the arm. Returns the
    detection or None. This is the only 'locate' step — first-person view only."""
    for _ in range(tries):
        checkpoint()
        joints, rgb, _ = observe()
        tr = finder(rgb, T_cam_of(joints))
        if tr is not None:
            return tr
        time.sleep(0.03)
    return None


def gentle_scan(finder, label, span_deg=70.0):
    """Slow, continuous base pan that STOPS the instant the cube enters the
    view — no pose hops, no returning to a home pose. Only runs when the target
    is not already visible."""
    set_phase(f"SEARCH {label}", "slow scan — stops the moment it sees the cube")
    start_pan = float(observe()[0][0])
    for sign in (1.0, -1.0, -1.0, 1.0):   # sweep right, back through left, return
        target = start_pan + sign * span_deg
        for _ in range(400):
            checkpoint()
            joints, rgb, _ = observe()
            tr = finder(rgb, T_cam_of(joints))
            if tr is not None:
                say(f"{label} spotted at uv=({tr.uv[0]:.0f},{tr.uv[1]:.0f})")
                return tr
            cur = float(joints[0])
            if abs(cur - target) < 2.0:
                break
            q = joints.copy()
            q[0] += float(np.clip(target - cur, -0.9, 0.9))   # ~13 deg/s, smooth
            send_joints(q)
            time.sleep(LOOP_DT)
    raise Abort(f"{label}: not found in the gripper view")


# ---------------- mission ----------------
# ================= CLEAN PICK (the Start button) ===========================
# Everything the pick needs is in ONE gripper-camera frame: the cube and the
# fingertips are both visible. So the whole algorithm is four honest steps:
#   1. SEE the cube, cast its ground-contact pixels onto the table -> (x, y)
#      (the exact same math the 2D map already uses)
#   2. IK the fingertip ON TOP of (x, y), hand pointing straight down
#   3. DESCEND straight down onto (x, y, table)
#   4. CLOSE on torque, then LIFT
# No stereo, no push-out fudge, no multi-vantage triangulation, no gaze servo,
# no camera self-calibration in the loop. If it lands off, we SEE by how much and
# trim — we do not guess.
PICK_HOVER_Z = 0.06        # hover this far above the table before descending
PICK_GRASP_Z = 0.015       # descend to here (grab the cube's lower body)
PICK_LIFT_M = 0.10


def _target_finder():
    """(finder, tracker, label) for the colour named in the detection query.

    Use whichever colour is named FIRST. A plain `"green" in query` test picked
    GREEN out of the default "red cube, green cube" — so the arm dutifully drove at
    the green cube while the user was waiting for it to grab the red one."""
    q = (state.get("query") or "red").lower()
    ir, ig = q.find("red"), q.find("green")
    if ig != -1 and (ir == -1 or ig < ir):
        return find_green, green_tracker, "green"
    return find_red, red_tracker, "red"


def locate_target_xy(finder, tries=8):
    """Where the cube sits on the table, base-frame (x, y): cast its ground-contact
    pixels onto the table plane and take the median over a few reads. This is the
    SAME localisation the 2D map uses. None if the cube isn't clearly in view."""
    pts = []
    for _ in range(tries):
        checkpoint()
        j, rgb, _ = observe()
        tr = finder(rgb, T_cam_of(j))
        if tr is None or tr.clipped:
            time.sleep(0.05)
            continue
        xy, _st, _sz = obj_xy_2d(tr.bbox_xyxy, T_cam_of(j))
        if xy is not None:
            pts.append(xy)
        time.sleep(0.03)
    if len(pts) < 3:
        return None
    return np.median(np.array(pts), axis=0)


# ---- uncalibrated visual servo -------------------------------------------
# We never trust where the camera is bolted. Instead the arm MEASURES its own
# image response: move the fingertip a known 2 cm in base X, watch which way the
# cube's pixel slides; same for Y. That 2x2 "pixels per metre" matrix replaces the
# camera-mount transform completely. Then we simply drive the cube's pixel onto
# the fingertip pixel (both are in the same picture), descending as we go. The
# DESCENT is pure kinematics (joint angles + URDF), which is always trustworthy.
SERVO_TOL_PX = 26.0        # cube pixel this close to the fingertip pixel = aligned
SERVO_MAX_STEP = 0.030     # m of horizontal correction per iteration
SERVO_PROBE = 0.020        # m probe step used to MEASURE the pixel response
SERVO_DESC = 0.015         # m descend per iteration once roughly centred
SERVO_DESC_GATE_PX = 120.0 # only descend while the cube is at least this centred
CUBE_EDGE_M = 0.0508       # the cube's real edge: 2 inches. Back-solved from a
                           # known 26cm sighting (edge = range*bbox_px/fx). At 1in
                           # it read 12cm for a cube really 26cm out, and the
                           # under-read range also squashed the cubes together — turns apparent size into
                           # a range. If your cube isn't 4cm, change THIS number.
# Lateral aim trim, in pixels, applied to the fingertip aim point. The gripper was
# consistently ending up LEFT of the cube. Moving the aim point LEFT (negative)
# makes the arm travel FURTHER RIGHT before it thinks it is lined up, because
# swinging the camera right slides the cube left in the picture. If it now
# overshoots to the right, make this less negative; if still left, more negative.
# The arm is still landing left, so push the aim point further left.
AIM_DU = -45.0   # final centering bias: robot lands on the cube's RIGHT
AIM_DV = 0.0
# Global scale on computed range. Increase (>1.0) to push mapped objects FURTHER
# OUT and spread them apart; decrease (<1.0) to pull them closer together. This
# is a coarse calibration knob for when the apparent-size / stereo depth numbers
# are consistently off in scale.
RANGE_SCALE = [1.0]
# Rotate all localized (x, y) positions in the base horizontal plane. Positive
# = counter-clockwise. Use this when the camera shows objects on opposite sides
# but the map clusters them on one side (a heading/yaw error in the hand-eye).
MAP_BEARING_OFFSET_DEG = [0.0]
# Damping. Taking the FULL Jacobian step overshoots and the arm hunts back and
# forth. Move a fraction of the computed correction each step, and shrink the step
# as the error shrinks, so it eases in instead of shaking.
SERVO_GAIN = 0.55
SERVO_DESC_RANGE = 0.12    # do NOT descend until the cube is actually this close.
                           # Pixel alignment alone is not proximity.
SERVO_WORK_Z = 0.07        # hold the fingertip at THIS height (absolute) while
                           # approaching. Commanding only relative height changes let
                           # per-step error accumulate upward — measured live, the tip
                           # climbed 7cm -> 22cm and "converged" in mid-air.
MAX_SERVO_ITERS = 26       # short steps now, so allow more of them


def _cube_track(finder, tries=4):
    """The cube's detection right now (pixel + bbox), or None."""
    for _ in range(tries):
        checkpoint()
        j, rgb, _ = observe()
        tr = finder(rgb, T_cam_of(j))
        if tr is not None:
            return tr
        time.sleep(0.05)
    return None


def _cube_uv(finder, tries=4):
    tr = _cube_track(finder, tries)
    return None if tr is None else np.array(tr.uv, np.float64)


def _cube_range_m(tr):
    """How far away the cube is, from how BIG IT LOOKS.

        range = focal_length * real_edge / apparent_width_px      (pinhole)

    This needs only the lens focal length and the cube's real size — no camera-mount
    transform — so it stays honest even while the mount numbers are wrong. It is the
    signal that tells us whether we are actually NEAR the cube, as opposed to merely
    lined up with it in the picture (a cube 40cm away can sit exactly on the
    fingertip pixel, which is why descending on pixel-alignment alone landed the
    gripper on bare board half way out)."""
    w = float(max(4.0, tr.bbox_xyxy[2] - tr.bbox_xyxy[0]))
    return float(fx * CUBE_EDGE_M / w)


def _tip(q):
    return np.asarray(kin.forward_kinematics(np.asarray(q, np.float64)))[:3, 3]


def _move_tip(p_tgt, pitch, j5, settle=0.12, step=2.5):
    """Move the FINGERTIP to a base-frame POSITION. Returns the pitch used, or None.

    Demanding ONE exact wrist pitch is why the arm sat still: measured live, it was
    commanded 3cm and achieved 0.0cm over and over at r=21cm — nowhere near a reach
    limit. Holding a single pitch simply has no IK solution in much of this
    workspace. Getting the fingertip to the POSITION is what matters; the precise
    wrist angle does not until the final grasp. So try the requested pitch, then
    sweep outward and take the first angle that actually solves.
    """
    q = observe()[0].astype(np.float64)
    p_tgt = np.asarray(p_tgt, np.float64)
    cands = [float(pitch)]
    for dp in (-8, 8, -16, 16, -25, 25, -35, 35, -45, 45, -55, 55, -65, -75):
        pc = float(pitch) + dp
        if 0.0 <= pc <= 88.0:
            cands.append(pc)
    for pc in cands:
        q_t, e = _ik_hold_pitch(q, p_tgt, pc, j5, ret_err=True)
        # Match the sliding tolerance in plan_grasp_pitch so shallow / far-reach
        # poses that were accepted there are not rejected here.
        tol = 0.012 if pc <= 15.0 else 0.008
        if e <= tol:
            goto_smooth(q_t, settle=settle, step=step)
            return pc
    return None


ALIGN_TOL_PX = 40.0        # cube this close to the gripper pixel = aligned
ALIGN_ITERS = 3            # approach already gets close, so quick final centering
ALIGN_PROBE_M = 0.015      # metres to probe when measuring the image response
ALIGN_CAP_M = 0.020        # max lateral correction per iteration


def _center_on_cube(finder, gp, j5):
    """Center the cube under the jaws with DECOUPLED single-DOF servos:

      * horizontal pixel error  ->  rotate the BASE (shoulder_pan)
      * vertical pixel error    ->  reach RADIALLY in/out

    Each axis is one joint and monotonic, so the sign is trivial to measure and the
    loop is stable. The previous 2D-Cartesian Jacobian oscillated (109->96->119px)
    because base rotation, reach and an auto-swept wrist pitch all mixed into it.
    Pitch is held FIXED here. Returns the final (x, y), or None."""
    tgt_u = HAND_UV[0] + AIM_DU
    tgt_v = HAND_UV[1] + AIM_DV
    tr = _cube_track(finder, tries=5)
    if tr is None:
        say("center: cube not in view")
        return None
    uv0 = np.array(tr.uv, np.float64)
    q0 = observe()[0].astype(np.float64)
    p0 = _tip(q0)

    # Small, slow probe moves so the measurement is clean and the arm does not jerk.
    PAN_PROBE = 3.0      # deg
    RAD_PROBE = 0.015    # m
    # Speed is scaled by estimated cube range so the arm slows down as it gets close.
    # The BASE is the main source of jerk, so its max step scales the most.
    r_m = _cube_range_m(tr)
    close = r_m < 0.12          # within ~12 cm -> slow/close mode
    speed = 0.9 if close else 1.6
    CENTER_STEP = 1.8 * speed   # deg per goto_smooth tick
    CENTER_SETTLE = 0.12 if close else 0.08  # s
    MAX_PAN_STEP = (4.5 if close else 8.0)  # deg per iteration
    MAX_RAD_STEP = (0.020 if close else 0.040)  # m per iteration

    # --- probe the HORIZONTAL sign: cube_u change per +PAN_PROBE of base pan ---
    qp = q0.copy(); qp[0] = float(np.clip(q0[0] + PAN_PROBE, -100, 100))
    goto_smooth(qp, settle=CENTER_SETTLE, step=CENTER_STEP)
    trp = _cube_track(finder, tries=4)
    goto_smooth(q0, settle=CENTER_SETTLE, step=CENTER_STEP)
    if trp is None or abs(float(qp[0] - q0[0])) < 0.5:
        say("center: horizontal probe failed")
        return _tip(observe()[0])[:2]
    du_dpan = (float(trp.uv[0]) - uv0[0]) / (qp[0] - q0[0])       # px per deg
    if abs(du_dpan) < 3.0:
        say("center: base rotation barely moves the cube")
        return _tip(observe()[0])[:2]

    # --- probe the VERTICAL sign: cube_v change per +RAD_PROBE radial reach ---
    r0 = float(np.hypot(p0[0], p0[1])); uo = np.array([p0[0], p0[1]]) / max(r0, 1e-6)
    dv_dr = None
    if _move_tip(np.array([p0[0] + uo[0]*RAD_PROBE, p0[1] + uo[1]*RAD_PROBE, p0[2]]),
                 gp, j5, settle=CENTER_SETTLE, step=CENTER_STEP) is not None:
        trr = _cube_track(finder, tries=4)
        _move_tip(np.array([p0[0], p0[1], p0[2]]), gp, j5,
                  settle=CENTER_SETTLE, step=CENTER_STEP)
        if trr is not None:
            dv_dr = (float(trr.uv[1]) - uv0[1]) / RAD_PROBE        # px per metre
    say(f"center: du/dpan={du_dpan:.1f}px/deg" +
        (f", dv/dr={dv_dr:.0f}px/m" if dv_dr else ", (no vertical probe)"))

    for it in range(ALIGN_ITERS):
        checkpoint()
        tr = _cube_track(finder, tries=3)
        if tr is None:
            say("center: cube gone (likely under the jaws) - stopping")
            break
        du = float(tr.uv[0]) - tgt_u
        dv = float(tr.uv[1]) - tgt_v
        say(f"center {it}: {abs(du):.0f}px {'right' if du > 0 else 'left'}, "
            f"{abs(dv):.0f}px {'below' if dv > 0 else 'above'} the jaws")
        if abs(du) < ALIGN_TOL_PX and abs(dv) < ALIGN_TOL_PX * 1.3:
            say("centered on the cube")
            break
        moved = False
        q = observe()[0].astype(np.float64)
        if abs(du) >= ALIGN_TOL_PX:                # horizontal via base rotation
            dpan = float(np.clip(-du / du_dpan, -MAX_PAN_STEP, MAX_PAN_STEP))
            q[0] = float(np.clip(q[0] + dpan, -100, 100))
            goto_smooth(q, settle=CENTER_SETTLE, step=CENTER_STEP); moved = True
        if dv_dr and abs(dv) >= ALIGN_TOL_PX * 1.3:   # vertical via radial reach
            p = _tip(observe()[0]); r = float(np.hypot(p[0], p[1]))
            uo = np.array([p[0], p[1]]) / max(r, 1e-6)
            dr = float(np.clip(-dv / dv_dr, -MAX_RAD_STEP, MAX_RAD_STEP))
            if _move_tip(np.array([p[0] + uo[0]*dr, p[1] + uo[1]*dr, p[2]]),
                         gp, j5, settle=CENTER_SETTLE, step=CENTER_STEP) is not None:
                moved = True
        if not moved:
            break
    tip = _tip(observe()[0])
    return np.array([tip[0], tip[1]], np.float64)


def _step_tip(p_from, delta, pitch, j5):
    """Move as far along `delta` as the arm can ACTUALLY reach.

    Bailing out the moment the full step is unreachable is what made the approach
    "go forward a little and give up": one 3 cm request fails IK and the whole
    servo stops. Instead, shorten the step (and, as a last resort, flatten the hand
    a little — a shallower pitch reaches further forward) until something is
    reachable.

    Returns the pitch ACTUALLY used (the fallback may flatten the hand), or None if
    nothing was reachable. The caller must adopt the returned pitch — otherwise the
    next move snaps the wrist back to the old angle, the wrist oscillates, and the
    camera swings off the cube.
    """
    # Smaller commanded moves get a gentler servo rate and a longer settle, so the
    # final centimetres glide in instead of snapping and ringing.
    n = float(np.linalg.norm(delta))
    fine = n < 0.012
    settle = 0.22 if fine else 0.12
    rate = 1.5 if fine else 2.5
    for scale in (1.0, 0.7, 0.45, 0.3, 0.18, 0.1):
        used = _move_tip(p_from + delta * scale, pitch, j5, settle=settle, step=rate)
        if used is not None:
            return used
    return None


def _measure_pixel_response(finder, pitch, j5):
    """Probe the arm against its own camera: step the fingertip in base X, then Y,
    and watch the cube's pixel move. Returns the 2x2 matrix (px per metre)."""
    uv0 = _cube_uv(finder)
    if uv0 is None:
        return None
    p0 = _tip(observe()[0])
    J = np.zeros((2, 2))
    for c, d in enumerate((np.array([SERVO_PROBE, 0.0, 0.0]),
                           np.array([0.0, SERVO_PROBE, 0.0]))):
        if _move_tip(p0 + d, pitch, j5) is None:
            _move_tip(p0, pitch, j5)
            return None
        # VERIFY THE ARM ACTUALLY MOVED. If it didn't, the pixel difference we are
        # about to divide by is pure detector noise, and the resulting direction
        # matrix sends the arm confidently the WRONG WAY (seen live: it drove
        # backwards, away from the cube, every step).
        moved = float(np.linalg.norm(_tip(observe()[0]) - p0))
        uv1 = _cube_uv(finder)
        _move_tip(p0, pitch, j5)                  # always return to the start
        if uv1 is None:
            return None
        if moved < 0.5 * SERVO_PROBE:
            say(f"probe {'X' if c == 0 else 'Y'}: asked for {SERVO_PROBE*100:.0f}cm "
                f"but the arm moved {moved*100:.1f}cm — cannot measure direction here")
            return None
        J[:, c] = (uv1 - uv0) / moved             # divide by what REALLY happened
    if abs(float(np.linalg.det(J))) < 1e3:        # probe produced no usable motion
        return None
    return J


def _mapped_xy(label):
    """The mapped (x, y) of the best-supported object of this colour, or None."""
    want = label.split()[0].lower()
    best = None
    with w2d_lock:
        for o in W2D["objs"].values():
            if want in o["label"].lower():
                if best is None or o["n"] > best["n"]:
                    best = o
        return None if best is None else np.array(best["xy"], float)


TARGET_RIGHT_TRIM_M = 0.050 # shift the APPROACH hover target this far to the
                            # cube's RIGHT, so the cube stays on the LEFT side of the
                            # camera view during approach.
APPROACH_STEPS = 3          # default approach waypoints. Step 1 closes ~90% of the
                            # gap so the arm gets close fast; remaining steps close
                            # the rest without passing the target.


def run_mission():
    """Close on the mapped cube in smooth stages, re-checking the map at every
    stage, then descend gradually and grip.

    One long move to the target is a dive: if the mapped position is a little off,
    nothing notices until the gripper is already there. Instead cover the distance
    in stages -- each one closes part of the remaining gap, then looks again and
    refines the target. Errors get corrected while there is still room to correct
    them, and the motion rates are low enough not to jerk.
    """
    def _shift_right(p, d):
        r = float(np.hypot(*p))
        if r < 1e-6 or d == 0.0:
            return np.array(p, dtype=np.float64)
        u = np.array(p, dtype=np.float64) / r
        # Perpendicular direction that puts the gripper on the cube's RIGHT.
        # If this still lands left, the base frame y-axis is inverted.
        perp = np.array([u[1], -u[0]])
        return np.array(p, dtype=np.float64) + perp * d

    def _approach_target(cube_xy, back_m=0.005):
        """Hover position for the approach: short of the cube and to its right.

        back_m: stop this many metres radially BEFORE the cube (keeps it in view).
        The lateral offset is TARGET_RIGHT_TRIM_M (tunable in the UI).
        """
        cube_xy = np.asarray(cube_xy, dtype=np.float64)
        r = float(np.hypot(*cube_xy))
        if r > 1e-6:
            backed = cube_xy * ((r - back_m) / r)
        else:
            backed = cube_xy.copy()
        return _shift_right(backed, TARGET_RIGHT_TRIM_M)

    try:
        stop_flag.clear()
        with lock:
            state["running"] = True
            state["t0"] = time.time()
        finder, tracker, label = _target_finder()

        cube_xy = _mapped_xy(label)
        if cube_xy is None:
            raise Abort(f"{label} cube is not on the 2D map - press 'Scan -> 2D map' first")
        say(f"{label} cube mapped at x={cube_xy[0]*100:.1f} y={cube_xy[1]*100:.1f} cm "
            f"(r={np.hypot(*cube_xy)*100:.1f}cm) - approaching in {APPROACH_STEPS} stages")
        send_joints(observe()[0], gripper=95.0)

        # ---- approach: stay to the right and 2 cm short of the cube ----
        # The hover target is always computed from the latest cube estimate, so it
        # keeps the cube on the LEFT side of the image and stops before the arm
        # passes it. Refinements update the cube position, not the approach offset.
        for i in range(APPROACH_STEPS):
            checkpoint()
            q = observe()[0].astype(np.float64)
            j5 = float(q[4])         # keep the wrist as-is; do NOT twist while approaching
            tip = _tip(q)
            target_xy = _approach_target(cube_xy)
            say(f"approach {i+1}: cube=({cube_xy[0]*100:.1f},{cube_xy[1]*100:.1f})cm "
                f"target=({target_xy[0]*100:.1f},{target_xy[1]*100:.1f})cm "
                f"offset=({(target_xy-cube_xy)[0]*100:.1f},{(target_xy-cube_xy)[1]*100:.1f})cm")
            delta_xy = target_xy - tip[:2]
            dist_xy = float(np.linalg.norm(delta_xy))

            if dist_xy < 0.015:
                say(f"approach {i+1}: already at approach hover")
                break

            last = (i == APPROACH_STEPS - 1)
            frac = 1.0 if last else 0.9
            # Last step reaches the hover target exactly; earlier steps close 90%
            # of the remaining gap, capped so the camera stays on the cube.
            step_len = dist_xy * frac if last else min(dist_xy * frac, 0.12)
            step_xy = (delta_xy / dist_xy * step_len) if dist_xy > 1e-6 else np.zeros(2)
            wx = float(tip[0] + step_xy[0])
            wy = float(tip[1] + step_xy[1])

            pitch, e = plan_grasp_pitch(np.array([cube_xy[0], cube_xy[1], PICK_GRASP_Z]), q)
            if pitch is None:
                raise Abort(f"r={np.hypot(*cube_xy)*100:.0f}cm is out of reach "
                            f"(best IK {e*1e3:.0f}mm)")
            with lock:
                state["obj3d"] = [float(cube_xy[0]), float(cube_xy[1]), PICK_GRASP_Z]
                state["obj3d_label"] = label
            set_phase("PICK", f"approach {i+1}/{APPROACH_STEPS} "
                              f"-> x={wx*100:.0f} y={wy*100:.0f} cm")
            if _move_tip(np.array([wx, wy, PICK_HOVER_Z]), pitch, j5,
                         settle=0.15, step=1.4) is None:
                raise Abort(f"IK cannot reach x={wx*100:.0f} y={wy*100:.0f}")

            # Refine the cube position from the new vantage. The approach target
            # will recompute with the right/short offset next loop.
            time.sleep(0.05)
            sense_2d()
            xy2 = None
            trf = _cube_track(finder, tries=3)
            if trf is not None and not trf.clipped:
                jf, _rgbf, _ = observe()
                zfs = [z for z in (read_depth_m(trf.uv) for _ in range(3)) if z is not None]
                zf = float(np.median(zfs)) if zfs else None
                xyf, _rngf, _szf = obj_xy_2d(trf.bbox_xyxy, T_cam_of(jf), z_m=zf)
                if xyf is not None:
                    xy2 = xyf
            if xy2 is None:
                xy2 = _mapped_xy(label)
            if xy2 is not None:
                adj = float(np.linalg.norm(xy2 - cube_xy))
                if adj <= 0.05:
                    say(f"check {i+1}/{APPROACH_STEPS}: refined cube by {adj*100:.1f}cm")
                    cube_xy = xy2
                else:
                    say(f"check {i+1}/{APPROACH_STEPS}: refined target jump {adj*100:.1f}cm - ignoring")

        # ---- FINAL CENTERING BY EYE, then descend on the aligned spot ----
        q = observe()[0].astype(np.float64)
        j5 = float(q[4])             # no wrist twist
        gp, _e = plan_grasp_pitch(np.array([cube_xy[0], cube_xy[1], PICK_GRASP_Z]), q)
        gp = gp if gp else 70.0
        set_phase("PICK", "centering the cube under the jaws")
        aligned = _center_on_cube(finder, gp, j5)
        if aligned is not None:
            gx, gy = float(aligned[0]), float(aligned[1])
        else:
            gx, gy = float(cube_xy[0]), float(cube_xy[1])
        q = observe()[0].astype(np.float64)
        z0 = float(_tip(q)[2])
        for f in (0.4, 0.75, 1.0):
            checkpoint()
            z = float(z0 + (PICK_GRASP_Z - z0) * f)
            last = (f == 1.0)
            set_phase("PICK", f"descending to z={z*100:.1f}cm")
            step = 0.9 if last else 1.4
            settle = 0.18 if last else 0.12
            if _move_tip(np.array([gx, gy, z]), gp, j5,
                         settle=settle, step=step) is None:
                say("could not reach that height - closing from here")
                break

        set_phase("PICK", "closing the gripper")
        held, _i = close_with_current(step=4.0, delay=0.08)
        set_phase("PICK", "lifting")
        ee_move_rel([0, 0, PICK_LIFT_M], settle=0.25)
        set_phase("DONE" if held else "PICK",
                  f"{label} cube {'picked' if held else 'missed - closed on air'}")

    except Abort as e:
        set_phase("ABORTED", str(e))
    except Exception as e:
        set_phase("ERROR", f"{type(e).__name__}: {e}")
    finally:
        try:
            with lock:
                g = state.get("gripper")
            if g is not None and g < 15.0:
                c = gripper_current()
                if c is None or abs(c) < 4.0:      # not carrying load -> relax
                    send_joints(observe(overlay=False)[0], gripper=40.0)
        except Exception:
            pass
        with lock:
            state["running"] = False


# ---------------- web ----------------
app = Flask(__name__)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAX stack mission</title><style>
body{margin:0;background:#141a21;color:#e6ecf1;font:15px/1.5 "Segoe UI",system-ui,sans-serif}
main{max-width:1280px;margin:0 auto;padding:18px;display:grid;grid-template-columns:2fr 1fr;gap:16px}
@media(max-width:900px){main{grid-template-columns:1fr}}
h1{font-size:17px;margin:0 0 10px;grid-column:1/-1}
h1 .tail{color:#93a1ae;font-weight:400;font-size:13px;margin-left:10px}
.cams{display:flex;flex-direction:column;gap:10px}
img{width:100%;border:1px solid #2b3540;border-radius:6px;background:#000;display:block}
.lbl{font-size:11px;color:#93a1ae;letter-spacing:.1em;text-transform:uppercase;margin:0 0 4px;font-family:ui-monospace,Consolas,monospace}
.panel{background:#1b232c;border:1px solid #2b3540;border-radius:6px;padding:12px}
.phase{font:600 20px/1.2 ui-monospace,Consolas,monospace;color:#4cc275}
.phase.bad{color:#d4795f}.detail{color:#93a1ae;font-size:13px;min-height:18px;margin:4px 0 10px}
table{width:100%;border-collapse:collapse;font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
td{padding:3px 6px;border-bottom:1px solid #2b3540;color:#93a1ae}td+td{color:#e6ecf1;text-align:right}
.btns{display:flex;gap:8px;margin-top:12px}
button{flex:1;padding:9px 0;border:0;border-radius:5px;font:600 13px "Segoe UI";cursor:pointer}
#b-start{background:#2e9e5b;color:#fff}#b-stop{background:#b0533c;color:#fff}#b-home{background:#2b3540;color:#e6ecf1}
pre{background:#10161c;border:1px solid #2b3540;border-radius:6px;padding:10px;font-size:11.5px;
line-height:1.5;height:240px;overflow-y:auto;white-space:pre-wrap;margin:12px 0 0}
.jog{margin-top:14px}
#jogmsg{color:#6fb2ff;font-family:ui-monospace,Consolas,monospace;font-size:11px;margin-left:8px}
.pad{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:6px}
.pad .j{padding:14px 0;background:#26313c;color:#e6ecf1;border:1px solid #37454f;border-radius:6px;
font:600 14px "Segoe UI";cursor:pointer;touch-action:manipulation;user-select:none}
.pad .j:active{background:#3a6ea5}
.pad .g{background:#2e5b46}
.stp{grid-column:1/-1;display:flex;align-items:center;gap:8px;color:#93a1ae;font-size:12px;margin-top:4px}
.stp input{flex:1}
.qbox{margin-top:14px}
.qrow{display:flex;gap:8px;margin-top:6px;align-items:center}
.qrow input{flex:1;padding:10px;background:#10161c;border:1px solid #37454f;border-radius:6px;
color:#e6ecf1;font:14px "Segoe UI"}
.qrow input[type=number]{max-width:90px}
.qrow label{color:#93a1ae;font-size:12px;white-space:nowrap;min-width:70px}
.qrow button{padding:10px 18px;background:#2e6ea5;color:#fff;border:0;border-radius:6px;
font:600 13px "Segoe UI";cursor:pointer}
.sticks{display:flex;gap:18px;justify-content:center;align-items:center;margin:8px 0 8px;flex-wrap:wrap}
.stick{position:relative;width:118px;height:118px;border-radius:50%;flex:0 0 auto;
background:radial-gradient(circle at 50% 42%,#222c37,#141a21 72%);border:1px solid #37454f;
touch-action:none;user-select:none;display:flex;align-items:center;justify-content:center}
.stick::before{content:"";position:absolute;inset:0;border-radius:50%;opacity:.55;
background:linear-gradient(#2b3540,#2b3540) center/1px 54% no-repeat,linear-gradient(#2b3540,#2b3540) center/54% 1px no-repeat}
.stick.vert::before{background:linear-gradient(#2b3540,#2b3540) center/1px 54% no-repeat}
.nub{width:46px;height:46px;border-radius:50%;pointer-events:none;
background:radial-gradient(circle at 36% 30%,#4a93c9,#22506f);border:1px solid #2e6ea5;box-shadow:0 3px 12px rgba(0,0,0,.5)}
.sl{position:absolute;bottom:-14px;left:50%;transform:translateX(-50%);white-space:nowrap;
font:600 9px ui-monospace,Consolas;letter-spacing:.4px;color:#6b7a86;text-transform:uppercase}
@media(max-width:760px){
  main{padding:10px;gap:10px}
  .panel{display:flex;flex-direction:column}
  .jog{order:-1;margin-top:0}
  .cams img{max-height:42vh;object-fit:contain}
  .phone-hide{display:none}
  .pad{grid-template-columns:repeat(2,1fr)}
}
</style></head><body><main>
<h1>RAX · pick green → place on red<span class="tail">v2 · continuous servo · dual cam</span></h1>
<div class="cams">
  <div><p class="lbl">gripper camera (OAK-D)</p><img src="/stream" alt="gripper view"></div>
  <div class="phone-hide"><p class="lbl">room camera (camsurv)</p><img src="/stream2" alt="room view"
       onerror="this.parentElement.style.display='none'"></div>
  <div class="phone-hide">
    <p class="lbl">robot + object 3d · drag to rotate ·
      <a id="r3d-link" href="#" target="_blank" style="color:#6fb2ff;text-transform:none">rerun ↗</a></p>
    <canvas id="v3d"
       style="width:100%;height:340px;border:1px solid #2b3540;border-radius:6px;background:#0b0f14;display:block;touch-action:none"></canvas>
  </div>
  <div class="phone-hide">
    <p class="lbl">2D map (top-down · base frame) · click a dot / row to go on top</p>
    <canvas id="v2d"
       style="width:100%;height:300px;border:1px solid #2b3540;border-radius:6px;background:#0b0f14;display:block;touch-action:none"></canvas>
    <table id="map2dtab" style="margin-top:6px"><tbody></tbody></table>
    <div class="btns">
      <button class="dim" id="b-scan2d" onclick="fetch('/scan2d',{method:'POST'})"
        style="background:#2e6ea5;color:#fff">Scan → 2D map</button>
      <button class="dim" onclick="fetch('/clearmap2d',{method:'POST'})">Clear 2D map</button>
    </div>
  </div>
</div>
<div class="panel">
  <div class="phase" id="phase">—</div><div class="detail" id="detail"></div>
  <table><tbody id="rows"></tbody></table>
  <div class="btns">
    <button id="b-start" onclick="fetch('/start',{method:'POST'})">Start</button>
    <button id="b-stop" onclick="fetch('/stop',{method:'POST'})">Stop</button>
    <button id="b-home" onclick="fetch('/home',{method:'POST'})">Fold home</button>
  </div>
  <div class="btns">
    <button id="b-home" onclick="fetch('/reset',{method:'POST'})">Reset pose</button>
  </div>
  <div class="jog">
    <p class="lbl">FPV polar · drag sticks, or W/S radius · A/D base · R/F height · T/G tilt · Q/E roll · space grip <span id="jogmsg"></span></p>
    <div class="sticks">
      <div class="stick" id="st-move"><div class="nub"></div><span class="sl">reach · base</span></div>
      <div class="stick vert" id="st-lift"><div class="nub"></div><span class="sl">height</span></div>
    </div>
    <div class="pad">
      <button data-d="up"    class="j">R ▲ higher</button>
      <button data-d="fwd"   class="j">W radius+</button>
      <button data-d="down"  class="j">F ▼ lower</button>
      <button data-d="left"  class="j">A ↺ base</button>
      <button data-d="back"  class="j">S radius−</button>
      <button data-d="right" class="j">D ↻ base</button>
      <button data-d="roll_ccw" class="j">Q ↺ roll</button>
      <button data-d="roll_cw"  class="j">E ↻ roll</button>
      <button data-d="pitch_up" class="j">T ⤒ tilt up</button>
      <button data-d="pitch_dn" class="j">G ⤓ tilt dn</button>
      <button data-d="open"  class="j g">OPEN gripper</button>
      <button data-d="close" class="j g">CLOSE gripper</button>
    </div>
  </div>
  <div class="qbox">
    <p class="lbl">detection query (YOLO prompt)</p>
    <div class="qrow">
      <input id="query" type="text" placeholder="e.g. red cube, toy block">
      <button id="q-set" onclick="setQuery()">Set</button>
      <button id="q-run" onclick="runYoloApproach()">Run YOLO Approach</button>
    </div>
  </div>
  <div class="qbox">
    <p class="lbl">approach tuning · change while IDLE, takes effect on next Start</p>
    <div class="qrow">
      <label>aim px</label>
      <input id="tune-aim" type="number" step="5" value="-90" title="Lateral aim offset (px). More negative = aim further LEFT in image = robot moves RIGHT.">
      <button onclick="setTune('aim','/setaimdu?px=')">Set</button>
    </div>
    <div class="qrow">
      <label>right trim cm</label>
      <input id="tune-trim" type="number" step="0.5" value="3.0" title="Approach target shifted this many cm to the cube's RIGHT.">
      <button onclick="setTune('trim','/settrim?cm=')">Set</button>
    </div>
    <div class="qrow">
      <label>steps</label>
      <input id="tune-steps" type="number" step="1" value="4" title="Number of staged approach waypoints.">
      <button onclick="setTune('steps','/setsteps?n=')">Set</button>
    </div>
    <div class="qrow">
      <label>cube cm</label>
      <input id="tune-cube" type="number" step="0.5" value="5.08" title="Real cube edge in cm. Used for apparent-size range fallback.">
      <button onclick="setTune('cube','/setcubesize?cm=')">Set</button>
    </div>
    <div class="qrow">
      <label>range scale</label>
      <input id="tune-scale" type="number" step="0.05" value="1.0" title="Multiply all ranges by this. >1 spreads objects out; <1 compresses.">
      <button onclick="setTune('scale','/setrangescale?scale=')">Set</button>
    </div>
    <div class="qrow">
      <label>bearing deg</label>
      <input id="tune-bearing" type="number" step="5" value="0.0" title="Rotate the map left/right. Use when camera shows objects on opposite sides but map clusters them on one side.">
      <button onclick="setTune('bearing','/setbearing?deg=')">Set</button>
    </div>
  </div>
  <pre id="log"></pre>
</div></main><script>
// ---- self-contained robot + object 3D viewer (no deps) --------------------
(function(){
  const a = document.getElementById('r3d-link'); if(a) a.href = `http://${location.hostname}:9090`;
  const cv = document.getElementById('v3d'); if(!cv) return;
  const ctx = cv.getContext('2d');
  // az 0 = look straight down the robot's forward axis from behind. It was -1.05
  // (-60deg), an off-axis orbit that makes a straight base look turned and a
  // straight wrist look sideways. Drag still rotates; this is only the default.
  let az = 0.0, el = 0.62, geom = {links:[], ee:null, obj:null};
  // Turn the robot 30° left ON the table: yaw the robot + objects about the base
  // vertical (z) axis while the ground grid stays fixed. (Orbiting the camera via
  // `az` rotates the whole scene together, so the robot never turns on the table.)
  // YAW was 0.524 (30deg) to "turn the robot on the table". It rotates the robot
  // AND the object markers, so a cube straight ahead got drawn 30deg to the left
  // and the wrist looked twisted when wrist_roll was actually 0. That made the 3D
  // view disagree with the camera and the map. Back to 0 so the view is truthful.
  // Scene yaw = the base-zero error (shoulder_pan reads ~-14deg while the arm is
  // physically straight). Rotating the WHOLE scene keeps the robot and the object
  // markers consistent with each other; offsetting only the arm pulled them apart.
  const YAW = 0.246, YC = Math.cos(YAW), YS = Math.sin(YAW);
  function yz(p){ return [p[0]*YC - p[1]*YS, p[0]*YS + p[1]*YC, p[2]]; }
  function resize(){ const r = cv.getBoundingClientRect(); const dpr = window.devicePixelRatio||1;
    cv.width = Math.round(r.width*dpr); cv.height = Math.round(r.height*dpr); ctx.setTransform(dpr,0,0,dpr,0,0); }
  window.addEventListener('resize', resize); resize();
  // The REAL URDF meshes (so101_new_calib.urdf -> assets/*.stl), fetched once in
  // link-local coords; /geom streams a 4x4 per link. This used to draw a bare
  // polyline through the link ORIGINS, which is why the arm looked like a stick
  // figure and its size/reach couldn't be judged against the cube.
  let mesh = null;
  fetch('/urdf').then(r=>r.json()).then(d=>{
    mesh = (d.links||[]).map(L=>({name:L.name, v:Float32Array.from(L.v), f:Int32Array.from(L.f)}));
    if(!mesh.length) mesh = null;
    draw();
  }).catch(()=>{});
  const LINKCOL = {base_link:[122,134,152], shoulder_link:[100,140,200],
    upper_arm_link:[120,160,215], lower_arm_link:[100,140,200],
    wrist_link:[130,170,220], gripper_link:[190,200,214],
    moving_jaw_so101_v1_link:[225,232,241]};
  // rotate: base-frame point -> screen. z is up. also returns view depth.
  function proj(p){ const ca=Math.cos(az), sa=Math.sin(az), ce=Math.cos(el), se=Math.sin(el);
    const rx = -p[0]*sa + p[1]*ca;            // screen right
    const dep =  p[0]*ca + p[1]*sa;           // into-screen (before tilt)
    const uy =  p[2]*ce - dep*se;             // screen up
    const W = cv.clientWidth, H = cv.clientHeight, s = Math.min(W,H)/0.60;
    return [W*0.5 + s*rx, H*0.62 - s*uy, dep*ce + p[2]*se]; }
  function line(a,b,col,w){ const p=proj(a), q=proj(b); ctx.strokeStyle=col; ctx.lineWidth=w||1;
    ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]); ctx.stroke(); }
  function drawMeshes(){
    const xf = geom.xf || {}; const tris = [];
    // View direction (into the screen) in BASE coords — depth = dot(p, vd).
    const ca=Math.cos(az), sa=Math.sin(az), ce=Math.cos(el), se=Math.sin(el);
    // yaw the view direction WITH the geometry so backface culling stays correct
    const vd0x=ca*ce, vd0y=sa*ce; const vdx=vd0x*YC - vd0y*YS, vdy=vd0x*YS + vd0y*YC, vdz=se;
    for(const L of mesh){
      const T = xf[L.name]; if(!T) continue;
      const nv = L.v.length/3, sx=new Float64Array(nv), sy=new Float64Array(nv), sd=new Float64Array(nv);
      const bx=new Float64Array(nv), by=new Float64Array(nv), bz=new Float64Array(nv);
      for(let i=0;i<nv;i++){
        const x=L.v[3*i], y=L.v[3*i+1], z=L.v[3*i+2];
        const X0 = T[0]*x+T[1]*y+T[2]*z+T[3];      // row-major 4x4
        const Y0 = T[4]*x+T[5]*y+T[6]*z+T[7];
        const Z = T[8]*x+T[9]*y+T[10]*z+T[11];
        const X = X0*YC - Y0*YS, Y = X0*YS + Y0*YC;   // yaw about base z (turn on table)
        bx[i]=X; by[i]=Y; bz[i]=Z;
        const s = proj([X,Y,Z]); sx[i]=s[0]; sy[i]=s[1]; sd[i]=s[2];
      }
      const c = LINKCOL[L.name] || [140,160,190];
      for(let t=0;t<L.f.length;t+=3){
        const a=L.f[t], b=L.f[t+1], q=L.f[t+2];
        // backface/shade from the base-frame normal; cheap lambert
        const ux=bx[b]-bx[a], uy2=by[b]-by[a], uz=bz[b]-bz[a];
        const vx=bx[q]-bx[a], vy=by[q]-by[a], vz=bz[q]-bz[a];
        let nx=uy2*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy2*vx;
        const nl=Math.hypot(nx,ny,nz)||1; nx/=nl; ny/=nl; nz/=nl;
        // BACKFACE CULL. Winding is preserved through decimation now, so the
        // normal is a real outward normal. Drawing both sides (what it did before)
        // paints the INSIDE of the arm on top of the outside — that is what made
        // it look transparent/x-ray, not the triangle budget. Skip faces pointing
        // away from the viewer and only the outer skin remains.
        if(nx*vdx + ny*vdy + nz*vdz >= 0) continue;
        const lam = Math.max(0.34, Math.min(1, 0.42 + 0.58*(0.35*nx - 0.45*ny + 0.82*nz)));
        tris.push([(sd[a]+sd[b]+sd[q])/3, sx[a],sy[a],sx[b],sy[b],sx[q],sy[q],
                   `rgb(${Math.round(c[0]*lam)},${Math.round(c[1]*lam)},${Math.round(c[2]*lam)})`]);
      }
    }
    tris.sort((p,q)=>q[0]-p[0]);                 // painter's: far first
    ctx.lineJoin='round';
    for(const t of tris){
      ctx.fillStyle = t[7]; ctx.strokeStyle = t[7]; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(t[1],t[2]); ctx.lineTo(t[3],t[4]); ctx.lineTo(t[5],t[6]); ctx.closePath();
      ctx.fill();
      ctx.stroke();   // seal the sub-pixel seams between fills — un-stroked
                      // canvas triangles leave hairline gaps that show the dark
                      // background through and make the arm look transparent.
    }
  }
  function draw(){
    const W=cv.clientWidth, H=cv.clientHeight; ctx.clearRect(0,0,W,H);
    // ground grid at z=0
    const g=0.30, n=6; ctx.globalAlpha=0.5;
    for(let i=0;i<=n;i++){ const t=-g+2*g*i/n;
      line([t,-g,0],[t,g,0],'#243040',1); line([-g,t,0],[g,t,0],'#243040',1); }
    ctx.globalAlpha=1;
    // base axes (turn with the robot's base frame)
    line(yz([0,0,0]),yz([0.08,0,0]),'#c9524a',2); line(yz([0,0,0]),yz([0,0.08,0]),'#4a93c9',2); line(yz([0,0,0]),yz([0,0,0.08]),'#4ac275',2);
    if(mesh && geom.xf){
      drawMeshes();
    } else {
      // fallback: link-origin polyline (what this viewer used to be)
      const L=geom.links||[];
      for(let i=0;i+1<L.length;i++) line(yz(L[i]),yz(L[i+1]),'#7f9fd8',4);
      for(const p of L){ const s=proj(yz(p)); ctx.fillStyle='#b9c9ea'; ctx.beginPath(); ctx.arc(s[0],s[1],3.5,0,7); ctx.fill(); }
    }
    if(geom.ee){ const s=proj(yz(geom.ee)); ctx.fillStyle='#e6ecf1'; ctx.beginPath(); ctx.arc(s[0],s[1],4.5,0,7); ctx.fill(); }
    // object cube
    if(geom.obj){ const o=geom.obj, h=(geom.obj_size||0.03)/2;
      const c=[]; for(let dx of [-h,h]) for(let dy of [-h,h]) for(let dz of [-h,h]) c.push(yz([o[0]+dx,o[1]+dy,o[2]+dz]));
      const E=[[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]];
      const col = (geom.obj_label==='green')?'#3fc46b':'#e2574c';
      for(const e of E) line(c[e[0]],c[e[1]],col,2);
      const s=proj(yz(o)); ctx.fillStyle=col; ctx.font='11px ui-monospace,Consolas';
      ctx.fillText(`${(geom.obj_label||'obj')}  r=${Math.hypot(o[0],o[1]).toFixed(2)}m`, s[0]+8, s[1]-8); }
    // mapped objects (2D map) — each as a cube RESTING on the table (z: 0..size)
    const E2=[[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]];
    for(const o of (geom.objs2d||[])){ const sz=o.s||0.03, hh=sz/2;
      const cl=/green/.test(o.label)?'#3fc46b':/blue/.test(o.label)?'#4a93c9':/red/.test(o.label)?'#e2574c':'#e0b040';
      const c=[]; for(const dx of [-hh,hh]) for(const dy of [-hh,hh]) for(const dz of [0,sz]) c.push(yz([o.x+dx,o.y+dy,dz]));
      for(const e of E2) line(c[e[0]],c[e[1]],cl,1.5);
      const s=proj(yz([o.x,o.y,sz])); ctx.fillStyle=cl; ctx.font='10px ui-monospace,Consolas';
      ctx.fillText(`${o.label.split(' ')[0]}#${o.tag}`, s[0]+7, s[1]-6); }
  }
  let drag=null;
  cv.addEventListener('pointerdown', e=>{ drag=[e.clientX,e.clientY]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e=>{ if(!drag) return; az -= (e.clientX-drag[0])*0.01; el += (e.clientY-drag[1])*0.01;
    el=Math.max(-0.2,Math.min(1.5,el)); drag=[e.clientX,e.clientY]; draw(); });
  cv.addEventListener('pointerup', ()=>{ drag=null; });
  async function poll(){ try{ geom = await (await fetch('/geom')).json(); }catch(e){} draw(); }
  poll(); setInterval(poll, 180);
})();
function jmsg(t){ document.getElementById('jogmsg').textContent = t; }
async function setQuery(){
  const q = document.getElementById('query').value.trim();
  if(!q) return;
  const btn = document.getElementById('q-set'); btn.textContent = '…';
  try{ const r = await (await fetch('/setquery?q='+encodeURIComponent(q),{method:'POST'})).json();
    btn.textContent = r.ok ? 'Set ✓' : 'Set'; }
  catch(e){ btn.textContent = 'Set'; }
  setTimeout(()=>document.getElementById('q-set').textContent='Set', 1200);
}
document.getElementById('query').addEventListener('keydown', e => { if(e.key==='Enter') setQuery(); });
async function setTune(id, url){
  const el = document.getElementById('tune-'+id);
  const v = el.value.trim();
  if(v === '') return;
  const btn = el.nextElementSibling; btn.textContent = '…';
  try{
    const r = await (await fetch(url+encodeURIComponent(v),{method:'POST'})).json();
    btn.textContent = r.ok ? 'Set ✓' : 'Set';
    if(!r.ok) alert(r.reason || 'failed');
  }catch(e){ btn.textContent = 'Set'; alert(e.message); }
  setTimeout(()=>btn.textContent='Set', 1200);
}
async function runYoloApproach(){
  const q = document.getElementById('query').value.trim();
  if(!q) { alert('Enter a query first (e.g. "red cube")'); return; }
  const btn = document.getElementById('q-run'); btn.textContent = '⟳ running…'; btn.disabled = true;
  try{
    const r = await (await fetch('/yolo-approach?q='+encodeURIComponent(q),{method:'POST'})).json();
    btn.textContent = r.ok ? '✓ Done' : 'Run YOLO Approach';
    if(!r.ok) alert('Error: ' + (r.reason||'unknown'));
  }
  catch(e){ btn.textContent = 'Run YOLO Approach'; alert('Connection error: ' + e.message); }
  finally { btn.disabled = false; setTimeout(()=>document.getElementById('q-run').textContent='Run YOLO Approach', 2000); }
}
async function setPush(){
  const cm = parseFloat(document.getElementById('pushout').value);
  const btn = document.getElementById('p-set'); btn.textContent = '…';
  try{ await fetch('/pushout',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({cm})});
    await fetch('/relocate',{method:'POST'});   // re-locate so the 3D lock box moves now
    btn.textContent = 'Applied ✓'; }
  catch(e){ btn.textContent = 'Apply + re-locate'; }
  setTimeout(()=>{ document.getElementById('p-set').textContent='Apply + re-locate'; }, 1400);
}
async function press(d){ try{ const r = await (await fetch(`/jogpress?dir=${d}`,{method:'POST'})).json();
  if(!r.ok) jmsg('✗ '+r.reason); else if(r.grip) jmsg(r.grip); }catch(e){} }
async function release(d){ try{ await fetch(`/jogrelease?dir=${d}`,{method:'POST'}); }catch(e){} }
const HOLD = new Set(['fwd','back','left','right','up','down','roll_ccw','roll_cw','pitch_up','pitch_dn']);
// buttons: hold-to-move (pointer events cover mouse + touch)
document.querySelectorAll('.j').forEach(b => {
  const d = b.dataset.d;
  if(HOLD.has(d)){
    const go = e => { e.preventDefault(); press(d); };
    const stop = () => release(d);
    b.addEventListener('pointerdown', go);
    b.addEventListener('pointerup', stop);
    b.addEventListener('pointerleave', stop);
    b.addEventListener('pointercancel', stop);
  } else {
    b.addEventListener('click', () => press(d));   // open/close = one shot
  }
});
// ---- phone joysticks: proportional analog jog via /jogvec -----------------
const jv = {r:0, th:0, z:0};
let jvTimer = null;
function jvSend(){
  fetch(`/jogvec?r=${jv.r.toFixed(3)}&th=${jv.th.toFixed(3)}&z=${jv.z.toFixed(3)}`,{method:'POST'}).catch(()=>{});
}
function jvActive(){ return Math.abs(jv.r)>1e-3 || Math.abs(jv.th)>1e-3 || Math.abs(jv.z)>1e-3; }
function jvStart(){ if(!jvTimer) jvTimer = setInterval(jvSend, 60); }        // ~16 Hz
function jvMaybeStop(){ if(!jvActive() && jvTimer){ jvSend(); clearInterval(jvTimer); jvTimer=null; } }
function makeStick(id, vert, apply){
  const pad = document.getElementById(id); if(!pad) return;
  const nub = pad.querySelector('.nub');
  let on=false, cx=0, cy=0, R=1;
  const dz = v => Math.abs(v) < 0.09 ? 0 : v;                                // deadzone
  function begin(e){ on=true; const r=pad.getBoundingClientRect();
    cx=r.left+r.width/2; cy=r.top+r.height/2; R=r.width/2-22;
    pad.setPointerCapture(e.pointerId); jvStart(); move(e); e.preventDefault(); }
  function move(e){ if(!on) return; let dx=e.clientX-cx, dy=e.clientY-cy; if(vert) dx=0;
    const d=Math.min(Math.hypot(dx,dy),R), a=Math.atan2(dy,dx);
    const kx=vert?0:d*Math.cos(a), ky=d*Math.sin(a);
    nub.style.transform=`translate(${kx}px,${ky}px)`;
    apply(dz(kx/R), dz(ky/R)); e.preventDefault(); }
  function end(){ on=false; nub.style.transform='translate(0,0)'; apply(0,0); jvMaybeStop(); }
  pad.addEventListener('pointerdown', begin);
  pad.addEventListener('pointermove', move);
  pad.addEventListener('pointerup', end);
  pad.addEventListener('pointercancel', end);
}
makeStick('st-move', false, (x,y)=>{ jv.r=-y; jv.th=-x; });   // up = reach+, left = base CCW
makeStick('st-lift', true,  (x,y)=>{ jv.z=-y; });             // up = higher
// keyboard: keydown starts, keyup stops; guard OS key-repeat with a held set
const KEYMAP = {w:'fwd', s:'back', a:'left', d:'right', r:'up', f:'down',
  q:'roll_ccw', e:'roll_cw', t:'pitch_up', g:'pitch_dn'};
const down = new Set();
document.addEventListener('keydown', ev => {
  if(ev.target.tagName === 'INPUT') return;
  const k = ev.key.toLowerCase();
  if(KEYMAP[k]){ ev.preventDefault(); if(!down.has(k)){ down.add(k); press(KEYMAP[k]); } }
  else if(k === 'o'){ ev.preventDefault(); press('open'); }     // O = open
  else if(k === 'c'){ ev.preventDefault(); press('close'); }    // C = close
  else if(ev.key === ' '){ ev.preventDefault(); if(!down.has(' ')){ down.add(' ');
    press(window._gripOpen ? 'close':'open'); window._gripOpen = !window._gripOpen; } }
});
document.addEventListener('keyup', ev => {
  const k = ev.key.toLowerCase();
  if(KEYMAP[k]){ down.delete(k); release(KEYMAP[k]); }
  else if(ev.key === ' '){ down.delete(' '); }
});
window.addEventListener('blur', () => { down.clear(); release('all'); });  // safety
async function tick(){
  try{
    const s = await (await fetch('/status')).json();
    const ph = document.getElementById('phase');
    ph.textContent = s.phase; ph.className = 'phase' + (/ABORT|ERROR/.test(s.phase) ? ' bad':'');
    document.getElementById('detail').textContent = s.detail || '';
    const rows = [
      ['joints (deg)', (s.joints||[]).join('  ')],
      ['gripper %', s.gripper ?? '—'],
      ['servo loop', (s.loop_hz||0) + ' Hz'],
      ['dist to target', s.dist_mm != null ? s.dist_mm + ' mm' : '—'],
      ['gripper xyz (m)', s.jog_xyz ? s.jog_xyz.join(', ') : '—'],
      ['gripper angle', s.pitch != null ? s.pitch + '°  (0=flat, 90=down)' : '—'],
      ['located object (m)', s.located ? s.located.join(', ') : '—'],
      ['target radius/angle', s.target_polar ? (s.target_polar[0]+' cm  '+s.target_polar[1]+'°  h='+s.target_polar[2]+' cm') : '—'],
      ['red (base, m)', s.p_red ? s.p_red.join(', ') : '—'],
      ['green (base, m)', s.p_green ? s.p_green.join(', ') : '—'],
      ['elapsed', s.t0 && s.running ? Math.round(Date.now()/1000 - s.t0) + ' s' : '—'],
    ];
    document.getElementById('rows').innerHTML =
      rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('');
    document.getElementById('log').textContent = (s.log||[]).join('\\n');
    const qi = document.getElementById('query');
    if(s.query && !qi.value && document.activeElement !== qi) qi.value = s.query;
    if(s.tune){
      const ta = document.getElementById('tune-aim');
      if(ta && document.activeElement !== ta) ta.value = s.tune.aim_du;
      const tt = document.getElementById('tune-trim');
      if(tt && document.activeElement !== tt) tt.value = s.tune.right_trim_cm;
      const ts = document.getElementById('tune-steps');
      if(ts && document.activeElement !== ts) ts.value = s.tune.approach_steps;
      const tc = document.getElementById('tune-cube');
      if(tc && document.activeElement !== tc) tc.value = s.tune.cube_edge_cm;
      const tsc = document.getElementById('tune-scale');
      if(tsc && document.activeElement !== tsc) tsc.value = s.tune.range_scale;
      const tb = document.getElementById('tune-bearing');
      if(tb && document.activeElement !== tb) tb.value = s.tune.bearing_deg;
    }
  }catch(e){}
  setTimeout(tick, 700);
}
tick();

// ---- 2D top-down object map ----
(function(){
  const cv=document.getElementById('v2d'); if(!cv) return;
  const ctx=cv.getContext('2d'); let objs=[];
  function resize(){const r=cv.getBoundingClientRect(),dpr=devicePixelRatio||1;
    cv.width=Math.round(r.width*dpr);cv.height=Math.round(r.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);}
  addEventListener('resize',resize);resize();
  // base at bottom-centre; +x = up (forward), +y = left. scale: fit ~55cm radius.
  function W2S(x,y){const W=cv.clientWidth,H=cv.clientHeight,s=Math.min(W/1.4,(H-30)/0.80);
    return [W/2 - y*s, H-20 - x*s];}
  function col(l){return /red/.test(l)?'#e2574c':/green/.test(l)?'#3fc46b':/blue/.test(l)?'#4a93c9':'#e0b040';}
  function draw(){const W=cv.clientWidth,H=cv.clientHeight;ctx.clearRect(0,0,W,H);
    const s=Math.min(W/1.4,(H-30)/0.80),[ox,oy]=W2S(0,0);
    ctx.strokeStyle='#1b2733';ctx.fillStyle='#4a5a68';ctx.font='10px ui-monospace,Consolas';
    for(let r=10;r<=70;r+=10){ctx.beginPath();ctx.arc(ox,oy,r/100*s,Math.PI,2*Math.PI);ctx.stroke();
      ctx.fillText(r+'cm',ox+3,oy-r/100*s+11);}
    // reach limit ~42cm
    ctx.strokeStyle='#3a5a3a';ctx.beginPath();ctx.arc(ox,oy,0.62*s,Math.PI,2*Math.PI);ctx.stroke();
    ctx.fillStyle='#c9524a';ctx.beginPath();ctx.arc(ox,oy,5,0,7);ctx.fill();
    ctx.fillStyle='#8aa';ctx.fillText('base',ox+7,oy+4);
    for(const o of objs){const sz=o.size||0.03,hh=sz/2,c=col(o.label);
      // the object's GROUND FOOTPRINT: an axis-aligned square of side `size`
      const P=[W2S(o.x-hh,o.y-hh),W2S(o.x-hh,o.y+hh),W2S(o.x+hh,o.y+hh),W2S(o.x+hh,o.y-hh)];
      ctx.beginPath();ctx.moveTo(P[0][0],P[0][1]);for(let k=1;k<4;k++)ctx.lineTo(P[k][0],P[k][1]);ctx.closePath();
      ctx.globalAlpha=0.30;ctx.fillStyle=c;ctx.fill();ctx.globalAlpha=1;
      ctx.strokeStyle=c;ctx.lineWidth=2;ctx.stroke();
      const [sx,sy]=W2S(o.x,o.y);
      ctx.fillStyle=c;ctx.beginPath();ctx.arc(sx,sy,2,0,7);ctx.fill();
      ctx.fillStyle='#e6ecf1';ctx.font='11px ui-monospace,Consolas';
      ctx.fillText(o.label.split(' ')[0]+'#'+o.tag+' '+(sz*100).toFixed(0)+'cm',sx+9,sy-6);}
  }
  cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    let best=null,bd=22;for(const o of objs){const[sx,sy]=W2S(o.x,o.y);const d=Math.hypot(sx-mx,sy-my);
      if(d<bd){bd=d;best=o;}} if(best)fetch('/goto2d?tag='+best.tag,{method:'POST'});});
  async function poll(){try{const d=await(await fetch('/map2d')).json();objs=d.objs||[];draw();
    const tb=document.querySelector('#map2dtab tbody');
    tb.innerHTML=objs.map(o=>`<tr class="obj" style="cursor:pointer" onclick="fetch('/goto2d?tag=${o.tag}',{method:'POST'})">`+
      `<td>${o.label.split(' ')[0]}#${o.tag}</td><td>x=${(o.x*100).toFixed(0)}</td><td>y=${(o.y*100).toFixed(0)}</td>`+
      `<td>r=${o.r_cm}cm</td><td>stereo=${o.stereo_cm!=null?o.stereo_cm+'cm':'—'}</td><td>n=${o.n}</td></tr>`).join('')
      ||'<tr><td colspan=6 style="color:#8aa">empty — Scan → 2D map</td></tr>';
  }catch(e){} setTimeout(poll,500);}
  poll();
})();
</script></body></html>"""


@app.route("/")
def index():
    resp = Response(PAGE, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/status")
def status():
    with lock:
        s = {k: v for k, v in state.items()}
    s["log"] = list(log)
    s["tune"] = {
        "aim_du": round(AIM_DU, 1),
        "right_trim_cm": round(TARGET_RIGHT_TRIM_M * 100, 2),
        "approach_steps": APPROACH_STEPS,
        "cube_edge_cm": round(CUBE_EDGE_M * 100, 2),
        "range_scale": round(float(RANGE_SCALE[0]), 2),
        "bearing_deg": round(float(MAP_BEARING_OFFSET_DEG[0]), 1),
    }
    return jsonify(s)


def _decimate(V, F, voxel):
    """Voxel-cluster a dense STL down to something a browser can draw. Snaps verts
    to a `voxel`-sized grid, drops the triangles that collapse, dedupes. Keeps the
    true silhouette (and the gap between the jaws) — unlike a convex hull."""
    key = np.floor(V / voxel).astype(np.int64)
    _uniq, inv = np.unique(key, axis=0, return_inverse=True)
    n = len(_uniq)
    Vn = np.zeros((n, 3), np.float64)
    cnt = np.zeros(n, np.float64)
    np.add.at(Vn, inv, V)
    np.add.at(cnt, inv, 1.0)
    Vn /= np.maximum(cnt, 1.0)[:, None]
    Fn = inv[F]
    ok = (Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])
    Fn = Fn[ok]
    # Dedupe WITHOUT destroying winding. Sorting the 3 indices inside a face (the
    # obvious way to dedupe) scrambles its orientation, so half the normals end up
    # pointing inward and the shading goes random light/dark — that is what made
    # the robot look like transparent shattered glass. Rotate each face so its
    # smallest index leads: that is canonical for dedupe AND preserves cyclic order.
    roll = np.argmin(Fn, axis=1)
    idx = (np.arange(3)[None, :] + roll[:, None]) % 3
    Fn = np.unique(np.take_along_axis(Fn, idx, axis=1), axis=0)
    return Vn, Fn


# The moving jaw is NOT part of the FK chain to gripper_frame_link, so its pose
# must be composed by hand: URDF joint `gripper`, parent gripper_link,
# origin xyz="0.0202 0.0188 -0.0234" rpy="1.5708 0 0".
_c, _s = math.cos(1.5708), math.sin(1.5708)
JAW_T = np.array([[1, 0, 0, 0.0202],
                  [0, _c, -_s, 0.0188],
                  [0, _s, _c, -0.0234],
                  [0, 0, 0, 1.0]], dtype=np.float64)

_urdf_payload = [None]


@app.route("/urdf")
def urdf_route():
    """The ACTUAL lerobot URDF visual meshes (SO101/so101_new_calib.urdf ->
    assets/*.stl), decimated once and sent to the browser in LINK-LOCAL coords.
    /geom then streams a 4x4 per link, so the page draws the real robot instead
    of the stick figure it used to draw from bare link origins."""
    if _urdf_payload[0] is None:
        try:
            from lerobot.utils.urdf_visual_meshes import load_link_visual_meshes_cached
            meshes = load_link_visual_meshes_cached(kin.urdf_dir) or {}
            out = []
            for name, (V, F) in meshes.items():
                Vd, Fd = _decimate(np.asarray(V, np.float64), np.asarray(F, np.int64), 0.006)
                out.append({"name": name,
                            "v": [round(float(x), 4) for x in Vd.ravel()],
                            "f": [int(i) for i in Fd.ravel()]})
            _urdf_payload[0] = out
            say(f"URDF viewer meshes: {sum(len(l['f']) // 3 for l in out)} tris "
                f"across {len(out)} links")
        except Exception as e:
            say(f"URDF viewer meshes failed: {type(e).__name__}: {e}")
            _urdf_payload[0] = []
    return jsonify(links=_urdf_payload[0])


@app.route("/geom")
def geom():
    """Live 3D geometry: a 4x4 pose per URDF link (so the browser can pose the
    real meshes), plus the link origins (legacy stick-figure fallback), the EE,
    and the tracked object — all in the robot base frame."""
    with lock:
        jlist = state.get("joints")
        obj = state.get("obj3d")
        olbl = state.get("obj3d_label", "target")
    links, ee, xf = [], None, {}
    if jlist and kin is not None:
        try:
            q = np.array(jlist, dtype=np.float64)
            # The URDF's wrist_roll zero does not line up with the servo's zero:
            # the joint reads ~0 while the rendered gripper sits rotated. Offset the
            # rendered angle so the picture matches the real hand. DISPLAY ONLY -
            # IK/FK for actual motion are untouched.
            q[4] += WRIST_RENDER_OFFSET
            chain = kin.get_link_transforms_chain(q)
            links = [[round(float(T[0, 3]), 4), round(float(T[1, 3]), 4),
                      round(float(T[2, 3]), 4)] for _n, T in chain]
            for _n, T in chain:
                xf[_n] = [round(float(v), 5) for v in np.asarray(T, np.float64).ravel()]
            Tg = dict(chain).get("gripper_link")
            if Tg is not None:
                xf["moving_jaw_so101_v1_link"] = [
                    round(float(v), 5) for v in (np.asarray(Tg, np.float64) @ JAW_T).ravel()]
            Tee = np.asarray(kin.forward_kinematics(q))
            ee = [round(float(Tee[i, 3]), 4) for i in range(3)]
        except Exception:
            pass
    # every mapped object (2D map) as a table-resting cube for the 3D viewer
    objs2d = [{"x": o["x"], "y": o["y"], "s": o["size"], "label": o["label"], "tag": o["tag"]}
              for o in world2d_snapshot()]
    return jsonify(links=links, xf=xf, ee=ee, obj=obj, obj_label=olbl,
                   obj_size=round(float(CUBE_EDGE_M), 3), objs2d=objs2d)


@app.route("/stream")
def stream():
    def gen():
        while True:
            with lock:
                buf = frame_jpeg[0]
            if buf is not None:
                yield (b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(buf)).encode() + b"\r\n\r\n" + buf + b"\r\n")
            time.sleep(0.1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/stream2")
def stream2():
    """Proxy the camsurv room camera (session login + MJPEG passthrough)."""
    def gen():
        try:
            s = pyrequests.Session()
            s.post(CAMSURV[0] + "/", data={"password": CAMSURV[1]}, timeout=5)
            r = s.get(CAMSURV[0] + "/stream/1", stream=True, timeout=10)
            for chunk in r.iter_content(chunk_size=8192):
                yield chunk
        except Exception:
            return
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start", methods=["POST"])
def start():
    with lock:
        busy = state["running"]
    if not busy and (mission_thread[0] is None or not mission_thread[0].is_alive()):
        mission_thread[0] = threading.Thread(target=run_mission, daemon=True)
        mission_thread[0].start()
        return jsonify(ok=True)
    return jsonify(ok=False, reason="already running")


@app.route("/stop", methods=["POST"])
def stop():
    stop_flag.set()
    say("STOP requested")
    return jsonify(ok=True)


# ---- smooth continuous jog (browser teleop) -----------------------------
# Model matching the operator's mental picture:
#  * The gripper holds a FIXED angle (pitch + roll) while you translate; its
#    yaw simply follows the arm's reach direction (the only orientation DOF a
#    5-axis arm can't hold independently). This is why translating no longer
#    dumps the wrist down.
#  * W/S = reach out/in, A/D = strafe left/right  -> move in the xy plane
#  * R/F = up/down in z                            -> same gripper angle
#  * T/G = wrist PITCH (tilt the hand up/down)     -> a SEPARATE control
#  * Q/E = wrist ROLL                              -> a SEPARATE control
#  * space = gripper open/close (eased, no snap)
jog_held = set()
jog_held_lock = threading.Lock()
JOG_SPEED = 0.05           # m/s top radius/height speed
JOG_AZIM = 14.0           # deg/s top base-rotation (azimuth) speed — slow pan
JOG_ACC = 0.16            # velocity-EMA per tick (accel limiting -> no jerk)
JOG_DT = 0.05             # 20 Hz
JOG_WRIST = 45.0         # deg/s top wrist pitch/roll speed
GRIP_RATE = 28.0         # %/s — gripper eases slowly toward target (no snap)
JOG_DIRS = ("fwd", "back", "left", "right", "up", "down",
            "roll_cw", "roll_ccw", "pitch_up", "pitch_dn")
grip_target = [95.0]      # eased toward by the loop; set by open/close
grip_cmd = [50.0]         # current commanded gripper %, persistent

# Proportional analog jog from the phone joysticks (set by /jogvec, -1..1 per
# axis: r=reach/radius, th=base azimuth, z=height). Fresher than JOG_VEC_TTL
# means a stick is being held; it adds onto the held-key rates, clamped to the
# same top speeds. Guarded by jog_held_lock.
jog_vec = {"r": 0.0, "th": 0.0, "z": 0.0, "t": 0.0}
JOG_VEC_TTL = 0.30


# The SO-101 pitch joints (servo 2 lift, 3 elbow, 4 wrist_flex) are parallel,
# so the gripper's world pitch is EXACTLY their sum:  pitch = j1 + j2 + j3
# (0-indexed), independent of pan (j0) and roll (j4). Measured, constant = 0.
# => to hold the gripper angle we algebraically slave wrist_flex:
#        j3 = pitch_target - j1 - j2
# This is exact — no IK convergence needed — so the angle never drifts.
# THE REAL JOINT LIMITS, read from so101_new_calib.urdf. An IK that does not know
# these is not an IK, it is a wish: it returns elbow_flex=+162 deg on a joint that
# stops at +96.8, the servo silently clamps, the arm parks at the stop, and the
# solver reports a 0.2 mm residual on a pose the robot cannot hold. That is exactly
# what froze the approach for 21 identical hops at pitch 65 while being commanded
# to 80 (2026-07-13). CLAMP EVERY ITERATION AND SCORE THE CLAMPED POSE.
J_LO = np.array([-110.0, -100.0, -96.8, -95.0, -157.2])
J_HI = np.array([+110.0, +100.0, +96.8, +95.0, +162.8])
WFLEX_MIN, WFLEX_MAX = float(J_LO[3]), float(J_HI[3])   # wrist_flex safe range (deg)


def _gripper_pitch(T):
    z = T[:3, 2]
    return math.degrees(math.atan2(-z[2], math.hypot(z[0], z[1])))


def _slave_wflex(j1, j2, pitch_tgt):
    return float(np.clip(pitch_tgt - j1 - j2, WFLEX_MIN, WFLEX_MAX))


def _ik_hold_pitch(q_seed, p_tgt, pitch_tgt, j5_fixed, iters=80, tol=2e-3,
                   ret_err=False, _retry=True):
    """Position IK on servos 1-3 (pan, lift, elbow) with wrist_flex (servo 4)
    ALGEBRAICALLY SLAVED to hold the gripper pitch, and wrist_roll (servo 5) fixed.

    THIS SOLVER USED TO SILENTLY NOT CONVERGE, and that was the "the arm grabs at
    air / just moves out" bug (fixed 2026-07-13). It ran a FIXED 10 iterations with
    a +-4 deg/iter clamp -- a total travel budget of 40 deg -- while a perfectly
    ordinary reach like tip -> (0.15, 0, 0.02) needs 80-160 deg of elbow. Measured
    residual for that exact target with the old code: 107 mm at pitch 0, 93 mm at
    pitch 20, 35 mm at pitch 60 -- for a point THIS code hits to 0.2 mm. It returned
    a half-solved pose, goto_smooth faithfully drove to it, the next hop re-seeded
    from there, and the fingertip crept outward and UPWARD forever
    (`descend: tip_z +61mm -> +114mm` while being commanded DOWN to +15mm).

    Why 10 iterations: FK here costs 790 us and a fresh numeric Jacobian is 3 more
    FK, so 10 iters was already 32 ms -- near the 70 ms jog tick. Fix is to stop
    rebuilding J every step: over a <=3 cm step it barely rotates, so reuse it for
    8 iterations. That buys convergence AND is faster than before (16 ms worst case,
    9 ms for a jog-sized step).

    It now RETURNS THE RESIDUAL (ret_err=True). Callers MUST check it: a target the
    arm cannot reach is a fact to report, not a pose to drive to.
    """
    q = np.array(q_seed, dtype=np.float64)
    q[4] = float(np.clip(j5_fixed, J_LO[4], J_HI[4]))
    q[3] = _slave_wflex(q[1], q[2], pitch_tgt)
    J = None
    for it in range(iters):
        T = np.asarray(kin.forward_kinematics(q))
        err = p_tgt - T[:3, 3]
        if np.linalg.norm(err) < 3e-4:
            break
        if J is None or it % 8 == 0:
            J = np.empty((3, 3))
            for c, ji in enumerate((0, 1, 2)):
                dq = q.copy()
                dq[ji] = float(np.clip(dq[ji] + 0.5, J_LO[ji], J_HI[ji]))
                if ji in (1, 2):                      # keep pitch held while
                    dq[3] = _slave_wflex(dq[1], dq[2], pitch_tgt)   # perturbing
                J[:, c] = (np.asarray(kin.forward_kinematics(dq))[:3, 3] - T[:3, 3]) / 0.5
        dth = np.clip(J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), err), -8.0, 8.0)
        q[:3] = np.clip(q[:3] + dth, J_LO[:3], J_HI[:3])   # <-- STAY INSIDE THE ROBOT
        q[3] = _slave_wflex(q[1], q[2], pitch_tgt)
    # Score the CLAMPED pose, and only call the pitch "held" if wrist_flex did not
    # saturate -- otherwise we are reporting success on a pose the servos will not hold.
    e = float(np.linalg.norm(p_tgt - np.asarray(kin.forward_kinematics(q))[:3, 3]))
    if abs((q[1] + q[2] + q[3]) - pitch_tgt) > 2.0:
        e = max(e, 0.05)          # pitch could not be held here: treat as unreachable
    if e > tol and _retry:
        # Wrong IK branch. There is a genuine elbow-flip dead band (mapped 2026-07-13):
        # at r=10-15cm the arm simply cannot hold a shallow pitch at all. Re-seed.
        # The last two seeds extend the arm forward for far / shallow-pitch targets.
        for alt in ([q_seed[0], -95.0, 90.0, 30.0, j5_fixed],
                    [q_seed[0], -30.0, 50.0, 60.0, j5_fixed],
                    [q_seed[0], -60.0, 20.0, 80.0, j5_fixed],
                    [q_seed[0], -20.0, 75.0, 0.0, j5_fixed],
                    [q_seed[0], -10.0, 85.0, 0.0, j5_fixed]):
            q2, e2 = _ik_hold_pitch(np.array(alt, np.float64), p_tgt, pitch_tgt,
                                    j5_fixed, iters, tol, ret_err=True, _retry=False)
            if e2 < e:
                q, e = q2, e2
            if e <= tol:
                break
    return (q, e) if ret_err else q


def jog_loop():
    # CYLINDRICAL (polar) FPV control centred on the base — the arm's natural
    # coordinates. The camera rides steady on a level gimbal (pitch held); you
    # dolly it in/out and swing it around the base:
    #   W/S = radius  r  (dolly the view forward/back, same heading & height)
    #   A/D = azimuth θ  (rotate the base — pan the whole view around)
    #   R/F = height  z
    #   T/G = gimbal tilt   Q/E = roll   space = gripper
    vr_f = vth_f = vz_f = 0.0        # smoothed radius / azimuth / height rates
    roll_f = tilt_f = 0.0
    q_cmd = None
    r_tgt = th_tgt = z_tgt = None
    pitch_tgt = None
    j5_cmd = 0.0
    while True:
        t0 = time.time()
        with lock:
            busy = state["running"]
        with jog_held_lock:
            held = set(jog_held)
            va = dict(jog_vec)
        analog_fresh = (time.time() - va["t"]) < JOG_VEC_TTL
        analog_active = analog_fresh and (abs(va["r"]) + abs(va["th"]) + abs(va["z"]) > 1e-3)
        if busy:
            vr_f = vth_f = vz_f = roll_f = tilt_f = 0.0
            q_cmd = None; r_tgt = None
            time.sleep(0.05)
            continue
        moving = (held or analog_active or abs(vr_f) > 1e-4 or abs(vth_f) > 1e-4 or abs(vz_f) > 1e-4
                  or abs(roll_f) > 0.5 or abs(tilt_f) > 0.5)
        gripping = abs(grip_cmd[0] - grip_target[0]) > 0.5
        if not moving and not gripping:
            q_cmd = None; r_tgt = None
            time.sleep(0.03)
            continue
        try:
            joints, _rgb, obs = observe(overlay=True)
        except Exception:
            time.sleep(0.05)
            continue
        T_meas = np.asarray(kin.forward_kinematics(joints))
        p = T_meas[:3, 3]
        r_now = math.hypot(p[0], p[1])
        th_now = math.atan2(p[1], p[0])
        if q_cmd is None or float(np.max(np.abs(q_cmd - joints))) > 6.0:
            q_cmd = joints.copy()
            r_tgt, th_tgt, z_tgt = r_now, th_now, float(p[2])
            pitch_tgt = float(joints[1] + joints[2] + joints[3])   # gripper pitch
            j5_cmd = float(joints[4])
            gp = obs.get("gripper.pos")
            if isinstance(gp, (int, float)) and gp >= 0:
                grip_cmd[0] = float(gp)

        # polar rate commands (W/S radius, A/D azimuth, R/F height)
        vr_des  = (JOG_SPEED if "fwd" in held else 0.0)  - (JOG_SPEED if "back" in held else 0.0)
        vth_des = (JOG_AZIM  if "left" in held else 0.0) - (JOG_AZIM  if "right" in held else 0.0)
        vz_des  = (JOG_SPEED if "up" in held else 0.0)   - (JOG_SPEED if "down" in held else 0.0)
        if analog_fresh:                      # phone joysticks add proportionally
            vr_des  += JOG_SPEED * va["r"]
            vth_des += JOG_AZIM  * va["th"]
            vz_des  += JOG_SPEED * va["z"]
        vr_des  = float(np.clip(vr_des,  -JOG_SPEED, JOG_SPEED))
        vth_des = float(np.clip(vth_des, -JOG_AZIM,  JOG_AZIM))
        vz_des  = float(np.clip(vz_des,  -JOG_SPEED, JOG_SPEED))
        vr_f  = (1 - JOG_ACC) * vr_f  + JOG_ACC * vr_des
        vth_f = (1 - JOG_ACC) * vth_f + JOG_ACC * vth_des
        vz_f  = (1 - JOG_ACC) * vz_f  + JOG_ACC * vz_des
        tilt_des = (JOG_WRIST if "pitch_dn" in held else 0.0) - (JOG_WRIST if "pitch_up" in held else 0.0)
        roll_des = (JOG_WRIST if "roll_cw" in held else 0.0) - (JOG_WRIST if "roll_ccw" in held else 0.0)
        tilt_f = (1 - JOG_ACC) * tilt_f + JOG_ACC * tilt_des
        roll_f = (1 - JOG_ACC) * roll_f + JOG_ACC * roll_des
        pitch_tgt = float(np.clip(pitch_tgt + tilt_f * JOG_DT, -20.0, 100.0))
        j5_cmd += roll_f * JOG_DT
        grip_cmd[0] += float(np.clip(grip_target[0] - grip_cmd[0], -GRIP_RATE * JOG_DT, GRIP_RATE * JOG_DT))

        # integrate polar target as INDEPENDENT axes. Held axes must NOT chase
        # the measured value (that coupled radius into height) — instead each
        # target integrates freely and is only "leashed": we stop adding more
        # if the arm has fallen too far behind, but never drag the target.
        if abs(r_tgt - r_now) < 0.06:
            r_tgt += vr_f * JOG_DT
        if abs(th_tgt - th_now) < math.radians(20):
            th_tgt += math.radians(vth_f) * JOG_DT
        if abs(z_tgt - p[2]) < 0.06:
            z_tgt += vz_f * JOG_DT
        r_tgt = float(np.clip(r_tgt, 0.12, 0.42))       # reach limits
        z_tgt = max(z_tgt, -0.10)
        p_tgt = np.array([r_tgt * math.cos(th_tgt), r_tgt * math.sin(th_tgt), z_tgt])
        q_t = _ik_hold_pitch(q_cmd, p_tgt, pitch_tgt, j5_cmd)
        dq = np.clip(q_t - q_cmd, -JOINT_RATE_MAX * JOG_DT, JOINT_RATE_MAX * JOG_DT)
        q_cmd = q_cmd + dq
        q_cmd = joints + np.clip(q_cmd - joints, -3.0, 3.0)
        try:
            send_joints(q_cmd, gripper=grip_cmd[0])
            with lock:
                state["joints"] = [round(float(v), 1) for v in q_cmd]
                state["jog_xyz"] = [round(float(v), 3) for v in p_tgt]
                state["gripper"] = round(float(grip_cmd[0]), 1)
                state["pitch"] = round(float(pitch_tgt), 1)
        except Exception:
            pass
        time.sleep(max(0.0, JOG_DT - (time.time() - t0)))


@app.route("/jogpress", methods=["POST"])
def jogpress():
    with lock:
        busy = state["running"]
    if busy:
        return jsonify(ok=False, reason="mission running")
    d = request.args.get("dir", "")
    if d in ("open", "close"):
        grip_target[0] = 95.0 if d == "open" else 5.0   # loop eases toward it
        return jsonify(ok=True, grip=d)
    if d in JOG_DIRS:
        with jog_held_lock:
            jog_held.add(d)
        return jsonify(ok=True)
    return jsonify(ok=False, reason="bad dir")


@app.route("/jogrelease", methods=["POST"])
def jogrelease():
    d = request.args.get("dir", "")
    with jog_held_lock:
        if d == "all":
            jog_held.clear()
        else:
            jog_held.discard(d)
    return jsonify(ok=True)


@app.route("/jogvec", methods=["POST"])
def jogvec():
    """Proportional analog jog from the phone joysticks. Each axis is a signed
    fraction in [-1, 1]: r = reach/radius, th = base azimuth, z = height. The
    jog loop holds the gripper pitch level exactly as it does for the keys."""
    with lock:
        busy = state["running"]
    if busy:
        return jsonify(ok=False, reason="mission running")

    def _f(name):
        try:
            return max(-1.0, min(1.0, float(request.args.get(name, "0"))))
        except (TypeError, ValueError):
            return 0.0

    with jog_held_lock:
        jog_vec["r"] = _f("r")
        jog_vec["th"] = _f("th")
        jog_vec["z"] = _f("z")
        jog_vec["t"] = time.time()
    return jsonify(ok=True)


@app.route("/setquery", methods=["POST"])
def setquery():
    """Set the YOLO-World detection prompt live. Comma-separated synonyms help
    (e.g. 'red cube, toy block, red box')."""
    q = (request.args.get("q") or "").strip()
    if not q or detector is None:
        return jsonify(ok=False, reason="empty query or detector not ready")
    pending_query[0] = q        # applied by the YOLO thread within ~2.5 s
    with lock:
        state["query"] = q
    return jsonify(ok=True, query=q)


yolo_approach_running = [False]
yolo_approach_process = [None]


@app.route("/yolo-approach", methods=["POST"])
def yolo_approach():
    """Launch lerobot-yolo-track-approach with the given text query.
    Note: This requires stopping the main control loop first (close this UI).
    The YOLO approach will take full control of the robot via COM4."""
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify(ok=False, reason="empty query")

    if yolo_approach_running[0]:
        return jsonify(ok=False, reason="YOLO approach already running")

    def run_yolo():
        try:
            yolo_approach_running[0] = True
            say(f"Starting YOLO approach with query: {q}")
            # Run lerobot-yolo-track-approach with the query
            # NOTE: This will fail if COM4 is still in use by the main loop
            cmd = [
                "lerobot-yolo-track-approach",
                "--robot.type=so101_follower",
                "--robot.port=COM4",
                "--robot.cameras={\"front\": {\"type\": \"oakd\", \"fps\": 30, \"width\": 640, \"height\": 480, \"use_depth\": true}}",
                f"--query={q}",
                "--model-path=./yolov8s-worldv2.pt",
                "--camera-mount=gripper",
                "--camera-frame-convention=opencv",
                "--gripper-camera-tf=0.04,0,0.02,0,-0.35,0",
                "--target-from-gripper-tf=true",
                "--approach-style=plan_top",
                "--search-scan-enabled=true",
                "--depth-from-bbox-enabled=true",
                "--target-physical-size-m=0.03",
                "--depth-source-policy=bbox_preferred",
                "--top-approach-height-m=0.05",
                "--top-max-reach-m=0.35",
                "--table-z-m=-0.02",
                "--table-clearance-m=0.005",
                "--plan-top-tilt-fraction=0.0",
                "--plan-top-tilt-tolerance-deg=90",
                "--plan-top-final-hover-m=0.05",
                "--plan-top-gripper-tip-offset-m=0.02",
                "--plan-top-center-enable=true",
                "--smooth-center-alpha=0.25",
                "--plan-top-keep-in-frame-enable=true",
                "--plan-top-keep-in-frame-deadband-px=24",
                "--plan-top-keep-in-frame-kp=0.40",
                "--plan-top-keep-in-frame-max-step-deg=2.5",
                "--show-window=true",
                "--display-data=true",
                "--display-sim3d=true"
            ]
            proc = subprocess.Popen(cmd, cwd=LEROBOT)
            yolo_approach_process[0] = proc
            proc.wait()
            say("YOLO approach completed")
        except Exception as e:
            say(f"YOLO approach error: {e}")
        finally:
            yolo_approach_running[0] = False
            yolo_approach_process[0] = None

    threading.Thread(target=run_yolo, daemon=True).start()
    return jsonify(ok=True, reason="YOLO approach launched in background. Close this UI to free COM4 if it fails to connect.")


@app.route("/debugdepth", methods=["POST"])
def debugdepth():
    def _dbg():
        try:
            with bus_lock:
                d = cam.read_depth()
            if d is None:
                say("debugdepth: read_depth returned None"); return
            nz = int(np.count_nonzero(d))
            h, w = d.shape[:2]
            cen = d[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
            cen_valid = cen[cen > 0]
            say(f"depth {w}x{h} dtype={d.dtype} nonzero={nz}/{h*w} "
                f"({100*nz/(h*w):.0f}%) center_valid={cen_valid.size}/400 "
                f"center_med={'%.0f' % np.median(cen_valid) if cen_valid.size else 'NONE'}mm "
                f"range={int(d[d>0].min()) if nz else 0}-{int(d.max())}mm")
        except Exception as e:
            say(f"debugdepth error: {type(e).__name__}: {e}")
    threading.Thread(target=_dbg, daemon=True).start()
    return jsonify(ok=True)


@app.route("/pushout", methods=["POST"])
def pushout():
    """Live-set the radial localization correction (cm), no restart. See PUSH_OUT.
    POST {cm: N}. Re-locate to see it move the target in the 3D view."""
    try:
        cm = float((request.get_json(silent=True) or {}).get("cm", request.args.get("cm", 10)))
    except (TypeError, ValueError):
        return jsonify(ok=False, reason="need a number")
    cm = float(np.clip(cm, -5.0, 30.0))
    PUSH_OUT[0] = cm / 100.0
    say(f"localization push-out set to {cm:.0f}cm — re-locate to apply")
    return jsonify(ok=True, cm=cm)


@app.route("/setaimdu", methods=["POST"])
def setaimdu():
    """Live-tune the lateral aim offset (px). Negative shifts the aim point LEFT
    in the image, which makes the robot move RIGHT relative to the cube."""
    try:
        px = float(request.args.get("px", request.form.get("px", 0)))
    except (TypeError, ValueError):
        return jsonify(ok=False, reason="need a number")
    global AIM_DU
    AIM_DU = float(np.clip(px, -300.0, 300.0))
    say(f"AIM_DU set to {AIM_DU:.0f}px (left shift -> robot right)")
    return jsonify(ok=True, px=AIM_DU)


@app.route("/settrim", methods=["POST"])
def settrim():
    """Live-tune the approach rightward trim (cm). Positive shifts the staged
    approach target to the cube's right."""
    try:
        cm = float(request.args.get("cm", request.form.get("cm", 0)))
    except (TypeError, ValueError):
        return jsonify(ok=False, reason="need a number")
    global TARGET_RIGHT_TRIM_M
    TARGET_RIGHT_TRIM_M = float(np.clip(cm, -10.0, 15.0)) / 100.0
    say(f"approach right trim set to {TARGET_RIGHT_TRIM_M*100:.1f}cm")
    return jsonify(ok=True, cm=TARGET_RIGHT_TRIM_M*100)


@app.route("/setsteps", methods=["POST"])
def setsteps():
    """Live-tune the number of staged approach steps."""
    try:
        n = int(request.args.get("n", request.form.get("n", 4)))
    except (TypeError, ValueError):
        return jsonify(ok=False, reason="need an integer")
    global APPROACH_STEPS
    APPROACH_STEPS = int(np.clip(n, 1, 12))
    say(f"approach steps set to {APPROACH_STEPS}")
    return jsonify(ok=True, n=APPROACH_STEPS)


@app.route("/setcubesize", methods=["POST"])
def setcubesize():
    """Live-tune the assumed cube edge size (cm). Used by the 2D map apparent-size
    fallback. If the 3D lock / map cubes look too close/far or the wrong size,
    adjust this to the real cube edge."""
    try:
        cm = float(request.args.get("cm", request.form.get("cm", 5.08)))
    except (TypeError, ValueError):
        return jsonify(ok=False, reason="need a number")
    global CUBE_EDGE_M
    cm = float(np.clip(cm, 1.0, 30.0))
    CUBE_EDGE_M = cm / 100.0
    say(f"cube edge size set to {cm:.2f}cm")
    return jsonify(ok=True, cm=cm)


@app.route("/setrangescale", methods=["POST"])
def setrangescale():
    """Live-tune the global range scale. >1.0 pushes mapped objects further out
    (more spread); <1.0 pulls them closer. Use this if the whole map looks
    compressed or stretched."""
    try:
        scale = float(request.args.get("scale", request.form.get("scale", 1.0)))
    except (TypeError, ValueError):
        return jsonify(ok=False, reason="need a number")
    scale = float(np.clip(scale, 0.3, 3.0))
    RANGE_SCALE[0] = scale
    say(f"range scale set to {scale:.2f} — clear the 2D map and re-scan to see it")
    return jsonify(ok=True, scale=scale)


@app.route("/setbearing", methods=["POST"])
def setbearing():
    """Live-tune the map bearing offset (deg). Positive = rotate map CCW.
    Use this when the camera shows objects on opposite sides but the 2D map
    clusters them on one side (camera heading error)."""
    try:
        deg = float(request.args.get("deg", request.form.get("deg", 0.0)))
    except (TypeError, ValueError):
        return jsonify(ok=False, reason="need a number")
    deg = float(np.clip(deg, -180.0, 180.0))
    MAP_BEARING_OFFSET_DEG[0] = deg
    say(f"map bearing offset set to {deg:.1f}deg — clear the 2D map and re-scan to see it")
    return jsonify(ok=True, deg=deg)


@app.route("/relocate", methods=["POST"])
def relocate():
    """Locate the RED cube the SAME WAY the mission does (table-ray + push-out) WITHOUT
    moving the arm, and publish it as the 3D lock so the yellow FPV box and the 3D view
    both update. This is the honest localization check: if the yellow box is not on the
    cube, the number to change is the push-out (or the hand-eye)."""
    with lock:
        busy = state["running"]
    if busy:
        return jsonify(ok=False, reason="mission running — stop first")

    def _rl():
        try:
            stop_flag.clear()
            red_tracker.reset()
            if detect_now(find_red, tries=15) is None:
                set_phase("IDLE", "RED not in the gripper view — point the camera at it")
                return
            p = locate_on_table(find_red, red_tracker, "RED")
            with lock:
                state["obj3d"] = [float(v) for v in p]
                state["obj3d_label"] = "red"
                state["target_polar"] = [round(float(np.hypot(p[0], p[1])) * 100, 1),
                                         round(math.degrees(math.atan2(p[1], p[0])), 1),
                                         round(float(p[2]) * 100, 1)]
            set_phase("IDLE", "RED lock updated — check the yellow box sits on the cube")
        except Abort as e:
            set_phase("IDLE", f"could not locate ({e})")
        except Exception as e:
            set_phase("ERROR", f"relocate: {type(e).__name__}: {e}")

    threading.Thread(target=_rl, daemon=True).start()
    return jsonify(ok=True)


@app.route("/calibmount", methods=["POST"])
def calibmount():
    """Pin the camera mount by MULTI-VIEW CONSISTENCY: the same stationary cube must
    map to the same table spot from every viewing pose. Keep one cube visible and
    press once. See calibrate_mount_multiview."""
    with lock:
        busy = state["running"]
    if busy:
        return jsonify(ok=False, reason="mission running — stop first")

    def _cal():
        try:
            stop_flag.clear()
            with lock:
                state["running"] = True
            finder, tracker, label = _target_finder()
            tracker.reset()
            calibrate_mount_multiview(finder, label)
        except Abort as e:
            say(f"[CALIB FAILED] {e}")
            set_phase("IDLE", "mount unchanged")
        except Exception as e:
            say(f"[CALIB ERROR] {type(e).__name__}: {e}")
            set_phase("IDLE", "mount unchanged")
        finally:
            with lock:
                state["running"] = False

    threading.Thread(target=_cal, daemon=True).start()
    return jsonify(ok=True)


@app.route("/calib", methods=["POST"])
def calib():
    """Re-fit the hand-eye TF from the robot's own motion. Put the cube in the
    gripper view, press this once, done — it persists across restarts.
    Watch the FPV: the magenta circle (TF's idea of the fingertip) should snap
    onto the cyan cross (the real fingertip). See calibrate_handeye."""
    with lock:
        busy = state["running"]
    if busy:
        return jsonify(ok=False, reason="mission running — stop first")

    def _cal():
        try:
            stop_flag.clear()
            with lock:
                state["running"] = True
            red_tracker.reset()
            calibrate_handeye(find_red)
        except Abort as e:
            say(f"[CALIB FAILED] {e}")
            set_phase("IDLE", "hand-eye unchanged")
        except Exception as e:
            say(f"[CALIB ERROR] {type(e).__name__}: {e}")
            set_phase("IDLE", "hand-eye unchanged")
        finally:
            with lock:
                state["running"] = False

    threading.Thread(target=_cal, daemon=True).start()
    return jsonify(ok=True)


@app.route("/locate", methods=["POST"])
def locate():
    """Single-shot metric localization of the target using stereo depth: detect
    it, read the OAK-D depth at that pixel, back-project + FK -> base-frame 3D
    coordinate. Returns 'the G-code of the object'."""
    with lock:
        busy = state["running"]
    if busy:
        return jsonify(ok=False, reason="mission running — stop first")

    def _loc():
        try:
            stop_flag.clear()
            set_phase("LOCATE", "reading stereo depth")
            green_tracker.reset(); red_tracker.reset()
            # 1) fast single-shot stereo depth (works when the object is >~32cm).
            # Try RED first (the mission target); only fall back to green. This used
            # to localize green FIRST, which is why the target defaulted to green.
            p = locate_object(find_red, red_tracker, "red")
            if p is None:
                p = locate_object(find_green, green_tracker, "green")
            # 2) fallback: multi-vantage triangulation (works at close grasp
            #    range where the OAK-D stereo is blind). Move-and-intersect rays.
            if p is None:
                joints, rgb, _ = observe()
                T = T_cam_of(joints)
                if find_red(rgb, T) is not None:
                    fnd, trk, lbl = find_red, red_tracker, "red"
                else:
                    fnd, trk, lbl = find_green, green_tracker, "green"
                set_phase("LOCATE", "close range — triangulating (move + intersect)")
                trk.reset()
                try:
                    p = triangulate(fnd, trk, lbl)
                except Abort as e:
                    set_phase("IDLE", f"could not locate ({e})")
                    return
            with lock:
                state["p_red"] = [round(float(v), 3) for v in p]
                state["located"] = [round(float(v), 3) for v in p]
            set_phase("IDLE", f"located @ ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m")
        except Exception as e:
            set_phase("ERROR", f"locate: {e}")
    threading.Thread(target=_loc, daemon=True).start()
    return jsonify(ok=True)


@app.route("/caltip", methods=["POST"])
def caltip():
    """Measure where the black fingertips actually sit in the image right now.
    Descend to the grasp deck, close the fingers, find the dark blob, report
    its grip point. That pixel is the true HAND_UV to servo cubes onto."""
    with lock:
        busy = state["running"]
    if busy:
        return jsonify(ok=False, reason="mission running — stop first")

    def _cal():
        try:
            stop_flag.clear()
            set_phase("CAL TIP", "descending to grasp deck")
            # a nominal grasp pose over the table centre (measured deck poses)
            goto_smooth(np.array([-7.5, 49.0, 29.0, -58.0, -40.0]), settle=0.5)
            send_joints(observe()[0], gripper=15.0)   # close black fingers
            time.sleep(0.3)
            joints, rgb, _ = observe(overlay=False)
            img = np.ascontiguousarray(rgb)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            # dark = low value; the black fingertips are the darkest thing in
            # the lower-centre of the frame (the table is bright wood).
            dark = cv2.inRange(hsv, (0, 0, 0), (180, 90, 70))
            dark[: int(0.45 * dark.shape[0]), :] = 0   # ignore upper frame
            n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
            best, best_a = None, 0
            for k in range(1, n):
                a = stats[k, cv2.CC_STAT_AREA]
                if a > best_a and a > 800:
                    best_a, best = a, k
            if best is None:
                set_phase("CAL TIP", "no fingertip blob found — check lighting")
                return
            x, y, w, h, a = stats[best]
            # grip point: horizontal centre of the finger blob, near its TOP
            # (the fingertip tips point up toward the incoming object).
            gx = float(cent[best][0])
            gy = float(y + 0.15 * h)
            vis = img.copy()
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)
            cv2.drawMarker(vis, (int(gx), int(gy)), (0, 255, 0),
                           cv2.MARKER_CROSS, 26, 3)
            cv2.putText(vis, f"HAND_UV=({gx:.0f},{gy:.0f}) area={a}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            publish(vis[:, :, ::-1])
            cv2.imwrite(OUT + r"\fingertip_cal.jpg", vis[:, :, ::-1])
            with lock:
                state["hand_uv"] = [round(gx, 1), round(gy, 1)]
            global HAND_UV
            HAND_UV = (gx, gy)
            send_joints(observe()[0], gripper=95.0)
            set_phase("CAL TIP", f"fingertips at ({gx:.0f},{gy:.0f}) — HAND_UV updated")
        except Exception as e:
            set_phase("ERROR", f"caltip: {e}")
    threading.Thread(target=_cal, daemon=True).start()
    return jsonify(ok=True)


@app.route("/probe3d", methods=["POST"])
def probe3d():
    """Diagnostic: drive gripper_frame to the last triangulated GREEN point at
    a series of heights, photograph each, so we can SEE the finger-vs-cube
    geometry instead of guessing from pixel numbers."""
    with lock:
        busy = state["running"]
        pg = state.get("p_green")
    if busy or pg is None:
        return jsonify(ok=False, reason="need idle + a triangulated green")

    def _probe():
        try:
            stop_flag.clear()
            p = np.array(pg, float)
            send_joints(observe()[0], gripper=95.0)
            time.sleep(0.15)
            for hz in (0.09, 0.06, 0.04, 0.02, 0.005):
                set_phase("PROBE3D", f"gripper_frame -> green xy, z={hz:.3f}")
                q, err = ik_to_point(np.array([p[0], p[1], hz]), observe()[0])
                if err > 0.02:
                    say(f"probe z={hz:.3f}: unreachable (err {err*1e3:.0f}mm)")
                    continue
                goto_smooth(q, settle=0.35)
                joints, rgb, _ = observe(overlay=False)
                img = np.ascontiguousarray(rgb)
                # mark the calibrated fingertip pixel for reference
                cv2.drawMarker(img, (int(HAND_UV[0]), int(HAND_UV[1])),
                               (0, 255, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
                cv2.putText(img, f"gripper_frame z={hz:.3f}m", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imwrite(OUT + rf"\probe_{int(hz*1000):03d}.jpg", img[:, :, ::-1])
                publish(img[:, :, ::-1])
                time.sleep(0.12)
            goto_smooth(VIEW, settle=0.35)
            set_phase("PROBE3D", "done — images saved")
        except Exception as e:
            set_phase("ERROR", f"probe3d: {e}")
    threading.Thread(target=_probe, daemon=True).start()
    return jsonify(ok=True)


@app.route("/reset", methods=["POST"])
def reset_pose():
    """Drive to the VIEW working pose — a good, singularity-free spot to jog
    from (folded/retracted poses make the IK misbehave)."""
    with lock:
        busy = state["running"]
    if busy:
        return jsonify(ok=False, reason="mission running — stop first")
    with jog_held_lock:
        jog_held.clear()

    def _rst():
        try:
            stop_flag.clear()
            set_phase("RESET", "moving to working pose")
            goto_smooth(VIEW, settle=0.4)
            send_joints(observe()[0], gripper=95.0)
            set_phase("IDLE", "at working pose — ready to jog")
        except Exception as e:
            set_phase("ERROR", str(e))
    threading.Thread(target=_rst, daemon=True).start()
    return jsonify(ok=True)


@app.route("/home", methods=["POST"])
def home():
    with lock:
        busy = state["running"]
    if busy:
        return jsonify(ok=False, reason="mission running — stop first")

    def _fold():
        try:
            stop_flag.clear()
            set_phase("FOLD HOME")
            goto_smooth(HOME, settle=0.5)
            set_phase("IDLE", "folded")
        except Exception as e:
            set_phase("ERROR", str(e))
    threading.Thread(target=_fold, daemon=True).start()
    return jsonify(ok=True)


# ---- 2D BEV map routes ----
@app.route("/map2d")
def r_map2d():
    return jsonify(objs=world2d_snapshot())


@app.route("/scan2d", methods=["POST"])
def r_scan2d():
    with lock:
        if state["running"]:
            return jsonify(ok=False, reason="busy")

    def _t():
        stop_flag.clear()
        with lock:
            state["running"] = True
        try:
            scan_2d()
        except Abort as e:
            set_phase("ABORTED", str(e))
        except Exception as e:
            set_phase("ERROR", f"{type(e).__name__}: {e}")
        finally:
            with lock:
                state["running"] = False
    threading.Thread(target=_t, daemon=True).start()
    return jsonify(ok=True)


@app.route("/goto2d", methods=["POST"])
def r_goto2d():
    tag = int(request.args.get("tag", "0"))
    with lock:
        if state["running"]:
            return jsonify(ok=False, reason="busy")

    def _t():
        stop_flag.clear()
        with lock:
            state["running"] = True
        try:
            goto_2d(tag)
        except Abort as e:
            set_phase("ABORTED", str(e))
        except Exception as e:
            set_phase("ERROR", f"{type(e).__name__}: {e}")
        finally:
            with lock:
                state["running"] = False
    threading.Thread(target=_t, daemon=True).start()
    return jsonify(ok=True)


@app.route("/clearmap2d", methods=["POST"])
def r_clearmap2d():
    global W2D
    with w2d_lock:
        W2D = {"objs": {}, "next": 1}
    say("2D map cleared")
    return jsonify(ok=True)


def idle_view():
    while True:
        with lock:
            busy = state["running"]
        with jog_held_lock:
            jogging = bool(jog_held)
        if not busy and not jogging:   # jog_loop owns the camera while jogging
            try:
                joints, rgb, _ = observe()
                sense_2d(joints, rgb)   # keep the 2D map fresh while idle
            except Exception:
                time.sleep(1.0)
        time.sleep(0.25)


def clear_gripper_overload():
    """The gripper servo latches overload protection easily (worn gears). Clear
    it with a raw torque cycle before lerobot's handshake reads hit the error."""
    try:
        import scservo_sdk as scs
        ph = scs.PortHandler("COM4")
        if not ph.openPort():
            return
        ph.setBaudRate(1000000)
        pk = scs.PacketHandler(0)
        pk.write1ByteTxRx(ph, 6, 40, 0)   # torque off
        time.sleep(0.6)
        pk.write1ByteTxRx(ph, 6, 40, 1)   # torque on (clears latched error)
        time.sleep(0.2)
        _pos, _c, err = pk.read2ByteTxRx(ph, 6, 56)
        ph.closePort()
        say(f"gripper overload cleared (status {err:#04x})")
    except Exception as e:
        say(f"gripper overload clear skipped: {e}")


# ---------------- Rerun 3D (robot mesh + object) served to the browser ----------------
RERUN_WEB_PORT = 9090
RERUN_GRPC_PORT = 9877
HOST_IP = "100.110.89.78"          # Tailscale IP the remote browser reaches
_rerun_ok = [False]


def start_rerun():
    """Serve the lerobot Rerun web viewer from this process. The viewer is a
    wasm app that runs in the REMOTE browser, so it must be told to connect to
    the gRPC data server at the host's reachable IP (not localhost)."""
    try:
        import rerun as rr
        rr.init("rax_mission", spawn=False)
        rr.serve_grpc(grpc_port=RERUN_GRPC_PORT)
        rr.serve_web_viewer(
            open_browser=False, web_port=RERUN_WEB_PORT,
            connect_to=f"rerun+http://{HOST_IP}:{RERUN_GRPC_PORT}/proxy")
        _rerun_ok[0] = True
        say(f"Rerun 3D viewer: http://{HOST_IP}:{RERUN_WEB_PORT}")
    except Exception as e:
        say(f"Rerun 3D disabled: {e}")


def rerun_thread():
    """Stream the robot pose (URDF meshes) + the tracked object cube into Rerun
    at ~8 Hz. Joints come from the shared state cache (updated by whichever loop
    owns the camera), so this never contends for the OAK-D."""
    if not _rerun_ok[0]:
        return
    try:
        from lerobot.utils.manipulation_sim3d import log_manipulation_sim3d
    except Exception as e:
        say(f"Rerun 3D disabled (sim3d import): {e}")
        return

    # URDF MESH SELF-CHECK. log_manipulation_sim3d silently falls back to a blue
    # stick-figure (`sim3d/robot/arm_chain`, LineStrips3D) whenever no chain link
    # matches a loaded mesh — and this thread used to swallow every exception with
    # a bare `except: pass`, so a mesh failure was invisible. Say it out loud once.
    try:
        from lerobot.utils.urdf_visual_meshes import load_link_visual_meshes_cached
        _meshes = load_link_visual_meshes_cached(kin.urdf_dir) or {}
        _chain = [n for n, _ in (kin.get_link_transforms_chain(np.zeros(len(ARM_MOTORS))) or [])]
        _hit = [n for n in _chain if n in _meshes]
        if _hit:
            say(f"Rerun 3D: URDF meshes OK — {len(_hit)}/{len(_chain)} links "
                f"({kin.urdf_dir}): {', '.join(_hit)}")
        else:
            say(f"Rerun 3D: NO URDF meshes matched — falling back to stick figure. "
                f"urdf_dir={kin.urdf_dir} meshes={list(_meshes)} chain={_chain}")
    except Exception as e:
        say(f"Rerun 3D: mesh self-check failed: {type(e).__name__}: {e}")

    frame = 0
    warned = [False]
    while True:
        try:
            with lock:
                jlist = state.get("joints")
                obj = state.get("obj3d")
                olbl = state.get("obj3d_label", "target")
            if jlist:
                joints = np.array(jlist, dtype=np.float64)
                centers = half = labels = None
                if obj is not None:
                    centers = np.array([obj], dtype=np.float64)
                    half = np.array([[0.015, 0.015, 0.015]], dtype=np.float64)
                    labels = [olbl]
                log_manipulation_sim3d(
                    frame_sequence=frame, kinematics=kin, joint_deg=joints,
                    object_centers_base=centers, object_half_sizes_base=half,
                    object_labels=labels,
                    focus_object_index=0 if centers is not None else None,
                    ground_plane_z_m=0.0)
                frame += 1
        except Exception as e:
            if not warned[0]:                      # was `pass` — that hid the bug
                warned[0] = True
                say(f"Rerun 3D log error: {type(e).__name__}: {e}")
        time.sleep(0.12)


def main():
    global robot, kin, fx, fy, cx0, cy0, detector, cam
    clear_gripper_overload()
    say("connecting robot + camera…")
    # The gripper servo (ID 6) answers intermittently — a marginal cable. Retry
    # the whole handshake with a FRESH robot object each time (a failed connect
    # leaves the bus in a half-open state that refuses reconnection).
    for attempt in range(6):
        robot = make_robot_from_config(SO101FollowerConfig(
            port="COM4", id="so101_follower",
            # DEPTH OFF. The pick locates the cube purely geometrically (cast its
            # pixel ray onto the table plane), so stereo depth buys us nothing — and
            # the stereo pipeline is what kept crashing the OAK-D mid-run
            # (X_LINK_ERROR + firmware crash dump, taking the whole server with it).
            # Dropping it also roughly halves the USB bandwidth. read_depth_m()
            # degrades gracefully to None.
            cameras={"front": OAKDCameraConfig(
                fps=30, width=640, height=480, use_depth=False)},
        ))
        try:
            robot.connect()
            break
        except RuntimeError as e:
            if "Missing motor" not in str(e) or attempt == 5:
                raise
            say(f"motor handshake incomplete (attempt {attempt + 1}/6) — retrying…")
            try:
                robot.bus.port_handler.closePort()
            except Exception:
                pass
            time.sleep(2.0)
    kin = RobotKinematics(LEROBOT + r"\SO101\so101_new_calib.urdf", "gripper_frame_link", ARM_MOTORS)
    cam = robot.cameras["front"]
    say("colour stream only (depth OFF) — the pick works by eye, not by stereo")
    intr = {"fx": 517.0, "fy": 517.0, "cx": 329.5, "cy": 231.4}
    if hasattr(cam, "get_depth_intrinsics"):
        try:
            intr = dict(cam.get_depth_intrinsics())
        except Exception:
            pass
    fx, fy, cx0, cy0 = (float(intr[k]) for k in ("fx", "fy", "cx", "cy"))

    # A hand-eye TF we FITTED (see calibrate_handeye) overrides the constant.
    _tfd = load_tf_override()
    if _tfd:
        say(f"hand-eye: using CALIBRATED TF from {_tfd.get('fitted','?')} "
            f"(reprojection {_tfd.get('rms_px',0):.0f}px) -> {_tfd['tf']}")
    # Sanity gate that costs nothing and would have caught this two days ago: the
    # camera is bolted to the gripper, so the fingertip has ONE fixed pixel, and we
    # measured it (HAND_UV). If the TF disagrees, every back-projected ray is wrong
    # and every range is wrong with it — say so loudly instead of quietly missing.
    try:
        _uv = tip_pixel(np.array(HOME, np.float64))
        _gap = 1e9 if _uv is None else math.hypot(_uv[0] - HAND_UV[0], _uv[1] - HAND_UV[1])
        if _gap > 40.0:
            say(f"*** HAND-EYE TF IS BAD: it puts the fingertip at "
                f"({_uv[0]:.0f},{_uv[1]:.0f}) but the fingers are really at "
                f"({HAND_UV[0]:.0f},{HAND_UV[1]:.0f}) — {_gap:.0f}px off. Every range "
                f"will be short. Put the cube in view and press CALIB. ***")
        else:
            say(f"hand-eye check: fingertip reprojects {_gap:.0f}px from HAND_UV — OK")
    except Exception as e:
        say(f"hand-eye check skipped: {type(e).__name__}: {e}")

    say("loading YOLO (parallel validation thread)…")
    detector = YoloWorldDetector(LEROBOT + r"\yolov8s-worldv2.pt", conf=0.10, imgsz=320,
                                 color_filter_min_frac=0.15)
    _q0 = "red cube, green cube"      # colour-ONLY: colourless terms like "toy
                                      # block"/"box" make YOLO-World fire on the
                                      # green cube and the mission then dives to it
    detector.set_query(_q0)
    with lock:
        state["query"] = _q0
    threading.Thread(target=yolo_worker, daemon=True).start()
    threading.Thread(target=idle_view, daemon=True).start()
    threading.Thread(target=jog_loop, daemon=True).start()
    start_rerun()
    threading.Thread(target=rerun_thread, daemon=True).start()
    set_phase("IDLE", "ready — press Start")
    say(f"UI: http://100.110.89.78:{PORT}  (Tailscale)")
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
