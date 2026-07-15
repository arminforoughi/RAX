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
import json, math, os, sys, tempfile, threading, time
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
VIEW = np.array([-6.7, 37.1, 48.1, -40.4, -28.4])
HOME = np.array([-9.67, -102.022, 98.066, 32.879, 0.0])
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
ACC_ALPHA = 0.22          # velocity EMA (lower = smoother, more damped)
K_LAT = 0.45              # lateral P gain (0.9 caused left-right hunting)
DEADBAND_PX = 18          # no lateral correction inside this — kills the limit cycle
JOINT_RATE_MAX = 35.0     # deg/s per joint hard clamp

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
            cv2.putText(img, name, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
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


def goto_smooth(target, settle=0.3, step=2.0):
    """Transit move (non-servo phases)."""
    joints, _r, obs = observe(overlay=False)
    gp = float(obs.get("gripper.pos", 50.0))
    with bus_lock:
        send_joint_target_smoothly(robot, ARM_MOTORS, joints, np.asarray(target, np.float64),
                                   step_deg=step, sleep_s=0.03,
                                   gripper_open=gp >= 50.0, gripper_width_pct=gp)
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
            hist.clear()
            stall_cool = now + 2.0
            time.sleep(0.35)
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
        if abs(pan_f) > 0.05:
            q_cmd[0] += float(np.clip(pan_f, -30.0, 30.0)) * LOOP_DT
            moved = True
        if moved:
            # anti-windup: if the arm can't follow (obstacle, limit), don't let
            # the command grind into it.
            q_cmd = joints + np.clip(q_cmd - joints, -2.5, 2.5)
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
        goto_smooth(q, settle=0.7)
        for _ in range(3):
            joints, rgb, _ = observe()
            tr = finder(rgb, T_cam_of(joints))
            if tr is not None:
                say(f"{label} found: uv=({tr.uv[0]:.0f},{tr.uv[1]:.0f}) area={tr.area_px}")
                return
            time.sleep(0.25)
    raise Abort(f"{label} cube not found in sweep")


def triangulate(finder, tracker, label):
    set_phase(f"TRIANGULATE {label}")
    rays = []
    base = observe()[0]
    for dq in (np.zeros(5), np.array([+8, 0, 0, +3, 0]), np.array([-8, -5, +4, +3, 0]),
               np.array([0, -8, +6, +4, 0]), np.array([+6, +5, -4, -3, 0])):
        if len(rays) >= 4:
            break
        goto_smooth(base + dq, settle=1.1)
        tr = None
        for _ in range(3):
            joints, rgb, _ = observe()
            tr = finder(rgb, T_cam_of(joints))
            if tr is not None and not tr.clipped:
                break
            time.sleep(0.35)
        if tr is None or tr.clipped:
            say(f"{label} vantage {dq}: unusable — skipped")
            continue
        T = T_cam_of(joints)
        d_cam = np.array([(tr.uv[0] - cx0) / fx, (tr.uv[1] - cy0) / fy, 1.0])
        d = T[:3, :3] @ (d_cam / np.linalg.norm(d_cam))
        rays.append((T[:3, 3], d))
    goto_smooth(base, settle=0.6)
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


def close_with_current(step=4.0, delay=0.1):
    """Close the gripper in small increments, watching the servo current, and
    stop the instant it rises (torque change = fingers on the object). Smaller
    step / longer delay = the slow, gentle close the user asked for."""
    idle = [c for c in (gripper_current() for _ in range(10)) if c is not None]
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
            time.sleep(0.4)
            return True, i_idle
    # Closed on air: DO NOT stay stalled shut (that's what tripped the servo's
    # overload protection earlier) — relax to a neutral opening.
    send_joints(observe(overlay=False)[0], gripper=40.0)
    time.sleep(0.4)
    return False, i_idle


def ee_move_rel(d_base, step=1.4, settle=0.5):
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
PUSH_OUT = [0.10]


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

    if rms(sol.x) > 25.0 or tipgap(sol.x) > 40.0:
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
GRASP_PITCH = (75.0, 80.0, 70.0, 85.0, 90.0, 65.0, 60.0, 55.0)


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
        if worst < 0.004:
            return pitch, worst
        if best is None or worst < best[1]:
            best = (pitch, worst)
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
    time.sleep(0.35)

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
        goto_smooth(q_tgt, settle=0.12, step=1.5)
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
    time.sleep(0.35)

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
        goto_smooth(q_tgt, settle=0.10, step=1.0)
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
        time.sleep(0.05)
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
def run_mission():
    try:
        stop_flag.clear()
        with lock:
            state["running"] = True
            state["t0"] = time.time()
        # ---- LOOK (first-person view only): do NOT move first. If the cube is
        #      already in the gripper view, go straight to gaze-center+approach.
        #      Only slow-scan (and, if truly blind, raise to a table view) when
        #      it isn't visible. No blind pan-sweep, no pre-dive. ----
        set_phase("LOOK", "checking the gripper view for the RED cube")
        red_tracker.reset()
        if detect_now(find_red, tries=12) is None:
            set_phase("SEARCH", "not in view — raising to a table view")
            goto_smooth(VIEW, settle=0.6)
            if detect_now(find_red, tries=8) is None:
                gentle_scan(find_red, "RED")

        # ---- APPROACH: gaze centers it (points the camera at it) AND closes
        #      the radius in one continuous loop. GRASP with a slow torque-sensed
        #      close. Retry up to 3×. ----
        held_vis = False
        for attempt in range(3):
            if attempt:
                say(f"grasp retry {attempt + 1}/3")
                send_joints(observe()[0], gripper=95.0)
                time.sleep(0.4)
                ee_move_rel([0, 0, 0.05], settle=0.5)   # small lift to re-see
                red_tracker.reset()
                if detect_now(find_red, tries=10) is None:
                    gentle_scan(find_red, "RED")
            # LOCK the cube from a distance (where the detector is honest), fly the
            # fingertip to a waypoint ABOVE it with the hand pitched DOWN (so the wrist
            # has somewhere legal to be), then descend straight onto the locked point.
            p_obj, pitch = approach_over_the_top(find_red, red_tracker, "RED")
            if not descend_and_close(p_obj, pitch, "RED"):
                say("descend did not reach the cube — closing anyway to see what happens")
            set_phase("GRASP", "slow torque-sensed close")
            contact, i_idle = close_with_current(step=3.0, delay=0.16)
            say(f"close: contact={contact}")

            set_phase("LIFT")
            ee_move_rel([0, 0, 0.10], settle=0.6)
            hold = [abs(c - i_idle) for c in (gripper_current() for _ in range(12)) if c is not None]
            hold_di = float(np.mean(hold)) if hold else 0.0
            joints, rgb, _ = observe()
            trh = find_red(rgb, T_cam_of(joints))
            held_vis = (trh is not None and trh.area_px > 6000
                        and math.hypot(trh.uv[0] - HAND_UV[0], trh.uv[1] - HAND_UV[1]) < 110)
            say(f"lifted: hold ΔI={hold_di:.1f} in-hand-visual={held_vis}")
            if held_vis or hold_di >= 3.0:
                break
        else:
            raise Abort("grasp failed after 3 attempts — red never held")

        set_phase("DONE", "RED cube picked and lifted")
    except Abort as e:
        set_phase("ABORTED", str(e))
        try:
            goto_smooth(VIEW, settle=0.5)
        except Exception:
            pass
    except Exception as e:
        set_phase("ERROR", f"{type(e).__name__}: {e}")
    finally:
        # Never park the gripper stalled: if it isn't holding anything at the
        # end (aborted / missed), relax it so the servo doesn't overload.
        try:
            g = None
            with lock:
                g = state.get("gripper")
            if g is not None and g < 15.0:
                hold = [abs(c) for c in (gripper_current(),) if c is not None]
                # keep the grip only if it's actually carrying load
                if not hold or hold[0] < 4.0:
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
.qrow{display:flex;gap:8px;margin-top:6px}
.qrow input{flex:1;padding:10px;background:#10161c;border:1px solid #37454f;border-radius:6px;
color:#e6ecf1;font:14px "Segoe UI"}
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
    <button id="b-home" onclick="fetch('/locate',{method:'POST'})">Locate object</button>
    <button id="b-home" onclick="fetch('/caltip',{method:'POST'})">Calibrate fingertip</button>
    <button id="b-calib" onclick="fetch('/calib',{method:'POST'})"
      title="Put the cube in the gripper view, then press. Fits the gripper→camera
transform from the robot's own motion (~30s). Watch the FPV: the magenta circle
should snap onto the cyan cross. Persists across restarts.">Calibrate HAND-EYE</button>
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
    </div>
  </div>
  <div class="qbox">
    <p class="lbl">push target outward (cm) — the camera sits behind the gripper, so
      the cube reads too close. Raise until the yellow "3D lock" box sits on the cube.</p>
    <div class="qrow">
      <input id="pushout" type="number" value="10" step="1" min="-5" max="30" style="width:5em">
      <button id="p-set" onclick="setPush()">Apply + re-locate</button>
    </div>
  </div>
  <pre id="log"></pre>
</div></main><script>
// ---- self-contained robot + object 3D viewer (no deps) --------------------
(function(){
  const a = document.getElementById('r3d-link'); if(a) a.href = `http://${location.hostname}:9090`;
  const cv = document.getElementById('v3d'); if(!cv) return;
  const ctx = cv.getContext('2d');
  let az = -1.05, el = 0.62, geom = {links:[], ee:null, obj:null};
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
    const vdx=ca*ce, vdy=sa*ce, vdz=se;
    for(const L of mesh){
      const T = xf[L.name]; if(!T) continue;
      const nv = L.v.length/3, sx=new Float64Array(nv), sy=new Float64Array(nv), sd=new Float64Array(nv);
      const bx=new Float64Array(nv), by=new Float64Array(nv), bz=new Float64Array(nv);
      for(let i=0;i<nv;i++){
        const x=L.v[3*i], y=L.v[3*i+1], z=L.v[3*i+2];
        const X = T[0]*x+T[1]*y+T[2]*z+T[3];      // row-major 4x4
        const Y = T[4]*x+T[5]*y+T[6]*z+T[7];
        const Z = T[8]*x+T[9]*y+T[10]*z+T[11];
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
    // base axes
    line([0,0,0],[0.08,0,0],'#c9524a',2); line([0,0,0],[0,0.08,0],'#4a93c9',2); line([0,0,0],[0,0,0.08],'#4ac275',2);
    if(mesh && geom.xf){
      drawMeshes();
    } else {
      // fallback: link-origin polyline (what this viewer used to be)
      const L=geom.links||[];
      for(let i=0;i+1<L.length;i++) line(L[i],L[i+1],'#7f9fd8',4);
      for(const p of L){ const s=proj(p); ctx.fillStyle='#b9c9ea'; ctx.beginPath(); ctx.arc(s[0],s[1],3.5,0,7); ctx.fill(); }
    }
    if(geom.ee){ const s=proj(geom.ee); ctx.fillStyle='#e6ecf1'; ctx.beginPath(); ctx.arc(s[0],s[1],4.5,0,7); ctx.fill(); }
    // object cube
    if(geom.obj){ const o=geom.obj, h=(geom.obj_size||0.03)/2;
      const c=[]; for(let dx of [-h,h]) for(let dy of [-h,h]) for(let dz of [-h,h]) c.push([o[0]+dx,o[1]+dy,o[2]+dz]);
      const E=[[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]];
      const col = (geom.obj_label==='green')?'#3fc46b':'#e2574c';
      for(const e of E) line(c[e[0]],c[e[1]],col,2);
      const s=proj(o); ctx.fillStyle=col; ctx.font='11px ui-monospace,Consolas';
      ctx.fillText(`${(geom.obj_label||'obj')}  r=${Math.hypot(o[0],o[1]).toFixed(2)}m`, s[0]+8, s[1]-8); }
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
  }catch(e){}
  setTimeout(tick, 700);
}
tick();
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
    return jsonify(links=links, xf=xf, ee=ee, obj=obj, obj_label=olbl, obj_size=0.03)


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
            r = s.get(CAMSURV[0] + "/stream/0", stream=True, timeout=10)
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
        for alt in ([q_seed[0], -95.0, 90.0, 30.0, j5_fixed],
                    [q_seed[0], -30.0, 50.0, 60.0, j5_fixed],
                    [q_seed[0], -60.0, 20.0, 80.0, j5_fixed]):
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
            # 1) fast single-shot stereo depth (works when the object is >~32cm)
            p = locate_object(find_green, green_tracker, "green")
            if p is None:
                p = locate_object(find_red, red_tracker, "red")
            # 2) fallback: multi-vantage triangulation (works at close grasp
            #    range where the OAK-D stereo is blind). Move-and-intersect rays.
            if p is None:
                joints, rgb, _ = observe()
                T = T_cam_of(joints)
                if find_green(rgb, T) is not None:
                    fnd, trk, lbl = find_green, green_tracker, "green"
                else:
                    fnd, trk, lbl = find_red, red_tracker, "red"
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
            goto_smooth(np.array([-7.5, 49.0, 29.0, -58.0, -40.0]), settle=1.2)
            send_joints(observe()[0], gripper=15.0)   # close black fingers
            time.sleep(0.8)
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
            time.sleep(0.4)
            for hz in (0.09, 0.06, 0.04, 0.02, 0.005):
                set_phase("PROBE3D", f"gripper_frame -> green xy, z={hz:.3f}")
                q, err = ik_to_point(np.array([p[0], p[1], hz]), observe()[0])
                if err > 0.02:
                    say(f"probe z={hz:.3f}: unreachable (err {err*1e3:.0f}mm)")
                    continue
                goto_smooth(q, settle=0.9)
                joints, rgb, _ = observe(overlay=False)
                img = np.ascontiguousarray(rgb)
                # mark the calibrated fingertip pixel for reference
                cv2.drawMarker(img, (int(HAND_UV[0]), int(HAND_UV[1])),
                               (0, 255, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
                cv2.putText(img, f"gripper_frame z={hz:.3f}m", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imwrite(OUT + rf"\probe_{int(hz*1000):03d}.jpg", img[:, :, ::-1])
                publish(img[:, :, ::-1])
                time.sleep(0.3)
            goto_smooth(VIEW, settle=0.8)
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
            goto_smooth(VIEW, settle=1.0)
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
            goto_smooth(HOME, settle=1.0)
            set_phase("IDLE", "folded")
        except Exception as e:
            set_phase("ERROR", str(e))
    threading.Thread(target=_fold, daemon=True).start()
    return jsonify(ok=True)


def idle_view():
    while True:
        with lock:
            busy = state["running"]
        with jog_held_lock:
            jogging = bool(jog_held)
        if not busy and not jogging:   # jog_loop owns the camera while jogging
            try:
                observe()
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
            cameras={"front": OAKDCameraConfig(
                fps=30, width=640, height=480, use_depth=True,
                stereo_extended_disparity=True,      # halves the min depth (~20cm)
                stereo_confidence_threshold=150)},    # looser -> more valid pixels
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
    say("stereo depth stream enabled — metric object localization active")
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
    _q0 = "red cube, toy block, red box"
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
