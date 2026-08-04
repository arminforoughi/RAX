# RAX robot-arm server: pick / place, an open-vocabulary object map, an admin UI
# and a public guest UI, all on one Flask port (:8484).
#
# ARCHITECTURE
#  * DETECT: YOLO-World (open vocabulary) runs in a PARALLEL thread (~2.5 s) so it
#    never sits in the control path; the two cube colours also have dedicated
#    strict-HSV trackers that are tighter during close approach. publish() draws
#    every detection on the FPV, not just the cubes.
#  * MAP: sense_2d folds detections into a 2D bird's-eye map. Range is measured
#    monocularly (the table IS the base plane) via measure_object, or falls back to
#    apparent size; elongated objects are ranged from their long axis. Entries are
#    merged by POSITION (one object fires under several labels), expire after
#    MAP_TTL_S, and only carry an orientation when one was actually measured.
#  * PICK: run_mission takes the mapped (x, y), approaches in 3 stages (step 1
#    closes ~90%, the rest are small corrections), centres by eye onto the
#    fingertip pixel (HAND_UV), descends and closes on the servo current.
#  * PLACE: place_at puts a held object on a mapped object or a clicked spot;
#    grip_to_bottom is measured at grasp, so release height needs only the
#    destination's height. Typed tasks ("green on red") via /task.
#  * SAFETY: the arm relaxes (torque off) after IDLE_RELAX_S and wakes on the next
#    motion; a latched servo overload is cleared and retried at connect. Run under
#    supervise.py, which restarts this and camserver if either dies.
#  * GUEST: a Host-header gate exposes only a small allowlist over the public
#    Tailscale-Funnel host; everything else is admin-only. See ui/*.html.
#
# NOTE the hand-eye TF (handeye_tf.json) is rotationally wrong — table rays come
# out too shallow — so absolute ranges are compressed and the map's positions are
# approximate. Most of the localization care in here works around that.
import json, math, os, re, secrets, subprocess, sys, tempfile, threading, time
import json, math, os, re, secrets, subprocess, sys, tempfile, threading, time
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
VIEW = np.array([5.0, 37.1, 48.1, -40.4, GRASP_ROLL])
# New home: captured from the physically-correct folded pose (2026-07-20).
HOME = np.array([-14.1, -99.1, 90.8, 33.2, -4.7])
HAND_UV = (440.0, 394.0)   # measured via /caltip against the real black fingertip
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
PORT = 8484
CAMSURV = ("http://127.0.0.1:5000", "camsurv123")

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
# Latest YOLO detections keyed by class label. Each value is a LIST of instances,
# {"xyxy": tuple, "conf": float}, confidence-sorted — a table can hold three cups
# and the map has to carry all three, so this cannot collapse to one box per label.
yolo_latest = {"t": 0.0, "dets": {}}
YOLO_NMS_IOU = 0.55        # merge duplicate boxes of the same label
pending_query = [None]      # set by /setquery, applied inside the YOLO thread


def _query_labels():
    """Parse the active detection query into class labels, preserving order."""
    q = (state.get("query") or "red cube, green cube").lower()
    return [p.strip() for p in q.split(",") if p.strip()]


# Minimum fraction of a box that must actually be the colour its LABEL names.
# Only labels containing a colour word are gated at all, so "pen" is never asked
# to be red while "red cube" still cannot latch onto the green one.
COLOR_MIN_FRAC = 0.15      # matches the original blanket filter for cubes


def _colour_ok(rgb, label, xyxy):
    try:
        from lerobot.perception.detection_filters import (
            bbox_color_match_fraction, color_names_in_query)
    except Exception:
        return True
    cols = color_names_in_query(label)
    if not cols:
        return True                       # colourless label: nothing to check
    box = tuple(int(v) for v in xyxy)
    try:
        return bbox_color_match_fraction(rgb, box, cols) >= COLOR_MIN_FRAC
    except Exception:
        return True


def _box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 1e-9 else 0.0


def _dedupe_boxes(insts):
    """Greedy NMS within one label — two prompts often fire on the same object."""
    kept = []
    for e in insts:
        if all(_box_iou(e["xyxy"], k["xyxy"]) < YOLO_NMS_IOU for k in kept):
            kept.append(e)
    return kept


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
            labels = _query_labels()
            by_label = {}
            for d in dets:
                if 0 <= d.class_id < len(labels):
                    lbl = labels[d.class_id]
                    if not _colour_ok(rgb, lbl, d.xyxy):
                        continue
                    by_label.setdefault(lbl, []).append(
                        {"xyxy": tuple(float(v) for v in d.xyxy), "conf": float(d.confidence)})
            for lbl, inst in by_label.items():
                inst.sort(key=lambda e: -e["conf"])
                by_label[lbl] = _dedupe_boxes(inst)
            yolo_latest["t"] = time.time()
            yolo_latest["dets"] = by_label


def find_red(rgb, T_base_cam=None):
    """Acquire via YOLO (parallel thread) OR a big strict-HSV blob (the S>=110
    gate already excludes wood grain, and YOLO misses edge-clipped slivers);
    afterwards window/anchor continuity tracks."""
    if red_tracker.last is not None or red_tracker.p_anchor is not None:
        tr = red_tracker.track(rgb, T_base_cam)
        if tr is not None:
            return tr
    box = None
    with lock:
        inst = yolo_latest["dets"].get("red cube")
        t = yolo_latest["t"]
    if inst:
        box = inst[0]["xyxy"]
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


def _track_from_box(rgb, xyxy, t):
    x1, y1, x2, y2 = xyxy
    clipped = x1 <= 1 or y1 <= 1 or x2 >= rgb.shape[1] - 2 or y2 >= rgb.shape[0] - 2
    return Track(((x1 + x2) / 2.0, (y1 + y2) / 2.0), tuple(xyxy),
                 int((x2 - x1) * (y2 - y1)), clipped, t)


def find_labels(rgb, label, max_age=4.0):
    """EVERY current instance of a label, as Tracks. The map needs all of them;
    find_label() picks the single best one for the approach code."""
    label = str(label).strip().lower()
    with lock:
        inst = list(yolo_latest["dets"].get(label, ()))
        t = yolo_latest["t"]
    if not inst or time.time() - t > max_age:
        return []
    return [_track_from_box(rgb, e["xyxy"], t) for e in inst]


def find_label(rgb, label, T_base_cam=None):
    """Generic YOLO-driven finder for any query label.

    Falls back to the legacy colour trackers for 'red cube' / 'green cube' because
    those trackers are tighter during close approach than a 2.5 Hz YOLO refresh.
    """
    label = str(label).strip().lower()
    # legacy colour trackers for the original cube colours
    if label == "red cube":
        return find_red(rgb, T_base_cam)
    if label == "green cube":
        return find_green(rgb, T_base_cam)
    tracks = find_labels(rgb, label)
    return tracks[0] if tracks else None


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
    # EVERY OTHER YOLO DETECTION. Until now the overlay drew boxes for the two
    # colour trackers and nothing else, so a pen (or cup, or anything) could be
    # detected at high confidence and still show NO BOX - which looks exactly like
    # "the detector cannot see it". Draw whatever the detector currently reports.
    with lock:
        dets = dict(yolo_latest["dets"])
        d_age = time.time() - yolo_latest["t"]
    if d_age < 4.0:
        # One physical object often matches SEVERAL words in the query - a pen fires
        # as both "pen" and "knife" - and drawing each one stacks unreadable labels
        # on top of each other. Collapse overlapping boxes and keep the best-scoring
        # name, so what you see is one object with one label.
        flat = []
        for lbl, insts in dets.items():
            if lbl in ("red cube", "green cube"):
                continue                      # already drawn by their trackers above
            for e in insts:
                flat.append((float(e["conf"]), lbl, tuple(e["xyxy"])))
        flat.sort(key=lambda t: -t[0])
        kept = []
        for conf, lbl, box in flat:
            if all(_box_iou(box, k[2]) < 0.45 for k in kept):
                kept.append((conf, lbl, box))
        for conf, lbl, box in kept:
            x1, y1, x2, y2 = (int(v) for v in box)
            col = (0, 200, 255)               # amber: a generic YOLO hit
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            w_px = float(max(4, x2 - x1))
            rng_cm = fx * class_size_m(lbl) / w_px * 100.0
            txt = f"{lbl} {conf:.2f} {rng_cm:.0f}cm"
            ty = max(14, y1 - 6)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, ty - th - 3), (x1 + tw + 4, ty + 3), (20, 20, 20), -1)
            cv2.putText(img, txt, (x1 + 2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
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


# ================= relaxed (torque-off) idle =================
# WHY. Holding a pose costs current, and holding it for a long time is what heats
# the servos until one latches its overload flag - which then refuses to answer
# reads and takes the whole server down at the next connect. Between uses the arm
# does not need to hold anything, so fold it somewhere gravity does the work and
# switch torque OFF. Nothing to heat, nothing to latch.
#
# ORDER MATTERS: fold FIRST, then cut torque. Cutting torque with the arm extended
# would just drop it.
ARM_RELAXED = [False]
IDLE_RELAX_S = [240.0]         # relax after this long with nothing happening; 0 = never
last_activity = [time.time()]


def note_activity():
    last_activity[0] = time.time()


def wake_arm(why=""):
    """Re-energise the servos. Goal position is re-synced to where the arm ACTUALLY
    is first, otherwise enabling torque snaps it back to the stale goal it held
    before relaxing - a jerk, and exactly the kind of load that trips overload."""
    if not ARM_RELAXED[0]:
        return
    try:
        with bus_lock:
            obs = robot.get_observation()
            q_now = [float(obs[f"{m}.pos"]) for m in ARM_MOTORS]
            act = {f"{m}.pos": v for m, v in zip(ARM_MOTORS, q_now)}
            act["gripper.pos"] = float(obs.get("gripper.pos", 50.0))
            robot.bus.enable_torque()
            robot.send_action(act)          # hold where it is, do not snap
        ARM_RELAXED[0] = False
        say(f"arm awake{(' — ' + why) if why else ''}")
    except Exception as e:
        say(f"wake failed: {type(e).__name__}: {e}")


def relax_arm(fold=True):
    """Fold home, then cut torque so nothing is being held."""
    if ARM_RELAXED[0]:
        return
    try:
        if fold:
            set_phase("RELAX", "folding home before going limp")
            goto_smooth(HOME, settle=0.4)
        with bus_lock:
            robot.bus.disable_torque()
        ARM_RELAXED[0] = True
        set_phase("RELAXED", "torque off — send any command to wake")
        say("arm relaxed: torque OFF, servos cool. Any motion command wakes it.")
    except Exception as e:
        say(f"relax failed: {type(e).__name__}: {e}")


def idle_relax_watch():
    """Relax the arm once it has been unused for IDLE_RELAX_S."""
    while True:
        time.sleep(5.0)
        try:
            limit = float(IDLE_RELAX_S[0])
            if limit <= 0 or ARM_RELAXED[0]:
                continue
            with lock:
                busy = state["running"]
            with jog_held_lock:
                jogging = bool(jog_held)
            active, _left, _t = guest_state()
            if busy or jogging or active:
                note_activity()
                continue
            if time.time() - last_activity[0] > limit:
                relax_arm(fold=True)
        except Exception:
            pass


def send_joints(q, gripper=None):
    # Single choke point for every motion in the program, so waking belongs here:
    # anything that wants to move the arm gets a live arm without having to know
    # about relaxing at all.
    if ARM_RELAXED[0]:
        wake_arm("motion requested")
    note_activity()
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
# Nothing that matters is outside the arm's own workspace. The reach is ~42 cm, so
# a "cube" localized at 92 cm is a broken solve, not a distant object — and letting
# those into the map is what filled it with ghosts strung out along the sightline.
# Gate every localization on this before it is ever stored.
MAP_R_MIN = 0.08
MAP_R_MAX = 0.55
# An entry not re-observed for this long is STALE: the object was moved or taken
# away, and the map should stop asserting it is there. Without this the map keeps
# reporting a scene that no longer exists - and worse, a stale high-n entry sits in
# the way and swallows observations of whatever is now at that spot.
MAP_TTL_S = 90.0
# How long an entry has to have gone unseen before a DIFFERENT label arriving at
# its position is allowed to take it over. Fresh disagreement (a pen firing as both
# "pen" and "knife" in the same instant) is genuine ambiguity and the better-
# supported name should win; stale disagreement means the object changed.
LABEL_TAKEOVER_S = 4.0


def _may_merge_labels(a, b):
    """May two DIFFERENTLY-labelled detections at the same spot be one object?

    Merging across labels is what collapsed the pen's five aliases (pen, knife,
    scissors, toothbrush, remote) into one entry. But applied bluntly it also
    merged the RED and GREEN cubes, because the bad hand-eye puts them within the
    merge radius of each other - and then the green cube vanished under a
    better-supported "red cube".

    The distinction: if BOTH names are things we were explicitly asked to look for,
    they are meant to be told apart and must never merge. If only one is in the
    query, the other is the detector reaching for a different word for the same
    thing, and merging is right.
    """
    if a == b:
        return True
    q = set(_query_labels())
    return not (a in q and b in q)
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


def obj_xy_2d(bbox, T_cam, z_m=None, label=None):
    """Where the object is, base-frame (x, y).

    RANGE COMES FROM STEREO DEPTH when it is available and sane; otherwise it
    falls back to apparent size:

        range = focal * real_object_width / bbox_width_px

    The real width is looked up from CLASS_META by label. Stereo depth is
    independent of the object's size, so it anchors the range and kills the
    random jumps.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 4:
        return None, float("nan"), class_size_m(label)

    real_w = class_size_m(label)

    # APPARENT-SIZE RANGING IS ONLY VALID FOR OBJECTS THAT LOOK THE SAME FROM EVERY
    # SIDE. range = fx * assumed_width / bbox_width treats the bbox width as the
    # object's real width. For a cube or a cup that holds at any angle. For an
    # elongated object it is nonsense: measured, a 14 cm pen 40 cm away reports
    # 150 cm when it lies across the view, 14 cm when diagonal and 11 cm end-on -
    # a 13x swing driven purely by an angle nobody measured. Each frame it rotates
    # slightly, the range jumps, and the map grows another ghost somewhere new.
    #
    # So: refuse to place elongated objects from apparent size alone. Better a gap
    # in the map than a confident wrong coordinate the arm will then drive at.
    pm = class_meta(label)
    aspect = max(pm["w_m"], pm["d_m"]) / max(min(pm["w_m"], pm["d_m"]), 1e-4)
    if aspect > 2.2 and (z_m is None or not (0.03 < z_m < 1.20)):
        # ELONGATED OBJECT: measure against its LONG axis, not its width.
        # class_size_m returns the geometric mean of w and d (3.7 cm for a pen) and
        # comparing that to the bbox WIDTH is meaningless - the width is whatever
        # angle the pen happens to lie at, which is why the overlay read "15cm" for
        # a pen 30 cm away. The bbox's LONGEST side, however, always corresponds to
        # the object's LONGEST axis, foreshortened by the viewing angle. That over-
        # estimates range when foreshortened, but it is bounded and roughly right,
        # instead of being wrong by a factor that swings with rotation.
        #
        # (The table-plane methods would be better still, but they need the hand-eye
        # rotation to be correct and it is not - rays from the upper frame graze out
        # to ~5.8 m. Apparent size is the only range that does not go through it.)
        long_px = float(max(w, h))
        rng = float(fx * max(pm["w_m"], pm["d_m"]) / max(long_px, 4.0))
        if not (0.05 < rng < 1.20):
            return None, float("nan"), real_w
        rng *= float(RANGE_SCALE[0])
        rng = float(np.clip(rng, 0.03, 1.50))
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        d = np.array([(u - cx0) / fx, (v - cy0) / fy, 1.0], dtype=np.float64)
        d /= np.linalg.norm(d)
        pt = T_cam[:3, 3] + T_cam[:3, :3] @ (d * rng)
        xy = pt[:2]
        if not (MAP_R_MIN < float(np.hypot(*xy)) < MAP_R_MAX):
            return None, float("nan"), real_w
        return _rotate_xy(xy, MAP_BEARING_OFFSET_DEG[0]), float(rng), real_w

    # Prefer stereo depth if the caller passed a valid metric range.
    if z_m is not None and 0.03 < z_m < 1.20:
        rng = float(z_m)
    else:
        rng = float(fx * real_w / float(w))
        if not (0.05 < rng < 1.20):
            return None, float("nan"), real_w

    rng *= float(RANGE_SCALE[0])
    rng = float(np.clip(rng, 0.03, 1.50))

    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0
    d = np.array([(u - cx0) / fx, (v - cy0) / fy, 1.0], dtype=np.float64)
    d /= np.linalg.norm(d)
    p = T_cam[:3, 3] + T_cam[:3, :3] @ (d * rng)
    xy = p[:2]
    if not (MAP_R_MIN < float(np.hypot(*xy)) < MAP_R_MAX):
        return None, float("nan"), real_w
    # Correct heading/yaw error in the hand-eye by rotating the bearing.
    xy = _rotate_xy(xy, MAP_BEARING_OFFSET_DEG[0])
    return xy, float(rng), real_w


def _consolidate_2d():
    """Collapse map entries that are really ONE physical object.

    MERGING ONLY WITHIN A LABEL WAS THE BUG. An open vocabulary gives one object
    several names - a single pen fired as pen, knife, scissors, toothbrush AND
    remote, so it became five "objects" that could never combine no matter how
    close together they sat (measured: five entries within 6-9 cm of each other).
    Two detections at the same place ARE the same thing; the label is the least
    reliable part of the observation, so position decides and the best-supported
    name wins. Weighted by observation count. Caller holds w2d_lock.
    """
    objs = W2D["objs"]
    changed = True
    while changed:
        changed = False
        items = list(objs.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                ta, a = items[i]
                tb, b = items[j]
                if ta not in objs or tb not in objs:
                    continue
                if not _may_merge_labels(a["label"], b["label"]):
                    continue
                if float(np.hypot(*(a["xy"] - b["xy"]))) < _merge_radius(a, b):
                    keep, drop = (ta, tb) if a["n"] >= b["n"] else (tb, ta)
                    ko, do = objs[keep], objs[drop]
                    wsum = ko["n"] + do["n"]
                    ko["xy"] = (ko["n"] * ko["xy"] + do["n"] * do["xy"]) / wsum
                    for k in ("w_m", "d_m", "h_m"):
                        ko[k] = (ko["n"] * ko[k] + do["n"] * do[k]) / wsum
                    ko["yaw"] = _yaw_blend(ko["yaw"], do["yaw"], do["n"] / wsum)
                    # a measurement beats a prior, whichever entry it came from
                    if do.get("measured") and not ko.get("measured"):
                        ko["shape"], ko["measured"] = do["shape"], True
                    # two ghosts of one object each hold caliper readings from the
                    # bearings they were seen from — pooling them is exactly the
                    # extra evidence the footprint fit wants
                    for bk, bv in do.get("sup", {}).items():
                        ko["sup"][bk] = 0.5 * (ko["sup"][bk] + bv) if bk in ko["sup"] else bv
                    fit = _fit_rect_from_support(ko["sup"])
                    if fit is not None:
                        ko["w_m"], ko["d_m"], ko["yaw"] = fit
                    if ko["label"] != do["label"]:
                        alt = set(ko.get("aka", ())) | set(do.get("aka", ())) | {do["label"]}
                        ko["aka"] = sorted(alt - {ko["label"]})
                    ko["n"] = wsum
                    ko["t"] = max(ko["t"], do["t"])
                    del objs[drop]
                    changed = True
                    break
            if changed:
                break


# Wrist roll that lines the JAWS UP ACROSS an object's long axis — the only way a
# parallel gripper closes on a pen, a fork or a book. GRASP_ROLL is the roll that
# was measured to put the jaws square to a cube when the arm points straight at it,
# so it is the zero of this mapping; the correction is the object's yaw measured
# RELATIVE to the arm's own bearing (shoulder_pan already turns the whole hand with
# the reach direction), plus 90 deg to cross the long axis.
# >>> UNVERIFIED ON HARDWARE. If the jaws come in ALONG the object instead of
# across it, drop the +90. If they are mirrored, flip GRASP_YAW_SIGN to -1. <<<
GRASP_YAW_SIGN = 1.0


def grasp_roll_for_yaw(yaw_deg, xy):
    bearing = math.degrees(math.atan2(float(xy[1]), float(xy[0])))
    rel = ((float(yaw_deg) - bearing + 90.0) % 180.0) - 90.0
    want = GRASP_ROLL + GRASP_YAW_SIGN * (rel + 90.0)
    # A parallel jaw is 180 deg symmetric, so roll and roll+-180 are the SAME grasp.
    # Pick whichever representative the wrist can actually reach — clamping instead
    # saturates at the limit for most yaws and silently discards the orientation.
    cands = [w for w in (want - 360.0, want - 180.0, want, want + 180.0, want + 360.0)
             if -157.2 <= w <= 162.8]
    if not cands:
        return float(GRASP_ROLL)
    return float(min(cands, key=lambda w: abs(w - GRASP_ROLL)))


def _merge_radius(a, b=None):
    """How close two same-label detections must be to count as one object.

    THE FLOOR IS SET BY LOCALIZATION NOISE, NOT BY OBJECT SIZE. Scaling this down
    to 0.55x the object's own footprint (3.5 cm for a cube) was wrong and produced
    the 16-ghost map: consecutive views of ONE cube land 5-15 cm apart, so every
    observation spawned a fresh tag. Object size may only ever WIDEN the radius —
    a laptop needs more than 14 cm — never narrow it below what the jitter demands.
    """
    r = W2D_MERGE
    for o in (a, b):
        if o is not None:
            r = max(r, 0.55 * max(o["w_m"], o["d_m"]))
    return float(np.clip(r, W2D_MERGE, 0.30))


def _yaw_blend(y_old, y_new, w_new):
    """Circular mean of two axis angles. A footprint rectangle has no front, so
    yaw lives mod 180 deg — averaging -89 and +89 naively gives 0, which is a
    right angle away from both. Average the doubled angle instead."""
    a = math.radians(2.0 * float(y_old))
    b = math.radians(2.0 * float(y_new))
    s = (1 - w_new) * math.sin(a) + w_new * math.sin(b)
    c = (1 - w_new) * math.cos(a) + w_new * math.cos(b)
    if abs(s) < 1e-9 and abs(c) < 1e-9:
        return float(y_new)
    return float(((math.degrees(math.atan2(s, c)) / 2.0 + 90.0) % 180.0) - 90.0)


def world2d_update(label, xy, stereo, w_m, d_m, h_m, shape, yaw, measured,
                   across_m=None, u_deg=None):
    """Fold one observation of one object into the map.

    across_m / u_deg are one caliper reading of the footprint (its width along the
    across-view direction u_deg) — the only footprint fact a single view actually
    establishes. They accumulate per direction bin, and once three bearings are in,
    the footprint and yaw are re-fitted from all of them.
    """
    xy = np.asarray(xy, float)
    obs = {"w_m": float(w_m), "d_m": float(d_m)}
    with w2d_lock:
        # Match on POSITION, not on the label: the same object arrives under
        # different names from an open vocabulary, and a new name must land on the
        # existing entry rather than spawn a rival ghost beside it.
        best, bd = None, None
        for t, o in W2D["objs"].items():
            if not _may_merge_labels(o["label"], label):
                continue
            dist = float(np.hypot(*(o["xy"] - xy)))
            if dist < _merge_radius(o, obs) and (bd is None or dist < bd):
                best, bd = t, dist
        if best is None:
            best = W2D["next"]; W2D["next"] += 1
            W2D["objs"][best] = {"label": label, "xy": xy,
                                 "w_m": float(w_m), "d_m": float(d_m), "h_m": float(h_m),
                                 "shape": str(shape), "yaw": float(yaw),
                                 "measured": bool(measured), "sup": {},
                                 "n": 1, "stereo": stereo, "t": time.time()}
        else:
            o = W2D["objs"][best]
            # A DIFFERENT label landing on a STALE entry means the thing at this
            # spot changed - the old name is not evidence any more, however many
            # times it was seen. Take the position over outright rather than let a
            # stale n=1793 "red cube" swallow every new observation of the green one
            # that is actually sitting there now.
            # Compare against when this entry was last confirmed UNDER ITS OWN
            # NAME, not when it was last touched at all. Touch-time never goes
            # stale: every incoming green observation refreshed the leftover "red
            # cube" entry it was being merged into, so the relabel that would have
            # fixed it could never fire - the wrong label kept itself alive.
            seen_as_itself = o.get("label_t", o["t"])
            if o["label"] != label and (time.time() - seen_as_itself) > LABEL_TAKEOVER_S:
                say(f"map: {o['label']}#{best} not confirmed as '{o['label']}' for "
                    f"{time.time() - seen_as_itself:.0f}s — relabelling as '{label}'")
                o["label"], o["aka"], o["n"] = label, [], 0
                o["measured"] = False
            if o["label"] == label:
                o["label_t"] = time.time()
            # Give fresh observations more weight so the map converges faster and
            # does not stay stuck on an early bad localization.
            o["xy"] = 0.55 * o["xy"] + 0.45 * xy
            if measured and not o.get("measured"):
                o["w_m"], o["d_m"], o["h_m"] = float(w_m), float(d_m), float(h_m)
                o["yaw"], o["shape"], o["measured"] = float(yaw), str(shape), True
            elif measured or not o.get("measured"):
                o["h_m"] = 0.7 * o["h_m"] + 0.3 * float(h_m)
                if not o["sup"]:            # no caliper readings yet — keep blending
                    o["w_m"] = 0.7 * o["w_m"] + 0.3 * float(w_m)
                    o["d_m"] = 0.7 * o["d_m"] + 0.3 * float(d_m)
                    o["yaw"] = _yaw_blend(o["yaw"], yaw, 0.3)
            if o["label"] != label:
                o["aka"] = sorted(set(o.get("aka", ())) | {label} - {o["label"]})
            o["n"] += 1
            o["stereo"] = stereo
            o["t"] = time.time()

        o = W2D["objs"][best]
        if across_m is not None and u_deg is not None:
            k = _sup_bin(u_deg)
            o["sup"][k] = (0.6 * o["sup"][k] + 0.4 * float(across_m)
                           if k in o["sup"] else float(across_m))
            fit = _fit_rect_from_support(o["sup"])
            if fit is not None:
                o["w_m"], o["d_m"], o["yaw"] = fit
                o["shape"] = _classify_shape(o["w_m"], o["d_m"], o["h_m"],
                                             class_meta(label)["shape"])
        _consolidate_2d()


def sense_2d(joints=None, rgb=None):
    """Fold every detected instance of every queried label into the 2D map.

    Each instance is MEASURED off the frame (footprint, height, yaw — see
    measure_object); only when that solve fails does it fall back to the old
    apparent-size range against the class prior, flagged measured=False so the
    map can tell a measurement from a guess.
    """
    if joints is None:
        joints, rgb, _ = observe()
    T = T_cam_of(joints)
    H_img, W_img = rgb.shape[:2]
    for label in _query_labels():
        for tr in _map_tracks(rgb, label, T):
            # WHICH EDGE IS CLIPPED MATTERS. Skipping every clipped box threw away a
            # pen detected at 0.63 whose box merely touched the TOP of the frame -
            # it never reached the map and so could never be picked. Only the BOTTOM
            # edge carries the table contact: cut off there and the position is a
            # lie, cut off anywhere else and the bottom edge (and the width) are
            # still perfectly good.
            x1b, y1b, x2b, y2b = tr.bbox_xyxy
            if y2b >= H_img - 2:
                continue
            full_view = not (x1b <= 1 or y1b <= 1 or x2b >= W_img - 2)
            # the silhouette solve needs the WHOLE object, so it only runs on a
            # fully-visible box; the position fallbacks below do not.
            m = measure_object(rgb, tr.bbox_xyxy, T, label) if full_view else None
            if m is not None:
                xy = _rotate_xy(m["xy"], MAP_BEARING_OFFSET_DEG[0])
                world2d_update(label, xy, m["rng_m"], m["w_m"], m["d_m"], m["h_m"],
                               m["shape"], m["yaw_deg"], True,
                               across_m=m["across_m"],
                               u_deg=m["u_deg"] + MAP_BEARING_OFFSET_DEG[0])
                continue
            # fallback 1: apparent-size range against the class prior
            zs = [z for z in (read_depth_m(tr.uv) for _ in range(3)) if z is not None]
            z = float(np.median(zs)) if zs else None
            xy, st, _sz = obj_xy_2d(tr.bbox_xyxy, T, z_m=z, label=label)
            if xy is None:
                # fallback 2: WHERE THE BOX MEETS THE TABLE.
                # For an elongated object apparent-size ranging is invalid (its
                # bbox width depends on an unmeasured angle) so obj_xy_2d refuses
                # it - but that left a pen detected at 0.61 and still absent from
                # the map, i.e. unpickable. Casting the bbox's BOTTOM-CENTRE onto
                # the table plane needs neither the object's size nor its
                # orientation: anything resting on the table meets it there. Good
                # for position only, which is what the grasp actually needs.
                x1, y1, x2, y2 = tr.bbox_xyxy
                p3 = ray_to_table(((x1 + x2) / 2.0, y2), T, TABLE_Z0)
                if p3 is not None:
                    cand = _rotate_xy(p3[:2], MAP_BEARING_OFFSET_DEG[0])
                    if MAP_R_MIN < float(np.hypot(*cand)) < MAP_R_MAX:
                        xy, st = cand, float("nan")
            if xy is not None:
                p = class_meta(label)
                world2d_update(label, xy, st, p["w_m"], p["d_m"], p["h_m"],
                               p["shape"], 0.0, False)


def _map_tracks(rgb, label, T):
    """Every instance of a label to fold into the map. The two cube colours still
    go through their dedicated HSV trackers (tighter than a 2.5 Hz YOLO refresh),
    everything else comes straight from YOLO, all instances."""
    if label in ("red cube", "green cube"):
        tr = find_label(rgb, label, T)
        return [tr] if tr is not None else []
    return find_labels(rgb, label)


def world2d_snapshot():
    now = time.time()
    with w2d_lock:
        # forget objects that have not been seen in a while: the map should
        # describe the table as it is, not as it once was
        for t in [t for t, o in W2D["objs"].items() if now - o["t"] > MAP_TTL_S]:
            del W2D["objs"][t]
        out = []
        for t, o in W2D["objs"].items():
            w_m, d_m, h_m = float(o["w_m"]), float(o["d_m"]), float(o["h_m"])
            out.append({"tag": t, "label": o["label"],
                        "x": round(float(o["xy"][0]), 3), "y": round(float(o["xy"][1]), 3),
                        # "size" stays the single characteristic edge that the older
                        # callers (3D viewer, goto2d) read; the real geometry is w/d/h.
                        "size": round(float(math.sqrt(max(w_m, 1e-3) * max(d_m, 1e-3))), 3),
                        "shape": o["shape"],
                        "yaw": round(float(o["yaw"]), 1),
                        "w_m": round(w_m, 3),
                        "d_m": round(d_m, 3),
                        "h_m": round(h_m, 3),
                        "measured": bool(o.get("measured", False)),
                        # yaw is only real if the silhouette solve produced it;
                        # otherwise it is 0 because nothing measured it, and drawing
                        # that as a definite orientation is a lie
                        "yaw_known": bool(o.get("measured", False)),
                        "aka": list(o.get("aka", ())),
                        "r_cm": round(float(np.hypot(*o["xy"])) * 100, 1),
                        "ang": round(math.degrees(math.atan2(o["xy"][1], o["xy"][0]))),
                        "stereo_cm": (round(o["stereo"] * 100) if o["stereo"] == o["stereo"] else None),
                        "n": o["n"], "age": round(now - o["t"], 1)})
        return out


def _apply_query_now(q, timeout=6.0):
    """Switch the detector vocabulary and WAIT for it to take effect.

    The model may only be mutated on the thread that runs it (set_classes from
    Flask crashes it), so the change goes through pending_query and the YOLO
    worker picks it up on its ~2.5 s cycle. Scanning before that lands would sweep
    the whole table with the OLD vocabulary and find nothing.
    """
    pending_query[0] = q
    t0 = time.time()
    while time.time() - t0 < timeout:
        with lock:
            if (state.get("query") or "") == q:
                time.sleep(0.4)          # let one frame run through the new classes
                return True
        time.sleep(0.15)
    say(f"scan: detector did not switch vocabulary within {timeout:.0f}s")
    return False


def scan_2d(broad=True):
    """Pan the base across the front arc, sensing into the 2D map.

    broad=True (the default) sweeps with the WHOLE tabletop vocabulary rather than
    whatever single thing the query happens to name — so a scan just tells you what
    is on the table without anyone having to type the categories first. The prior
    query is restored afterwards, because the pick engine aims at the query and
    leaving it as 29 classes would change what Start picks up.
    """
    # Scan at SCAN_IMGSZ for recall (a pen is a few pixels wide at 320), then put
    # PICK_IMGSZ back - the approach trims were tuned at 320 and range comes from
    # bbox width, so leaving the scan resolution set would shift every pick range.
    prev_imgsz = getattr(detector, "imgsz", None) if detector else None
    if detector is not None:
        detector.imgsz = SCAN_IMGSZ
    prev_q = None
    if broad:
        with lock:
            prev_q = state.get("query")
        want = ", ".join(TABLE_CLASSES)
        if want != prev_q:
            set_phase("SCAN2D", f"loading {len(TABLE_CLASSES)} tabletop categories…")
            _apply_query_now(want)
    try:
        _scan_sweep()
    finally:
        if detector is not None and prev_imgsz is not None:
            detector.imgsz = prev_imgsz
        with lock:
            now_q = state.get("query") or ""
        if broad and prev_q and prev_q != now_q:
            # 8 s, not 4: the worker's cycle is ~2.5 s and a 29-class pass is
            # slower than a 2-class one, so a tight timeout logs a false alarm for
            # a switch that lands a moment later anyway.
            if _apply_query_now(prev_q, timeout=8.0):
                say(f"scan: detection query restored to '{prev_q}'")
            else:
                say(f"scan: query restore to '{prev_q}' is still pending")


def _scan_sweep():
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
    # Hover height follows the object's MEASURED height, not a fixed 6 cm: a
    # keyboard is 2 cm tall and a bottle 23 cm, and the old constant flew the
    # gripper straight into anything taller than a cube.
    # Height is the least reliable of the three measured dimensions — on the
    # synthetic bench it reads LOW on tall objects (a 10 cm cup measured 5.6) —
    # and reading low is the direction that flies the gripper into the object. So
    # clear the TALLER of the measurement and the class prior.
    h_clear = float(np.clip(max(o["h_m"], class_height_m(label)), 0.005, 0.30))
    z_mid = float(np.clip(TABLE_Z0 + 0.5 * float(o["h_m"]), 0.005, 0.20))
    z_hover = float(np.clip(TABLE_Z0 + h_clear + 0.04, 0.05, 0.26))
    h = h_clear
    set_phase("GOTO2D", f"{label} #{tag} @ ({xy[0]*100:.0f},{xy[1]*100:.0f})cm "
                        f"h={h*100:.0f}cm yaw={o['yaw']:+.0f}deg")
    j = observe()[0].astype(np.float64)
    pitch, e = plan_grasp_pitch(np.array([xy[0], xy[1], z_mid]), j)
    if pitch is None:
        raise Abort(f"{label} #{tag}: out of reach (best IK {e*1e3:.0f}mm, "
                    f"r={np.hypot(*xy)*100:.0f}cm)")
    roll = grasp_roll_for_yaw(o["yaw"], xy)
    q, err = _ik_hold_pitch(observe()[0], np.array([xy[0], xy[1], z_hover]), pitch,
                            roll, ret_err=True)
    if err > 0.008:
        raise Abort(f"hover pose unreachable (IK {err*1e3:.0f}mm)")
    goto_smooth(q, settle=0.25)
    with lock:
        state["obj3d"] = [float(xy[0]), float(xy[1]), z_mid]
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


# ---- FIRST-PERSON APPROACH ------------------------------------------------
# The camera is the head of the snake: it rides on the gripper, so every move
# changes the view, and the closer we get the WORSE the detector behaves (it
# hallucinates, then stops recognising the cube at all once the cube fills the
# frame and slides under the fingers). So we do NOT servo on pixels all the way
# in. We look from a distance where the detector is honest, LOCK the cube's 3D
# point, and run the last leg on the lock -- the cube is static, so a remembered
# coordinate beats a close-range guess.
STANDOFF_H = 0.05        # m, fingertip parks this far ABOVE the cube, then descends
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

# ---------------- carry + place ----------------
# WHAT WE KNOW WHILE HOLDING SOMETHING, AND WHY IT IS ENOUGH.
# The height of the object in the jaws never has to be estimated. At the instant
# the grasp closes, the fingertip is at a known z and the object's bottom is
# resting on the table, so
#       grip_to_bottom = z_tip_at_grasp - TABLE_Z0
# is the distance from the fingertip down to the underside of whatever is now held
# — measured, not guessed, and constant for as long as the grip holds. Setting that
# underside on top of a destination of height h_top is then just
#       z_release = TABLE_Z0 + h_top + grip_to_bottom + PLACE_CLEAR_M
# so the only quantity the map has to supply is the DESTINATION's height. That is
# what measure_object provides — and because its height reads LOW (see goto_2d),
# and reading low here means driving the carried object down into the target, the
# destination height is taken as max(measured, class prior).
PLACE_CLEAR_M = 0.008      # gap left under the carried object at release
PLACE_HOVER_M = 0.07       # hover this far above z_release before descending
PLACE_TRANSIT_Z = 0.16     # carry the object at this height while traversing
PLACE_OPEN_PCT = 62.0      # gripper opening that releases without flicking
PLACE_RETREAT_M = 0.09     # straight-up retreat after releasing

# Detection resolution. 320 is what the approach trims (AIM_DU, TARGET_RIGHT_TRIM_M,
# TARGET_BACK_M) were tuned against, and range comes straight from bbox width
#     range = fx * real_width / bbox_width
# so changing this SHIFTS EVERY RANGE and silently invalidates that tuning. Raising
# it to 640 found the pen but made cube picking worse; the pen is a scan-time
# concern, the trims are a pick-time concern, so they get different resolutions.
DET_CONF = [0.06]          # detector confidence floor; live-tunable via /setconf
PICK_IMGSZ = 320           # used for picking - do not change without re-tuning trims
SCAN_IMGSZ = 640           # used only while scanning, where recall matters more

# Locate from one canonical pose instead of the map average. OFF by default: it is
# repeatable (0.4 cm) but its systematic offset differs from the map's, so it needs
# its own trim values before it can beat the tuned map path.
USE_SURVEY_LOCATE = [False]

# What is currently in the jaws. grip_to_bottom is the measured fingertip->underside
# distance described above; h_m is the carried object's own height, needed only to
# grow the destination's height in the map so a SECOND place stacks on top of the
# first instead of into it.
carry = {"held": False, "label": None, "h_m": 0.0, "grip_to_bottom": PICK_GRASP_Z,
         "tag": None}


def _set_carry(held, label=None, h_m=0.0, grip_to_bottom=None, tag=None):
    with lock:
        carry.update(held=bool(held), label=label, h_m=float(h_m), tag=tag)
        if grip_to_bottom is not None:
            carry["grip_to_bottom"] = float(grip_to_bottom)
        state["carry"] = (f"{label} (h={h_m*100:.1f}cm, "
                          f"grip->bottom {carry['grip_to_bottom']*100:.1f}cm)"
                          if held else None)


# ================= self-calibrated table plane =================
# THE SCRAPE. The code treats the table as z = 0 (the base plane) and grasps at a
# fixed PICK_GRASP_Z. Two things break that, and they ADD UP as the arm reaches out:
#   * the table is not exactly parallel to the robot's base plane;
#   * the arm SAGS under its own weight, and the droop grows with extension - so the
#     fingertip sits lower than the FK says, by more at r=35cm than at r=18cm.
# Either one alone tilts the effective floor; together they are why it clears fine
# near the base and scrapes at full reach.
#
# We do not have to separate them. TOUCH THE TABLE AND WRITE DOWN THE FK z WHERE
# CONTACT HAPPENS. That number already contains the table height, the tilt AND the
# sag at that reach, because it is measured in the same coordinates the arm is
# commanded in. Probe several points, fit a plane, and use it as the floor.
#
#     z_floor(x, y) = a*x + b*y + c
#
# Everything that used to assume 0 (or the hand-tuned -0.022) then reads off this.
FLOOR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "floor_plane.json")
# Default c matches the value someone measured by hand and left in ee_move_rel's
# comment ("table contact is z=-0.022, sag included"); a=b=0 means "flat and level"
# until a calibration says otherwise.
FLOOR_PLANE = [0.0, 0.0, -0.022]
FLOOR_PROBE_STEP = 0.0025      # descend in 2.5 mm bites - gentle enough not to
                               # slam the servos or trip the overload latch
FLOOR_PROBE_DROP = 0.055       # give up after this much descent from the start
FLOOR_FOLLOW_MIN = 0.45        # measured/commanded travel below this = blocked
FLOOR_LOAD_RISE = 90           # raw Present_Load rise over baseline = pushing
FLOOR_RETREAT = 0.020          # lift this much after each touch
FLOOR_GRASP_CLEAR = 0.012      # grasp this far ABOVE the measured floor


def floor_z(x, y):
    """Table height in base z at (x, y), from the calibrated plane."""
    a, b, c = FLOOR_PLANE
    return float(a * float(x) + b * float(y) + c)


def load_floor_plane():
    global FLOOR_PLANE
    try:
        with open(FLOOR_FILE) as f:
            d = json.load(f)
        FLOOR_PLANE = [float(d["a"]), float(d["b"]), float(d["c"])]
        say(f"floor: calibrated plane loaded — z = {FLOOR_PLANE[0]:+.4f}x "
            f"{FLOOR_PLANE[1]:+.4f}y {FLOOR_PLANE[2]:+.4f}  "
            f"(tilt {d.get('tilt_deg', 0):.2f}deg, rms {d.get('rms_mm', 0):.1f}mm, "
            f"fitted {d.get('fitted', '?')})")
        return d
    except FileNotFoundError:
        say(f"floor: no calibration yet — assuming z={FLOOR_PLANE[2]*100:.1f}cm and level. "
            f"Press 'Calibrate floor' to measure it.")
    except Exception as e:
        say(f"floor: ignoring bad {os.path.basename(FLOOR_FILE)} ({e})")
    return None


def _arm_load():
    """Summed |Present_Load| over the joints that carry the arm's weight. Rises
    sharply when the fingertip pushes into something."""
    tot = 0.0
    for m in ("shoulder_lift", "elbow_flex"):
        try:
            with bus_lock:
                tot += abs(float(robot.bus.read("Present_Load", m, normalize=False)))
        except Exception:
            return None
    return tot


def probe_floor_at(x, y, pitch, j5, z_start=None):
    """Lower the fingertip at (x, y) until it touches, and return the FK z where it
    did. Contact is called on EITHER the arm stopping following the command or the
    load rising - two independent signals, because either alone has a failure mode
    (a servo can stall silently; load can drift)."""
    z0 = (floor_z(x, y) + 0.030) if z_start is None else float(z_start)
    if _move_tip(np.array([x, y, z0]), pitch, j5, settle=0.25, step=1.2) is None:
        return None, "cannot reach the probe point"
    base_load = _arm_load()
    z_cmd = z0
    stuck = 0
    dropped = 0.0
    while dropped < FLOOR_PROBE_DROP:
        checkpoint()
        z_before = float(_tip(observe(overlay=False)[0])[2])
        z_cmd -= FLOOR_PROBE_STEP
        if _move_tip(np.array([x, y, z_cmd]), pitch, j5, settle=0.14, step=0.7) is None:
            return None, f"IK gave up at z={z_cmd*100:.1f}cm"
        dropped += FLOOR_PROBE_STEP
        z_after = float(_tip(observe(overlay=False)[0])[2])
        moved = z_before - z_after
        load = _arm_load()
        pushing = (base_load is not None and load is not None
                   and load - base_load > FLOOR_LOAD_RISE)
        if moved < FLOOR_FOLLOW_MIN * FLOOR_PROBE_STEP or pushing:
            stuck += 1
        else:
            stuck = 0
        if stuck >= 2 or pushing:
            z_touch = z_after
            # back off so we are not leaning on the table while we think
            _move_tip(np.array([x, y, z_touch + FLOOR_RETREAT]), pitch, j5,
                      settle=0.2, step=1.0)
            why = "load" if pushing else "not following"
            return z_touch, why
    _move_tip(np.array([x, y, z0]), pitch, j5, settle=0.2, step=1.2)
    return None, f"no contact within {FLOOR_PROBE_DROP*100:.0f}cm"


# Probe points spread over the working area: two reaches x three bearings, so the
# fit sees both how the floor tilts sideways AND how the sag grows with extension.
FLOOR_PROBE_POINTS = [(0.19, -18.0), (0.19, 0.0), (0.19, 18.0),
                      (0.29, -18.0), (0.29, 0.0), (0.29, 18.0),
                      (0.34, 0.0)]


def calibrate_floor():
    """Touch the table at several places and fit z_floor(x, y). Self-calibration:
    no ruler, no hand-tuned constant, and it absorbs arm sag for free."""
    global FLOOR_PLANE
    set_phase("FLOORCAL", f"probing the table at {len(FLOOR_PROBE_POINTS)} points")
    say("=" * 52)
    say("FLOOR CALIBRATION — touching the table to find its real height")
    q0 = observe(overlay=False)[0].astype(np.float64)
    send_joints(q0, gripper=8.0)          # jaws closed: one definite contact point
    time.sleep(0.6)
    pts = []
    for r, ang in FLOOR_PROBE_POINTS:
        checkpoint()
        x = r * math.cos(math.radians(ang))
        y = r * math.sin(math.radians(ang))
        pitch, e = plan_grasp_pitch(np.array([x, y, floor_z(x, y) + 0.02]), q0)
        if pitch is None:
            say(f"  r={r*100:.0f}cm ang={ang:+.0f}deg — unreachable (IK {e*1e3:.0f}mm), skipped")
            continue
        set_phase("FLOORCAL", f"probing r={r*100:.0f}cm ang={ang:+.0f}deg")
        z, why = probe_floor_at(x, y, pitch, float(q0[4]))
        if z is None:
            say(f"  r={r*100:.0f}cm ang={ang:+.0f}deg — {why}")
            continue
        pts.append((x, y, z))
        say(f"  r={r*100:.0f}cm ang={ang:+.0f}deg -> touched at z={z*100:+.2f}cm ({why})")

    if len(pts) < 3:
        raise Abort(f"floor: only {len(pts)} touch points — need 3 to fit a plane")
    A = np.array([[p[0], p[1], 1.0] for p in pts])
    zz = np.array([p[2] for p in pts])
    (a, b, c), *_ = np.linalg.lstsq(A, zz, rcond=None)
    resid = zz - A @ np.array([a, b, c])
    rms = float(np.sqrt(np.mean(resid ** 2)))
    tilt = math.degrees(math.atan(math.hypot(a, b)))
    FLOOR_PLANE = [float(a), float(b), float(c)]
    d = {"a": float(a), "b": float(b), "c": float(c),
         "tilt_deg": round(tilt, 3), "rms_mm": round(rms * 1000, 2),
         "points": [[round(v, 4) for v in p] for p in pts],
         "fitted": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        with open(FLOOR_FILE, "w") as f:
            json.dump(d, f, indent=1)
    except Exception as e:
        say(f"floor: could not save ({e})")
    say(f"floor plane: z = {a:+.4f}x {b:+.4f}y {c:+.4f}   "
        f"tilt {tilt:.2f}deg   fit rms {rms*1000:.1f}mm over {len(pts)} points")
    say(f"  at r=18cm the floor is z={floor_z(0.18,0)*100:+.2f}cm, "
        f"at r=34cm it is z={floor_z(0.34,0)*100:+.2f}cm  "
        f"(difference {abs(floor_z(0.34,0)-floor_z(0.18,0))*1000:.0f}mm — "
        f"that gap IS the scrape)")
    if rms > 0.004:
        say(f"  NOTE rms {rms*1000:.1f}mm is high for a flat table — a probe point "
            f"may have caught an object rather than the surface")
    say("=" * 52)
    set_phase("IDLE", f"floor calibrated: tilt {tilt:.2f}deg, rms {rms*1000:.1f}mm")
    return d


def _target_finder(label=None):
    """(finder, tracker, label) for a pick target.

    With no argument, uses whichever colour is named FIRST in the detection query.
    A plain `"green" in query` test picked GREEN out of the default
    "red cube, green cube" — so the arm dutifully drove at the green cube while the
    user was waiting for it to grab the red one.

    With an explicit label (from a typed task like "green on red"), the two cube
    colours still route to their dedicated HSV trackers — those are tighter during
    close approach than a 2.5 Hz YOLO refresh — and anything else gets the generic
    open-vocabulary finder, so `cup on book` works as well as `green on red`.
    """
    if label:
        lab = str(label).strip().lower()
        if "red" in lab:
            return find_red, red_tracker, "red"
        if "green" in lab:
            return find_green, green_tracker, "green"
        return (lambda rgb, T=None, _l=lab: find_label(rgb, _l, T)), None, lab
    # Use whatever the query names FIRST, not just the two cube colours. This used
    # to test only for "red"/"green" and fall through to RED for anything else - so
    # setting the query to "pen" and pressing Start silently hunted a red cube and
    # then aborted with "cannot see 'red'", which is baffling when you asked for a
    # pen. Ordering matters too: a plain `"green" in query` test picked GREEN out of
    # the default "red cube, green cube" while the user waited for the red one.
    labels = _query_labels()
    return _target_finder(labels[0] if labels else "red cube")


# ================= object class priors (the whole YOLO vocabulary) =============
# Per-class PRIORS: shape + typical real-world (width, depth, height) in metres.
# These are only a starting guess and a fallback — measure_object() measures the
# real size and yaw off the picture, and the map prefers the measurement whenever
# it succeeds. The priors matter when the object is clipped, tiny, or blends into
# the table so the silhouette solve fails.
#
# YOLO-World is open-vocabulary, so the label string IS the class key: anything
# you type in the query box works, it just gets the generic fallback if it is not
# listed here. The 80 COCO names are all present so "all YOLO classes" maps with
# sane numbers out of the box.
#
# Format: label -> (shape, width_m, depth_m, height_m), width/depth being the
# footprint on the table and height the vertical extent.
_CLASS_TABLE = {
    # --- the original cubes (measured on the real blocks) ---
    "red cube":      ("cube",     0.0508, 0.0508, 0.0508),
    "green cube":    ("cube",     0.0508, 0.0508, 0.0508),
    "blue cube":     ("cube",     0.0508, 0.0508, 0.0508),
    "yellow cube":   ("cube",     0.0508, 0.0508, 0.0508),
    "toy block":     ("cube",     0.0508, 0.0508, 0.0508),
    # --- COCO: people & animals ---
    "person":        ("cylinder", 0.45,  0.30,  1.70),
    "bird":          ("cuboid",   0.10,  0.22,  0.16),
    "cat":           ("cuboid",   0.18,  0.46,  0.25),
    "dog":           ("cuboid",   0.25,  0.70,  0.50),
    "horse":         ("cuboid",   0.60,  2.20,  1.60),
    "sheep":         ("cuboid",   0.40,  1.20,  0.90),
    "cow":           ("cuboid",   0.70,  2.40,  1.50),
    "elephant":      ("cuboid",   1.50,  4.00,  3.00),
    "bear":          ("cuboid",   0.80,  1.80,  1.20),
    "zebra":         ("cuboid",   0.60,  2.20,  1.50),
    "giraffe":       ("cuboid",   0.80,  2.50,  4.50),
    # --- COCO: vehicles & street ---
    "bicycle":       ("cuboid",   0.60,  1.75,  1.10),
    "car":           ("cuboid",   1.80,  4.50,  1.50),
    "motorcycle":    ("cuboid",   0.80,  2.10,  1.20),
    "airplane":      ("cuboid",  30.0,  35.0,  10.0),
    "bus":           ("cuboid",   2.55, 12.0,   3.20),
    "train":         ("cuboid",   3.00, 25.0,   4.00),
    "truck":         ("cuboid",   2.50,  8.00,  3.00),
    "boat":          ("cuboid",   2.00,  6.00,  2.00),
    "traffic light": ("cuboid",   0.30,  0.30,  1.00),
    "fire hydrant":  ("cylinder", 0.30,  0.30,  0.75),
    "stop sign":     ("cuboid",   0.75,  0.05,  2.10),
    "parking meter": ("cuboid",   0.15,  0.15,  1.20),
    "bench":         ("cuboid",   0.55,  1.50,  0.85),
    # --- COCO: accessories & sport ---
    "backpack":      ("cuboid",   0.32,  0.20,  0.45),
    "umbrella":      ("cylinder", 0.06,  0.06,  0.90),
    "handbag":       ("cuboid",   0.32,  0.14,  0.26),
    "tie":           ("cuboid",   0.08,  0.02,  0.55),
    "suitcase":      ("cuboid",   0.45,  0.22,  0.65),
    "frisbee":       ("cylinder", 0.27,  0.27,  0.03),
    "skis":          ("cuboid",   0.12,  1.70,  0.05),
    "snowboard":     ("cuboid",   0.28,  1.50,  0.03),
    "sports ball":   ("sphere",   0.22,  0.22,  0.22),
    "kite":          ("cuboid",   1.00,  0.60,  0.05),
    "baseball bat":  ("cylinder", 0.07,  0.07,  0.85),
    "baseball glove":("cuboid",   0.25,  0.15,  0.30),
    "skateboard":    ("cuboid",   0.21,  0.80,  0.11),
    "surfboard":     ("cuboid",   0.50,  2.10,  0.07),
    "tennis racket": ("cuboid",   0.28,  0.68,  0.03),
    # --- COCO: tabletop (the ones this arm can actually pick) ---
    "bottle":        ("cylinder", 0.068, 0.068, 0.23),
    "wine glass":    ("cylinder", 0.080, 0.080, 0.20),
    "cup":           ("cylinder", 0.080, 0.080, 0.10),
    "pen cup":       ("cylinder", 0.075, 0.075, 0.10),
    "mug":           ("cylinder", 0.085, 0.085, 0.10),
    "fork":          ("cuboid",   0.025, 0.19,  0.012),
    "knife":         ("cuboid",   0.022, 0.22,  0.012),
    "spoon":         ("cuboid",   0.035, 0.18,  0.012),
    "bowl":          ("cylinder", 0.15,  0.15,  0.07),
    "banana":        ("cuboid",   0.045, 0.19,  0.040),
    "apple":         ("sphere",   0.078, 0.078, 0.078),
    "sandwich":      ("cuboid",   0.12,  0.12,  0.05),
    "orange":        ("sphere",   0.075, 0.075, 0.075),
    "broccoli":      ("sphere",   0.12,  0.12,  0.14),
    "carrot":        ("cuboid",   0.035, 0.17,  0.035),
    "hot dog":       ("cuboid",   0.050, 0.16,  0.050),
    "pizza":         ("cylinder", 0.30,  0.30,  0.03),
    "donut":         ("cylinder", 0.095, 0.095, 0.045),
    "cake":          ("cylinder", 0.22,  0.22,  0.10),
    # --- COCO: furniture & appliances ---
    "chair":         ("cuboid",   0.45,  0.45,  0.90),
    "couch":         ("cuboid",   0.90,  2.00,  0.80),
    "potted plant":  ("cylinder", 0.22,  0.22,  0.40),
    "bed":           ("cuboid",   1.50,  2.00,  0.60),
    "dining table":  ("cuboid",   0.90,  1.60,  0.75),
    "toilet":        ("cuboid",   0.38,  0.70,  0.75),
    "microwave":     ("cuboid",   0.50,  0.38,  0.30),
    "oven":          ("cuboid",   0.60,  0.60,  0.85),
    "toaster":       ("cuboid",   0.28,  0.18,  0.20),
    "sink":          ("cuboid",   0.55,  0.45,  0.20),
    "refrigerator":  ("cuboid",   0.70,  0.70,  1.80),
    # --- COCO: electronics & small objects ---
    "tv":            ("cuboid",   1.10,  0.08,  0.65),
    "laptop":        ("cuboid",   0.33,  0.24,  0.02),
    "mouse":         ("cuboid",   0.062, 0.11,  0.038),
    "remote":        ("cuboid",   0.045, 0.16,  0.022),
    "keyboard":      ("cuboid",   0.44,  0.14,  0.025),
    "cell phone":    ("cuboid",   0.072, 0.15,  0.009),
    "book":          ("cuboid",   0.15,  0.22,  0.030),
    "clock":         ("cylinder", 0.25,  0.25,  0.05),
    "vase":          ("cylinder", 0.12,  0.12,  0.25),
    "scissors":      ("cuboid",   0.065, 0.18,  0.010),
    "teddy bear":    ("cuboid",   0.22,  0.15,  0.32),
    "hair drier":    ("cuboid",   0.085, 0.22,  0.22),
    "toothbrush":    ("cuboid",   0.015, 0.19,  0.015),
    # --- handy extras that are not COCO but come up on this table ---
    "pen":           ("cuboid",   0.010, 0.14,  0.010),
    "pencil":        ("cuboid",   0.008, 0.17,  0.008),
    "marker":        ("cylinder", 0.017, 0.017, 0.14),
    "eraser":        ("cuboid",   0.022, 0.055, 0.012),
    "screwdriver":   ("cuboid",   0.028, 0.21,  0.028),
    "tape":          ("cylinder", 0.075, 0.075, 0.025),
    "battery":       ("cylinder", 0.014, 0.014, 0.050),
    "usb stick":     ("cuboid",   0.018, 0.055, 0.009),
    "box":           ("cuboid",   0.10,  0.10,  0.10),
    "can":           ("cylinder", 0.066, 0.066, 0.12),
}
# The 80 COCO names, in order — the "all YOLO classes" preset for the query box.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
# Objects bigger than this in any footprint dimension cannot be on this table —
# used to reject a nonsense measurement, not to reject the detection.
MAX_TABLE_OBJ_M = 0.45
CUBE_EDGE_M = 0.0508       # generic fallback edge for an unlisted label

CLASS_META = {k: {"shape": v[0], "w_m": v[1], "d_m": v[2], "h_m": v[3]}
              for k, v in _CLASS_TABLE.items()}


def class_meta(label):
    """Prior for a label: {shape, w_m, d_m, h_m}. Unlisted labels get a cube guess."""
    return CLASS_META.get(str(label).strip().lower(),
                          {"shape": "cube", "w_m": CUBE_EDGE_M,
                           "d_m": CUBE_EDGE_M, "h_m": CUBE_EDGE_M})


def class_size_m(label):
    """Characteristic width for apparent-size ranging (what the bbox width maps to).

    For an object of unknown yaw the bbox width is somewhere between the footprint's
    minor and major axis, so the geometric mean is the least-wrong single number.
    """
    m = class_meta(label)
    return float(math.sqrt(max(m["w_m"], 1e-3) * max(m["d_m"], 1e-3)))


def class_height_m(label):
    """Vertical extent above the table — used for hover/grasp height."""
    return float(class_meta(label)["h_m"])


# ---------------- monocular size + orientation measurement ----------------
# Stereo depth is OFF on this rig (it crashed the OAK-D mid-run, see main()), so
# size and orientation are measured MONOCULARLY, using the one extra fact we have:
# everything sits on a known plane — the table IS the robot's base plane, z=TABLE_Z0.
#
# That fact turns a picture into metric geometry:
#   * every pixel where the object MEETS THE TABLE (the bottom of its silhouette,
#     column by column) back-projects onto z=TABLE_Z0 at a definite (x, y). Those
#     points are the object's real FOOTPRINT, in metres, in the base frame.
#   * cv2.minAreaRect over that footprint gives width, depth and YAW directly.
#   * the top of the silhouette, intersected with the vertical line through the
#     footprint centre, gives the HEIGHT.
# No object-size assumption enters any of this — the prior is only the fallback.
SEG_RING = 8               # px ring around the bbox sampled as table background
SEG_MIN_FRAC = 0.06        # mask must cover this fraction of the bbox to be usable
# Rays that graze the table are useless for ranging: near the horizon one pixel of
# segmentation noise slides the intersection by many centimetres. Require the ray
# to come down onto the plane at least this steeply — sin(incidence) >= this.
# About 13 deg off the table. Shallow, and shallow rays ARE where the range gets
# unreliable — but raising this to 0.35 (20 deg) rejected 100% of real rays on this
# rig (measure_stats: no_contact_line 69/69), because objects at r~40 cm sit near
# the top of the gripper camera's view and are genuinely seen near-grazing. The
# far-flung ghosts it was meant to stop are better caught by the two gates that say
# what is actually wrong with them — size_vs_prior and out_of_workspace — so this
# stays permissive and those do the rejecting.
MIN_TABLE_INCIDENCE = 0.22
# A TALL object's silhouette is WIDEST AT ITS TOP, not at its base — the top is
# nearer the camera, so perspective spreads it. That means the outermost columns of
# the silhouette are the object's near-vertical SIDE edges, and their bottom pixel
# is somewhere up the side wall, NOT on the table. Back-projecting those onto the
# table plane throws them far outward: measured on the synthetic bench, an 8 cm cup
# came out 11.5 cm across from exactly this. A genuine contact pixel sits on the
# base edge, where the silhouette's lower boundary runs roughly HORIZONTALLY; on a
# side edge it plunges. So reject columns where the lower boundary is steeper than
# this many pixels of drop per pixel across.
MAX_CONTACT_SLOPE = 2.5


def _pixels_to_table(uv, T_base_cam, z_plane, min_incidence=0.0):
    """Back-project an (N,2) array of pixels onto the horizontal plane z=z_plane.

    Returns (points (M,3), keep_mask (N,)). Rays that point up, that meet the plane
    behind the camera or absurdly far away, or that graze it more shallowly than
    min_incidence, are dropped.
    """
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    d = np.stack([(uv[:, 0] - cx0) / fx, (uv[:, 1] - cy0) / fy,
                  np.ones(len(uv))], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d @ np.asarray(T_base_cam[:3, :3], np.float64).T
    o = np.asarray(T_base_cam[:3, 3], np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z_plane - o[2]) / d[:, 2]
    keep = ((d[:, 2] < -max(1e-4, float(min_incidence))) & np.isfinite(t)
            & (t > 0.03) & (t < 1.50))
    return o + d[keep] * t[keep, None], keep


def _silhouette_mask(rgb, bbox):
    """Separate the object from the table inside a YOLO box.

    Colour-distance segmentation, not GrabCut: a ring of pixels just OUTSIDE the
    box is the table, so any pixel inside the box far enough from that background
    colour (in Lab, which is roughly perceptually uniform) is object. Otsu picks
    the cut so it adapts to contrast instead of needing a tuned threshold. This
    costs ~1 ms against GrabCut's ~60 ms, which matters because sense_2d runs
    inside the scan sweep.

    Returns a uint8 mask in FULL-FRAME coordinates, or None.
    """
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    lab = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    # background colour = median of a ring just outside the box (that is table)
    rx1, ry1 = max(0, x1 - SEG_RING), max(0, y1 - SEG_RING)
    rx2, ry2 = min(W, x2 + SEG_RING), min(H, y2 + SEG_RING)
    ring = np.ones((ry2 - ry1, rx2 - rx1), bool)
    ring[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
    ring_px = lab[ry1:ry2, rx1:rx2][ring]
    if ring_px.shape[0] < 40:
        return None
    bg = np.median(ring_px, axis=0)

    roi = lab[y1:y2, x1:x2]
    dist = np.linalg.norm(roi - bg, axis=2)
    dmax = float(dist.max())
    if dmax < 8.0:                       # object is the same colour as the table
        return None
    d8 = np.clip(dist / dmax * 255.0, 0, 255).astype(np.uint8)
    _thr, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    # keep the component that actually covers the box centre — Otsu on a
    # background gradient can light up a corner of the ROI instead of the object
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    cx_r, cy_r = (x2 - x1) // 2, (y2 - y1) // 2
    inner = lbl[max(0, cy_r - 3):cy_r + 4, max(0, cx_r - 3):cx_r + 4]
    inner = inner[inner > 0]
    if inner.size:
        best = int(np.bincount(inner).argmax())
    else:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < SEG_MIN_FRAC * (x2 - x1) * (y2 - y1):
        return None
    full = np.zeros((H, W), np.uint8)
    full[y1:y2, x1:x2] = np.where(lbl == best, 255, 0).astype(np.uint8)
    return full


def _contact_points(mask, bbox, T_base_cam, z_plane):
    """Base-frame footprint points where the object meets the table, plus the
    matching top-of-silhouette pixel for each of those columns.

    For a convex object standing on a plane, the LOWEST object pixel in each image
    column is the point where that column's surface touches the table — so those
    pixels, and only those, can be back-projected onto z=z_plane honestly. Columns
    whose bottom pixel sits on the box's own bottom edge are dropped: the object is
    cut off there and its real contact line is outside the frame.

    Returns (contact_pts (M,3), top_uv (M,2)) column-for-column, so the height
    solve can pair each roof pixel with the floor pixel DIRECTLY BELOW IT rather
    than with the footprint centre — which is what a flat elongated object needs
    (the highest pixel of a lying remote is its far END, not its top face).
    """
    H, W = mask.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    sub = mask[y1:y2, x1:x2] > 0
    cols = np.where(sub.any(axis=0))[0]
    if cols.size < 6:
        return None, None
    bottom = (sub.shape[0] - 1) - np.argmax(sub[::-1, :], axis=0)
    top = np.argmax(sub, axis=0)
    # Clipping means the IMAGE ran out, not the box: a bounding box touches the
    # silhouette on all four sides by construction, so testing the bottom pixel
    # against the box's own floor discards the entire true contact line. (It did
    # exactly that — a cup kept 28 of 147 columns, all of them up on its far rim.)
    if y2 >= H - 2:
        return None, None
    # drop the side-edge columns (see MAX_CONTACT_SLOPE) — keep only the stretch of
    # the lower boundary that is genuinely lying along the object's base
    b = bottom[cols].astype(np.float64)
    slope = np.gradient(b, cols.astype(np.float64))
    flat = np.abs(slope) <= MAX_CONTACT_SLOPE
    if flat.sum() >= 6:
        cols, b = cols[flat], b[flat]
    else:
        b = bottom[cols].astype(np.float64)
    uv_bot = np.stack([cols + x1 + 0.5, b + y1 + 0.5], axis=1)
    pts, ok = _pixels_to_table(uv_bot, T_base_cam, z_plane, MIN_TABLE_INCIDENCE)
    if pts.shape[0] < 6:
        return None, None
    uv_top = np.stack([cols + x1 + 0.5, top[cols] + y1 + 0.5], axis=1)[ok]
    return pts, uv_top


def _solve_height(uv_top, xy, half_along, T_base_cam, z_plane):
    """Height of the object above the table, from the top of its silhouette.

    The highest silhouette pixel is the object's FAR TOP edge — looking down at a
    box you see its top face, and its skyline is the far rim. That rim stands
    vertically above the FAR edge of the footprint, so the ray is walked to the
    vertical line there, not to the one through the centre. Anchoring on the centre
    is what made a 2.2 cm remote measure 8.4 cm: a long object's far edge is half
    its length away, and the ray keeps climbing over that distance.

    half_along is the footprint's half-extent along the horizontal viewing
    direction — i.e. how far the far edge sits behind the centre.
    """
    uv = np.asarray(uv_top, np.float64).reshape(-1, 2)
    if uv.shape[0] == 0:
        return None
    o = np.asarray(T_base_cam[:3, 3], np.float64)
    xy = np.asarray(xy, np.float64)
    view = xy - o[:2]
    n = float(np.linalg.norm(view))
    if n < 1e-6:
        return None
    far_xy = xy + view / n * float(half_along)     # the skyline stands over here

    hs = []
    for k in np.argsort(uv[:, 1])[:max(3, uv.shape[0] // 10)]:   # the highest pixels
        d = np.array([(uv[k, 0] - cx0) / fx, (uv[k, 1] - cy0) / fy, 1.0], np.float64)
        d /= np.linalg.norm(d)
        d = np.asarray(T_base_cam[:3, :3], np.float64) @ d
        denom = float(d[0] ** 2 + d[1] ** 2)
        if denom < 1e-9:
            continue
        t = float((far_xy - o[:2]) @ d[:2] / denom)
        if not (0.03 < t < 1.50):
            continue
        h = float(o[2] + t * d[2] - z_plane)
        if -0.005 < h < 0.60:
            hs.append(h)
    return float(np.median(hs)) if len(hs) >= 3 else None


def _classify_shape(w_m, d_m, h_m, prior):
    """Name the solid from its measured proportions, keeping the class prior when
    the label is one we know (a 'cup' stays a cylinder even if the footprint arc
    came out slightly rectangular)."""
    if prior in ("cylinder", "sphere"):
        return prior
    lo, hi = min(w_m, d_m), max(w_m, d_m)
    if hi < 1e-4:
        return prior
    if hi / max(lo, 1e-4) > 2.5:
        return "cuboid"                       # clearly elongated: pen, knife, book
    if h_m > 1.6 * hi:
        return "cylinder"                     # tall and square-ish on the table
    if abs(h_m - hi) / hi < 0.30:
        return "cube"
    return "cuboid"


# Why measurements get rejected, so "everything says (prior)" is diagnosable
# instead of a mystery. Surfaced in /status as measure_stats.
MEAS_STATS = {}


def _meas_fail(why):
    MEAS_STATS[why] = MEAS_STATS.get(why, 0) + 1
    return None


def measure_object(rgb, bbox, T_base_cam, label, z_plane=None):
    """Measure an object's position, footprint, height and yaw from ONE frame.

    Returns a dict {xy, w_m, d_m, h_m, yaw_deg, shape, measured, rng_m} or None if
    the silhouette solve did not produce something believable — the caller then
    falls back to the class prior (obj_xy_2d).

    'yaw_deg' is the direction of the footprint's MAJOR axis in the base frame,
    normalised to [-90, 90) because a rectangle has no front. By convention d_m is
    the extent ALONG yaw (the long axis) and w_m the extent across it, matching
    both the class priors and the way the map draws footprints.
    """
    if fx <= 0:
        return None
    z_plane = TABLE_Z0 if z_plane is None else float(z_plane)
    mask = _silhouette_mask(rgb, bbox)
    if mask is None:
        return _meas_fail("no_silhouette")
    pts, uv_top = _contact_points(mask, bbox, T_base_cam, z_plane)
    if pts is None:
        return _meas_fail("no_contact_line")

    # WHAT ONE VIEW CAN AND CANNOT SEE. Measured on the synthetic bench:
    #   * the extent ACROSS the sightline is recovered to about a millimetre
    #     (5.1 cm cube -> 5.1, 15 cm book -> 15.0, 4.5 cm remote -> 4.6,
    #     8 cm cup -> 7.9) — the silhouette's width is that extent, full stop.
    #   * the extent ALONG the sightline is NOT observable: the object's far side
    #     is behind the object. It over-reads on tall things (8 cm cup -> 11.5)
    #     and under-reads when the long axis points at the camera (22 cm book ->
    #     16.4, the rest of it hidden).
    # So this returns the across-view width as the measurement and hands it to the
    # map with the direction it was taken along; the map fuses the widths gathered
    # from the different bearings of a scan sweep into the actual footprint (see
    # _fit_rect_from_support). One view can only ever give one caliper reading.
    xy_pts = pts[:, :2].astype(np.float32)
    (cx_f, cy_f), (a, b), ang = cv2.minAreaRect(xy_pts)
    if max(a, b) < 0.004 or max(a, b) > MAX_TABLE_OBJ_M * 2.2:
        return _meas_fail("rect_size")

    prior = class_meta(label)
    cam_xy = np.asarray(T_base_cam[:3, 3], np.float64)[:2]
    view = cam_xy - np.array([cx_f, cy_f])
    n_view = float(np.linalg.norm(view))
    if n_view < 1e-6:
        return _meas_fail("degenerate_view")
    v = view / n_view                       # unit vector back toward the camera
    u = np.array([-v[1], v[0]])             # ACROSS the sightline: the caliper axis

    proj_u = xy_pts.astype(np.float64) @ u
    across = float(proj_u.max() - proj_u.min())
    if not (0.004 < across < MAX_TABLE_OBJ_M):
        return _meas_fail("across_range")
    # SANITY-CHECK THE MEASUREMENT AGAINST WHAT THE CLASS IS. A 5.1 cm cube coming
    # out 0.9 cm or 9.3 cm wide is a broken silhouette, not a surprising cube, and
    # letting those through is what scattered one-off ghosts across the map. The
    # band is deliberately wide (the prior is only a guess) — it rejects nonsense,
    # not disagreement. Outside it, the caller falls back to the prior path.
    p_size = float(math.sqrt(max(prior["w_m"], 1e-3) * max(prior["d_m"], 1e-3)))
    if not (0.35 * p_size < across < 2.8 * p_size):
        return _meas_fail("size_vs_prior")

    # The contact arc is the NEAR side of the footprint, so its centroid sits about
    # half a depth too close to the camera. We do not know the depth yet, so push
    # back by half the across-width (a circle's worth) — the map's multi-bearing
    # average then cancels most of what this leaves behind.
    xy = np.array([proj_u.mean() * u[0], proj_u.mean() * u[1]], np.float64) \
        + v * float((xy_pts.astype(np.float64) @ v).mean()) - v * (0.5 * across)
    if not (MAP_R_MIN < float(np.hypot(*xy)) < MAP_R_MAX):
        return _meas_fail("out_of_workspace")
    rng = float(np.linalg.norm(cam_xy - xy))

    # HEIGHT NEEDS TO KNOW HOW FAR BACK THE OBJECT'S FAR EDGE IS, and using the
    # across-width for that is wrong for anything elongated. A pen pointing along
    # the sightline measures across=1.8cm (correctly - that is its width), but it
    # extends ~7cm away from us, so anchoring the roof ray 0.9cm behind centre walks
    # it nowhere near far enough and the height over-reads: measured, a flat pen
    # came out 4.5cm tall and got classified as a standing cylinder.
    #
    # We do not know the along-view extent from one view, so use the best estimate
    # available - the class prior's long axis - and never less than the across
    # width. Errors here stay second-order because the roof ray is steep.
    prior_long = max(prior["w_m"], prior["d_m"])
    half_along = 0.5 * max(across, min(prior_long, MAX_TABLE_OBJ_M))
    h_m = _solve_height(uv_top, xy, half_along, T_base_cam, z_plane)
    if h_m is None or not (0.002 < h_m < 0.60):
        h_m = float(prior["h_m"])
    # And do not let a single view claim an object is TALL when its footprint was
    # never determined: h > footprint reads as "standing up", which for a pen seen
    # end-on is exactly the wrong conclusion.
    if h_m > 1.5 * max(across, 1e-3) and prior_long > 2.2 * min(prior["w_m"], prior["d_m"]):
        h_m = float(prior["h_m"])

    # SANITY-CHECK THE SOLID AGAINST WHAT THE CLASS IS, and fall back to the prior
    # rather than publish a shape that cannot be true. From ONE viewpoint an
    # elongated object pointing along the sightline has no measurable length -
    # `across` is its WIDTH, correctly measured, and the 14 cm of a pen is simply
    # invisible. Left alone that produced "pen: 2.6 x 2.6 x 3.6 cm", i.e. a stubby
    # thing STANDING UP, which is worse than admitting we do not know: the map drew
    # a confident orientation for an object whose shape it had not determined.
    # Recovering the real footprint needs several bearings (the support fusion in
    # world2d_update), which a Scan sweep provides and idle sensing does not.
    if h_m > 2.5 * max(prior["h_m"], 0.003):
        return _meas_fail("height_vs_prior")
    MEAS_STATS["ok"] = MEAS_STATS.get("ok", 0) + 1
    yaw = math.degrees(math.atan2(u[1], u[0]))
    return {"xy": xy, "across_m": across, "u_deg": yaw, "h_m": float(h_m),
            "w_m": across, "d_m": max(across, min(prior["w_m"], prior["d_m"])),
            "yaw_deg": float(((yaw + 90.0) % 180.0) - 90.0),
            "shape": prior["shape"], "measured": True, "rng_m": rng}


# ---- fusing per-bearing caliper readings into a footprint ----
# Each view measures the footprint's width along ONE direction (its across-view
# axis). That is the footprint's SUPPORT WIDTH along that direction, and a convex
# shape is determined by its support widths — so a scan sweep, which sees each
# object from a spread of bearings, measures the whole footprint between them.
SUP_BINS = 12              # direction bins over 180 deg (15 deg each)
SUP_MIN_BINS = 3           # fit a rectangle only once this many bearings are in


def _sup_bin(u_deg):
    return int(((float(u_deg) % 180.0) / 180.0) * SUP_BINS) % SUP_BINS


def _fit_rect_from_support(sup):
    """Least-squares rectangle through the accumulated support widths.

    A rectangle with half-sides (a, b) at yaw phi has support width
        s(theta) = 2a|cos(theta - phi)| + 2b|sin(theta - phi)|
    Sweep phi over 1 deg steps; for each, a and b fall out of a 2x2 linear solve.
    Keep the phi with the smallest residual. Returns (w_m, d_m, yaw_deg) with d_m
    the long side and yaw along it, or None when too few bearings have been seen.
    """
    obs = [(math.radians((k + 0.5) * 180.0 / SUP_BINS), s)
           for k, s in sorted(sup.items()) if s > 0]
    if len(obs) < SUP_MIN_BINS:
        return None
    th = np.array([o[0] for o in obs])
    s = np.array([o[1] for o in obs])
    best = None
    for phi_deg in range(0, 180):
        phi = math.radians(phi_deg)
        A = np.stack([np.abs(np.cos(th - phi)), np.abs(np.sin(th - phi))], axis=1)
        try:
            x, *_ = np.linalg.lstsq(A, s, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if x[0] <= 0 or x[1] <= 0:
            continue
        r = float(np.linalg.norm(A @ x - s))
        if best is None or r < best[0]:
            best = (r, float(x[0]), float(x[1]), phi_deg)
    if best is None:
        return None
    _r, e1, e2, phi_deg = best
    # e1 is the extent along phi, e2 across it; report the long side as d/yaw
    if e1 >= e2:
        d_m, w_m, yaw = e1, e2, phi_deg
    else:
        d_m, w_m, yaw = e2, e1, phi_deg + 90.0
    if not (0.004 < w_m < MAX_TABLE_OBJ_M and 0.004 < d_m < MAX_TABLE_OBJ_M):
        return None
    return float(w_m), float(d_m), float(((yaw + 90.0) % 180.0) - 90.0)
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
    # The BASE carries the whole arm's inertia and is what visibly jerks, so it gets
    # its own slower rate rather than sharing the reach rate.
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


# ---------------- locate from ONE fixed pose ----------------
# WHY THIS EXISTS. The hand-eye rotation is wrong, and a rotation error ROTATES
# WITH THE ARM: the same stationary cube localizes anywhere from r=31cm to r=92cm
# across a scan sweep. The 2D map then AVERAGES those fixes, so it averages an
# error that is different in every sample and the answer smears - which is why the
# grasp missed in a different place each time and why no trim could fix it.
#
# Localizing from ONE fixed pose collapses that: the error stops varying and
# becomes a CONSTANT offset, which a single trim can absorb (or the final pixel
# centring can). Same broken TF, but now it is broken the same way every time.
#
# So: swing to SURVEY_POSE, take several reads of the object from exactly there,
# take the median, and hand that one coordinate to the approach. No averaging
# across viewpoints, no multi-pose map fusion in the path that decides the grasp.
# A CANONICAL SURVEY CONFIGURATION, with the base free to turn.
#
# Pinning ALL FIVE joints (the first version of this) fixes the error nicely but
# then the object has to happen to be in that one frame - measured live: the arm
# reached the pose and got "0 good reads" because the cube simply was not in view.
#
# What actually matters is the camera's TILT, which is set by lift/elbow/wrist: the
# range error comes from the sightline being too steep or too shallow. Base pan only
# swings the bearing, and bearing is the well-conditioned part. So hold joints 2-5
# at the canonical values and let joint 1 turn to face the object. The range error
# stays constant (same tilt every time) while the view can still cover the table.
# Use HOME's tilt, NOT VIEW's. The constant is named VIEW but nothing ever sensed
# from it: idle_view and scan_2d both sense from whatever pose the arm is in, which
# after startup is HOME, and the scan only changes pan. Measured - surveying at
# VIEW's tilt gave "0 good reads" three times in a row while the map, built at
# HOME's tilt, saw the same cube at r=36.8cm. VIEW's lift is +37 against HOME's
# -99, so the camera is pointing somewhere else entirely.
SURVEY_TILT = HOME[1:].copy()   # lift, elbow, wrist_flex, wrist_roll - always these
SURVEY_READS = 9                # reads to median over (rejects detector jitter)
SURVEY_SPREAD_MAX = 0.05        # m; if reads disagree by more than this from ONE
                                # pose the detector is unstable - say so rather
                                # than averaging noise into a confident answer
SURVEY_PAN_LIMIT = 100.0


def _visible_now(label, finder=None, tries=4):
    """Is the object in frame, unclipped, right now?"""
    for _ in range(tries):
        j, rgb, _ = observe()
        tr = finder(rgb, T_cam_of(j)) if finder else find_label(rgb, label, T_cam_of(j))
        if tr is not None and not tr.clipped:
            return True
        time.sleep(0.12)
    return False


def survey_pose_for(bearing_deg=None):
    """Canonical tilt, base turned toward `bearing_deg`.

    With no bearing this defaults to HOME's pan, NOT the arm's current pan. Using
    the current pan made failures compound: a survey that could not find the object
    swept the base looking for it, left it 48 deg off, and the next survey started
    from there and swept further - the arm ratcheted away from the table until it
    was staring at bare wood and even the idle map went empty.
    """
    pan = float(HOME[0]) if bearing_deg is None else float(bearing_deg)
    return np.concatenate(([np.clip(pan, -SURVEY_PAN_LIMIT, SURVEY_PAN_LIMIT)],
                           SURVEY_TILT))


def locate_from_survey(label, finder=None, bearing_deg=None):
    """The object's (x, y) from the canonical survey configuration. Returns
    (xy, spread) or (None, reason). Does NOT touch the 2D map - the map is the
    operator's picture of the table; this is what the grasp aims at.

    bearing_deg turns the base to face the object first. The 2D map is a fine
    source for that even though its RANGE is unreliable: a rotation error spoils
    range far more than it spoils bearing.
    """
    if bearing_deg is None:
        rough = _mapped_xy(label)
        if rough is not None:
            bearing_deg = math.degrees(math.atan2(rough[1], rough[0]))
            say(f"survey: turning to face {label} at {bearing_deg:+.0f}deg "
                f"(rough bearing from the map)")
    set_phase("LOCATE", f"surveying for {label} from the canonical pose")
    q_start = observe(overlay=False)[0].astype(np.float64).copy()
    goto_smooth(survey_pose_for(bearing_deg), settle=0.45)

    # If it is not in frame from there, sweep the base a little - same tilt, so the
    # range error stays constant, we are only looking around.
    if not _visible_now(label, finder):
        base = float(bearing_deg) if bearing_deg is not None else float(HOME[0])
        for dpan in (-14.0, 14.0, -28.0, 28.0):
            checkpoint()
            goto_smooth(survey_pose_for(base + dpan), settle=0.3)
            if _visible_now(label, finder):
                say(f"survey: found {label} after turning {dpan:+.0f}deg")
                break
    fixes = []
    for _ in range(SURVEY_READS):
        checkpoint()
        j, rgb, _ = observe()
        T = T_cam_of(j)
        tr = finder(rgb, T) if finder else find_label(rgb, label, T)
        if tr is None or tr.clipped:
            time.sleep(0.08)
            continue
        m = measure_object(rgb, tr.bbox_xyxy, T, label)
        if m is not None:
            xy = _rotate_xy(m["xy"], MAP_BEARING_OFFSET_DEG[0])
        else:
            xy, _st, _sz = obj_xy_2d(tr.bbox_xyxy, T, z_m=None, label=label)
            if xy is None:
                time.sleep(0.05)
                continue
        fixes.append(np.asarray(xy, np.float64))
        time.sleep(0.05)

    if len(fixes) < 3:
        # leave the arm where the survey began, so a failure does not move the
        # camera off the table for whatever runs next
        try:
            goto_smooth(q_start, settle=0.3)
        except Exception:
            pass
        return None, f"only {len(fixes)} good reads of '{label}' from the survey pose"
    arr = np.array(fixes)
    xy = np.median(arr, axis=0)
    spread = float(np.max(np.linalg.norm(arr - xy, axis=1)))
    say(f"survey: {len(fixes)} reads -> x={xy[0]*100:+.1f} y={xy[1]*100:+.1f} cm "
        f"(r={np.hypot(*xy)*100:.1f}cm), spread {spread*100:.1f}cm")
    if spread > SURVEY_SPREAD_MAX:
        say(f"survey: reads disagree by {spread*100:.1f}cm from ONE pose — "
            f"detector is unstable, treat this fix as rough")
    return xy, spread


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


def _picked_height(label, gx, gy):
    """Height of the object we just grasped, for growing the destination's height
    after the place. Prefers the mapped measurement nearest the grasp point, and
    marks that entry so the place can retire it. Falls back to the class prior."""
    with w2d_lock:
        best, bd = None, 0.10
        for t, o in W2D["objs"].items():
            d = float(np.hypot(o["xy"][0] - gx, o["xy"][1] - gy))
            if d < bd and label.split()[0] in o["label"]:
                best, bd = t, d
        if best is not None:
            W2D["objs"][best]["picked"] = True
            return float(max(W2D["objs"][best]["h_m"], 0.005))
    for cand in (label, f"{label} cube"):
        if str(cand).strip().lower() in CLASS_META:
            return class_height_m(cand)
    return CUBE_EDGE_M


def _dest_geometry(tag=None, xy=None):
    """Resolve a placement destination to (xy, top_height, label).

    A mapped object's top is its measured height, floored by the class prior — the
    measurement reads low, and low here means burying the carried object in the
    target. A bare table spot has a top of zero.
    """
    if tag is not None:
        with w2d_lock:
            o = W2D["objs"].get(int(tag))
            if o is None:
                raise Abort(f"destination tag {tag} is not on the 2D map")
            return (np.array(o["xy"], np.float64),
                    float(max(o["h_m"], class_height_m(o["label"]))),
                    str(o["label"]))
    if xy is None:
        raise Abort("no destination given")
    return np.asarray(xy, np.float64), 0.0, None


def place_at(tag=None, xy=None, recenter=True):
    """Put the carried object down on a mapped object, or on a clicked table spot.

    Blind from the map to get close, then ONE visual correction before descending —
    the map is good enough to arrive over the destination, not good enough to land
    on a 5 cm cube. See PLACE_CLEAR_M for the release-height derivation.
    """
    with lock:
        held, g2b, h_carry, carry_label = (carry["held"], carry["grip_to_bottom"],
                                           carry["h_m"], carry["label"])
    if not held:
        raise Abort("nothing in the jaws — pick something first")

    dest_xy, h_top, dest_label = _dest_geometry(tag, xy)
    z_rel = TABLE_Z0 + h_top + g2b + PLACE_CLEAR_M
    z_hov = z_rel + PLACE_HOVER_M
    where = (f"{dest_label} #{tag}" if tag is not None
             else f"({dest_xy[0]*100:.0f},{dest_xy[1]*100:.0f})cm")
    say(f"place {carry_label} on {where}: dest top={h_top*100:.1f}cm "
        f"+ grip->bottom {g2b*100:.1f}cm -> release tip z={z_rel*100:.1f}cm")

    # ---- 1. lift to transit height, straight up, before going anywhere ----
    set_phase("PLACE", "lifting to transit height")
    q = observe()[0].astype(np.float64)
    j5, tip = float(q[4]), _tip(q)
    pitch, e = plan_grasp_pitch(np.array([dest_xy[0], dest_xy[1], z_rel]), q)
    if pitch is None:
        raise Abort(f"destination {where} out of reach (best IK {e*1e3:.0f}mm, "
                    f"r={np.hypot(*dest_xy)*100:.0f}cm)")
    z_tr = max(PLACE_TRANSIT_Z, z_hov)
    if _move_tip(np.array([tip[0], tip[1], z_tr]), pitch, j5, settle=0.18, step=1.6) is None:
        say("place: could not lift to transit height — traversing from here")

    # ---- 2. blind traverse to the mapped destination, still high ----
    set_phase("PLACE", f"traversing to {where}")
    if _move_tip(np.array([dest_xy[0], dest_xy[1], z_tr]), pitch, j5,
                 settle=0.20, step=1.6) is None:
        raise Abort(f"cannot traverse to {where}")
    if _move_tip(np.array([dest_xy[0], dest_xy[1], z_hov]), pitch, j5,
                 settle=0.20, step=1.4) is None:
        say("place: could not drop to hover height — correcting from transit height")

    # ---- 3. ONE visual correction, from the hover ----
    # The carried object hangs in front of the camera, so the destination is only
    # visible from up here, past it. This is also why the correction happens once,
    # at hover, rather than as a servo all the way down.
    place_xy = dest_xy
    if recenter and dest_label:
        set_phase("PLACE", f"re-centering on {where}")
        finder = (lambda rgb, T=None, _l=dest_label: find_label(rgb, _l, T))
        aligned = _center_on_cube(finder, pitch, j5)
        if aligned is None:
            say(f"place: {dest_label} not visible from the hover — "
                f"using the mapped position blind")
        else:
            moved = float(np.linalg.norm(np.asarray(aligned) - dest_xy))
            if moved <= 0.06:
                place_xy = np.asarray(aligned, np.float64)
                say(f"place: re-centred {moved*100:.1f}cm off the mapped spot")
            else:
                say(f"place: re-centre jumped {moved*100:.1f}cm — rejected, staying blind")

    # ---- 4. descend to the release height ----
    q = observe()[0].astype(np.float64)
    z0 = float(_tip(q)[2])
    for f in (0.5, 1.0):
        checkpoint()
        z = float(z0 + (z_rel - z0) * f)
        set_phase("PLACE", f"descending to z={z*100:.1f}cm")
        if _move_tip(np.array([place_xy[0], place_xy[1], z]), pitch, j5,
                     settle=0.18, step=1.0) is None:
            say("place: could not reach release height — releasing from here")
            break

    # ---- 5. release, then retreat straight up so the jaws do not drag it ----
    set_phase("PLACE", "releasing")
    send_joints(observe()[0], gripper=PLACE_OPEN_PCT)
    time.sleep(0.35)
    _set_carry(False)
    set_phase("PLACE", "retreating")
    ee_move_rel([0, 0, PLACE_RETREAT_M], settle=0.25)

    # ---- 6. the stack got taller: record it, or the next place lands INSIDE it ----
    if tag is not None and h_carry > 0:
        with w2d_lock:
            o = W2D["objs"].get(int(tag))
            if o is not None:
                o["h_m"] = float(h_top + h_carry)
                o["t"] = time.time()
        say(f"map: {where} is now {(h_top + h_carry)*100:.1f}cm tall")
    if carry_label:
        # the carried object is no longer where it was picked from
        with w2d_lock:
            for t, o in list(W2D["objs"].items()):
                if o["label"] == carry_label and o.get("picked"):
                    del W2D["objs"][t]
    set_phase("DONE", f"placed {carry_label} on {where}")


TARGET_RIGHT_TRIM_M = 0.050 # shift the APPROACH hover target this far to the
                            # cube's RIGHT, so the cube stays on the LEFT side of the
                            # camera view during approach.
TARGET_BACK_M = 0.010       # stop the approach hover target this far SHORT of the
                            # cube radially. Tunable in the UI; 1 cm default.
APPROACH_STEPS = 3          # step 1 closes ~90% of the gap, the rest are small
                            # corrections. Tried 2 with a full-distance first move
                            # and it missed more, so this is back to 3.


def run_mission(target_label=None):
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

    def _approach_target(cube_xy):
        """Hover position for the approach: short of the cube and to its right.

        TARGET_BACK_M: stop this many metres radially BEFORE the cube (keeps it in view).
        TARGET_RIGHT_TRIM_M: shift this far to the cube's right (tunable in the UI).
        """
        cube_xy = np.asarray(cube_xy, dtype=np.float64)
        r = float(np.hypot(*cube_xy))
        if r > 1e-6:
            backed = cube_xy * ((r - TARGET_BACK_M) / r)
        else:
            backed = cube_xy.copy()
        return _shift_right(backed, TARGET_RIGHT_TRIM_M)

    try:
        stop_flag.clear()
        with lock:
            state["running"] = True
            state["t0"] = time.time()
        finder, tracker, label = _target_finder(target_label)
        say("=" * 52)
        say(f"PICK START  target='{label}'  query='{state.get('query')}'")
        q_now = observe()[0]
        say(f"  [1/7] pose      joints={np.round(q_now,1).tolist()} "
            f"tip={np.round(_tip(q_now)*100,1).tolist()}cm")

        # Locate from ONE fixed pose (see locate_from_survey) instead of taking the
        # map's average across viewpoints. Falls back to the map if the survey
        # cannot see the object at all, so a scan is still useful.
        cube_xy = None
        if USE_SURVEY_LOCATE[0]:
            cube_xy, spread = locate_from_survey(label, finder)
            if cube_xy is None:
                say(f"survey failed ({spread}); falling back to the 2D map")
        if cube_xy is None:
            cube_xy = _mapped_xy(label)
        if cube_xy is None:
            how = "the survey pose or " if USE_SURVEY_LOCATE[0] else ""
            raise Abort(f"'{label}' is not on the 2D map (looked in {how}the map). "
                        f"Press 'Scan -> 2D map', and check the query names it — "
                        f"the query is currently '{state.get('query')}'")
        say(f"  [2/7] target    x={cube_xy[0]*100:+.1f} y={cube_xy[1]*100:+.1f} cm  "
            f"r={np.hypot(*cube_xy)*100:.1f}cm "
            f"bearing={math.degrees(math.atan2(cube_xy[1], cube_xy[0])):+.0f}deg")
        say(f"  [3/7] approach   {APPROACH_STEPS} stages, "
            f"trim={TARGET_RIGHT_TRIM_M*100:.1f}cm right, back={TARGET_BACK_M*100:.1f}cm, "
            f"hover z={PICK_HOVER_Z*100:.0f}cm")
        say("        opening the gripper")
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
            # Step 1 closes 90% of the gap, then the remaining stages make small
            # corrections. REVERTED to this after trying a single full-distance
            # move: going 100% in one go was worse - it arrives with no margin left
            # to correct, so any residual localization error lands as a miss. 90%
            # then small re-centring leaves room to fix things while still close.
            step_len = dist_xy * frac if last else min(dist_xy * frac, 0.12)
            step_xy = (delta_xy / dist_xy * step_len) if dist_xy > 1e-6 else np.zeros(2)
            wx = float(tip[0] + step_xy[0])
            wy = float(tip[1] + step_xy[1])

            pitch, e = plan_grasp_pitch(
                np.array([cube_xy[0], cube_xy[1], PICK_GRASP_Z]), q)
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
        say(f"  [4/7] centering  pitch={gp:.0f}deg wrist_roll={j5:.0f}deg "
            f"aim=({HAND_UV[0]+AIM_DU:.0f},{HAND_UV[1]+AIM_DV:.0f})px")
        aligned = _center_on_cube(finder, gp, j5)
        if aligned is not None:
            gx, gy = float(aligned[0]), float(aligned[1])
            say(f"        centred -> grasp at x={gx*100:+.1f} y={gy*100:+.1f} cm "
                f"(moved {np.linalg.norm(np.array([gx,gy])-cube_xy)*100:.1f}cm from the map fix)")
        else:
            gx, gy = float(cube_xy[0]), float(cube_xy[1])
            say("        centring FAILED - falling back to the mapped position")
        q = observe()[0].astype(np.float64)
        z0 = float(_tip(q)[2])
        for f in (0.4, 0.75, 1.0):
            checkpoint()
            z = float(z0 + (PICK_GRASP_Z - z0) * f)
            last = (f == 1.0)
            set_phase("PICK", f"descending to z={z*100:.1f}cm")
            # the final rung goes slower and settles longer — that is the one that
            # has to land accurately
            used = _move_tip(np.array([gx, gy, z]), gp, j5,
                             settle=0.18 if last else 0.12,
                             step=0.9 if last else 1.4)
            if used is None:
                say(f"  [5/7] descend   z={z*100:.1f}cm UNREACHABLE - closing from here")
                break
            say(f"  [5/7] descend   {int(f*100):3d}%  z={z*100:5.1f}cm  "
                f"pitch={used:.0f}deg  tip={np.round(_tip(observe(overlay=False)[0])*100,1).tolist()}cm")

        set_phase("PICK", "closing the gripper")
        # Measure the fingertip height at the INSTANT of the grasp, before lifting.
        # The object's underside is on the table right now, so this one number is
        # the fingertip->underside distance for the whole carry (see PLACE_CLEAR_M).
        z_grasp = float(_tip(observe(overlay=False)[0])[2])
        say(f"  [6/7] grasp     closing at tip z={z_grasp*100:.1f}cm, "
            f"watching the gripper current")
        held, i_idle = close_with_current(step=4.0, delay=0.08)
        say(f"        {'CONTACT' if held else 'NO CONTACT'} "
            f"(idle current {i_idle:.0f})")
        if held:
            g2b = max(0.002, z_grasp - TABLE_Z0)
            h_obj = _picked_height(label, gx, gy)
            _set_carry(True, label=label, h_m=h_obj, grip_to_bottom=g2b)
            say(f"holding {label}: grip->bottom {g2b*100:.1f}cm, "
                f"object height {h_obj*100:.1f}cm")
        else:
            _set_carry(False)
        set_phase("PICK", "lifting")
        say(f"  [7/7] lift      +{PICK_LIFT_M*100:.0f}cm")
        ee_move_rel([0, 0, PICK_LIFT_M], settle=0.25)
        say(f"PICK {'SUCCESS' if held else 'FAILED - closed on air'}  "
            f"tip={np.round(_tip(observe(overlay=False)[0])*100,1).tolist()}cm")
        say("=" * 52)
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
                holding = carry["held"]
            # Never relax a grip we believe is holding something: the current read
            # returning None used to be enough to open the jaws and drop the object
            # mid-sequence, because `c is None` takes the relax branch.
            if g is not None and g < 15.0 and not holding:
                c = gripper_current()
                if c is None or abs(c) < 4.0:      # not carrying load -> relax
                    send_joints(observe(overlay=False)[0], gripper=40.0)
        except Exception:
            pass
        with lock:
            state["running"] = False


# ---------------- web ----------------
app = Flask(__name__)

# ================= guest access =================
# The public tunnel exposes this ONE port, so the admin UI would be on the open
# internet too if nothing stopped it. The security boundary is therefore the Host
# header: a request that arrived on the tunnel hostname is "public" and may only
# touch the handful of endpoints in GUEST_ENDPOINTS, and only while holding the
# active session cookie. Everything else — calibration, tuning, the task engine,
# the map — is refused for public requests no matter what it asks for.
#
# Sessions are first-come-first-served so a QR code on the wall never has to
# change: whoever opens /guest while it is free claims the arm for GUEST_MINUTES,
# everyone else sees a countdown until it frees up.
GUEST_MINUTES = 5.0
GUEST_COOLDOWN_MIN = 10.0     # after your turn, how long before you may claim again
QUEUE_TTL = 25.0              # a waiter who stops polling for this long drops out
PUBLIC_HOST = [None]          # set once the public tunnel reports its URL

# THE QUEUE. One printed QR points everyone at /guest, which is a LOBBY: it holds
# your place and tells you how long until your turn. A visitor is identified by a
# long-lived cookie, backed up by their IP so clearing cookies does not jump the
# line. Everything below lives in memory only and is never written to disk — an IP
# is personal data and we keep it exactly as long as it takes to be fair.
#
# The cooldown is deliberately CONDITIONAL: it only applies when somebody else is
# actually waiting. Alone with the robot you can keep playing; the moment a queue
# forms, a repeat visitor goes behind the people who have not had a turn yet.
guest = {"token": None, "expires": 0.0, "started": 0.0}
visitors = {}                 # vid -> {"ip", "last_end", "turns", "seen"}
queue = []                    # vids waiting, front of list = next up
guest_lock = threading.Lock()


# PERSISTENT VISITOR LOG. This writes IP addresses to disk, which earlier versions
# deliberately did not do — enabled on request so the operator can see who has been
# driving the arm and settle "who is on it right now" without guessing. One
# append-only JSONL file next to the script; delete it and the history is gone.
GUEST_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guest_log.jsonl")
guest_log_lock = threading.Lock()


def _log_guest(event, vid, **extra):
    rec = {"t": round(time.time(), 1),
           "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
           "event": event, "vid": vid,
           "ip": visitors.get(vid, {}).get("ip", "?")}
    rec.update(extra)
    try:
        with guest_log_lock:
            with open(GUEST_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    except Exception as e:
        say(f"guest log write failed: {e}")


def _guest_history(limit=400):
    """Recent log lines, newest first, plus a per-IP roll-up."""
    rows = []
    try:
        with guest_log_lock:
            with open(GUEST_LOG, encoding="utf-8") as f:
                lines = f.readlines()[-limit:]
        for ln in lines:
            try:
                rows.append(json.loads(ln))
            except ValueError:
                continue
    except FileNotFoundError:
        pass
    by_ip = {}
    for r in rows:
        ip = r.get("ip", "?")
        e = by_ip.setdefault(ip, {"ip": ip, "turns": 0, "secs": 0.0, "last": 0})
        if r.get("event") == "turn_end":
            e["turns"] += 1
            e["secs"] += float(r.get("held_s") or 0)
        e["last"] = max(e["last"], r.get("t", 0))
    return rows, sorted(by_ip.values(), key=lambda e: -e["last"])


def _client_ip():
    """The visitor's real IP. Behind cloudflared/Funnel the socket peer is the
    tunnel, so prefer the forwarded headers those set."""
    for h in ("CF-Connecting-IP", "X-Forwarded-For"):
        v = request.headers.get(h)
        if v:
            return v.split(",")[0].strip()
    return request.remote_addr or "?"


def _visitor(resp_cookies=None):
    """Stable id for this phone. Returns (vid, is_new)."""
    vid = request.cookies.get("rax_id")
    new = False
    if not vid or len(vid) < 8:
        vid, new = secrets.token_urlsafe(12), True
    fresh = vid not in visitors
    v = visitors.setdefault(vid, {"ip": _client_ip(), "last_end": 0.0,
                                  "turns": 0, "seen": 0.0,
                                  "first": time.time()})
    v["seen"] = time.time()
    v["ip"] = _client_ip()
    if fresh:
        _log_guest("scanned", vid, ua=(request.headers.get("User-Agent") or "")[:90])
    return vid, new


def _prune(now):
    """Drop waiters who stopped polling. Caller holds guest_lock."""
    global queue
    queue = [v for v in queue
             if v in visitors and now - visitors[v]["seen"] < QUEUE_TTL]


def _cooling(vid, now):
    """Seconds of cooldown left for this visitor, 0 if none."""
    v = visitors.get(vid)
    if not v or not v["turns"]:
        return 0.0
    return max(0.0, v["last_end"] + GUEST_COOLDOWN_MIN * 60.0 - now)


def _guest_cleanup():
    """Leave the arm safe and tidy for the next visitor: drop whatever is in the
    jaws and fold back to HOME. Runs on its own thread — _end_turn is called with
    guest_lock held, and this takes seconds."""
    def _t():
        try:
            stop_flag.set()
            time.sleep(0.4)          # let any running mission notice and unwind
            stop_flag.clear()
            with lock:
                state["running"] = True
            try:
                set_phase("RESET", "guest turn over — opening and folding home")
                # The jog loop normally owns the gripper (it eases grip_cmd toward
                # grip_target) but it parks itself while a mission is running, so
                # nothing would move it here. Drive it ourselves at the same rate,
                # and leave grip_target/grip_cmd in sync so the loop does not snap
                # the jaws back when the next guest starts jogging.
                grip_target[0] = 95.0
                t_open = time.time()
                while time.time() - t_open < 4.0:
                    g = grip_cmd[0] + GRIP_RATE * 0.05
                    grip_cmd[0] = float(min(95.0, g))
                    send_joints(observe(overlay=False)[0], gripper=grip_cmd[0])
                    if grip_cmd[0] >= 94.9:
                        break
                    time.sleep(0.05)
                say(f"guest cleanup: gripper opened to {grip_cmd[0]:.0f}%")
                _set_carry(False)
                # goto_smooth holds whatever the gripper reads at its start, which
                # is now open — so the fold cannot re-close it.
                goto_smooth(HOME, settle=0.3)
                set_phase("IDLE", "ready for the next guest")
            finally:
                with lock:
                    state["running"] = False
        except Exception as e:
            say(f"guest cleanup failed: {type(e).__name__}: {e}")
    threading.Thread(target=_t, daemon=True).start()


def _end_turn(now):
    """Finish the active session and record the cooldown. Caller holds the lock."""
    tok = guest["token"]
    if tok:
        for vid, v in visitors.items():
            if v.get("token") == tok:
                v["last_end"], v["turns"] = now, v["turns"] + 1
                _log_guest("turn_end", vid, turns=v["turns"],
                           held_s=round(now - guest["started"], 1))
                break
        _guest_cleanup()
    guest.update(token=None, expires=0.0, started=0.0)


def _promote(now):
    """Hand the arm to the front of the queue. Caller holds the lock."""
    _prune(now)
    while queue:
        vid = queue[0]
        # someone still cooling down yields to anyone who has not played yet
        if _cooling(vid, now) > 0 and any(_cooling(o, now) <= 0 for o in queue[1:]):
            queue.append(queue.pop(0))
            continue
        queue.pop(0)
        tok = secrets.token_urlsafe(16)
        guest.update(token=tok, expires=now + GUEST_MINUTES * 60.0, started=now)
        visitors[vid]["token"] = tok
        _log_guest("turn_start", vid, waiting=len(queue),
                   turns=visitors[vid].get("turns", 0))
        say(f"guest turn started ({GUEST_MINUTES:.0f} min) — "
            f"{visitors[vid]['ip']} — {len(queue)} waiting")
        return vid
    return None


def lobby_tick(vid):
    """Register/refresh this visitor's place and report where they stand.

    This is the ONE call the lobby page makes. It expires the running turn, hands
    the arm to whoever is next, and keeps the queue swept — so the whole system is
    driven by visitors polling, with no background thread to get out of sync.
    """
    now = time.time()
    with guest_lock:
        if guest["token"] and now >= guest["expires"]:
            _end_turn(now)
        _prune(now)
        mine = visitors.get(vid, {}).get("token")
        playing = bool(guest["token"]) and mine == guest["token"]

        if not playing:
            if vid not in queue:
                # a repeat visitor joins behind everyone still on their first turn
                if _cooling(vid, now) > 0:
                    queue.append(vid)
                else:
                    at = len(queue)
                    for i, o in enumerate(queue):
                        if _cooling(o, now) > 0:
                            at = i
                            break
                    queue.insert(at, vid)
            if not guest["token"]:
                got = _promote(now)
                playing = (got == vid)

        pos = 0 if playing else (queue.index(vid) + 1 if vid in queue else 0)
        left = max(0.0, guest["expires"] - now) if guest["token"] else 0.0
        ahead = max(0, pos - 1)
        eta = left + ahead * GUEST_MINUTES * 60.0 if pos else 0.0
        return {"playing": playing, "position": pos, "waiting": len(queue),
                "left": round(left), "eta": round(eta),
                "cooldown": round(_cooling(vid, now)),
                "minutes": GUEST_MINUTES,
                "token": guest["token"] if playing else None}

# View-function names a public visitor may reach. Deliberately tiny: look, drive,
# pick, stop. No parameters, no calibration, no map, no placing.
GUEST_ENDPOINTS = {
    "index", "guest_page", "guest_state_route", "ui_asset",
    "stream", "guest_status", "jogpress", "jogrelease", "jogvec",
    "guest_pick", "guest_scan", "guest_query", "guest_targets",
    "guest_home", "feedback", "stop",
    "urdf_route", "geom",          # read-only, needed by the shared 3D viewer
}


def is_public_request():
    """Did this request come in over the public tunnel (rather than LAN/Tailscale)?"""
    pub = PUBLIC_HOST[0]
    if not pub:
        return False
    return (request.host or "").split(":")[0].lower() == pub.lower()


def guest_state():
    """(active, seconds_left, token). Expiry is lazy — checked on read."""
    with guest_lock:
        now = time.time()
        if guest["token"] and now >= guest["expires"]:
            _end_turn(now)
        left = max(0.0, guest["expires"] - now) if guest["token"] else 0.0
        return bool(guest["token"]), left, guest["token"]


def guest_is_caller():
    """Is the caller the visitor whose turn it currently is?"""
    active, left, tok = guest_state()
    if not (active and left > 0):
        return False
    vid = request.cookies.get("rax_id")
    return bool(vid) and visitors.get(vid, {}).get("token") == tok


@app.before_request
def _gate_public():
    if not is_public_request():
        return None                      # LAN / Tailscale: admin, unrestricted
    ep = request.endpoint or ""
    if ep not in GUEST_ENDPOINTS:
        return jsonify(ok=False, reason="not available to guests"), 403
    # the driving endpoints additionally need to hold the live session
    if ep in ("jogpress", "jogrelease", "jogvec",
              "guest_pick", "guest_scan", "guest_query", "guest_home"):
        if not guest_is_caller():
            return jsonify(ok=False, reason="your turn has ended"), 403
    return None


@app.route("/guest")
def guest_page():
    """The one page the printed QR points at. It is the LOBBY: it decides for
    itself whether to show the controls or the queue, so a single URL works
    forever and nobody lands on a dead end."""
    vid, _new = _visitor()
    resp = _nocache(load_ui("guest.html"))
    resp.set_cookie("rax_id", vid, max_age=30 * 24 * 3600, samesite="Lax")
    return resp


@app.route("/guest/lobby", methods=["POST"])
def guest_state_route():
    """Heartbeat + queue state. Called every second by the lobby page."""
    vid, _new = _visitor()
    info = lobby_tick(vid)
    resp = jsonify(ok=True, **info)
    resp.set_cookie("rax_id", vid, max_age=30 * 24 * 3600, samesite="Lax")
    return resp


@app.route("/guest/status")
def guest_status():
    """The tiny slice of state a guest UI needs — no tuning, no calibration."""
    with lock:
        phase, detail, running = state["phase"], state["detail"], state["running"]
        grip = state.get("gripper")
    _a, left, _t = guest_state()
    return jsonify(phase=phase, detail=detail, running=running, gripper=grip,
                   left=round(left), mine=guest_is_caller())


@app.route("/guest/pick", methods=["POST"])
def guest_pick():
    """The pick engine, and nothing else."""
    with lock:
        if state["running"]:
            return jsonify(ok=False, reason="already running")
    return _run_bg(run_mission)


@app.route("/guest/home", methods=["POST"])
def guest_home():
    """Let a guest park the arm back at HOME — the safe reset when they have
    driven it somewhere awkward and want to start over."""
    with lock:
        if state["running"]:
            return jsonify(ok=False, reason="already running")

    def _fold():
        set_phase("FOLD HOME", "guest asked to park the arm")
        goto_smooth(HOME, settle=0.4)
        set_phase("IDLE", "parked")
    return _run_bg(_fold)


@app.route("/guest/scan", methods=["POST"])
def guest_scan():
    """Let a guest run the table scan — it only pans the base and fills the map."""
    with lock:
        if state["running"]:
            return jsonify(ok=False, reason="already running")
    return _run_bg(scan_2d, broad=True)


# What a guest may hunt for. A FIXED LIST, not free text: the query is fed straight
# to YOLO-World's open vocabulary, and letting the public type anything into the
# thing that decides where the arm lunges is not a knob to hand out. Each entry is
# (button label, emoji, detection query).
GUEST_TARGETS = [
    ("Red cube",   "🟥", "red cube"),
    ("Green cube", "🟩", "green cube"),
    ("Cup",        "🥤", "cup"),
    ("Bottle",     "🍼", "bottle"),
    ("Banana",     "🍌", "banana"),
    ("Phone",      "📱", "cell phone"),
]


@app.route("/guest/targets")
def guest_targets():
    with lock:
        cur = state.get("query") or ""
    return jsonify(targets=[{"label": n, "emoji": e, "query": q}
                            for n, e, q in GUEST_TARGETS], current=cur)


@app.route("/guest/query", methods=["POST"])
def guest_query():
    """Switch what the robot is looking for — only to one of GUEST_TARGETS."""
    want = (request.args.get("q") or "").strip().lower()
    if want not in {q for _n, _e, q in GUEST_TARGETS}:
        return jsonify(ok=False, reason="not one of the choices")
    if detector is None:
        return jsonify(ok=False, reason="detector not ready")
    pending_query[0] = want            # applied by the YOLO thread within ~2.5 s
    with lock:
        state["query"] = want
    return jsonify(ok=True, query=want)


@app.route("/ui/<path:name>")
def ui_asset(name):
    """Serve shared front-end assets (viewer3d.js). Filename-only, no traversal."""
    if "/" in name or "\\" in name or not name.endswith((".js", ".css")):
        return "", 404
    try:
        body = load_ui(name)
    except FileNotFoundError:
        return "", 404
    mt = "application/javascript" if name.endswith(".js") else "text/css"
    resp = Response(body, mimetype=mt)
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------- public tunnel + QR ----------------
# cloudflared's free quick-tunnel: no account, and the hostname is random enough
# that the URL itself is the secret. It dies with the process, so the exposure
# window is exactly as long as the server runs.
CF_BIN = r"C:\Program Files (x86)\cloudflared\cloudflared.exe"
tunnel = {"url": None, "proc": None, "error": None, "kind": None}
_CF_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


TS_BIN = r"C:\Program Files\Tailscale\tailscale.exe"


def start_funnel():
    """Prefer Tailscale Funnel: its hostname never changes, so ONE printed QR
    keeps working forever. Requires the `funnel` node attribute in the tailnet
    ACL — without it the node has no Funnel capability and this returns False,
    and we fall back to a cloudflared quick tunnel (fresh URL each run).
    """
    if not os.path.exists(TS_BIN):
        return False
    try:
        st = json.loads(subprocess.run([TS_BIN, "status", "--json"], timeout=20,
                                       capture_output=True, text=True).stdout)
        caps = st.get("Self", {}).get("Capabilities") or []
        if not any("funnel" in str(c).lower() for c in caps):
            say("tunnel: tailnet has no Funnel capability — using a cloudflared "
                "quick tunnel (URL changes each restart). Add the `funnel` "
                "nodeAttr in your tailnet ACL for a permanent QR.")
            return False
        host = (st.get("Self", {}).get("DNSName") or "").rstrip(".")
        if not host:
            return False
        # Funnel may only listen on 443/8443/10000 (see the funnel-ports capability),
        # so it CANNOT bind our PORT directly — it listens on 443 and proxies inward.
        # 443 also keeps the public URL bare, which is what a printed QR wants.
        r = subprocess.run([TS_BIN, "funnel", "--bg", "--https=443",
                            f"http://127.0.0.1:{PORT}"],
                           timeout=60, capture_output=True, text=True)
        out = (r.stderr or "") + (r.stdout or "")
        if r.returncode != 0 or "not enabled" in out.lower():
            # The ACL nodeAttr is necessary but NOT sufficient: Tailscale also wants
            # a one-time per-node consent click, and it prints the link here.
            link = re.search(r"https://login\.tailscale\.com/\S+", out)
            say("tunnel: Funnel needs one-time approval for this node"
                + (f" — open {link.group(0)}" if link else f" — {out.strip()[:160]}"))
            return False
        tunnel["url"] = f"https://{host}"
        tunnel["kind"] = "funnel"
        PUBLIC_HOST[0] = host
        say(f"guest link is LIVE (permanent): {tunnel['url']}/guest")
        return True
    except Exception as e:
        say(f"tunnel: funnel check failed — {e}")
        return False


def start_tunnel():
    """Launch the public tunnel: Funnel if the tailnet allows it, else cloudflared."""
    if start_funnel():
        return
    if not os.path.exists(CF_BIN):
        tunnel["error"] = "cloudflared not installed"
        say(f"tunnel: {CF_BIN} not found — guest link stays LAN-only")
        return
    try:
        p = subprocess.Popen(
            [CF_BIN, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
    except Exception as e:
        tunnel["error"] = str(e)
        say(f"tunnel: failed to launch — {e}")
        return
    tunnel["proc"] = p

    def _read():
        for line in p.stdout:
            m = _CF_URL_RE.search(line)
            if m and not tunnel["url"]:
                tunnel["url"] = m.group(0)
                tunnel["kind"] = "cloudflared"
                PUBLIC_HOST[0] = m.group(0).split("://", 1)[1]
                say(f"guest link is LIVE: {tunnel['url']}/guest  "
                    f"({GUEST_MINUTES:.0f} min per visitor)")
        if not tunnel["url"]:
            tunnel["error"] = "cloudflared exited without a URL"
    threading.Thread(target=_read, daemon=True).start()


def guest_url():
    return f"{tunnel['url']}/guest" if tunnel["url"] else None


@app.route("/guestlink")
def guestlink():
    """Admin-only: the guest URL, its QR, and who is currently playing."""
    url = guest_url()
    active, left, _t = guest_state()
    svg = None
    if url:
        import segno, io
        buf = io.BytesIO()            # segno's svg writer emits bytes
        segno.make(url, error="m").save(buf, kind="svg", scale=1, border=2,
                                        dark="#0b0f14", light=None, xmldecl=False,
                                        svgclass=None, lineclass=None)
        svg = buf.getvalue().decode("utf-8")
    with guest_lock:
        waiting = len(queue)
    return jsonify(ok=bool(url), url=url, qr_svg=svg, error=tunnel["error"],
                   kind=tunnel["kind"], permanent=(tunnel["kind"] == "funnel"),
                   busy=active, left=round(left), minutes=GUEST_MINUTES,
                   waiting=waiting, cooldown_min=GUEST_COOLDOWN_MIN)


FEEDBACK_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.jsonl")
feedback_lock = threading.Lock()
FEEDBACK_MAX = 1200            # characters; anything longer is somebody's script


@app.route("/feedback", methods=["POST"])
def feedback():
    """Anyone (guest or admin) can leave a note. Appended to feedback.jsonl and
    shown in the admin panel — no mail server, nothing leaves this machine."""
    body = request.get_json(silent=True) or {}
    msg = str(body.get("message") or "").strip()[:FEEDBACK_MAX]
    contact = str(body.get("contact") or "").strip()[:120]
    if not msg:
        return jsonify(ok=False, reason="say something first")
    vid = request.cookies.get("rax_id") or "-"
    rec = {"t": round(time.time(), 1),
           "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
           "message": msg, "contact": contact,
           "ip": _client_ip(), "vid": vid[:8],
           "from": "guest" if is_public_request() else "admin"}
    try:
        with feedback_lock:
            with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    except Exception as e:
        say(f"feedback write failed: {e}")
        return jsonify(ok=False, reason="could not save that, sorry")
    say(f"FEEDBACK from {rec['ip']}: {msg[:120]}")
    return jsonify(ok=True)


@app.route("/feedbacklist")
def feedbacklist():
    """Admin-only: everything anyone has written."""
    out = []
    try:
        with feedback_lock:
            with open(FEEDBACK_LOG, encoding="utf-8") as f:
                lines = f.readlines()[-200:]
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    except FileNotFoundError:
        pass
    return jsonify(items=out[::-1], count=len(out))


@app.route("/floorcal", methods=["POST"])
def r_floorcal():
    """Touch the table at several points and fit z_floor(x, y). Admin only - it
    deliberately drives the fingertip into the table, gently."""
    return _run_bg(calibrate_floor)


@app.route("/setconf", methods=["POST"])
def setconf():
    """Detector confidence floor. Lower finds marginal objects (a pen reads as
    'knife' at only ~0.09 in a cluttered frame) at the cost of more phantoms."""
    try:
        v = float(request.args.get("v", ""))
    except ValueError:
        return jsonify(ok=False, reason="need v=0.01..0.9")
    if not (0.01 <= v <= 0.9):
        return jsonify(ok=False, reason="need v=0.01..0.9")
    DET_CONF[0] = v
    if detector is not None:
        detector.conf = v
    say(f"detector confidence floor set to {v:.3f}")
    return jsonify(ok=True, conf=v)


@app.route("/relax", methods=["POST"])
def r_relax():
    """Fold home and cut torque. Any later motion command wakes it automatically."""
    with lock:
        if state["running"]:
            return jsonify(ok=False, reason="busy — stop first")
    return _run_bg(relax_arm)


@app.route("/wake", methods=["POST"])
def r_wake():
    wake_arm("asked to wake")
    note_activity()
    return jsonify(ok=True, relaxed=ARM_RELAXED[0])


@app.route("/setrelaxidle", methods=["POST"])
def setrelaxidle():
    """Seconds of inactivity before the arm relaxes itself. 0 disables it."""
    try:
        v = float(request.args.get("s", ""))
    except ValueError:
        return jsonify(ok=False, reason="need s=<seconds>, 0 to disable")
    IDLE_RELAX_S[0] = max(0.0, v)
    say(f"idle relax {'disabled' if v <= 0 else f'set to {v:.0f}s'}")
    return jsonify(ok=True, seconds=IDLE_RELAX_S[0])


@app.route("/floor")
def r_floor():
    a, b, c = FLOOR_PLANE
    return jsonify(a=a, b=b, c=c,
                   tilt_deg=round(math.degrees(math.atan(math.hypot(a, b))), 3),
                   at_18cm=round(floor_z(0.18, 0.0), 4),
                   at_34cm=round(floor_z(0.34, 0.0), 4),
                   grasp_clear=FLOOR_GRASP_CLEAR,
                   calibrated=os.path.exists(FLOOR_FILE))


@app.route("/survey", methods=["POST"])
def r_survey():
    """Locate the target from the fixed survey pose and REPORT it, without moving
    to grasp. Repeat it a few times to see whether the fix is repeatable - that is
    the number that decides whether the approach can be accurate at all."""
    label = (request.args.get("label") or "").strip().lower()

    def _t():
        finder = None
        lab = label
        if not lab:
            finder, _tr, lab = _target_finder()
        xy, spread = locate_from_survey(lab, finder)
        if xy is None:
            set_phase("ABORTED", f"survey: {spread}")
            return
        with lock:
            state["obj3d"] = [float(xy[0]), float(xy[1]), 0.02]
            state["obj3d_label"] = lab.split()[0]
        set_phase("IDLE", f"{lab} at x={xy[0]*100:+.1f} y={xy[1]*100:+.1f} cm "
                          f"r={np.hypot(*xy)*100:.1f}cm spread={spread*100:.1f}cm")
    return _run_bg(_t)


@app.route("/guests")
def guests():
    """Admin-only: who is driving, who is queued, and who has been here.

    Deliberately NOT in GUEST_ENDPOINTS — visitors must not be able to enumerate
    each other's addresses.
    """
    now = time.time()
    with guest_lock:
        _prune(now)
        tok = guest["token"]
        playing = None
        for vid, v in visitors.items():
            if tok and v.get("token") == tok:
                playing = {"ip": v["ip"], "vid": vid[:6],
                           "left": round(max(0.0, guest["expires"] - now)),
                           "held": round(now - guest["started"]),
                           "turns": v.get("turns", 0)}
                break
        q = [{"pos": i + 1, "ip": visitors.get(v, {}).get("ip", "?"),
              "vid": v[:6],
              "waiting": round(now - visitors.get(v, {}).get("first", now)),
              "turns": visitors.get(v, {}).get("turns", 0),
              "cooldown": round(_cooling(v, now))}
             for i, v in enumerate(queue)]
        seen = len(visitors)
    rows, by_ip = _guest_history()
    return jsonify(playing=playing, queue=q, seen_now=seen,
                   by_ip=by_ip[:25], recent=rows[::-1][:40],
                   minutes=GUEST_MINUTES, cooldown_min=GUEST_COOLDOWN_MIN)


@app.route("/guestkick", methods=["POST"])
def guestkick():
    """Admin-only: end the current guest's turn immediately."""
    with guest_lock:
        now = time.time()
        _end_turn(now)
        _promote(now)
    stop_flag.set()
    say("guest turn ended by admin")
    return jsonify(ok=True)

UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")


def load_ui(name):
    """Read a UI file from ui/. Read per request, not cached, so editing the
    HTML shows up on refresh without restarting the robot server."""
    with open(os.path.join(UI_DIR, name), encoding="utf-8") as f:
        return f.read()



def _nocache(html):
    resp = Response(html, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/")
def index():
    if is_public_request():
        return _nocache(load_ui("guest_closed.html"))
    return _nocache(load_ui("admin.html"))


@app.route("/status")
def status():
    with lock:
        s = {k: v for k, v in state.items()}
    s["log"] = list(log)
    s["measure_stats"] = dict(MEAS_STATS)
    s["relaxed"] = ARM_RELAXED[0]
    s["idle_relax_s"] = IDLE_RELAX_S[0]
    s["idle_for"] = round(time.time() - last_activity[0])
    s["tune"] = {
        "aim_du": round(AIM_DU, 1),
        "right_trim_cm": round(TARGET_RIGHT_TRIM_M * 100, 2),
        "back_cm": round(TARGET_BACK_M * 100, 2),
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
    # every mapped object (2D map) as a table-resting, yawed box for the 3D viewer
    objs2d = [{"x": o["x"], "y": o["y"], "s": o["size"],
               "w": o["w_m"], "d": o["d_m"], "h": o["h_m"], "yaw": o["yaw"],
               "label": o["label"], "tag": o["tag"]}
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


# Query presets. "coco" is every class YOLO knows; "table" is the subset this arm
# can physically pick, which detects faster and keeps street furniture out of the map.
TABLE_CLASSES = [
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "orange", "book", "clock", "vase", "scissors", "teddy bear",
    "cell phone", "mouse", "remote", "keyboard", "laptop", "toothbrush",
    "pen", "pencil", "marker", "tape", "can", "box", "red cube", "green cube",
]


@app.route("/preset")
def preset():
    q = {"coco": ", ".join(COCO_CLASSES),
         "table": ", ".join(TABLE_CLASSES),
         "cubes": "red cube, green cube"}.get(request.args.get("name", ""))
    if q is None:
        return jsonify(ok=False, reason="unknown preset")
    return jsonify(ok=True, q=q)


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


@app.route("/setback", methods=["POST"])
def setback():
    """Live-tune the approach radial back-off (cm). Positive stops SHORT of the cube."""
    try:
        cm = float(request.args.get("cm", request.form.get("cm", 1.0)))
    except (TypeError, ValueError):
        return jsonify(ok=False, reason="need a number")
    global TARGET_BACK_M
    TARGET_BACK_M = float(np.clip(cm, 0.0, 10.0)) / 100.0
    say(f"approach back-off set to {TARGET_BACK_M*100:.1f}cm")
    return jsonify(ok=True, cm=TARGET_BACK_M*100)


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
    broad = request.args.get("broad", "1") != "0"
    with lock:
        if state["running"]:
            return jsonify(ok=False, reason="busy")

    def _t():
        stop_flag.clear()
        with lock:
            state["running"] = True
        try:
            scan_2d(broad=broad)
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


def _run_bg(fn, *a, **kw):
    """Run a mission step on the mission thread, with the usual busy latch."""
    with lock:
        if state["running"]:
            return jsonify(ok=False, reason="busy")

    def _t():
        stop_flag.clear()
        with lock:
            state["running"] = True
        try:
            fn(*a, **kw)
        except Abort as e:
            set_phase("ABORTED", str(e))
        except Exception as e:
            set_phase("ERROR", f"{type(e).__name__}: {e}")
        finally:
            with lock:
                state["running"] = False
    threading.Thread(target=_t, daemon=True).start()
    return jsonify(ok=True)


@app.route("/place", methods=["POST"])
def r_place():
    """Place what is held: /place?tag=N onto a mapped object, or /place?x=..&y=..
    (centimetres, base frame) onto a bare table spot."""
    with lock:
        if not carry["held"]:
            return jsonify(ok=False, reason="nothing in the jaws — pick something first")
    tag = request.args.get("tag")
    recenter = request.args.get("recenter", "1") != "0"
    if tag is not None:
        return _run_bg(place_at, tag=int(tag), recenter=recenter)
    try:
        xy = (float(request.args["x"]) / 100.0, float(request.args["y"]) / 100.0)
    except (KeyError, ValueError):
        return jsonify(ok=False, reason="need tag=N or x=..&y=.. in cm")
    return _run_bg(place_at, xy=xy, recenter=False)


# ---------------- typed tasks: "green on red, blue on green" ----------------
# The detection query says WHAT TO LOOK FOR; a task says WHAT TO DO. Kept separate
# because one query ("red cube, green cube") serves many different tasks, and
# folding the two together is how `_target_finder` used to grab the wrong cube.
TASK_SEP = (" on ", " onto ", " ontop ", " on top of ", ">", "->")


def parse_task(text):
    """'green on red, blue on green' -> [('green','red'), ('blue','green')].

    Each step is <pick> on <destination>. Steps run in order, so a tower is just a
    longer list. Labels are matched loosely against the map later, so 'green' and
    'green cube' both work.
    """
    steps = []
    for chunk in str(text).replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        cut = None
        low = chunk.lower()
        for sep in TASK_SEP:
            i = low.find(sep)
            if i != -1 and (cut is None or i < cut[0]):
                cut = (i, len(sep))
        if cut is None:
            raise Abort(f"cannot read step {chunk!r} — write it as "
                        f"'<object> on <destination>', e.g. 'green on red'")
        pick = chunk[:cut[0]].strip()
        dest = chunk[cut[0] + cut[1]:].strip()
        if not pick or not dest:
            raise Abort(f"step {chunk!r} is missing an object or a destination")
        steps.append((pick, dest))
    if not steps:
        raise Abort("empty task")
    return steps


def _find_map_tag(label):
    """The best-supported map entry whose label mentions `label`. Loose on purpose:
    the user types 'red', the map holds 'red cube'."""
    want = str(label).strip().lower()
    with w2d_lock:
        best, bn = None, -1
        for t, o in W2D["objs"].items():
            if o.get("picked"):
                continue
            if want in o["label"].lower() or o["label"].lower() in want:
                if o["n"] > bn:
                    best, bn = t, o["n"]
        return best


def run_task(text):
    """Run a typed task: pick each object and place it on its destination."""
    steps = parse_task(text)
    say(f"task: {len(steps)} step(s) — " +
        ", ".join(f"{p} on {d}" for p, d in steps))
    # Fail before moving if any destination is missing from the map, rather than
    # picking something up and then discovering there is nowhere to put it.
    for i, (pick, dest) in enumerate(steps, 1):
        if _find_map_tag(dest) is None:
            raise Abort(f"step {i}: no '{dest}' on the 2D map — Scan → 2D map first")
        if _find_map_tag(pick) is None:
            raise Abort(f"step {i}: no '{pick}' on the 2D map — Scan → 2D map first")
    for i, (pick, dest) in enumerate(steps, 1):
        checkpoint()
        tag = _find_map_tag(dest)      # re-resolve: earlier steps reshape the map
        if tag is None:
            raise Abort(f"step {i}: '{dest}' vanished from the map")
        set_phase("TASK", f"step {i}/{len(steps)}: {pick} on {dest}")
        say(f"--- task step {i}/{len(steps)}: {pick} -> {dest} (tag {tag}) ---")
        pick_then_place(dest_tag=tag, target_label=pick)
    set_phase("DONE", f"task complete: {len(steps)} step(s)")


@app.route("/task", methods=["POST"])
def r_task():
    """Run a typed task, e.g. /task?q=green on red, blue on green"""
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify(ok=False, reason="empty task")
    try:
        parse_task(q)          # reject bad syntax before claiming the arm
    except Abort as e:
        return jsonify(ok=False, reason=str(e))
    return _run_bg(run_task, q)


def pick_then_place(dest_tag=None, dest_xy=None, target_label=None):
    run_mission(target_label=target_label)
    with lock:
        held = carry["held"]
        # run_mission clears state["running"] in its own finally, and checkpoint()
        # only honours the Stop button while that flag is set — so re-arm it before
        # the place, or Stop silently does nothing for the rest of the sequence.
        state["running"] = True
    if stop_flag.is_set():
        raise Abort("stopped by user")
    if not held:
        raise Abort("pick failed — not placing")
    place_at(tag=dest_tag, xy=dest_xy)


@app.route("/pickplace", methods=["POST"])
def r_pickplace():
    """Pick the object named by the current query, then place it. Destination is
    tag=N (a mapped object) or x=..&y=.. in cm."""
    tag = request.args.get("tag")
    if tag is not None:
        return _run_bg(pick_then_place, dest_tag=int(tag))
    try:
        xy = (float(request.args["x"]) / 100.0, float(request.args["y"]) / 100.0)
    except (KeyError, ValueError):
        return jsonify(ok=False, reason="need tag=N or x=..&y=.. in cm")
    return _run_bg(pick_then_place, dest_xy=xy)


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


def clear_gripper_overload(ids=(1, 2, 3, 4, 5, 6)):
    """Clear a latched overload on ANY motor with a raw torque cycle, before
    lerobot's handshake reads hit the error.

    It used to touch only the gripper (id 6), which latches easily on worn gears -
    but shoulder_lift (id 2) latches too after sustained holding, and THAT is what
    kills the server: `Failed to read 'Min_Position_Limit' on id_=2 ... Overload
    error!` at connect, before anything is running to catch it. Cycle them all.
    """
    try:
        import scservo_sdk as scs
        ph = scs.PortHandler("COM4")
        if not ph.openPort():
            return
        ph.setBaudRate(1000000)
        pk = scs.PacketHandler(0)
        cleared = []
        for mid in ids:
            pk.write1ByteTxRx(ph, mid, 40, 0)   # torque off
            time.sleep(0.12)
            pk.write1ByteTxRx(ph, mid, 40, 1)   # torque on (clears latched error)
            time.sleep(0.06)
            _pos, _c, err = pk.read2ByteTxRx(ph, mid, 56)
            cleared.append(f"{mid}:{err:#04x}")
        ph.closePort()
        say("overload cleared — " + " ".join(cleared))
    except Exception as e:
        say(f"overload clear skipped: {e}")


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
        except (RuntimeError, ConnectionError) as e:
            msg = str(e)
            if attempt == 5:
                raise
            if "Overload" in msg or "no status packet" in msg.lower():
                # a latched servo: cycle torque on every id and try again, rather
                # than exiting and leaving the operator to run a script by hand
                say(f"connect blocked by a latched servo (attempt {attempt + 1}/6) — "
                    f"clearing overload and retrying")
                try:
                    robot.bus.port_handler.closePort()
                except Exception:
                    pass
                time.sleep(1.5)
                clear_gripper_overload()
                time.sleep(1.0)
                continue
            if "Missing motor" not in msg:
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
    load_floor_plane()
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
    # imgsz 640, not 320: a pen is a few pixels wide at 320 and is simply not
    # found (measured on a real frame - 320 -> no pen, 640 -> pen at 0.65 conf,
    # the strongest detection in the scene). Costs 489 ms vs 171 ms per pass, which
    # is free here because the YOLO worker sleeps 2.5 s between passes anyway.
    #
    # color_filter_min_frac=0: the library filter is applied to EVERY box whenever
    # ANY colour word appears in the query, so with the tabletop vocabulary it
    # demanded that a pen contain red/orange/green pixels and threw it away. (It
    # even reads the FRUIT "orange" as a colour name.) Colour gating is done
    # per-label in yolo_worker instead - see _colour_ok.
    detector = YoloWorldDetector(LEROBOT + r"\yolov8s-worldv2.pt", conf=DET_CONF[0], imgsz=PICK_IMGSZ,
                                 color_filter_min_frac=0.0)
    _q0 = "red cube, green cube"      # colour-ONLY: colourless terms like "toy
                                      # block"/"box" make YOLO-World fire on the
                                      # green cube and the mission then dives to it
    detector.set_query(_q0)
    with lock:
        state["query"] = _q0
    threading.Thread(target=yolo_worker, daemon=True).start()
    threading.Thread(target=idle_view, daemon=True).start()
    threading.Thread(target=jog_loop, daemon=True).start()
    threading.Thread(target=idle_relax_watch, daemon=True).start()
    start_rerun()
    threading.Thread(target=rerun_thread, daemon=True).start()
    start_tunnel()
    set_phase("IDLE", "ready — press Start")
    say(f"UI: http://100.110.89.78:{PORT}  (Tailscale)")
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
