# map2d.py — simple 2D bird's-eye-view object map for the SO-101 + OAK-D.
#
# The whole idea, kept deliberately small:
#   * Everything the arm cares about lives on ONE plane: the table, at the base's
#     height. So the world is a 2D X-Y grid (base frame), not a 3D map.
#   * The camera rides on the gripper. Its pose is known EXACTLY from the servo
#     angles via forward kinematics — this is SLAM with known poses, i.e. pure
#     mapping, no estimation.
#   * To place a detected object on the grid we use INVERSE PERSPECTIVE MAPPING
#     (IPM): cast the detection's sightline onto the table plane. For an object
#     resting on the table that intersection IS its position — no dependence on
#     the OAK-D's (biased, close-range) stereo depth. Stereo is used only to GATE
#     a detection (is it really near the table, within reach?).
#   * Moving to an object is then plain IK to (x, y, table) — "just math".
#
# Reach measured from the URDF at table height: usable annulus r≈8..42cm over a
# ±110° arc. The grid and the reachable mask below come from that.
import json, math, os, sys, tempfile, threading, time
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

sys.path.insert(0, r"C:\Users\labot\Documents\lerobot\src")

from lerobot.model.kinematics import RobotKinematics
from lerobot.perception.yolo_world import YoloWorldDetector
from lerobot.manipulation.visual_servo.gaze_engine import ARM_MOTORS, parse_tf_string
from lerobot.manipulation.yolo_track.motion_primitives import send_joint_target_smoothly
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig
from lerobot.motors.feetech.feetech import FeetechMotorsBus as _FTBus

_FTBus._handshake = lambda self: None      # gripper ID6 handshake is flaky; writes need no ACK

LEROBOT = r"C:\Users\labot\Documents\lerobot"
PORT = 8486
HOST_IP = "100.110.89.78"

# ---- world geometry ----
TABLE_Z = -0.02          # table surface in the base frame (measured FK tip contact ≈ -0.022)
REACH_MIN, REACH_MAX = 0.08, 0.42     # usable radius (m) — from the URDF reach sweep
ARC_DEG = 110.0          # usable pan half-arc (shoulder_pan limit)
GRID_RES = 0.005         # 5 mm cells
GRID_X = (-0.12, 0.46)   # base-frame x span of the map
GRID_Y = (-0.46, 0.46)
MERGE_M = 0.18           # detections of the SAME label within this distance are the same
                         # object (merged into one running-mean entry). Set well above
                         # the residual localization drift so one cube = one map dot,
                         # not a smear of duplicate tags. (Trade-off: two same-colour
                         # cubes closer than this would merge — fine for now.)
TABLE_TOL = 0.06         # a detection's stereo range must agree with its ray∩table range to this
STEREO_TRUST = 0.60      # only trust stereo depth below this (m). A close cube is stereo-BLIND
                         # and its pixel reads the far wall (~2-3 m); ignore that, trust IPM.

# ---- poses ----
VIEW = np.array([-6.7, 37.1, 48.1, -40.4, -28.4])     # a neutral raised pose
HOME = np.array([-9.67, -102.022, 98.066, 32.879, 0.0])
SCAN_PITCH = 45.0        # (legacy) gripper pitch guess; scan() now solves for the pitch
CAM_DOWN = 40.0          # desired CAMERA depression (deg below horizon) while scanning —
                         # steep enough that IPM (ray∩table) is well-conditioned
GRIP_TIP = 0.007         # gripper_frame_link is the fingertip +7mm
HAND_UV = (440.0, 394.0)

# ---- IK joint limits (from so101_new_calib.urdf) ----
J_LO = np.array([-110.0, -100.0, -96.8, -95.0, -157.2])
J_HI = np.array([+110.0, +100.0, +96.8, +95.0, +162.8])
WFLEX_MIN, WFLEX_MAX = float(J_LO[3]), float(J_HI[3])
GRASP_PITCH = (75.0, 80.0, 70.0, 85.0, 90.0, 65.0)

state = {"phase": "IDLE", "detail": "", "joints": [], "gripper": None,
         "running": False, "query": "", "dets": []}
log = deque(maxlen=120)
frame_jpeg = [None]
lock = threading.Lock()
bus_lock = threading.RLock()
kin_lock = threading.Lock()
map_lock = threading.Lock()
stop_flag = threading.Event()
mission_thread = [None]
latest_snap = [None]
pending_query = [None]
det_classes = [["red cube", "green cube"]]

robot = kin = cam = detector = None
T_ee_cam = parse_tf_string("-0.0390,-0.0290,-0.0043,1.7407,1.6823,1.8316")
fx = fy = cx0 = cy0 = 0.0
TF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handeye_tf.json")


def say(msg):
    log.appendleft(f"{time.strftime('%H:%M:%S')}  {msg}")
    print(msg, flush=True)


def set_phase(ph, detail=""):
    with lock:
        state["phase"], state["detail"] = ph, detail
    say(f"[{ph}] {detail}" if detail else f"[{ph}]")


class Abort(Exception):
    pass


def checkpoint():
    with lock:
        running = state["running"]
    if stop_flag.is_set() and running:
        raise Abort("stopped by user")


# ---------------- kinematics (serialized; NumericRobotKinematics mutates state) ----------------
def fk(q):
    with kin_lock:
        return np.asarray(kin.forward_kinematics(np.asarray(q, np.float64)))


def T_cam_of(q):
    return fk(q) @ T_ee_cam


def tip_of(q):
    T = fk(q)
    return T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, GRIP_TIP])


def cam_depression(q):
    """Camera optical-axis angle below the horizon (deg). The camera is much
    shallower than the hand because of the hand-eye rotation, so scan() solves
    for the hand pitch that gives the camera depression it actually wants."""
    z = T_cam_of(q)[:3, 2]
    return math.degrees(math.atan2(-z[2], math.hypot(z[0], z[1])))


def _slave_wflex(j1, j2, pitch):
    return float(np.clip(pitch - j1 - j2, WFLEX_MIN, WFLEX_MAX))


def ik_xy(q_seed, p_tgt, pitch_tgt, j5, iters=80, tol=2e-3, ret_err=False, _retry=True):
    """Position IK on pan/lift/elbow with wrist_flex slaved to hold gripper pitch
    (the 3 pitch joints are parallel, so pitch = j1+j2+j3 exactly). Limit-clamped
    every step; returns the residual so callers can refuse unreachable targets."""
    q = np.array(q_seed, np.float64)
    q[4] = float(np.clip(j5, J_LO[4], J_HI[4]))
    q[3] = _slave_wflex(q[1], q[2], pitch_tgt)
    J = None
    for it in range(iters):
        T = fk(q)
        err = p_tgt - T[:3, 3]
        if np.linalg.norm(err) < 3e-4:
            break
        if J is None or it % 8 == 0:
            J = np.empty((3, 3))
            for c, ji in enumerate((0, 1, 2)):
                dq = q.copy()
                dq[ji] = float(np.clip(dq[ji] + 0.5, J_LO[ji], J_HI[ji]))
                if ji in (1, 2):
                    dq[3] = _slave_wflex(dq[1], dq[2], pitch_tgt)
                J[:, c] = (fk(dq)[:3, 3] - T[:3, 3]) / 0.5
        dth = np.clip(J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), err), -8.0, 8.0)
        q[:3] = np.clip(q[:3] + dth, J_LO[:3], J_HI[:3])
        q[3] = _slave_wflex(q[1], q[2], pitch_tgt)
    e = float(np.linalg.norm(p_tgt - fk(q)[:3, 3]))
    if abs((q[1] + q[2] + q[3]) - pitch_tgt) > 2.0:
        e = max(e, 0.05)
    if e > tol and _retry:
        for alt in ([q_seed[0], -95, 90, 30, j5], [q_seed[0], -30, 50, 60, j5],
                    [q_seed[0], -60, 20, 80, j5]):
            q2, e2 = ik_xy(np.array(alt, np.float64), p_tgt, pitch_tgt, j5, iters, tol,
                           ret_err=True, _retry=False)
            if e2 < e:
                q, e = q2, e2
            if e <= tol:
                break
    return (q, e) if ret_err else q


def reachable(x, y):
    r = math.hypot(x, y)
    a = abs(math.degrees(math.atan2(y, x)))
    return REACH_MIN <= r <= REACH_MAX and a <= ARC_DEG


def grasp_pitch_for(x, y, q_seed):
    """Steepest feasible grasp pitch for a table cell, proven by IK. Close cells
    need a near-vertical hand (the fingertip is 98mm ahead of the wrist)."""
    p = np.array([x, y, TABLE_Z + 0.02])
    p_hi = np.array([x, y, TABLE_Z + 0.07])
    best = None
    for pitch in GRASP_PITCH:
        _, e1 = ik_xy(q_seed, p, pitch, float(q_seed[4]), ret_err=True)
        _, e2 = ik_xy(q_seed, p_hi, pitch, float(q_seed[4]), ret_err=True)
        w = max(e1, e2)
        if w < 0.006:
            return pitch, w
        if best is None or w < best[1]:
            best = (pitch, w)
    return None, best[1]


# ---------------- robot I/O ----------------
_prev = [None]


def observe(overlay=True):
    checkpoint()
    joints = None
    err = None
    for _ in range(12):
        try:
            with bus_lock:
                pos = robot.bus.sync_read("Present_Position", ARM_MOTORS, num_retry=2)
            joints = np.array([float(pos[m]) for m in ARM_MOTORS])
            break
        except Exception as e:      # flaky bus: arm-only read + retries
            err = e
            time.sleep(0.06)
    if joints is None:
        raise err
    with bus_lock:
        rgb = np.asarray(cam.read_latest())
        try:
            d = cam.read_depth()
        except Exception:
            d = None
        try:
            gp = float(robot.bus.sync_read("Present_Position", ["gripper"], num_retry=0)["gripper"])
        except Exception:
            gp = -1.0
    depth = None
    if d is not None:
        depth = np.asarray(d, np.float32) / 1000.0
        depth[depth <= 0.05] = np.nan
    with lock:
        state["joints"] = [round(float(v), 1) for v in joints]
        state["gripper"] = round(gp, 1)
        latest_snap[0] = {"rgb": rgb, "q": joints.copy(), "depth": depth, "t": time.time()}
    if overlay:
        publish(rgb, joints)
    return joints, rgb, gp


def send_joints(q, gripper=None):
    act = {f"{m}.pos": float(v) for m, v in zip(ARM_MOTORS, q)}
    if gripper is not None:
        act["gripper.pos"] = float(gripper)
    with bus_lock:
        robot.send_action(act)


def goto(target, settle=0.4, step=2.0):
    joints, _r, gp = observe(overlay=False)
    if gp < 0:
        gp = 50.0
    with bus_lock:
        send_joint_target_smoothly(robot, ARM_MOTORS, joints, np.asarray(target, np.float64),
                                   step_deg=step, sleep_s=0.03,
                                   gripper_open=gp >= 50.0, gripper_width_pct=gp)
    time.sleep(settle)


def gripper_current():
    try:
        with bus_lock:
            return float(robot.bus.read("Present_Current", "gripper", normalize=False))
    except Exception:
        return None


def clear_overload():
    try:
        import scservo_sdk as scs
        ph = scs.PortHandler("COM4")
        if not ph.openPort():
            return
        ph.setBaudRate(1000000)
        pk = scs.PacketHandler(0)
        # clear latched overload on EVERY servo, not just the gripper: a joint
        # driven out of range (e.g. a fresh, mis-homed base) latches overload and
        # then even lerobot's calibration READ throws "Overload error!" before the
        # prompt. A torque off/on cycle clears the latch on each id.
        for mid in (1, 2, 3, 4, 5, 6):
            pk.write1ByteTxRx(ph, mid, 40, 0)
        time.sleep(0.5)
        for mid in (1, 2, 3, 4, 5, 6):
            pk.write1ByteTxRx(ph, mid, 40, 1)
        time.sleep(0.2)
        ph.closePort()
        say("overload cleared on ids 1-6")
    except Exception as e:
        say(f"overload clear skipped: {e}")


# ---------------- geometry: pixel <-> base, ray∩table ----------------
def project_base(p_base, T_base_cam):
    pc = np.linalg.inv(np.asarray(T_base_cam, np.float64)) @ np.append(p_base, 1.0)
    if pc[2] <= 1e-4:
        return None
    return (float(fx * pc[0] / pc[2] + cx0), float(fy * pc[1] / pc[2] + cy0))


def ray_to_table(uv, T_base_cam, z=TABLE_Z):
    """Inverse perspective mapping: intersect a pixel's sightline with the table
    plane -> the base-frame (x, y) that pixel looks at ON the table."""
    o = T_base_cam[:3, 3]
    d_cam = np.array([(uv[0] - cx0) / fx, (uv[1] - cy0) / fy, 1.0])
    d = T_base_cam[:3, :3] @ d_cam
    if d[2] >= -1e-3:          # ray not pointing down at the plane
        return None
    t = (z - o[2]) / d[2]
    if t <= 0:
        return None
    return (o + t * d)[:2]


def stereo_range(uv, depth, win=6):
    """Median stereo depth (m) around a pixel, or nan. Used only to GATE a
    detection (is it near the table plane / on a real surface?)."""
    if depth is None:
        return float("nan")
    u, v = int(round(uv[0])), int(round(uv[1]))
    h, w = depth.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return float("nan")
    patch = depth[max(0, v - win):v + win + 1, max(0, u - win):u + win + 1]
    good = patch[np.isfinite(patch)]
    return float(np.median(good)) if good.size >= 8 else float("nan")


def tip_pixel(joints):
    T = fk(joints)
    return project_base(T[:3, 3], T @ T_ee_cam)


def load_tf():
    global T_ee_cam
    try:
        with open(TF_FILE) as f:
            d = json.load(f)
        T_ee_cam = parse_tf_string(d["tf"])
        return d
    except Exception:
        return None


# ---------------- detection (YOLO-S + HSV fallback) ----------------
HSV = {"red": [((0, 110, 80), (9, 255, 255)), ((170, 110, 80), (179, 255, 255))],
       "green": [((38, 80, 60), (85, 255, 255))],
       "blue": [((95, 90, 60), (130, 255, 255))]}


def label_color(label):
    for c in HSV:
        if c in label.lower():
            return c
    return None


def hsv_box(rgb, color):
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, np.uint8), cv2.COLOR_RGB2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in HSV[color]:
        m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _l, st, _c = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < 900:
            continue
        if best is None or a > best[0]:
            x, y, w, h = (int(st[i, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                  cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            best = (a, (x, y, x + w, y + h))
    return best[1] if best else None


def detect(rgb):
    """Return {label: (bbox, conf, src)} — best YOLO box per class, HSV fallback."""
    out = {}
    try:
        dets = detector.predict_rgb(np.ascontiguousarray(rgb)) if detector else []
    except Exception:
        dets = []
    classes = det_classes[0]
    for d in dets:
        if not (0 <= d.class_id < len(classes)):
            continue
        lbl = classes[d.class_id]
        if lbl not in out or d.confidence > out[lbl][1]:
            out[lbl] = (tuple(d.xyxy), float(d.confidence), "yolo")
    for lbl in classes:
        if out.get(lbl) and out[lbl][1] >= 0.14:
            continue
        col = label_color(lbl)
        if col:
            b = hsv_box(rgb, col)
            if b:
                out[lbl] = (b, 0.5, "hsv")
    return out


# ---------------- the 2D MAP ----------------
def object_xy(bbox, T_cam, depth):
    """Where an object rests on the table (base x,y), by IPM. Casts rays through
    the object's LOWER pixels (its ground-contact band) onto the table plane and
    takes the median — exact for a body sitting on the plane, and independent of
    the OAK-D's biased close-range depth. Stereo only gates the result."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 4:
        return None
    # sample the bottom third of the box (nearest the table), inset horizontally
    us = np.linspace(x1 + 0.25 * w, x2 - 0.25 * w, 5)
    vs = np.linspace(y1 + 0.66 * h, y2 - 0.05 * h, 5)
    hits = []
    for v in vs:
        for u in us:
            p = ray_to_table((u, v), T_cam)
            # accept any hit on the TABLE in front (a broad region) — do NOT gate
            # on pick-reach here, or an object a little too far never gets mapped.
            # Reachability is a separate flag for picking, computed from the map.
            if p is not None and 0.05 < math.hypot(p[0], p[1]) < 0.60 \
                    and abs(math.degrees(math.atan2(p[1], p[0]))) < 125:
                hits.append(p)
    if len(hits) < 3:
        return None
    xy = np.median(np.array(hits), axis=0)
    # Stereo gate — but ONLY when stereo is a plausible CLOSE reading. A cube near
    # the base is inside the OAK-D's stereo blind zone, so its pixel depth falls
    # through to the far wall (~2-3 m); trusting that would reject every valid IPM
    # hit (which is exactly what "IPM failed, stereo=195cm" was). So we consult
    # stereo only when it reads < STEREO_TRUST m — close enough to actually be the
    # object — and use it to reject a not-on-the-table thing (a hand at close
    # range). Beyond that, IPM (+ the reachable filter) stands on its own.
    rng = stereo_range(((x1 + x2) / 2, (y1 + y2) / 2), depth)
    if np.isfinite(rng) and rng < STEREO_TRUST:
        ray_rng = float(np.linalg.norm(np.append(xy, TABLE_Z) - T_cam[:3, 3]))
        if abs(rng - ray_rng) > 0.12:
            return None
    return xy


class World2D:
    """A flat, top-down object map in the base X-Y plane. Each object is a tag +
    label + running-mean (x,y). Association is nearest-neighbour within MERGE_M —
    that's the whole tracker."""

    def __init__(self):
        self.objs = {}       # tag -> {label, xy, n, conf, last_seen}
        self._next = 1

    def update(self, label, xy, conf):
        with map_lock:
            best, bd = None, MERGE_M
            for tag, o in self.objs.items():
                if o["label"] != label:
                    continue
                d = float(np.linalg.norm(o["xy"] - xy))
                if d < bd:
                    best, bd = tag, d
            if best is None:
                tag = self._next
                self._next += 1
                self.objs[tag] = {"label": label, "xy": np.asarray(xy, np.float64),
                                  "n": 1, "conf": conf, "last_seen": time.time()}
                say(f"map: new {label} #{tag} @ ({xy[0]*100:.0f},{xy[1]*100:.0f})cm")
            else:
                o = self.objs[best]
                a = 0.3
                o["xy"] = (1 - a) * o["xy"] + a * np.asarray(xy, np.float64)
                o["n"] += 1
                o["conf"] = max(o["conf"], conf)
                o["last_seen"] = time.time()

    def find(self, label):
        """Best (most-seen) object of a label, or None."""
        with map_lock:
            cand = [(t, o) for t, o in self.objs.items() if o["label"] == label]
        if not cand:
            return None
        t, o = max(cand, key=lambda to: to[1]["n"])
        return t, o["xy"].copy()

    def get(self, tag):
        with map_lock:
            o = self.objs.get(tag)
            return (o["label"], o["xy"].copy()) if o else None

    def remove(self, tag):
        with map_lock:
            self.objs.pop(tag, None)

    def snapshot(self):
        with map_lock:
            now = time.time()
            return [{"tag": t, "label": o["label"],
                     "x": round(float(o["xy"][0]), 3), "y": round(float(o["xy"][1]), 3),
                     "r_cm": round(float(np.hypot(*o["xy"])) * 100, 1),
                     "ang": round(math.degrees(math.atan2(o["xy"][1], o["xy"][0])), 0),
                     "n": o["n"], "reach": reachable(*o["xy"]),
                     "age": round(now - o["last_seen"], 1)}
                    for t, o in self.objs.items()]


world = World2D()


def sense(joints, rgb, depth):
    """Detect + drop every on-table object into the 2D map from ONE view."""
    T_cam = T_cam_of(joints)
    dets = detect(rgb)
    ui = []
    for label, (bbox, conf, src) in dets.items():
        ui.append({"label": label, "bbox": [float(v) for v in bbox], "conf": round(conf, 2),
                   "src": src, "t": time.time()})
        if conf < 0.14:
            continue
        x1, y1, x2, y2 = bbox
        if x1 <= 2 or y1 <= 2 or x2 >= rgb.shape[1] - 3 or y2 >= rgb.shape[0] - 3:
            continue                     # clipped: partial object
        xy = object_xy(bbox, T_cam, depth)
        if xy is not None:
            world.update(label, xy, conf)
    with lock:
        state["dets"] = ui


def probe():
    """GROUND TRUTH, no motion. For the current view, report each detection's IPM
    (x,y) on the table, the camera depression at it (shallower = less accurate),
    and stereo-range vs ray-to-table-range. Put an object at a ruler-measured
    (x,y), hit Probe, and compare — this is the one test that says whether the map
    is right. A consistent 'reads N cm nearer than the ruler' is the localization
    BIAS behind 'it lands short', and its size tells us how to fix it."""
    joints, rgb, _ = observe()
    T_cam = T_cam_of(joints)
    z = T_cam[:3, 2]
    cp = math.degrees(math.atan2(-z[2], math.hypot(z[0], z[1])))
    depth = latest_snap[0]["depth"]
    dets = detect(rgb)
    head = f"cam {cp:.0f}° down, {(T_cam[2, 3] - TABLE_Z) * 100:.0f}cm high"
    if not dets:
        set_phase("PROBE", head + " — nothing detected"); return
    for label, (bbox, conf, src) in dets.items():
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        rng = stereo_range((cx, cy), depth)
        xy = object_xy(bbox, T_cam, depth)
        if xy is None:
            say(f"probe {label}: IPM failed (off-table/out-of-reach) "
                f"stereo={rng * 100:.0f}cm"); continue
        ray_rng = float(np.linalg.norm(np.append(xy, TABLE_Z) - T_cam[:3, 3]))
        say(f"probe {label}: MAP x={xy[0] * 100:.1f} y={xy[1] * 100:.1f}cm "
            f"(r={np.hypot(*xy) * 100:.1f}cm) | ray-range={ray_rng * 100:.0f}cm "
            f"stereo={rng * 100:.0f}cm [{src}]")
    set_phase("PROBE", head + " — see log; compare MAP x/y to a ruler")


# ---------------- scan: sweep the arc, build the map ----------------
def scan():
    """Look down-and-out at the table and pan the base across the reachable arc,
    dropping every detected object into the 2D map. FK gives the camera pose at
    every tick, so each detection lands at the right world (x,y)."""
    set_phase("SCAN", "finding a down-look, then sweeping the arc")
    # Find the gripper pitch that makes the CAMERA look ~CAM_DOWN deg down. This
    # is NOT the gripper pitch: the hand-eye offset makes the camera much
    # shallower than the hand (a 45° hand gave only ~2° camera depression), which
    # is why IPM was landing objects out to the wall. Search gripper pitch for the
    # one that actually points the optical axis down at the table.
    j = observe()[0].astype(np.float64)
    best = None
    for gp in np.arange(40.0, 100.0, 4.0):
        q, e = ik_xy(j, np.array([0.24, 0.0, TABLE_Z + 0.18]), gp, float(j[4]), ret_err=True)
        if e > 0.02:
            continue
        cd = cam_depression(q)
        if best is None or abs(cd - CAM_DOWN) < abs(best[2] - CAM_DOWN):
            best = (q, gp, cd)
    if best is None:
        raise Abort("could not find a reachable down-look scan pose")
    q0, hold_pitch, cd = best
    say(f"scan look: hand {hold_pitch:.0f}° -> camera {cd:.0f}° down")
    goto(q0, settle=0.5)
    base = observe()[0].astype(np.float64)
    start = float(base[0])
    # Sweep LEFT then RIGHT across the front arc SMOOTHLY: stream a fine, steady
    # pan target (~0.6°/tick at 30 Hz ≈ 18°/s) so the servo glides instead of
    # jumping. Sensing runs every few ticks (it's slower than the motion), so the
    # camera motion stays smooth regardless of detector/bus latency.
    for target in (start - 50.0, start + 50.0, start):
        target = float(np.clip(target, -ARC_DEG, ARC_DEG))
        cur = float(observe()[0][0])
        n = 0
        while abs(cur - target) > 1.0:
            checkpoint()
            cur += float(np.clip(target - cur, -0.6, 0.6))      # fine, smooth step
            base[0] = cur
            base[3] = _slave_wflex(base[1], base[2], hold_pitch)
            send_joints(base)
            if n % 4 == 0:                                       # sense a few times/sec
                joints, rgb, _ = observe()
                base[1:3] = joints[1:3]                          # keep lift/elbow fresh
                sense(joints, rgb, latest_snap[0]["depth"])
            n += 1
            time.sleep(0.033)
    n = len(world.snapshot())
    set_phase("MAPPED", f"{n} object(s) on the map")


# ---------------- pick: IK to the cell, descend, grasp ----------------
def close_gripper():
    idle = [c for c in (gripper_current() for _ in range(8)) if c is not None]
    i0 = float(np.mean(idle)) if idle else 0.0
    pct = 95.0
    while pct > 2.0:
        checkpoint()
        pct -= 4.0
        j = observe()[0]
        send_joints(j, gripper=pct)
        time.sleep(0.14)
        c = gripper_current()
        if c is not None and abs(c - i0) >= 8.0:
            send_joints(j, gripper=max(0.0, pct - 12.0))
            time.sleep(0.4)
            return True
    send_joints(observe(overlay=False)[0], gripper=40.0)
    return False


def pick(tag, dry=False):
    got = world.get(tag)
    if got is None:
        raise Abort(f"tag {tag} not on the map")
    label, xy = got
    if not reachable(*xy):
        raise Abort(f"{label} #{tag} at r={np.hypot(*xy)*100:.0f}cm is out of reach")
    set_phase("PICK", f"{label} #{tag} @ ({xy[0]*100:.0f},{xy[1]*100:.0f})cm")
    j = observe()[0].astype(np.float64)
    pitch, e = grasp_pitch_for(xy[0], xy[1], j)
    if pitch is None:
        raise Abort(f"{label} #{tag}: no reachable grasp pitch (IK {e*1e3:.0f}mm)")
    send_joints(j, gripper=95.0); time.sleep(0.3)

    # 1) hover above the cell, 2) descend onto the table, 3) close.
    above = np.array([xy[0], xy[1], TABLE_Z + 0.06])
    q_above, e1 = ik_xy(observe()[0], above, pitch, float(j[4]), ret_err=True)
    if e1 > 0.006:
        raise Abort(f"hover unreachable (IK {e1*1e3:.0f}mm)")
    goto(q_above, settle=0.5)

    if dry:
        set_phase("DONE", f"dry: hovering over {label} #{tag}")
        return
    grip = np.array([xy[0], xy[1], TABLE_Z + 0.015])
    q_grip, e2 = ik_xy(observe()[0], grip, pitch, float(j[4]), ret_err=True)
    if e2 < 0.008:
        goto(q_grip, settle=0.4, step=1.0)
    set_phase("PICK", "closing")
    held = close_gripper()
    q_lift, e3 = ik_xy(observe()[0], above + np.array([0, 0, 0.06]), pitch, float(j[4]), ret_err=True)
    if e3 < 0.02:
        goto(q_lift, settle=0.5)
    if held:
        world.remove(tag)
    set_phase("DONE" if held else "PICK", f"{label} #{tag} {'picked' if held else 'missed'}")


# ---------------- overlay + detector thread ----------------
def publish(rgb, joints=None):
    img = np.ascontiguousarray(rgb[:, :, ::-1])
    cv2.drawMarker(img, (int(HAND_UV[0]), int(HAND_UV[1])), (60, 200, 255), cv2.MARKER_CROSS, 24, 2)
    if joints is not None and kin is not None and fx > 0:
        try:
            uv = tip_pixel(joints)
        except Exception:
            uv = None
        if uv is not None:
            gap = math.hypot(uv[0] - HAND_UV[0], uv[1] - HAND_UV[1])
            cv2.putText(img, f"hand-eye {gap:.0f}px", (10, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if gap < 40 else (255, 0, 255), 1)
    with lock:
        dets = list(state.get("dets") or [])
        ph = state["phase"]
    now = time.time()
    for d in dets:
        if now - d["t"] > 0.8:
            continue
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"{d['label']} {d['conf']:.2f}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    # reproject the mapped objects into the FPV (yellow = honest check: on the object?)
    if joints is not None and fx > 0:
        try:
            T_cam = T_cam_of(joints)
        except Exception:
            T_cam = None
        if T_cam is not None:
            for o in world.snapshot():
                uv = project_base(np.array([o["x"], o["y"], TABLE_Z]), T_cam)
                if uv is None:
                    continue
                u, v = int(round(uv[0])), int(round(uv[1]))
                if -50 < u < img.shape[1] + 50 and -50 < v < img.shape[0] + 50:
                    cv2.drawMarker(img, (u, v), (0, 235, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
                    cv2.putText(img, f"{o['label']}#{o['tag']}", (u + 8, v),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 235, 255), 1)
    cv2.putText(img, ph, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 255, 120), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if ok:
        with lock:
            frame_jpeg[0] = buf.tobytes()


def yolo_worker():
    while True:
        time.sleep(0.05)
        if pending_query[0] is not None and detector is not None:
            q = pending_query[0]; pending_query[0] = None
            try:
                detector.set_query(q)
                det_classes[0] = [p.strip() for p in q.split(",") if p.strip()] or [q]
                with lock:
                    state["query"] = q
                say(f"query set: {q}")
            except Exception as e:
                say(f"query change failed: {e}")


def idle_view():
    while True:
        with lock:
            busy = state["running"]
        if not busy:
            try:
                joints, rgb, _ = observe()
                sense(joints, rgb, latest_snap[0]["depth"])   # keep mapping while idle
            except Exception:
                time.sleep(0.5)
        time.sleep(0.2)


# ---------------- web ----------------
app = Flask(__name__)
PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RAX 2D map</title><style>
body{margin:0;background:#0f141a;color:#e6ecf1;font:15px/1.5 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px;display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:800px){main{grid-template-columns:1fr}}
h1{grid-column:1/-1;font-size:16px;margin:0}
img,canvas{width:100%;border:1px solid #263340;border-radius:6px;background:#000;display:block}
.lbl{font:11px ui-monospace,monospace;color:#8aa;text-transform:uppercase;letter-spacing:.1em;margin:0 0 4px}
.panel{grid-column:1/-1;background:#18212b;border:1px solid #263340;border-radius:6px;padding:12px}
.phase{font:600 18px ui-monospace,monospace;color:#4cc275}.phase.bad{color:#d4795f}
table{width:100%;border-collapse:collapse;font:12.5px ui-monospace,monospace}
td,th{padding:3px 6px;border-bottom:1px solid #263340;text-align:left;color:#9ab}
tr.obj{cursor:pointer}tr.obj:hover td{background:#223}
.btns{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
button{padding:9px 14px;border:0;border-radius:5px;font:600 13px system-ui;cursor:pointer;background:#2b3846;color:#e6ecf1}
#b-scan{background:#2e9e5b}#b-stop{background:#b0533c}
input{padding:8px;background:#0f141a;border:1px solid #37454f;border-radius:5px;color:#e6ecf1}
pre{background:#0b0f14;border:1px solid #263340;border-radius:6px;padding:8px;height:150px;overflow:auto;font-size:11px;white-space:pre-wrap}
</style></head><body><main>
<h1>RAX · 2D object map <span style="color:#8aa;font-weight:400">· stereo → base-plane grid → IK</span></h1>
<div><p class="lbl">gripper camera · yellow = mapped object reprojected</p><img src="/stream"></div>
<div><p class="lbl">top-down map (base frame) · click a row below to pick</p>
  <canvas id="map" width="520" height="520"></canvas></div>
<div class="panel">
  <div class="phase" id="phase">—</div><div id="detail" style="color:#8aa;font-size:13px;min-height:18px"></div>
  <table id="tab"><thead><tr><th>tag</th><th>label</th><th>x cm</th><th>y cm</th><th>r</th><th>ang</th><th>n</th><th>reach</th></tr></thead><tbody></tbody></table>
  <div class="btns">
    <button id="b-scan" onclick="fetch('/scan',{method:'POST'})">Scan + Map</button>
    <button onclick="fetch('/probe',{method:'POST'})" title="ruler-check localization for the current view (no motion)">Probe</button>
    <button id="b-stop" onclick="fetch('/stop',{method:'POST'})">Stop</button>
    <button onclick="fetch('/home',{method:'POST'})">Home</button>
    <button onclick="fetch('/reset',{method:'POST'})">Reset pose</button>
    <button onclick="fetch('/clearmap',{method:'POST'})">Clear map</button>
    <button onclick="fetch('/calib',{method:'POST'})" title="put an object in view first">Calib hand-eye</button>
    <input id="q" placeholder="red cube, green cube" style="flex:1;min-width:140px">
    <button onclick="setq()">Set query</button>
  </div>
  <pre id="log"></pre>
</div></main><script>
const cv=document.getElementById('map'),ctx=cv.getContext('2d');
let M={objs:[],reach:{rmin:8,rmax:42,arc:110}};
function W2S(x,y){ // base metres -> screen px. base +x = up, +y = left (top-down)
  const s=cv.width/1.0, ox=cv.width/2, oy=cv.height*0.82;
  return [ox - y*s, oy - x*s];
}
function drawMap(){
  ctx.fillStyle='#0b0f14';ctx.fillRect(0,0,cv.width,cv.height);
  const s=cv.width/1.0;
  // reachable annulus + arc
  ctx.strokeStyle='#1d2a36';ctx.fillStyle='#121b24';
  for(const rr of [M.reach.rmax, M.reach.rmin]){
    ctx.beginPath();
    const a0=-M.reach.arc*Math.PI/180, a1=M.reach.arc*Math.PI/180;
    const [ox,oy]=W2S(0,0);
    ctx.arc(ox,oy,rr/100*s,-Math.PI/2-a1,-Math.PI/2-a0);
    ctx.stroke();
  }
  // grid rings every 10cm
  ctx.strokeStyle='#15202b';ctx.fillStyle='#456';ctx.font='10px monospace';
  const [ox,oy]=W2S(0,0);
  for(let r=10;r<=40;r+=10){ctx.beginPath();ctx.arc(ox,oy,r/100*s,0,7);ctx.stroke();
    ctx.fillText(r+'cm',ox+2,oy-r/100*s+12);}
  // base + axes
  ctx.fillStyle='#c9524a';ctx.beginPath();ctx.arc(ox,oy,5,0,7);ctx.fill();
  ctx.fillStyle='#8aa';ctx.fillText('base',ox+6,oy+4);
  // objects
  for(const o of M.objs){
    const [sx,sy]=W2S(o.x,o.y);
    const c=/red/.test(o.label)?'#e2574c':/green/.test(o.label)?'#3fc46b':/blue/.test(o.label)?'#4a93c9':'#e0b040';
    ctx.fillStyle=o.reach?c:'#666';ctx.beginPath();ctx.arc(sx,sy,7,0,7);ctx.fill();
    ctx.fillStyle='#e6ecf1';ctx.font='11px monospace';
    ctx.fillText(`${o.label}#${o.tag}`,sx+9,sy+4);
  }
}
async function poll(){
  try{const s=await (await fetch('/status')).json();
    document.getElementById('phase').textContent=s.phase;
    document.getElementById('phase').className='phase'+(/ABORT|ERROR/.test(s.phase)?' bad':'');
    document.getElementById('detail').textContent=s.detail||'';
    document.getElementById('log').textContent=(s.log||[]).join('\\n');
    M.objs=s.map||[];drawMap();
    const tb=document.querySelector('#tab tbody');
    tb.innerHTML=(s.map||[]).map(o=>`<tr class="obj" onclick="pick(${o.tag})"><td>${o.tag}</td><td>${o.label}</td><td>${(o.x*100).toFixed(0)}</td><td>${(o.y*100).toFixed(0)}</td><td>${o.r_cm}</td><td>${o.ang}°</td><td>${o.n}</td><td>${o.reach?'✓':'—'}</td></tr>`).join('')||'<tr><td colspan=8>empty — Scan + Map</td></tr>';
    const qi=document.getElementById('q');if(s.query&&!qi.value&&document.activeElement!==qi)qi.value=s.query;
  }catch(e){}
  setTimeout(poll,500);
}
function pick(t){fetch('/pick?tag='+t,{method:'POST'});}
function setq(){const q=document.getElementById('q').value.trim();if(q)fetch('/setquery?q='+encodeURIComponent(q),{method:'POST'});}
poll();
</script></body></html>"""


@app.route("/")
def index():
    r = Response(PAGE, mimetype="text/html")
    r.headers["Cache-Control"] = "no-store"
    return r


@app.route("/status")
def status():
    with lock:
        s = {k: v for k, v in state.items()}
    s["log"] = list(log)
    s["map"] = world.snapshot()
    return jsonify(s)


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


def _run(fn, *a):
    with lock:
        if state["running"]:
            return False
    def _t():
        stop_flag.clear()
        with lock:
            state["running"] = True
        try:
            fn(*a)
        except Abort as e:
            set_phase("ABORTED", str(e))
        except Exception as e:
            set_phase("ERROR", f"{type(e).__name__}: {e}")
        finally:
            with lock:
                state["running"] = False
    mission_thread[0] = threading.Thread(target=_t, daemon=True)
    mission_thread[0].start()
    return True


@app.route("/scan", methods=["POST"])
def r_scan():
    return jsonify(ok=_run(scan))


@app.route("/probe", methods=["POST"])
def r_probe():
    return jsonify(ok=_run(probe))


@app.route("/pick", methods=["POST"])
def r_pick():
    tag = int(request.args.get("tag", "0"))
    dry = request.args.get("dry") in ("1", "true")
    return jsonify(ok=_run(pick, tag, dry))


@app.route("/stop", methods=["POST"])
def r_stop():
    stop_flag.set(); say("STOP")
    return jsonify(ok=True)


@app.route("/clearmap", methods=["POST"])
def r_clearmap():
    global world
    world = World2D()
    say("map cleared")
    return jsonify(ok=True)


@app.route("/reset", methods=["POST"])
def r_reset():
    return jsonify(ok=_run(lambda: (goto(VIEW, settle=0.8), send_joints(observe()[0], gripper=95.0),
                                    set_phase("IDLE", "at view pose"))))


@app.route("/home", methods=["POST"])
def r_home():
    return jsonify(ok=_run(lambda: (goto(HOME, settle=1.0), set_phase("IDLE", "folded"))))


@app.route("/setquery", methods=["POST"])
def r_setquery():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify(ok=False)
    pending_query[0] = q
    return jsonify(ok=True)


@app.route("/calib", methods=["POST"])
def r_calib():
    return jsonify(ok=_run(calibrate_handeye))


# ---------------- hand-eye self-calibration (motion-based, same fit as before) ----------------
def calibrate_handeye(n_target=14):
    global T_ee_cam
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    set_phase("CALIB", "sampling an object from several poses")
    label = det_classes[0][0]
    with lock:
        cur = [d["label"] for d in (state.get("dets") or [])]
    if cur:
        label = cur[0]
    q0 = observe()[0].astype(np.float64)

    def see():
        for _ in range(20):
            joints, rgb, _ = observe()
            depth = latest_snap[0]["depth"]
            d = detect(rgb).get(label)
            if d:
                x1, y1, x2, y2 = d[0]
                uv = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                return joints, uv, stereo_range(uv, depth)   # + camera-frame range (stereo)
            time.sleep(0.05)
        return None
    s0 = see()
    if s0 is None:
        raise Abort(f"no '{label}' in view")
    # Bigger, more diverse sweep than before: translation + rotation must change
    # ENOUGH that the object's bearing shifts, or the camera position is not
    # observable and the fit drifts (the 19cm/pointing-up minimum).
    deltas = ([np.array([p, 0, 0, 0, 0.]) for p in (-15, -8, 8, 15)]
              + [np.array([0, 0, 0, w, 0.]) for w in (-14, -7, 7, 14)]
              + [np.array([0, a, b, 0, 0.]) for a, b in ((-8, 8), (8, -8), (-6, 14), (6, -14), (-12, 6))])
    samples = [s0]
    for d in deltas:
        checkpoint()
        goto(q0 + d, settle=0.3, step=1.5)
        s = see()
        if s:
            samples.append(s)
        if len(samples) >= n_target:
            break
    goto(q0, settle=0.3)
    if len(samples) < 8:
        raise Abort(f"only {len(samples)} views")
    T_ee = [fk(j) for j, _, _ in samples]
    uvs = [uv for _, uv, _ in samples]
    rngs = [r for _, _, r in samples]
    n = len(samples)
    w_tip = math.sqrt(n)
    w_depth = 200.0                    # px-equivalent weight on the stereo range residual (m→px)
    w_tmag = 300.0                     # px-equivalent weight on the |t|≈CAM_TIP_M prior
    CAM_TIP_M = 0.10                   # measured camera→fingertip mount distance

    def unpack(x):
        tf = np.eye(4); tf[:3, 3] = x[:3]; tf[:3, :3] = Rotation.from_rotvec(x[3:6]).as_matrix()
        return tf, np.array(x[6:9])

    def resid(x):
        tf, p = unpack(x); r = []
        for T, uv, rng in zip(T_ee, uvs, rngs):
            Tc = T @ tf
            pu = project_base(p, Tc)
            r += [400, 400] if pu is None else [pu[0] - uv[0], pu[1] - uv[1]]
            # STEREO DEPTH constraint: the object's modelled camera-frame depth
            # must match the measured stereo range. This is what actually pins the
            # camera DISTANCE (and kills the 19cm drift) — pixels alone can't.
            if np.isfinite(rng) and 0.12 < rng < 0.60:
                pc = np.linalg.inv(Tc) @ np.append(p, 1.0)
                r.append(w_depth * (float(pc[2]) - rng))
        pt = project_base(T_ee[0][:3, 3], T_ee[0] @ tf)
        r += [400, 400] if pt is None else [w_tip * (pt[0] - HAND_UV[0]), w_tip * (pt[1] - HAND_UV[1])]
        # translation-magnitude prior: the camera is a MEASURED ~10cm from the tip.
        r.append(w_tmag * (float(np.linalg.norm(x[:3])) - CAM_TIP_M))
        return r
    p0 = ray_to_table(uvs[0], T_ee[0] @ T_ee_cam)
    p_seed = np.array([p0[0], p0[1], TABLE_Z]) if p0 is not None else np.array([0.25, 0, TABLE_Z])
    # seed the translation at the MEASURED 10cm (keep the old direction, rescaled)
    t_old = np.asarray(T_ee_cam[:3, 3], np.float64)
    t_seed = t_old / max(1e-6, np.linalg.norm(t_old)) * CAM_TIP_M
    x0 = np.concatenate([t_seed, Rotation.from_matrix(T_ee_cam[:3, :3]).as_rotvec(), p_seed])
    lo = np.array([-.14, -.14, -.14, -4, -4, -4, -.45, -.45, TABLE_Z - .02])
    hi = np.array([.14, .14, .14, 4, 4, 4, .45, .45, TABLE_Z + .08])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)

    def rms(x):
        tf, p = unpack(x)
        rr = [project_base(p, T @ tf) for T in T_ee]
        e = [math.hypot(pu[0] - uv[0], pu[1] - uv[1]) for pu, (uv) in zip(rr, uvs) if pu is not None]
        return float(np.sqrt(np.mean(np.square(e)))) if e else 999.0
    sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=6000)
    tf, p = unpack(sol.x)
    tmag = float(np.linalg.norm(tf[:3, 3]))
    zc = (fk(q0) @ tf)[:3, 2]
    cd = math.degrees(math.atan2(-zc[2], math.hypot(zc[0], zc[1])))
    say(f"hand-eye fit: RMS={rms(sol.x):.0f}px |t|={tmag*100:.0f}cm cam≈{cd:.0f}° down ({n} views)")
    if rms(sol.x) > 40.0:
        raise Abort(f"fit did not converge (RMS {rms(sol.x):.0f}px)")
    rv = Rotation.from_matrix(tf[:3, :3]).as_rotvec()
    tf_str = ",".join(f"{v:.4f}" for v in list(tf[:3, 3]) + list(rv))
    json.dump({"tf": tf_str, "rms_px": rms(sol.x), "fitted": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(TF_FILE, "w"), indent=2)
    T_ee_cam = tf
    set_phase("CALIB", f"hand-eye fixed — {rms(sol.x):.0f}px")


# ---------------- main ----------------
def main():
    global robot, kin, cam, detector, fx, fy, cx0, cy0
    clear_overload()
    say("connecting robot + camera…")
    for attempt in range(6):
        robot = make_robot_from_config(SO101FollowerConfig(
            port="COM4", id="so101_follower",
            cameras={"front": OAKDCameraConfig(fps=30, width=640, height=480, use_depth=True,
                                               stereo_extended_disparity=True,
                                               stereo_confidence_threshold=150)}))
        try:
            robot.connect(); break
        except (RuntimeError, ConnectionError) as e:
            if attempt == 5:
                raise
            say(f"connect glitch ({str(e)[:50]}) retry {attempt+1}/6")
            try:
                robot.bus.port_handler.closePort()
            except Exception:
                pass
            time.sleep(2.0)
    kin = RobotKinematics(LEROBOT + r"\SO101\so101_new_calib.urdf", "gripper_frame_link", ARM_MOTORS)
    cam = robot.cameras["front"]
    intr = {"fx": 517.0, "fy": 517.0, "cx": 329.5, "cy": 231.4}
    if hasattr(cam, "get_depth_intrinsics"):
        try:
            intr = dict(cam.get_depth_intrinsics())
        except Exception:
            pass
    fx, fy, cx0, cy0 = (float(intr[k]) for k in ("fx", "fy", "cx", "cy"))
    d = load_tf()
    if d:
        say(f"hand-eye: {d.get('fitted','?')} ({d.get('rms_px',0):.0f}px)")
    try:
        uv = tip_pixel(np.array(HOME, np.float64))
        gap = math.hypot(uv[0] - HAND_UV[0], uv[1] - HAND_UV[1]) if uv else 999
        say(f"hand-eye check: fingertip {gap:.0f}px from HAND_UV" + ("  *** run Calib ***" if gap > 40 else " — OK"))
    except Exception:
        pass
    say("loading YOLO-World S…")
    detector = YoloWorldDetector(LEROBOT + r"\yolov8s-worldv2.pt", conf=0.08, imgsz=640,
                                 color_filter_min_frac=0.0)
    detector.set_query(", ".join(det_classes[0]))
    with lock:
        state["query"] = ", ".join(det_classes[0])
    threading.Thread(target=yolo_worker, daemon=True).start()
    threading.Thread(target=idle_view, daemon=True).start()
    set_phase("IDLE", "ready — Scan + Map")
    say(f"UI: http://{HOST_IP}:{PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
