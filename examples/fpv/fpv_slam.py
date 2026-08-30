# fpv_slam.py — FPV table-top SLAM: 2D object map + map-based picking. One file.
#
# ================================ THE PHYSICS ================================
# The robot sits ON the table. Everything it can ever grasp also sits on the
# table. So the world is TWO-DIMENSIONAL: the table plane, which is the base
# plane (z = TABLE_Z in the base frame). An object is fully described by
# (label, x, y) — nothing else is needed to find it, approach it, or pick it.
#
# The camera rides on the gripper head (first-person view), ~10 cm behind the
# fingertips, pitched ~30 deg down at them. Its pose is never estimated — it is
# COMPUTED EXACTLY from the servo angles:
#
#       T_cam = FK(joint_angles) @ T_ee_cam          (hand-eye, CAD-measured)
#
# FK from servo encoders is exact odometry. SLAM with known poses is not SLAM
# any more — it is pure mapping. That is why this file is short.
#
# LOCALIZATION (vision problem #1): a pixel is a ray. An object resting on the
# table intersects that ray exactly where the ray pierces the table plane:
#
#       p = o + t d ,  t = (TABLE_Z - o_z) / d_z     (inverse perspective map)
#
# Cast rays through the BOTTOM band of the detection box (the object's ground
# contact) -> (x, y). Distance to the object is ||p - o|| for free. The OAK-D's
# stereo depth is biased/blind at grasp range (it reads the far wall through a
# close cube), so stereo is used ONLY as a plausibility gate, never as position.
# The table plane is the measurement; the camera is just the bearing sensor.
#
# TRACKING: nearest-neighbour association in the 2D plane + running mean. One
# tag per physical object. Objects persist when they leave the view (the table
# is static; the base is bolted down). That is the whole tracker.
#
# APPROACH (problem #2): with every object at a known base (x, y), motion is
# "just math": IK the fingertip above (x, y), descend, close. NO pixel-space
# servoing — pixel Jacobians through this mount flip sign and stall (the
# stack_mission2 failure). And YOLO hallucinating at close range does not
# matter: the final leg runs on the remembered coordinate, and success is
# FELT (gripper current), not seen.
#
# Reach (URDF sweep at table height): usable annulus r ~= 10..42 cm over a
# +/-110 deg pan arc. The map only marks objects inside it as pickable.
#
# Flow:  SCAN (pan sweep, engine fuses detections into the map)
#     -> LOOK (park the camera on a tag from a good vantage; a few more fused
#        views tighten its coordinate — then FREEZE it)
#     -> PICK (pure IK: hover 6 cm over the frozen cell, descend, close, lift)
# ============================================================================
import json, math, os, sys, threading, time
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
PORT = 8487
HOST_IP = "100.110.89.78"

# ---- world geometry (base frame; +x forward, +y left, z up) ----
TABLE_Z = -0.02                 # table surface, measured FK tip contact (~-0.022)
REACH_MIN, REACH_MAX = 0.10, 0.42   # usable radius at table height (URDF sweep)
ARC_DEG = 110.0                 # usable pan half-arc (shoulder_pan joint limit)
MERGE_M = 0.10                  # same-label detections closer than this = same object.
                                # (Live: an UNCALIBRATED hand-eye smears one cube ~10 cm
                                # along its bearing across a scan; 10 cm keeps one cube =
                                # one tag. Run Calib hand-eye to shrink the scatter, then
                                # this can come down to ~0.05 for dense scenes.)

# ---- camera / hand-eye ----
TF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handeye_tf.json")
T_ee_cam = parse_tf_string("0.0000,-0.0500,-0.0796,-0.4562,-0.3195,-1.1922")  # CAD
fx = fy = cx0 = cy0 = 0.0
HAND_UV = (440.0, 394.0)        # measured fingertip pixel in the FPV
GRIP_TIP = 0.007                # gripper_frame_link is the fingertip +7 mm
CAM_DOWN_SCAN = 38.0            # camera depression while scanning (deg below horizon)
STEREO_TRUST = 0.55             # stereo depth is consulted only below this range (m)

# ---- poses / IK ----
VIEW = np.array([-6.7, 37.1, 48.1, -40.4, -28.4])
HOME = np.array([-9.67, -102.022, 98.066, 32.879, 0.0])
J_LO = np.array([-110.0, -100.0, -96.8, -95.0, -157.2])   # so101_new_calib.urdf
J_HI = np.array([+110.0, +100.0, +96.8, +95.0, +162.8])
WFLEX_MIN, WFLEX_MAX = float(J_LO[3]), float(J_HI[3])
GRASP_PITCH = (75.0, 80.0, 70.0, 85.0, 90.0, 65.0, 60.0, 55.0, 50.0)  # steep-first
HOVER_H, GRIP_H = 0.06, 0.015   # fingertip height above the table for hover / grasp

state = {"phase": "IDLE", "detail": "", "joints": [], "gripper": None,
         "running": False, "query": "", "dets": []}
log = deque(maxlen=120)
frame_jpeg = [None]
lock = threading.Lock()
bus_lock = threading.RLock()
kin_lock = threading.Lock()
map_lock = threading.Lock()
stop_flag = threading.Event()
latest_snap = [None]            # {"rgb","depth","q","t"} — joints PAIRED with the frame
pending_query = [None]
det_classes = [["red cube", "green cube"]]
robot = kin = cam = detector = None


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


# ---------------- kinematics ----------------
def fk(q):
    with kin_lock:
        return np.asarray(kin.forward_kinematics(np.asarray(q, np.float64)))


def T_cam_of(q):
    return fk(q) @ T_ee_cam


def cam_depression(q):
    """Optical-axis angle below the horizon (deg)."""
    z = T_cam_of(q)[:3, 2]
    return math.degrees(math.atan2(-z[2], math.hypot(z[0], z[1])))


def _slave_wflex(j1, j2, pitch):
    # lift/elbow/wrist_flex are parallel axes: gripper world pitch = j1+j2+j3 exactly.
    return float(np.clip(pitch - j1 - j2, WFLEX_MIN, WFLEX_MAX))


def ik_xy(q_seed, p_tgt, pitch_tgt, j5, iters=80, tol=2e-3, ret_err=False, _retry=True):
    """Position IK on pan/lift/elbow with wrist_flex slaved to the gripper pitch.
    Limit-clamped; returns the residual so callers can refuse unreachable targets."""
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
    return REACH_MIN <= r <= REACH_MAX and abs(math.degrees(math.atan2(y, x))) <= ARC_DEG


def solve_pitch(x, y, z, q_seed):
    """Steepest gripper pitch the IK can actually hold at (x, y, z), steep-first.
    Close cells need a near-vertical hand (fingertip is 98 mm ahead of the wrist)."""
    best = None
    for pitch in GRASP_PITCH:
        _, e = ik_xy(q_seed, np.array([x, y, z]), pitch, float(q_seed[4]), ret_err=True)
        if e < 0.006:
            return pitch, e
        if best is None or e < best[1]:
            best = (pitch, e)
    return None, best[1]


# ---------------- robot I/O ----------------
def observe(overlay=True):
    """Read arm joints + camera frame (+ aligned depth), paired into one snapshot."""
    checkpoint()
    joints = None
    err = None
    for _ in range(12):
        try:
            with bus_lock:
                pos = robot.bus.sync_read("Present_Position", ARM_MOTORS, num_retry=2)
            joints = np.array([float(pos[m]) for m in ARM_MOTORS])
            break
        except Exception as e:
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
        latest_snap[0] = {"rgb": rgb, "depth": depth, "q": joints.copy(), "t": time.time()}
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


def close_gripper():
    """Close slowly, stop the instant servo current rises (contact = FELT, not seen)."""
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
    send_joints(observe(overlay=False)[0], gripper=40.0)   # closed on air: relax
    return False


def clear_overload():
    """Torque off/on cycle on every servo id — clears latched overload errors."""
    try:
        import scservo_sdk as scs
        ph = scs.PortHandler("COM4")
        if not ph.openPort():
            return
        ph.setBaudRate(1000000)
        pk = scs.PacketHandler(0)
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


# ---------------- geometry: pixel <-> base plane ----------------
def project_base(p_base, T_base_cam):
    pc = np.linalg.inv(np.asarray(T_base_cam, np.float64)) @ np.append(p_base, 1.0)
    if pc[2] <= 1e-4:
        return None
    return (float(fx * pc[0] / pc[2] + cx0), float(fy * pc[1] / pc[2] + cy0))


def ray_to_table(uv, T_base_cam, z=TABLE_Z):
    """Inverse perspective mapping: a pixel's sightline ∩ the table plane."""
    o = T_base_cam[:3, 3]
    d = T_base_cam[:3, :3] @ np.array([(uv[0] - cx0) / fx, (uv[1] - cy0) / fy, 1.0])
    if d[2] >= -1e-3:                # ray not pointing down at the plane
        return None
    t = (z - o[2]) / d[2]
    if t <= 0:
        return None
    return (o + t * d)[:2]


def stereo_range(uv, depth, win=6):
    """Median stereo depth near a pixel — plausibility gate only, never position."""
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


# ---------------- detection: YOLO-World + HSV fallback ----------------
HSV = {"red": [((0, 110, 80), (9, 255, 255)), ((170, 110, 80), (179, 255, 255))],
       "green": [((38, 80, 60), (85, 255, 255))],
       "blue": [((95, 90, 60), (130, 255, 255))],
       "yellow": [((20, 90, 80), (34, 255, 255))]}


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
    """{label: (bbox, conf, src)} — best YOLO box per class, HSV fallback per class."""
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


# ---------------- the 2D map ----------------
def ipm_xy(bbox, T_cam, depth):
    """Detection -> (x, y) on the table plane + camera range. Rays through the
    box's bottom band (ground contact), median. Stereo only gates plausibility."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 4:
        return None
    hits = []
    for v in np.linspace(y1 + 0.66 * h, y2 - 0.05 * h, 3):
        for u in np.linspace(x1 + 0.25 * w, x2 - 0.25 * w, 3):
            p = ray_to_table((u, v), T_cam)
            if p is not None and 0.05 < math.hypot(p[0], p[1]) < 0.60 \
                    and abs(math.degrees(math.atan2(p[1], p[0]))) < 125:
                hits.append(p)
    if len(hits) < 3:
        return None
    xy = np.median(np.array(hits), axis=0)
    rng = stereo_range(((x1 + x2) / 2, (y1 + y2) / 2), depth)
    if np.isfinite(rng) and rng < STEREO_TRUST:
        # stereo only speaks when it plausibly sees the object itself; reject
        # detections whose ray∩table range disagrees wildly (hand in view etc.)
        ray_rng = float(np.linalg.norm(np.append(xy, TABLE_Z) - T_cam[:3, 3]))
        if abs(rng - ray_rng) > 0.12:
            return None
    return xy


class World2D:
    """Flat object map in the base X-Y plane: tag -> label + running-mean (x,y).
    Nearest-neighbour association within MERGE_M is the whole tracker."""

    def __init__(self):
        self.objs = {}
        self._next = 1

    def update(self, label, xy, conf):
        with map_lock:
            best, bd = None, MERGE_M
            for tag, o in self.objs.items():
                if o["label"] != label or o["frozen"]:
                    continue
                d = float(np.linalg.norm(o["xy"] - xy))
                if d < bd:
                    best, bd = tag, d
            if best is None:
                tag = self._next
                self._next += 1
                self.objs[tag] = {"label": label, "xy": np.asarray(xy, np.float64),
                                  "n": 1, "conf": conf, "last_seen": time.time(),
                                  "frozen": False}
                say(f"map: new {label} #{tag} @ ({xy[0]*100:.0f},{xy[1]*100:.0f})cm")
            else:
                o = self.objs[best]
                o["xy"] = 0.7 * o["xy"] + 0.3 * np.asarray(xy, np.float64)
                o["n"] += 1
                o["conf"] = max(o["conf"], conf)
                o["last_seen"] = time.time()

    def get(self, tag):
        with map_lock:
            o = self.objs.get(tag)
            return (o["label"], o["xy"].copy()) if o else None

    def set_frozen(self, tag, val):
        with map_lock:
            if tag in self.objs:
                self.objs[tag]["frozen"] = val

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
                     "frozen": o["frozen"], "age": round(now - o["last_seen"], 1)}
                    for t, o in self.objs.items()]


world = World2D()


# ---------------- perception engine (one worker thread does ALL fusion) ----------------
def engine():
    """Every new snapshot -> detect -> IPM -> map. Runs continuously, so the map
    builds while scanning, while parked, whenever the camera sees the table."""
    last_t = 0.0
    while True:
        time.sleep(0.03)
        if pending_query[0] is not None and detector is not None:
            q = pending_query[0]
            pending_query[0] = None
            try:
                detector.set_query(q)
                det_classes[0] = [p.strip() for p in q.split(",") if p.strip()] or [q]
                with lock:
                    state["query"] = q
                say(f"query set: {q}")
            except Exception as e:
                say(f"query change failed: {e}")
        with lock:
            snap = latest_snap[0]
        if snap is None or detector is None or snap["t"] == last_t:
            continue
        last_t = snap["t"]
        dets = detect(snap["rgb"])
        T_cam = T_cam_of(snap["q"])          # pose AT CAPTURE — never "now"
        ui = []
        for label, (bbox, conf, src) in dets.items():
            ui.append({"label": label, "bbox": [float(v) for v in bbox],
                       "conf": round(conf, 2), "src": src, "t": snap["t"]})
            if conf < 0.14:
                continue
            x1, y1, x2, y2 = bbox
            if x1 <= 2 or y1 <= 2 or x2 >= snap["rgb"].shape[1] - 3 \
                    or y2 >= snap["rgb"].shape[0] - 3:
                continue                     # clipped box = partial object
            xy = ipm_xy(bbox, T_cam, snap["depth"])
            if xy is not None:
                world.update(label, xy, conf)
        with lock:
            state["dets"] = ui


# ---------------- scan: sweep the arc, the engine maps everything ----------------
def find_scan_pose():
    """Solve ONE raised down-look pose: tip forward and up, camera ~CAM_DOWN_SCAN
    degrees down. The camera is much shallower than the hand because of the
    hand-eye mount, so search the gripper pitch for the right CAMERA depression."""
    j = observe()[0].astype(np.float64)
    best = None
    for gp in np.arange(40.0, 100.0, 4.0):
        q, e = ik_xy(j, np.array([0.24, 0.0, TABLE_Z + 0.18]), gp, float(j[4]), ret_err=True)
        if e > 0.02:
            continue
        cd = cam_depression(q)
        if best is None or abs(cd - CAM_DOWN_SCAN) < abs(best[2] - CAM_DOWN_SCAN):
            best = (q, gp, cd)
    if best is None:
        raise Abort("could not find a reachable down-look scan pose")
    say(f"scan look: hand {best[1]:.0f}° -> camera {best[2]:.0f}° down")
    return best[0], best[1]


def scan():
    """Pan the base across the full reachable arc; the engine fuses every frame.
    Pure motion here — mapping happens on the engine thread."""
    set_phase("SCAN", "sweeping the table into the 2D map")
    q0, hold_pitch = find_scan_pose()
    goto(q0, settle=0.5)
    base = observe()[0].astype(np.float64)
    start = float(base[0])
    for target in (start - ARC_DEG, start + ARC_DEG, start):
        target = float(np.clip(target, -ARC_DEG, ARC_DEG))
        cur = float(observe()[0][0])
        while abs(cur - target) > 1.0:
            checkpoint()
            cur += float(np.clip(target - cur, -0.6, 0.6))     # ~18°/s glide
            base[0] = cur
            base[3] = _slave_wflex(base[1], base[2], hold_pitch)
            send_joints(base)
            observe()                                          # feed the engine
            base[1:3] = latest_snap[0]["q"][1:3]
            time.sleep(0.033)
    n = len(world.snapshot())
    set_phase("MAPPED", f"{n} object(s) on the map")


# ---------------- look: SLAM-style navigation to re-observe a map tag ----------------
def look_at(tag, refine_s=1.2):
    """Park the camera on a mapped object from a good vantage (hand above and
    behind it, steep pitch), so the engine gets a few more well-conditioned
    views of exactly that cell. Then FREEZE the tag: from here the coordinate
    is memory, and memory beats a close-range hallucination."""
    got = world.get(tag)
    if got is None:
        raise Abort(f"tag {tag} not on the map")
    label, xy = got
    j = observe()[0].astype(np.float64)
    pitch, e = solve_pitch(xy[0], xy[1], TABLE_Z + 0.14, j)
    if pitch is None:
        raise Abort(f"{label} #{tag}: no reachable look pose (IK {e*1e3:.0f}mm)")
    q_look, e = ik_xy(j, np.array([xy[0], xy[1], TABLE_Z + 0.14]), pitch,
                      float(j[4]), ret_err=True)
    if e > 0.008:
        raise Abort(f"look pose unreachable (IK {e*1e3:.0f}mm)")
    set_phase("LOOK", f"re-observing {label} #{tag} ({refine_s:.0f}s)")
    goto(q_look, settle=0.4)
    t_end = time.time() + refine_s
    while time.time() < t_end:               # let the engine pump views into the mean
        checkpoint()
        observe()
        time.sleep(0.1)
    world.set_frozen(tag, True)
    _l, xy = world.get(tag)
    say(f"{label} #{tag} frozen @ ({xy[0]*100:.1f},{xy[1]*100:.1f})cm "
        f"r={np.hypot(*xy)*100:.1f}cm")
    return label, xy


# ---------------- pick: pure IK through the map ----------------
def pick(tag, dry=False):
    got = world.get(tag)
    if got is None:
        raise Abort(f"tag {tag} not on the map")
    if not reachable(*got[1]):
        raise Abort(f"{got[0]} #{tag} at r={np.hypot(*got[1])*100:.0f}cm is out of reach")
    label, xy = look_at(tag)
    set_phase("PICK", f"{label} #{tag} @ ({xy[0]*100:.0f},{xy[1]*100:.0f})cm")
    j = observe()[0].astype(np.float64)
    pitch, e = solve_pitch(xy[0], xy[1], TABLE_Z + HOVER_H, j)
    if pitch is None:
        raise Abort(f"{label} #{tag}: no reachable grasp pitch (IK {e*1e3:.0f}mm)")
    send_joints(j, gripper=95.0)
    time.sleep(0.3)

    q_hover, e1 = ik_xy(observe()[0], np.array([xy[0], xy[1], TABLE_Z + HOVER_H]),
                        pitch, float(j[4]), ret_err=True)
    if e1 > 0.006:
        raise Abort(f"hover unreachable (IK {e1*1e3:.0f}mm)")
    goto(q_hover, settle=0.5)
    if dry:
        set_phase("DONE", f"dry: hovering over {label} #{tag}")
        return
    q_grip, e2 = ik_xy(observe()[0], np.array([xy[0], xy[1], TABLE_Z + GRIP_H]),
                       pitch, float(j[4]), ret_err=True)
    if e2 < 0.008:
        goto(q_grip, settle=0.4, step=1.0)
    set_phase("PICK", "closing — feeling for contact")
    held = close_gripper()
    q_lift, e3 = ik_xy(observe()[0], np.array([xy[0], xy[1], TABLE_Z + HOVER_H + 0.06]),
                       pitch, float(j[4]), ret_err=True)
    if e3 < 0.02:
        goto(q_lift, settle=0.5)
    if held:
        world.remove(tag)                    # in the hand now, not on the table
    else:
        world.set_frozen(tag, False)
    set_phase("DONE" if held else "PICK", f"{label} #{tag} {'picked' if held else 'missed'}")


# ---------------- probe: the honest ruler check (no motion) ----------------
def probe():
    """For each detection in the CURRENT view: map (x,y), camera range, stereo
    range. Put an object at a ruler-measured (x,y), hit Probe, compare."""
    joints, rgb, _ = observe()
    T_cam = T_cam_of(joints)
    depth = latest_snap[0]["depth"]
    dets = detect(rgb)
    head = f"cam {cam_depression(joints):.0f}° down, {(T_cam[2,3]-TABLE_Z)*100:.0f}cm high"
    if not dets:
        set_phase("PROBE", head + " — nothing detected")
        return
    for label, (bbox, conf, src) in dets.items():
        rng = stereo_range(((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2), depth)
        xy = ipm_xy(bbox, T_cam, depth)
        if xy is None:
            say(f"probe {label}: IPM rejected (off-table / stereo gate) "
                f"stereo={rng*100:.0f}cm [{src}]")
            continue
        ray_rng = float(np.linalg.norm(np.append(xy, TABLE_Z) - T_cam[:3, 3]))
        say(f"probe {label}: MAP x={xy[0]*100:.1f} y={xy[1]*100:.1f}cm "
            f"(r={np.hypot(*xy)*100:.1f}cm) | ray-range={ray_rng*100:.0f}cm "
            f"stereo={rng*100:.0f}cm [{src}]")
    set_phase("PROBE", head + " — compare MAP x/y against a ruler")


# ---------------- hand-eye calib: bounded cross-view consistency fit ----------------
def calibrate(n_views=9):
    """Fit ONLY the hand-eye rotation + table height (translation stays at the
    CAD measurement). One object, N pan poses: a wrong TF smears its IPM (x,y)
    across poses; the right one makes all poses agree. Objective = localization
    spread. Tight bounds (±0.2 rad, ±2 cm) — seeded at CAD, cannot go degenerate."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    global T_ee_cam, TABLE_Z
    set_phase("CALIB", "sampling one object across the arc — keep it in view")
    label = det_classes[0][0]
    q0, hold_pitch = find_scan_pose()
    goto(q0, settle=0.5)
    start = float(observe()[0][0])
    pans = np.linspace(max(-ARC_DEG, start - 70), min(ARC_DEG, start + 70), n_views)
    samples = []                              # (T_cam, uv) of the bottom-band point
    for p in pans:
        checkpoint()
        q = observe()[0].astype(np.float64)
        q[0] = float(p)
        q[3] = _slave_wflex(q[1], q[2], hold_pitch)
        goto(q, settle=0.5, step=1.5)
        joints, rgb, _ = observe()
        d = detect(rgb).get(label)
        if d is None:
            say(f"calib: no '{label}' at pan {p:.0f}° — skipping")
            continue
        x1, y1, x2, y2 = d[0]
        uv = ((x1 + x2) / 2, y1 + 0.8 * (y2 - y1))       # ground-contact band centre
        samples.append((T_cam_of(joints), uv, joints.copy()))
    goto(q0, settle=0.4)
    if len(samples) < 5:
        raise Abort(f"calib: only {len(samples)} views of '{label}' (need 5)")

    T0 = T_ee_cam.copy()
    r0 = Rotation.from_matrix(T0[:3, :3]).as_rotvec()

    def localizations(x):
        tf = T0.copy()
        tf[:3, :3] = Rotation.from_rotvec(r0 + x[:3]).as_matrix()
        pts = []
        for T_ee_q, uv, q in samples:
            T_cam = fk(q) @ tf
            p = ray_to_table(uv, T_cam, z=TABLE_Z + x[3])
            if p is None:
                return None
            pts.append(p)
        return np.array(pts)

    def resid(x):
        pts = localizations(x)
        if pts is None:
            return np.full(len(samples) * 2, 0.5)
        c = pts.mean(axis=0)
        return ((pts - c) * 25.0).ravel()     # metres -> ~px-scale residuals

    x0 = np.zeros(4)
    lo, hi = np.array([-0.2]*3 + [-0.02]), np.array([0.2]*3 + [0.02])
    sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=2000)
    pts = localizations(sol.x)
    spread0 = float(np.max(np.linalg.norm(np.ptp(localizations(x0), axis=0)))) \
        if localizations(x0) is not None else float("nan")
    spread = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    say(f"calib: view-to-view scatter {spread0*100:.1f}cm -> {spread*100:.1f}cm "
        f"({len(samples)} views)")
    if spread > 0.05:
        raise Abort(f"calib: scatter still {spread*100:.1f}cm — TF NOT changed")
    rv = r0 + sol.x[:3]
    tf = T0.copy()
    tf[:3, :3] = Rotation.from_rotvec(rv).as_matrix()
    T_ee_cam = tf
    TABLE_Z = float(TABLE_Z + sol.x[3])
    tf_str = ",".join(f"{v:.4f}" for v in list(tf[:3, 3]) + list(rv))
    json.dump({"tf": tf_str, "table_z": TABLE_Z, "scatter_cm": spread * 100,
               "fitted": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(TF_FILE, "w"), indent=2)
    set_phase("CALIB", f"hand-eye fixed — cross-view scatter {spread*100:.1f}cm, "
                       f"table_z={TABLE_Z*100:.1f}cm")


def load_tf():
    global T_ee_cam, TABLE_Z
    try:
        with open(TF_FILE) as f:
            d = json.load(f)
        T_ee_cam = parse_tf_string(d["tf"])
        if "table_z" in d:
            TABLE_Z = float(d["table_z"])
        return d
    except Exception:
        return None


# ---------------- FPV overlay ----------------
def publish(rgb, joints=None):
    img = np.ascontiguousarray(rgb[:, :, ::-1])
    cv2.drawMarker(img, (int(HAND_UV[0]), int(HAND_UV[1])), (60, 200, 255),
                   cv2.MARKER_CROSS, 24, 2)
    if joints is not None and kin is not None and fx > 0:
        try:
            uv = tip_pixel(joints)
        except Exception:
            uv = None
        if uv is not None:
            gap = math.hypot(uv[0] - HAND_UV[0], uv[1] - HAND_UV[1])
            cv2.putText(img, f"hand-eye {gap:.0f}px", (10, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if gap < 40 else (255, 0, 255), 1)
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
    if joints is not None and fx > 0:
        try:
            T_cam = T_cam_of(joints)
        except Exception:
            T_cam = None
        if T_cam is not None:
            for o in world.snapshot():       # yellow = mapped object reprojected
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


def idle_view():
    while True:
        with lock:
            busy = state["running"]
        if not busy:
            try:
                observe()
            except Exception:
                time.sleep(0.5)
        time.sleep(0.2)


# ---------------- web ----------------
app = Flask(__name__)
PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RAX FPV SLAM</title><style>
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
<h1>RAX · FPV SLAM <span style="color:#8aa;font-weight:400">· known-pose mapping on the base plane → IK pick</span></h1>
<div><p class="lbl">gripper camera · yellow = mapped object reprojected</p><img src="/stream"></div>
<div><p class="lbl">top-down map (base frame) · click a row to pick</p>
  <canvas id="map" width="520" height="520"></canvas></div>
<div class="panel">
  <div class="phase" id="phase">—</div><div id="detail" style="color:#8aa;font-size:13px;min-height:18px"></div>
  <table id="tab"><thead><tr><th>tag</th><th>label</th><th>x cm</th><th>y cm</th><th>r</th><th>ang</th><th>n</th><th>reach</th><th></th></tr></thead><tbody></tbody></table>
  <div class="btns">
    <button id="b-scan" onclick="fetch('/scan',{method:'POST'})">Scan + Map</button>
    <button onclick="fetch('/probe',{method:'POST'})" title="ruler-check localization for the current view (no motion)">Probe</button>
    <button onclick="fetch('/calib',{method:'POST'})" title="one object in view; fits hand-eye rotation across the arc">Calib hand-eye</button>
    <button id="b-stop" onclick="fetch('/stop',{method:'POST'})">Stop</button>
    <button onclick="fetch('/home',{method:'POST'})">Home</button>
    <button onclick="fetch('/reset',{method:'POST'})">Reset pose</button>
    <button onclick="fetch('/clearmap',{method:'POST'})">Clear map</button>
    <input id="q" placeholder="red cube, green cube" style="flex:1;min-width:140px">
    <button onclick="setq()">Set query</button>
  </div>
  <pre id="log"></pre>
</div></main><script>
const cv=document.getElementById('map'),ctx=cv.getContext('2d');
let M={objs:[],reach:{rmin:10,rmax:42,arc:110}};
function W2S(x,y){const s=cv.width/1.0,ox=cv.width/2,oy=cv.height*0.82;return [ox-y*s,oy-x*s];}
function drawMap(){
  ctx.fillStyle='#0b0f14';ctx.fillRect(0,0,cv.width,cv.height);
  const s=cv.width/1.0,[ox,oy]=W2S(0,0);
  ctx.strokeStyle='#1d2a36';
  const a0=-M.reach.arc*Math.PI/180,a1=M.reach.arc*Math.PI/180;
  for(const rr of [M.reach.rmax,M.reach.rmin]){
    ctx.beginPath();ctx.arc(ox,oy,rr/100*s,-Math.PI/2-a1,-Math.PI/2-a0);ctx.stroke();}
  ctx.strokeStyle='#15202b';ctx.fillStyle='#456';ctx.font='10px monospace';
  for(let r=10;r<=40;r+=10){ctx.beginPath();ctx.arc(ox,oy,r/100*s,0,7);ctx.stroke();
    ctx.fillText(r+'cm',ox+2,oy-r/100*s+12);}
  ctx.fillStyle='#c9524a';ctx.beginPath();ctx.arc(ox,oy,5,0,7);ctx.fill();
  ctx.fillStyle='#8aa';ctx.fillText('base',ox+6,oy+4);
  for(const o of M.objs){
    const [sx,sy]=W2S(o.x,o.y);
    const c=/red/.test(o.label)?'#e2574c':/green/.test(o.label)?'#3fc46b':/blue/.test(o.label)?'#4a93c9':'#e0b040';
    ctx.fillStyle=o.reach?c:'#666';ctx.beginPath();ctx.arc(sx,sy,7,0,7);ctx.fill();
    ctx.fillStyle='#e6ecf1';ctx.font='11px monospace';
    ctx.fillText(`${o.label}#${o.tag}${o.frozen?'*':''}`,sx+9,sy+4);}
}
async function poll(){
  try{const s=await (await fetch('/status')).json();
    document.getElementById('phase').textContent=s.phase;
    document.getElementById('phase').className='phase'+(/ABORT|ERROR/.test(s.phase)?' bad':'');
    document.getElementById('detail').textContent=s.detail||'';
    document.getElementById('log').textContent=(s.log||[]).join('\\n');
    M.objs=s.map||[];drawMap();
    document.querySelector('#tab tbody').innerHTML=(s.map||[]).map(o=>
      `<tr class="obj" onclick="pick(${o.tag})"><td>${o.tag}</td><td>${o.label}</td><td>${(o.x*100).toFixed(1)}</td><td>${(o.y*100).toFixed(1)}</td><td>${o.r_cm}</td><td>${o.ang}°</td><td>${o.n}</td><td>${o.reach?'✓':'—'}</td><td><a href="#" onclick="event.stopPropagation();fetch('/look?tag=${o.tag}',{method:'POST'});return false" style="color:#4a93c9">look</a></td></tr>`
    ).join('')||'<tr><td colspan=9>empty — Scan + Map</td></tr>';
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
    threading.Thread(target=_t, daemon=True).start()
    return True


@app.route("/scan", methods=["POST"])
def r_scan():
    return jsonify(ok=_run(scan))


@app.route("/probe", methods=["POST"])
def r_probe():
    return jsonify(ok=_run(probe))


@app.route("/calib", methods=["POST"])
def r_calib():
    return jsonify(ok=_run(calibrate))


@app.route("/look", methods=["POST"])
def r_look():
    tag = int(request.args.get("tag", "0"))
    return jsonify(ok=_run(lambda: (look_at(tag), set_phase("IDLE", "look done"))))


@app.route("/pick", methods=["POST"])
def r_pick():
    tag = int(request.args.get("tag", "0"))
    dry = request.args.get("dry") in ("1", "true")
    return jsonify(ok=_run(pick, tag, dry))


@app.route("/stop", methods=["POST"])
def r_stop():
    stop_flag.set()
    say("STOP")
    return jsonify(ok=True)


@app.route("/clearmap", methods=["POST"])
def r_clearmap():
    global world
    world = World2D()
    say("map cleared")
    return jsonify(ok=True)


@app.route("/reset", methods=["POST"])
def r_reset():
    return jsonify(ok=_run(lambda: (goto(VIEW, settle=0.8),
                                    send_joints(observe()[0], gripper=95.0),
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
            robot.connect()
            break
        except (RuntimeError, ConnectionError) as e:
            if attempt == 5:
                raise
            say(f"connect glitch ({str(e)[:50]}) retry {attempt+1}/6")
            try:
                robot.bus.port_handler.closePort()
            except Exception:
                pass
            time.sleep(2.0)
    kin = RobotKinematics(LEROBOT + r"\SO101\so101_new_calib.urdf",
                          "gripper_frame_link", ARM_MOTORS)
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
        say(f"hand-eye: {d.get('fitted','?')}")
    try:
        uv = tip_pixel(np.array(HOME, np.float64))
        gap = math.hypot(uv[0] - HAND_UV[0], uv[1] - HAND_UV[1]) if uv else 999
        say(f"hand-eye check: fingertip {gap:.0f}px from HAND_UV"
            + ("  *** run Calib ***" if gap > 40 else " — OK"))
    except Exception:
        pass
    say("loading YOLO-World S…")
    detector = YoloWorldDetector(LEROBOT + r"\yolov8s-worldv2.pt", conf=0.08, imgsz=640,
                                 color_filter_min_frac=0.0)
    detector.set_query(", ".join(det_classes[0]))
    with lock:
        state["query"] = ", ".join(det_classes[0])
    threading.Thread(target=engine, daemon=True).start()
    threading.Thread(target=idle_view, daemon=True).start()
    set_phase("IDLE", "ready — Scan + Map")
    say(f"UI: http://{HOST_IP}:{PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
