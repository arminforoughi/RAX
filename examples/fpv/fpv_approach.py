# FPV map-based approach — first-person-view object mapping + approach + grasp.
#
# The camera is the head of the snake: it rides the gripper (~10 cm BEHIND the
# fingertips, pitched ~30 deg down at them), so every move changes the view and
# the detector gets WORSE the closer we get. The old approach conflated
# camera->object range with fingertip->object range and papered over the 10 cm
# gap with a radial fudge (PUSH_OUT). This server does the physics instead:
#
#   * Everything lives in the BASE frame. A detection becomes
#         p_base = (FK(q_at_capture) @ T_ee_cam) @ p_cam
#     and the 10 cm camera-behind-tip offset falls out of T_ee_cam naturally.
#   * Remaining travel is ALWAYS ||p_obj - fingertip(q)||. Camera range is used
#     only to decide whether to TRUST the detector and which range source to use.
#   * A persistent multi-object MAP (base frame, one EKF per object) is the
#     single source of truth. Base frame is bolted to the table and FK is exact
#     odometry, so this is SLAM with known poses = pure mapping. Objects persist
#     when they leave the view — that is the point.
#   * Approach = flying the fingertip through the map. The map refines while the
#     detector is honest (camera range in the trust band); inside COMMIT_M the
#     detector is IGNORED and the final leg runs open-loop on the frozen map
#     point — the object is static, so a remembered coordinate beats a
#     close-range hallucination.
import json, math, os, sys, tempfile, threading, time
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

sys.path.insert(0, r"C:\Users\labot\Documents\lerobot\src")

from lerobot.model.kinematics import RobotKinematics
from lerobot.perception.yolo_world import YoloWorldDetector
from lerobot.perception.bearing_ekf import (
    BearingRangeEKF, Intrinsics, ray_in_base, triangulate_midpoint)
from lerobot.manipulation.visual_servo.gaze_engine import ARM_MOTORS, parse_tf_string
from lerobot.manipulation.yolo_track.motion_primitives import send_joint_target_smoothly
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig

# The repo's SOLVED stereo->object primitives: back-project an object's depth
# region to a base-frame point cloud, then take its centroid. This replaces the
# bespoke bearing-EKF that reinvented (worse) what these already do.
from rax.perception.depth_cloud.backproject import backproject_masked, box_mask
from rax.models.depth.stereo import StereoIntrinsics

# The gripper servo (ID 6) replies too slowly for lerobot's handshake ping
# timeout. Skip the existence assert: sync WRITES need no ACK, and every read
# here already tolerates a miss.
from lerobot.motors.feetech.feetech import FeetechMotorsBus as _FTBus

_FTBus._handshake = lambda self: None

LEROBOT = r"C:\Users\labot\Documents\lerobot"
OUT = os.path.join(tempfile.gettempdir(), "rax_fpv_approach")
os.makedirs(OUT, exist_ok=True)

PORT = 8485                       # stack_mission2 owns 8484
HOST_IP = "100.110.89.78"         # Tailscale IP the remote browser reaches
RERUN_WEB_PORT = 9091             # 9090/9877 belong to stack_mission2
RERUN_GRPC_PORT = 9878

VIEW = np.array([-6.7, 37.1, 48.1, -40.4, -28.4])
HOME = np.array([-9.67, -102.022, 98.066, 32.879, 0.0])
HAND_UV = (440.0, 394.0)          # fingertips in the image, measured via /caltip

# gripper_frame_link IS the fingertip / grasp centre (measured from the URDF +
# jaw mesh: the jaws hinge 75-85 mm behind it, tips reach +7 mm past it). FK of
# that frame is the tip; do NOT add a 10 cm "tip offset" — that is the CAMERA's
# offset and it lives in T_ee_cam.
GRIP_TIP_OFFSET_M = 0.007

# ---- localization / map tunables ----
TRUST_NEAR = 0.15     # m camera range: closer than this the detector lies (looming/hallucinating)
TRUST_FAR = 0.70      # m: beyond this the object is ~15 px and the bearing is mush
STEREO_MIN_TRUST = 0.25   # extended-disparity stereo floor ~0.20 m + speckle margin
COMMIT_M = 0.13       # camera range below which the map entry FREEZES and the detector is ignored
UNFREEZE_M = 0.30     # camera range above which a frozen entry may refine again
SIGMA_LOCK = 0.008    # m EKF sigma to end SURVEY — under one finger-width
SIGMA_OK = 0.020      # m: accept at survey timeout if at least this good
SURVEY_MIN_OBS = 8
SURVEY_TIMEOUT_S = 14.0
DET_FRESH_S = 0.8     # 2x YOLO-L worst-case latency; older boxes are stale mid-approach
CONF_FUSE = 0.14      # fuse into the map above this (detector conf floor is 0.10 = draw-only)
MOTION_GATE_DPS = 30.0  # max joint speed at capture for a detection to be fused (blur)
ASSOC_PX = 80.0       # detection must reproject within this of its entry, else new instance
TABLE_Z0 = -0.02      # table surface in base frame (measured FK tip contact ≈ -0.022)
SIZE_DEFAULT = 0.030  # object edge until stereo has measured it

# ---- approach / motion tunables ----
STANDOFF_H = 0.05     # m fingertip parks this far ABOVE the object, then descends
PITCH_RAMP_M = 0.12   # blend gaze pitch -> grasp pitch over the last 12 cm of tip travel
APPR_KP = 1.2         # forward speed P-gain on remaining tip distance
V_MAX = 0.045         # m/s
V_MIN = 0.008         # m/s creep at arrival
LOOP_DT = 0.066       # ~15 Hz; _ik_hold_pitch worst case 16 ms leaves headroom
JOINT_RATE_MAX = 35.0 # deg/s per joint hard clamp
IK_RESID_MAX = 0.006  # m: above this the pose is unreachable — report, never drive
OBJ_FLOOR_Z = -0.020  # measured table contact (FK z with servo sag) is -0.022
GRASP_Z_MAX = 0.035   # every object rests on the table, so the GRASP height is at
                      # table level — cap the target z here. The map faithfully
                      # tracks the object's visible FACE centre (~7cm up on a big
                      # close cube from a shallow view), but grasping there is both
                      # unreachable (needs the wrist inside the base) and wrong; we
                      # grasp the table FOOTPRINT under it. (Uses the on-table premise.)
GAZE_V_TGT = 0.38 * 480.0  # aim the object at the UPPER-centre pixel row — the
                           # fingers own the frame bottom (HAND_UV v=394), so
                           # upper-centre keeps it in view longest while diving
GAZE_DEADBAND_PX = 18.0
GRASP_PITCH = (75.0, 80.0, 70.0, 85.0, 90.0, 65.0, 60.0, 55.0)  # steep first — see plan_grasp_pitch

state = {
    "phase": "IDLE", "detail": "", "joints": [], "gripper": None,
    "t0": None, "running": False, "loop_hz": 0.0, "dist_mm": None,
    "query": "", "pick_label": None, "map": {}, "dets": [],
}
log = deque(maxlen=140)
frame_jpeg = [None]
lock = threading.Lock()
bus_lock = threading.RLock()      # Feetech bus + OAK-D are not thread-safe
kin_lock = threading.Lock()       # NumericRobotKinematics mutates internal state on
                                  # every FK — the YOLO worker and the mission loop
                                  # both call it, so it MUST be serialized (this is
                                  # new vs stack_mission2, where only one thread FK'd)
map_lock = threading.Lock()
stop_flag = threading.Event()
mission_thread = [None]
latest_snap = [None]              # {"rgb","q","t","speed"} — joints PAIRED with frame
approach_active = [False]         # worker must not reseed a moving-object entry mid-approach
raw_dets = {}                     # label -> {"uv","bbox","conf","t","q"} freshest raw detection
DRY_DEFAULT = False


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
detector = None
T_ee_cam = parse_tf_string("-0.0390,-0.0290,-0.0043,1.7407,1.6823,1.8316")  # overridden by handeye_tf.json
fx = fy = cx0 = cy0 = 0.0
INTR = None                       # bearing_ekf.Intrinsics, built in main()
STEREO_INTR = None                # models.depth.StereoIntrinsics, built in main()


# ---------------- kinematics (all calls serialized) ----------------
def fk(q):
    with kin_lock:
        return np.asarray(kin.forward_kinematics(np.asarray(q, np.float64)))


def ik_pose(q_seed, T_goal, pw=1.0, ow=0.0):
    with kin_lock:
        return kin.inverse_kinematics(q_seed, T_goal, position_weight=pw, orientation_weight=ow)


def link_chain(q):
    with kin_lock:
        return kin.get_link_transforms_chain(q)


def T_cam_of(joints):
    return fk(joints) @ T_ee_cam


# ---------------- robot I/O ----------------
_prev_obs = [None]                # (t, joints) for the capture-motion gate


def observe(overlay=True):
    checkpoint()
    # Read ONLY the 5 arm motors on the critical path. robot.get_observation()
    # sync-reads all 6 IDs, and the flaky gripper (ID 6, marginal cable) drops
    # often enough that its miss fails the WHOLE group read ("no status packet")
    # and aborts the mission. The arm motors are healthy; read them alone, with
    # retries, so one bad gripper reply can't blind the arm.
    joints = None
    last_err = None
    for attempt in range(12):
        try:
            with bus_lock:
                pos = robot.bus.sync_read("Present_Position", ARM_MOTORS, num_retry=2)
            joints = np.array([float(pos[m]) for m in ARM_MOTORS])
            break
        except Exception as e:      # noqa: BLE001 — any bus hiccup, retry
            last_err = e
            time.sleep(0.06)
    if joints is None:
        raise last_err
    # camera frame + aligned depth + gripper position. Depth is paired WITH the
    # frame and joints so the worker back-projects the depth that matches the
    # image it detected on (the object is static; the pose that took the picture
    # is exact). Gripper read may fail (returns -1) without killing the observation.
    with bus_lock:
        rgb = np.asarray(cam.read_latest())
        try:
            depth_raw = cam.read_depth()          # (H,W) uint16 mm, aligned to RGB
        except Exception:
            depth_raw = None
        try:
            g = robot.bus.sync_read("Present_Position", ["gripper"], num_retry=0)
            gpos = float(g["gripper"])
        except Exception:
            gpos = -1.0
    depth_m = None
    if depth_raw is not None:
        depth_m = np.asarray(depth_raw, np.float32) / 1000.0   # -> metres
        depth_m[depth_m <= 0.05] = np.nan                      # invalid -> nan (dropped downstream)
    now = time.time()
    speed = 0.0
    if _prev_obs[0] is not None:
        dt = now - _prev_obs[0][0]
        if 1e-3 < dt < 1.0:
            speed = float(np.max(np.abs(joints - _prev_obs[0][1]))) / dt
    _prev_obs[0] = (now, joints.copy())
    with lock:
        state["joints"] = [round(float(v), 1) for v in joints]
        state["gripper"] = round(gpos, 1)
        # joints PAIRED with the frame: YOLO-L takes 200-400 ms, and a measurement
        # back-projected through the WRONG pose is exactly the smear that made the
        # old localization drift. The worker uses this snapshot's q, never "now".
        latest_snap[0] = {"rgb": rgb, "q": joints.copy(), "t": now, "speed": speed,
                          "depth": depth_m}
    obs = {"gripper.pos": gpos}      # compat shim for callers that want the grip
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
    if gp < 0:                       # gripper read dropped — hold it open, don't command -1%
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


# ---------------- camera geometry ----------------
def read_depth_m(uv, win=7):
    """Median stereo depth (metres) in a small window around pixel uv, or None.
    The OAK-D returns uint16 millimetres aligned to the RGB frame."""
    try:
        with bus_lock:
            depth = cam.read_depth()
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
    vals = patch[(patch > 80) & (patch < 2000)]   # 8cm-2m valid band
    if vals.size < 8:
        return None
    return float(np.median(vals)) / 1000.0


def project_base(p_base, T_base_cam):
    """Base-frame point -> pixel. The honest check on every localization."""
    pc = np.linalg.inv(np.asarray(T_base_cam, np.float64)) @ np.append(
        np.asarray(p_base, np.float64), 1.0)
    if pc[2] <= 1e-4:
        return None                      # behind the camera
    return (float(fx * pc[0] / pc[2] + cx0), float(fy * pc[1] / pc[2] + cy0))


def tip_pixel(joints):
    """Where the hand-eye TF SAYS the fingertip appears in the FPV. The camera is
    bolted to the gripper, so this is ONE fixed pixel — it must equal HAND_UV.
    The gap between them is the hand-eye error, in pixels, live."""
    T = fk(joints)
    return project_base(T[:3, 3], T @ T_ee_cam)


def table_grasp_point(p):
    """Project a map point down to its table FOOTPRINT for grasping. The object
    rests on the table (user's premise), so the graspable point is at table height
    below the observed face — not the face centre the camera happens to see."""
    p = np.asarray(p, np.float64)
    return np.array([p[0], p[1], float(np.clip(p[2], OBJ_FLOOR_Z, GRASP_Z_MAX))])


def cam_range_to(p_obj, T_base_cam):
    """Camera->object range along the optical axis (the z of p_obj in cam coords).
    Detector-TRUST gating only — never remaining travel."""
    pc = np.asarray(T_base_cam, np.float64)
    d = np.asarray(p_obj, np.float64) - pc[:3, 3]
    return float(pc[:3, 2] @ d)


def cam_pitch_of(T_base_cam):
    """Optical-axis angle below the horizon, degrees (positive = looking down)."""
    z = np.asarray(T_base_cam, np.float64)[:3, 2]
    return math.degrees(math.atan2(-z[2], math.hypot(z[0], z[1])))


def gaze_steps(p_obj, joints, max_pan=2.0, max_pitch=2.5):
    """(pan_target, pitch_target): one damped 2x2 Newton step walking p_obj's
    REPROJECTION toward the aim pixel (cx0, GAZE_V_TGT). Works with ZERO
    detections (map only).

    The pixel Jacobian is computed NUMERICALLY through the true FK + hand-eye
    each call. Do not hand-model it: the mount can carry roll/yaw offsets that
    scale, cross-couple, or even FLIP the raw-pixel signs (measured slope -0.43
    at the VIEW pose under the fitted TF), and a wrong-sign gaze diverges."""
    pan_now = float(joints[0])
    pitch_now = float(joints[1] + joints[2] + joints[3])
    q = np.asarray(joints, np.float64)
    try:
        uv0 = project_base(p_obj, T_cam_of(q))
    except Exception:
        return pan_now, pitch_now
    if uv0 is None:
        return pan_now, pitch_now
    J = np.zeros((2, 2))
    for col, ji in enumerate((0, 3)):   # pan probe; wrist probe = pitch 1:1 (parallel axes)
        q2 = q.copy()
        q2[ji] = float(np.clip(q2[ji] + 3.0, J_LO[ji], J_HI[ji]))
        d = float(q2[ji] - q[ji])
        if abs(d) < 0.5:                # against the limit: probe the other way
            q2[ji] = float(np.clip(q[ji] - 3.0, J_LO[ji], J_HI[ji]))
            d = float(q2[ji] - q[ji])
        try:
            uv1 = project_base(p_obj, T_cam_of(q2))
        except Exception:
            uv1 = None
        if uv1 is None or abs(d) < 1e-6:
            return pan_now, pitch_now
        J[0, col] = (uv1[0] - uv0[0]) / d
        J[1, col] = (uv1[1] - uv0[1]) / d
    err = np.array([cx0 - uv0[0], GAZE_V_TGT - uv0[1]])
    try:
        step = np.linalg.solve(J.T @ J + 4.0 * np.eye(2), J.T @ err)
    except np.linalg.LinAlgError:
        return pan_now, pitch_now
    return (pan_now + float(np.clip(step[0], -max_pan, max_pan)),
            pitch_now + float(np.clip(step[1], -max_pitch, max_pitch)))


# ---------------- hand-eye (shared handeye_tf.json with stack_mission2) ----------------
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


def wait_raw_det(label, after_t, timeout=3.0):
    """Block until the YOLO worker reports a detection of `label` captured AFTER
    after_t (i.e. from the current pose), keeping frames flowing meanwhile."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        checkpoint()
        observe()
        with lock:
            d = raw_dets.get(label)
            d = dict(d) if d else None
        if d and d["t"] > after_t:
            return d
        time.sleep(0.1)
    return None


def calibrate_handeye(label, n_target=14):
    """Fit the gripper->camera transform FROM THE ROBOT'S OWN MOTION (~30 s).

    Two independent facts pin the TF down:
      (A) THE FINGERTIP IS IN THE PICTURE: project(FK_tip) must equal HAND_UV.
          Nails the camera's POINTING DIRECTION.
      (B) A STATIC OBJECT LOOKS THE SAME FROM EVERYWHERE: N poses that keep the
          object in view give sightlines that must pass through ONE point.
          Nails the camera's POSITION on the gripper.
    Unknowns: TF translation (3) + rotvec (3) + the object (3). Seeded from the
    current TF, so a good TF stays put. Same fit as stack_mission2's /calib —
    both scripts share handeye_tf.json."""
    global T_ee_cam
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    set_phase("CALIB", f"hand-eye: sampling '{label}' from several poses")
    q0 = observe()[0].astype(np.float64)
    d0 = wait_raw_det(label, time.time() - 1.0, timeout=4.0)
    if d0 is None:
        raise Abort(f"hand-eye: no '{label}' in view — put it in the gripper view first")

    deltas = []
    for dpan in (-9.0, -4.5, 0.0, 4.5, 9.0):
        deltas.append(np.array([dpan, 0.0, 0.0, 0.0, 0.0]))
    for dwf in (-9.0, -4.0, 4.0, 9.0):
        deltas.append(np.array([0.0, 0.0, 0.0, dwf, 0.0]))
    for dl, de in ((-6.0, 6.0), (6.0, -6.0), (-4.0, 10.0), (4.0, -10.0), (-8.0, 4.0)):
        deltas.append(np.array([0.0, dl, de, 0.0, 0.0]))

    samples = []          # (joints_at_capture, uv) — the worker pairs them for us
    for dq in deltas:
        checkpoint()
        goto_smooth(q0 + dq, settle=0.3, step=1.5)
        t_mark = time.time()
        d = wait_raw_det(label, t_mark, timeout=2.0)
        if d is None:
            continue
        samples.append((np.asarray(d["q"], np.float64), np.array(d["uv"], np.float64)))
        if len(samples) >= n_target:
            break
    goto_smooth(q0, settle=0.3, step=1.5)

    if len(samples) < 6:
        raise Abort(f"hand-eye: only {len(samples)} usable views (need 6) — "
                    "keep the object in the gripper view for the whole sweep")

    T_ee = [fk(j) for j, _ in samples]
    uvs = [uv for _, uv in samples]
    w_tip = math.sqrt(len(samples))   # the fingertip anchor is ONE constraint; weight ~ sqrt(N)

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
        pt = project_base(T_ee[0][:3, 3], T_ee[0] @ tf)
        r += ([400.0, 400.0] if pt is None else
              [w_tip * (pt[0] - HAND_UV[0]), w_tip * (pt[1] - HAND_UV[1])])
        return r

    o, dirv = ray_in_base(T_ee[0] @ T_ee_cam, tuple(uvs[0]), INTR)
    p_seed = np.array([0.18, 0.0, 0.02])
    if dirv[2] < -1e-6:
        t = (0.015 - o[2]) / dirv[2]
        if t > 0:
            p_seed = o + t * dirv

    x0 = np.concatenate([np.asarray(T_ee_cam[:3, 3], np.float64),
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

    say(f"hand-eye BEFORE: reprojection RMS={rms(x0):.0f}px  "
        f"fingertip off by {tipgap(x0):.0f}px  ({len(samples)} views)")
    sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac",
                        max_nfev=4000, ftol=1e-10, xtol=1e-10)
    tf, p = unpack(sol.x)
    say(f"hand-eye AFTER:  reprojection RMS={rms(sol.x):.0f}px  "
        f"fingertip off by {tipgap(sol.x):.0f}px")
    # 35px (was 25): a big cube's bbox centre wanders a few px as its apparent
    # shape changes across poses, so a large-target fit floors a little higher
    # than a chessboard would. 35px still rejects a bad fit but accepts a good
    # one — at 25cm range 35px is ~1.7cm/view, averaged much tighter by the EKF.
    if rms(sol.x) > 35.0 or tipgap(sol.x) > 40.0:
        raise Abort(f"hand-eye: fit did not converge (RMS {rms(sol.x):.0f}px, "
                    f"tip {tipgap(sol.x):.0f}px) — TF NOT changed.")

    rv = Rotation.from_matrix(tf[:3, :3]).as_rotvec()
    tf_str = ",".join(f"{v:.4f}" for v in list(tf[:3, 3]) + list(rv))
    with open(TF_FILE, "w") as f:
        json.dump({"tf": tf_str, "rms_px": rms(sol.x), "tip_px": tipgap(sol.x),
                   "views": len(samples), "fitted": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    T_ee_cam = tf
    say(f"hand-eye CALIBRATED -> {tf_str}  (saved to {os.path.basename(TF_FILE)})")
    set_phase("CALIB", f"hand-eye fixed — reprojection {rms(sol.x):.0f}px")
    return tf


# ---------------- the MAP (stereo point-cloud centroid) ----------------
# Metric object mapping from a stereo camera is a SOLVED problem, so we use the
# repo's own primitives instead of a bespoke filter: back-project the object's
# depth pixels to a 3D point cloud (perception.depth_cloud.backproject), move it
# to the base frame with FK x hand-eye, and take the outlier-robust CENTROID
# (perception.depth_cloud.object_cloud). No triangulation, no table-plane fudge,
# no per-object EKF, no bbox-centre single-pixel depth (that read the cube's near
# FACE and localized it too high). NOTE: the OAK-D is stereo-blind below ~20cm,
# so mapping happens from a survey vantage (~25-45cm); the entry is then FROZEN
# and the approach runs on the remembered centroid.
MIN_CLOUD_PTS = 40        # valid depth pixels needed to trust a centroid
CLOUD_JUMP_M = 0.15       # a centroid this far from the estimate is a bad frame
                          # (background bleed / partial occlusion) — skip it


def _robust_centroid(pts):
    """Median-centred, MAD-gated centroid of an Nx3 cloud — drops the background
    and box-edge outliers a plain mean would chase."""
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    mad = float(np.median(np.abs(d - np.median(d)))) + 1e-6
    keep = d < (float(np.median(d)) + 3.0 * mad)
    inl = pts[keep] if int(keep.sum()) >= 8 else pts
    return inl.mean(axis=0), inl


class MapEntry:
    def __init__(self, label):
        self.label = label
        self.pos = None                              # base-frame centroid = object position
        self.cloud = np.empty((0, 3), np.float64)    # last inlier cloud (for the 3D viz)
        self.size_m = None                           # cloud horizontal extent ~ object edge
        self.n_obs = 0
        self.created_t = time.time()
        self.last_seen = 0.0
        self.last_uv = None
        self.last_bbox = None
        self.range_cam_m = float("nan")              # median camera->object range (vantage gate)
        self.frozen = False
        self.quality = "none"
        self._sigma = float("inf")                   # standard error of the centroid
        self._jump = 0

    @property
    def initialized(self):
        return self.pos is not None

    @property
    def size(self):
        s = self.size_m if self.size_m else SIZE_DEFAULT
        return float(np.clip(s, 0.02, 0.10))

    def position(self):
        return None if self.pos is None else self.pos.copy()

    def sigma(self):
        return self._sigma

    def reset(self):
        self.pos = None
        self.cloud = np.empty((0, 3), np.float64)
        self.size_m = None
        self.n_obs = 0
        self._sigma = float("inf")
        self._jump = 0

    def reproject(self, T_base_cam):
        """Centroid -> pixel in a view (the honest localization check)."""
        if self.pos is None:
            return None
        return project_base(self.pos, T_base_cam)


class ObjectMap:
    """Persistent base-frame object map. One cloud-centroid entry per label;
    entries never expire on lost sight. All fusion on the detector thread."""

    def __init__(self):
        self.entries = {}   # label -> MapEntry, guarded by map_lock

    def get(self, label):
        with map_lock:
            return self.entries.get(label)

    def entry(self, label):
        with map_lock:
            e = self.entries.get(label)
            if e is None:
                e = self.entries[label] = MapEntry(label)
            return e

    def remove(self, label):
        with map_lock:
            self.entries.pop(label, None)

    def summary(self):
        out = {}
        now = time.time()
        with map_lock:
            for lbl, e in self.entries.items():
                p = e.position()
                out[lbl] = {
                    "p": None if p is None else [round(float(v), 3) for v in p],
                    "r_cm": None if p is None else round(float(np.hypot(p[0], p[1])) * 100, 1),
                    "ang_deg": None if p is None else round(math.degrees(math.atan2(p[1], p[0])), 1),
                    "z_cm": None if p is None else round(float(p[2]) * 100, 1),
                    "sigma_mm": None if not e.initialized else round(min(e.sigma(), 9.99) * 1e3, 1),
                    "n_obs": e.n_obs, "age_s": round(now - e.last_seen, 1) if e.last_seen else None,
                    "quality": e.quality, "frozen": e.frozen,
                    "size_cm": round(e.size * 100, 1) if e.size_m else None,
                    "range_cm": round(e.range_cam_m * 100, 1) if np.isfinite(e.range_cam_m) else None,
                }
        return out

    # -- fusion (detector thread only) ----------------------------------
    def observe_detection(self, label, uv, bbox, conf, T_cam, speed, depth_m, shape):
        H, W = shape[:2]
        now = time.time()
        with map_lock:
            e = self.entries.get(label)
            if e is None:
                e = self.entries[label] = MapEntry(label)
            e.last_seen = now
            e.last_uv = tuple(uv)
            e.last_bbox = tuple(bbox)
            frozen = e.frozen

        # trust gates: any fail => display-only, never fused
        if conf < CONF_FUSE or frozen or depth_m is None:
            return
        x1, y1, x2, y2 = bbox
        if x1 <= 2 or y1 <= 2 or x2 >= W - 3 or y2 >= H - 3:
            return                          # clipped box: partial object, biased cloud
        if speed > MOTION_GATE_DPS:
            return                          # motion blur / stale depth-frame pairing

        # back-project the object's depth region to a base-frame cloud, robustly
        # (box shrunk 15% to keep table/background off the edges).
        mask = box_mask((H, W), (x1, y1, x2, y2), inset=0.15)
        cloud = backproject_masked(depth_m, mask, STEREO_INTR, T_cam,
                                   z_min_m=0.05, z_max_m=1.5, max_points=3000)
        if cloud.shape[0] < MIN_CLOUD_PTS:
            return                          # too close (stereo-blind) or no valid depth here
        c, inl = _robust_centroid(cloud)
        rng = float(np.median(np.linalg.norm(inl - T_cam[:3, 3], axis=1)))
        ext = inl[:, :2]
        size = float(np.linalg.norm(ext.max(0) - ext.min(0)) / math.sqrt(2.0))
        std = float(np.sqrt(np.mean(np.sum((inl - inl.mean(0)) ** 2, axis=1))))

        # DEPTH from the table, BEARING from the cloud. The cloud centroid gives a
        # well-calibrated DIRECTION to the object and its size, but this OAK-D's
        # range is biased at close range (it put the on-table cube at z=17cm). The
        # objects rest on the table (base plane), so intersect the camera->centroid
        # ray with z = TABLE_Z0 + size/2 to recover the true on-table position
        # along the correct sightline. Falls back to the raw centroid if the
        # sightline is too shallow to hit the plane cleanly.
        o = np.asarray(T_cam[:3, 3], np.float64)
        dirv = c - o
        nd = float(np.linalg.norm(dirv))
        if nd > 1e-6:
            dirv /= nd
            if dirv[2] < -0.08:
                t = (TABLE_Z0 + 0.5 * size - o[2]) / dirv[2]
                if 0.05 < t < 1.0:
                    c = o + t * dirv

        with map_lock:
            if e.frozen:
                return
            if e.pos is None:
                e.pos = c
            else:
                if float(np.linalg.norm(c - e.pos)) > CLOUD_JUMP_M and e.n_obs > 3:
                    # a big jump = background bleed / occlusion / it moved. Skip a
                    # few, then follow (a genuinely moved object should win).
                    e._jump += 1
                    if e._jump < 4:
                        return
                    e._jump = 0
                    e.pos = c
                else:
                    e._jump = 0
                    e.pos = 0.6 * e.pos + 0.4 * c
            e.cloud = inl
            e.range_cam_m = rng
            e.size_m = size if e.size_m is None else 0.8 * e.size_m + 0.2 * size
            # standard error of the centroid = cloud std / sqrt(N): shrinks as the
            # cloud is tight and well-populated. Survey locks when this is small.
            e._sigma = std / math.sqrt(max(1, inl.shape[0]))
            e.n_obs += 1
            e.quality = "cloud"


obj_map = ObjectMap()


# ---------------- detector worker (all fusion happens here) ----------------
pending_query = [None]
det_classes = [["red cube", "green cube"]]

# HSV color-blob fallback. On THIS rig the YOLO-World checkpoints are prompt-
# sensitive (the L model returns 0 dets on a clear green cube; the S model is
# better but still flickers), yet the cubes are vivid and segment trivially. So
# every color-named label ALSO gets a saturation-gated HSV blob — the proven
# acquisition path from stack_mission2 — and whichever gives a box feeds the map.
# The map's own trust gates + EKF still decide what actually fuses.
HSV_BANDS = {
    "red": [((0, 110, 80), (9, 255, 255)), ((170, 110, 80), (179, 255, 255))],
    "green": [((38, 80, 60), (85, 255, 255))],
    "blue": [((95, 90, 60), (130, 255, 255))],
    "yellow": [((20, 90, 80), (34, 255, 255))],
}
HSV_MIN_AREA = 900


def label_color(label):
    for c in HSV_BANDS:
        if c in label.lower():
            return c
    return None


def color_blob(rgb, color):
    """Largest saturation-gated blob of `color` -> (uv, bbox_xyxy) or None."""
    bands = HSV_BANDS.get(color)
    if bands is None:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, np.uint8), cv2.COLOR_RGB2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in bands:
        m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _l, stats, _c = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < HSV_MIN_AREA:
            continue
        if best is None or a > best[0]:
            x, y, w, h = (int(stats[i, j]) for j in (
                cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            best = (a, (x, y, x + w, y + h))
    if best is None:
        return None
    x1, y1, x2, y2 = best[1]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0), (x1, y1, x2, y2)


def yolo_worker():
    last_t = 0.0
    while True:
        time.sleep(0.05)
        # the model must only be mutated on the thread that runs it
        if pending_query[0] is not None and detector is not None:
            q = pending_query[0]
            pending_query[0] = None
            try:
                detector.set_query(q)
                det_classes[0] = [p.strip() for p in q.split(",") if p.strip()] or [q]
                with lock:
                    state["query"] = q
                say(f"detection query set: {q}  (each comma term = a distinct map object)")
            except Exception as e:
                say(f"query change failed: {e}")
        with lock:
            snap = latest_snap[0]
        if snap is None or detector is None or snap["t"] == last_t:
            continue
        last_t = snap["t"]
        try:
            dets = detector.predict_rgb(np.ascontiguousarray(snap["rgb"]))
        except Exception:
            dets = []
        T_cam = T_cam_of(snap["q"])          # pose AT CAPTURE, not now
        classes = det_classes[0]
        rgb = snap["rgb"]

        # best YOLO box per label (highest conf wins)
        best = {}                            # label -> (uv, bbox, conf, src)
        for d in dets:
            if not (0 <= d.class_id < len(classes)):
                continue
            label = classes[d.class_id]
            x1, y1, x2, y2 = d.xyxy
            uv = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            if label not in best or d.confidence > best[label][2]:
                best[label] = (uv, tuple(d.xyxy), float(d.confidence), "yolo")

        # HSV fallback: any color-named label WITHOUT a trustworthy YOLO box gets
        # its saturation-gated blob (conf 0.5 — a real, if colour-only, detection).
        for label in classes:
            if best.get(label) and best[label][2] >= CONF_FUSE:
                continue
            col = label_color(label)
            if col is None:
                continue
            cb = color_blob(rgb, col)
            if cb is not None:
                best[label] = (cb[0], cb[1], 0.5, "hsv")

        ui = []
        for label, (uv, bbox, conf, src) in best.items():
            ui.append({"label": label, "bbox": [float(v) for v in bbox],
                       "conf": round(conf, 2), "t": snap["t"], "src": src})
            with lock:
                raw_dets[label] = {"uv": uv, "bbox": tuple(bbox), "conf": conf,
                                   "t": snap["t"], "src": src,
                                   "q": np.asarray(snap["q"], np.float64).copy()}
            try:
                obj_map.observe_detection(label, uv, bbox, conf, T_cam,
                                          snap["speed"], snap.get("depth"), rgb.shape)
            except Exception as e:
                say(f"map fusion error ({label}): {type(e).__name__}: {e}")
        with lock:
            state["dets"] = ui
            state["map"] = obj_map.summary()


def raw_det(label, max_age=DET_FRESH_S):
    with lock:
        d = raw_dets.get(label)
        d = dict(d) if d else None
    if d and time.time() - d["t"] < max_age:
        return d
    return None


# ---------------- IK: pitch-slaved position solver (proven, from stack_mission2) ----------------
# THE REAL JOINT LIMITS from so101_new_calib.urdf. An IK that does not know these
# returns poses the servos silently clamp — half-solved poses the arm cannot hold.
J_LO = np.array([-110.0, -100.0, -96.8, -95.0, -157.2])
J_HI = np.array([+110.0, +100.0, +96.8, +95.0, +162.8])
WFLEX_MIN, WFLEX_MAX = float(J_LO[3]), float(J_HI[3])


def _gripper_pitch(T):
    z = T[:3, 2]
    return math.degrees(math.atan2(-z[2], math.hypot(z[0], z[1])))


def _slave_wflex(j1, j2, pitch_tgt):
    # the SO-101 pitch joints (lift, elbow, wrist_flex) are PARALLEL, so gripper
    # world pitch = j1+j2+j3 exactly => wrist_flex is algebraic, no IK drift.
    return float(np.clip(pitch_tgt - j1 - j2, WFLEX_MIN, WFLEX_MAX))


def _ik_hold_pitch(q_seed, p_tgt, pitch_tgt, j5_fixed, iters=80, tol=2e-3,
                   ret_err=False, _retry=True):
    """Position IK on servos 1-3 (pan, lift, elbow) with wrist_flex ALGEBRAICALLY
    SLAVED to hold the gripper pitch and wrist_roll fixed. Limit-clamped every
    iteration; scores the CLAMPED pose; RETURNS THE RESIDUAL — callers must check
    it (an unreachable target is a fact to report, not a pose to drive to)."""
    q = np.array(q_seed, dtype=np.float64)
    q[4] = float(np.clip(j5_fixed, J_LO[4], J_HI[4]))
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
        e = max(e, 0.05)          # pitch could not be held here: treat as unreachable
    if e > tol and _retry:
        # wrong IK branch — genuine elbow-flip dead band at r=10-15cm. Re-seed.
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


def plan_grasp_pitch(p_obj, q_seed):
    """Choose the gripper pitch to grasp with, and PROVE the arm can get there.
    A close object can ONLY be grasped nearly straight down: the fingertip is
    98 mm in front of the wrist, so a horizontal hand at r=15cm puts the wrist
    inside the base column. Steep-first, IK-residual veto."""
    p_above = np.array([p_obj[0], p_obj[1], p_obj[2] + STANDOFF_H])
    best = None
    for pitch in GRASP_PITCH:
        _, e_hi = _ik_hold_pitch(q_seed, p_above, pitch, float(q_seed[4]), ret_err=True)
        _, e_lo = _ik_hold_pitch(q_seed, np.asarray(p_obj, np.float64), pitch,
                                 float(q_seed[4]), ret_err=True)
        worst = max(float(e_hi), float(e_lo))
        if worst < 0.004:
            return pitch, worst
        if best is None or worst < best[1]:
            best = (pitch, worst)
    return None, best[1]


# ---------------- gaze + search + survey ----------------
def entry_uv(label, joints):
    """The pixel to gaze at: the fresh detection when the detector is honest,
    the MAP REPROJECTION when it is not — the gaze never starves."""
    d = raw_det(label)
    if d is not None:
        return d["uv"], True
    e = obj_map.get(label)
    if e is None:
        return None, False
    with map_lock:
        if not e.initialized:
            return None, False
        uv_r = e.reproject(T_cam_of(joints))
    return (uv_r, False) if uv_r is not None else (None, False)


def scan_for(label, span_deg=70.0):
    """Slow continuous base pan that STOPS the instant the worker reports the
    label. The worker fuses EVERY label it sees during the sweep (~13 deg/s
    passes the motion gate), so searching for one object maps the others free."""
    set_phase(f"SEARCH {label}", "slow scan — stops the moment it is seen")
    start_pan = float(observe()[0][0])
    for sign in (1.0, -1.0, -1.0, 1.0):
        target = float(np.clip(start_pan + sign * span_deg, J_LO[0], J_HI[0]))
        for _ in range(500):
            checkpoint()
            joints, _rgb, _ = observe()
            if raw_det(label) is not None:
                say(f"'{label}' spotted")
                return True
            cur = float(joints[0])
            if abs(cur - target) < 2.0:
                break
            q = joints.copy()
            q[0] += float(np.clip(target - cur, -0.9, 0.9))   # ~13 deg/s
            send_joints(q)
            time.sleep(LOOP_DT)
    return False


def scan_map(span_deg=70.0):
    """One full sweep with NO early exit — the worker builds the map as the view
    pans across the table."""
    set_phase("SEARCH", "mapping sweep — the worker fuses everything it sees")
    start_pan = float(observe()[0][0])
    for target in (start_pan + span_deg, start_pan - span_deg, start_pan):
        target = float(np.clip(target, J_LO[0], J_HI[0]))
        for _ in range(600):
            checkpoint()
            joints, _rgb, _ = observe()
            cur = float(joints[0])
            if abs(cur - target) < 2.0:
                break
            q = joints.copy()
            q[0] += float(np.clip(target - cur, -0.9, 0.9))
            send_joints(q)
            time.sleep(LOOP_DT)


def aim_at(label):
    """Coarse pan toward the map entry / fresh detection so SURVEY starts with
    the object in frame."""
    joints = observe()[0]
    e = obj_map.get(label)
    p = e.position() if e is not None else None
    if p is None:
        return
    T_cam = T_cam_of(joints)
    z = T_cam[:3, 2]
    az_cam = math.atan2(z[1], z[0])
    d = p - T_cam[:3, 3]
    az_obj = math.atan2(d[1], d[0])
    dpan = math.degrees(math.atan2(math.sin(az_obj - az_cam), math.cos(az_obj - az_cam)))
    if abs(dpan) < 3.0:
        return
    tgt = float(np.clip(joints[0] + dpan, J_LO[0], J_HI[0]))
    goto_smooth(np.array([tgt, *joints[1:]]), settle=0.3, step=1.2)


def survey(label):
    """Hold a good vantage and pump measurements into the map until the entry is
    tight. Gaze = pixel-P on pan + wrist_flex (rotation only). If stereo hasn't
    initialized the entry, small LATERAL dollies manufacture parallax — the EKF
    turns the arm's own motion into metric range."""
    set_phase(f"SURVEY {label}", "localizing — table-solve from a downward view")
    e = obj_map.entry(label)
    # a stale/garbage anchor (huge sigma, or seeded from a shallow view) only
    # fights the good downward-view measurements — drop it and re-seed clean.
    with map_lock:
        if e.initialized and e.sigma() > 0.10:
            e.reset()
    t0 = time.time()
    misses = 0
    last_dolly = t0
    dolly_sign = 1.0
    while time.time() - t0 < SURVEY_TIMEOUT_S:
        t_tick = time.time()
        checkpoint()
        joints, _rgb, _ = observe()
        uv, fresh = entry_uv(label, joints)
        if uv is None:
            misses += 1
            if misses > 40:
                raise Abort(f"{label}: lost during survey")
            time.sleep(LOOP_DT)
            continue
        misses = 0

        with map_lock:
            inited = e.initialized
            sig = e.sigma()
            n = e.n_obs
            p = e.position()
        if inited and sig < SIGMA_LOCK and n >= SURVEY_MIN_OBS:
            say(f"{label}: locked @ r={np.hypot(p[0], p[1])*100:.1f}cm "
                f"z={p[2]*100:.1f}cm sigma={sig*1e3:.1f}mm n={n} [{e.quality}]")
            return

        # gaze: rotate only (pan + wrist), object to the upper-centre pixel.
        # Once the entry is initialized, BOTH axes go through the numeric
        # reprojection derivative — the raw-pixel signs are properties of the
        # MOUNT, not the arm, and a refit hand-eye TF can flip them. Before
        # gaze on the REAL cube via the mount-correct numeric Jacobian. When a
        # fresh detection exists we servo a point on ITS bearing (so gaze follows
        # the detector, not a still-converging map point); otherwise we servo the
        # map reprojection. gaze_steps drives that point toward (cx0, GAZE_V_TGT).
        du, dv = uv[0] - cx0, uv[1] - GAZE_V_TGT
        q = joints.copy()
        T_now = T_cam_of(joints)
        if fresh:
            rng0 = 0.30
            if p is not None:
                rng0 = float(np.clip(cam_range_to(p, T_now), 0.12, 0.5))
            o, dv3 = ray_in_base(T_now, tuple(uv), INTR)
            p_gaze = o + dv3 * rng0
        else:
            p_gaze = p
        if p_gaze is not None and (abs(du) > GAZE_DEADBAND_PX or abs(dv) > GAZE_DEADBAND_PX):
            pan_t, pitch_t = gaze_steps(p_gaze, joints, max_pan=2.0, max_pitch=2.0)
            q[0] = float(np.clip(pan_t, J_LO[0], J_HI[0]))
            q[3] = float(np.clip(_slave_wflex(q[1], q[2], pitch_t), J_LO[3], J_HI[3]))
            send_joints(q)

        # vantage keeping: the OAK-D is stereo-BLIND below ~20cm, so the cloud
        # map only forms from a stand-off. Two ways to be too close:
        #   (a) we HAVE a range and it's short, or
        #   (b) no cloud has formed at all after a couple seconds (the object is
        #       likely inside the blind zone — that's WHY there's no cloud).
        # In both cases pull the fingertip radially IN toward the base, which
        # backs the head-mounted camera away from an object ahead of it.
        with map_lock:
            rng_meas = e.range_cam_m
            inited2 = e.initialized
        too_close = (np.isfinite(rng_meas) and rng_meas < STEREO_MIN_TRUST + 0.03) \
            or (not inited2 and time.time() - t0 > 2.5)
        if too_close and time.time() - last_dolly > 2.0:
            tip = fk(joints)[:3, 3]
            rvec = tip[:2]
            rn = float(np.linalg.norm(rvec))
            if rn > 1e-3:
                move = np.array([-rvec[0] / rn, -rvec[1] / rn, 0.0]) * 0.03   # reach IN 3cm
                pitch_now = float(joints[1] + joints[2] + joints[3])
                q_t, err = _ik_hold_pitch(joints, tip + move, pitch_now,
                                          float(joints[4]), ret_err=True)
                if err < IK_RESID_MAX:
                    rtxt = f"{rng_meas*100:.0f}cm" if np.isfinite(rng_meas) else "stereo-blind"
                    say(f"{label}: too close ({rtxt}) — backing the camera off to map")
                    goto_smooth(q_t, settle=0.3, step=1.2)
                last_dolly = time.time()
        time.sleep(max(0.0, LOOP_DT - (time.time() - t_tick)))

    with map_lock:
        sig = e.sigma() if e.initialized else float("inf")
    if sig < SIGMA_OK:
        say(f"{label}: survey timeout, accepting sigma={sig*1e3:.1f}mm")
        return
    raise Abort(f"{label}: could not localize (sigma {sig*1e3:.0f}mm after "
                f"{SURVEY_TIMEOUT_S:.0f}s)")


# ---------------- approach + grasp ----------------
def approach(label):
    """Fly the FINGERTIP down a straight base-frame line to the standoff point
    above the object, camera gazing at it the whole way (geometric pitch, so it
    works even with zero detections). The map refines in the worker while the
    detector is honest; at COMMIT_M camera range the entry FREEZES and the rest
    is pure memory."""
    e = obj_map.get(label)
    p_obj = e.position() if e is not None else None
    if p_obj is None:
        raise Abort(f"{label}: not in the map")
    set_phase(f"APPROACH {label}", "flying the fingertip through the map")
    send_joints(observe()[0], gripper=95.0)   # fingers open early
    time.sleep(0.3)

    q0 = observe()[0].astype(np.float64)
    p_grasp = table_grasp_point(p_obj)          # project to the table footprint
    pitch_hint, e_ik = plan_grasp_pitch(p_grasp, q0)
    if pitch_hint is None:
        raise Abort(f"{label}: unreachable at ANY grasp pitch (best IK residual "
                    f"{e_ik*1e3:.0f}mm, r={np.hypot(p_grasp[0], p_grasp[1])*100:.0f}cm "
                    f"z={p_grasp[2]*100:.0f}cm)")
    say(f"{label}: grasp pitch {pitch_hint:.0f}deg down (IK residual {e_ik*1e3:.1f}mm) "
        f"at r={np.hypot(p_grasp[0], p_grasp[1])*100:.0f}cm z={p_grasp[2]*100:.0f}cm")

    approach_active[0] = True
    try:
        q_cmd = None
        hist = []
        stall_cool = 0.0
        t_end = time.time() + 90.0
        n, t_hz = 0, time.time()
        while time.time() < t_end:
            t0 = time.time()
            joints, _rgb, _ = observe()
            if q_cmd is None or float(np.max(np.abs(q_cmd - joints))) > 8.0:
                q_cmd = joints.copy().astype(np.float64)

            with map_lock:
                p_obj = e.position()
                frozen = e.frozen
            p_grasp = table_grasp_point(p_obj)          # grasp the table footprint
            p_stand = np.array([p_grasp[0], p_grasp[1], p_grasp[2] + STANDOFF_H])
            T_now = fk(joints)
            tip = T_now[:3, 3]
            d_stand = float(np.linalg.norm(p_stand - tip))
            d_obj = float(np.linalg.norm(p_grasp - tip))
            rng = cam_range_to(p_obj, T_now @ T_ee_cam)   # camera range to the SEEN point
            with lock:
                state["dist_mm"] = round(d_obj * 1e3)
                state["obj3d"] = [float(v) for v in p_obj]
                state["obj3d_label"] = label

            if rng < COMMIT_M and not frozen:
                with map_lock:
                    e.frozen = True
                say(f"{label}: camera range {rng*100:.0f}cm < {COMMIT_M*100:.0f}cm — "
                    f"map FROZEN, detector ignored from here (memory beats a "
                    f"close-range hallucination)")
            if d_stand <= 0.012:
                say(f"{label}: at the standoff, {STANDOFF_H*100:.0f}cm above the object "
                    f"(tip->grasp {d_obj*1e3:.0f}mm)")
                return p_grasp.copy(), pitch_hint

            # stall: commanded but not moving. On target = arrival; else unjam up.
            now = time.time()
            hist.append((now, joints.copy()))
            while hist and now - hist[0][0] > 1.6:
                hist.pop(0)
            if (now > stall_cool and now - hist[0][0] > 1.3
                    and float(np.max(np.abs(joints - hist[0][1]))) < 0.25):
                if d_obj < 0.03:
                    say(f"{label}: stalled ON the object — treating as arrival")
                    return p_grasp.copy(), pitch_hint
                say(f"{label}: blocked — retreating 12mm up to unjam")
                q_up, err = _ik_hold_pitch(joints, tip + np.array([0, 0, 0.012]),
                                           float(joints[1] + joints[2] + joints[3]),
                                           float(joints[4]), ret_err=True)
                if err < 0.02:
                    send_joints(q_up)
                    q_cmd = np.asarray(q_up, np.float64).copy()
                hist.clear()
                stall_cool = now + 2.0
                time.sleep(0.35)
                continue

            # straight-line glide, speed ∝ remaining TIP distance (never camera range)
            v = float(np.clip(APPR_KP * d_stand, V_MIN, V_MAX))
            step = min(v * LOOP_DT, d_stand)
            p_tgt = tip + (p_stand - tip) / d_stand * step
            z_meas = float(tip[2])
            if z_meas < -0.012:                       # deck: never push further down
                p_tgt[2] = max(p_tgt[2], z_meas)
            p_tgt[2] = max(p_tgt[2], 0.0)             # approach stays above the base plane

            w = float(np.clip(1.0 - d_stand / PITCH_RAMP_M, 0.0, 1.0))
            _pan_gz, pitch_gz = gaze_steps(p_obj, joints)
            pitch_tgt = float(np.clip((1.0 - w) * pitch_gz + w * pitch_hint, -20.0, 100.0))

            q_t, err = _ik_hold_pitch(q_cmd, p_tgt, pitch_tgt, float(q_cmd[4]), ret_err=True)
            if err > IK_RESID_MAX:
                q_t, err = _ik_hold_pitch(q_cmd, p_tgt, pitch_hint, float(q_cmd[4]), ret_err=True)
            if err > IK_RESID_MAX:
                raise Abort(f"{label}: waypoint unreachable (IK residual {err*1e3:.0f}mm). "
                            f"Refusing to drive to a half-solved pose.")
            dq = np.clip(q_t - q_cmd, -JOINT_RATE_MAX * LOOP_DT, JOINT_RATE_MAX * LOOP_DT)
            q_cmd = q_cmd + dq
            # anti-windup: never let the command grind ahead of a blocked arm
            q_cmd = joints + np.clip(q_cmd - joints, -2.5, 2.5)
            send_joints(q_cmd)

            n += 1
            if n % 20 == 0:
                with lock:
                    state["loop_hz"] = round(20.0 / max(1e-3, time.time() - t_hz), 1)
                t_hz = time.time()
            time.sleep(max(0.0, LOOP_DT - (time.time() - t0)))
        raise Abort(f"{label}: approach timed out")
    finally:
        approach_active[0] = False


def descend_and_grasp(label, p_obj, pitch, dry):
    """Straight down onto the FROZEN map point, hand pitched down. DELIBERATELY
    NO PIXELS: a down-looking camera puts the object at the frame bottom while
    the hand is still 2cm above it. We know where it is; go there."""
    set_phase(f"GRASP {label}", "descending onto the locked point")
    e = obj_map.get(label)
    size = e.size if e is not None else SIZE_DEFAULT
    p_grip = np.asarray(p_obj, np.float64).copy()
    p_grip[2] = float(np.clip(p_grip[2], max(size / 2.0 - 0.005, OBJ_FLOOR_Z), 0.06))

    send_joints(observe()[0].astype(np.float64), gripper=95.0)
    time.sleep(0.3)
    for i in range(14):
        checkpoint()
        q = observe()[0].astype(np.float64)
        tip = fk(q)[:3, 3]
        d = p_grip - tip
        n = float(np.linalg.norm(d))
        say(f"descend {i}: tip->obj={n*1e3:.0f}mm")
        if n <= 0.008:
            break
        p_tgt = tip + d / n * min(0.015, n)
        q_t, err = _ik_hold_pitch(q, p_tgt, pitch, float(q[4]), ret_err=True)
        if err > 0.005:
            say(f"{label}: descend blocked — IK residual {err*1e3:.0f}mm, stopping here")
            break
        goto_smooth(q_t, settle=0.10, step=1.0)
    else:
        say(f"{label}: descend ran out of steps")

    if dry:
        say(f"{label}: DRY RUN — skipping the close, retreating up")
        q = observe()[0].astype(np.float64)
        tip = fk(q)[:3, 3]
        q_up, err = _ik_hold_pitch(q, tip + np.array([0, 0, 0.06]), pitch,
                                   float(q[4]), ret_err=True)
        if err < 0.02:
            goto_smooth(q_up, settle=0.4)
        return True

    set_phase(f"GRASP {label}", "slow torque-sensed close")
    contact, i_idle = close_with_current(step=3.0, delay=0.16)
    say(f"close: contact={contact}")

    set_phase("LIFT")
    q = observe()[0].astype(np.float64)
    tip = fk(q)[:3, 3]
    q_up, err = _ik_hold_pitch(q, tip + np.array([0, 0, 0.10]), pitch,
                               float(q[4]), ret_err=True)
    if err < 0.02:
        goto_smooth(q_up, settle=0.6)

    hold = [abs(c - i_idle) for c in (gripper_current() for _ in range(12)) if c is not None]
    hold_di = float(np.mean(hold)) if hold else 0.0
    observe()
    d = raw_det(label, max_age=1.2)
    held_vis = (d is not None and
                math.hypot(d["uv"][0] - HAND_UV[0], d["uv"][1] - HAND_UV[1]) < 110)
    say(f"lifted: hold dI={hold_di:.1f} in-hand-visual={held_vis}")
    return held_vis or hold_di >= 3.0


def close_with_current(step=4.0, delay=0.1):
    """Close in small increments, watching servo current; stop the instant it
    rises (torque change = fingers on the object)."""
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
            send_joints(joints, gripper=max(0.0, pct - 14.0))   # firm hold for transit
            time.sleep(0.4)
            return True, i_idle
    # closed on air: never park stalled shut (overload latch) — relax
    send_joints(observe(overlay=False)[0], gripper=40.0)
    time.sleep(0.4)
    return False, i_idle


# ---------------- missions ----------------
def run_search_and_map():
    try:
        stop_flag.clear()
        with lock:
            state["running"] = True
            state["t0"] = time.time()
        scan_map(span_deg=70.0)
        labels = list(det_classes[0])
        found, missing = [], []
        for lbl in labels:
            e = obj_map.get(lbl)
            if e is None or e.position() is None:
                if raw_det(lbl, max_age=10.0) is None:
                    missing.append(lbl)
                    continue
            aim_at(lbl)
            try:
                survey(lbl)
                found.append(lbl)
            except Abort as ex:
                say(f"{lbl}: {ex}")
                missing.append(lbl)
        detail = f"mapped: {', '.join(found) or 'nothing'}"
        if missing:
            detail += f" — not found: {', '.join(missing)}"
        set_phase("MAPPED", detail)
    except Abort as ex:
        set_phase("ABORTED", str(ex))
    except Exception as ex:
        set_phase("ERROR", f"{type(ex).__name__}: {ex}")
    finally:
        with lock:
            state["running"] = False
            state["map"] = obj_map.summary()


def run_pick(label, dry=False):
    try:
        stop_flag.clear()
        with lock:
            state["running"] = True
            state["t0"] = time.time()
            state["pick_label"] = label
        e = obj_map.get(label)
        need_survey = (e is None or e.position() is None
                       or e.sigma() > SIGMA_OK or e.frozen)
        if e is not None:
            with map_lock:
                e.frozen = False     # a new pick re-earns the freeze
        if need_survey:
            if raw_det(label, max_age=5.0) is None and not scan_for(label):
                raise Abort(f"'{label}' not found in the gripper view")
            aim_at(label)
            survey(label)

        for attempt in range(3):
            if attempt:
                say(f"pick retry {attempt + 1}/3")
                send_joints(observe()[0], gripper=95.0)
                time.sleep(0.4)
                q = observe()[0].astype(np.float64)
                tip = fk(q)[:3, 3]
                q_up, err = _ik_hold_pitch(q, tip + np.array([0, 0, 0.06]),
                                           float(q[1] + q[2] + q[3]), float(q[4]),
                                           ret_err=True)
                if err < 0.02:
                    goto_smooth(q_up, settle=0.5)
                with map_lock:
                    ee = obj_map.entries.get(label)
                    if ee is not None:
                        ee.frozen = False
                survey(label)
            p_obj, pitch = approach(label)
            ok = descend_and_grasp(label, p_obj, pitch, dry)
            if dry:
                set_phase("DONE", f"dry run complete — map said '{label}' is at "
                          f"({p_obj[0]:+.3f},{p_obj[1]:+.3f},{p_obj[2]:+.3f})")
                return
            if ok:
                obj_map.remove(label)   # it's in the hand now, not on the table
                set_phase("DONE", f"'{label}' picked and lifted")
                return
        raise Abort(f"grasp failed after 3 attempts — '{label}' never held")
    except Abort as ex:
        set_phase("ABORTED", str(ex))
        try:
            goto_smooth(VIEW, settle=0.5)
        except Exception:
            pass
    except Exception as ex:
        set_phase("ERROR", f"{type(ex).__name__}: {ex}")
    finally:
        try:
            with lock:
                g = state.get("gripper")
            if g is not None and 0 <= g < 15.0:
                c = gripper_current()
                if c is None or abs(c) < 4.0:      # not carrying load: relax, no overload
                    send_joints(observe(overlay=False)[0], gripper=40.0)
        except Exception:
            pass
        with lock:
            state["running"] = False
            state["pick_label"] = None
            state["map"] = obj_map.summary()


# ---------------- FPV overlay ----------------
def publish(rgb, joints=None):
    img = np.ascontiguousarray(rgb[:, :, ::-1])
    now = time.time()
    # CYAN CROSS = measured fingertips; MAGENTA CIRCLE = TF-predicted fingertips.
    # The gap between them IS the hand-eye error, live. Run Calib to close it.
    cv2.drawMarker(img, (int(HAND_UV[0]), int(HAND_UV[1])), (60, 200, 255),
                   cv2.MARKER_CROSS, 26, 2)
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
            cv2.putText(img, f"hand-eye err {gap:.0f}px", (10, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    # fresh raw detections (green boxes)
    with lock:
        dets = list(state.get("dets") or [])
    for d in dets:
        if now - d["t"] > DET_FRESH_S:
            continue
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"{d['label']} {d['conf']:.2f}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    # MAP REPROJECTIONS (yellow) = the honest localization check: the box either
    # sits on the object you can see, or the map is wrong. Logs can't fake this.
    if joints is not None and kin is not None and fx > 0:
        with map_lock:
            snap = [(lbl, e.position(), e.frozen, e.quality)
                    for lbl, e in obj_map.entries.items() if e.initialized]
        try:
            T_cam = T_cam_of(joints)
        except Exception:
            T_cam = None
        if T_cam is not None:
            for lbl, p, frozen, quality in snap:
                uvc = project_base(p, T_cam)
                if uvc is None:
                    continue
                u, v = int(round(uvc[0])), int(round(uvc[1]))
                if not (-300 < u < img.shape[1] + 300 and -300 < v < img.shape[0] + 300):
                    continue
                col = (0, 165, 255) if frozen else (0, 235, 255)
                cv2.rectangle(img, (u - 16, v - 16), (u + 16, v + 16), col, 2)
                cv2.drawMarker(img, (u, v), col, cv2.MARKER_TILTED_CROSS, 12, 1)
                tag = f"{lbl}" + (" [frozen]" if frozen else f" [{quality}]")
                cv2.putText(img, tag, (u - 18, v - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    with lock:
        cv2.putText(img, state["phase"], (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (80, 255, 120), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if ok:
        with lock:
            frame_jpeg[0] = buf.tobytes()


# ---------------- jog (browser teleop, unchanged model from stack_mission2) ----------------
jog_held = set()
jog_held_lock = threading.Lock()
JOG_SPEED = 0.05
JOG_AZIM = 14.0
JOG_ACC = 0.16
JOG_DT = 0.05
JOG_WRIST = 45.0
GRIP_RATE = 28.0
JOG_DIRS = ("fwd", "back", "left", "right", "up", "down",
            "roll_cw", "roll_ccw", "pitch_up", "pitch_dn")
grip_target = [95.0]
grip_cmd = [50.0]
jog_vec = {"r": 0.0, "th": 0.0, "z": 0.0, "t": 0.0}
JOG_VEC_TTL = 0.30


def jog_loop():
    vr_f = vth_f = vz_f = 0.0
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
            q_cmd = None
            r_tgt = None
            time.sleep(0.05)
            continue
        moving = (held or analog_active or abs(vr_f) > 1e-4 or abs(vth_f) > 1e-4
                  or abs(vz_f) > 1e-4 or abs(roll_f) > 0.5 or abs(tilt_f) > 0.5)
        gripping = abs(grip_cmd[0] - grip_target[0]) > 0.5
        if not moving and not gripping:
            q_cmd = None
            r_tgt = None
            time.sleep(0.03)
            continue
        try:
            joints, _rgb, obs = observe(overlay=True)
        except Exception:
            time.sleep(0.05)
            continue
        T_meas = fk(joints)
        p = T_meas[:3, 3]
        r_now = math.hypot(p[0], p[1])
        th_now = math.atan2(p[1], p[0])
        if q_cmd is None or float(np.max(np.abs(q_cmd - joints))) > 6.0:
            q_cmd = joints.copy()
            r_tgt, th_tgt, z_tgt = r_now, th_now, float(p[2])
            pitch_tgt = float(joints[1] + joints[2] + joints[3])
            j5_cmd = float(joints[4])
            gp = obs.get("gripper.pos")
            if isinstance(gp, (int, float)) and gp >= 0:
                grip_cmd[0] = float(gp)
        vr_des = (JOG_SPEED if "fwd" in held else 0.0) - (JOG_SPEED if "back" in held else 0.0)
        vth_des = (JOG_AZIM if "left" in held else 0.0) - (JOG_AZIM if "right" in held else 0.0)
        vz_des = (JOG_SPEED if "up" in held else 0.0) - (JOG_SPEED if "down" in held else 0.0)
        if analog_fresh:
            vr_des += JOG_SPEED * va["r"]
            vth_des += JOG_AZIM * va["th"]
            vz_des += JOG_SPEED * va["z"]
        vr_des = float(np.clip(vr_des, -JOG_SPEED, JOG_SPEED))
        vth_des = float(np.clip(vth_des, -JOG_AZIM, JOG_AZIM))
        vz_des = float(np.clip(vz_des, -JOG_SPEED, JOG_SPEED))
        vr_f = (1 - JOG_ACC) * vr_f + JOG_ACC * vr_des
        vth_f = (1 - JOG_ACC) * vth_f + JOG_ACC * vth_des
        vz_f = (1 - JOG_ACC) * vz_f + JOG_ACC * vz_des
        tilt_des = (JOG_WRIST if "pitch_dn" in held else 0.0) - (JOG_WRIST if "pitch_up" in held else 0.0)
        roll_des = (JOG_WRIST if "roll_cw" in held else 0.0) - (JOG_WRIST if "roll_ccw" in held else 0.0)
        tilt_f = (1 - JOG_ACC) * tilt_f + JOG_ACC * tilt_des
        roll_f = (1 - JOG_ACC) * roll_f + JOG_ACC * roll_des
        pitch_tgt = float(np.clip(pitch_tgt + tilt_f * JOG_DT, -20.0, 100.0))
        j5_cmd += roll_f * JOG_DT
        grip_cmd[0] += float(np.clip(grip_target[0] - grip_cmd[0],
                                     -GRIP_RATE * JOG_DT, GRIP_RATE * JOG_DT))
        if abs(r_tgt - r_now) < 0.06:
            r_tgt += vr_f * JOG_DT
        if abs(th_tgt - th_now) < math.radians(20):
            th_tgt += math.radians(vth_f) * JOG_DT
        if abs(z_tgt - p[2]) < 0.06:
            z_tgt += vz_f * JOG_DT
        r_tgt = float(np.clip(r_tgt, 0.12, 0.42))
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


def idle_view():
    while True:
        with lock:
            busy = state["running"]
        with jog_held_lock:
            jogging = bool(jog_held)
        if not busy and not jogging:
            try:
                observe()
            except Exception:
                time.sleep(1.0)
        time.sleep(0.25)


# ---------------- web ----------------
app = Flask(__name__)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAX FPV map approach</title><style>
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
td,th{padding:3px 6px;border-bottom:1px solid #2b3540;color:#93a1ae;text-align:left}
td+td{color:#e6ecf1}
#maptab tr.obj{cursor:pointer}#maptab tr.obj:hover td{background:#232e39}
.btns{display:flex;gap:8px;margin-top:12px}
button{flex:1;padding:9px 0;border:0;border-radius:5px;font:600 13px "Segoe UI";cursor:pointer}
#b-map{background:#2e9e5b;color:#fff}#b-stop{background:#b0533c;color:#fff}
.dim{background:#2b3540;color:#e6ecf1}
pre{background:#10161c;border:1px solid #2b3540;border-radius:6px;padding:10px;font-size:11.5px;
line-height:1.5;height:220px;overflow-y:auto;white-space:pre-wrap;margin:12px 0 0}
.jog{margin-top:14px}
#jogmsg{color:#6fb2ff;font-family:ui-monospace,Consolas,monospace;font-size:11px;margin-left:8px}
.pad{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:6px}
.pad .j{padding:14px 0;background:#26313c;color:#e6ecf1;border:1px solid #37454f;border-radius:6px;
font:600 14px "Segoe UI";cursor:pointer;touch-action:manipulation;user-select:none}
.pad .j:active{background:#3a6ea5}
.pad .g{background:#2e5b46}
.qbox{margin-top:14px}
.qrow{display:flex;gap:8px;margin-top:6px;align-items:center}
.qrow input[type=text]{flex:1;padding:10px;background:#10161c;border:1px solid #37454f;border-radius:6px;
color:#e6ecf1;font:14px "Segoe UI"}
.qrow button{padding:10px 18px;background:#2e6ea5;color:#fff;border:0;border-radius:6px;
font:600 13px "Segoe UI";cursor:pointer;flex:0 0 auto}
.qrow label{color:#93a1ae;font-size:12px;white-space:nowrap}
.sticks{display:flex;gap:18px;justify-content:center;align-items:center;margin:8px 0 8px;flex-wrap:wrap}
.stick{position:relative;width:118px;height:118px;border-radius:50%;flex:0 0 auto;
background:radial-gradient(circle at 50% 42%,#222c37,#141a21 72%);border:1px solid #37454f;
touch-action:none;user-select:none;display:flex;align-items:center;justify-content:center}
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
<h1>RAX · FPV map approach<span class="tail">detect → map (base frame) → fly the fingertip through the map</span></h1>
<div class="cams">
  <div><p class="lbl">gripper camera (OAK-D) · yellow box = map reprojection (must sit ON the object)</p>
    <img src="/stream" alt="gripper view"></div>
  <div class="phone-hide">
    <p class="lbl">robot + object map · drag to rotate ·
      <a id="r3d-link" href="#" target="_blank" style="color:#6fb2ff;text-transform:none">rerun ↗</a></p>
    <canvas id="v3d"
       style="width:100%;height:340px;border:1px solid #2b3540;border-radius:6px;background:#0b0f14;display:block;touch-action:none"></canvas>
  </div>
</div>
<div class="panel">
  <div class="phase" id="phase">—</div><div class="detail" id="detail"></div>
  <table><tbody id="rows"></tbody></table>
  <p class="lbl" style="margin-top:12px">object map (click a row to pick it)</p>
  <table id="maptab"><thead><tr><th>label</th><th>r</th><th>ang</th><th>z</th><th>σ</th><th>n</th><th>src</th></tr></thead>
  <tbody></tbody></table>
  <div class="btns">
    <button id="b-map" onclick="fetch('/start',{method:'POST'})">Search + Map</button>
    <button id="b-stop" onclick="fetch('/stop',{method:'POST'})">Stop</button>
    <button class="dim" onclick="fetch('/home',{method:'POST'})">Fold home</button>
  </div>
  <div class="btns">
    <button class="dim" onclick="fetch('/reset',{method:'POST'})">Reset pose</button>
    <button class="dim" onclick="fetch('/caltip',{method:'POST'})">Calibrate fingertip</button>
    <button class="dim" onclick="fetch('/calib',{method:'POST'})"
      title="Put an object in view first. Re-fits the gripper→camera transform from the robot's own motion (~30s).">Calibrate HAND-EYE</button>
  </div>
  <div class="qbox">
    <p class="lbl">pick an object from the map</p>
    <div class="qrow">
      <input id="picklbl" type="text" placeholder="e.g. red cube">
      <label><input id="dry" type="checkbox"> dry run</label>
      <button onclick="doPick()">Pick</button>
    </div>
  </div>
  <div class="qbox">
    <p class="lbl">detection query — each comma term is a DISTINCT map object</p>
    <div class="qrow">
      <input id="query" type="text" placeholder="e.g. red cube, green cube, bottle">
      <button id="q-set" onclick="setQuery()">Set</button>
    </div>
  </div>
  <div class="jog">
    <p class="lbl">FPV polar jog · sticks, or W/S radius · A/D base · R/F height · T/G tilt · Q/E roll · space grip <span id="jogmsg"></span></p>
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
  <pre id="log"></pre>
</div></main><script>
// ---- 3D viewer: real URDF meshes + ALL map objects ------------------------
(function(){
  const a = document.getElementById('r3d-link'); if(a) a.href = `http://${location.hostname}:%%RERUN%%`;
  const cv = document.getElementById('v3d'); if(!cv) return;
  const ctx = cv.getContext('2d');
  let az = -1.05, el = 0.62, geom = {links:[], ee:null, objs:[]};
  function resize(){ const r = cv.getBoundingClientRect(); const dpr = window.devicePixelRatio||1;
    cv.width = Math.round(r.width*dpr); cv.height = Math.round(r.height*dpr); ctx.setTransform(dpr,0,0,dpr,0,0); }
  window.addEventListener('resize', resize); resize();
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
  function proj(p){ const ca=Math.cos(az), sa=Math.sin(az), ce=Math.cos(el), se=Math.sin(el);
    const rx = -p[0]*sa + p[1]*ca;
    const dep =  p[0]*ca + p[1]*sa;
    const uy =  p[2]*ce - dep*se;
    const W = cv.clientWidth, H = cv.clientHeight, s = Math.min(W,H)/0.60;
    return [W*0.5 + s*rx, H*0.62 - s*uy, dep*ce + p[2]*se]; }
  function line(a,b,col,w){ const p=proj(a), q=proj(b); ctx.strokeStyle=col; ctx.lineWidth=w||1;
    ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]); ctx.stroke(); }
  function drawMeshes(){
    const xf = geom.xf || {}; const tris = [];
    const ca=Math.cos(az), sa=Math.sin(az), ce=Math.cos(el), se=Math.sin(el);
    const vdx=ca*ce, vdy=sa*ce, vdz=se;
    for(const L of mesh){
      const T = xf[L.name]; if(!T) continue;
      const nv = L.v.length/3, sx=new Float64Array(nv), sy=new Float64Array(nv), sd=new Float64Array(nv);
      const bx=new Float64Array(nv), by=new Float64Array(nv), bz=new Float64Array(nv);
      for(let i=0;i<nv;i++){
        const x=L.v[3*i], y=L.v[3*i+1], z=L.v[3*i+2];
        const X = T[0]*x+T[1]*y+T[2]*z+T[3];
        const Y = T[4]*x+T[5]*y+T[6]*z+T[7];
        const Z = T[8]*x+T[9]*y+T[10]*z+T[11];
        bx[i]=X; by[i]=Y; bz[i]=Z;
        const s = proj([X,Y,Z]); sx[i]=s[0]; sy[i]=s[1]; sd[i]=s[2];
      }
      const c = LINKCOL[L.name] || [140,160,190];
      for(let t=0;t<L.f.length;t+=3){
        const a=L.f[t], b=L.f[t+1], q=L.f[t+2];
        const ux=bx[b]-bx[a], uy2=by[b]-by[a], uz=bz[b]-bz[a];
        const vx=bx[q]-bx[a], vy=by[q]-by[a], vz=bz[q]-bz[a];
        let nx=uy2*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy2*vx;
        const nl=Math.hypot(nx,ny,nz)||1; nx/=nl; ny/=nl; nz/=nl;
        if(nx*vdx + ny*vdy + nz*vdz >= 0) continue;
        const lam = Math.max(0.34, Math.min(1, 0.42 + 0.58*(0.35*nx - 0.45*ny + 0.82*nz)));
        tris.push([(sd[a]+sd[b]+sd[q])/3, sx[a],sy[a],sx[b],sy[b],sx[q],sy[q],
                   `rgb(${Math.round(c[0]*lam)},${Math.round(c[1]*lam)},${Math.round(c[2]*lam)})`]);
      }
    }
    tris.sort((p,q)=>q[0]-p[0]);
    ctx.lineJoin='round';
    for(const t of tris){
      ctx.fillStyle = t[7]; ctx.strokeStyle = t[7]; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(t[1],t[2]); ctx.lineTo(t[3],t[4]); ctx.lineTo(t[5],t[6]); ctx.closePath();
      ctx.fill(); ctx.stroke();
    }
  }
  function objColor(lbl){
    if(/red/i.test(lbl)) return '#e2574c';
    if(/green/i.test(lbl)) return '#3fc46b';
    if(/blue/i.test(lbl)) return '#4a93c9';
    let h=0; for(const ch of lbl) h=(h*31+ch.charCodeAt(0))>>>0;
    return `hsl(${h%360},60%,60%)`;
  }
  function draw(){
    const W=cv.clientWidth, H=cv.clientHeight; ctx.clearRect(0,0,W,H);
    const g=0.30, n=6; ctx.globalAlpha=0.5;
    for(let i=0;i<=n;i++){ const t=-g+2*g*i/n;
      line([t,-g,0],[t,g,0],'#243040',1); line([-g,t,0],[g,t,0],'#243040',1); }
    ctx.globalAlpha=1;
    line([0,0,0],[0.08,0,0],'#c9524a',2); line([0,0,0],[0,0.08,0],'#4a93c9',2); line([0,0,0],[0,0,0.08],'#4ac275',2);
    if(mesh && geom.xf){ drawMeshes(); }
    else {
      const L=geom.links||[];
      for(let i=0;i+1<L.length;i++) line(L[i],L[i+1],'#7f9fd8',4);
    }
    if(geom.ee){ const s=proj(geom.ee); ctx.fillStyle='#e6ecf1'; ctx.beginPath(); ctx.arc(s[0],s[1],4.5,0,7); ctx.fill(); }
    for(const o of (geom.objs||[])){
      const c=[], h=(o.size||0.03)/2, p=o.p;
      for(let dx of [-h,h]) for(let dy of [-h,h]) for(let dz of [-h,h]) c.push([p[0]+dx,p[1]+dy,p[2]+dz]);
      const E=[[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]];
      const col = objColor(o.label);
      for(const e of E) line(c[e[0]],c[e[1]],col,2);
      const s=proj(p); ctx.fillStyle=col; ctx.font='11px ui-monospace,Consolas';
      ctx.fillText(`${o.label}${o.frozen?' ❄':''} r=${Math.hypot(p[0],p[1]).toFixed(2)}m`, s[0]+8, s[1]-8);
    }
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
async function doPick(){
  const l = document.getElementById('picklbl').value.trim();
  if(!l) return;
  const dry = document.getElementById('dry').checked ? '&dry=1' : '';
  try{ const r = await (await fetch('/pick?label='+encodeURIComponent(l)+dry,{method:'POST'})).json();
    if(!r.ok) jmsg('✗ '+r.reason); }catch(e){}
}
async function press(d){ try{ const r = await (await fetch(`/jogpress?dir=${d}`,{method:'POST'})).json();
  if(!r.ok) jmsg('✗ '+r.reason); else if(r.grip) jmsg(r.grip); }catch(e){} }
async function release(d){ try{ await fetch(`/jogrelease?dir=${d}`,{method:'POST'}); }catch(e){} }
const HOLD = new Set(['fwd','back','left','right','up','down','roll_ccw','roll_cw','pitch_up','pitch_dn']);
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
    b.addEventListener('click', () => press(d));
  }
});
const jv = {r:0, th:0, z:0};
let jvTimer = null;
function jvSend(){
  fetch(`/jogvec?r=${jv.r.toFixed(3)}&th=${jv.th.toFixed(3)}&z=${jv.z.toFixed(3)}`,{method:'POST'}).catch(()=>{});
}
function jvActive(){ return Math.abs(jv.r)>1e-3 || Math.abs(jv.th)>1e-3 || Math.abs(jv.z)>1e-3; }
function jvStart(){ if(!jvTimer) jvTimer = setInterval(jvSend, 60); }
function jvMaybeStop(){ if(!jvActive() && jvTimer){ jvSend(); clearInterval(jvTimer); jvTimer=null; } }
function makeStick(id, vert, apply){
  const pad = document.getElementById(id); if(!pad) return;
  const nub = pad.querySelector('.nub');
  let on=false, cx=0, cy=0, R=1;
  const dz = v => Math.abs(v) < 0.09 ? 0 : v;
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
makeStick('st-move', false, (x,y)=>{ jv.r=-y; jv.th=-x; });
makeStick('st-lift', true,  (x,y)=>{ jv.z=-y; });
const KEYMAP = {w:'fwd', s:'back', a:'left', d:'right', r:'up', f:'down',
  q:'roll_ccw', e:'roll_cw', t:'pitch_up', g:'pitch_dn'};
const down = new Set();
document.addEventListener('keydown', ev => {
  if(ev.target.tagName === 'INPUT') return;
  const k = ev.key.toLowerCase();
  if(KEYMAP[k]){ ev.preventDefault(); if(!down.has(k)){ down.add(k); press(KEYMAP[k]); } }
  else if(k === 'o'){ ev.preventDefault(); press('open'); }
  else if(k === 'c'){ ev.preventDefault(); press('close'); }
  else if(ev.key === ' '){ ev.preventDefault(); if(!down.has(' ')){ down.add(' ');
    press(window._gripOpen ? 'close':'open'); window._gripOpen = !window._gripOpen; } }
});
document.addEventListener('keyup', ev => {
  const k = ev.key.toLowerCase();
  if(KEYMAP[k]){ down.delete(k); release(KEYMAP[k]); }
  else if(ev.key === ' '){ down.delete(' '); }
});
window.addEventListener('blur', () => { down.clear(); release('all'); });
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
      ['tip → target', s.dist_mm != null ? s.dist_mm + ' mm' : '—'],
      ['picking', s.pick_label || '—'],
      ['elapsed', s.t0 && s.running ? Math.round(Date.now()/1000 - s.t0) + ' s' : '—'],
    ];
    document.getElementById('rows').innerHTML =
      rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('');
    const mb = document.querySelector('#maptab tbody');
    const m = s.map || {};
    mb.innerHTML = Object.keys(m).map(l => {
      const e = m[l];
      const sig = e.sigma_mm != null ? e.sigma_mm+'mm' : '—';
      return `<tr class="obj" onclick="document.getElementById('picklbl').value='${l.replace(/'/g,"\\\\'")}'">`+
        `<td>${l}${e.frozen?' ❄':''}</td><td>${e.r_cm??'—'}cm</td><td>${e.ang_deg??'—'}°</td>`+
        `<td>${e.z_cm??'—'}cm</td><td>${sig}</td><td>${e.n_obs}</td><td>${e.quality}</td></tr>`;
    }).join('') || '<tr><td colspan="7">empty — run Search + Map</td></tr>';
    document.getElementById('log').textContent = (s.log||[]).join('\\n');
    const qi = document.getElementById('query');
    if(s.query && !qi.value && document.activeElement !== qi) qi.value = s.query;
  }catch(e){}
  setTimeout(tick, 700);
}
tick();
</script></body></html>""".replace("%%RERUN%%", str(RERUN_WEB_PORT))


@app.route("/")
def index():
    resp = Response(PAGE, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/status")
def status():
    with lock:
        s = {k: v for k, v in state.items()}
    s["log"] = list(log)
    return jsonify(s)


@app.route("/map")
def map_route():
    return jsonify(obj_map.summary())


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


def _decimate(V, F, voxel):
    """Voxel-cluster a dense STL down to something a browser can draw."""
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
    # dedupe WITHOUT destroying winding: rotate each face so its smallest index leads
    roll = np.argmin(Fn, axis=1)
    idx = (np.arange(3)[None, :] + roll[:, None]) % 3
    Fn = np.unique(np.take_along_axis(Fn, idx, axis=1), axis=0)
    return Vn, Fn


_c, _s = math.cos(1.5708), math.sin(1.5708)
JAW_T = np.array([[1, 0, 0, 0.0202],
                  [0, _c, -_s, 0.0188],
                  [0, _s, _c, -0.0234],
                  [0, 0, 0, 1.0]], dtype=np.float64)
_urdf_payload = [None]


@app.route("/urdf")
def urdf_route():
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
        except Exception as e:
            say(f"URDF viewer meshes failed: {type(e).__name__}: {e}")
            _urdf_payload[0] = []
    return jsonify(links=_urdf_payload[0])


@app.route("/geom")
def geom():
    """Live 3D geometry: per-link poses + the WHOLE object map, base frame."""
    with lock:
        jlist = state.get("joints")
    links, ee, xf = [], None, {}
    if jlist and kin is not None:
        try:
            q = np.array(jlist, dtype=np.float64)
            chain = link_chain(q)
            links = [[round(float(T[0, 3]), 4), round(float(T[1, 3]), 4),
                      round(float(T[2, 3]), 4)] for _n, T in chain]
            for _n, T in chain:
                xf[_n] = [round(float(v), 5) for v in np.asarray(T, np.float64).ravel()]
            Tg = dict(chain).get("gripper_link")
            if Tg is not None:
                xf["moving_jaw_so101_v1_link"] = [
                    round(float(v), 5) for v in (np.asarray(Tg, np.float64) @ JAW_T).ravel()]
            Tee = fk(q)
            ee = [round(float(Tee[i, 3]), 4) for i in range(3)]
        except Exception:
            pass
    objs = []
    with map_lock:
        for lbl, e in obj_map.entries.items():
            p = e.position()
            if p is None:
                continue
            objs.append({"label": lbl, "p": [round(float(v), 4) for v in p],
                         "size": round(e.size, 3), "frozen": e.frozen})
    return jsonify(links=links, xf=xf, ee=ee, objs=objs)


@app.route("/start", methods=["POST"])
def start():
    with lock:
        busy = state["running"]
    if not busy and (mission_thread[0] is None or not mission_thread[0].is_alive()):
        mission_thread[0] = threading.Thread(target=run_search_and_map, daemon=True)
        mission_thread[0].start()
        return jsonify(ok=True)
    return jsonify(ok=False, reason="already running")


@app.route("/pick", methods=["POST"])
def pick():
    label = (request.args.get("label") or "").strip()
    dry = request.args.get("dry") in ("1", "true", "yes")
    if not label:
        return jsonify(ok=False, reason="need ?label=")
    with lock:
        busy = state["running"]
    if busy or (mission_thread[0] is not None and mission_thread[0].is_alive()):
        return jsonify(ok=False, reason="already running")
    if label not in det_classes[0] and obj_map.get(label) is None:
        return jsonify(ok=False, reason=f"'{label}' is not in the query or the map")
    mission_thread[0] = threading.Thread(target=run_pick, args=(label, dry), daemon=True)
    mission_thread[0].start()
    return jsonify(ok=True)


@app.route("/stop", methods=["POST"])
def stop():
    stop_flag.set()
    say("STOP requested")
    return jsonify(ok=True)


@app.route("/setquery", methods=["POST"])
def setquery():
    q = (request.args.get("q") or "").strip()
    if not q or detector is None:
        return jsonify(ok=False, reason="empty query or detector not ready")
    pending_query[0] = q        # applied inside the YOLO thread (set_classes is thread-affine)
    with lock:
        state["query"] = q
    return jsonify(ok=True, query=q)


@app.route("/jogpress", methods=["POST"])
def jogpress():
    with lock:
        busy = state["running"]
    if busy:
        return jsonify(ok=False, reason="mission running")
    d = request.args.get("dir", "")
    if d in ("open", "close"):
        grip_target[0] = 95.0 if d == "open" else 5.0
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


@app.route("/calib", methods=["POST"])
def calib():
    with lock:
        busy = state["running"]
    if busy:
        return jsonify(ok=False, reason="mission running — stop first")
    # calibrate against whatever is actually in view: an explicit ?label=, else
    # the freshest detected label, else the first query class.
    label = (request.args.get("label") or "").strip()
    if not label:
        with lock:
            fresh = [d["label"] for d in (state.get("dets") or [])]
        label = fresh[0] if fresh else det_classes[0][0]

    def _cal():
        try:
            stop_flag.clear()
            with lock:
                state["running"] = True
            calibrate_handeye(label)
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


@app.route("/caltip", methods=["POST"])
def caltip():
    """Measure where the black fingertips actually sit in the image right now."""
    with lock:
        busy = state["running"]
    if busy:
        return jsonify(ok=False, reason="mission running — stop first")

    def _cal():
        try:
            stop_flag.clear()
            set_phase("CAL TIP", "descending to grasp deck")
            goto_smooth(np.array([-7.5, 49.0, 29.0, -58.0, -40.0]), settle=1.2)
            send_joints(observe()[0], gripper=15.0)
            time.sleep(0.8)
            joints, rgb, _ = observe(overlay=False)
            img = np.ascontiguousarray(rgb)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            dark = cv2.inRange(hsv, (0, 0, 0), (180, 90, 70))
            dark[: int(0.45 * dark.shape[0]), :] = 0
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
            gx = float(cent[best][0])
            gy = float(y + 0.15 * h)
            global HAND_UV
            HAND_UV = (gx, gy)
            send_joints(observe()[0], gripper=95.0)
            set_phase("CAL TIP", f"fingertips at ({gx:.0f},{gy:.0f}) — HAND_UV updated")
        except Exception as e:
            set_phase("ERROR", f"caltip: {e}")
    threading.Thread(target=_cal, daemon=True).start()
    return jsonify(ok=True)


@app.route("/reset", methods=["POST"])
def reset_pose():
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
            set_phase("IDLE", "at working pose — ready")
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


# ---------------- Rerun 3D ----------------
_rerun_ok = [False]


def start_rerun():
    try:
        import rerun as rr
        rr.init("rax_fpv_approach", spawn=False)
        rr.serve_grpc(grpc_port=RERUN_GRPC_PORT)
        rr.serve_web_viewer(
            open_browser=False, web_port=RERUN_WEB_PORT,
            connect_to=f"rerun+http://{HOST_IP}:{RERUN_GRPC_PORT}/proxy")
        _rerun_ok[0] = True
        say(f"Rerun 3D viewer: http://{HOST_IP}:{RERUN_WEB_PORT}")
    except Exception as e:
        say(f"Rerun 3D disabled: {e}")


def rerun_thread():
    if not _rerun_ok[0]:
        return
    try:
        from lerobot.utils.manipulation_sim3d import log_manipulation_sim3d
    except Exception as e:
        say(f"Rerun 3D disabled (sim3d import): {e}")
        return
    frame = 0
    warned = [False]
    while True:
        try:
            with lock:
                jlist = state.get("joints")
            centers = half = labels = None
            with map_lock:
                pts = [(lbl, e.position(), e.size) for lbl, e in obj_map.entries.items()
                       if e.initialized]
            if pts:
                centers = np.array([p for _l, p, _s in pts], dtype=np.float64)
                half = np.array([[s / 2, s / 2, s / 2] for _l, _p, s in pts], dtype=np.float64)
                labels = [l for l, _p, _s in pts]
            if jlist:
                joints = np.array(jlist, dtype=np.float64)
                with kin_lock:
                    log_manipulation_sim3d(
                        frame_sequence=frame, kinematics=kin, joint_deg=joints,
                        object_centers_base=centers, object_half_sizes_base=half,
                        object_labels=labels,
                        focus_object_index=0 if centers is not None else None,
                        ground_plane_z_m=0.0)
                frame += 1
        except Exception as e:
            if not warned[0]:
                warned[0] = True
                say(f"Rerun 3D log error: {type(e).__name__}: {e}")
        time.sleep(0.12)


# ---------------- main ----------------
def main():
    global robot, kin, fx, fy, cx0, cy0, detector, cam, INTR, STEREO_INTR
    clear_gripper_overload()
    say("connecting robot + camera…")
    for attempt in range(6):
        robot = make_robot_from_config(SO101FollowerConfig(
            port="COM4", id="so101_follower",
            cameras={"front": OAKDCameraConfig(
                fps=30, width=640, height=480, use_depth=True,
                stereo_extended_disparity=True,       # halves the min depth (~20cm)
                stereo_confidence_threshold=150)},
        ))
        try:
            robot.connect()
            break
        except (RuntimeError, ConnectionError) as e:
            # "Missing motor" = flaky gripper handshake; ConnectionError = a bus
            # write/read glitched mid-configure (torque-enable on a servo). Both
            # are the marginal cable and both clear on a fresh retry.
            transient = "Missing motor" in str(e) or isinstance(e, ConnectionError)
            if not transient or attempt == 5:
                raise
            say(f"connect glitched (attempt {attempt + 1}/6: {str(e)[:60]}) — retrying…")
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
    INTR = Intrinsics(fx=fx, fy=fy, cx=cx0, cy=cy0)
    STEREO_INTR = StereoIntrinsics(fx=fx, fy=fy, cx=cx0, cy=cy0,
                                   baseline_m=0.075, width=640, height=480)

    _tfd = load_tf_override()
    if _tfd:
        say(f"hand-eye: using CALIBRATED TF from {_tfd.get('fitted', '?')} "
            f"(reprojection {_tfd.get('rms_px', 0):.0f}px)")
        if float(_tfd.get("rms_px", 0) or 0) > 40.0:
            say(f"*** HAND-EYE TF FIT QUALITY IS POOR (object reprojection RMS "
                f"{_tfd['rms_px']:.0f}px). A perfect fingertip pixel does NOT clear "
                f"it — rotation about the tip ray is unobservable from the tip alone. "
                f"Expect the yellow map boxes to miss the objects; press Calibrate "
                f"HAND-EYE with an object in view before trusting any localization. ***")
    # sanity gate: the fingertip has ONE fixed pixel and we measured it. If the
    # TF disagrees, every back-projected ray is wrong — say so loudly.
    try:
        _uv = tip_pixel(np.array(HOME, np.float64))
        _gap = 1e9 if _uv is None else math.hypot(_uv[0] - HAND_UV[0], _uv[1] - HAND_UV[1])
        if _gap > 40.0:
            say(f"*** HAND-EYE TF IS BAD: fingertip reprojects "
                f"({_uv[0]:.0f},{_uv[1]:.0f}) but the fingers are at "
                f"({HAND_UV[0]:.0f},{HAND_UV[1]:.0f}) — {_gap:.0f}px off. Every range "
                f"will be wrong. Put an object in view and press Calibrate HAND-EYE. ***")
        else:
            say(f"hand-eye check: fingertip reprojects {_gap:.0f}px from HAND_UV — OK")
    except Exception as e:
        say(f"hand-eye check skipped: {type(e).__name__}: {e}")

    # yolov8s-worldv2 detects these cubes far more reliably than the L checkpoint
    # on this rig (measured 2026-07-16: L returned 0 dets on a clear green cube at
    # conf 0.02; S gave 0.30-0.70). Bigger != better for THIS worldv2 export. The
    # HSV blob fallback in the worker backs it up regardless.
    say("loading YOLO-World S (measured more reliable than L on this rig) + HSV fallback…")
    detector = YoloWorldDetector(LEROBOT + r"\yolov8s-worldv2.pt", conf=0.08, imgsz=640,
                                 color_filter_min_frac=0.0)
    q0 = ", ".join(det_classes[0])
    detector.set_query(q0)
    with lock:
        state["query"] = q0

    threading.Thread(target=yolo_worker, daemon=True).start()
    threading.Thread(target=idle_view, daemon=True).start()
    threading.Thread(target=jog_loop, daemon=True).start()
    start_rerun()
    threading.Thread(target=rerun_thread, daemon=True).start()
    set_phase("IDLE", "ready — Search + Map, then Pick")
    say(f"UI: http://{HOST_IP}:{PORT}  (Tailscale)")
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
