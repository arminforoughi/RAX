# RAX first-person-view manipulation server: see an object, locate it, drive at it,
# grasp it, put it somewhere. Admin UI, guest UI and MJPEG stream on one Flask port
# (:8484). This is the demo rig; the reusable stack it sits on is `rax.*` in src/.
#
# WHERE TO START READING
#  If you want the pipeline rather than the demo, read src/rax/grasp.py — 400 lines,
#  no Flask, no threads, runnable headless (`python -m rax.grasp --arm mock`). This
#  file is that pipeline plus everything an unattended public demo needs, so most of
#  its bulk is the demo, not the robot.
#
# THE LOOP
#  * DETECT: YOLO-World (open vocabulary) runs in a PARALLEL thread (~2.5 s) so it
#    never sits in the control path; the two cube colours also have dedicated
#    strict-HSV trackers that are tighter during close approach. publish() draws
#    every detection on the FPV, not just the cubes. A label naming a colour is
#    gated on actually containing that colour — rax.models.detection.color_filters.
#  * LOCATE: monocular. The table IS the base plane, so an object's pixel ray meets
#    it at a known height (PlaneRayLocalizer); apparent size against a class prior is
#    the fallback (ApparentSizeLocalizer). Neither needs depth, which is why the
#    OAK-D runs colour-only here — see the DEPTH OFF note in main().
#  * MAP: sense_2d folds detections into a 2D bird's-eye map. Entries are merged by
#    POSITION (one object fires under several labels), expire after MAP_TTL_S, and
#    only carry an orientation when one was actually measured.
#  * APPROACH: run_mission takes the mapped (x, y) and closes in 3 stages (step 1
#    covers ~90%, the rest correct), re-measuring between them so error is corrected
#    while there is still room to correct it. IK holds the tool pitch through it.
#  * CENTRE: the last correction is by eye — servo the object onto the fingertip
#    pixel (HAND_UV). The servo MEASURES its own gains by probing rather than
#    modelling them, because the hand-eye TF is wrong (see the NOTE below) and a
#    modelled sign would drive the wrong way. approach/visual_center.py.
#  * GRASP: descend, close on the servo current, and measure fingertip height at the
#    instant of contact — that one number is the whole carry's grip->bottom distance.
#  * PLACE: place_at puts a held object on a mapped object or a clicked spot; release
#    height needs only the destination's height plus that measured offset.
#
# TASKS: "red on green"
#  A phrase naming a destination is an instruction, not a vocabulary. Typed into
#  EITHER the task box or the detection-query box it runs as a task: Start checks
#  reads_as_a_task() first, and /setquery expands it to the vocabulary the task needs
#  so both the object AND its destination can reach the map. This used to be a trap —
#  the phrase in the query box read as two class names, Start picked whichever came
#  first, and the arm grasped the red cube and folded home reporting success. See
#  tests/test_task_routing.py, which pins that exact run.
#
# SAFETY: the arm relaxes (torque off) after IDLE_RELAX_S and wakes on the next
#  motion; a latched servo overload is cleared and retried at connect. Run under
#  supervise.py, which restarts this and camserver if either dies.
# GUEST: a Host-header gate exposes only a small allowlist over the public
#  Tailscale-Funnel host; everything else is admin-only. See ui/*.html.
#
# RUNNING IT
#  examples/mission_server/run_server.ps1   (detached; -Status, -Stop, -Port COMn)
#  Read TROUBLESHOOTING.md before debugging any "camera not found" — on this rig it
#  has twice been a host-side flag, not the hardware.
#
#  Environment:
#    RAX_ARM=<profile>     which arm (default so101; see robots/profiles/)
#    RAX_ARM_PORT=<port>   the arm's serial port, else the profile's default
#    RAX_PORT=<n>          this server's HTTP port (default 8484)
#    RAX_LEROBOT_SRC=<dir> a lerobot source checkout, if it is not pip-installed
#    RAX_YOLO_WEIGHTS=<f>  YOLO-World weights (default yolov8s-worldv2.pt)
#    GOOGLE_API_KEY=<k>    optional; enables the advisory vision checks
#    RAX_CAMSURV_URL / RAX_CAMSURV_PASSWORD   optional room camera at /stream2
#
#  lerobot is a HARDWARE dependency and is imported at connect time only (see
#  connect_hardware()), so this module imports, --help works and the tests run on a
#  machine without it. It used to be an absolute path to one developer's home
#  directory at module scope, which is why nobody else could run the file.
#
# RELATIONSHIP TO examples/legacy/stack_mission2.py
#  That is the original single-file server, kept unchanged and runnable so the two
#  can be compared on the same hardware rather than the old one replaced on trust.
#  The extracted pieces were verified bit-identical against it over a 447-case golden
#  grid (tests/test_extraction_parity.py); the knobs and /status payload are
#  unchanged. RAX_PORT=<n> runs this alongside it — but only one process at a time
#  can hold the arm's serial port and the camera.
#
# NOTE the hand-eye TF (handeye_tf.json) is rotationally wrong — table rays come
# out too shallow — so absolute ranges are compressed and the map's positions are
# approximate. Most of the localization care in here works around that: locating
# from ONE fixed pose (locate_from_survey) turns a varying error into a constant one,
# and the visual centring absorbs what is left.
import json, math, os, re, secrets, subprocess, sys, tempfile, threading, time
from typing import NamedTuple

import cv2
import numpy as np
import requests as pyrequests
from flask import Flask, Response, jsonify, request

# ---- lerobot: an optional HARDWARE dependency, resolved at connect time ---------
# This server drives the arm and the OAK-D through lerobot. That is a DRIVER
# dependency, not an algorithm one — everything above the hardware line is RAX — and
# every lerobot import now sits inside the function that needs it, so this module
# imports (and --help works, and the tests run) on a machine that has never heard of
# lerobot. See connect_hardware().
#
# RAX_LEROBOT_SRC points at a lerobot source checkout when it is not pip-installed.
# This used to be an absolute path into one developer's home directory, which is
# exactly why nobody else could run the file.
LEROBOT = os.environ.get("RAX_LEROBOT_SRC", "")
if LEROBOT:
    _src = os.path.join(LEROBOT, "src")
    sys.path.insert(0, _src if os.path.isdir(_src) else LEROBOT)

#: YOLO-World weights; ultralytics downloads them on first use if absent.
YOLO_WEIGHTS = os.environ.get("RAX_YOLO_WEIGHTS", "yolov8s-worldv2.pt")

OUT = os.path.join(tempfile.gettempdir(), "rax_stack_mission")  # debug-image dumps

# A vision model for the questions geometry cannot answer — see gemini_vision.py.
# None when there is no key or no SDK, and every call site checks, so the pick behaves
# exactly as it did before if it is absent. Built lazily in main() once say() works.
GEMINI = None
os.makedirs(OUT, exist_ok=True)

# ---- the arm, described as data ------------------------------------------------
# Everything that used to be a literal here now comes from a profile, so pointing
# this server at a different arm is a matter of writing one (URDF + extrinsics +
# which joint does what) rather than editing the algorithms. The measurements and
# the reasons behind them live in robots/profiles/so101.py; the aliases below keep
# the existing names so the rest of this file is unchanged.
#   RAX_ARM=<name> selects a profile; see robots/profiles/available_profiles().
from rax.robots.profiles import load_profile
from rax.perception.camera_geometry import (
    CameraGeometry, EyeInHand, FixedCamera, intrinsics_from_dict, parse_tf,
    tf_to_string)
from rax.perception.object_priors import (
    PRIORS, CLASS_META, COCO_CLASSES, TABLE_CLASSES)
from rax.perception.table_plane import Plane, fit_plane
from rax.manipulation.arms.ik_strategy import make_ik
from rax.manipulation.arms.kinematics import make_kinematics
from rax.manipulation.arms.motion import MotionLimits, quintic_waypoints
from rax.perception.locate import ApparentSizeLocalizer, PlaneRayLocalizer, rotate_xy
from rax.perception.measure import ObjectMeasurer, classify_shape
from rax.perception.handeye import (
    GraspSample, fit_to_known_points,
    HandEyeSample, fit_consistency, fit_reprojection, load_hand_eye, save_hand_eye)
from rax.models.detection.tracking import (
    AnchorTracker, Track, clipped_edges, table_ray_is_usable)
from rax.models.detection.detector_service import (
    DetectorConfig, DetectorService, box_iou)
from rax.mobility.slam.object_map import (
    ObjectMap, fit_rect_from_support, sup_bin, yaw_blend)
from rax.manipulation.approach import (
    ApproachConfig, approach_target, cap_reach, stage_step, stage_trim)
from rax.manipulation.approach.derive import (
    grasp_aim_offset_px,
    grasp_bias_m,
    align_tolerance_px, grasp_height, right_trim_for_visibility)
from rax.perception.selfcal import (
    MIN_SAMPLES, LocalizationModel, LocalizationSample, apply_to_config, diagnose,
    fit_localization)
from rax.manipulation.approach.visual_center import center_on_object
from rax.common.mission_state import Abort, MissionState

# Sibling module, not part of the published package: queueing strangers onto one
# shared robot is this demo's problem, not the pick stack's.
from guest_sessions import GuestConfig, GuestSessions
from gemini_vision import ADVISORY_ONLY, GeminiVision, MISS_REASONS
from gemini_vision import available as gemini_available

ARM = load_profile(os.environ.get("RAX_ARM", "so101"))

ARM_MOTORS = list(ARM.joint_names)     # was imported from lerobot's gaze_engine
TF = ARM.camera.extrinsics
GRASP_ROLL = ARM.gripper.grasp_roll_deg
WRIST_RENDER_OFFSET = ARM.gripper.render_offset_deg
VIEW = np.array(ARM.view_deg)
HOME = np.array(ARM.home_deg)
HAND_UV = ARM.gripper.hand_uv
GRIP_TIP_OFFSET_M = ARM.gripper.tip_offset_m
CAM_TIP_M = ARM.camera.cam_tip_m
ARM_PORT = ARM.port                    # the arm's serial port (was "COM4", 3 places)
PORT = int(os.environ.get("RAX_PORT", 8484))   # this server's HTTP port
CAMSURV = (os.environ.get("RAX_CAMSURV_URL", "http://127.0.0.1:5000"),
           os.environ.get("RAX_CAMSURV_PASSWORD", ""))

JOINT_RATE_MAX = ARM.joint_rate_max_dps   # deg/s per joint hard clamp

# gaze-engine approach: point at the target (direct-joint pixel P-control, no
# IK, so it can't swing) then step straight down the line of sight, decreasing
# the radius, toward the object's back-projected 3D point (radial-to-object —
# descends ONTO the cube instead of hovering above it).
Z_TABLE = ARM.table_z_m   # base-frame height of the object CENTRE; the sightline is
                          # intersected with THIS plane to localize. See the profile.
TARGET_SIZE_M = 0.03      # cube edge — pinhole range from bbox size (survives close range)

# Phase, rolling log and stop flag now come from rax.common.mission_state, which was
# written as a drop-in for the dict that used to live here: it subscripts, gets and
# updates like one, so the ~290 call sites below did not have to change. The aliases
# keep `say(...)`, `set_phase(...)` and `with lock:` reading the way they always have.
MS = MissionState()
state = MS
log = MS.log
lock = MS.lock
stop_flag = MS.stop_flag
say = MS.say
set_phase = MS.set_phase
checkpoint = MS.checkpoint

frame_jpeg = [None]
bus_lock = threading.RLock()   # Feetech bus is not thread-safe
mission_thread = [None]
latest_rgb = [None]            # for the YOLO thread


robot = None
kin = None
cam = None
T_ee_cam = parse_tf(TF)
fx = fy = cx0 = cy0 = 0.0

# ---- shared perception objects -------------------------------------------------
# The camera geometry (project / back-project / ray-to-plane / tip pixel) and the
# table plane now live in perception/, parameterized rather than reading globals.
# GEOM's pose provider resolves `kin` at call time, so it is usable from module
# scope even though the robot connects later in main(). A fixed (non-wrist) camera
# swaps EyeInHand for FixedCamera and everything downstream is unchanged.
GEOM = CameraGeometry(
    intrinsics_from_dict(
        dict(zip(("fx", "fy", "cx", "cy"), ARM.camera.intrinsics_fallback)),
        width=ARM.camera.width, height=ARM.camera.height),
    EyeInHand(lambda q: kin.forward_kinematics(q), T_ee_cam)
    if ARM.camera.eye_in_hand else FixedCamera(parse_tf(ARM.camera.extrinsics)),
)
FLOOR = Plane()          # the measured table surface; re-fitted by calibrate_floor

# Every tunable of the approach, in one object. Replaces two competing idioms for the
# same job (`global X` rebinds and the `X[0]` one-element-list trick) and gives the
# autotuner and the UI a real get/set-by-name API. See manipulation/approach/config.py.
CFG = ApproachConfig()

_IK = [None]


def ik_strategy():
    """The profile's IK strategy, built once the robot model exists.

    Rebuilt if `kin` is swapped (tests construct one after import), so there is no
    stale solver silently holding a different robot's kinematics.
    """
    if _IK[0] is None or _IK[0].kin is not kin:
        _IK[0] = make_ik(kin, ARM, grasp_pitches=GRASP_PITCH, standoff_m=STANDOFF_H)
    return _IK[0]


def _sync_geometry():
    """Push the current intrinsics + hand-eye into GEOM.

    Both are discovered late (intrinsics when the camera connects, the hand-eye
    whenever a calibration re-fits it), so every site that rebinds them calls this.
    """
    GEOM.set_intrinsics(intrinsics_from_dict(
        {"fx": fx, "fy": fy, "cx": cx0, "cy": cy0},
        width=ARM.camera.width, height=ARM.camera.height))
    if isinstance(GEOM.pose, EyeInHand):
        GEOM.pose.T_ee_cam = np.asarray(T_ee_cam, dtype=np.float64)

# The colour-anchor and CamShift trackers now live in models/detection/tracking.py.
red_tracker = AnchorTracker("red", GEOM)
green_tracker = AnchorTracker("green", GEOM)



# ---------------- parallel detection ----------------
# The throttled worker, query handling, NMS and the per-label trackers now live in
# models/detection/detector_service.py. This keeps the old function names as thin
# delegations so the ~40 call sites below are unchanged.
detector = None
DETECT = None              # DetectorService, built in main() once the detector loads
DETECT_CFG = DetectorConfig(
    period_s=2.5,
    nms_iou=0.55,          # two prompts routinely fire on the same object
    max_age_s=4.0,
    colour_min_frac=0.15,
    # the cube colours keep their dedicated HSV+anchor trackers, which are tighter
    # during close approach than the detector's refresh rate
    reserved_labels=("red cube", "green cube"),
)


def _query_labels():
    """Parse the active detection query into class labels, preserving order.

    Prefers what the detector is ACTUALLY running, then the expanded vocabulary, and
    only then the raw query box — which since reads_as_a_task() may hold a task phrase
    ("red on green") rather than a class list, and splitting that on commas would hand
    back one nonsense label.
    """
    if DETECT is not None:
        return DETECT.labels()
    q = (state.get("vocabulary") or state.get("query") or "red cube, green cube").lower()
    if reads_as_a_task(q) is not None:
        q = "red cube, green cube"
    return [p.strip() for p in q.split(",") if p.strip()]


def _colour_ok(rgb, label, xyxy):
    """Reject a box that is not the colour its LABEL names. Colourless labels pass."""
    try:
        from rax.models.detection.color_filters import (
            bbox_color_match_fraction, color_names_in_query)
    except Exception:
        return True
    cols = color_names_in_query(label)
    if not cols:
        return True
    try:
        return bbox_color_match_fraction(
            rgb, tuple(int(v) for v in xyxy), cols) >= DETECT_CFG.colour_min_frac
    except Exception:
        return True


def yolo_worker():
    DETECT.run_forever()


#: A strict-HSV blob smaller than this is a speck of glare or a shadow edge, not the
#: cube. Latching one anchors the tracker to noise and every later frame then "tracks"
#: it, so acquisition is refused below this and the stale lock is cleared.
CUBE_ACQUIRE_MIN_AREA_PX = 2500

#: A detector box older than this is not evidence about where the cube is NOW. The
#: detector runs at ~2.5s, so this allows roughly one missed cycle.
CUBE_DET_MAX_AGE_S = 4.0


# ---------------- vision-model assist, where the evidence says it helps ------------
#: Off with RAX_VLM_ASSIST=0. On by default because it only ever runs where the
#: classical detectors have ALREADY returned nothing — its alternative is not a working
#: locate, it is an abandoned one.
VLM_ASSIST = os.environ.get("RAX_VLM_ASSIST", "1") != "0"

#: Calls allowed per locate. The model costs 1.3-1.6s typical (7s worst seen), and the
#: survey takes SURVEY_READS frames — asking on every one would put a network round
#: trip in the control path. One or two is enough: if the object is in frame at all,
#: one look finds it.
VLM_BUDGET_PER_LOCATE = 2
_vlm_left = [0]


def vlm_budget_reset(n: int = VLM_BUDGET_PER_LOCATE):
    _vlm_left[0] = int(n) if VLM_ASSIST else 0


def vlm_track(rgb, label):
    """A Track from the vision model, or None. Only for when the detectors found none.

    MEASURED, on this rig's own saved frames, which is why it sits here and not at the
    hover. Where the object is genuinely in the picture the model's box centre agrees
    with the strict-HSV blob to within ~5px on 5 of 8 frames — and on the one frame
    where the green cube photographed TEAL from a steep angle, HSV latched onto an
    unrelated blob 211px away while the model put the box on the actual cube.

    It was tried at the centring hover first and did NOT help: of the eight frames the
    servo saved after reporting "object not in view", seven contain no cube at all —
    the model correctly answers "there is no green cube visible in the image, only a
    blurry surface and part of the gripper". That failure is the camera being aimed
    somewhere else, and no detector can fix a pose error. Hence: upstream, at the
    survey, while the object is still in frame.
    """
    if not VLM_ASSIST or GEMINI is None or _vlm_left[0] <= 0:
        return None
    _vlm_left[0] -= 1
    lab = str(label).strip().lower()
    if lab in ("red", "green", "blue", "yellow"):
        lab = f"{lab} cube"          # a bare colour is a poor thing to ask a model for
    try:
        fix = GEMINI.locate_object(rgb, lab)
    except Exception as e:
        say(f"vlm: locate failed ({type(e).__name__}: {e})")
        return None
    if not fix.ok:
        say(f"vlm: no fix for '{lab}' — {fix.reason[:110]}")
        return None
    x1, y1, x2, y2 = (float(v) for v in fix.bbox_xyxy)
    say(f"vlm: located '{lab}' at ({fix.uv[0]:.0f},{fix.uv[1]:.0f})px "
        f"conf {fix.confidence:.0%}{' CLIPPED' if fix.clipped else ''} "
        f"[{fix.latency_s:.1f}s] — the detector had nothing")
    return Track((float(fix.uv[0]), float(fix.uv[1])), (x1, y1, x2, y2),
                 int(max(x2 - x1, 0) * max(y2 - y1, 0)), bool(fix.clipped), time.time())

def _find_cube(tracker, label, rgb, T_base_cam=None):
    """Acquire-or-track one of the colour cubes. Three ways in, tried in order:

    1. CONTINUITY. If the tracker already holds a fix (or a 3D anchor to reproject),
       search near it. Cheapest, and the only one tight enough during close approach.
    2. THE DETECTOR. Seed from the open-vocabulary box, which sees the cube in poses
       and lighting the strict HSV bands miss.
    3. A FULL-FRAME STRICT MASK. Catches what the detector misses -- edge-clipped
       slivers especially -- and is the only path that works before the detector has
       ever run.

    WHY THIS IS SHARED. ``find_green`` used to be a bare one-liner, ``return
    green_tracker.track(...)``: continuity only, no detector seed, and no area gate,
    while red had all three. So green could not re-acquire after anything reset it --
    and ``/calibmount`` resets the tracker before its sweep, which is precisely when
    re-acquisition is the whole job. Observed: the green cube plainly in frame,
    ``p_green`` empty, the mount calibration collecting 6 views instead of 10. Two
    cube colours should not have two different levels of robustness by accident.
    """
    if tracker.last is not None or tracker.p_anchor is not None:
        tr = tracker.track(rgb, T_base_cam)
        if tr is not None:
            return tr
    if DETECT is not None:
        inst, t = DETECT.instances(label)
        if inst and time.time() - t < CUBE_DET_MAX_AGE_S:
            x1, y1, x2, y2 = inst[0]["xyxy"]
            tracker.last = Track(((x1 + x2) / 2, (y1 + y2) / 2), tuple(inst[0]["xyxy"]),
                                 int((x2 - x1) * (y2 - y1)), False, time.time())
            return tracker.track(rgb, T_base_cam)
    tr = tracker.track(rgb, None)          # full-frame strict mask
    if tr is not None and tr.area_px >= CUBE_ACQUIRE_MIN_AREA_PX:
        return tr
    tracker.last = None                    # too small: don't latch a speck
    return None


def find_red(rgb, T_base_cam=None):
    return _find_cube(red_tracker, "red cube", rgb, T_base_cam)


def find_green(rgb, T_base_cam=None):
    return _find_cube(green_tracker, "green cube", rgb, T_base_cam)


def find_labels(rgb, label, max_age=4.0):
    """EVERY current instance of a label, as Tracks."""
    return DETECT.find_all(rgb, label, max_age)


def find_label(rgb, label, T_base_cam=None):
    """Generic finder for any query label.

    The two cube colours keep their dedicated HSV+anchor trackers, which are tighter
    during close approach than the detector's refresh rate. Everything else is tracked
    frame-to-frame by the service.
    """
    label = str(label).strip().lower()
    if label == "red cube":
        return find_red(rgb, T_base_cam)
    if label == "green cube":
        return find_green(rgb, T_base_cam)
    return DETECT.find(rgb, label)


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
            # THE EXACT PIXEL the tracker is locked to — same marker the generic
            # (non-cube) path draws below. This loop is a SEPARATE code path (the
            # dedicated red/green HSV trackers) that used to have no marker at
            # all, which is why "add an X" appeared to do nothing when the query
            # was the default "red cube, green cube" — that query never reaches
            # the generic block the marker was first added to.
            cv2.drawMarker(img, (int(round(tr.uv[0])), int(round(tr.uv[1]))), color,
                           cv2.MARKER_TILTED_CROSS, 18, 2)
            # label with the DISTANCE, from apparent size: range = f * edge / width.
            # Needs only the lens focal length and the cube's real size, so it stays
            # honest regardless of the camera-mount numbers.
            w_px = float(max(4, x2 - x1))
            rng_cm = fx * PRIORS.fallback_edge_m / w_px * 100.0
            cv2.putText(img, f"{name} {rng_cm:.0f}cm", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # EVERY OTHER YOLO DETECTION. Until now the overlay drew boxes for the two
    # colour trackers and nothing else, so a pen (or cup, or anything) could be
    # detected at high confidence and still show NO BOX - which looks exactly like
    # "the detector cannot see it". Draw whatever the detector currently reports.
    with lock:
        dets = dict(DETECT.latest.dets)
        d_age = time.time() - DETECT.latest.t
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
            if all(box_iou(box, k[2]) < 0.45 for k in kept):
                kept.append((conf, lbl, box))
        for conf, lbl, box in kept:
            # Prefer the LIVE PIXEL-TRACKED box over the raw YOLO cache: the cache
            # only moves every ~2.5s (yolo_worker's cycle) and drawing it directly
            # is what made the overlay look frozen-then-jumping as the arm moved.
            # publish() only READS pt.last here, never calls track() itself - the
            # control loop (find_label) is what steps CamShift each tick.
            pt = DETECT.trackers.get(lbl)
            tracked = pt.last if (pt is not None and pt.last is not None
                                  and now - pt.last.t < 0.5) else None
            x1, y1, x2, y2 = (int(v) for v in (tracked.bbox_xyxy if tracked else box))
            col = (255, 210, 60) if tracked else (0, 200, 255)  # cyan=tracked, amber=raw
            # THE TAG, ON THE VIEW: fill in the ACTUAL pixels driving the track
            # (pt.pixel_mask, Otsu-thresholded back-projection — see track()),
            # not just its bounding box. This is what "which pixels are being
            # tracked" looks like frame to frame, and it moves/reshapes with the
            # object instead of sitting fixed to a rectangle.
            if tracked and pt.pixel_mask is not None:
                mh, mw = pt.pixel_mask.shape
                mx, my = pt.pixel_origin
                if 0 <= mx and 0 <= my and mx + mw <= img.shape[1] and my + mh <= img.shape[0]:
                    roi = img[my:my + mh, mx:mx + mw]
                    overlay = np.full_like(roi, col)
                    blended = cv2.addWeighted(roi, 0.55, overlay, 0.45, 0)
                    m3 = pt.pixel_mask[:, :, None]
                    roi[:] = np.where(m3, blended, roi)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            # THE EXACT PIXEL the tag is anchored to — an X at tr.uv, the same
            # point every downstream calculation (bearing, range, servo
            # centering) actually uses. The box shows the extent; this shows the
            # single coordinate that matters.
            ux, uy = tracked.uv if tracked else ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            cv2.drawMarker(img, (int(round(ux)), int(round(uy))), col,
                           cv2.MARKER_TILTED_CROSS, 18, 2)
            w_px = float(max(4, x2 - x1))
            rng_cm = fx * class_size_m(lbl) / w_px * 100.0
            tag = "" if tracked else " (raw)"
            txt = f"{lbl} {conf:.2f} {rng_cm:.0f}cm{tag}"
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


_GOTO_LIMITS = MotionLimits.from_profile(ARM)


def goto_smooth(target, settle=0.15, step=2.0):
    """Transit move with a quintic S-curve profile: zero start/end velocity and
    acceleration. This removes the base jump at motion start/stop.

    The old send_joint_target_smoothly moved at a fixed step per tick, which is
    just a velocity cap — it still commanded abrupt starts and stops. Here the
    velocity ramps up and down smoothly, so the camera/gripper "head" glides.

    The profile maths lives in manipulation/arms/motion.py; this owns the send loop
    and its real-time pacing.
    """
    joints, _r, obs = observe(overlay=False)
    gp = float(obs.get("gripper.pos", 50.0))
    # step=2.0 was the old default degrees/tick; use it as a speed scale.
    limits = _GOTO_LIMITS.scaled(float(step) / 2.0)
    waypoints, _T = quintic_waypoints(joints, target, limits)

    t0 = time.time()
    for k, q_cmd in enumerate(waypoints):
        send_joints(q_cmd, gripper=gp)
        # sleep to maintain the command rate, accounting for command overhead
        to_sleep = t0 + (k + 1) * limits.dt_s - time.time()
        if to_sleep > 0:
            time.sleep(to_sleep)

    time.sleep(settle)


def T_cam_of(joints):
    return GEOM.T_base_cam(np.asarray(joints, dtype=np.float64))


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
    return GEOM.backproject(uv, z_m, T_base_cam)


def project_base(p_base, T_base_cam):
    """Base-frame point -> pixel. The inverse of locate_3d; the ground truth test
    for the hand-eye TF."""
    return GEOM.project(p_base, T_base_cam)


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
    return GEOM.tip_pixel(joints)


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
    r_tri = float(np.hypot(p[0], p[1]))
    if max(gaps) > 0.05 or not (MAP_R_MIN < r_tri < MAP_R_MAX) or not (-0.12 < p[2] < 0.15):
        raise Abort(f"{label}: triangulation implausible")
    tracker.p_anchor = p.copy()
    tracker.anchor_t = time.time()
    return p


def close_with_current(step=None, delay=None):
    """Close the gripper in small increments, watching the servo current, and
    stop the instant it rises (torque change = fingers on the object). Smaller
    step / longer delay = the slow, gentle close the user asked for.

    The thresholds come from the profile's GripperProfile, so a different gripper
    (different gearing, different current scale) declares its own.
    """
    g = ARM.gripper
    step = g.close_step_pct if step is None else step
    delay = g.close_delay_s if delay is None else delay
    idle = [c for c in (gripper_current() for _ in range(5)) if c is not None]
    i_idle = float(np.mean(idle)) if idle else 0.0
    pct = g.open_pct
    while pct > g.closed_pct:
        checkpoint()
        pct -= step
        joints = observe(overlay=True)[0]
        send_joints(joints, gripper=pct)
        time.sleep(delay)
        c = gripper_current()
        if c is not None and abs(c - i_idle) >= g.contact_current_delta:
            # firmer squeeze — ΔI=1.8 holds slipped the cube during transit
            send_joints(joints, gripper=max(0.0, pct - g.squeeze_extra_pct))
            time.sleep(0.18)
            return True, i_idle
    # Closed on air: DO NOT stay stalled shut (that's what tripped the servo's
    # overload protection earlier) — relax to a neutral opening.
    send_joints(observe(overlay=False)[0], gripper=g.relax_on_miss_pct)
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
# Localization push-out lives on CFG now (see manipulation/approach/config.py).


def push_out_radial(p):
    """Move a base-frame point radially OUTWARD (away from base z-axis) by CFG.push_out_m.
    Applied to EVERY localization (initial lock AND every approach refine) so the
    correction is consistent — otherwise a raw re-measure would drag the target back
    inward and undo the push. See PUSH_OUT."""
    p = np.asarray(p, np.float64).copy()
    r = float(np.hypot(p[0], p[1]))
    push = float(CFG.push_out_m)
    if r > 1e-3 and push != 0.0:
        p[0] += p[0] / r * push
        p[1] += p[1] / r * push
    return p


def ray_to_table(uv, T_base_cam, z_plane=None):
    """Intersect the pixel's back-projected sightline with the table plane. The
    plane height is TABLE_Z[0], which locate_on_table MEASURES from the bbox
    pinhole range rather than assuming."""
    z = TABLE_Z[0] if z_plane is None else float(z_plane)
    return GEOM.ray_to_plane(uv, T_base_cam, z)


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
MAP_R_MIN = ARM.reach_min_m
MAP_R_MAX = ARM.reach_max_m
# How far the fingertip can actually be DRIVEN, from the profile's measured
# envelope. This was hardcoded as 0.42 in two places while the profile said 0.55
# and the arm measures 0.47 -- three different numbers, none of them right. The
# jog cap was the tightest, which is why jogging out flat stopped 5cm short of
# the arm's real limit and looked like 'it will not straighten all the way'.
GRASP_R_MAX = ARM.reach_grasp_max_m
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
# The bird's-eye object map. Its association rules, the support-ring footprint fit
# and the consolidation pass live in mobility/slam/object_map.py; the callables below
# are injected because they depend on things the map does not own — the active query,
# and how a measured footprint maps to a shape name. They are wrapped in lambdas
# because they are defined further down this file.
WORLD = ObjectMap(
    merge_m=W2D_MERGE,
    ttl_s=MAP_TTL_S,
    may_merge_labels=lambda a, b: _may_merge_labels(a, b),
    classify_shape=classify_shape,
    prior_shape=lambda label: class_meta(label)["shape"],
    log=say,
)
w2d_lock = WORLD.lock


_rotate_xy = rotate_xy      # now perception/locate.py


def _correct_xy(xy):
    """Apply the live polar corrections to a localization that did NOT come from the
    localizer — the silhouette solve (measure_object) and the bbox-to-table ray.

    ``obj_xy_2d`` needs none of this: localizers() pushes range_scale and
    bearing_offset_deg onto the strategies and ApparentSizeLocalizer applies both
    internally. The other two position solves bypass it entirely, and until now they
    were handed only the BEARING — so a fitted range_scale moved the map's apparent-size
    fixes and left its silhouette fixes exactly where they were, in the same map, at the
    same time. Bearing was already applied uniformly here; this makes range match.

    Worth being explicit about the physics, because "they measure range differently" is
    a real objection: the silhouette solve gets position by intersecting the table
    plane, so a proportional range error is not its natural failure mode the way it is
    for apparent-size ranging. But selfcal's model is defined on the OUTPUT of
    localization — ``r_true = range_scale * r_observed + push_out_m``, whatever produced
    r_observed — and it is fitted from samples drawn through whichever path the pick
    happened to use. A correction that lands on some of those paths and not others
    cannot be inverted, which is what :func:`_selfcal_uncorrect` has to do to keep the
    fit absolute. Uniform application is what makes the model mean anything.

    push_out is deliberately NOT applied here: run_mission applies it once, at the
    single point where the pick's target is decided.
    """
    return LocalizationModel(range_scale=float(CFG.range_scale), push_out_m=0.0,
                             bearing_offset_deg=float(CFG.bearing_offset_deg)).apply(xy)


_LOC = [None]


class Localizers(NamedTuple):
    """The two range strategies, named so neither can be taken for the other.

    They used to come back as a bare 2-tuple, unpacked positionally at each call site
    — while rax.grasp.localizers() returns its pair in the OPPOSITE order (plane
    first, because with a fixed camera the plane solve is the stronger estimator).
    Two functions of the same name, same repo, reversed contract: nothing enforced
    it and nothing warned. Fields make the ordering irrelevant.
    """

    apparent: object      # ApparentSizeLocalizer — range from the object's size prior
    plane: object         # PlaneRayLocalizer — range from where its ray meets the table


def localizers():
    """The localization strategies, with the live-tunable corrections refreshed.

    RANGE_SCALE and MAP_BEARING_OFFSET_DEG are dialled from the UI against the 3D
    view, so they are pushed on every call rather than captured at construction.
    """
    if _LOC[0] is None:
        reach = (MAP_R_MIN, MAP_R_MAX)
        _apparent = ApparentSizeLocalizer(GEOM, PRIORS, reach_m=reach)
        # THE PINHOLE RELATION GIVES DEPTH, NOT RANGE. Walking it along the sightline
        # puts an off-axis object too close by cos(off-axis angle) — always inward,
        # growing with the angle: ~10mm at 20cm, ~90mm at 45cm. The approach
        # deliberately keeps the object off-centre, so this is live on every pick, and
        # a radial push-out fudge is what was compensating for it.
        #
        # Seen directly with two cubes: the near-axis green cube mapped at 20.2cm from
        # the camera against an observed 20cm, while the off-axis red cube mapped at
        # 14.8cm against an observed 26cm — 11cm inward, and enough to report the two
        # in the WRONG ORDER. RAX_AXIAL_DEPTH=0 restores the old behaviour.
        _apparent.axial_depth = os.environ.get("RAX_AXIAL_DEPTH", "1") != "0"
        _LOC[0] = Localizers(
            apparent=_apparent,
            # The table solve is used by callers that judge the raw point themselves,
            # so it does not additionally gate on the workspace.
            plane=PlaneRayLocalizer(GEOM, PRIORS, reach_m=reach, z_plane=TABLE_Z0,
                                    gate_reach=False),
        )
    for loc in _LOC[0]:
        loc.range_scale = float(CFG.range_scale)
        loc.bearing_offset_deg = float(CFG.bearing_offset_deg)
    return _LOC[0]


def obj_xy_2d(bbox, T_cam, z_m=None, label=None):
    """Where the object is, base-frame (x, y) — see perception/locate.py.

    Returns (xy | None, range_m, assumed_size_m). The strategy, including the
    elongated-object long-axis branch and why it exists, lives with the code there.
    """
    apparent = localizers().apparent
    fix = apparent.locate(bbox, T_cam, label=label, z_m=z_m)
    return (fix.xy if fix.ok else None), fix.range_m, fix.size_m



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



_yaw_blend = yaw_blend      # circular mean of two axis angles; now object_map.py


def world2d_update(label, xy, stereo, w_m, d_m, h_m, shape, yaw, measured,
                   across_m=None, u_deg=None):
    """Fold one observation of one object into the map — ObjectMap.update.

    across_m / u_deg are one caliper reading of the footprint (its width along the
    across-view direction u_deg) — the only footprint fact a single view actually
    establishes. The association rules, and the two bugs that shaped them, are
    documented in mobility/slam/object_map.py.
    """
    return WORLD.update(label, xy, stereo=stereo, w_m=w_m, d_m=d_m, h_m=h_m,
                        shape=shape, yaw=yaw, measured=measured,
                        across_m=across_m, u_deg=u_deg)


#: How far the silhouette solve's range may differ from the object's apparent size
#: before it is disbelieved, as a ratio. Outside this band the two disagree about
#: something basic and the transform-free one wins.
MEASURE_RANGE_BAND = (0.72, 1.38)

#: How many silhouette solves this gate has dropped, reported in /status.
_measure_rejects = [0]


def apparent_range_m(bbox_xyxy, label):
    """Range implied by how big the object LOOKS, from the class prior. Or None.

    ``fx * real_width / pixel_width`` — no hand-eye transform, no table plane, no arm
    pose. That is the point: it is wrong only if the prior is wrong or the box is, so
    it makes an honest referee for the solves that do depend on all of those.
    """
    lab = str(label).strip().lower()
    # Only a MEASURED prior may referee. PRIORS.size_m falls back to a default edge for
    # anything it does not know, and judging a laptop or a bottle against a cube-sized
    # default would reject every honest measurement of it and empty the map of exactly
    # the objects the class table does not cover.
    known = next((c for c in (lab, f"{lab} cube") if c in CLASS_META), None)
    if known is None:
        return None
    x1, _y1, x2, _y2 = (float(v) for v in bbox_xyxy)
    w_px = x2 - x1
    size = float(PRIORS.size_m(known) or 0.0)
    if w_px < 4.0 or size <= 0.0 or GEOM.fx <= 0.0:
        return None
    return float(GEOM.fx * size / w_px)


def measured_range_is_credible(m, bbox_xyxy, label):
    """Does the silhouette solve agree with the object's own apparent size?

    WHY THIS GATE EXISTS. The map preferred measure_object's position over the
    apparent-size fallback whenever the box was fully visible, and on this rig that
    solve degrades badly at the shallow viewing angles the arm actually surveys from.
    It does not degrade toward noise; it degrades toward a confident wrong number, and
    because sense_2d folds every frame in, the map AVERAGES those.

    Observed with two cubes on the table: the operator could see green nearer than red
    and the detector's own overlay agreed (green 20cm, red 26cm), while the map had red
    at 35cm and green at 42cm — both far too distant, and their ORDER swapped. A map
    that cannot say which of two objects is nearer cannot be driven on, whatever its
    RMS is.

    Apparent size is the referee because it shares no machinery with the thing it is
    judging: no hand-eye transform, no table plane, no arm pose. When the two disagree
    by more than a third, the one that depends on the known-wrong rotation is the one
    to drop, and sense_2d falls through to the apparent-size path it already has.
    """
    if m is None:
        return False
    ref = apparent_range_m(bbox_xyxy, label)
    if ref is None or ref <= 0.0:
        return True                     # no referee available; nothing to object with
    try:
        rng = float(m["rng_m"])
    except (KeyError, TypeError, ValueError):
        return True
    if rng <= 0.0:
        return False
    ratio = rng / ref
    return MEASURE_RANGE_BAND[0] <= ratio <= MEASURE_RANGE_BAND[1]

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
            if m is not None and not measured_range_is_credible(m, tr.bbox_xyxy, label):
                # Disagrees with the object's own apparent size: drop it and use the
                # transform-free path below. See measured_range_is_credible.
                _measure_rejects[0] += 1
                m = None
            if m is not None:
                xy = _correct_xy(m["xy"])
                world2d_update(label, xy, m["rng_m"], m["w_m"], m["d_m"], m["h_m"],
                               m["shape"], m["yaw_deg"], True,
                               across_m=m["across_m"],
                               u_deg=m["u_deg"] + CFG.bearing_offset_deg)
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
                    cand = _correct_xy(p3[:2])
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
    # forget objects that have not been seen in a while: the map should describe the
    # table as it is, not as it once was
    WORLD.prune()
    now = time.time()
    with w2d_lock:
        out = []
        for t, o in WORLD.objs.items():
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
                        # "no stereo reading" arrives as either None or NaN depending on
                        # which path created the entry; both must serialize, not crash
                        # the viewer. (`x == x` is the NaN test.)
                        "stereo_cm": (round(o["stereo"] * 100)
                                      if o["stereo"] is not None and o["stereo"] == o["stereo"]
                                      else None),
                        "n": o["n"], "age": round(now - o["t"], 1)})
        return out


def _apply_query_now(q, timeout=6.0):
    """Switch the detector vocabulary and WAIT for it to take effect.

    The model may only be mutated on the thread that runs it (calling into it from
    Flask crashes the process), so the change is queued and the detector thread picks
    it up on its next cycle. Scanning before that lands would sweep the whole table
    with the OLD vocabulary and find nothing.
    """
    if DETECT.wait_for_query(q, timeout):
        return True
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
        o = WORLD.objs.get(tag)
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
    plane = localizers().plane
    fix = plane.locate(tr.bbox_xyxy, T_base_cam, uv=tr.uv)
    if not fix.ok:
        return None, None
    return np.array([fix.xy[0], fix.xy[1], fix.z_m], dtype=np.float64), fix.size_m


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
        f"| raw r={r_raw*100:.1f}cm + pushed out {CFG.push_out_m*100:.0f}cm "
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
        d = load_hand_eye(TF_FILE)
    except ValueError as e:
        say(f"hand-eye: ignoring {e}")
        return None
    if d is None:
        return None
    T_ee_cam = parse_tf(d["tf"])
    _sync_geometry()
    return d


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

    # The consistency fit lives in perception/handeye.py; this owns the pan sweep.
    fit = fit_consistency(
        [HandEyeSample(T_ee, uv) for T_ee, uv in samples],
        GEOM, tip_uv=HAND_UV, T_seed=T_ee_cam, z_plane=TABLE_Z0,
        tip_offset_m=GRIP_TIP_OFFSET_M)
    say(f"multi-view spread: {fit.before['spread_m']*100:.1f}cm -> {fit.spread_m*100:.1f}cm "
        f"(how much the same cube moves between viewpoints)")
    if not fit.converged:
        raise Abort(f"multi-view calibration {fit.reason}")
    T_ee_cam = fit.T_ee_cam
    _sync_geometry()
    save_hand_eye(TF_FILE, fit)
    say(f"mount CALIBRATED (multi-view) -> {fit.tf}")
    set_phase("IDLE",
              f"mount calibrated - same cube now agrees to {fit.spread_m*100:.1f}cm across views")



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

    # The fit itself lives in perception/handeye.py — this owns the motion that
    # collected the views, which is the part that needs a robot.
    T_ee0 = np.asarray(kin.forward_kinematics(samples[0][0]))
    fit = fit_reprojection(
        [HandEyeSample(np.asarray(kin.forward_kinematics(j)), uv) for j, uv in samples],
        GEOM, tip_uv=HAND_UV, T_seed=T_ee_cam,
        target_seed=_measure_point(tr0, T_ee0 @ T_ee_cam), cam_tip_m=CAM_TIP_M)

    say(f"hand-eye BEFORE: cube reprojection RMS={fit.before['rms_px']:.0f}px  "
        f"fingertip off by {fit.before['tip_gap_px']:.0f}px  ({fit.n_views} views)")
    say(f"hand-eye AFTER:  cube reprojection RMS={fit.rms_px:.0f}px  "
        f"fingertip off by {fit.tip_gap_px:.0f}px")
    if not fit.converged:
        raise Abort(f"hand-eye: {fit.reason} — TF NOT changed.")

    save_hand_eye(TF_FILE, fit)
    T_ee_cam = fit.T_ee_cam
    _sync_geometry()
    p = fit.target_p
    say(f"hand-eye CALIBRATED -> {fit.tf}")
    say(f"  (saved to {os.path.basename(TF_FILE)}; loaded automatically on every restart)")
    say(f"  cube now solves to r={np.hypot(p[0], p[1])*100:.1f}cm "
        f"ang={math.degrees(math.atan2(p[1], p[0])):+.0f}deg z={p[2]*100:.1f}cm")
    set_phase("CALIB", f"hand-eye fixed — reprojection {fit.rms_px:.0f}px")
    return fit.T_ee_cam


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


# ---------------- hand-eye from grasps: the only ground truth the rig has ----------
#: (uv, T_base_ee) of the object, observed just before the descent. Paired at contact
#: with FK's answer for where it actually was, and written to GRASP_FILE.
_grasp_obs = [None]
GRASP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_samples.jsonl")


def note_grasp_observation(finder, label):
    """Remember where the object APPEARED, just before committing to the descent.

    Half of a calibration sample. The other half arrives when the jaws close: the
    object is then between the fingertips and forward kinematics says where those are,
    which owes nothing to the camera, the hand-eye transform or the table plane.

    That independence is what the existing fitters lack. One solves for the transform
    AND the target together and drifts along a flat direction; the other asks only that
    the viewpoints agree, and on this rig they agreed to 0.3cm on the wrong place.
    A grasp is the one moment the robot learns where something REALLY was.
    """
    _grasp_obs[0] = None
    try:
        j, rgb, _ = observe(overlay=False)
        tr = finder(rgb, T_cam_of(j))
        if tr is None or tr.clipped:
            return
        T_ee = np.asarray(kin.forward_kinematics(np.asarray(j, np.float64)), np.float64)
        _grasp_obs[0] = (tuple(float(v) for v in tr.uv), T_ee.tolist(), str(label))
    except Exception:
        _grasp_obs[0] = None


def record_grasp_sample(p_base):
    """Pair the remembered pixel with where the grasp proved the object was."""
    obs = _grasp_obs[0]
    _grasp_obs[0] = None
    if obs is None:
        return False
    uv, T_ee, label = obs
    try:
        with open(GRASP_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "label": label, "uv": list(uv),
                                "T_base_ee": T_ee,
                                "p_base": [float(v) for v in p_base]}) + "\n")
    except Exception as e:
        say(f"handeye: could not record the grasp sample ({type(e).__name__}: {e})")
        return False
    return True


def load_grasp_samples():
    """Every recorded (pixel, pose, true position) triple. Skips unreadable lines."""
    out = []
    try:
        with open(GRASP_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    out.append(GraspSample(np.asarray(d["T_base_ee"], np.float64),
                                           np.asarray(d["uv"], np.float64),
                                           np.asarray(d["p_base"], np.float64)))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return out

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
    return ik_strategy().plan_pitch(p_obj, q_seed)


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
# Fallback grasp height, used only when the object's height is unknown. When the map
# HAS a height, grasp_z_for() derives the grip point from it instead — see
# manipulation/approach/derive.py. The two agree exactly on the 5.08cm cube this
# constant was tuned on; they differ where the constant was silently wrong (it closes
# ABOVE a 0.9cm phone and grips a 23cm bottle at its base).
PICK_GRASP_Z = 0.015
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
PLACE_OPEN_PCT = ARM.gripper.place_open_pct   # releases without flicking the object
PLACE_RETREAT_M = 0.09     # straight-up retreat after releasing

# Detection resolution. 320 is what the approach trims (CFG.aim_du_px, CFG.right_trim_m,
# CFG.back_m) were tuned against, and range comes straight from bbox width
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
    if not held:
        # Outside the lock: clear_picked_flags takes w2d_lock, and nesting the two the
        # other way round is how a deadlock gets built.
        clear_picked_flags()


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
FLOOR_PROBE_STEP = 0.0025      # descend in 2.5 mm bites - gentle enough not to
                               # slam the servos or trip the overload latch
FLOOR_PROBE_DROP = 0.055       # give up after this much descent from the start
FLOOR_FOLLOW_MIN = 0.45        # measured/commanded travel below this = blocked
FLOOR_LOAD_RISE = 90           # raw Present_Load rise over baseline = pushing
FLOOR_RETREAT = 0.020          # lift this much after each touch
FLOOR_GRASP_CLEAR = 0.012      # grasp this far ABOVE the measured floor


def floor_z(x, y):
    """Table height in base z at (x, y), from the calibrated plane."""
    return FLOOR.z(x, y)


def load_floor_plane():
    try:
        d = FLOOR.load(FLOOR_FILE)
    except ValueError as e:
        say(f"floor: ignoring {e}")
        return None
    if d is None:
        say(f"floor: no calibration yet — assuming z={FLOOR.c*100:.1f}cm and level. "
            f"Press 'Calibrate floor' to measure it.")
        return None
    say(f"floor: calibrated plane loaded — {FLOOR.describe()}  "
        f"(tilt {d.get('tilt_deg', 0):.2f}deg, rms {d.get('rms_mm', 0):.1f}mm, "
        f"fitted {d.get('fitted', '?')})")
    return d


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
    try:
        res = fit_plane(pts)
    except ValueError as e:
        raise Abort(f"floor: {e}")
    a, b, c = res.plane.a, res.plane.b, res.plane.c
    rms, tilt = res.rms_m, res.tilt_deg
    FLOOR.set(a, b, c)          # in place: every holder of FLOOR sees the new surface
    d = res.to_dict()
    try:
        # Plane.save writes a/b/c itself; pass only the fit metadata alongside.
        FLOOR.save(FLOOR_FILE, **{k: v for k, v in d.items() if k not in ("a", "b", "c")})
    except Exception as e:
        say(f"floor: could not save ({e})")
    say(f"floor plane: {FLOOR.describe()}   "
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
# The per-class size/shape priors, the COCO name list and the tabletop subset now
# live in perception/object_priors.py and are imported at the top of this file.
# Objects bigger than this in any footprint dimension cannot be on this table —
# used to reject a nonsense measurement, not to reject the detection.
def class_meta(label):
    """Prior for a label: {shape, w_m, d_m, h_m}. Unlisted labels get a cube guess."""
    return PRIORS.meta(label)


def class_size_m(label):
    """Characteristic width for apparent-size ranging (what the bbox width maps to)."""
    return PRIORS.size_m(label)


def class_height_m(label):
    """Vertical extent above the table — used for hover/grasp height."""
    return PRIORS.height_m(label)


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
# The monocular silhouette metrology now lives in perception/measure.py — what one
# view can and cannot establish, and every gate that rejects a bad solve, is
# documented with the code there.
MEASURER = ObjectMeasurer(GEOM, PRIORS, reach_m=(MAP_R_MIN, MAP_R_MAX),
                          table_z=TABLE_Z0)
MEAS_STATS = MEASURER.stats     # surfaced in /status as measure_stats


def measure_object(rgb, bbox, T_base_cam, label, z_plane=None):
    """Measure an object's position, footprint, height and yaw from ONE frame."""
    return MEASURER.measure(rgb, bbox, T_base_cam, label, z_plane)


# ---- fusing per-bearing caliper readings into a footprint ----
# Each view measures the footprint's width along ONE direction (its across-view
# axis). That is the footprint's SUPPORT WIDTH along that direction, and a convex
# shape is determined by its support widths — so a scan sweep, which sees each
# object from a spread of bearings, measures the whole footprint between them.
# SUP_BINS / SUP_MIN_BINS now live with the fit in mobility/slam/object_map.py


_sup_bin = sup_bin                              # now mobility/slam/object_map.py
_fit_rect_from_support = fit_rect_from_support  # least-squares footprint rectangle


# Lateral aim trim, in pixels, applied to the fingertip aim point. The gripper was
# consistently ending up LEFT of the cube. Moving the aim point LEFT (negative)
# makes the arm travel FURTHER RIGHT before it thinks it is lined up, because
# swinging the camera right slides the cube left in the picture. If it now
# overshoots to the right, make this less negative; if still left, more negative.
# The arm is still landing left, so push the aim point further left.
# CFG.aim_du_px / CFG.aim_dv_px live on CFG now.
# Global scale on computed range. Increase (>1.0) to push mapped objects FURTHER
# OUT and spread them apart; decrease (<1.0) to pull them closer together. This
# is a coarse calibration knob for when the apparent-size / stereo depth numbers
# are consistently off in scale.
# Rotate all localized (x, y) positions in the base horizontal plane. Positive
# = counter-clockwise. Use this when the camera shows objects on opposite sides
# but the map clusters them on one side (a heading/yaw error in the hand-eye).


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


def _refix_here(finder, label=None, tries=3):
    """Re-locate the object from wherever the arm is standing NOW.

    Returns ``(xy, None)`` or ``(None, why)``. The reason is half the point: "the
    detector never saw it" and "it is half out of frame" are different problems and
    the approach reacts to them differently, so a caller handed a bare None could
    neither choose nor log anything true about what happened.

    Corrected the SAME way as the initial fix — bearing offset (inside the localizer),
    then push-out. A raw re-measure would drag the target back inward and undo the
    correction, and because the refine is accepted out to max_refine_jump_m it would do
    so silently.
    """
    tr = _cube_track(finder, tries=tries)
    if tr is None:
        return None, "the detector did not find it from here"
    j, rgb, _ = observe()
    if tr.clipped:
        # WHICH EDGE IS CLIPPED MATTERS — the same conclusion sense_2d reached, applied
        # here too. Refusing every clipped box threw away the majority of the approach's
        # re-measures: a real pick logged "half out of frame" on two of its three checks
        # and drove the whole approach on the original estimate, which was 5.2cm out.
        #
        # A clipped box breaks APPARENT-SIZE ranging, because that reads distance from
        # the box's width and a cut-off box is narrower than the object. It does not
        # break the TABLE RAY, which reads position from where the object's bottom edge
        # meets a known plane and needs neither the object's size nor its orientation.
        # So the question is not "is it clipped" but "is the bottom edge real".
        x1, y1, x2, y2 = tr.bbox_xyxy
        cut = clipped_edges(tr.bbox_xyxy, rgb.shape[:2])
        if not table_ray_is_usable(cut):
            # Either the bottom is off-frame, so the contact point is below the picture,
            # or a side is, so the visible centroid is not the object's centre and the
            # bearing taken from it is biased inward by up to half the hidden width.
            # (The range would survive a side cut — the ray's elevation comes from y2,
            # which is intact — but the approach takes a refine's BEARING in full and
            # caps only its reach, see cap_reach, which is the wrong way round for it.)
            return None, (f"it is cut off at the {'+'.join(cut)}, so "
                          + ("where it meets the table cannot be seen"
                             if "bottom" in cut else
                             "its centre, and the bearing taken from it, is not where "
                             "it looks"))
        p3 = ray_to_table(((x1 + x2) / 2.0, y2), T_cam_of(j), TABLE_Z0)
        if p3 is None:
            return None, f"cut off at the {'+'.join(cut)} and its table ray did not solve"
        xy = _correct_xy(p3[:2])
        if not (MAP_R_MIN < float(np.hypot(*xy)) < MAP_R_MAX):
            return None, (f"cut off at the {'+'.join(cut)}; its table ray lands at "
                          f"r={np.hypot(*xy)*100:.0f}cm, outside the mapped reach")
        say("        (cut off at the %s — ranged from where its bottom edge "
            "meets the table instead of from its width)" % "+".join(cut))
        return push_out_radial(xy), None
    zs = [z for z in (read_depth_m(tr.uv) for _ in range(3)) if z is not None]
    z = float(np.median(zs)) if zs else None
    # Pass the LABEL. Without it the localizer sizes every object with the fallback
    # cube edge, so anything smaller than that reads too far away — the same forward
    # walk as a clipped box, applied to every re-measure of a small object.
    xy, _rng, _sz = obj_xy_2d(tr.bbox_xyxy, T_cam_of(j), z_m=z, label=label)
    if xy is None:
        return None, "it was in view but did not localize inside the mapped reach"
    # push-out only. The bearing offset is applied by the LOCALIZER — localizers()
    # pushes CFG.bearing_offset_deg onto both strategies on every call, and
    # ApparentSizeLocalizer.locate returns Fix(rotate_xy(xy, bearing_offset_deg), ...)
    # (perception/locate.py:117). Rotating again here applied it TWICE to every
    # close-up re-measure, so the refines steered onto a bearing double the intended
    # correction while the initial fix used it once — the two sources of the pick's
    # target disagreeing by exactly the offset.
    #
    # It was invisible because the offset defaults to 0.0, where double is still zero.
    # The grasp-outcome selfcal below is the first thing that makes it non-zero
    # automatically, which would have turned a latent bug into a live one.
    return push_out_radial(xy), None


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
    return float(fx * PRIORS.fallback_edge_m / w)


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


# ALIGN_TOL_PX / ALIGN_ITERS are CFG.align_tol_px / CFG.align_iters now.


class _CenteringOps:
    """Binds this server's arm to the visual servo's interface.

    The servo itself is arm-agnostic (manipulation/approach/visual_center.py); this is
    the thin layer that knows about `finder`, the grasp pitch being held, and the wrist
    roll that must not twist mid-approach.
    """

    def __init__(self, finder, pitch_deg, roll_deg):
        self.finder, self.pitch, self.roll = finder, pitch_deg, roll_deg

    def joints(self):
        return observe()[0].astype(np.float64)

    def tip(self, q=None):
        return _tip(self.joints() if q is None else q)

    def track(self, tries=3):
        return _cube_track(self.finder, tries=tries)

    def range_m(self, track):
        return _cube_range_m(track)

    def move_pan(self, delta_deg, *, settle, step):
        """Rotate the base and report the degrees actually achieved.

        The clip below is why this has to be measured rather than assumed: with the
        pan joint on its limit the commanded delta silently becomes zero, and the
        centring probe would then divide its pixel measurement by a rotation that
        never happened. Reading the encoder back also folds in servo under-travel.
        """
        q = self.joints()
        lo, hi = J_LO[ARM.pan_joint], J_HI[ARM.pan_joint]
        before = float(q[ARM.pan_joint])
        q[ARM.pan_joint] = float(np.clip(before + float(delta_deg), lo, hi))
        goto_smooth(q, settle=settle, step=step)
        return float(self.joints()[ARM.pan_joint]) - before

    def move_tip(self, p_base, *, settle, step):
        return _move_tip(np.asarray(p_base, np.float64), self.pitch, self.roll,
                         settle=settle, step=step) is not None

    say = staticmethod(say)
    checkpoint = staticmethod(checkpoint)


def _center_on_cube(finder, gp, j5, label=None):
    """Center the object under the jaws — manipulation/approach/visual_center.py.

    Returns the final (x, y), or None if the object was never in view.
    """
    ops = _CenteringOps(finder, gp, j5)
    # Both the tolerance AND the lateral grasp bias are derived from how big the
    # object actually looks right now. A fixed pixel count means a different physical
    # distance at every range — see derive.grasp_aim_offset_px for what that cost.
    tol = None
    aim_du = CFG.aim_du_px
    tr = ops.track(tries=3)
    if tr is not None:
        rng = _cube_range_m(tr)
        if rng and rng > 0.01:
            size_m = PRIORS.size_m(label)
            tol = align_tolerance_px(GEOM, size_m, rng)
            # Bias the fingertip to the object's RIGHT so the near finger passes it
            # instead of shoving it. CFG.aim_du_px stays an operator trim on top.
            aim_du = grasp_aim_offset_px(GEOM, size_m, rng) + CFG.aim_du_px
            say(f"center: aim bias {aim_du:.0f}px "
                f"({-aim_du * rng / GEOM.fx * 100:.1f}cm right of the object at {rng*100:.0f}cm)")
    aim = (HAND_UV[0] + aim_du, HAND_UV[1] + CFG.aim_dv_px)
    res = center_on_object(ops, aim, CFG, tolerance_px=tol)
    if res.xy is None:
        # "object not in view" fires on EVERY pick, so the final visual correction
        # never runs and the grasp lands on the staged estimate alone. The trim is
        # supposed to keep the object visible here, so either it does not or the
        # detector cannot see it this close. Dump the frame the tracker actually got
        # rather than reasoning about it: one look settles which.
        try:
            _j, _rgb, _ = observe()
            path = os.path.join(OUT, f"center_miss_{int(time.time())}.jpg")
            cv2.imwrite(path, _rgb)
            say(f"center: saved the frame it could not find '{label}' in -> {path}")
            # ...and ASK, rather than filing it for a human who will not look. The dump
            # has existed for a while and the log kept saying "object not in view" for
            # situations needing opposite responses: two real frames from this rig were
            # (a) the cube present but cut off by the BOTTOM of the picture and motion
            # blurred, and (b) the camera aimed at a bag of clutter with no cube at all.
            # One wants a reframe, the other a re-survey. Off-thread and advisory — the
            # answer arrives after the pick has moved on, and it is for the log.
            if GEMINI is not None:
                GEMINI.ask_async("explain_miss", _rgb, label)
        except Exception as e:
            say(f"center: could not save the debug frame ({type(e).__name__})")
    if not res.centered:
        # A centring that did not converge returns the arm's CURRENT TIP, not the
        # object — see CenteringResult in manipulation/approach/visual_center.py.
        # Handing that back made the caller grasp at the standoff pose, which the
        # right-trim deliberately offsets from the object by ~5cm, so the jaws closed
        # on air while every stage reported success. None is the honest answer: the
        # caller already falls back to the mapped position, which is the measurement
        # we actually trust.
        if res.xy is not None:
            say(f"center: {res.reason or 'did not converge'} — "
                f"not trusting the servo's position")
        return None
    return res.xy


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
SURVEY_TILT = np.array(ARM.survey_tilt_deg)   # every joint after the pan — always these
# ...except the wrist pitch, which is live-tunable. It is the only joint that aims the
# camera without changing the arm's shape, and every range the survey produces depends
# on how steeply the sightline meets the table — so it belongs on a dial, not in a
# constant. The profile still owns the default; CFG owns what the operator dialled.
CFG.survey_pitch_deg = float(SURVEY_TILT[2])
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
    tilt = SURVEY_TILT.copy()
    tilt[2] = float(CFG.survey_pitch_deg)      # live-tunable; see ApproachConfig
    return np.concatenate(([np.clip(pan, -SURVEY_PAN_LIMIT, SURVEY_PAN_LIMIT)], tilt))


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
    vlm_budget_reset()
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
        if tr is None:
            tr = vlm_track(rgb, label)      # only when the detectors found nothing
        if tr is None or tr.clipped:
            time.sleep(0.08)
            continue
        m = measure_object(rgb, tr.bbox_xyxy, T, label)
        if m is not None:
            xy = _correct_xy(m["xy"])
        else:
            xy, _st, _sz = obj_xy_2d(tr.bbox_xyxy, T, z_m=None, label=label)
            if xy is None:
                time.sleep(0.05)
                continue
        # NOTE: push_out is NOT applied here. It is applied once in run_mission, at the
        # single point where the pick's target is decided, so it lands on the map path
        # too (USE_SURVEY_LOCATE is off by default, so this function is not even in the
        # pick's path). Correcting here as well would double-apply it.
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


def _mapped_size_m(label):
    """The characteristic edge the MAP measured for this label, or None.

    Compared against the class prior it is a free consistency check on the range:
    the two are the same measurement seen from different ends.
    """
    want = str(label).strip().lower()
    with w2d_lock:
        best, bn = None, -1
        for _t, o in WORLD.objs.items():
            lab = str(o.get("label", "")).lower()
            if (want in lab or lab in want) and o.get("n", 0) > bn:
                w_m, d_m = float(o.get("w_m") or 0.0), float(o.get("d_m") or 0.0)
                if w_m > 0 and d_m > 0:
                    best, bn = math.sqrt(w_m * d_m), o.get("n", 0)
    return best


def _mapped_xy(label):
    """The mapped (x, y) of the best-supported object of this colour, or None."""
    want = label.split()[0].lower()
    best = None
    with w2d_lock:
        for o in WORLD.objs.values():
            if want in o["label"].lower():
                if best is None or o["n"] > best["n"]:
                    best = o
        return None if best is None else np.array(best["xy"], float)


def mapped_height(label, xy, max_dist_m=0.10):
    """The map's MEASURED height for the object near xy, or None.

    Read-only on purpose. _picked_height answers a similar question but marks the entry
    as picked so the place step can retire it — calling that here would retire the
    object before it had been grasped.

    Only a genuine measurement is returned. A class prior is a guess about a category,
    and grasping a specific object by a category guess is what the fixed constant
    already did.
    """
    with w2d_lock:
        best, bd = None, float(max_dist_m)
        for _t, o in WORLD.objs.items():
            d = float(np.hypot(o["xy"][0] - xy[0], o["xy"][1] - xy[1]))
            if d < bd and str(label).split()[0] in o["label"]:
                best, bd = o, d
        if best is None or not best.get("measured"):
            return None
        h = float(best["h_m"])
    if h <= 0.004:
        return None
    # THE HEIGHT MEASUREMENT READS LOW. _dest_geometry already compensates for this
    # when stacking ("the measurement reads low, and low here means burying the carried
    # object in the target"), and it matters more here: measured against a 5.08cm cube
    # this solve returned 2.7cm, which would put the grip point 8mm off the table
    # instead of 15mm. Gripping too LOW drives the jaws into the surface; too high
    # merely misses and can be retried. So floor by the class prior, the same rule and
    # for the same reason as the place path.
    prior = class_height_m(label)
    return max(h, prior) if prior > 0 else h


def grasp_z_for(label, xy=None):
    """Where to close the jaws: derived from the object's height when it was measured.

    Falls back to the tuned constant otherwise, so an unmapped or unmeasured target
    behaves exactly as it does today rather than being grasped from a guess. The two
    agree on the 5.08 cm cube the constant was tuned on.
    """
    h = mapped_height(label, xy) if xy is not None else None
    if h is None:
        return PICK_GRASP_Z
    return float(grasp_height(h, table_z_m=TABLE_Z0))


def _picked_height(label, gx, gy):
    """Height of the object we just grasped, for growing the destination's height
    after the place. Prefers the mapped measurement nearest the grasp point, and
    marks that entry so the place can retire it. Falls back to the class prior."""
    with w2d_lock:
        best, bd = None, 0.10
        for t, o in WORLD.objs.items():
            d = float(np.hypot(o["xy"][0] - gx, o["xy"][1] - gy))
            if d < bd and label.split()[0] in o["label"]:
                best, bd = t, d
        if best is not None:
            WORLD.objs[best]["picked"] = True
            return float(max(WORLD.objs[best]["h_m"], 0.005))
    for cand in (label, f"{label} cube"):
        if str(cand).strip().lower() in CLASS_META:
            return class_height_m(cand)
    return PRIORS.fallback_edge_m


def _dest_geometry(tag=None, xy=None):
    """Resolve a placement destination to (xy, top_height, label).

    A mapped object's top is its measured height, floored by the class prior — the
    measurement reads low, and low here means burying the carried object in the
    target. A bare table spot has a top of zero.
    """
    if tag is not None:
        with w2d_lock:
            o = WORLD.objs.get(int(tag))
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
        aligned = _center_on_cube(finder, pitch, j5, dest_label)
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

    # ---- 3b. last look at the carry, before this becomes irreversible ----
    # The grasp check runs off-thread and can land AFTER place_at read carry["held"] at
    # entry. Measured: CONTACT at 17:34:55, place committed at 17:34:57, the camera's
    # "the red object is clearly visible on the surface below the gripper" at 17:35:02 —
    # five seconds too late to matter. The carry was cleared and the place carried on
    # regardless, opened its empty jaws over the destination, and the task reported
    # "placed red on green cube" with the map recording a 10.2cm stack that did not
    # exist. The cubes were side by side on the table.
    #
    # So the entry check is not enough: re-read it here, at the last moment before the
    # descent, where the verdict has had the whole traverse to arrive. The gripper
    # current cannot tell the object from a fingertip fouled on its corner; the camera
    # can, and on that run it was the one that was right.
    with lock:
        still_held = carry["held"]
    if not still_held:
        raise Abort("the jaws are empty — the grasp check overturned the pick during "
                    "the traverse, so there is nothing to place. Re-run the pick")

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
            o = WORLD.objs.get(int(tag))
            if o is not None:
                o["h_m"] = float(h_top + h_carry)
                o["t"] = time.time()
        say(f"map: {where} is now {(h_top + h_carry)*100:.1f}cm tall")
    if carry_label:
        # the carried object is no longer where it was picked from
        with w2d_lock:
            for t, o in list(WORLD.objs.items()):
                if o.get("picked") and map_label_matches(carry_label, o["label"]):
                    del WORLD.objs[t]
    set_phase("DONE", f"placed {carry_label} on {where}")




#: Believe "the jaws are empty" over the current sensor at or above this confidence.
#: Only in that ONE direction: an empty verdict makes the robot do LESS (it declines to
#: carry on with a place it would botch), while a "holding" verdict would make it do
#: more on the model's word alone. Fail-safe is not symmetric and neither is this.
GRASP_EMPTY_TRUST = 0.85

#: Whether to act on that. Deliberately NOT behind ADVISORY_ONLY: that flag guards
#: verdicts which would AUTHORISE a motion on a remote model's word, and this one only
#: ever withholds one. The worst case if the model is wrong is a pick reported as
#: failed that actually held — recoverable, and visible in the log. The worst case of
#: the reverse is the arm traversing to a destination and opening empty jaws over it.
GRASP_TRUST_EMPTY = os.environ.get("RAX_GRASP_TRUST_EMPTY", "1") != "0"


def _log_grasp_verdict(v, held_by_current):
    """Report the model's read of the grasp next to the current sensor's — and, in one
    direction only, act on it.

    Disagreement points both ways. "Current says held, camera says empty" is a finger
    fouled on the object or a grip on the table edge; "current says empty, camera says
    holding" means the contact threshold is too high and good picks are being discarded.

    Only the first is acted on. The gripper current says the jaws met RESISTANCE, which
    reads identically for the object, a fingertip fouled on its corner, and the table
    edge — it cannot tell you WHAT it met. The camera can, and it was right about every
    miss observed on this rig. So a confident "empty" clears the carry flag: the arm
    then refuses to place, instead of traversing to the destination and solemnly opening
    its empty jaws, which is what it did before. The reverse would have a remote model
    authorise a motion on nothing but its own say-so, and is left advisory.
    """
    if not v.ok:
        say(f"        grasp check unavailable ({v.reason})")
        return
    agrees = (v.answer == "holding") == bool(held_by_current)
    if v.answer == "unsure":
        say(f"        grasp check: model could not tell — {v.reason}")
        return
    if agrees:
        say(f"        grasp check: model agrees ({v.answer}, {v.confidence:.0%})")
        return
    say(f"        grasp check: DISAGREES — current said "
        f"{'HELD' if held_by_current else 'EMPTY'}, camera says {v.answer.upper()} "
        f"({v.confidence:.0%}) — {v.reason}")
    if (held_by_current and v.answer == "empty"
            and v.confidence >= GRASP_EMPTY_TRUST and GRASP_TRUST_EMPTY):
        with lock:
            still_holding = carry["held"]
        if still_holding:
            _set_carry(False)
            set_phase("PICK", "grasp check says the jaws are empty — not carrying")
            say(f"        carry CLEARED on the camera's word ({v.confidence:.0%}): a "
                f"place from here would put nothing down. Re-run the pick.")


def run_mission(target_label=None, reraise=False):
    """Close on the mapped cube in smooth stages, re-checking the map at every
    stage, then descend gradually and grip.

    One long move to the target is a dive: if the mapped position is a little off,
    nothing notices until the gripper is already there. Instead cover the distance
    in stages -- each one closes part of the remaining gap, then looks again and
    refines the target. Errors get corrected while there is still room to correct
    them, and the motion rates are low enough not to jerk.
    """
    def _approach_target(cube_xy, stage=0):
        """Hover position for the approach: short of the cube and to its right.

        The geometry itself is approach_target() in manipulation/approach; what this
        adds is the visibility cap on the trim, which needs the object's size prior and
        the camera — things the pure geometry deliberately does not know about.

        CFG.back_m: stop this many metres radially BEFORE the cube (keeps it in view).
        CFG.right_trim_m: shift this far to the cube's right (tunable in the UI).
        """
        cube_xy = np.asarray(cube_xy, dtype=np.float64)
        # The trim keeps the object off to one side so it stays in frame while the
        # gripper closes in. CFG.right_trim_m is AUTHORITATIVE: it is what the UI
        # shows, what the operator tunes, and a value they confirmed works.
        #
        # right_trim_for_visibility is a CAP, not a replacement. It answers "how far
        # could I sit and still see the object", which is an upper bound — it was
        # being used as the trim itself, so it sat at 75% of the maximum offset that
        # still fits in frame, saturating its own 12cm cap at normal ranges. Measured
        # from a real failing run: the log printed "trim=5.0cm" while applying 12.0cm,
        # putting the hover 12cm to the side of the cube. The cube then fell outside
        # the view, "center: object not in view" fired on EVERY run, the centring step
        # bailed to the mapped position, and the descent ran on an uncorrected estimate
        # and closed on air. Take the smaller of the two: never further than the
        # operator asked, and never further than stays visible.
        closest = max(float(np.hypot(*cube_xy)) - CFG.back_m, 0.05)
        # DECAY the trim across the stages — stage_trim() in manipulation/approach.
        # Full offset on the first hop, where losing sight of the object is the real
        # risk, shrinking to trim_final_frac by the last one, where the offset is
        # nothing but pixel error handed to the centring servo. Applied BEFORE the
        # visibility cap so the cap still bounds whatever the decay asked for.
        # Land the decay on the offset the GRASP wants, not on a fraction of the
        # visibility trim. One lateral offset, held from the first hop to the jaws,
        # so the centring servo has no sideways move left to make beside the object.
        trim = stage_trim(float(CFG.right_trim_m), stage=stage, total=int(CFG.steps),
                          final_frac=float(CFG.trim_final_frac),
                          final_m=grasp_bias_m(PRIORS.size_m(label)))
        cap = right_trim_for_visibility(GEOM, PRIORS.size_m(label), closest)
        if cap > 0.0:
            trim = min(trim, cap)
        return approach_target(cube_xy, back_m=CFG.back_m, right_trim_m=trim), trim

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
        used_survey = False
        if USE_SURVEY_LOCATE[0]:
            cube_xy, spread = locate_from_survey(label, finder)
            used_survey = cube_xy is not None
            if cube_xy is None:
                say(f"survey failed ({spread}); falling back to the 2D map")
        if cube_xy is None:
            cube_xy = _mapped_xy(label)
        if cube_xy is None:
            # LAST RESORT, and the place it actually earns its keep. The map is empty
            # for this label, so the alternative is not a worse pick — it is no pick at
            # all: "'red cube' is not on the 2D map", with the cube sitting in plain
            # view of the camera. That happened repeatedly, because the strict-HSV
            # bands miss an object whose colour shifts with the viewing angle (a green
            # cube photographs teal from a steep look and is not found at all).
            #
            # Measured against this rig's own frames: where the object IS in the
            # picture the model's box centre agrees with the HSV blob to within ~5px,
            # and on the teal frame it boxed the cube while HSV had latched onto an
            # unrelated blob 211px away. One look, then the ordinary geometry converts
            # the pixel — the model is never asked how far away anything is.
            vlm_budget_reset()
            try:
                j_v, rgb_v, _ = observe(overlay=False)
                tr_v = vlm_track(rgb_v, label)
            except Exception:
                tr_v = None
            if tr_v is not None:
                xy_v, rng_v, _sz = obj_xy_2d(tr_v.bbox_xyxy, T_cam_of(j_v),
                                             z_m=None, label=label)
                if xy_v is not None:
                    cube_xy = xy_v
                    say(f"  [2/7] target    from the vision model: "
                        f"({xy_v[0]*100:.1f},{xy_v[1]*100:.1f})cm, range {rng_v*100:.0f}cm "
                        f"— the map had no '{label}'")
        if cube_xy is None:
            how = "the survey pose or " if USE_SURVEY_LOCATE[0] else ""
            raise Abort(f"'{label}' is not on the 2D map (looked in {how}the map). "
                        f"Press 'Scan -> 2D map', and check the query names it — "
                        f"the query is currently '{state.get('query')}'")
        # THE single place the reach correction is applied, deliberately AFTER both
        # sources have had their say. It used to live inside locate_on_table (which the
        # pick never calls) and then locate_from_survey (which is off by default), so
        # the "reach cm" knob moved nothing on a real pick no matter what it was set to.
        # Applied here it lands on whichever source produced the target, and the [2/7]
        # line below reports the corrected number the arm will actually drive at.
        raw_xy = np.asarray(cube_xy, np.float64)
        cube_xy = push_out_radial(raw_xy)
        if abs(float(CFG.push_out_m)) > 1e-9:
            say(f"        reach correction {CFG.push_out_m*100:+.1f}cm: "
                f"r {np.hypot(*raw_xy)*100:.1f} -> {np.hypot(*cube_xy)*100:.1f}cm")
        say(f"  [2/7] target    x={cube_xy[0]*100:+.1f} y={cube_xy[1]*100:+.1f} cm  "
            f"r={np.hypot(*cube_xy)*100:.1f}cm "
            f"bearing={math.degrees(math.atan2(cube_xy[1], cube_xy[0])):+.0f}deg")
        say(f"  [3/7] approach   {CFG.steps} stages, "
            f"trim={CFG.right_trim_m*100:.1f}cm right, back={CFG.back_m*100:.1f}cm, "
            f"hover z={PICK_HOVER_Z*100:.0f}cm")
        say("        opening the gripper")
        send_joints(observe()[0], gripper=95.0)

        # ---- approach: stay to the right and short of the cube ----
        # The hover target is recomputed from the LATEST cube estimate every stage, so
        # it keeps the cube on the LEFT side of the image and stops before the arm
        # passes it. Refinements update the cube position, not the approach offset.
        #
        # Two real picks, two different ways for that to go wrong, both handled here.
        #
        # When the look FAILS, it used to fail silently: all three checks logged
        # "re-measured 0.0cm away" because the close-up detection had found nothing and
        # the code fell back to _mapped_xy — the same map entry the target came from —
        # then reported that non-answer as a successful refine. The arm drove three
        # hops, the last at the full remaining distance, on an estimate nothing had ever
        # confirmed. The map fallback is gone: it cannot correct an error it is the
        # source of. A failed look now says how it failed and holds the margin back.
        #
        # When the look SUCCEEDS, its range is biased outward and the staging compounds
        # it: measured, r = 41.0 -> 44.9 -> 46.5cm over three stages while the cube sat
        # at 43.3, each stage stepping further out on the last one's over-estimate,
        # until the arm was past the object with it out of frame. So a re-measure gets
        # its BEARING taken in full — that half converged on the truth, 44 -> 38deg
        # against an actual 34 — while cap_reach holds its REACH to the original
        # estimate plus a fixed budget. See cap_reach for why the bias has a sign.
        seen_close = False       # has a close-up sighting agreed with the target yet?
        r_cap = float(np.hypot(*cube_xy)) + CFG.max_refine_out_m
        for i in range(CFG.steps):
            checkpoint()
            q = observe()[0].astype(np.float64)
            j5 = float(q[4])         # keep the wrist as-is; do NOT twist while approaching
            tip = _tip(q)
            target_xy, trim_used = _approach_target(cube_xy, stage=i)
            # Log the trim ACTUALLY applied. The [3/7] banner used to print the knob
            # while a derived value overrode it, so the log disagreed with the robot.
            say(f"approach {i+1}: cube=({cube_xy[0]*100:.1f},{cube_xy[1]*100:.1f})cm "
                f"target=({target_xy[0]*100:.1f},{target_xy[1]*100:.1f})cm "
                f"offset=({(target_xy-cube_xy)[0]*100:.1f},{(target_xy-cube_xy)[1]*100:.1f})cm "
                f"trim={trim_used*100:.1f}cm")
            # Stage 1 closes most of the gap, the rest are corrections. `commit` is why
            # the sighting is tracked: only a stage that follows a confirming close-up
            # read may spend the last of the margin in one move.
            waypoint, dist_xy = stage_step(
                tip[:2], target_xy, stage=i, total=CFG.steps,
                first_frac=CFG.first_step_frac, max_first_m=CFG.max_first_step_m,
                commit=seen_close)
            if waypoint is None or dist_xy < CFG.arrived_m:
                say(f"approach {i+1}: already at approach hover")
                break
            wx, wy = float(waypoint[0]), float(waypoint[1])

            pitch, e = plan_grasp_pitch(
                np.array([cube_xy[0], cube_xy[1], PICK_GRASP_Z]), q)
            if pitch is None:
                r_cm = float(np.hypot(*cube_xy)) * 100.0
                msg = (f"r={r_cm:.0f}cm is out of reach (best IK {e*1e3:.0f}mm; "
                       f"this arm reaches {GRASP_R_MAX*100:.0f}cm with a flat wrist, "
                       f"less as the grasp steepens)")
                # Before blaming the arm, check whether the RANGE is even believable.
                # Range and apparent size are the same measurement: an object twice as
                # far is inferred twice as big. So a "5cm cube" that measures 10cm is
                # not a big cube -- it is a doubled range, and the reach limit is a
                # symptom, not the fault. Seen live: a cube 20cm away, its box clipped
                # at the frame edge, mapped at 42cm and reported "51cm is out of reach"
                # while the arm sat correctly refusing a target that was never there.
                measured = _mapped_size_m(label)
                prior = float(PRIORS.size_m(label))
                if measured and prior > 0 and not (0.6 < measured / prior < 1.7):
                    msg = (f"r={r_cm:.0f}cm is out of reach, BUT that range is not "
                           f"trustworthy: '{label}' measures {measured*100:.1f}cm across "
                           f"when a {label} is {prior*100:.1f}cm, so the range is off by "
                           f"about {measured/prior:.1f}x and the object is nearer than "
                           f"{r_cm:.0f}cm. Re-scan with it fully in frame (a box clipped "
                           f"at the frame edge ranges badly), rather than raising reach")
                raise Abort(msg)
            with lock:
                state["obj3d"] = [float(cube_xy[0]), float(cube_xy[1]), PICK_GRASP_Z]
                state["obj3d_label"] = label
            set_phase("PICK", f"approach {i+1}/{CFG.steps} "
                              f"-> x={wx*100:.0f} y={wy*100:.0f} cm")
            if _move_tip(np.array([wx, wy, PICK_HOVER_Z]), pitch, j5,
                         settle=0.15, step=1.4) is None:
                raise Abort(f"IK cannot reach x={wx*100:.0f} y={wy*100:.0f}")

            # Look again from the new vantage — the closest one so far.
            time.sleep(0.05)
            sense_2d()
            xy2, why = _refix_here(finder, label)
            if xy2 is None:
                if seen_close:
                    # It was in view a stage ago and is not now: the arm is closing on
                    # it and it has gone under the gripper or out of frame. Every
                    # further stage would be a blind dive past it. Stop at this
                    # vantage — the last one that could see it — and let the centring
                    # step work from here, where it still has something to centre on.
                    say(f"check {i+1}/{CFG.steps}: lost it — {why}. Stopping the "
                        f"approach here rather than closing in blind")
                    break
                say(f"check {i+1}/{CFG.steps}: no fresh fix — {why}. Still driving on "
                    f"the original estimate, so the approach keeps its margin")
                continue

            # Take the bearing, hold the reach. The cap is against the ORIGINAL fix, so
            # three stages of outward-biased reads cannot add up to a drive past it.
            r_raw = float(np.hypot(*xy2))
            xy2 = cap_reach(xy2, r_cap)
            if r_raw > r_cap + 1e-6:
                say(f"check {i+1}/{CFG.steps}: read r={r_raw*100:.1f}cm, further out "
                    f"than anything before it — held at {r_cap*100:.1f}cm and kept only "
                    f"its bearing")
            adj = float(np.linalg.norm(xy2 - cube_xy))
            # Beyond the bound, assume a DIFFERENT object was detected. Inside it, steer
            # toward the new fix rather than snapping onto it, so one bad read cannot
            # hijack the target. (The bound is CFG.max_refine_jump_m — the config field
            # existed while the code hardcoded 0.05, so tuning it did nothing.)
            if adj > CFG.max_refine_jump_m:
                say(f"check {i+1}/{CFG.steps}: re-measure {adj*100:.1f}cm away - "
                    f"beyond the {CFG.max_refine_jump_m*100:.0f}cm sanity bound, "
                    f"treating as a different object and ignoring")
                continue
            gain = float(CFG.first_refine_gain if not seen_close else CFG.refine_gain)
            cube_xy = cube_xy + (xy2 - cube_xy) * gain
            seen_close = True
            say(f"check {i+1}/{CFG.steps}: re-measured {adj*100:.1f}cm away, took "
                f"{gain*100:.0f}% of it -> "
                f"({cube_xy[0]*100:.1f},{cube_xy[1]*100:.1f})cm")

        # ---- FINAL CENTERING BY EYE, then descend on the aligned spot ----
        q = observe()[0].astype(np.float64)
        j5 = float(q[4])             # no wrist twist
        grasp_z = grasp_z_for(label, cube_xy)
        if abs(grasp_z - PICK_GRASP_Z) > 1e-6:
            say(f"        grasp height {grasp_z*100:.1f}cm from the measured object "
                f"(fixed default is {PICK_GRASP_Z*100:.1f}cm)")
        gp, _e = plan_grasp_pitch(np.array([cube_xy[0], cube_xy[1], grasp_z]), q)
        gp = gp if gp else 70.0
        set_phase("PICK", "centering the cube under the jaws")
        # The lateral aim is DERIVED per-object inside _center_on_cube, which logs it;
        # printing HAND_UV+aim_du here would report an aim point that is not the one
        # used, which is exactly how the old dialled offset stayed invisible.
        say(f"  [4/7] centering  pitch={gp:.0f}deg wrist_roll={j5:.0f}deg "
            f"fingertip=({HAND_UV[0]:.0f},{HAND_UV[1]:.0f})px")
        aligned = _center_on_cube(finder, gp, j5, label)
        # Half a hand-eye sample: where the object APPEARED, from a pose we still know.
        # The other half arrives at contact. See note_grasp_observation.
        note_grasp_observation(finder, label)
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
            z = float(z0 + (grasp_z - z0) * f)
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
            # CALIBRATION, FOR FREE. The jaws are around the object, so FK now says
            # where it actually was — the same ground truth /selfcal/touch asks an
            # operator to jog to by hand. Pair it with raw_xy, the localization this
            # pick was aimed from, and this grasp becomes one calibration sample.
            #
            # BEFORE the lift, because lifting moves the object and FK would then
            # describe where it is being carried, not where it was found. And only on
            # CONTACT: a grasp that closed on air knows nothing about where anything is.
            try:
                tip_now = _tip(observe(overlay=False)[0])
                _selfcal_record(label, raw_xy, tip_now[:2],
                                source="survey" if used_survey else "map")
                # The same instant, kept in the form a hand-eye fit can use: the pixel
                # it was seen at, the pose it was seen from, and now the position the
                # grasp PROVED. Unlike the map's estimate this owes nothing to the
                # transform being fitted, which is what makes it usable as truth.
                if record_grasp_sample(tip_now):
                    say(f"handeye: grasp sample #{len(load_grasp_samples())} recorded")
            except Exception as e:
                say(f"selfcal: could not record this grasp ({type(e).__name__}: {e})")
        else:
            _set_carry(False)
        set_phase("PICK", "lifting")
        say(f"  [7/7] lift      +{PICK_LIFT_M*100:.0f}cm")
        ee_move_rel([0, 0, PICK_LIFT_M], settle=0.25)
        say(f"PICK {'SUCCESS' if held else 'FAILED - closed on air'}  "
            f"tip={np.round(_tip(observe(overlay=False)[0])*100,1).tolist()}cm")
        # SECOND OPINION ON THE GRASP. `held` comes from a gripper current delta, which
        # says the jaws met resistance — not that they met the OBJECT. It reads the same
        # for a finger fouled on the cube's corner, for the table edge, and for a real
        # grasp. The arm is stationary and holding at this point, so a frame now costs
        # nothing on the critical path and answers the question directly. Advisory: it
        # is logged next to `held` so the two can be compared over a run of picks before
        # anything is allowed to branch on it.
        if GEMINI is not None:
            try:
                GEMINI.ask_async("verify_grasp", observe(overlay=False)[1], label,
                                 on_done=lambda v: _log_grasp_verdict(v, held))
            except Exception:
                pass
        say("=" * 52)
        set_phase("DONE" if held else "PICK",
                  f"{label} cube {'picked' if held else 'missed - closed on air'}")

    except Abort as e:
        set_phase("ABORTED", str(e))
        if reraise:
            raise
    except Exception as e:
        set_phase("ERROR", f"{type(e).__name__}: {e}")
        if reraise:
            raise
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
# The queue itself lives in guest_sessions.GuestSessions — see GUESTS below, built
# once the robot-side callbacks it needs are defined.


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
    fresh = GUESTS.see(vid, _client_ip(),
                       ua=(request.headers.get("User-Agent") or "")[:90])
    if fresh:
        # First-seen time is only for the admin roster's "waiting" column, so it is
        # the server's bookkeeping rather than the queue policy's.
        visitors[vid]["first"] = time.time()
    return vid, new


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


# THE QUEUE OBJECT. Everything above this line is robot- and Flask-specific; the
# turn-taking policy itself is not, so it lives in guest_sessions.py where it can be
# tested without a robot (tests/test_guest_sessions.py runs an evening of visitor
# behaviour in a millisecond). The three things it cannot know are injected: how to
# park the arm when a turn ends, how to announce a turn starting, and where to log.
GUESTS = GuestSessions(
    GuestConfig(minutes=GUEST_MINUTES, cooldown_min=GUEST_COOLDOWN_MIN,
                queue_ttl_s=QUEUE_TTL),
    on_turn_end=_guest_cleanup,
    on_turn_start=lambda vid, waiting: say(
        f"guest turn started ({GUEST_MINUTES:.0f} min) — "
        f"{GUESTS.visitors[vid]['ip']} — {waiting} waiting"),
    log=_log_guest,
)

# Read-only aliases so the routes below still read the way they did. `queue` is NOT
# aliased: _prune rebinds it, so a module-level name would go stale after the first
# sweep — the routes ask GUESTS.queue instead.
guest = GUESTS.holder
visitors = GUESTS.visitors
guest_lock = GUESTS.lock


def lobby_tick(vid):
    """Register/refresh this visitor's place and report where they stand.

    This is the ONE call the lobby page makes. It expires the running turn, hands
    the arm to whoever is next, and keeps the queue swept — so the whole system is
    driven by visitors polling, with no background thread to get out of sync.
    """
    return GUESTS.tick(vid).to_dict()


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
    return GUESTS.state()


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
    DETECT.request_query(want)         # applied by the detector thread
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
        waiting = len(GUESTS.queue)
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
    a, b, c = FLOOR.as_list()
    return jsonify(a=a, b=b, c=c,
                   tilt_deg=round(FLOOR.tilt_deg, 3),
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
              "cooldown": round(GUESTS.cooldown_left(v, now))}
             for i, v in enumerate(GUESTS.queue)]
        seen = len(visitors)
    rows, by_ip = _guest_history()
    return jsonify(playing=playing, queue=q, seen_now=seen,
                   by_ip=by_ip[:25], recent=rows[::-1][:40],
                   minutes=GUEST_MINUTES, cooldown_min=GUEST_COOLDOWN_MIN)


@app.route("/guestkick", methods=["POST"])
def guestkick():
    """Admin-only: end the current guest's turn immediately."""
    # end_current only ends. The next lobby poll promotes whoever is next, which is
    # within a second and gives _guest_cleanup time to park the arm first.
    GUESTS.end_current()
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
    s = state.snapshot()
    s["log"] = state.log_lines()
    s["measure_stats"] = dict(MEAS_STATS)
    s["relaxed"] = ARM_RELAXED[0]
    s["idle_relax_s"] = IDLE_RELAX_S[0]
    s["idle_for"] = round(time.time() - last_activity[0])
    s["tune"] = CFG.as_dict()
    s["tune"]["cube_edge_cm"] = round(PRIORS.fallback_edge_m * 100, 2)
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
            meshes = load_link_visual_meshes_cached(ARM.mesh_path) or {}
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
                   obj_size=round(float(PRIORS.fallback_edge_m), 3), objs2d=objs2d)


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
    """Start button. Runs a TASK when the query names a destination, else a pick.

    See reads_as_a_task(): "red on green" typed into the query box used to pick the
    red cube and stop, because Start only ever ran run_mission and run_mission takes
    the first label in the query as its target.

    The two paths latch differently and must not be swapped: run_mission clears the
    stop flag and owns state["running"] itself, while run_task relies on _run_bg for
    both. Routing a task through the bare thread below would leave it unstoppable and
    invisible to the busy check.
    """
    with lock:
        busy = state["running"]
        q = pending_instruction(state)   # not state["query"] - see pending_instruction
    if busy or (mission_thread[0] is not None and mission_thread[0].is_alive()):
        return jsonify(ok=False, reason="already running")
    steps = reads_as_a_task(q)
    if steps is not None:
        say(f"start: '{q}' reads as a task ({len(steps)} step(s)) — placing, not just picking")
        return _run_bg(run_task, q)
    mission_thread[0] = threading.Thread(target=run_mission, daemon=True)
    mission_thread[0].start()
    return jsonify(ok=True, task=False)


@app.route("/stop", methods=["POST"])
def stop():
    stop_flag.set()
    say("STOP requested")
    return jsonify(ok=True)


@app.route("/shutdown", methods=["POST"])
def shutdown_route():
    """Stop the server the way that actually releases the camera.

    THIS IS THE ONLY RELIABLE GRACEFUL STOP ON WINDOWS. PowerShell's Stop-Process is a
    hard TerminateProcess — equivalent to SIGKILL — so no signal handler runs, and
    Python on Windows does not deliver an external SIGTERM to a handler either. Killing
    the server that way leaves the OAK-D booted with no owner, which is what makes the
    next start fail with "No available devices" until somebody unplugs it physically.

    So: POST here instead of killing the process. It releases the camera and the servo
    bus first, then exits.
    """
    if is_public_request():
        return jsonify(ok=False, reason="not available to guests"), 403
    with lock:
        busy = state["running"]
    if busy and not request.args.get("force"):
        return jsonify(ok=False, reason="a mission is running; pass ?force=1"), 409

    def _bye():
        time.sleep(0.3)          # let this response reach the client first
        _shutdown("/shutdown requested")
        os._exit(0)

    threading.Thread(target=_bye, daemon=True).start()
    say("shutdown requested — releasing camera and bus")
    return jsonify(ok=True, note="camera and bus released before exit")


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
# Which joints those are is now declared by the profile (pan_joint / pitch_chain /
# roll_joint), so an arm with a different layout describes itself instead of being
# assumed here.
#
# THE REAL JOINT LIMITS, now READ FROM THE URDF rather than transcribed. An IK that
# does not know these is not an IK, it is a wish: it returns elbow_flex=+162 deg on a
# joint that stops at +96.8, the servo silently clamps, the arm parks at the stop, and
# the solver reports a 0.2 mm residual on a pose the robot cannot hold. That is exactly
# what froze the approach for 21 identical hops at pitch 65 while being commanded
# to 80 (2026-07-13). CLAMP EVERY ITERATION AND SCORE THE CLAMPED POSE.
J_LO, J_HI = ARM.limits()
_WFLEX = ARM.slaved_joint                               # the algebraically-slaved joint
WFLEX_MIN, WFLEX_MAX = float(J_LO[_WFLEX]), float(J_HI[_WFLEX])



def _slave_wflex(j1, j2, pitch_tgt):
    """Hold the tool pitch by slaving the last pitch joint. See PitchHoldIK.slave.

    Not called from this file — the IK strategy does its own slaving internally. It
    stays because tests/test_extraction_parity.py pins 60 golden values through it,
    which is what keeps the extracted PitchHoldIK.slave honest against the original
    monolith. Delete it and that check silently stops covering anything.
    """
    q = np.zeros(ARM.n_joints, dtype=np.float64)
    q[1], q[2] = j1, j2
    return ik_strategy().slave(q, pitch_tgt)

def _ik_hold_pitch(q_seed, p_tgt, pitch_tgt, j5_fixed, iters=80, tol=2e-3,
                   ret_err=False, _retry=True):
    """Position IK holding the tool pitch — now manipulation/arms/ik_strategy.py.

    The solver moved out verbatim (verified bit-identical over a 120-case grid); only
    the joint indices are read from the profile instead of being literals, which is
    what lets a different arm use it. The comments explaining WHY it looks the way it
    does — the fixed-10-iteration bug that made the arm grab at air, the Jacobian
    reuse, and the two invariants about clamping to the joint limits — live with the
    code there.

    Callers MUST check the residual: a target the arm cannot reach is a fact to
    report, not a pose to drive to.
    """
    q, e = ik_strategy().solve(q_seed, p_tgt, pitch_deg=pitch_tgt, roll_deg=j5_fixed,
                               iters=iters, tol=tol, _retry=_retry)
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
        # Was a hardcoded 0.42. The arm reaches 0.47 with a flat wrist, so this
        # clipped 5cm short of the real limit -- jogging outward simply stopped,
        # with nothing said, and the arm looked unable to extend.
        r_tgt = float(np.clip(r_tgt, ARM.reach_min_m + 0.04, GRASP_R_MAX))
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
    (e.g. 'red cube, toy block, red box').

    A task phrase ("red on green") is not a vocabulary, but it is a perfectly natural
    thing to type here. Rather than accept it as two odd class names, expand it to the
    vocabulary the task actually needs — every object it mentions — so both the pick
    and its destination can reach the map. Start then runs it as a task.
    """
    q = (request.args.get("q") or "").strip()
    if not q or detector is None:
        return jsonify(ok=False, reason="empty query or detector not ready")
    steps = reads_as_a_task(q)
    vocab = q if steps is None else task_vocabulary(steps)
    DETECT.request_query(vocab)     # applied by the detector thread
    with lock:
        state["query"] = q
        state["vocabulary"] = vocab
        # state["query"] is NOT a safe place to keep the phrase: the detector thread
        # reports the vocabulary it actually applied back into it (on_query_change in
        # main()), so a moment later "red on green" has become "red cube, green cube"
        # and Start sees an ordinary two-class query again. Caught on the live server,
        # not by the tests — the overwrite only happens once the detector cycles.
        # state["intent"] holds what the operator ASKED for and nothing overwrites it.
        state["intent"] = q if steps is not None else None
    if steps is not None:
        say(f"query: '{q}' reads as a task — detecting '{vocab}' so both the object "
            f"and its destination can reach the map. Press Start (or Run task).")
    return jsonify(ok=True, query=q, vocabulary=vocab, task=steps is not None)


@app.route("/preset")
def preset():
    q = {"coco": ", ".join(COCO_CLASSES),
         "table": ", ".join(TABLE_CLASSES),
         "cubes": "red cube, green cube"}.get(request.args.get("name", ""))
    if q is None:
        return jsonify(ok=False, reason="unknown preset")
    return jsonify(ok=True, q=q)


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
    cm = CFG.set_knob("push_out_cm", cm)
    say(f"localization push-out set to {cm:.0f}cm — re-locate to apply")
    return jsonify(ok=True, cm=cm)


@app.route("/setknob", methods=["POST"])
def setknob():
    """Live-tune ANY knob by name: /setknob?name=survey_pitch_deg&value=68.2

    The single tuning route. Six per-knob shims (/setaimdu, /settrim, /setback,
    /setsteps, /setrangescale, /setbearing) used to sit alongside it, each
    re-implementing the same clamp-and-log against one hardcoded knob name; they
    predated ApproachConfig owning its own bounds. A knob added to KNOBS is now
    tunable the moment it exists. GET /setknob lists what there is to tune.

    /setcubesize is deliberately NOT folded in here: it sets a perception prior
    (PRIORS.fallback_edge_m), not an approach knob, so it has no entry in KNOBS.
    """
    name = (request.args.get("name") or "").strip()
    if not name:
        return jsonify(ok=False, reason="need name=",
                       knobs={n: CFG.get_knob(n) for n in CFG.knob_names()})
    if name not in CFG.knob_names():
        return jsonify(ok=False, reason=f"unknown knob {name!r}",
                       knobs=CFG.knob_names()), 400
    try:
        value = float(request.args.get("value", request.form.get("value")))
    except (TypeError, ValueError):
        return jsonify(ok=False, reason="need a numeric value="), 400
    was = CFG.get_knob(name)
    now = CFG.set_knob(name, value)      # clamps to the knob's own bounds
    say(f"{name}: {was:g} -> {now:g}" + ("  (clamped)" if abs(now - value) > 1e-9 else ""))
    return jsonify(ok=True, name=name, was=was, value=now)


@app.route("/setcubesize", methods=["POST"])
def setcubesize():
    """Live-tune the assumed cube edge size (cm). Used by the 2D map apparent-size
    fallback. If the 3D lock / map cubes look too close/far or the wrong size,
    adjust this to the real cube edge."""
    try:
        cm = float(request.args.get("cm", request.form.get("cm", 5.08)))
    except (TypeError, ValueError):
        return jsonify(ok=False, reason="need a number")
    cm = float(np.clip(cm, 1.0, 30.0))
    PRIORS.fallback_edge_m = cm / 100.0
    say(f"cube edge size set to {cm:.2f}cm")
    return jsonify(ok=True, cm=cm)


# ---------------- self-calibration: measure the localization error instead of dialling it ----------------
# push_out / range_scale / bearing were dials someone turned until grasps stopped
# missing. They can be solved instead: the arm's OWN forward kinematics is ground
# truth, so observing an object from a normal viewing distance and then TOUCHING it
# gives a (camera said, arm found) pair with no ruler and no guessing. A few pairs
# across the workspace and perception/selfcal.py fits push_out/range_scale/bearing
# directly — and refuses to apply a fit when the residual has a known bad-geometry
# signature (like the axial-depth bias) that a linear correction cannot actually fix.
#
# Two-step, operator-in-the-loop, because autonomous grasp is not yet proven on this
# rig and this needs no grasp at all:
#   1. POST /selfcal/observe   — from wherever the arm is now, look at the object and
#                                 record where the CAMERA says it is.
#   2. jog the fingertip down until it touches the object (by hand, via /jogpress /
#      /jogvec — whatever is already on screen)
#   3. POST /selfcal/touch     — record where the ARM says it is (FK, ground truth).
# Repeat at a few positions across the workspace — different ranges AND bearings, or
# the fit cannot separate the effects (see the "cannot be separated" warning). Then:
#   POST /selfcal/fit    — see the numbers without changing anything
#   POST /selfcal/apply  — write them to CFG, but only if the fit is trustworthy
_selfcal = {"pending": None, "samples": []}

# AND THE SAME MEASUREMENT, TAKEN FOR FREE. The operator-in-the-loop procedure above
# is correct and nobody runs it: it costs six jog-and-press cycles before it will even
# report a fit, so push_out/range_scale/bearing stayed hand-dialled and the pick kept
# missing in the way they were invented to patch.
#
# But a SUCCESSFUL GRASP IS ALREADY THIS EXACT SAMPLE. At the instant the jaws close on
# something, forward kinematics says where that something is — the same ground truth
# /selfcal/touch collects, and produced by the same fingertip, except the robot got
# there by itself and the operator pressed nothing. Pair it with the localization the
# pick aimed at and every successful pick is one calibration point, gathered during
# ordinary use, spread across exactly the radii and bearings that are actually used.
#
# Samples persist to JSONL so they accumulate across restarts — a calibration that
# needs six points is worthless if the buffer empties every time the server reloads.
SELFCAL_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "selfcal_samples.jsonl")
selfcal_log_lock = threading.Lock()

#: Grasps this far from where the camera said the object was are not calibration data.
#: Beyond it the pick was rescued by the centring servo or got lucky, and the pairing
#: "camera said X, object was at Y" no longer describes one localization error.
SELFCAL_MAX_PAIR_M = 0.15


def _selfcal_off_axis_deg(uv):
    """Angle between a pixel's ray and the camera's own view direction."""
    d = np.array([(uv[0] - cx0) / fx, (uv[1] - cy0) / fy, 1.0])
    return float(math.degrees(math.acos(np.clip(1.0 / np.linalg.norm(d), -1.0, 1.0))))


def _selfcal_uncorrect(xy, range_scale, bearing_offset_deg):
    """Strip the corrections that were live when an observation was taken.

    The fit must be ABSOLUTE — "the rig's range is 6% long" — not a delta on whatever
    was dialled in at the time, or applying it a second time would double the very
    correction it just measured. The localizer's forward correction is
    ``r' = range_scale * r`` then ``theta' = theta + bearing`` (perception/locate.py),
    and this is its exact inverse.

    This inversion is only valid because every localization path applies the SAME
    correction. It did not used to: the silhouette solve and the bbox-to-table ray got
    the bearing offset but never range_scale, so a fitted scale moved some of the map's
    fixes and not others and no single inverse existed. :func:`_correct_xy` is what
    makes them uniform — change one and this stops being an inverse.
    """
    # LocalizationModel.unapply is the exact inverse of the correction the localizer
    # applies, and lives with the model so the two cannot drift apart. push_out is 0
    # here on purpose: raw_xy is captured BEFORE run_mission applies the push-out.
    return LocalizationModel(range_scale=float(range_scale), push_out_m=0.0,
                             bearing_offset_deg=float(bearing_offset_deg)).unapply(xy)


def _selfcal_record(label, xy_observed, xy_true, source, off_axis_deg=0.0):
    """Log one (camera said, arm found) pair. Returns the sample, or None if refused."""
    obs = np.asarray(xy_observed, dtype=np.float64)
    true = np.asarray(xy_true, dtype=np.float64)
    err = float(np.linalg.norm(true - obs))
    if err > SELFCAL_MAX_PAIR_M:
        say(f"selfcal: not recording this grasp — camera and arm disagree by "
            f"{err*100:.1f}cm, past the {SELFCAL_MAX_PAIR_M*100:.0f}cm bound where the "
            f"pair still describes one localization error")
        return None
    sample = {
        "label": str(label), "source": str(source),
        "xy_true": [float(true[0]), float(true[1])],
        "xy_observed": [float(obs[0]), float(obs[1])],
        "off_axis_deg": float(off_axis_deg),
        "range_m": float(np.hypot(*obs)),
        # The correction state the observation was taken under, so it can be undone.
        "range_scale": float(CFG.range_scale),
        "bearing_offset_deg": float(CFG.bearing_offset_deg),
        "t": time.time(),
    }
    _selfcal["samples"].append(sample)
    try:
        with selfcal_log_lock, open(SELFCAL_LOG, "a") as f:
            f.write(json.dumps(sample) + "\n")
    except Exception as e:
        say(f"selfcal: sample kept in memory but not written ({type(e).__name__})")
    say(f"selfcal: sample #{len(_selfcal['samples'])} from a successful grasp — "
        f"camera said r={np.hypot(*obs)*100:.1f}cm, arm found r={np.hypot(*true)*100:.1f}cm "
        f"(off by {(np.hypot(*true) - np.hypot(*obs))*100:+.1f}cm)")
    return sample


def _selfcal_load_log():
    """Re-read persisted samples at startup. A bad line is skipped, not fatal."""
    try:
        with open(SELFCAL_LOG) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return 0
    n = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            _selfcal["samples"].append(json.loads(line))
            n += 1
        except Exception:
            continue
    return n


@app.route("/gemini", methods=["GET", "POST"])
def r_gemini():
    """Ask the vision model about the frame the camera is looking at RIGHT NOW.

    The point is to be able to tune framing without burning a pick: aim the arm, hit
    this, and it says whether the object is in view, clipped, occluded or absent —
    the same question the centring step asks when it fails, minus the failure.
    """
    if GEMINI is None:
        return jsonify(ok=False, reason="gemini is not configured (no GOOGLE_API_KEY "
                                        "in the environment or .env.local, or the "
                                        "google-genai SDK is not installed)")
    label = request.args.get("label") or (_query_labels() or ["red cube"])[0]
    try:
        _j, rgb, _ = observe()
    except Exception as e:
        return jsonify(ok=False, reason=f"could not grab a frame ({type(e).__name__})")
    v = GEMINI.explain_miss(rgb, label)
    say(v.describe())
    return jsonify(ok=v.ok, answer=v.answer, reason=v.reason,
                   confidence=round(v.confidence, 2), latency_s=round(v.latency_s, 2),
                   means=MISS_REASONS.get(v.answer, ""), model=GEMINI.model,
                   calls=GEMINI.calls, failures=GEMINI.failures)


@app.route("/selfcal/reset", methods=["POST"])
def selfcal_reset():
    _selfcal["pending"] = None
    _selfcal["samples"] = []
    say("selfcal: buffer cleared")
    return jsonify(ok=True)


@app.route("/selfcal/observe", methods=["POST"])
def selfcal_observe():
    """Step 1: from here, where does the CAMERA say the object is?

    Uses the same localizer the map uses, so the calibration measures the error the
    map actually has — not a different, idealized one.
    """
    label = request.args.get("label") or (_query_labels() or ["red cube"])[0]
    finder, _tracker, lbl = _target_finder(label)
    q, rgb, _ = observe()
    T = T_cam_of(q)
    tr = finder(rgb, T)
    if tr is None:
        return jsonify(ok=False, reason=f"'{lbl}' not visible from here — move to where "
                                        f"the camera can see it, then try again")
    xy, rng, _size = obj_xy_2d(tr.bbox_xyxy, T, label=lbl)
    if xy is None:
        return jsonify(ok=False, reason="localization failed from this pose (range or "
                                        "geometry out of bounds) — try a different pose")
    u = (tr.bbox_xyxy[0] + tr.bbox_xyxy[2]) / 2.0
    v = (tr.bbox_xyxy[1] + tr.bbox_xyxy[3]) / 2.0
    off_axis = _selfcal_off_axis_deg((u, v))
    _selfcal["pending"] = {"label": lbl, "xy_observed": [float(xy[0]), float(xy[1])],
                           "range_m": float(rng), "off_axis_deg": off_axis, "t": time.time()}
    say(f"selfcal: observed '{lbl}' at r={rng*100:.1f}cm, {off_axis:.0f}deg off-axis — "
        f"now jog the tip down to TOUCH it, then press Mark True")
    return jsonify(ok=True, observed=_selfcal["pending"])


@app.route("/selfcal/touch", methods=["POST"])
def selfcal_touch():
    """Step 2: now the fingertip is ON the object. Where does the ARM say it is?"""
    p = _selfcal["pending"]
    if p is None:
        return jsonify(ok=False, reason="no pending observation — POST /selfcal/observe "
                                        "first, from a normal viewing distance")
    tip = _tip(observe()[0])
    sample = {"label": p["label"], "xy_true": [float(tip[0]), float(tip[1])],
             "xy_observed": p["xy_observed"], "off_axis_deg": p["off_axis_deg"],
             "range_m": p["range_m"]}
    _selfcal["samples"].append(sample)
    _selfcal["pending"] = None
    r_true = math.hypot(*sample["xy_true"])
    r_obs = math.hypot(*sample["xy_observed"])
    say(f"selfcal: sample #{len(_selfcal['samples'])} — camera said r={r_obs*100:.1f}cm, "
        f"arm found r={r_true*100:.1f}cm  (off by {(r_true-r_obs)*100:+.1f}cm)")
    return jsonify(ok=True, n=len(_selfcal["samples"]), sample=sample)


def _selfcal_load_samples():
    """The buffer as fittable samples, with each one's live corrections undone.

    Samples predating the auto-recorder carry no correction state; they were taken
    through /selfcal/observe, which calls obj_xy_2d directly, so they have whatever was
    dialled in at the time and no record of it. Defaulting to the identity treats them
    as raw — right whenever the knobs were untouched, which is the case they were
    collected in.
    """
    out = []
    for s in _selfcal["samples"]:
        obs = _selfcal_uncorrect(np.array(s["xy_observed"], dtype=np.float64),
                                 s.get("range_scale", 1.0),
                                 s.get("bearing_offset_deg", 0.0))
        out.append(LocalizationSample(np.array(s["xy_true"], dtype=np.float64), obs,
                                      s.get("off_axis_deg", 0.0),
                                      label=s.get("label", "")))
    return out


@app.route("/selfcal/status")
def selfcal_status():
    return jsonify(ok=True, n=len(_selfcal["samples"]), pending=_selfcal["pending"] is not None,
                   samples=_selfcal["samples"])


@app.route("/selfcal/fit", methods=["POST"])
def selfcal_fit():
    """Report what the samples imply, WITHOUT changing anything."""
    samples = _selfcal_load_samples()
    if len(samples) < MIN_SAMPLES:
        return jsonify(ok=False, reason=f"only {len(samples)} samples (need {MIN_SAMPLES}+, "
                                        f"spread across both range and bearing)")
    fit = fit_localization(samples)
    for line in fit.summary().split("\n"):
        say(f"selfcal: {line}")
    return jsonify(ok=True, trustworthy=fit.trustworthy, n=fit.n,
                   model={"range_scale": fit.model.range_scale,
                         "push_out_cm": round(fit.model.push_out_m * 100, 2),
                         "bearing_deg": round(fit.model.bearing_offset_deg, 2)},
                   rms_before_mm=round(fit.rms_before_m * 1000, 1),
                   rms_after_mm=round(fit.rms_after_m * 1000, 1),
                   warnings=fit.warnings, diagnosis=diagnose(samples))


@app.route("/selfcal/apply", methods=["POST"])
def selfcal_apply():
    """Write the fitted model to CFG's live knobs. Refuses an untrustworthy fit
    unless ?force=1 — installing numbers that do not describe the rig is exactly
    how the dial-turning this replaces got started."""
    samples = _selfcal_load_samples()
    if len(samples) < MIN_SAMPLES:
        return jsonify(ok=False, reason=f"only {len(samples)} samples (need {MIN_SAMPLES}+)")
    fit = fit_localization(samples)
    force = request.args.get("force") == "1"
    applied = apply_to_config(fit, CFG, force=force)
    if applied:
        say(f"selfcal APPLIED: {fit.model.describe()}")
    else:
        say(f"selfcal: NOT applied ({'; '.join(fit.warnings) or 'fit not trustworthy'}) "
            f"— pass ?force=1 to override")
    return jsonify(ok=applied, applied=applied, trustworthy=fit.trustworthy,
                   tune=CFG.as_dict(), warnings=fit.warnings)


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


@app.route("/handeye/graspfit", methods=["GET", "POST"])
def handeye_graspfit():
    """Fit the mount against the objects grasps have proved the position of.

    GET reports what the samples imply and changes nothing. POST?apply=1 writes it.

    This is the third fitter, and it exists because the other two cannot be right by
    construction on this rig: /calib solves for the transform AND the target together
    and drifts along a flat direction while reporting a perfect residual, and
    /calibmount asks only that the viewpoints agree — which they did, to 0.3cm, on the
    wrong place, making picking worse. Neither objective knows where anything actually
    is. A grasp does.
    """
    global T_ee_cam
    samples = load_grasp_samples()
    fit = fit_to_known_points(samples, GEOM, T_seed=T_ee_cam)
    body = {"ok": bool(fit.converged), "n": len(samples),
            "rms_px": round(float(fit.rms_px), 2) if fit.rms_px == fit.rms_px else None,
            "rms_px_before": round(float(fit.before.get("rms_px", float("nan"))), 2),
            "tf": tf_to_string(fit.T_ee_cam), "reason": fit.reason, "applied": False}
    if request.method == "POST" and request.args.get("apply") == "1":
        if not fit.converged:
            body["reason"] = f"refusing to apply: {fit.reason}"
            return jsonify(body), 400
        T_ee_cam = fit.T_ee_cam
        _sync_geometry()
        save_hand_eye(TF_FILE, fit)
        _LOC[0] = None                     # localizers cached the old geometry
        say(f"handeye: mount fitted from {len(samples)} grasps, "
            f"{fit.before.get('rms_px', 0):.1f}px -> {fit.rms_px:.1f}px reprojection")
        set_phase("IDLE", f"mount fitted from {len(samples)} grasps "
                          f"({fit.rms_px:.1f}px)")
        body["applied"] = True
    return jsonify(body)


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


def reads_as_a_task(text):
    """The parsed steps if `text` names a destination ("red on green"), else None.

    THE TRAP THIS CLOSES. "red on green" is a placement instruction, but typed into
    the detection-query box it was accepted as a *vocabulary* and Start then picked
    whichever label came first — so the arm grabbed the red cube, reported DONE, and
    never went near the green one. Nothing was broken; the phrase had simply been read
    as two class names instead of as a task. A real run shows exactly that:

        PICK START  target='red'  query='red cube, green cube'
        ... PICK SUCCESS ... [DONE] red cube picked

    The phrase is unambiguous — no detection vocabulary needs the word "on" — so both
    entry points now honour it rather than one of them silently doing half the job.
    """
    try:
        return parse_task(text)
    except Abort:
        return None


def pending_instruction(st):
    """What the operator last ASKED for — not what the detector ended up running.

    state["query"] cannot answer this. The detector thread reports the vocabulary it
    actually applied back into that key (on_query_change in main()), so a task phrase
    put there survives only until the detector next cycles: "red on green" silently
    becomes "red cube, green cube", and Start sees an ordinary two-class query and
    runs a plain pick. That is the original bug wearing a different hat, and it only
    shows up on a live server, which is where it was found.
    """
    return str(st.get("intent") or st.get("query") or "")

def task_vocabulary(steps):
    """Detection query covering every object a task mentions, picks and destinations.

    A task needs BOTH objects on the map at once: the pick to drive at, the
    destination to place on. Deriving the vocabulary from the task is what guarantees
    that — otherwise the map only ever holds whatever the query happened to name, and
    run_task aborts with "no 'green' on the 2D map" through no fault of the operator.

    Bare colour words are expanded to the cube class they almost always mean
    ("green" -> "green cube"), because a one-word colour is a weak YOLO-World prompt.
    """
    out = []
    for pick, dest in steps:
        for lab in (pick, dest):
            lab = str(lab).strip().lower()
            if not lab:
                continue
            if lab not in CLASS_META and f"{lab} cube" in CLASS_META:
                lab = f"{lab} cube"
            if lab not in out:
                out.append(lab)
    return ", ".join(out)

def map_label_matches(want, label) -> bool:
    """Does this map entry's label mean the thing the caller asked for?

    Loose in BOTH directions on purpose: the operator types "red", the map holds
    "red cube", and _target_finder hands the mission the bare colour "red" while the
    detector wrote "red cube".

    ONE function because the mismatch is what broke it. The `picked` flag was SET with
    a loose test (`label.split()[0] in o["label"]`, so "red" marked "red cube") and
    RETIRED with an exact one (`o["label"] == carry_label`, so "red cube" == "red" was
    never true). The flag went on and never came off, `_find_map_tag` skips picked
    entries, and so every object the arm successfully picked became permanently
    invisible to tasks: "no 'red cube' on the 2D map" while the map plainly held one
    and the camera was looking straight at it.
    """
    a, b = str(want).strip().lower(), str(label).strip().lower()
    return bool(a) and bool(b) and (a in b or b in a)


def clear_picked_flags(label=None):
    """Un-flag map entries marked as 'in the jaws'. Returns how many were cleared.

    ``picked`` means "this entry is the object currently held" — so when the jaws are
    empty, by definition nothing is picked. Clearing it here rather than only on a
    successful place is what makes it self-correcting: a pick that missed, a place, a
    dropped object, and a carry cleared by the grasp check all end at the same place,
    and none of them should leave the map poisoned.
    """
    n = 0
    with w2d_lock:
        for _t, o in WORLD.objs.items():
            if o.get("picked") and (label is None or map_label_matches(label, o["label"])):
                o["picked"] = False
                n += 1
    return n

def _live_range_and_tip(label):
    """(apparent range to `label` right now, current tip position), or (None, None).

    Apparent size is the referee because it is transform-free: ``fx * real_width /
    pixel_width`` needs no hand-eye rotation, no table plane and no arm pose. That is
    exactly what a ghost map entry cannot survive — an entry 52cm away is refuted on
    the spot by a cube filling 100 pixels.
    """
    try:
        j, rgb, _ = observe(overlay=False)
        finder, _tracker, _lab = _target_finder(label)
        tr = finder(rgb, T_cam_of(j))
        if tr is None or tr.clipped:
            return None, None
        rng = apparent_range_m(tr.bbox_xyxy, label)
        if rng is None:
            return None, None
        tip = np.asarray(kin.forward_kinematics(np.asarray(j, np.float64)),
                         dtype=np.float64)[:3, 3]
        return float(rng), tip
    except Exception:
        return None, None


def _find_map_tag(label):
    """The map entry for `label` that best matches what the camera can see NOW.

    Loose on the label: the user types 'red', the map holds 'red cube'.

    WHY NOT SIMPLY THE BEST-SUPPORTED ENTRY, which is what this did. Observations taken
    from different arm poses are rotated to different places by a hand-eye transform
    that is wrong, so one physical object grows several map entries too far apart to
    merge — and the ghost is not the lonely one. Measured, on a map cleared seconds
    earlier and rebuilt from a single pose:

        'green cube'  base r=37.1cm   18.6cm from the tip   n=40   <- the real cube
        'green cube'  base r=52.0cm   34.3cm from the tip   n=98   <- the ghost, and
                                                                      better supported

    Picking by ``n`` chose the ghost, the arm drove at 52cm, hit its 47cm limit and
    reported "out of reach" — an accurate message about entirely the wrong thing.

    So: when the object is in view, prefer the entry whose distance from the gripper
    matches how big the object actually looks. Support count only breaks ties among
    entries the camera cannot arbitrate, and is the whole rule when nothing is visible.
    """
    want = str(label).strip().lower()
    rng, tip = _live_range_and_tip(label)
    with w2d_lock:
        cands = [(t, o) for t, o in WORLD.objs.items()
                 if not o.get("picked") and map_label_matches(want, o["label"])]
        if not cands:
            return None
        if rng is None or tip is None:
            return max(cands, key=lambda c: c[1]["n"])[0]
        def mismatch(item):
            o = item[1]
            d = float(np.linalg.norm(
                np.array([o["xy"][0], o["xy"][1], TABLE_Z0], dtype=np.float64) - tip))
            return abs(d - rng)
        best = min(cands, key=mismatch)
        gap = mismatch(best)
        # If even the best entry disagrees with the picture by more than the object is
        # wide, the camera is not arbitrating anything useful — say so rather than
        # quietly acting on it.
        tol = max(3.0 * float(PRIORS.size_m(label) or 0.05), 0.12)
        if gap > tol:
            say(f"map: no '{want}' entry matches what the camera sees "
                f"(closest is {gap*100:.0f}cm out) — using the best-supported one")
            return max(cands, key=lambda c: c[1]["n"])[0]
        if len(cands) > 1:
            worst = max(cands, key=mismatch)
            if worst[0] != best[0]:
                say(f"map: {len(cands)} '{want}' entries; picked the one the camera "
                    f"agrees with ({gap*100:.0f}cm out, vs {mismatch(worst)*100:.0f}cm)")
        return best[0]

def run_task(text):
    """Run a typed task: pick each object and place it on its destination."""
    steps = parse_task(text)
    say(f"task: {len(steps)} step(s) — " +
        ", ".join(f"{p} on {d}" for p, d in steps))
    # Make sure the detector can SEE everything this task names before checking the
    # map for it. A task's destination is often nothing the current query mentions, so
    # without this the map cannot hold it and the check below fails on the operator's
    # behalf rather than on the robot's.
    vocab = task_vocabulary(steps)
    with lock:
        running_vocab = state.get("vocabulary") or state.get("query") or ""
    if vocab and vocab != running_vocab:
        set_phase("TASK", f"loading the task vocabulary: {vocab}")
        _apply_query_now(vocab)
        with lock:
            state["vocabulary"] = vocab

    # Fail before moving if any object is missing from the map, rather than picking
    # something up and then discovering there is nowhere to put it. A missing object
    # is far more often "nobody has scanned yet" than "it is not on the table", so
    # scan once and look again before giving up on it.
    def _missing():
        want = []
        for i, (pick, dest) in enumerate(steps, 1):
            for lab in (dest, pick):
                if _find_map_tag(lab) is None and (i, lab) not in want:
                    want.append((i, lab))
        return want

    if _missing():
        say("task: not everything is on the map yet — scanning first")
        scan_2d(broad=False)
    gone = _missing()
    if gone:
        raise Abort("step {}: no '{}' on the 2D map after a scan — check it is on the "
                    "table and in view".format(*gone[0]))

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
    # reraise=True: pressed on its own, Start swallows a failed pick and leaves the
    # reason in the phase line, which is right for one button press. In a task the
    # caller has to KNOW, or every failure downstream is reported as the generic
    # "pick failed - not placing" and the real cause ("cannot see 'red'", an IK miss,
    # a stop) is lost behind it.
    run_mission(target_label=target_label, reraise=True)
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
    WORLD.clear()
    say("2D map cleared")
    return jsonify(ok=True)


def merge_radius_watch():
    """Adopt the merge radius the map's own measured noise implies.

    The radius has to cover the localization scatter or one object spawns a fresh tag
    per view — that is what produced a 16-ghost map. The scatter was never measured, so
    the number was guessed at 14cm and left. The map can measure it: every update's
    residual against the running estimate is one sample of that error, because the
    object did not move, the estimate did.

    Adopted only once there is enough evidence, only when it differs enough to matter,
    and never below the configured floor — a radius that shrinks below the real noise
    is the failure this is meant to prevent, so the measurement may widen it but not
    tighten it past the default.
    """
    floor = WORLD.merge_m
    while True:
        time.sleep(60.0)
        try:
            want = WORLD.suggested_merge_radius(k=3.0, min_samples=60)
            if want is None:
                continue
            want = max(float(want), floor)
            if abs(want - WORLD.merge_m) < 0.01:
                continue
            scatter = WORLD.observation_scatter()
            say(f"map: merge radius {WORLD.merge_m*100:.0f}cm -> {want*100:.0f}cm "
                f"(measured localization scatter {scatter*1000:.0f}mm, 3 sigma)")
            WORLD.merge_m = want
        except Exception as e:
            say(f"merge radius watch: {type(e).__name__}: {e}")


def idle_view():
    """Keep the FPV and the map warm while nothing else is driving.

    Bails out on `shutting_down` so it is not mid-read when the camera is released.
    """
    while not shutting_down.is_set():
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


def clear_gripper_overload(ids=None):
    """Clear a latched overload on ANY motor with a raw torque cycle, before
    lerobot's handshake reads hit the error.

    It used to touch only the gripper (id 6), which latches easily on worn gears -
    but shoulder_lift (id 2) latches too after sustained holding, and THAT is what
    kills the server: `Failed to read 'Min_Position_Limit' on id_=2 ... Overload
    error!` at connect, before anything is running to catch it. Cycle them all.

    The bus layout (ids, baud, register numbers) comes from the profile, so an arm on
    a different servo bus declares its own instead of inheriting Feetech's.
    """
    bus = ARM.bus
    if bus is None:
        return                      # this arm exposes no raw servo bus
    if ids is None:
        ids = bus.motor_ids
    try:
        import scservo_sdk as scs
        ph = scs.PortHandler(ARM_PORT)
        if not ph.openPort():
            return
        ph.setBaudRate(bus.baud)
        pk = scs.PacketHandler(0)
        cleared = []
        for mid in ids:
            pk.write1ByteTxRx(ph, mid, bus.torque_register, 0)   # torque off
            time.sleep(0.12)
            pk.write1ByteTxRx(ph, mid, bus.torque_register, 1)   # on: clears the latch
            time.sleep(0.06)
            _pos, _c, err = pk.read2ByteTxRx(ph, mid, bus.status_register)
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
        _meshes = load_link_visual_meshes_cached(ARM.mesh_path) or {}
        _chain = [n for n, _ in (kin.get_link_transforms_chain(np.zeros(len(ARM_MOTORS))) or [])]
        _hit = [n for n in _chain if n in _meshes]
        if _hit:
            say(f"Rerun 3D: URDF meshes OK — {len(_hit)}/{len(_chain)} links "
                f"({ARM.mesh_path}): {', '.join(_hit)}")
        else:
            say(f"Rerun 3D: NO URDF meshes matched — falling back to stick figure. "
                f"mesh_dir={ARM.mesh_path} meshes={list(_meshes)} chain={_chain}")
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


_shut = [False]
# Set first thing in _shutdown so the frame-grabbing threads stop touching the camera
# BEFORE it is disconnected. Without this they keep calling into a device that is being
# torn down, which is what leaves it booted-with-no-owner and needing a USB re-plug.
shutting_down = threading.Event()


def _shutdown(why=""):
    """Release the camera and the servo bus before the process goes away.

    THIS IS NOT HOUSEKEEPING. An OAK-D whose owner dies without disconnecting stays
    BOOTED with no one holding it: Windows keeps enumerating it, depthai skips it with
    "X_LINK_BOOTED ... X_LINK_ERROR", and the next start fails with "No available
    devices" until somebody physically unplugs it. That is what left this robot blind
    for 41 hours — the server kept serving, the camera was gone, and nothing said so.

    Idempotent, and every step is independently guarded: a shutdown path that raises
    halfway leaves exactly the wedged device it was written to prevent.
    """
    if _shut[0]:
        return
    _shut[0] = True
    print(f"shutting down ({why}) — releasing camera and bus", flush=True)
    try:
        shutting_down.set()
        stop_flag.set()
        if DETECT is not None:
            DETECT.stop()
    except Exception:
        pass
    # Let the frame grabbers notice and fall out of their loops. Disconnecting the
    # camera underneath a thread still reading from it is what wedges the device.
    time.sleep(0.6)
    # Robot first: its disconnect owns the camera it was given, so releasing the camera
    # out from under it leaves the robot reporting "not connected" and skipping its own
    # bus teardown. The camera call after it is belt-and-braces for the case where the
    # robot never finished connecting.
    for label, fn in (("robot", lambda: robot.disconnect()),
                      ("camera", lambda: cam.disconnect())):
        try:
            fn()
            print(f"  {label} released", flush=True)
        except Exception as e:
            print(f"  {label} release: {type(e).__name__}: {e}", flush=True)


def _install_shutdown_handlers():
    """Run the release on normal exit and on a polite stop.

    WHAT THIS DOES AND DOES NOT COVER, on Windows specifically:
      * normal exit / an unhandled exception  -> atexit fires. Covered.
      * Ctrl-C in a console                   -> SIGINT fires. Covered.
      * POST /shutdown                        -> calls _shutdown directly. Covered.
      * PowerShell Stop-Process               -> NOT covered. It is a hard
        TerminateProcess, the same as SIGKILL, and no handler runs. Python on Windows
        also does not deliver an externally-sent SIGTERM to a handler.

    That last case is how this server has usually been stopped, and it is exactly the
    case that wedges the camera. Use POST /shutdown instead.
    """
    import atexit
    import signal

    atexit.register(_shutdown, "atexit")
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda s, _f: (_shutdown(f"signal {s}"), os._exit(0)))
        except (ValueError, OSError):
            pass            # not on the main thread, or unsupported on this platform


def connect_hardware():
    """Import the lerobot driver stack and return the pieces main() needs.

    Deferred to here on purpose: lerobot is a HARDWARE dependency. Importing it at
    module scope meant this file could not be imported — for a test, for --help, or
    to read the routes — without a lerobot checkout on the machine. Everything above
    this line is RAX and numpy.
    """
    try:
        from lerobot.robots.utils import make_robot_from_config
        from lerobot.robots.so_follower import SO101FollowerConfig
        from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig
        from lerobot.perception.yolo_world import YoloWorldDetector
        from lerobot.motors.feetech.feetech import FeetechMotorsBus as _FTBus
    except ImportError as e:
        raise SystemExit(
            "lerobot is required to talk to the arm and the OAK-D, but it did not "
            f"import ({e}).\n"
            "  pip install lerobot   -- or, for a source checkout:\n"
            "  set RAX_LEROBOT_SRC=<path to the lerobot repo>\n"
            "Everything else in RAX runs without it: python -m rax.grasp --arm mock"
        ) from e

    # The gripper servo (ID 6) replies too slowly for lerobot's handshake ping timeout
    # (raw pings see it 100%; lerobot's missed 48/48). Skip the existence assert: sync
    # WRITES need no ACK, and every read in this file already tolerates a miss
    # (gripper_current returns None, observe retries).
    _FTBus._handshake = lambda self: None
    return make_robot_from_config, SO101FollowerConfig, OAKDCameraConfig, YoloWorldDetector


def main():
    global robot, kin, fx, fy, cx0, cy0, detector, cam, DETECT
    (make_robot_from_config, SO101FollowerConfig,
     OAKDCameraConfig, YoloWorldDetector) = connect_hardware()
    clear_gripper_overload()
    say("connecting robot + camera…")
    # The gripper servo (ID 6) answers intermittently — a marginal cable. Retry
    # the whole handshake with a FRESH robot object each time (a failed connect
    # leaves the bus in a half-open state that refuses reconnection).
    for attempt in range(6):
        robot = make_robot_from_config(SO101FollowerConfig(
            port=ARM_PORT, id="so101_follower",
            # DEPTH OFF. The pick locates the cube purely geometrically (cast its
            # pixel ray onto the table plane), so stereo depth buys us nothing — and
            # the stereo pipeline is what kept crashing the OAK-D mid-run
            # (X_LINK_ERROR + firmware crash dump, taking the whole server with it).
            # Dropping it also roughly halves the USB bandwidth. read_depth_m()
            # degrades gracefully to None.
            cameras={"front": OAKDCameraConfig(
                fps=ARM.camera.fps, width=ARM.camera.width,
                height=ARM.camera.height, use_depth=ARM.camera.use_depth)},
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
    # The arm's own URDF, from the profile. This used to reach into the external
    # lerobot checkout; that copy only adds two FIXED camera frames, so FK to
    # gripper_frame_link is bit-identical (verified over 300 random poses) — and
    # owning the file here is what lets a non-lerobot arm supply its own.
    kin = make_kinematics(ARM.urdf_path, ARM.ee_frame, list(ARM.joint_names))
    cam = robot.cameras["front"]
    say("colour stream only (depth OFF) — the pick works by eye, not by stereo")
    _fb = ARM.camera.intrinsics_fallback
    intr = {"fx": _fb[0], "fy": _fb[1], "cx": _fb[2], "cy": _fb[3]}
    if hasattr(cam, "get_depth_intrinsics"):
        try:
            intr = dict(cam.get_depth_intrinsics())
        except Exception:
            pass
    fx, fy, cx0, cy0 = (float(intr[k]) for k in ("fx", "fy", "cx", "cy"))
    _sync_geometry()          # the camera reported its real intrinsics

    # A hand-eye TF we FITTED (see calibrate_handeye) overrides the constant.
    load_floor_plane()
    _tfd = load_tf_override()
    if _tfd:
        _rms = _tfd.get("rms_px", 0)
        say(f"hand-eye: using CALIBRATED TF from {_tfd.get('fitted','?')} "
            f"(reprojection {_rms:.0f}px) -> {_tfd['tf']}")
        # rms_px is written by save_hand_eye from an actual reprojection fit, and a
        # real fit never lands at exactly zero. A 0 means this file did NOT come from
        # either fitter in this server — the shipped one says
        # "source": "fingertip-constrained rotation fix", which nothing here writes —
        # so it carries no measured quality at all and "reprojection 0px" reads as
        # perfect when it means unmeasured. Say which it is.
        if not _rms:
            say("hand-eye: WARNING — that TF has no measured reprojection error "
                f"(source={_tfd.get('source','?')!r}). It was not produced by "
                "calibrate_handeye. Run /handeye to get a real residual before "
                "trusting any range it produces.")
    global GEMINI
    if gemini_available():
        GEMINI = GeminiVision(log=say)
        say(f"gemini: {GEMINI.model} available. Grasp checks can now CLEAR a carry "
            f"(a confident 'empty' only, never the reverse); miss diagnosis "
            f"stays advisory. RAX_GRASP_TRUST_EMPTY=0 to disable")
    else:
        say("gemini: no GOOGLE_API_KEY or SDK — miss diagnosis and grasp checks are off")
    _n_cal = _selfcal_load_log()
    if _n_cal:
        say(f"selfcal: {_n_cal} sample(s) carried over from previous runs "
            f"(need {MIN_SAMPLES}+ to fit; POST /selfcal/fit to see where it stands)")
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
    detector = YoloWorldDetector(YOLO_WEIGHTS, conf=DET_CONF[0], imgsz=PICK_IMGSZ,
                                 color_filter_min_frac=0.0)
    _q0 = "red cube, green cube"      # colour-ONLY: colourless terms like "toy
                                      # block"/"box" make YOLO-World fire on the
                                      # green cube and the mission then dives to it
    detector.set_query(_q0)
    with lock:
        state["query"] = _q0
    # The detection loop, query handling, NMS and the per-label trackers all live in
    # the service now. It reports the query back into `state` so /status is unchanged.
    DETECT = DetectorService(
        detector,
        frame_source=lambda: latest_rgb[0],
        config=DETECT_CFG,
        colour_ok=_colour_ok,
        log=say,
        on_query_change=lambda q: state.update(query=q),
    )
    DETECT.query = _q0
    DETECT.start()
    threading.Thread(target=idle_view, daemon=True).start()
    threading.Thread(target=jog_loop, daemon=True).start()
    threading.Thread(target=idle_relax_watch, daemon=True).start()
    threading.Thread(target=merge_radius_watch, daemon=True).start()
    start_rerun()
    threading.Thread(target=rerun_thread, daemon=True).start()
    start_tunnel()
    _install_shutdown_handlers()
    set_phase("IDLE", "ready — press Start")
    say(f"UI: http://100.110.89.78:{PORT}  (Tailscale)")
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True)
    finally:
        _shutdown("app.run returned")


if __name__ == "__main__":
    main()
