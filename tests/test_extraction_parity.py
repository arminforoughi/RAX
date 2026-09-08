"""Golden-value parity tests for the functions extracted out of the monolith.

The modularization moves ~1,100 lines of proven logic out of ``stack_mission2.py``
into packages, and the result is ``mission_server.py`` — the original is kept intact
and runnable so the two can be compared on the same hardware. Most of that logic is *pure* — it maps numbers to numbers — and much of it
encodes specific, expensively-learned corrections (the limit-clamping in
``_ik_hold_pitch``, the elongated-object branch in ``obj_xy_2d``). Moving it is only
safe if "the extracted version does exactly what the original did" is a fact we can
check, not a hope.

So: before touching anything, run every one of those functions over a fixed grid of
inputs and freeze the answers in ``tests/golden/extraction_parity.json``. After each
extraction phase, run this again. Any drift is a bug in the move, and it names the
function.

The fixture pins every module global the functions read (intrinsics, hand-eye TF,
table height, the live-tunable knobs) so the goldens do not depend on whatever
``handeye_tf.json`` / ``floor_plane.json`` happen to hold on disk.

Runs headless: lerobot falls back to pure-numpy URDF kinematics when placo is absent,
so no robot and no placo are needed.

    python tests/test_extraction_parity.py --record    # freeze goldens (do this ONCE, on known-good code)
    python tests/test_extraction_parity.py             # check against the goldens
    pytest tests/test_extraction_parity.py             # same, under pytest if installed

Re-baselines
------------
Deliberate behaviour changes are recorded here, with what moved and by how much.
Anything not on this list is a bug in an extraction.

* Phase 1 — joint limits now read from the URDF instead of the hand-transcribed
  table at the old ``the original stack_mission2.py:4661``. The transcription was rounded to one
  decimal, so the true bounds differ slightly and in both directions: elbow_flex
  ``96.8 -> 96.8299`` (looser), wrist_flex ``95.0 -> 94.9998`` and shoulder_pan
  ``110.0 -> 109.9999`` (tighter). Effect, measured over this grid: joint angles move
  by at most 0.03 deg, ``plan_grasp_pitch`` residuals by 1e-7 m, and only the three
  limit-dependent groups (``ik_hold_pitch``, ``slave_wflex``, ``plan_grasp_pitch``)
  changed at all — every geometry and class-prior group stayed bit-identical.
  0.03 deg is below one STS3215 servo step (0.088 deg), so no commanded pose changes.

* Seed derivation — the SO-101 profile's five hand-added IK seeds replaced by the one
  derived by ``manipulation.arms.workspace`` (the mirrored elbow branch). Different
  seeds means the solver lands on different valid branches, so ``ik_hold_pitch`` poses
  move. What matters is that nothing got worse: 36/60 cases solve before AND after,
  ZERO regressions (nothing reachable became unreachable), zero newly reachable, and
  ``plan_grasp_pitch`` chose an identical angle in all 10 cases with the same worst
  residual (0.56 mm). The 16 cases whose residual grew were already beyond tolerance —
  a different failed branch, which changes nothing the arm does. Independently
  measured over 700 poses across five heights: identical coverage, 2.6x faster.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

GOLDEN = REPO / "tests" / "golden" / "extraction_parity.json"

# The IK and the geometry are deterministic, so a faithful extraction should be
# bit-identical. Allow only enough slack to absorb numpy reduction-order changes.
ATOL = 1e-9

# Pinned fixture values ------------------------------------------------------------
# The documented intrinsics fallback (the original stack_mission2.py:5822), so the goldens do not
# depend on which OAK-D happens to be plugged in.
FX = FY = 517.0
CX = 329.5
CY = 231.4
URDF = REPO / "src" / "rax" / "robots" / "arms" / "lerobot_so101" / "SO101" / "so101_new_calib.urdf"
EE_FRAME = "gripper_frame_link"
ARM_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


#: The monolith these goldens were extracted from now lives with the other demos.
MISSION_SERVER_DIR = REPO / "examples" / "mission_server"


def _load_module():
    """Import ``mission_server`` and pin every global the pure functions read.

    Skips rather than fails when the demo server will not import. It depends on a
    lerobot checkout that only the original hardware machine has, and a contributor
    with no SO-101 should still be able to run the suite — the packages this pins
    against are covered by the other tests, which need nothing but numpy.
    """
    if str(MISSION_SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(MISSION_SERVER_DIR))
    try:
        import mission_server as S
        from rax.manipulation.arms.kinematics import make_kinematics
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"the reference monolith needs a lerobot checkout: {exc}")

    S.kin = make_kinematics(str(URDF), EE_FRAME, ARM_MOTORS)
    S.fx, S.fy, S.cx0, S.cy0 = FX, FY, CX, CY
    # Pin the hand-eye to the in-source constant, NOT to handeye_tf.json — that file
    # is re-fitted by /calib and would silently move every golden.
    S.T_ee_cam = S.parse_tf(S.TF)
    # Push both into the shared CameraGeometry the way the running server does. Doing
    # this explicitly matters: the pinned intrinsics happen to equal the profile's
    # fallback, so without it the geometry goldens would pass even if the sync were
    # broken — a test passing for the wrong reason.
    S._sync_geometry()
    S.FLOOR.set(0.0, 0.0, -0.022)
    S.TABLE_Z[0] = S.Z_TABLE
    # The approach tunables all live on one config object now.
    S.CFG.push_out_m = 0.0
    S.CFG.range_scale = 1.0
    S.CFG.bearing_offset_deg = 0.0
    S.PRIORS.fallback_edge_m = 0.0508
    # These goldens freeze the ORIGINAL monolith's answers, and it walked the
    # size-derived distance along the sightline rather than treating it as axial depth.
    # The running server no longer does — the sightline placement puts an off-axis
    # object too close by cos(angle), measured at 11cm on a real cube, enough to report
    # two objects in the wrong order. Pinning it off here keeps this test measuring what
    # it is for (did the EXTRACTION change behaviour) rather than re-litigating a
    # deliberate fix. tests/test_axial_depth.py covers the new behaviour.
    S.localizers().apparent.axial_depth = False
    return S


# Deterministic input grids ---------------------------------------------------------
JOINT_POSES = [
    [-14.1, -99.1, 90.8, 33.2, -4.7],   # HOME
    [5.0, 37.1, 48.1, -40.4, 90.0],     # VIEW
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [30.0, -20.0, 40.0, 10.0, 45.0],
    [-45.0, 60.0, -30.0, 25.0, -90.0],
]

TARGET_POINTS = [
    [0.22, 0.05, 0.02],
    [0.15, 0.00, 0.05],
    [0.30, -0.10, 0.02],
    [0.12, 0.12, 0.08],
    [0.40, 0.05, 0.02],   # near the edge of reach
]

PIXELS = [(320.0, 240.0), (440.0, 394.0), (100.0, 80.0), (600.0, 450.0), (10.0, 470.0)]

LABELS = [
    "red cube", "green cube", "blue cube", "cup", "bottle", "pen", "knife",
    "scissors", "banana", "apple", "mouse", "cell phone", "book", "spoon",
    "unknown thing that is not in the table",
]

BBOXES = [
    (300.0, 220.0, 340.0, 260.0),   # small, square
    (200.0, 150.0, 400.0, 350.0),   # large, square
    (100.0, 200.0, 380.0, 240.0),   # wide and flat -> the elongated branch
    (300.0, 100.0, 340.0, 400.0),   # tall and thin -> the elongated branch
    (0.0, 0.0, 6.0, 6.0),           # degenerate, below the 4 px floor
]


# Case generation -------------------------------------------------------------------
def collect(S) -> dict:
    """Run every pinned function over its grid and return a JSON-able result tree."""
    out: dict[str, object] = {}

    # --- _slave_wflex: the pitch-slaving identity ---
    out["slave_wflex"] = [
        S._slave_wflex(j1, j2, pitch)
        for j1 in (-99.1, -20.0, 0.0, 40.0)
        for j2 in (90.8, 0.0, -30.0)
        for pitch in (0.0, 45.0, 70.0, 90.0, -30.0)
    ]

    # --- class priors ---
    out["class_size_m"] = [S.class_size_m(l) for l in LABELS]
    out["class_height_m"] = [S.class_height_m(l) for l in LABELS]
    out["class_meta"] = [
        {k: (list(v) if isinstance(v, (list, tuple)) else v)
         for k, v in sorted(S.class_meta(l).items())}
        for l in LABELS
    ]

    # --- floor plane ---
    out["floor_z"] = [
        S.floor_z(x, y)
        for x in (0.0, 0.1, 0.25, 0.4) for y in (-0.2, 0.0, 0.15)
    ]

    # --- push_out_radial (identity at PUSH_OUT=0, and a non-zero probe) ---
    push = []
    for val in (0.0, 0.05):
        S.CFG.push_out_m = val
        push += [S.push_out_radial(np.array(p)).tolist() for p in TARGET_POINTS]
    S.CFG.push_out_m = 0.0
    out["push_out_radial"] = push

    # --- grasp roll from yaw ---
    out["grasp_roll_for_yaw"] = [
        S.grasp_roll_for_yaw(yaw, (x, y))
        for yaw in (0.0, 30.0, 90.0, 135.0, -45.0)
        for (x, y) in ((0.25, 0.0), (0.15, 0.15), (0.2, -0.1))
    ]

    # --- camera geometry: FK -> T_base_cam -> project / backproject / ray ---
    tcam = {}
    proj, loc3d, ray, tips = [], [], [], []
    for q in JOINT_POSES:
        qa = np.array(q, dtype=np.float64)
        T = np.asarray(S.T_cam_of(qa))
        tcam[str(q)] = T.tolist()
        tips.append(S.tip_pixel(qa))
        for p in TARGET_POINTS:
            proj.append(S.project_base(np.array(p), T))
        for uv in PIXELS:
            loc3d.append(S.locate_3d(uv, 0.25, T).tolist())
            r = S.ray_to_table(uv, T)
            ray.append(None if r is None else np.asarray(r).tolist())
    out["T_cam_of"] = tcam
    out["tip_pixel"] = tips
    out["project_base"] = proj
    out["locate_3d"] = loc3d
    out["ray_to_table"] = ray

    # --- obj_xy_2d: apparent-size AND the elongated long-axis branch ---
    objxy = []
    for q in JOINT_POSES[:3]:
        T = np.asarray(S.T_cam_of(np.array(q, dtype=np.float64)))
        for bbox in BBOXES:
            for label in ("red cube", "cup", "pen", "knife", None):
                for z_m in (None, 0.30):
                    xy, rng, size = S.obj_xy_2d(bbox, T, z_m=z_m, label=label)
                    objxy.append({
                        "xy": None if xy is None else np.asarray(xy).tolist(),
                        "rng": None if (rng is None or math.isnan(rng)) else float(rng),
                        "size": float(size),
                    })
    out["obj_xy_2d"] = objxy

    # --- _ik_hold_pitch: the solver the whole approach rides on ---
    ik = []
    for q in JOINT_POSES[:3]:
        qa = np.array(q, dtype=np.float64)
        for p in TARGET_POINTS:
            for pitch in (0.0, 45.0, 70.0, 90.0):
                qs, e = S._ik_hold_pitch(qa, np.array(p), pitch, float(qa[4]), ret_err=True)
                ik.append({"q": np.asarray(qs).tolist(), "e": float(e)})
    out["ik_hold_pitch"] = ik

    # --- plan_grasp_pitch: which pitch the arm commits to, and the residual ---
    plans = []
    for q in JOINT_POSES[:2]:
        qa = np.array(q, dtype=np.float64)
        for p in TARGET_POINTS:
            pitch, e = S.plan_grasp_pitch(np.array(p), qa)
            plans.append({"pitch": None if pitch is None else float(pitch), "e": float(e)})
    out["plan_grasp_pitch"] = plans

    return out


# Comparison ------------------------------------------------------------------------
def _jsonable(x):
    """Normalize to what a JSON round-trip would yield, so a freshly-collected tree and
    a loaded golden compare structurally (tuples and ndarrays both become lists)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _diff(path, a, b, bad):
    """Recursively compare two JSON trees, appending human-readable diffs to ``bad``."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                bad.append(f"{path}.{k}: present in only one side")
            else:
                _diff(f"{path}.{k}", a[k], b[k], bad)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            bad.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(f"{path}[{i}]", x, y, bad)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (math.isnan(a) and math.isnan(b)) and abs(a - b) > ATOL:
            bad.append(f"{path}: {a!r} != {b!r}  (delta {a - b:+.3e})")
    elif a != b:
        bad.append(f"{path}: {a!r} != {b!r}")


def compare(current: dict, golden: dict) -> list[str]:
    bad: list[str] = []
    _diff("", _jsonable(current), _jsonable(golden), bad)
    return bad


def test_extraction_parity():
    """The pinned functions still produce the frozen answers."""
    assert GOLDEN.exists(), (
        f"no goldens at {GOLDEN} — run `python {__file__} --record` on known-good code first"
    )
    S = _load_module()
    bad = compare(collect(S), json.loads(GOLDEN.read_text()))
    assert not bad, "extraction changed behaviour:\n  " + "\n  ".join(bad[:40])


def test_urdf_limits_match_hardcoded():
    """The hand-transcribed J_LO/J_HI really are the URDF's limits.

    ``the original stack_mission2.py:4661`` transcribes the joint limits by hand and the comment
    above it explains that an IK which does not know them "is not an IK, it is a
    wish". The extraction reads them from the URDF instead — so first prove the two
    agree, rather than assuming it.
    """
    import xml.etree.ElementTree as ET

    S = _load_module()
    root = ET.parse(URDF).getroot()
    lim = {}
    for j in root.findall("joint"):
        node = j.find("limit")
        if node is not None:
            lim[j.get("name")] = (
                math.degrees(float(node.get("lower", 0.0))),
                math.degrees(float(node.get("upper", 0.0))),
            )
    missing = [m for m in ARM_MOTORS if m not in lim]
    assert not missing, f"URDF has no <limit> for {missing}"

    lo = np.array([lim[m][0] for m in ARM_MOTORS])
    hi = np.array([lim[m][1] for m in ARM_MOTORS])
    # The transcription is rounded to 1 decimal (96.8 vs 96.83), so compare at that.
    assert np.allclose(lo, S.J_LO, atol=0.05), f"lower: urdf {lo} vs J_LO {S.J_LO}"
    assert np.allclose(hi, S.J_HI, atol=0.05), f"upper: urdf {hi} vs J_HI {S.J_HI}"


def test_world2d_snapshot_serializes_a_populated_map():
    """The server's snapshot wrapper, with objects actually in the map.

    This exists because a real bug shipped past the suite: replacing the inline TTL
    prune with WORLD.prune() removed a `now = time.time()` that the rest of the
    function still used, and /geom returned HTTP 500. Every map test exercised
    ObjectMap directly, so nothing touched this wrapper — and the failing line only
    runs when the map is NON-EMPTY, which no test made it.
    """
    S = _load_module()
    S.WORLD.clear()
    try:
        S.WORLD.update("red cube", [0.25, 0.03], w_m=0.05, d_m=0.05, h_m=0.05,
                       shape="cube", yaw=12.0, measured=True)
        # deliberately no stereo reading on this one: entries arrive with None or NaN
        # depending on which path made them, and both must serialize
        S.WORLD.update("cup", [0.32, -0.10], stereo=float("nan"), w_m=0.08, d_m=0.08,
                       h_m=0.10, shape="cylinder", yaw=0.0)
        snap = S.world2d_snapshot()
        assert len(snap) == 2, snap
        for o in snap:
            # the exact key set /geom and /map2d feed to the UI
            for k in ("tag", "label", "x", "y", "size", "shape", "yaw", "w_m", "d_m",
                      "h_m", "measured", "yaw_known", "aka", "r_cm", "ang", "n", "age"):
                assert k in o, f"{k} missing from the snapshot payload"
            assert isinstance(o["age"], float) and o["age"] >= 0.0
            assert o["yaw_known"] is o["measured"]
            assert o["stereo_cm"] is None, "no stereo reading -> None, not a crash"
        assert {o["label"] for o in snap} == {"red cube", "cup"}
    finally:
        S.WORLD.clear()


def test_grasp_height_uses_measurements_and_never_retires_an_object_early():
    """The derived grasp height must read the map, not mutate it.

    _picked_height answers a similar question but marks the entry picked so the place
    step can retire it. Reusing it here would retire the object BEFORE it was grasped.
    Also checks the fallback: an unmeasured object keeps today's fixed height, so
    switching this on cannot change behaviour where nothing measured anything.
    """
    S = _load_module()
    S.WORLD.clear()
    try:
        # measured, and much taller than the cube the constant was tuned on
        S.WORLD.update("bottle", [0.30, 0.0], w_m=0.07, d_m=0.07, h_m=0.23,
                       shape="cylinder", yaw=0.0, measured=True)
        tag = next(iter(S.WORLD.objs))
        z = S.grasp_z_for("bottle", np.array([0.30, 0.0]))
        assert z > S.PICK_GRASP_Z * 3, f"a 23cm bottle should not be gripped at {z*100:.1f}cm"
        assert not S.WORLD.objs[tag].get("picked"), "the object was retired before the grasp"

        # A measurement that reads LOW must not lower the grip point. The height solve
        # under-reads (measured 2.7cm against a real 5.08cm cube), and gripping low
        # drives the jaws into the table, while gripping high merely misses. So the
        # class prior is a floor — the same rule the place path already uses.
        S.WORLD.clear()
        S.WORLD.update("red cube", [0.25, 0.0], w_m=0.05, d_m=0.05, h_m=0.027,
                       shape="cube", yaw=0.0, measured=True)
        low = S.grasp_z_for("red cube", np.array([0.25, 0.0]))
        assert abs(low - S.PICK_GRASP_Z) < 1e-9, (
            f"an under-reading measurement dropped the grip to {low*100:.1f}cm")

        # unmeasured -> fall back to the tuned constant, i.e. no behaviour change
        S.WORLD.clear()
        S.WORLD.update("cup", [0.30, 0.0], w_m=0.08, d_m=0.08, h_m=0.10,
                       shape="cylinder", yaw=0.0, measured=False)
        assert S.grasp_z_for("cup", np.array([0.30, 0.0])) == S.PICK_GRASP_Z
        # nothing there at all -> also the constant
        assert S.grasp_z_for("cup", np.array([-0.30, 0.0])) == S.PICK_GRASP_Z
        assert S.grasp_z_for("cup") == S.PICK_GRASP_Z

        # and the object the constant was tuned on is unchanged
        S.WORLD.clear()
        S.WORLD.update("red cube", [0.25, 0.0], w_m=0.0508, d_m=0.0508, h_m=0.0508,
                       shape="cube", yaw=0.0, measured=True)
        got = S.grasp_z_for("red cube", np.array([0.25, 0.0]))
        assert abs(got - S.PICK_GRASP_Z) < 1e-9, f"cube grasp moved to {got*100:.2f}cm"
    finally:
        S.WORLD.clear()


def test_geometry_sync_is_live():
    """A re-fitted hand-eye or a late intrinsics read must reach the shared geometry.

    The intrinsics and the hand-eye are both discovered after import — one when the
    camera connects, the other whenever /calib re-fits it. If _sync_geometry() stopped
    being called, every projection would silently keep using the profile's fallback,
    which is the kind of failure that shows up as "the arm grabs at air" rather than
    as an exception. So: perturb each, and require the geometry to follow.
    """
    S = _load_module()
    q = np.array(JOINT_POSES[1], dtype=np.float64)
    T = np.asarray(S.T_cam_of(q))
    before = S.project_base(np.array([0.25, 0.0, 0.02]), T)

    S.fx = FX * 1.10                      # as if the camera reported a longer lens
    S._sync_geometry()
    after = S.project_base(np.array([0.25, 0.0, 0.02]), T)
    assert abs(after[0] - before[0]) > 1.0, "intrinsics change did not reach GEOM"

    S.fx = FX
    S._sync_geometry()
    assert S.project_base(np.array([0.25, 0.0, 0.02]), T) == before

    # A different hand-eye must move the camera pose itself.
    S.T_ee_cam = S.parse_tf("0.01,0.02,0.03,0,0,0")
    S._sync_geometry()
    assert not np.allclose(np.asarray(S.T_cam_of(q)), T), "hand-eye change did not reach GEOM"


def test_pose_ik_branch_runs():
    """The generic-6DOF IK branch must actually solve, not just import.

    The SO-101 exercises PitchHoldIK, so without this the PoseIK path would be dead
    code that only breaks the day someone plugs in a different arm. Declare the same
    arm as a generic wrist and require it to reach.
    """
    import dataclasses

    from rax.manipulation.arms.kinematics import make_kinematics

    from rax.manipulation.arms.ik_strategy import PoseIK, make_ik
    from rax.robots.profiles import load_profile

    p = load_profile("so101")
    kin = make_kinematics(p.urdf_path, p.ee_frame, list(p.joint_names))
    generic = dataclasses.replace(p, name="generic", ik="pose", pan_joint=None,
                                  pitch_chain=(), roll_joint=None, ik_seeds=())
    generic.validate()
    ik = make_ik(kin, generic)
    assert isinstance(ik, PoseIK)

    seed = np.array(p.home_deg, dtype=np.float64)
    for tgt in ([0.22, 0.05, 0.05], [0.28, -0.08, 0.04]):
        q, e = ik.solve(seed, np.array(tgt), pitch_deg=70.0, roll_deg=0.0)
        assert e < 0.005, f"PoseIK missed {tgt} by {e * 1000:.1f} mm"
        lo, hi = generic.limits()
        assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9), "solution outside limits"
    pitch, e = ik.plan_pitch(np.array([0.22, 0.05, 0.02]), seed)
    assert pitch is not None and e < 0.005


def test_fixed_camera_pose_is_supported():
    """The geometry must serve a world-mounted camera too, not just eye-in-hand —
    with tip_pixel correctly reporting that it has no answer for that rig."""
    from rax.perception.camera_geometry import CameraGeometry, FixedCamera, intrinsics_from_dict

    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 0.60]           # 60 cm above the base, looking down
    T[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    geom = CameraGeometry(
        intrinsics_from_dict({"fx": FX, "fy": FY, "cx": CX, "cy": CY}, 640, 480),
        FixedCamera(T))

    assert geom.tip_pixel(np.zeros(5)) is None, "a fixed camera has no fixed tip pixel"
    # Its pose does not depend on the joints.
    assert np.allclose(geom.T_base_cam(np.zeros(5)), geom.T_base_cam(np.ones(5)))
    # Round trip: a point on the table projects to a pixel that rays back to it.
    p = np.array([0.05, -0.03, 0.0])
    uv = geom.project(p, T)
    assert uv is not None
    back = geom.ray_to_plane(uv, T, 0.0)
    assert back is not None and np.allclose(back, p, atol=1e-9)


def main(argv):
    record = "--record" in argv
    S = _load_module()
    current = collect(S)
    if record:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(_jsonable(current), indent=1, sort_keys=True))
        n = sum(len(v) if isinstance(v, (list, dict)) else 1 for v in current.values())
        print(f"recorded {len(current)} groups / {n} cases -> {GOLDEN.relative_to(REPO)}")
        return 0
    if not GOLDEN.exists():
        print(f"no goldens at {GOLDEN}; run with --record first", file=sys.stderr)
        return 2
    bad = compare(current, json.loads(GOLDEN.read_text()))
    if bad:
        print(f"PARITY FAILED — {len(bad)} difference(s):", file=sys.stderr)
        for line in bad[:40]:
            print("  " + line, file=sys.stderr)
        if len(bad) > 40:
            print(f"  ... and {len(bad) - 40} more", file=sys.stderr)
        return 1
    test_urdf_limits_match_hardcoded()
    test_geometry_sync_is_live()
    test_fixed_camera_pose_is_supported()
    test_pose_ik_branch_runs()
    test_world2d_snapshot_serializes_a_populated_map()
    test_grasp_height_uses_measurements_and_never_retires_an_object_early()
    print(f"parity OK — {len(current)} groups match, URDF limits match J_LO/J_HI, "
          f"geometry sync live, fixed-camera and pose-IK branches work")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
