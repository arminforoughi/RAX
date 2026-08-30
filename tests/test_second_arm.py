"""End-to-end proof that a second arm plugs in: detect -> locate -> map -> approach.

The SO-101 is a parallel-pitch arm with a wrist camera. Every seam in the stack has a
second branch for arms that are not like it — generic 6-DOF IK, a world-fixed camera,
a profile with no URDF to read limits from — and none of those branches run during
normal operation. This exercises all of them together, with no hardware, so "another
arm can use this" is a test result rather than a claim.

The scene is synthetic and the loop is closed the same way the real one is: an object
sits at a known place on the table, the camera sees it as a bounding box, the localizer
recovers a position from that box alone, the map fuses repeated observations, and the
approach stages the arm toward the result. The check is that the arm converges on the
object it was never told the position of.

    python tests/test_second_arm.py
    pytest tests/test_second_arm.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rax.manipulation.approach import ApproachConfig, approach_target, stage_step  # noqa: E402
from rax.manipulation.arms.ik_strategy import PoseIK, make_ik  # noqa: E402
from rax.manipulation.arms.kinematics import CartesianKinematics  # noqa: E402
from rax.manipulation.arms.motion import MotionLimits, quintic_waypoints  # noqa: E402
from rax.mobility.slam.object_map import ObjectMap  # noqa: E402
from rax.perception.camera_geometry import (  # noqa: E402
    CameraGeometry,
    FixedCamera,
    intrinsics_from_dict,
    parse_tf,
)
from rax.perception.locate import ApparentSizeLocalizer  # noqa: E402
from rax.perception.object_priors import PRIORS  # noqa: E402
from rax.robots.profiles import load_profile  # noqa: E402

# Where the object really is. Nothing downstream is told this; it exists only to
# render the synthetic view and to score the result at the end.
TRUE_XY = np.array([0.26, -0.07])
TRUE_LABEL = "cup"


def _rig():
    """Build the whole stack from the mock profile, the way a server would."""
    profile = load_profile("mock")
    kin = CartesianKinematics()
    ik = make_ik(kin, profile)
    fb = profile.camera.intrinsics_fallback
    geom = CameraGeometry(
        intrinsics_from_dict(dict(zip(("fx", "fy", "cx", "cy"), fb)),
                             profile.camera.width, profile.camera.height),
        FixedCamera(parse_tf(profile.camera.extrinsics)))
    return profile, kin, ik, geom


def _render_bbox(geom, xy, *, label=TRUE_LABEL, z_m=0.0):
    """The bounding box the camera would report for an object at ``xy``.

    Projects the object's real footprint through the same intrinsics the localizer
    uses, so the test closes the loop through the geometry rather than hand-computing
    a box that happens to give the right answer.
    """
    meta = PRIORS.meta(label)
    half = 0.5 * float(np.sqrt(meta["w_m"] * meta["d_m"]))
    T = geom.T_base_cam()
    corners = []
    for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half)):
        uv = geom.project(np.array([xy[0] + dx, xy[1] + dy, z_m]), T)
        assert uv is not None, "the object projected behind the camera"
        corners.append(uv)
    us = [c[0] for c in corners]
    vs = [c[1] for c in corners]
    return (min(us), min(vs), max(us), max(vs))


def test_mock_profile_takes_the_other_branch_everywhere():
    profile, _kin, ik, geom = _rig()
    assert profile.ik == "pose" and isinstance(ik, PoseIK)
    assert profile.camera.mount == "fixed"
    assert not geom.pose.moves_with_arm
    assert geom.tip_pixel(np.zeros(profile.n_joints)) is None
    assert profile.bus is None, "no servo bus to clear overloads on"
    # No URDF, so the limits had to be declared rather than read — and resolve() must
    # accept that rather than insisting on a file.
    lo, hi = profile.limits()
    assert len(lo) == len(hi) == profile.n_joints


def test_localizer_recovers_the_object_from_its_bbox_alone():
    """The whole chain from pixels to metres, with a world-fixed camera.

    Uses the axial-depth placement, which is the geometrically correct one. See
    test_apparent_size_has_an_inward_bias_off_axis for why it is not yet the default.
    """
    profile, _kin, _ik, geom = _rig()
    loc = ApparentSizeLocalizer(geom, PRIORS,
                                reach_m=(profile.reach_min_m, profile.reach_max_m))
    loc.axial_depth = True
    fix = loc.locate(_render_bbox(geom, TRUE_XY), geom.T_base_cam(), label=TRUE_LABEL)
    assert fix.ok, f"localization failed: {fix}"
    err = float(np.linalg.norm(fix.xy - TRUE_XY))
    assert err < 0.005, f"located {fix.xy} vs true {TRUE_XY} — {err * 1000:.0f} mm out"


def test_apparent_size_has_an_inward_bias_off_axis():
    """Pin a real, pre-existing defect so it cannot be forgotten or silently changed.

    Apparent size yields AXIAL DEPTH (fx*W/w_px), but the default placement walks that
    distance ALONG THE SIGHTLINE. The two agree only on the optical axis; elsewhere the
    object lands too close by cos(off-axis angle) — always inward, growing with angle.

    This is inherited from the monolith's obj_xy_2d, which the extraction reproduces
    exactly, and it is live on the real rig because the approach deliberately keeps the
    object off-centre. Flipping ApparentSizeLocalizer.axial_depth fixes it, but that
    changes where the arm drives, so it needs a hardware check first.
    """
    profile, _kin, _ik, geom = _rig()
    reach = (profile.reach_min_m, profile.reach_max_m)
    biased = ApparentSizeLocalizer(geom, PRIORS, reach_m=reach)
    correct = ApparentSizeLocalizer(geom, PRIORS, reach_m=reach)
    correct.axial_depth = True

    T = geom.T_base_cam()
    # The on-axis point sits under the arm, inside the workspace floor, so compare
    # there with the reach gate off — the question is the geometry, not reachability.
    biased_raw = ApparentSizeLocalizer(geom, PRIORS, reach_m=reach, gate_reach=False)
    correct_raw = ApparentSizeLocalizer(geom, PRIORS, reach_m=reach, gate_reach=False)
    correct_raw.axial_depth = True
    on_axis = np.array([0.02, 0.0])
    a = biased_raw.locate(_render_bbox(geom, on_axis), T, label=TRUE_LABEL)
    b = correct_raw.locate(_render_bbox(geom, on_axis), T, label=TRUE_LABEL)
    assert a.ok and b.ok
    assert np.linalg.norm(a.xy - b.xy) < 0.001, "the two must agree on the optical axis"

    for xy, floor_mm in (([0.20, 0.0], 5.0), ([0.35, 0.0], 30.0)):
        xy = np.array(xy)
        got = biased.locate(_render_bbox(geom, xy), T, label=TRUE_LABEL)
        assert got.ok
        # Biased inward: closer to the base than the truth, never further out.
        assert np.hypot(*got.xy) < np.hypot(*xy)
        err_mm = float(np.linalg.norm(got.xy - xy)) * 1000
        assert err_mm > floor_mm, f"expected a >{floor_mm:.0f} mm bias at {xy}, got {err_mm:.1f}"


def test_map_fuses_repeated_views_into_one_object():
    profile, _kin, _ik, geom = _rig()
    loc = ApparentSizeLocalizer(geom, PRIORS,
                                reach_m=(profile.reach_min_m, profile.reach_max_m))
    loc.axial_depth = True
    world = ObjectMap(log=lambda _m: None)
    rng = np.random.default_rng(5)
    for _ in range(25):
        # jitter the box the way a real detector's box jitters
        bbox = np.array(_render_bbox(geom, TRUE_XY)) + rng.normal(0, 1.5, 4)
        fix = loc.locate(tuple(bbox), geom.T_base_cam(), label=TRUE_LABEL)
        if not fix.ok:
            continue
        meta = PRIORS.meta(TRUE_LABEL)
        world.update(TRUE_LABEL, fix.xy, w_m=meta["w_m"], d_m=meta["d_m"],
                     h_m=meta["h_m"], shape=meta["shape"], yaw=0.0)
    assert len(world) == 1, f"25 views of one cup became {len(world)} objects"
    entry = next(iter(world.objs.values()))
    err = float(np.linalg.norm(entry["xy"] - TRUE_XY))
    assert err < 0.02, f"mapped at {entry['xy']} vs true {TRUE_XY}"


def test_staged_approach_converges_on_the_mapped_object():
    """The approach must close the distance without overshooting, using pose IK."""
    profile, kin, ik, geom = _rig()
    cfg = ApproachConfig()
    loc = ApparentSizeLocalizer(geom, PRIORS,
                                reach_m=(profile.reach_min_m, profile.reach_max_m))
    loc.axial_depth = True

    fix = loc.locate(_render_bbox(geom, TRUE_XY), geom.T_base_cam(), label=TRUE_LABEL)
    assert fix.ok
    target = approach_target(fix.xy, back_m=cfg.back_m, right_trim_m=cfg.right_trim_m)

    q = np.array(profile.home_deg, dtype=np.float64)
    start = float(np.linalg.norm(np.asarray(kin.forward_kinematics(q))[:2, 3] - target))
    for stage in range(cfg.steps):
        tip = np.asarray(kin.forward_kinematics(q))[:3, 3]
        waypoint, remaining = stage_step(tip[:2], target, stage=stage, total=cfg.steps,
                                         first_frac=cfg.first_step_frac,
                                         max_first_m=cfg.max_first_step_m)
        if waypoint is None or remaining < cfg.arrived_m:
            break
        goal = np.array([waypoint[0], waypoint[1], cfg.hover_z_m])
        q, e = ik.solve(q, goal)
        assert e < 0.005, f"stage {stage}: IK missed by {e * 1000:.1f} mm"

    tip = np.asarray(kin.forward_kinematics(q))[:3, 3]
    final = float(np.linalg.norm(tip[:2] - target))
    assert final < start, "the approach moved away from the target"
    assert final < cfg.arrived_m, f"ended {final * 1000:.0f} mm from the hover target"
    # It should stop SHORT of the object and to one side, not on top of it — that
    # offset is what keeps the object in view during the real approach.
    assert np.linalg.norm(tip[:2] - TRUE_XY) > cfg.back_m


def test_transit_moves_respect_the_profiles_limits():
    """A profile that never declared per-joint ceilings still gets a valid profile."""
    profile, _kin, _ik, _geom = _rig()
    limits = MotionLimits.from_profile(profile)
    a = np.array(profile.home_deg, dtype=np.float64)
    b = np.array(profile.view_deg, dtype=np.float64)
    waypoints, T = quintic_waypoints(a, b, limits)
    assert T >= 0.12 and len(waypoints) >= 2
    assert np.allclose(waypoints[0], a) and np.allclose(waypoints[-1], b, atol=1e-9)
    # Velocity never exceeds the ceiling the profile asked for.
    v = np.abs(np.diff(np.array(waypoints), axis=0)) / limits.dt_s
    assert np.all(v <= limits.vmax_dps * 1.02 + 1e-9), f"peak {v.max():.1f} deg/s"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
