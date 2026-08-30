"""``python -m rax.grasp`` — point an arm and a camera at an object and pick it up.

This is the entry point the repo is published for. Everything else in the tree is
either a library it calls or a demo built on top of it.

    # No hardware at all — synthetic arm, synthetic camera, full pipeline.
    python -m rax.grasp --arm mock --query "red object"

    # A webcam on a mast watching any arm you have a profile for.
    python -m rax.grasp --arm head_mono --camera-source 0 --query "cup"

    # SO-101 with the OAK-D on its wrist.
    python -m rax.grasp --arm so101 --port /dev/ttyACM0 --query "red cube"

Which pipeline runs is decided by the rig, not by a flag, because the two are genuinely
different problems:

**Eye-in-hand** (camera on the gripper). The object's bearing *is* the thing being
controlled — every joint move changes the view — so the arm visually servos: keep the
object on the aim pixel, drive the range down, close. That is
:class:`~rax.manipulation.arms.gaze_engine.GazeEngine`.

**Fixed camera** (head mast, tripod, torso). The view never changes, so there is
nothing to servo. Instead the object is localized *once* into base frame, fused into a
map over several frames, and the arm is staged toward the result open-loop. This is the
path a head-camera rig takes, and it is the one that works with no depth sensor at all.

Both end in the same current-sensed close from
:mod:`rax.manipulation.arms.grasp`, and both drive the same profile-described arm.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from rax.manipulation.approach import ApproachConfig, approach_target, grasp_height, stage_step
from rax.manipulation.arms.grasp import GraspConfig, close_with_current, release
from rax.manipulation.arms.ik_strategy import make_ik
from rax.manipulation.arms.motion import MotionLimits, quintic_waypoints
from rax.manipulation.arms.rig import Rig, geometry_for
from rax.mobility.slam.object_map import ObjectMap
from rax.perception.locate import ApparentSizeLocalizer, PlaneRayLocalizer, chain
from rax.perception.object_priors import PRIORS
from rax.robots.profiles import available_profiles, load_profile

logger = logging.getLogger("rax.grasp")


# --- assembling the rig -------------------------------------------------------------
def build_arm(profile, args):
    """The arm half of the rig, chosen by profile unless overridden.

    A driver is the one piece a new robot genuinely has to supply. The contract is four
    methods (see :mod:`rax.manipulation.arms.arm_interface`); everything else about the
    arm is described by its profile as data.
    """
    driver = args.driver or ("so101" if profile.name == "so101" else "mock")

    if driver == "so101":
        from rax.robots.arms.lerobot_so101.driver import So101Arm

        port = args.port or profile.port
        if not port:
            raise SystemExit(
                "the SO-101 driver needs a serial port: pass --port (Linux "
                "/dev/ttyACM0, Windows COM4) or set it in the profile")
        arm = So101Arm(port=port, urdf=profile.urdf_path, ee_frame=profile.ee_frame,
                       gripper_camera_tf=profile.camera.extrinsics)
        return arm, arm.kin

    if driver == "mock":
        from rax.manipulation.arms.kinematics import CartesianKinematics
        from rax.manipulation.arms.mock_arm import MockArm, MockArmBody, table_scene

        kin = CartesianKinematics()
        q0 = np.array(profile.home_deg, dtype=np.float64) if profile.home_deg else None
        # An eye-in-hand mock owns its camera (it renders from its own EE pose); a
        # fixed-camera mock must not, or the rig would have two disagreeing views.
        if profile.camera.eye_in_hand:
            return MockArm(), kin
        # A fixed-camera rig has an ordinary Z-up base frame, so the scene has to be
        # placed on the table rather than in the mock arm's camera-convention frame.
        return MockArmBody(objects=table_scene(profile.table_z_m), q0=q0), kin

    raise SystemExit(f"unknown --driver {driver!r}; expected so101|mock")


def build_camera(profile, arm, args):
    """The camera half. ``--camera`` overrides what the profile declares."""
    kind = args.camera or profile.camera.kind

    if args.driver == "mock" or (args.driver is None and profile.name != "so101"):
        from rax.perception.cameras.synthetic import SyntheticCamera

        # The synthetic camera needs a pose to render from. For a fixed mount that is
        # the profile's extrinsics; the mock arm's own renderer covers eye-in-hand.
        geom = geometry_for(profile)
        objects = getattr(arm, "objects", None)
        return SyntheticCamera(lambda: geom.T_base_cam(), kind=kind, objects=objects,
                               width=profile.camera.width,
                               height=profile.camera.height,
                               fx=profile.camera.intrinsics_fallback[0])

    from rax.perception.cameras import make_camera

    return make_camera(profile, source=args.camera_source)


def build_rig(args):
    """Profile + arm + camera + geometry + depth strategy, ready to run."""
    profile = load_profile(args.arm)
    arm, kin = build_arm(profile, args)

    if profile.camera.eye_in_hand and hasattr(arm, "get_observation") and args.driver != "rig":
        # This arm already owns its wrist camera; wrapping it in a Rig would only
        # discard and re-render the same frames.
        return profile, arm, kin, geometry_for(profile, kin)

    camera = build_camera(profile, arm, args)
    geom = geometry_for(profile, kin)
    stereo = None
    if getattr(camera, "kind", "mono") == "stereo":
        from rax.models.depth import make_stereo

        stereo = make_stereo(args.stereo, max_disp_px=args.max_disp)
    return profile, Rig(arm, camera, geom, stereo=stereo), kin, geom


def build_detector(args, profile):
    """A detector that can actually see this rig's imagery.

    YOLO-World is the right default on a colour camera, but it returns nothing on the
    synthetic scene (plain untextured balls) and often nothing on the OAK-D's greyscale
    rectified stream. Choosing by rig rather than making the user discover this is the
    difference between a demo that runs and a demo that prints "no detections".
    """
    from rax.models.detection import make_detector

    backend = args.detector
    if backend == "auto":
        if args.driver == "mock" or profile.name in ("mock", "head_mono"):
            backend = "color_blob"
        else:
            from rax.models.detection.prompt_detector import _HSV_RANGES

            words = args.query.lower().split()
            backend = "color_blob" if any(w in _HSV_RANGES for w in words) else "blob"
    return make_detector(backend)


# --- the fixed-camera pipeline ------------------------------------------------------
def localizers(geom, profile, has_depth: bool):
    """The range strategies available to this rig, best first.

    With a fixed camera the object is on a known surface, so the plane intersection is
    tried first: it solves range and size together and is the strongest estimator for
    anything resting on the table. Apparent size is the fallback for objects the plane
    solve rejects. A depth sensor would add a third, but neither of these needs one,
    which is exactly why a webcam rig works.
    """
    reach = (profile.reach_min_m, profile.reach_max_m)
    return [
        PlaneRayLocalizer(geom, PRIORS, z_plane=profile.table_z_m, reach_m=reach),
        ApparentSizeLocalizer(geom, PRIORS, reach_m=reach),
    ]


def detect_bbox(detector, rgb, query: str):
    """Highest-confidence box matching the query, or None."""
    dets = detector.detect(rgb, query)
    if not dets:
        return None
    best = max(dets, key=lambda d: d.confidence)
    return tuple(float(v) for v in best.box)


def survey(rig, detector, locs, world, query: str, *, frames: int, settle_s: float):
    """Look at the scene ``frames`` times and fuse what is seen into the map.

    Repeated views of a *static* scene from a *static* camera are not redundant: the
    detector's box jitters by a pixel or two per frame, and each pixel is millimetres at
    the object, so the map's fusion averages away noise the single-shot fix carries.
    """
    meta = PRIORS.meta(query)
    seen = 0
    for _ in range(frames):
        obs = rig.get_observation()
        bbox = detect_bbox(detector, obs.left, query)
        if bbox is None:
            time.sleep(settle_s)
            continue
        fix = chain(locs, bbox, obs.T_base_cam, label=query)
        if not fix.ok:
            time.sleep(settle_s)
            continue
        world.update(query, fix.xy, w_m=meta["w_m"], d_m=meta["d_m"], h_m=meta["h_m"],
                     shape=meta["shape"], yaw=0.0)
        seen += 1
        time.sleep(settle_s)
    return seen


def goto(rig, kin, ik, profile, q, goal_xyz):
    """Solve for a Cartesian goal and drive there on a smooth trajectory."""
    q_next, err = ik.solve(np.asarray(q, dtype=np.float64), np.asarray(goal_xyz))
    if err > 0.02:
        return q, err
    limits = MotionLimits.from_profile(profile)
    waypoints, _T = quintic_waypoints(np.asarray(q, dtype=np.float64), q_next, limits)
    for w in waypoints:
        rig.send_joint_targets(w)
        time.sleep(limits.dt_s)
    return q_next, err


def pick_with_fixed_camera(rig, kin, geom, profile, args) -> int:
    """Locate once, fuse, stage in, descend, close, lift. No visual servo.

    The arm is deliberately NOT corrected from the camera during the descent. A head
    camera watching an arm reach toward an object sees the arm occlude the object at
    exactly the moment a correction would matter, so a servo loop there chases its own
    gripper. Fusing several clean views *before* moving, then executing open-loop, is
    both simpler and more reliable for this rig.
    """
    detector = build_detector(args, profile)
    locs = localizers(geom, profile, getattr(getattr(rig, "depth", None), "available", False))
    world = ObjectMap(log=lambda m: logger.debug("[map] %s", m))
    cfg = ApproachConfig()

    print(f"[rax] rig: {profile.name} + {getattr(rig.camera, 'kind', '?')} camera "
          f"({profile.camera.mount}), detector={detector.name}, depth={rig.depth.name}")
    print(f"[rax] surveying for {args.query!r}...")

    seen = survey(rig, detector, locs, world, args.query,
                  frames=args.survey_frames, settle_s=args.settle)
    if not len(world):
        print(f"[rax] never located {args.query!r} in {args.survey_frames} frames "
              f"({seen} usable detections). Is it in view and on the table?")
        return 1

    entry = next(iter(world.objs.values()))
    xy = np.asarray(entry["xy"], dtype=np.float64)
    h_m = float(entry.get("h_m") or PRIORS.meta(args.query)["h_m"])
    print(f"[rax] {args.query!r} at ({xy[0]:+.3f}, {xy[1]:+.3f}) m from {seen} views")

    if args.no_move:
        print("[rax] --no-move: located only, nothing commanded")
        return 0

    # Grasp height derived from the object's own measured height, not a constant that
    # was tuned on somebody else's cube.
    z_grasp = profile.table_z_m + grasp_height(h_m)
    target = approach_target(xy, back_m=cfg.back_m, right_trim_m=cfg.right_trim_m)
    q = np.array(profile.home_deg, dtype=np.float64)
    ik = make_ik(kin, profile)
    rig.set_gripper(profile.gripper.open_pct)

    for stage in range(cfg.steps):
        tip = np.asarray(kin.forward_kinematics(q))[:3, 3]
        waypoint, remaining = stage_step(tip[:2], target, stage=stage, total=cfg.steps,
                                         first_frac=cfg.first_step_frac,
                                         max_first_m=cfg.max_first_step_m)
        if waypoint is None or remaining < cfg.arrived_m:
            break
        q, err = goto(rig, kin, ik, profile, q,
                      [waypoint[0], waypoint[1], cfg.hover_z_m])
        print(f"[rax]   stage {stage + 1}/{cfg.steps}: {remaining * 100:.1f} cm to go "
              f"(ik {err * 1000:.1f} mm)")

    print(f"[rax] descending to z={z_grasp * 100:.1f} cm and closing")
    q, _ = goto(rig, kin, ik, profile, q, [xy[0], xy[1], z_grasp])

    grip = GraspConfig(open_pct=profile.gripper.open_pct,
                       close_pct=profile.gripper.closed_pct,
                       close_step_pct=profile.gripper.close_step_pct,
                       contact_delta_counts=profile.gripper.contact_current_delta)
    result = close_with_current(rig, grip)
    print(f"[rax] grasp: contact={result.contact} ({result.reason})")

    q, _ = goto(rig, kin, ik, profile, q, [xy[0], xy[1], cfg.hover_z_m])
    if not result.contact:
        release(rig, grip)
        print("[rax] closed on air — released and stopped")
        return 1
    print("[rax] holding. Done.")
    return 0


# --- the eye-in-hand pipeline -------------------------------------------------------
def pick_with_gaze(rig, kin, profile, args) -> int:
    """Visual servo: keep the object on the aim pixel and drive the range down."""
    from rax.manipulation.arms.gaze_engine import GazeConfig, GazeEngine
    from rax.models.depth import make_stereo
    from rax.models.detection import make_mask_tracker
    from rax.perception.depth_cloud import CloudTracker, PointCloudStream

    detector = build_detector(args, profile)
    stereo = make_stereo(args.stereo, max_disp_px=args.max_disp)
    cloud = CloudTracker(detector, make_mask_tracker(args.mask), stereo, args.query,
                         detect_every=args.detect_every, stream=PointCloudStream())
    cfg = GazeConfig(T_ee_cam=geometry_for(profile, kin).pose.T_ee_cam,
                     world_up=np.array([0.0, 0.0, 1.0]))
    engine = GazeEngine(rig, kin, cloud, cfg, cartesian=(profile.ik == "pose"))

    print(f"[rax] rig: {profile.name} eye-in-hand, detector={detector.name}, "
          f"stereo={stereo.name}")
    print(f"[rax] gaze-servoing onto {args.query!r}")
    final = engine.run(max_ticks=args.max_ticks)
    print(f"[rax] finished in state {final}")
    return 0 if str(final).lower().startswith("done") else 1


# --- CLI ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rax.grasp", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="mock",
                   help="profile name; --list shows what is installed")
    p.add_argument("--list", action="store_true", help="list arm profiles and exit")
    p.add_argument("--query", default="red object", help="what to pick up")

    g = p.add_argument_group("hardware")
    g.add_argument("--driver", default=None, choices=["so101", "mock"],
                   help="arm driver; defaults to the one the profile implies")
    g.add_argument("--port", default=None, help="arm serial port (overrides the profile)")
    g.add_argument("--camera", default=None, choices=["stereo", "rgbd", "mono"],
                   help="sensor kind; defaults to what the profile declares")
    g.add_argument("--camera-source", default=0,
                   help="mono camera index or URL (default 0)")

    g = p.add_argument_group("perception")
    g.add_argument("--detector", default="auto",
                   choices=["auto", "yolo", "blob", "color_blob"])
    g.add_argument("--mask", default="auto", choices=["auto", "sam2", "ellipse"])
    g.add_argument("--stereo", default="auto",
                   choices=["auto", "raft", "foundation", "sgbm"])
    g.add_argument("--max-disp", type=int, default=192)
    g.add_argument("--detect-every", type=int, default=5)
    g.add_argument("--survey-frames", type=int, default=12,
                   help="fixed-camera rigs: frames fused before moving")
    g.add_argument("--settle", type=float, default=0.05,
                   help="seconds between survey frames")

    g = p.add_argument_group("run")
    g.add_argument("--max-ticks", type=int, default=600)
    g.add_argument("--no-move", action="store_true",
                   help="locate and report, never command motion")
    g.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.list:
        for name in available_profiles():
            prof = load_profile(name)
            print(f"  {name:<12} {prof.n_joints} joints, ik={prof.ik}, "
                  f"camera={prof.camera.kind}/{prof.camera.mount}")
        return 0

    profile, rig, kin, geom = build_rig(args)
    try:
        if profile.camera.eye_in_hand:
            return pick_with_gaze(rig, kin, profile, args)
        return pick_with_fixed_camera(rig, kin, geom, profile, args)
    except KeyboardInterrupt:
        print("\n[rax] interrupted")
        return 130
    finally:
        for name in ("disconnect", "close"):
            fn = getattr(rig, name, None)
            if fn is not None:
                fn()
                break


if __name__ == "__main__":
    sys.exit(main())
