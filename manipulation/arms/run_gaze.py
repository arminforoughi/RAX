"""CLI entrypoint: detect + tag + stream point clouds, then gaze-grasp/place.

Examples
--------
Dev harness (no hardware, no model weights)::

    python -m manipulation.arms.run_gaze --backend mock --query "red box" --place-on "blue box"

Real SO-101 + OAK-D (needs the arm, the camera, and RAFT/FoundationStereo weights)::

    python -m manipulation.arms.run_gaze --backend so101 --interface eth0 \
        --query "red cube" --stereo raft

``--backend mock`` falls back to SGBM stereo + colour-blob detection + ellipse
masks automatically, so it runs anywhere; pass ``--stereo``/``--detector``/``--mask``
to force a specific backend.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from models.depth import make_stereo
from models.detection import make_detector, make_mask_tracker
from perception.depth_cloud import CloudTracker, PointCloudStream
from perception.depth_cloud.stream import CloudUpdate


def _build_arm(args):
    """Return (arm, kinematics, T_ee_cam, world_up)."""
    if args.backend == "mock":
        from manipulation.arms.kinematics import CartesianKinematics
        from manipulation.arms.mock_arm import WORLD_UP, MockArm

        arm = MockArm()
        return arm, CartesianKinematics(), np.eye(4), WORLD_UP

    if args.backend == "so101":
        from robots.arms.lerobot_so101.driver import So101Arm

        arm = So101Arm(
            port=args.port, urdf=args.urdf, ee_frame=args.ee_frame,
            gripper_camera_tf=args.gripper_camera_tf,
        )
        return arm, arm.kin, arm.T_ee_cam, np.array([0.0, 0.0, 1.0])

    raise ValueError(f"unknown backend {args.backend!r}")


def _save_snapshot(path: str, left, cloud) -> None:
    """Draw current tracks (box, tag, centroid depth) on the left frame and save it."""
    import cv2

    img = np.ascontiguousarray(left[:, :, ::-1])  # RGB -> BGR for cv2
    for tr in cloud.list_tracks():
        x1, y1, x2, y2 = (int(v) for v in tr.box)
        focus = tr.tag == cloud.focus_tag
        color = (0, 0, 255) if focus else (0, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if focus else 1)
        depth = tr.centroid[2] if tr.has_cloud else float("nan")
        label = f"{'*' if focus else ''}#{tr.tag} {tr.label} z={depth:.2f}m"
        cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(path, img)


def _print_stream(updates: list[CloudUpdate]) -> None:
    parts = [
        f"{'*' if u.is_focus else ' '}tag{u.tag}:{u.label}[{u.points.shape[0]}pts "
        f"c=({u.centroid[0]:+.2f},{u.centroid[1]:+.2f},{u.centroid[2]:+.2f})]"
        for u in updates
    ]
    print("  cloud:", " ".join(parts))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["mock", "so101"], default="mock")
    p.add_argument("--query", default="red box", help="object to focus and grasp")
    p.add_argument("--place-on", default=None, help="label of the object to place onto (optional)")
    p.add_argument("--stereo", default="auto", choices=["auto", "raft", "foundation", "sgbm"])
    p.add_argument("--max-disp", type=int, default=192,
                   help="max stereo disparity (px). Close objects need a large range: "
                        "depth = fx*baseline/disp, so ~18cm on the real OAK-D is ~340px "
                        "(run_gaze_engine.sh passes 384). Keep small (~192) for the mock scene.")
    p.add_argument("--detector", default="auto", choices=["auto", "yolo", "blob", "color_blob"])
    p.add_argument("--mask", default="auto", choices=["auto", "sam2", "ellipse"])
    p.add_argument("--detect-every", type=int, default=5)
    p.add_argument("--max-ticks", type=int, default=600)
    p.add_argument("--grasp-range", type=float, default=0.10,
                   help="stop approaching and close when depth reads this (m) ~ fingertip range")
    p.add_argument("--final-advance", type=float, default=0.02,
                   help="blind final inch along view axis after centred at grasp-range (m)")
    p.add_argument("--aim-v-offset", type=float, default=90.0,
                   help="pixels below image centre where the object should sit (gripper aim point)")
    p.add_argument("--approach-style", default="angled",
                   choices=["angled", "topdown", "horizontal"],
                   help="how the gripper comes in on the object (so101 Cartesian servo): "
                        "'angled' along the current view (least motion), 'topdown' from "
                        "straight above, 'horizontal' level from the base side.")
    p.add_argument("--pan-sign", type=float, default=None,
                   help="shoulder_pan sign (+1 default mock, -1 default so101). "
                        "Flip if the arm pans AWAY from the object.")
    p.add_argument("--tilt-sign", type=float, default=None,
                   help="shoulder_lift sign (-1 default so101). Flip if vertical gaze is wrong.")
    # so101 hardware knobs
    p.add_argument("--port", default="/dev/ttyACM0",
                   help="SO-101 serial port (Linux: /dev/ttyACM0, macOS: /dev/tty.usbmodem*)")
    p.add_argument("--urdf", default="SO101/so101_new_calib.urdf")
    p.add_argument("--ee-frame", default="gripper_frame_link")
    p.add_argument("--gripper-camera-tf", default="0.04,0,0.07,0,-0.5,0",
                   help="eye-in-hand extrinsic 'x,y,z,rx,ry,rz' (m, rotvec rad): camera pose in "
                        "the gripper frame. Camera sits ~7cm above the gripper, pitched down to "
                        "sight the fingertips. Measure/calibrate this for your mount.")
    p.add_argument("--rerun", action="store_true",
                   help="open the Rerun viewer: live camera + detection boxes + 3D point clouds")
    p.add_argument("--no-move", action="store_true",
                   help="perception only: connect, detect, tag, stream clouds — never command motion")
    p.add_argument("--snapshot", default=None, metavar="PATH",
                   help="with --no-move, save an annotated frame (boxes + tags + depth) to PATH")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    arm, kin, T_ee_cam, world_up = _build_arm(args)
    # The synthetic mock scene is plain colour balls; default it to the colour-blob
    # detector (YOLO-World won't recognise them). Real backends default to auto.
    det_backend = args.detector
    if det_backend == "auto" and args.backend == "mock":
        det_backend = "color_blob"
    elif det_backend == "auto" and args.backend == "so101":
        # Colour word in the query -> fast HSV detector. Plain noun -> the mono
        # foreground-blob detector: on the grayscale OAK-D stream it reliably finds a
        # clear object on the table, where YOLO-World (trained on RGB) often returns
        # nothing for a plain cube. Force YOLO explicitly with --detector yolo.
        from models.detection.prompt_detector import _HSV_RANGES

        if any(w in _HSV_RANGES for w in args.query.lower().split()):
            det_backend = "color_blob"
        else:
            det_backend = "blob"
    detector = make_detector(det_backend)
    mask_tracker = make_mask_tracker(args.mask)
    stereo = make_stereo(args.stereo, max_disp_px=args.max_disp)

    stream = PointCloudStream()
    stream.subscribe(_print_stream)
    extra = (args.place_on,) if args.place_on else ()
    cloud = CloudTracker(detector, mask_tracker, stereo, args.query,
                         extra_labels=extra, detect_every=args.detect_every, stream=stream)

    viz = None
    if args.rerun:
        from perception.depth_cloud.rerun_viz import RerunViz
        viz = RerunViz.try_create()
        if viz:
            print("[run_gaze] Rerun viewer open — camera + boxes + 3D point clouds")

    if args.no_move:
        # Connect to the real sensors and run perception only — the arm never moves.
        print("[run_gaze] --no-move: perception only, NO motion commanded")
        import time
        last = None
        for _ in range(args.max_ticks):
            last = arm.get_observation()
            cloud.update(last)
            if viz:
                viz.log(last, cloud, "no-move", kin=kin)
            time.sleep(0.05)
        print(f"[run_gaze] perception check done; {len(cloud.list_tracks())} tracks, "
              f"focus tag {cloud.focus_tag}")
        for tr in cloud.list_tracks():
            print(f"    tag{tr.tag} {tr.label}: centroid(base) "
                  f"({tr.centroid[0]:+.3f},{tr.centroid[1]:+.3f},{tr.centroid[2]:+.3f}) m, "
                  f"{tr.points.shape[0]} pts")
        if args.snapshot and last is not None:
            _save_snapshot(args.snapshot, last.left, cloud)
            print(f"[run_gaze] wrote annotated frame to {args.snapshot}")
        return

    # Lazy import to avoid a hard dependency when only the perception stack is used.
    from manipulation.arms.gaze_engine import GazeConfig, GazeEngine

    pan_sign = args.pan_sign if args.pan_sign is not None else 1.0
    tilt_sign = args.tilt_sign if args.tilt_sign is not None else (1.0 if args.backend == "so101" else 1.0)
    cfg = GazeConfig(
        T_ee_cam=T_ee_cam, world_up=world_up,
        grasp_range_m=args.grasp_range, final_advance_m=args.final_advance,
        aim_v_offset_px=args.aim_v_offset, pan_sign=pan_sign, tilt_sign=tilt_sign,
        approach_style=args.approach_style,
    )
    engine = GazeEngine(
        arm, kin, cloud, cfg, place_on_label=args.place_on,
        cartesian=(args.backend == "mock"),
    )

    print(f"[run_gaze] backend={args.backend} query={args.query!r} place_on={args.place_on!r}")
    print(f"[run_gaze] detector={detector.name} mask={mask_tracker.name} stereo={stereo.name}")
    print(f"[run_gaze] grasp_range={args.grasp_range}m aim_v_offset={args.aim_v_offset}px "
          f"advance={args.final_advance}m approach={args.approach_style} "
          f"pan_sign={pan_sign} tilt_sign={tilt_sign}")
    if viz:
        # Drive the loop ourselves so we can log every tick to Rerun.
        import time
        from manipulation.arms.gaze_engine import DONE, FAILED
        dt = 1.0 / max(1.0, cfg.loop_hz)
        final = engine.state
        for _ in range(args.max_ticks):
            final = engine.step(dt)
            if engine.last_obs is not None:
                focus = cloud.focus_track()
                depth_m = engine.range_m
                u_aim = v_aim = None
                if focus is not None:
                    u_aim, v_aim = engine._gaze_uv(focus, engine.last_obs)
                viz.log(
                    engine.last_obs, cloud, final,
                    kin=kin, depth_m=depth_m, u_aim=u_aim, v_aim=v_aim,
                    aim_v_offset_px=engine._aim_v_now,  # live ramping aim, centre -> bottom
                )
            if final in (DONE, FAILED):
                break
            time.sleep(dt)
    else:
        final = engine.run(max_ticks=args.max_ticks)
    print(f"[run_gaze] finished in state {final} after focusing tag {cloud.focus_tag}")


if __name__ == "__main__":
    main()
