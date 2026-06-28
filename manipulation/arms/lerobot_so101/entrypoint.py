"""RAX-native ``lerobot-gaze-engine`` — search → approach → grasp on SO-101 + OAK-D.

Mirrors the flag surface of the original lerobot console script so
``run_gaze_engine.sh`` works with only a shebang/path change.

Usage (matches lerobot/run_gaze_engine.sh)::

    python -m manipulation.arms.lerobot_so101 \\
      --robot.port /dev/ttyACM0 \\
      --urdf ./SO101/so101_new_calib.urdf \\
      --query "red cube" \\
      --model-path ./yolov8s-worldv2.pt \\
      --gripper-camera-tf "0.04,0,0.09,-0.2690,0.2824,-1.6014"

Key differences from the raw ``manipulation.arms.run_gaze`` CLI:
  - Uses the ``--robot.port`` / ``--robot.type`` naming convention
  - Accepts ``--gaze-*`` / ``--approach-*`` / ``--live-*`` flag aliases
  - ``--display-data`` streams a live OpenCV window (left + detection overlay)
  - ``--display-sim3d`` opens a Rerun viewer (same as ``--rerun`` in run_gaze)
  - ``--live-control-keypress`` enables WASD-style keyboard nudge
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing (mirrors lerobot-gaze-engine flag names)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RAX gaze engine — SO-101 + OAK-D (lerobot-gaze-engine compatible)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- robot connection -----------------------------------------------
    p.add_argument("--robot.port", dest="robot_port", default="",
                   help="Serial port for the SO-101 arm (e.g. /dev/ttyACM0).")
    p.add_argument("--robot.type", dest="robot_type", default="so101_follower",
                   help="Robot type (only so101_follower is supported here).")
    p.add_argument("--robot.cameras", dest="robot_cameras", default="",
                   help="Ignored — OAK-D is auto-detected from the arm.")
    p.add_argument("--urdf", default="SO101/so101_new_calib.urdf",
                   help="Path to the SO-101 URDF (absolute or relative to CWD).")
    p.add_argument("--ee-frame", default="gripper_frame_link",
                   help="End-effector frame name in the URDF.")
    p.add_argument("--gripper-camera-tf",
                   default="0.04,0,0.09,-0.2690,0.2824,-1.6014",
                   help="Eye-in-hand extrinsic as 'x,y,z,rx,ry,rz' (m, rotvec rad).")
    p.add_argument("--camera-key", default="front",
                   help="Camera key used in lerobot robot config.")

    # ---- detection ------------------------------------------------------
    p.add_argument("--query", default="cube",
                   help="Natural-language query for the object to grasp.")
    p.add_argument("--place-on", default=None,
                   help="Label of the support object for pick-and-place.")
    p.add_argument("--model-path", default="yolov8s-worldv2.pt",
                   help="YOLO-World model weights (.pt).")
    p.add_argument("--search-min-detection-confidence",
                   dest="min_conf", type=float, default=0.18,
                   help="Minimum YOLO detection confidence.")
    p.add_argument("--detector", default="auto",
                   choices=["auto", "yolo", "color_blob", "blob"],
                   help="Force a detection backend (auto = colour word → HSV, else YOLO).")
    p.add_argument("--mask", default="auto",
                   choices=["auto", "sam2", "ellipse"],
                   help="Mask tracker backend.")
    p.add_argument("--stereo", default="sgbm",
                   choices=["auto", "raft", "foundation", "sgbm"],
                   help="Stereo depth backend.")
    p.add_argument("--max-disp", dest="max_disp", type=int, default=384,
                   help="Maximum stereo disparity in pixels.")
    p.add_argument("--detect-every", type=int, default=5,
                   help="Run the open-vocab detector every N ticks.")

    # ---- gaze / approach ------------------------------------------------
    p.add_argument("--gaze-kp-tilt", type=float, default=0.48,
                   help="Proportional gain for the tilt (shoulder_lift) servo.")
    p.add_argument("--approach-el-deg", type=float, default=60.0,
                   help="Elevation angle of the approach axis (deg, 90 = top-down).")
    p.add_argument("--approach-coarse-look-at", type=bool, default=False,
                   help="Use look-at IK during the coarse approach phase.")
    p.add_argument("--approach-depth-priority", type=bool, default=True,
                   help="Prioritise depth servo over pixel centering.")
    p.add_argument("--approach-max-lin-vel-m-s", type=float, default=0.032,
                   help="Maximum Cartesian linear velocity during approach (m/s).")
    p.add_argument("--approach-gaze-scale-coarse", type=float, default=0.35,
                   help="Gaze gain scale during coarse approach phase.")
    p.add_argument("--approach-gaze-max-tilt-deg-coarse", type=float, default=3.0,
                   help="Maximum tilt step (deg) during coarse approach.")
    p.add_argument("--approach-steep-always-optical", type=bool, default=True,
                   help="When el > 45°, always approach along optical axis.")
    p.add_argument("--approach-style", default="angled",
                   choices=["angled", "topdown", "horizontal"],
                   help="How the gripper comes in on the object.")
    p.add_argument("--approach-coarse-ik-orientation-weight",
                   type=float, default=0.0)
    p.add_argument("--approach-pause-vertical-err-px",
                   type=float, default=0.0)
    p.add_argument("--final-standoff-m", type=float, default=0.06,
                   help="Camera-to-object distance to stop approaching (m).")
    p.add_argument("--target-physical-size-m", type=float, default=0.03)
    p.add_argument("--bbox-depth-scale", type=float, default=1.0)
    p.add_argument("--bbox-depth-offset-m", type=float, default=0.02)

    # ---- grasp ----------------------------------------------------------
    p.add_argument("--grasp-enable", type=bool, default=True)
    p.add_argument("--grasp-trigger-standoff-m", type=float, default=0.10)
    p.add_argument("--grasp-final-approach-m", type=float, default=0.04)
    p.add_argument("--grasp-final-approach-along-optical", type=bool, default=True)
    p.add_argument("--grasp-post-contact-squeeze-pct", type=float, default=8.0)
    p.add_argument("--grasp-lift-confirm", type=bool, default=True)

    # ---- search ---------------------------------------------------------
    p.add_argument("--search-startup-probe", type=bool, default=True)
    p.add_argument("--search-startup-lock-frames", type=int, default=2)

    # ---- preposition / live control ------------------------------------
    p.add_argument("--preposition-enabled", type=bool, default=False)
    p.add_argument("--live-control-stdin", type=bool, default=True,
                   help="Read single-key commands from stdin.")
    p.add_argument("--live-control-keypress", type=bool, default=True,
                   help="Same as --live-control-stdin (alias).")
    p.add_argument("--live-keys-auto-preposition", type=bool, default=True)
    p.add_argument("--live-keys-snap-targets", type=bool, default=True)
    p.add_argument("--live-key-max-steps-per-tick", type=int, default=8)
    p.add_argument("--live-el-slew-deg-s", type=float, default=72.0)
    p.add_argument("--live-preposition-boost-duration-s", type=float, default=1.2)
    p.add_argument("--live-preposition-boost-lin-vel-m-s", type=float, default=0.16)

    # ---- display -------------------------------------------------------
    p.add_argument("--display-data", type=bool, default=True,
                   help="Show live OpenCV window with detection overlay.")
    p.add_argument("--display-sim3d", type=bool, default=True,
                   help="Open Rerun 3-D viewer (same as --rerun).")
    p.add_argument("--rerun", action="store_true",
                   help="Alias for --display-sim3d=true.")

    # ---- misc ----------------------------------------------------------
    p.add_argument("--max-ticks", type=int, default=600)
    p.add_argument("--loop-hz", type=float, default=20.0)
    p.add_argument("-v", "--verbose", action="store_true")

    return p


# ---------------------------------------------------------------------------
# Keyboard nudge (--live-control-keypress)
# ---------------------------------------------------------------------------

class _KeyboardNudge:
    """Non-blocking keyboard listener that queues single-char commands.

    Keys:  [ ] move EE left/right   , . forward/back   - = depth in/out
    """

    HELP = (
        "Keys: [ ] ← →   , . ↑ ↓   - = depth   "
        "space=stop   q=quit"
    )

    def __init__(self) -> None:
        self._cmd: str | None = None
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()
        print(self.HELP)

    def _reader(self) -> None:
        try:
            import tty, termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                while True:
                    ch = sys.stdin.read(1)
                    with self._lock:
                        self._cmd = ch
                    if ch == "q":
                        break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            # Windows / not a tty — fall back to line input
            while True:
                try:
                    line = input()
                    with self._lock:
                        self._cmd = line.strip()[:1] if line.strip() else None
                except EOFError:
                    break

    def pop(self) -> str | None:
        with self._lock:
            cmd, self._cmd = self._cmd, None
            return cmd

    def wants_quit(self) -> bool:
        return self.pop() == "q"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _annotate_frame(left: np.ndarray, cloud, state: str) -> np.ndarray:
    """Draw detection boxes + state text on the left camera frame."""
    import cv2

    img = np.ascontiguousarray(left[:, :, ::-1])   # RGB → BGR
    for tr in cloud.list_tracks():
        x1, y1, x2, y2 = (int(v) for v in tr.box)
        is_focus = (tr.tag == cloud.focus_tag)
        color = (0, 0, 255) if is_focus else (0, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if is_focus else 1)
        depth = tr.centroid[2] if tr.has_cloud else float("nan")
        label = f"{'*' if is_focus else ''}#{tr.tag} {tr.label} z={depth:.2f}m"
        cv2.putText(img, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(img, f"[{state}]", (8, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return img


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ---- build the arm --------------------------------------------------
    port = args.robot_port
    if not port:
        print("!! --robot.port is required (e.g. /dev/ttyACM0 or /dev/ttyS3)")
        sys.exit(1)

    from robots.arms.lerobot_so101.driver import So101Arm
    arm = So101Arm(
        port=port,
        urdf=args.urdf,
        ee_frame=args.ee_frame,
        camera_key=args.camera_key,
        gripper_camera_tf=args.gripper_camera_tf,
    )

    # ---- build detection stack -----------------------------------------
    from models.depth import make_stereo
    from models.detection import make_detector, make_mask_tracker

    # Auto-select detector: colour word in query → HSV blob, else YOLO-World.
    det_backend = args.detector
    if det_backend == "auto":
        from models.detection.prompt_detector import _HSV_RANGES
        has_colour = any(w in _HSV_RANGES for w in args.query.lower().split())
        det_backend = "color_blob" if has_colour else "yolo"

    detector    = make_detector(det_backend, model_path=args.model_path,
                                min_confidence=args.min_conf)
    mask_tracker = make_mask_tracker(args.mask)
    stereo       = make_stereo(args.stereo, max_disp_px=args.max_disp)

    # ---- build cloud tracker -------------------------------------------
    from perception.depth_cloud import CloudTracker, PointCloudStream
    from perception.depth_cloud.stream import CloudUpdate

    stream = PointCloudStream()

    def _print_cloud(updates: list[CloudUpdate]) -> None:
        parts = [
            f"{'*' if u.is_focus else ' '}tag{u.tag}:{u.label}"
            f"[{u.points.shape[0]}pts c=({u.centroid[0]:+.2f},"
            f"{u.centroid[1]:+.2f},{u.centroid[2]:+.2f})]"
            for u in updates
        ]
        print(" cloud:", " ".join(parts))

    stream.subscribe(_print_cloud)
    extra = (args.place_on,) if args.place_on else ()
    cloud = CloudTracker(
        detector, mask_tracker, stereo, args.query,
        extra_labels=extra,
        detect_every=args.detect_every,
        stream=stream,
    )

    # ---- Rerun / display -----------------------------------------------
    viz = None
    if args.display_sim3d or args.rerun:
        try:
            from perception.depth_cloud.rerun_viz import RerunViz
            viz = RerunViz.try_create()
            if viz:
                print("[gaze-engine] Rerun viewer open")
        except Exception as e:
            print(f"[gaze-engine] Rerun unavailable: {e}")

    show_cv = args.display_data
    if show_cv:
        try:
            import cv2
            cv2.namedWindow("gaze-engine", cv2.WINDOW_NORMAL)
        except Exception:
            show_cv = False

    # ---- keyboard control ----------------------------------------------
    kb: _KeyboardNudge | None = None
    if args.live_control_keypress or args.live_control_stdin:
        try:
            kb = _KeyboardNudge()
        except Exception:
            pass

    # ---- gaze engine ---------------------------------------------------
    from manipulation.arms.gaze_engine import GazeConfig, GazeEngine, DONE, FAILED
    from manipulation.arms.grasp import GraspConfig

    cfg = GazeConfig(
        T_ee_cam=arm.T_ee_cam,
        grasp_range_m=args.grasp_trigger_standoff_m,
        final_advance_m=args.grasp_final_approach_m,
        max_lin_vel_m_s=args.approach_max_lin_vel_m_s,
        approach_style=args.approach_style,
        gaze_kp_tilt=args.gaze_kp_tilt,
    )

    engine = GazeEngine(arm, arm.kin, cloud, cfg, place_on_label=args.place_on)

    print(f"[gaze-engine] query={args.query!r} port={port} detector={detector.name} "
          f"stereo={stereo.name} approach={args.approach_style}")
    print("[gaze-engine] Starting — arm WILL move. Ctrl-C to stop.")

    dt = 1.0 / max(1.0, args.loop_hz)
    final = engine.state
    try:
        for _ in range(args.max_ticks):
            if kb and kb.wants_quit():
                print("[gaze-engine] user quit")
                break

            final = engine.step(dt)

            if viz and engine.last_obs is not None:
                focus  = cloud.focus_track()
                u_aim = v_aim = None
                if focus is not None:
                    u_aim, v_aim = engine._gaze_uv(focus, engine.last_obs)
                viz.log(engine.last_obs, cloud, final,
                        kin=arm.kin, depth_m=engine.range_m,
                        u_aim=u_aim, v_aim=v_aim,
                        aim_v_offset_px=engine._aim_v_now)

            if show_cv and engine.last_obs is not None:
                try:
                    import cv2
                    frame = _annotate_frame(engine.last_obs.left, cloud, final)
                    cv2.imshow("gaze-engine", frame)
                    cv2.waitKey(1)
                except Exception:
                    pass

            if final in (DONE, FAILED):
                break
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[gaze-engine] interrupted")
    finally:
        if show_cv:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        arm.disconnect()

    print(f"[gaze-engine] done — final state: {final}")


if __name__ == "__main__":
    main()
