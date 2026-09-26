#!/usr/bin/env python3
"""Locate a tube by cap colour, aim at it, approach it in steps, grasp it, verify.

The ordering is the whole point. The previous version opened by driving to an extended
standoff pose before it had located anything -- a single unverified lunge of +65 on the
shoulder and +74 on the elbow -- and only then started servoing. Everything after that
first move was careful, which did not matter, because the lunge is what hit things.

Here nothing extends until there is a target:

  LOOK     raise the arm to a retracted pose that can see the mat. Raising and pulling
           in are safe; reaching out is not.
  AIM      rotate the BASE only, in small steps, until the cap is centred horizontally.
           Base rotation sweeps across the table rather than into it.
  APPROACH extend shoulder and elbow together in ~15 increments. After each one, look
           again and correct the base. Any step that loses sight of the cap is undone.
  GRASP    close, then read the gripper. Holding reads > 31.9; closed on nothing reads
           30.2. Measured across 113 demonstrations with no overlap, so a failed grasp
           is detected rather than assumed.

Every pose is clamped to the joint box the 113 human demonstrations actually visited.
The operator never collided, so that box is a grounded safety bound -- and it catches
the specific bug that drove the arm into the bench, a base sweep of -40 units when no
demonstration ever moved the base outside -15.4 .. +12.7.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "Documents/lab-robot"))
sys.path.insert(0, str(Path.home() / "Documents/lab-robot/perception"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pick")

MOTORS = ["base", "shoulder_2", "elbow", "wrist", "tool", "gripper"]
RUNS = Path.home() / "smolvla_runs"
POSES = json.loads((RUNS / "poses.json").read_text())
ENVELOPE = json.loads((RUNS / "safe_envelope.json").read_text())
LOOK, GRASP = POSES["look"], POSES["grasp"]
PLACE = json.loads((RUNS / "place_poses.json").read_text())
# Routing comes from the demonstrations: across ~90 confidently-judged episodes the
# operator sent gold to the grey rack and blue/green to the black one, with no exceptions.
RACK_FOR = {"gold": "grey", "blue": "black", "green": "black"}
GRIP_OPEN, GRIP_SHUT, HOLDING = 54.0, 30.2, 31.9
# The gripper's place in the wrist view, measured once and stored. The camera is bolted
# to the wrist so this is a constant (+-1.8px over a whole session). Detecting it live
# every frame is what produced the wandering target and the "locked onto its own finger"
# failures, because the white tape reads as a blue cap.
_GEO = json.loads((RUNS / "gripper_geometry.json").read_text())
GRASP_POINT = tuple(_GEO["grasp_point"])
FINGERS = [tuple(_GEO["finger_left"]), tuple(_GEO["finger_right"])]
EXCLUDE_R = float(_GEO["exclude_radius"])
TARGET_FALLBACK = GRASP_POINT
# How much of the demonstrated descent to actually travel. Less than 1.0 keeps the jaws
# off the table; the grasp still closes because the open jaws are wider than a tube.
# A fixed fraction of the demonstrated descent cannot be right for every tube: that
# depth is a median, so a tube nearer than the median gets overshot by a couple of
# centimetres and a further one is not reached. 0.93 stopped short and closed on the cap
# (39.7, above the 31.7-34.8 body-grip band); 0.97 overshot forward.
#
# Use apparent cap SIZE as the proximity signal instead -- it grows as the camera closes
# in, and it was measured directly at the grasp moment across the demonstrations:
# median area 1548px, lower quartile 528px. Stop when the cap is that big, whatever
# depth that happens to correspond to, and keep the fraction only as a backstop.
DEPTH_FRACTION = 1.0
# How close, in pixels, the cap must be to the measured grasp point before closing.
# Roughly the jaw half-span, so the tube really is between the fingers.
GRASP_RADIUS = 70.0
# The view as a grid, which is how the operator reasons about it: the cap sits in a
# cell, the jaws sit in a cell, and the job is to make those the same cell.
GRID_COLS, GRID_ROWS = 10, 8
CELL_W, CELL_H = 640 / GRID_COLS, 480 / GRID_ROWS


def cell_of(x: float, y: float) -> tuple[int, int]:
    return int(min(max(x // CELL_W, 0), GRID_COLS - 1)), int(min(max(y // CELL_H, 0), GRID_ROWS - 1))


def clamp(pose: dict) -> dict:
    """Hold every joint inside the box the 113 demonstrations visited."""
    out = {}
    for m, v in pose.items():
        lo, hi = ENVELOPE.get(m, (-1e9, 1e9))
        out[m] = float(min(max(v, lo), hi))
    return out


def settled_gripper(robot, timeout: float = 1.5) -> float:
    """Read the gripper only once it has stopped moving.

    Reading a fixed 0.9s after commanding a close catches the jaws mid-travel: an empty
    close reported 37.70, which is above the holding threshold, so the run declared a
    successful grasp while the jaws were plainly empty. The closed-on-nothing value is
    30.2 and it is only reached once motion has finished.
    """
    t0 = time.time()
    last = robot.get_observation()["gripper.pos"]
    stable = 0
    while time.time() - t0 < timeout:
        time.sleep(0.08)
        v = robot.get_observation()["gripper.pos"]
        stable = stable + 1 if abs(v - last) < 0.06 else 0
        last = v
        if stable >= 3:
            break
    return float(last)


def pose_of(robot) -> dict:
    o = robot.get_observation()
    return {m: float(o[f"{m}.pos"]) for m in MOTORS}


def goto(robot, target: dict, steps: int = 18, dwell: float = 0.03,
         tol: float = 3.0, timeout: float = 1.2) -> bool:
    """Ramp to a pose AND WAIT until the arm is actually there.

    The previous version sent a fixed schedule of interpolated setpoints and returned.
    That is not the same as arriving: `max_relative_target` clamps every command to a few
    units from where the arm currently is, so if the arm lags the ramp runs out while the
    joints are still short of the goal -- the log filled with

        Relative goal position magnitude had to be clamped to be safe.
        {'elbow': {'original goal_pos': -58.0, 'safe goal_pos': -67.69}}

    and `goto` reported success ten units away. Everything downstream then reasoned about
    a pose the arm was never in: the LOOK pose was never reached, so the fingers were out
    of frame and the grasp point silently fell back to a guessed constant, the cap sat
    high, and the descent closed on air.

    Now the ramp is followed by a settle loop that re-sends the goal until the joints are
    within tolerance, and the caller learns whether it actually got there.
    """
    target = clamp(target)
    cur = pose_of(robot)
    for i in range(1, steps + 1):
        a = i / steps
        robot.send_action({f"{m}.pos": cur[m] + a * (target.get(m, cur[m]) - cur[m]) for m in MOTORS})
        time.sleep(dwell)

    # Settle: keep commanding the goal until the arm reaches it. The clamp still limits
    # each step, so this is simply how long that takes.
    deadline = time.time() + timeout
    watched = [m for m in MOTORS if m != "gripper"]
    # 1.5 units was tighter than the arm settles to under gravity; it timed out "off by
    # 7.4" while a single clamped step from the goal. 3.0 is about a millimetre at the
    # tool and is reached reliably.
    stalled = 0
    prev = None
    while time.time() < deadline:
        now = pose_of(robot)
        err = max(abs(now[m] - target.get(m, now[m])) for m in watched)
        if err <= tol:
            return True
        # If it stops improving it is against a limit or an obstruction; do not grind.
        if prev is not None and prev - err < 0.15:
            stalled += 1
            if stalled >= 4:
                logger.warning("  goto stalled %.1f from the goal - not forcing it", err)
                return False
        else:
            stalled = 0
        prev = err
        robot.send_action({f"{m}.pos": target.get(m, now[m]) for m in MOTORS})
        time.sleep(0.03)
    now = pose_of(robot)
    worst = max(watched, key=lambda m: abs(now[m] - target.get(m, now[m])))
    logger.warning("  goto did not arrive: %s off by %.1f", worst,
                   abs(now[worst] - target.get(worst, now[worst])))
    return False


def see(robot, colour: str, n: int = 2, near=None):
    """Where the chosen cap is. The grasp point is a constant, not a measurement.

    This used to locate the fingers every frame and derive the grasp point from them.
    That was the single biggest source of bad behaviour: the tape reads as a blue cap, so
    the pairing sometimes picked the wrong blobs, and the target the servo was driving
    toward moved between frames. The fingers are bolted to the camera -- the point is
    fixed, and caps detected on top of the fingers are simply discarded.
    """
    import cv2

    from caps2 import find_caps

    pts, frame = [], None
    for _ in range(n):
        o = robot.get_observation()
        frame = cv2.cvtColor(o["wrist"], cv2.COLOR_RGB2BGR)
        hits = [c for c in find_caps(frame, restrict_to_mat=True) if c.colour == colour]
        hits = [c for c in hits
                if all((c.x - fx) ** 2 + (c.y - fy) ** 2 > EXCLUDE_R ** 2 for fx, fy in FINGERS)]
        if hits:
            h = (min(hits, key=lambda c: (c.x - near[0]) ** 2 + (c.y - near[1]) ** 2)
                 if near else max(hits, key=lambda c: c.area))
            pts.append((h.x, h.y, h.area))
    cap = tuple(np.median(np.array(pts), axis=0)) if pts else None
    return cap, GRASP_POINT, frame


def measure_twist_gain(robot, cap_xy, d: float = 10.0) -> float | None:
    """Degrees the TUBE appears to rotate per unit of `tool`.

    First attempt measured this on the jaw line and returned "unmeasurable" every single
    run. The reason is the mounting: the wrist camera sits past the tool joint, so the
    camera and the fingers rotate together and the jaw line never moves in the image.
    What moves is the rest of the world -- so measure the tube instead. The alignment
    goal is then the same either way: tube axis perpendicular to the (fixed) jaw line.
    """
    import cv2

    from orient import tube_axis

    def ang(near):
        vals = []
        for _ in range(3):
            fr = cv2.cvtColor(robot.get_observation()["wrist"], cv2.COLOR_RGB2BGR)
            ax = tube_axis(fr, near)
            if ax is not None:
                vals.append(ax.angle_deg)
            time.sleep(0.04)
        return float(np.median(vals)) if vals else None

    a0 = ang(cap_xy)
    if a0 is None:
        return None
    home = pose_of(robot)
    goto(robot, {**home, "tool": home["tool"] + d}, steps=8)
    time.sleep(0.3)
    a1 = ang(cap_xy)
    goto(robot, home, steps=8)
    time.sleep(0.3)
    if a1 is None:
        return None
    delta = a1 - a0
    while delta > 90:
        delta -= 180
    while delta < -90:
        delta += 180
    g = delta / d
    return g if abs(g) > 0.15 else None


def measure_local(robot, colour: str, near, d: float = 2.0):
    """Measure how the cap moves in the image for a base nudge and a reach nudge, here.

    Columns are [d(image)/d(base), d(image)/d(reach)]. Measured fresh rather than
    assumed, because the wrist camera is carried by the arm and tilts as it extends.
    """
    base, _, _ = see(robot, colour, n=2, near=near)
    if base is None:
        return None
    cols = []
    for name, delta in (("base", {"base": d}), ("reach", {"shoulder_2": d, "elbow": d})):
        home = pose_of(robot)
        goto(robot, {**home, **{k: home[k] + v for k, v in delta.items()}}, steps=6)
        time.sleep(0.3)
        moved, _, _ = see(robot, colour, n=2, near=(base[0], base[1]))
        goto(robot, home, steps=6)
        time.sleep(0.3)
        if moved is None:
            return None
        cols.append([(moved[0] - base[0]) / d, (moved[1] - base[1]) / d])
    J = np.array(cols).T
    return J if abs(np.linalg.det(J)) > 0.4 else None


def place(robot, colour: str, ui, steps: int = 16) -> bool:
    """Carry the held tube to its rack and drop it into a free hole.

    Same split as the pick, for the same reason: the vertical profile comes from the
    demonstrations (every one ends with a tube released into a hole) and vision supplies
    the horizontal, because that is the part that varies -- which hole is free today.

    Holes are only found from the wrist camera. From overhead a rack is ~65x85px with
    ~8px holes and Hough finds two of them; from the demonstrated release pose the rack
    fills the wrist frame with ~45px holes.
    """
    import cv2

    from holes import draw as draw_holes
    from holes import find_holes, pick_free_hole
    rack = RACK_FOR[colour]
    prof = PLACE[rack]
    logger.info("[PLACE] %s cap -> %s rack", colour, rack)

    # Carry in stages, gently. A single move swung the base 34 units and rotated the
    # wrist at the same time, and the tube was repeatedly shed on the way -- the jaws
    # hold it by friction, so acceleration is what loses it. Lift clear first, then
    # rotate, then settle into the carry pose, checking the grip at each stage so a drop
    # is caught where it happened rather than at the end.
    logger.info("[PLACE] carrying to the %s rack, in stages", rack)
    carry = dict(prof["carry"])
    here = pose_of(robot)

    # FOLD, rotate, unfold. Lifting by a fixed number of units still left the arm
    # extended out over the table, so swinging toward the rack dragged the tube across
    # the mat and past the rack edges. Pull the arm back into the compact looking pose
    # first -- the same shoulder/elbow the LOOK phase uses -- then rotate, then reach
    # back out over the rack.
    #
    # Folding also puts the tube close to the base's axis of rotation, so the swing
    # applies far less force to a gently-held tube and can be taken faster than the
    # crawl an extended rotation needed.
    folded = {**here, "shoulder_2": LOOK["shoulder_2"], "elbow": LOOK["elbow"],
              "wrist": here["wrist"]}
    stages = [
        ("fold up", folded),
        ("rotate folded", {**folded, "base": carry["base"]}),
        ("unfold over rack", carry),
    ]
    for name, target in stages:
        # The rotation is the dangerous one and gets its own, much gentler profile. The
        # base swings ~34 units to reach the rack, and the gripper deliberately runs at a
        # reduced PWM limit (600 against the arm's 885) so it does not cook itself, which
        # means it holds the tube gently. Swinging that fast shears it out of the jaws.
        goto(robot, {**target, "gripper": GRIP_SHUT}, steps=26, dwell=0.03)
        time.sleep(0.25)
        g = settled_gripper(robot, timeout=1.5)
        if g <= HOLDING:
            logger.error("[PLACE] tube lost during '%s' (gripper %.2f)", name, g)
            return False
        logger.info("  [PLACE] %-16s ok (gripper %.2f)", name, g)
    ui("PLACE-CARRY", note=f"carrying to the {rack} rack")

    # Descend a third of the way before choosing a hole. At the carry pose the rack sits
    # small and in the corner of the wrist view -- measured 5 holes visible there against
    # 15 from lower down -- so choosing from the carry pose picks from a partial view.
    rel0 = prof["release"]
    c0 = pose_of(robot)
    goto(robot, {**c0,
                 "shoulder_2": c0["shoulder_2"] + 0.33 * (rel0["shoulder_2"] - c0["shoulder_2"]),
                 "elbow": c0["elbow"] + 0.33 * (rel0["elbow"] - c0["elbow"]),
                 "wrist": c0["wrist"] + 0.33 * (rel0["wrist"] - c0["wrist"]),
                 "gripper": GRIP_SHUT}, steps=16)
    time.sleep(0.5)

    # Find a free hole from here.
    frame = cv2.cvtColor(robot.get_observation()["wrist"], cv2.COLOR_RGB2BGR)
    gp = GRASP_POINT
    hs = find_holes(frame)
    target_hole = pick_free_hole(hs, prefer=gp)
    logger.info("[PLACE] %d holes visible, %d free", len(hs), sum(h.free for h in hs))
    if target_hole is None:
        logger.error("[PLACE] no free hole visible in the %s rack", rack)
        return False
    logger.info("[PLACE] target hole at (%.0f,%.0f), cell %s",
                target_hole.x, target_hole.y, cell_of(target_hole.x, target_hole.y))
    cv2.imwrite(str(RUNS / f"place_target_{rack}.png"), draw_holes(frame, hs, target_hole))

    # Descend along the demonstrated release profile, correcting base toward the hole.
    start = pose_of(robot)
    rel = prof["release"]
    sign = 1.0
    hole_xy = (target_hole.x, target_hole.y)
    for step in range(steps):
        a = (step + 1) / steps
        cur = pose_of(robot)
        goto(robot, {**cur,
                     "shoulder_2": start["shoulder_2"] + a * (rel["shoulder_2"] - start["shoulder_2"]),
                     "elbow": start["elbow"] + a * (rel["elbow"] - start["elbow"]),
                     "wrist": start["wrist"] + a * (rel["wrist"] - start["wrist"]),
                     "gripper": GRIP_SHUT}, steps=8)
        time.sleep(0.28)

        frame = cv2.cvtColor(robot.get_observation()["wrist"], cv2.COLOR_RGB2BGR)
        gp = GRASP_POINT
        hs = find_holes(frame)
        near = min((h for h in hs if h.free), key=lambda h: (h.x - hole_xy[0]) ** 2 + (h.y - hole_xy[1]) ** 2,
                   default=None)
        if near is None:
            logger.info("  [PLACE] %2d/%d: lost the hole this frame", step + 1, steps)
            ui("PLACE-DESCEND", note=f"descending {step + 1}/{steps} - hole not in view")
            continue
        hole_xy = (near.x, near.y)
        ex = gp[0] - near.x
        logger.info("  [PLACE] %2d/%d: hole cell %s -> jaw cell %s  dx %+.0f",
                    step + 1, steps, cell_of(near.x, near.y), cell_of(*gp), ex)
        ui("PLACE-DESCEND", cap=hole_xy, target=gp, err=abs(ex),
           note=f"descending {step + 1}/{steps} toward the free hole")
        # Same scaled gain as the pick: the camera closes on the rack as it descends.
        if abs(ex) > 35:
            q = sign * float(np.clip(abs(ex) / (18.0 * (1.0 + 2.6 * a)), 0.25, 2.0))
            cur = pose_of(robot)
            goto(robot, {**cur, "base": cur["base"] + q}, steps=6)
            time.sleep(0.2)
            f2 = cv2.cvtColor(robot.get_observation()["wrist"], cv2.COLOR_RGB2BGR)
            g2 = GRASP_POINT
            h2 = find_holes(f2)
            n2 = min((h for h in h2 if h.free), key=lambda h: (h.x - hole_xy[0]) ** 2 + (h.y - hole_xy[1]) ** 2,
                     default=None)
            if n2 is not None:
                if abs(g2[0] - n2.x) > abs(ex) + 8:
                    sign *= -1
                    logger.info("        base going the wrong way, flipped")
                hole_xy = (n2.x, n2.y)

    logger.info("[PLACE] releasing")
    cur = pose_of(robot)
    goto(robot, {**cur, "gripper": GRIP_OPEN}, steps=12)
    time.sleep(0.8)
    g = settled_gripper(robot, timeout=2.0)
    released = g > 45.0
    logger.info("[PLACE] gripper %.2f -> %s", g, "released" if released else "STILL GRIPPING")
    ui("PLACE-DONE", note=f"gripper {g:.2f}")

    # Verify from above: back off and check the hole now holds a cap.
    back = pose_of(robot)
    goto(robot, {**back, "shoulder_2": back["shoulder_2"] - 10.0, "gripper": GRIP_OPEN}, steps=14)
    time.sleep(0.6)
    frame = cv2.cvtColor(robot.get_observation()["wrist"], cv2.COLOR_RGB2BGR)
    hs = find_holes(frame)
    filled = [h for h in hs if not h.free]
    cv2.imwrite(str(RUNS / f"placed_{rack}.png"), draw_holes(frame, hs))
    logger.info("[PLACE] after release: %d occupied holes visible (photo -> %s)",
                len(filled), RUNS / f"placed_{rack}.png")
    return released


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colour", required=True, choices=["gold", "blue", "green"])
    ap.add_argument("--port", default="/dev/tty.usbserial-FTA9DQBQ")
    # -1 means "work it out from what the cameras see". macOS reshuffles OpenCV indices
    # between runs and the built-in FaceTime camera landed on index 1 three times in this
    # project, each time producing a full silent run against a picture of the room.
    ap.add_argument("--top-cam", type=int, default=-1)
    ap.add_argument("--wrist-cam", type=int, default=-1)
    ap.add_argument("--aim-steps", type=int, default=14)
    ap.add_argument("--aim-tol", type=float, default=150.0,
                    help="how centred is good enough before approaching")
    ap.add_argument("--approach-steps", type=int, default=15)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--ui", action="store_true")
    ap.add_argument("--place", action="store_true", help="after a confirmed grasp, place it in a rack")
    ap.add_argument("--twist", action="store_true",
                    help="rotate the wrist so the jaws close across the tube axis")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        logger.info("LOOK  %s", {k: round(v, 1) for k, v in LOOK.items()})
        logger.info("GRASP %s", {k: round(v, 1) for k, v in GRASP.items()})
        logger.info("envelope %s", {k: [round(x, 1) for x in v] for k, v in ENVELOPE.items()})
        return

    from lerobot.cameras.opencv import OpenCVCameraConfig

    from x250_driver import X250Follower, X250FollowerConfig

    if args.top_cam < 0 or args.wrist_cam < 0:
        import cameras as _cams

        # Motion test, not content scoring. Content scoring put the operator's FaceTime
        # camera in the wrist slot and a whole run executed against a picture of his face.
        # NOTE: the motion probe reliably identifies the wrist camera (45x margin) but
        # leaves one device in a mode the robot's connect then refuses. Until that is
        # sorted, pass --top-cam/--wrist-cam explicitly for a production run.
        t, w, diffs = _cams.find_by_motion(args.port)
        for i, v in sorted(diffs.items()):
            logger.info("  camera %d: frame change when the arm moved %.2f", i, v)
        scores = []
        if w is None:
            sys.exit("no camera is looking at the mat - check the wrist camera")
        args.wrist_cam = w if args.wrist_cam < 0 else args.wrist_cam
        args.top_cam = (t if t is not None else w) if args.top_cam < 0 else args.top_cam
    logger.info("cameras: overhead=%d  wrist=%d", args.top_cam, args.wrist_cam)
    # The probe just released these devices, and one of them comes back in 1920 mode,
    # which makes the robot's connect fail outright. Pin the mode before handing over.
    import cameras as _cams2

    modes = _cams2.force_mode([args.top_cam, args.wrist_cam])
    logger.info("camera modes after reset: %s", modes)

    # Only the WRIST camera belongs on the robot. The overhead feed is context for the
    # operator's window and nothing in the control loop reads it -- but while it was
    # attached to the robot, a stale overhead frame aborted runs that were otherwise
    # going fine (2260ms late, mid-pick). It is opened separately and best-effort below.
    cams = {"wrist": OpenCVCameraConfig(index_or_path=args.wrist_cam, width=640, height=480, fps=30)}
    robot = X250Follower(X250FollowerConfig(id="x250_follower", port=args.port, cameras=cams,
                                            max_relative_target="8.0", home_on_connect=False))
    robot.connect()

    # Best-effort overhead feed for the window. Failures here never stop a pick.
    top_cap = None
    if args.ui:
        try:
            import cv2 as _cv

            top_cap = _cv.VideoCapture(args.top_cam)
            if top_cap.isOpened():
                top_cap.set(_cv.CAP_PROP_FRAME_WIDTH, 640)
                top_cap.set(_cv.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                top_cap = None
        except Exception:
            top_cap = None

    def ui(phase, cap=None, target=None, err=None, note=""):
        if not args.ui:
            return
        try:
            import cv2

            import ui as _ui
            from caps2 import find_caps
            from jaws import find_jaws, grasp_point

            from orient import tube_axis, twist_error

            o = robot.get_observation()
            w = cv2.cvtColor(o["wrist"], cv2.COLOR_RGB2BGR)
            t = None
            if top_cap is not None:
                ok_t, f_t = top_cap.read()
                if ok_t:
                    t = f_t
            ax = tube_axis(w, cap) if cap else None
            te = twist_error(w, cap) if cap else None
            # Draw what the CONTROLLER sees, not the raw detector: the jaws' own tape
            # reads as a blue cap, and showing it made the display look like it was
            # tracking the gripper even when the servo had correctly locked the tube.
            shown = [c for c in find_caps(w, restrict_to_mat=True)
                     if all((c.x - fx) ** 2 + (c.y - fy) ** 2 > EXCLUDE_R ** 2 for fx, fy in FINGERS)]
            _ui.show(w, t, caps=shown, jaws_pair=(FINGERS[0], FINGERS[1]), grasp=GRASP_POINT,
                     chosen=cap, target=target, phase=phase, gripper=o["gripper.pos"], err=err,
                     pose={m: o[f"{m}.pos"] for m in ("base", "shoulder_2", "elbow", "gripper")},
                     note=note, axis=ax, twist=(te[0] if te else None))
        except Exception:
            pass

    try:
        for attempt in range(1, args.attempts + 1):
            logger.info("=== attempt %d/%d: %s cap ===", attempt, args.attempts, args.colour)

            # ---- LOOK ------------------------------------------------------------
            logger.info("[LOOK] raising to the retracted looking pose")
            goto(robot, {**LOOK, "gripper": GRIP_OPEN}, steps=26)
            time.sleep(0.5)
            ui("LOOK", note="raised, not extended - nothing committed yet")

            pinned = GRASP_POINT
            logger.info("[LOOK] grasp point (fixed): (%d, %d)", *GRASP_POINT)
            cap, _live, _ = see(robot, args.colour, n=4)
            target = pinned
            logger.info("[LOOK] cap %s", f"at ({cap[0]:.0f},{cap[1]:.0f})" if cap else "not visible")

            # ---- AIM: base rotation only ----------------------------------------
            # Aim means "get the tube into view and roughly toward the middle", not
            # "centre it perfectly". The cap sits high in the frame at this retracted
            # pose and its x error cannot be fully removed until the arm extends -- the
            # APPROACH phase corrects x continuously as it closes in. Demanding a tight
            # centring here just swept the base back and forth until it ran out of steps.
            logger.info("[AIM] rotating base only - no extension")
            locked = (cap[0], cap[1]) if cap else None
            sign_base = 1.0
            aimed = cap is not None and abs(target[0] - cap[0]) < args.aim_tol
            for step in range(args.aim_steps):
                if aimed:
                    break
                cap, _live, _ = see(robot, args.colour, n=2, near=locked)
                target = pinned
                cur = pose_of(robot)
                if cap is None:
                    # Not in view: step the base, always from where we are now. Mixing an
                    # absolute sweep with incremental corrections meant each sweep undid
                    # whatever aiming had been achieved.
                    q = sign_base * 3.0
                    goto(robot, {**cur, "base": cur["base"] + q}, steps=8)
                    time.sleep(0.2)
                    logger.info("  [AIM] %2d: not in view, base %+.1f", step, q)
                    after, _, _ = see(robot, args.colour, n=2)
                    if after is None and step % 4 == 3:
                        sign_base *= -1      # swept that way long enough, try the other
                    elif after is not None:
                        locked = (after[0], after[1])
                    ui("AIM", note="sweeping the base to bring the tube into view")
                    continue

                locked = (cap[0], cap[1])
                ex = target[0] - cap[0]
                logger.info("  [AIM] %2d: cap x=%.0f  want %.0f  dx=%+.0f", step, cap[0], target[0], ex)
                ui("AIM", cap=locked, target=target, err=abs(ex), note="aiming with base rotation")
                if abs(ex) < args.aim_tol:
                    logger.info("  [AIM] in view and roughly aimed")
                    aimed = True
                    break
                q = sign_base * float(np.clip(abs(ex) / 14.0, 1.0, 2.5))
                goto(robot, {**cur, "base": cur["base"] + q}, steps=8)
                time.sleep(0.2)
                after, _, _ = see(robot, args.colour, n=2, near=locked)
                if after is not None:
                    if abs(target[0] - after[0]) > abs(ex) + 4:
                        sign_base *= -1
                        logger.info("      wrong way, flipping direction")
                    locked = (after[0], after[1])
            # ---- APPROACH: demonstrated profile + live horizontal correction ----
            # Servoing the vertical error in image space does not work here. Measured
            # locally, extending the arm moves the cap UP 31.8px per unit, because the
            # 30-degree wrist camera tilts as the arm reaches: image alignment says
            # "retract" while physical alignment says "extend". The two disagree, so the
            # image cannot decide the descent.
            #
            # The 113 demonstrations already contain the right vertical profile -- every
            # one of them ends with a tube in the jaws. So follow that profile for the
            # descent, and use vision for the thing that genuinely varies between runs:
            # WHICH tube, and where it is horizontally. Base correction every step keeps
            # the chosen cap lined up as the arm comes down.
            # Establish which way the base moves the cap BEFORE descending. Relying on a
            # flip-on-worse rule failed: the scaled-down corrections changed the error by
            # less than the 8px the flip test needs, so a wrong sign was never detected
            # and dx sat at +150..+190 for a whole descent. The grasp then closed 156px
            # off-centre and held the tube by its edge at 31.98, just over the 31.9 line.
            # Measure the base direction properly, with a big enough move to beat the
            # detector's noise, and retry if the first attempt is ambiguous. A 3-unit
            # probe moved the cap less than the +-8px noise floor, so the sign stayed a
            # guess -- and then the flip-on-worse rule toggled it at random every step,
            # leaving dx pinned at +100 for an entire descent while "correcting".
            sign_base = None
            for probe_q in (7.0, -10.0, 14.0):
                if cap is None:
                    break
                probe_from = pose_of(robot)
                before_x = cap[0]
                goto(robot, {**probe_from, "base": probe_from["base"] + probe_q}, steps=10)
                time.sleep(0.3)
                p_after, _, _ = see(robot, args.colour, n=3, near=locked)
                goto(robot, probe_from, steps=10)
                time.sleep(0.3)
                if p_after is None:
                    continue
                moved = p_after[0] - before_x
                if abs(moved) < 18:
                    logger.info("[APPROACH] base %+.0f moved the cap only %+.0fpx - probing harder",
                                probe_q, moved)
                    continue
                # px of cap motion per unit of base, then the sign that shrinks the error
                gain = moved / probe_q
                sign_base = 1.0 if (pinned[0] - before_x) * gain > 0 else -1.0
                logger.info("[APPROACH] base gain %+.2f px/unit -> correcting with sign %+.0f",
                            gain, sign_base)
                break
            if sign_base is None:
                logger.warning("[APPROACH] could not measure the base direction - "
                               "descending without horizontal correction")
                sign_base = 0.0
            logger.info("[APPROACH] following the demonstrated descent, correcting base from vision")
            # Twist gain is measured LATER, a few steps into the descent. Measuring it
            # here failed every time ("unmeasurable") because the arm is still high and
            # its own fingers are not yet in frame, so there is no jaw line to rotate.
            twist_gain = None
            start_pose = pose_of(robot)
            n_steps = args.approach_steps
            reached = False
            steps_done = 0
            misses = 0
            progress = 0
            last_dist = None
            for step in range(n_steps):
                # LOOK FIRST, then move. The old order descended one increment and only
                # then checked, so a lost detection simply kept reaching: the log showed
                # "cap not in view" for five consecutive steps while the arm carried on
                # down, and it finished well forward of the tube with nothing between the
                # jaws. If the target is not visible the arm now holds position.
                cap, _live, _ = see(robot, args.colour, n=2, near=locked)
                target = pinned
                if cap is None:
                    misses += 1
                    # Losing sight at CLOSE range is how an approach is supposed to end:
                    # the tube passes under the jaws and out of the camera's view. Treating
                    # that as a failure left the arm stopped high, holding air. Losing it
                    # while still far away is a real failure and must not be driven through.
                    # Only treat a lost target as "arrived" once the arm is genuinely
                    # low. Completing the descent from high up grabbed the cap instead of
                    # the body (40.98, above the 31.7-34.8 band) and shed it on the lift.
                    if last_dist is not None and last_dist < 170 and (step + 1) / n_steps >= 0.7:
                        logger.info("  [APPROACH] target passed under the jaws at %.0fpx, %.0f%% down - "
                                    "completing the descent", last_dist, 100 * (step + 1) / n_steps)
                        cur = pose_of(robot)
                        goto(robot, {**cur,
                                     "shoulder_2": start_pose["shoulder_2"] + DEPTH_FRACTION * (GRASP["shoulder_2"] - start_pose["shoulder_2"]),
                                     "elbow": start_pose["elbow"] + DEPTH_FRACTION * (GRASP["elbow"] - start_pose["elbow"])},
                             steps=10)
                        time.sleep(0.2)
                        steps_done += 1
                        reached = True
                        break
                    logger.info("  [APPROACH] %2d: target not in view - holding (%d/4)",
                                step + 1, misses)
                    ui("APPROACH", note=f"target lost - holding, not descending ({misses}/4)")
                    if misses >= 4:
                        logger.warning("  [APPROACH] lost the %s cap while still %s - stopping",
                                       args.colour,
                                       f"{last_dist:.0f}px away" if last_dist else "far off")
                        break
                    time.sleep(0.25)
                    continue
                misses = 0
                progress += 1
                steps_done += 1
                # Depth follows the STEP, not the count of successful looks. Tying it to
                # `progress` meant four held steps left the arm at 59% depth -- too high,
                # closing on air well above the tube.
                a = ((step + 1) / n_steps) * DEPTH_FRACTION
                locked = (cap[0], cap[1])
                ex = target[0] - cap[0]
                ey = target[1] - cap[1]
                c_cap, c_jaw = cell_of(cap[0], cap[1]), cell_of(target[0], target[1])
                dist = float(np.hypot(ex, ey))
                last_dist = dist
                logger.info("  [APPROACH] %2d/%d: cap %s -> jaw %s  dx %+.0f dy %+.0f  dist %.0f",
                            step + 1, n_steps, c_cap, c_jaw, ex, ey, dist)
                ui("APPROACH", cap=locked, target=target, err=dist,
                   note=f"target {dist:.0f}px from the jaws")
                # Closing needs BOTH: lined up in the image AND down at tube height.
                # Image alignment alone fixes x and y and says nothing about depth -- one
                # attempt converged to 7px at 41% descent and closed on air well above the
                # tube. When it lines up early, hold the alignment and keep descending.
                at_depth = a >= 0.85
                if dist < GRASP_RADIUS and at_depth:
                    logger.info("  [APPROACH] between the jaws (%.0fpx) and at depth (%.0f%%) - closing",
                                dist, 100 * a)
                    reached = True
                    break
                if dist < GRASP_RADIUS:
                    logger.info("  [APPROACH] lined up (%.0fpx) but only %.0f%% down - continuing",
                                dist, 100 * a)

                # ONE observation, ONE move. Each step used to look, move the base, look
                # AGAIN to check that move, then move the descent -- two commands and two
                # observations per increment, which is where the slow back-and-forth came
                # from. Base and reach are independent joints, so they go in a single
                # command and the arm makes one smooth move per step.
                scale = 1.0 + 1.1 * a
                cur = pose_of(robot)
                nxt = dict(cur)
                if abs(ex) > 30 and sign_base != 0.0:
                    nxt["base"] = cur["base"] + sign_base * float(
                        np.clip(abs(ex) / (14.0 * scale), 0.4, 3.2 / scale))
                nxt["shoulder_2"] = start_pose["shoulder_2"] + a * (GRASP["shoulder_2"] - start_pose["shoulder_2"])
                nxt["elbow"] = start_pose["elbow"] + a * (GRASP["elbow"] - start_pose["elbow"])
                goto(robot, nxt, steps=6, timeout=0.5)

                # Twist the wrist so the jaws close ACROSS the tube rather than along it.
                # Both angles are read from the same image -- the tube axis from its
                # silhouette on the dark mat, the jaw line from the tape -- so aligning
                # them needs no calibration beyond the gain measured above.
                if args.twist and twist_gain is None and step == 3:
                    twist_gain = measure_twist_gain(robot, locked)
                    logger.info("      twist gain %s deg per unit of tool",
                                f"{twist_gain:+.2f}" if twist_gain else "unmeasurable, twist off")
                if twist_gain and step >= 4:
                    import cv2

                    from orient import twist_error

                    fr = cv2.cvtColor(robot.get_observation()["wrist"], cv2.COLOR_RGB2BGR)
                    te = twist_error(fr, locked)
                    if te is not None and abs(te[0]) > 10.0:
                        dt = float(np.clip(te[0] / twist_gain, -7.0, 7.0))
                        cur = pose_of(robot)
                        goto(robot, {**cur, "tool": cur["tool"] + dt}, steps=6)
                        time.sleep(0.2)
                        logger.info("        tube axis %+.0f deg, twisting tool %+.1f",
                                    te[1].angle_deg, dt)

                # Cell alignment alone is not enough to stop. A grid cell is 64x60px, so
                # the cells can match while the arm is still well above the tube -- two
                # runs "arrived" at shoulder -3.5 and +14.8 against a demonstrated grasp
                # depth of +29.4, closed on air, and read 30.20 (empty). Require the
                # descent to be essentially complete as well: it is the demonstrated
                # depth that puts the jaws at tube height, not the image.
            if steps_done == 0:
                logger.warning("[APPROACH] never advanced - not closing on nothing")
                continue
            if not reached:
                logger.info("[APPROACH] completed the demonstrated descent (%d steps)", steps_done)

            # ---- GRASP -----------------------------------------------------------
            ui("CLOSING", note="closing the jaws")
            logger.info("[GRASP] closing")
            cur = pose_of(robot)
            goto(robot, {**cur, "gripper": GRIP_SHUT}, steps=14)
            g = settled_gripper(robot)
            if g > HOLDING:
                logger.info("*** HOLDING A TUBE: gripper %.2f > %.1f ***", g, HOLDING)
                ui("HOLDING", note=f"gripper {g:.2f}")
                # Lift clear of the mat before anything else. Small 3-unit hops left the
                # tube barely off the surface, so the subsequent swing toward the rack
                # dragged it across the mat and over the rack edges.
                # The lift is the one move that must stay gentle: the tube is held by
                # friction at a deliberately low gripper PWM, and a brisk lift sheds it
                # even from a good grip. Everything else in this routine can be quick.
                for s in (4.0, 4.0, 4.0, 4.0, 4.0):
                    p = pose_of(robot)
                    goto(robot, {**p, "shoulder_2": p["shoulder_2"] - s,
                                 "elbow": p["elbow"] - s * 0.5, "gripper": GRIP_SHUT},
                         steps=18, dwell=0.04)
                    time.sleep(0.2)
                    if settled_gripper(robot, timeout=1.5) <= HOLDING:
                        logger.warning("  dropped during the lift")
                        break
                logger.info("lifted clear of the mat")
                g2 = settled_gripper(robot, timeout=2.0)
                logger.info("after lift: gripper %.2f -> %s", g2, "STILL HOLDING" if g2 > HOLDING else "dropped")
                if g2 > HOLDING:
                    import cv2

                    o = robot.get_observation()
                    cv2.imwrite(str(RUNS / f"held_{args.colour}.png"),
                                cv2.cvtColor(o["wrist"], cv2.COLOR_RGB2BGR))
                    logger.info("photo -> %s", RUNS / f"held_{args.colour}.png")
                    if args.place:
                        ok = place(robot, args.colour, ui)
                        logger.info("=== %s ===", "PICK AND PLACE COMPLETE" if ok else "placed but not confirmed")
                        return
                    ui("HOLDING", note="picked up - holding for 5s")
                    for _ in range(10):
                        p = pose_of(robot)
                        robot.send_action({f"{m}.pos": (GRIP_SHUT if m == "gripper" else p[m]) for m in MOTORS})
                        time.sleep(0.5)
                    return
            else:
                logger.warning("[GRASP] failed: gripper %.2f (empty reads %.1f)", g, GRIP_SHUT)
                ui("FAILED", note=f"gripper {g:.2f} - nothing in the jaws")
        logger.error("no successful grasp in %d attempts", args.attempts)
    except KeyboardInterrupt:
        logger.info("interrupted by operator")
    finally:
        if top_cap is not None:
            try:
                top_cap.release()
            except Exception:
                pass
        if args.ui:
            try:
                import ui as _ui

                _ui.close()
            except Exception:
                pass
        logger.info("disconnecting (torque released)")
        robot.disconnect()


if __name__ == "__main__":
    main()
