"""The tube UI, served from INSIDE the mission server, driving the SO-101.

WHY IT LIVES IN THIS PROCESS AND NOT ITS OWN. The SO-101's serial bus and its OAK-D are
owned by the mission server's threads. Opening either from a second process is not a
configuration problem to be worked around -- it is how this rig breaks. The camera dies,
and the bus returns `[TxRxResult] Port is in use!` and then cycles "servo bus stopped
answering -- closed and reopened the port" forever, which is exactly the state this server
was found in before this module was written. So the tube server does not connect to
anything: it is handed the mission server's own accessors and runs on a second port in the
same process, sharing the one bus lock that already exists.

WHAT IT ADDS, AND WHAT IT DELIBERATELY DOES NOT REIMPLEMENT. It adds the tube-shaped view
of the world: a 3D scene with the arm and the tubes, rack holes as destinations, and the
shared episode/verification machinery from `rax.manipulation`. It does NOT add a second
pick. The SO-101's pick is `attempt_pick` in the mission server -- measured at 6/6 on the
green cube, with the radial creep, the grasp-check override and the retry loop all tuned
against this arm's real failures. Writing a second one to drive tubes instead of cubes
would be the exact duplication the rest of this work removed.

So a tube pick here is: point the open-vocabulary detector at tubes, let the existing pick
run, and verify the result the shared way.

THE TUBE PRIOR IS WHAT MAKES IT WORK AT ALL. Apparent-size ranging divides by the object's
expected size, and before `rax.perception.object_priors` learned about tubes, a tube fell
back to the 50.8 mm cube default and therefore reported itself roughly three times further
away than it was -- the same failure that makes the pen unpickable here. That prior is a
prerequisite for this module, not a nicety.

WHAT IS HONEST ABOUT THE RACK. Nobody has measured where a rack physically sits on this
bench. The racks' holes are ESTIMATED from the top camera (TOP_RACKS), and the UI labels them
`assumed`. Set it with POST /rack before asking for a place, or leave the destination as
"hold it" and the arm simply picks the tube up, which needs no such number.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
TUBE_UI = os.path.join(os.path.dirname(HERE), "tube_server", "ui")

#: Labels the open-vocabulary detector is pointed at for a tube run. Several synonyms
#: because YOLO-World is genuinely prompt-sensitive here and one phrasing is a coin flip.
TUBE_QUERY = "test tube, sample tube, vial"

#: Words that mark a mapped object as a tube. Matched loosely, because the detector
#: reports whichever synonym fired.
TUBE_WORDS = ("tube", "vial")

#: Where the rack is ASSUMED to be, in base-frame metres, and its hole grid. NOT MEASURED
#: -- see the module docstring. POST /rack {"x":..,"y":..} to correct it.
#: Square the jaws to the tube before closing. ON, and deliberately NOT read from
#: CFG.yaw_align, which is off because it measured 0/6 against 6/6 -- ON CUBES. The
#: geometry is not the same argument twice:
#:
#:   a 5cm cube fits the jaws at any yaw except near its diagonal, so rolling the wrist
#:   buys a little and costs the tilt that rolling introduces (the roll axis is the
#:   approach direction, so rolling tips the jaws out of horizontal).
#:
#:   a tube is 16mm across and 100mm long. The jaws MUST close across the short axis;
#:   closing along the long one means closing on nothing, or on the far edge. There is no
#:   "mostly square" for a cylinder.
#:
#: So a cube measurement does not transfer, and the flag that carries it should not
#: either. This one earns or loses its place on tube episodes.
TUBE_YAW_ALIGN = True

#: Most the wrist may roll in one pick, degrees. SMALL, on purpose.
#:
#: The first run rolled 70 degrees in one step, on an axis reading of "+0deg" from an
#: elongation of 2.3 -- while the tube was plainly lying at about 85. A big roll on a
#: measurement that weak is the worst of both: it swings the camera off the tube (the
#: lens rides past the roll joint, so the whole view turns with the hand) and it does so
#: on evidence that did not deserve to move the arm at all.
#:
#: A nudge is also all that is usually needed. The jaws are ~12cm apart in the image at
#: grasping range and a tube is 16mm across, so being a few degrees off square costs
#: almost nothing in effective opening; it is only the large misalignments that make a
#: cylinder unpickable, and those are better fixed over several picks than in one lunge.
#:
#: 20 was the old value and it made the stage decorative: a tube lying 80 degrees out
#: of square needed a 105-degree roll and got 20, so the jaws still closed along the
#: cylinder. The joint's own limits are the real bound here, and they are enforced
#: separately -- this only stops a bad angle measurement from spinning the wrist.
TUBE_MAX_ROLL_DEG = 100.0

#: Largest lateral correction the grid stage will make in ONE move, metres. The cap is
#: re-measured after every step, so a real 4cm error still closes -- in two passes rather
#: than one lunge. 2cm is a nudge at this scale and keeps a bad fix from becoming a dive.
TUBE_MAX_STEP_M = 0.02

#: WHERE THIS GRIPPER ACTUALLY STOPS ON AIR. Measured, by commanding it shut with
#: nothing between the jaws and reading it back five times: 6.5, 6.5, 6.5, 6.4.
#:
#: The profile says closed_pct = 2.0, and that is a COMMAND value, not an achieved one --
#: the jaws meet their own stop before the servo reaches the commanded position. Using it
#: as the air baseline put the holding floor at 7.0, half a unit above where empty jaws
#: actually rest, so a close on nothing sat one sample of noise away from reading as a
#: grasp. One run settled at exactly 6.4 and was called EMPTY by luck.
GRIP_AIR_PCT = 1.2
#: 2026-09-27: RE-MEASURED at 1.2 with nothing in the jaws, and a hand-held tube
#: stopped them at 6.0-8.2 -- inside the old "air" value, so a real grip was called
#: EMPTY and the jaws opened: the tube dropped on the spot twice. The operator's rule is
#: the one used now: the jaws STOPPED SHORT of their closed stop, or the current rose
#: while they were not fully closed, means something is between them.
GRIP_BLOCKED_PCT = 3.5
#: Above this is a tube. The gap is deliberately wide because the populations have NOT
#: been measured on this gripper the way the X250's were over 113 demonstrations
#: (holding > 31.9, air 30.2, no overlap). Every close logs its settled value, which is
#: what turns this threshold into a measurement.
GRIP_HOLDING_PCT = 10.5
#: ...AND BELOW THIS, because holding is a band and not a floor. A parallel gripper on a
#: 16mm tube comes to rest in a narrow range -- measured here at 13.8, 15.5 and 20.6 on
#: three good picks -- and it CANNOT stop high: there is nothing in the workspace thick
#: enough. So a close that settles at 43.4, 76.3 or 86.9 (all three logged, all three
#: called HOLDING by a one-sided test) did not close on a tube at all; the jaws stopped
#: early on a false contact and the squeeze fired while they were still wide open. That
#: is the gripper "opening instead of closing": it never got near shut.
GRIP_JAMMED_PCT = 36.0
#: Opening above which a current rise is the motor starting rather than the fingers
#: touching something. Passed into the close; see the note there.
GRIP_TRUST_BELOW_PCT = 60.0
#: How far the jaws are parted before the close. Enough to clear a 16mm tube and its cap
#: with room either side, and nowhere near the 95 they used to swing through.
GRIP_PREOPEN_PCT = 45.0


#: Degrees out of square worth turning the wrist for. Below this the roll costs more in
#: disturbance than it buys in alignment.
TWIST_TOL_DEG = 15.0

#: THE JAW LINE THE TWIST SQUARES AGAINST, image degrees mod 180. NOT jaw_frame().axis_deg.
#: That one comes from HAND_UV and a moving tip of (210,476) -- a point clipped at the
#: frame's bottom edge, measured for cubes at a wider opening -- and it reads 160. The two
#: fingertips are plainly visible in a live frame (2026-09-27) at about (154,382) and
#: (400,376): a line at 179. Squaring a +84deg tube against the 160 line asked for 70,
#: so the wrist rolled 14deg AWAY from square while the log reported "6deg out" -- it
#: was measuring its success against the same wrong line. Against 179 that tube was
#: already square. Re-measure if the camera or the jaws are remounted.
TWIST_JAW_AXIS_DEG = 179.0
#: The roll used to MEASURE which way the picture turns, before correcting.
TWIST_PROBE_DEG = 10.0
#: Elongation the silhouette needs at the GRASP, where the tube is foreshortened.
TWIST_MIN_ELONGATION = 1.8
#: How far three consecutive angle reads may disagree and still be acted on.
TWIST_MAX_SPREAD_DEG = 18.0

#: How far the second look, taken from above, may move the answer before it stops being
#: a refinement and starts being a different object. Mirrors mission_server's own
#: FIX_REFINE_MAX_M, for the same reason.
FIX_REFINE_MAX_M = 0.04

#: TOP CAMERA -> TABLE, ROUGH. Two fingertip positions seen in the 1280x720 top view
#: (CamSurv camera 0) on 2026-09-27: FK (19.2, +0.1)cm at px (555,200) and
#: (17.8, -13.5)cm at px (515,330). That gives ~9.6 px/cm with the robot's +x pointing
#: image-right and turned ~12deg, and -y pointing image-down. Read by eye off the
#: gripper body, so trust it to +-2cm and no better. (A 3-point re-fit on 2026-09-28
#: made the drops WORSE and was reverted on the operator's word: "it was good before".)
TOP_ORIGIN_PX = (557.0, 185.0)
TOP_ORIGIN_XY = (0.192, 0.001)
TOP_PX_PER_M = 960.0
TOP_EX = (0.977, 0.203)        # robot +x, as a unit vector in the image
TOP_EY_DOWN = (-0.203, 0.977)  # robot -y, as a unit vector in the image


def top_px_to_xy(u, v):
    """Top-camera pixel -> base-frame (x, y) metres, by the rough fit above."""
    du, dv = u - TOP_ORIGIN_PX[0], v - TOP_ORIGIN_PX[1]
    return (TOP_ORIGIN_XY[0] + (du * TOP_EX[0] + dv * TOP_EX[1]) / TOP_PX_PER_M,
            TOP_ORIGIN_XY[1] - (du * TOP_EY_DOWN[0] + dv * TOP_EY_DOWN[1]) / TOP_PX_PER_M)


def top_px_delta_to_xy(du, dv):
    """A pixel DISPLACEMENT in the top view -> the base-frame displacement, metres."""
    return ((du * TOP_EX[0] + dv * TOP_EX[1]) / TOP_PX_PER_M,
            -(du * TOP_EY_DOWN[0] + dv * TOP_EY_DOWN[1]) / TOP_PX_PER_M)


def xy_to_top_px(x, y):
    """Base-frame (x, y) metres -> top-camera pixel, the inverse of top_px_to_xy."""
    dx, dy = (x - TOP_ORIGIN_XY[0]) * TOP_PX_PER_M, -(y - TOP_ORIGIN_XY[1]) * TOP_PX_PER_M
    return (TOP_ORIGIN_PX[0] + dx * TOP_EX[0] + dy * TOP_EY_DOWN[0],
            TOP_ORIGIN_PX[1] + dx * TOP_EX[1] + dy * TOP_EY_DOWN[1])


#: The racks as seen in the top view: each hole's PIXEL, read off the frame by eye. The
#: base-frame positions are derived through top_px_to_xy, so correcting the fit moves
#: every hole with it. ESTIMATES -- nothing has been dropped into one yet.
TOP_RACKS = [
    # Re-read after the racks were moved (2026-09-27 ~19:50): Hough on the top view,
    # top-face holes only -- the side-wall slots and shadows it also finds are dropped.
    {"name": "black rack (est)", "colour": "#d0a040",
     "holes_px": [(541, 378), (564, 373), (581, 375), (553, 388), (566, 399),
                  (542, 403), (580, 410), (556, 415), (569, 426), (545, 427),
                  (582, 438), (559, 442), (572, 454), (586, 465), (561, 469)]},
    {"name": "silver rack (est)", "colour": "#9fb4c8",
     "holes_px": [(646, 358), (669, 355), (691, 352), (660, 368), (683, 370),
                  (650, 385), (672, 382), (696, 380), (663, 397), (685, 391),
                  (654, 413), (677, 408), (700, 404), (667, 423), (691, 419),
                  (656, 439), (681, 435), (704, 433), (671, 450), (695, 447)]},
]

#: The left jaw's corner of the wrist view (x < this, y > that) is masked from the
#: detector: the jaw kept reading as a blue cap.
JAW_CORNER_X, JAW_CORNER_Y = 200.0, 340.0
#: Fingertip height where the hand is squared and the wrist twisted -- clear of a lying
#: tube (1.6cm) -- before the straight, fixed-angle descent to the grasp.
TWIST_Z = 0.04
#: The twist is small by design (the operator: "you don't need to twist much"): at most
#: this many degrees either way, and none at all within TWIST_TOL_DEG of square.
TWIST_MAX_DEG = 45.0
#: ...and only this fraction of the measured error (the operator: "do half of the angle").
TWIST_FRACTION = 0.5
#: Beyond this the tube reads nearly perpendicular and the direction is ambiguous; the
#: wrist then always turns NEGATIVE -- what the operator saw work (2026-09-28).
TWIST_AMBIGUOUS_DEG = 70.0
#: The body must stand out this much (grey levels) from either side to be read.
BODY_MIN_CONTRAST = 18.0
#: After a roll, the tube must have turned in the view to within this of the expectation.
TWIST_CHECK_TOL_DEG = 25.0
#: THE BOX: before closing, the cap must sit in the yellow jaw cell left of the fingers.
BOX_ABOVE_M = 0.015          # align this far above the grasp height (tips clear of the tube)
BOX_MARGIN_PX = 12.0         # how far outside the cell still counts
BOX_PROBE_PAN_DEG = 3.0      # test moves that measure how the cap moves, here
BOX_PROBE_R_M = 0.01
BOX_STEPS = 6
BOX_GAIN = 0.8
BOX_MAX_PAN_DEG = 4.0
BOX_MAX_R_M = 0.015
#: Every sideways move near the tube is made in the air: up this much, across, down.
HOP_UP_M = 0.03
#: The base gain the probe measures on this rig (px of cap per degree of base), used when
#: the probe itself loses the cap. Measured -5.05, -6.26, -6.55, -7.84, -9.81, -10.44.
BASE_GAIN_DEFAULT_PX_PER_DEG = -7.5
#: THE GUARDRAIL: while picking, the base never faces further right than this (deg,
#: + = left). The racks sit at about -25 to -45deg.
PICK_MIN_BEARING_DEG = -15.0
#: Bearings (deg, robot frame, + = left) the one wrist scan stops at.
SCAN_BEARINGS_DEG = (40.0, 20.0, 0.0)   # not toward the racks (right, ~-25..-45deg): their
                                         # place is known, the scan has no business there
#: At each scan bearing the wrist also tilts down this much, to see near the base.
SCAN_TILTS_DEG = (0.0, 25.0)
#: A mapped cap this close to a rack hole is in the rack, not on the mat.
RACK_EXCLUDE_M = 0.04
#: Cap movement against the grasp point, px, on the way down, that means "pushed".
PUSHED_PX = 45.0
MAX_HOPS = 0   # OFF: with several caps of one colour it jumped to another cap (366px)
               # and hopped on a tube that had not moved
#: Jaws settling this wide mean two tubes (one settles 6-10.5 on this gripper).
GRIP_TWO_PCT = 16.0
#: Caps this close to the jaw centre after the lift are in the jaws.
TWO_CAPS_PX = 170.0
#: ...and this big: a cap IN the jaws is right under the lens (~130x140px); tubes still
#: lying on the mat, 9cm further down, look a fraction of that.
TWO_CAPS_MIN_AREA = 2500.0
#: Pace of the tube mode's own moves: bigger joint steps, shorter settles.
SPEED_SCALE = 1.35
SETTLE_SCALE = 0.75
#: Smallest cap the mat scan believes, px (real ones measured 130-305; false 21-108).
TOP_SCAN_MIN_AREA = 115
#: A cap this close to the target hole, from above with the arm at home, means it went in.
#: The cap stands ~9cm above the hole, so the parallax is inside this.
HOLE_VERIFY_PX = 40.0


def top_body_angle(img, cap, gates):
    """Image angle (deg) from a cap toward its tube's body, in the top view, or None.

    Two measurements, each covering the other's failure. The CAP is a short cylinder
    and from above it is elongated along the tube, which gives the axis but not which
    end the body is on; the BODY is the bright streak on the dark mat, which gives the
    side but, measured alone, locks onto a neighbouring tube or a QR code. So: the axis
    from the cap's own blob, the side from brightness along that axis. 7 of 7 right on
    the live frame, where a generic silhouette fit got 1 of 7 (mat edges, QR codes).
    """
    from rax.perception.tube_caps import CAP_HSV
    x0, y0, x1, y1 = [int(v) for v in cap.bbox]
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    lo, hi = CAP_HSV[cap.colour]
    smin, vmin = gates[cap.colour][:2]
    ys, xs = np.nonzero((hsv[:, :, 0] >= lo) & (hsv[:, :, 0] <= hi)
                        & (hsv[:, :, 1] >= smin) & (hsv[:, :, 2] >= vmin))
    if len(xs) < 8:
        return None
    (_, _), (w, h), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
    cap_ax = (ang if w >= h else ang + 90.0) % 180.0
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    H, W = g.shape

    def ray(a):
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        v = [min(float(g[int(cap.y + sa * r), int(cap.x + ca * r)]), 170.0)
             for r in range(12, 60, 2)
             if 0 <= int(cap.y + sa * r) < H and 0 <= int(cap.x + ca * r) < W]
        return float(np.mean(v)) if v else 0.0

    cands = [a for a in range(0, 360, 4)
             if abs(((a - cap_ax) + 90.0) % 180.0 - 90.0) <= 35.0]
    return float(max(cands, key=ray))


def in_rack_zone(x, y, margin=None):
    """Is base-frame (x, y) on or within RACK_EXCLUDE_M of a rack's hole grid?"""
    m = RACK_EXCLUDE_M if margin is None else margin
    for r in TOP_RACKS:
        hs = [top_px_to_xy(u, v) for u, v in r["holes_px"]]
        xs, ys = [h[0] for h in hs], [h[1] for h in hs]
        if min(xs) - m <= x <= max(xs) + m and min(ys) - m <= y <= max(ys) + m:
            return True
    return False


#: Which rack each cap colour goes to -- the operator's rule, 2026-09-27.
RACK_FOR_COLOUR = {"blue": "black rack (est)", "green": "black rack (est)",
                   "red": "silver rack (est)"}

# ---- the UPRIGHT drop: grab looking down, stand the tube up, drop it in a hole --------
#
# Held from above, the tube lies along the gripper's y axis (across the jaws, across the
# approach). With the hand LEVEL (pitch 0) and the wrist at roll +-90, that axis is
# vertical -- checked on the arm's own model, both signs, before anything moved.
#: Wrist roll that stands the tube up; the other end up is -STAND_ROLL_DEG.
STAND_ROLL_DEG = 90.0
#: Top of the racks above the table. NOT MEASURED -- a guess from the photos.
RACK_TOP_Z = 0.055
#: How much tube hangs below the fingertips once it stands. The grasp is taken near the
#: cap, so nearly the whole 100mm tube is below the jaws. An estimate.
TUBE_BELOW_TIP_M = 0.085
#: Clearance of the tube's bottom over the rack top while lining up, and at the release.
HOVER_CLEAR_M = 0.05
DROP_CLEAR_M = 0.01     # lower, on request: the tube's bottom ~1cm over the rack
#: Extra height per cap colour, on top of both -- the operator asked for gold higher.
#: Gold goes to the silver rack, whose far holes cap the fingertip at ~22cm when level.
EXTRA_Z_BY_COLOUR = {"red": 0.06}     # red took gold's place (and its rack)
#: Where the tube is stood up, on the grasp's bearing. Checked on the arm's model: level
#: at either roll solves here with no residual; 20cm out does not (the shoulder hits its
#: -100 limit), and looking straight DOWN nothing this high solves at all -- which is why
#: the stand-up goes straight from the lift to level instead of rising first.
STAND_R_M, STAND_Z_M = 0.26, 0.16
#: Fingertip height for the swing to the rack (the tube hangs ~8.5cm below it).
CARRY_Z_M = 0.25
#: The table in the top view (x0, y0, x1, y1), and the cap gates for that camera.
TOP_TABLE_ROI = (370, 40, 1040, 600)
TOP_CAP_GATES = {"green": (90, 70), "blue": (90, 70), "gold": (55, 90), "red": (100, 60)}
#: Those gates are loose enough that the silver rack's holes read blue and the arm's
#: joints read gold. So a held cap is only believed within this many px of where the
#: fingertip's own position says it is (the height parallax is inside this too).
TOP_HELD_SEARCH_PX = 70.0
#: Top-camera alignment of the held cap over the hole.
TOP_ALIGN_TOL_PX = 6.0
TOP_ALIGN_STEPS = 5
TOP_ALIGN_GAIN = 0.7
TOP_ALIGN_MAX_M = 0.04       # total, a bound on a mis-detection

#: A lab tube's nominal shape, from `rax.perception.object_priors`: 16 mm across,
#: 100 mm long.
TUBE_D_M, TUBE_L_M = 0.016, 0.100

# The shape-based orientation classifier that used to live here is gone with the map it
# read from. It compared a mapped object's w/d/h against a standing tube and a lying one
# and returned None when it matched neither -- which was the right shape of answer, and
# the reason it existed was that the naive height test called a tube "upright" while the
# camera plainly showed it lying.
#
# THE PRINCIPLE SURVIVES, in `tubes()`, on better evidence. Orientation now comes from
# the tube's own silhouette rather than from three numbers fused over a base sweep, and
# it still has three answers: an elongated body means LYING at a measured angle, and no
# elongated body means UNKNOWN -- a tube standing in a rack and one pointing end-on at
# the camera look the same from here, and nothing available can separate them.



#: Two fixes of the same tube closer than this are the same tube. A tube is 16mm across
#: and the cast's own scatter is a couple of centimetres, so this has to be bigger than
#: the noise and smaller than the gap between two tubes an operator would set out.
MAP_MERGE_M = 0.045
#: A mapped tube nobody has seen for this long is stale, but NOT deleted -- it is still
#: roughly where it was. It is reported as "mapped" rather than "seen" so the UI can say
#: which, and so nothing downstream mistakes an old fix for a fresh one.
MAP_FRESH_S = 3.0


def start(ms, port: int = 8486) -> None:
    """Serve the tube UI on ``port``, backed by mission-server module ``ms``."""
    from rax.manipulation.episodes import EpisodeLog
    from rax.manipulation.grip import CurrentRise, reconcile, settled
    from rax.perception.tube_caps import draw as draw_caps
    from rax.perception.tube_caps import find_caps, fold, tube_axis
    from rax.robots.urdf_visuals import link_visuals

    # NO YOLO BOXES IN THIS MODE. Tubes are found by cap colour here; the
    # open-vocabulary detector's boxes are a different detector's opinion drawn over
    # the same picture, and they land on the tube the arm is aiming at.
    try:
        ms.DRAW_YOLO_BOXES[0] = False
    except Exception:
        pass

    app = Flask("tube_mode", static_folder=None)
    # THE MAP. Tubes persist here once seen, so the arm knows roughly where they are
    # after it has moved or backed off -- which the "only what is in frame right now"
    # version could not: a tube left the map the moment the camera looked elsewhere, so
    # a pick-all could never find the second tube after approaching the first.
    #
    # Each entry is a running position, not a snapshot, so repeated looks average out the
    # cast's scatter instead of the last one overwriting everything before it.
    tube_map = {}          # id -> {id, colour, x, y, n, t, angle, yaw_known, picked}
    next_id = [1]
    lock = threading.Lock()
    tstate = {"phase": "IDLE", "note": "ready", "running": False, "log": [],
              "idle_current": 0.0}
    #: ONE TUBE AT A TIME. While a pick runs, the overlays draw only this tube: its box,
    #: its path from the jaws and its direction -- not every tube in view. `uv` is kept
    #: up to date by the overlay itself (it tracks the nearest cap of the colour).
    focus = {"colour": None, "uv": None, "xy": None}
    mapping = [False]      # True only while the wrist scan is sweeping
    pick_guard = [False]   # True while picking: the base may not face the racks
    episodes = EpisodeLog(
        os.path.join(HERE, "tube_episodes.jsonl"),
        probe=lambda: {"joints": [round(float(v), 1) for v in ms.observe(False)[0]]},
        note=lambda m: tsay(m))

    # ---- caps, found every frame and drawn on the FPV overlay -------------------
    # THE BOXES THE OPERATOR ASKED FOR, and they are not only decoration: the same
    # detection is what the grid aim steers on. Registered as a hook so mission_server
    # never imports this module -- the dependency points one way.
    def _map_observe():
        """Fold the caps in view into the map. Called wherever the arm is looking.

        ONLY DURING THE SCAN. Every wrist frame used to fold its caps in, from every
        pose the arm passed through, false gold ones included -- one run's map grew to
        19 "tubes", some at the robot's own base, while the mat held one. The map is
        built once, by the sweep, and then left alone while the arm works through it.
        """
        if not mapping[0]:
            return
        caps, t = ms.LAST_CAPS[0], ms.LAST_CAPS[1]
        if not caps or time.time() - t > 2.0:
            return
        try:
            T = ms.T_cam_of(ms.observe(False)[0])
        except Exception:
            return
        for c in caps:
            try:
                pt = ms.ray_to_table((c["x"], c["y"]), T)
            except Exception:
                continue
            if pt is None:
                continue
            x, y = float(pt[0]), float(pt[1])
            if not (ms.ARM.reach_min_m <= math.hypot(x, y) <= ms.ARM.reach_max_m):
                continue
            if in_rack_zone(x, y):
                continue                      # a tube already in a rack, not one to pick
            with lock:
                hit = None
                for e in tube_map.values():
                    if e["colour"] != c["colour"] or e.get("picked"):
                        continue
                    if math.hypot(e["x"] - x, e["y"] - y) <= MAP_MERGE_M:
                        hit = e
                        break
                if hit is None:
                    tube_map[next_id[0]] = {
                        "id": next_id[0], "colour": c["colour"], "x": x, "y": y,
                        "n": 1, "t": time.time(), "angle": c.get("angle"),
                        "yaw_known": bool(c.get("confident")), "picked": False}
                    next_id[0] += 1
                else:
                    # RUNNING MEAN, not replacement: every look is a noisy cast and the
                    # last one is not better than the average of the ones before it.
                    k = hit["n"] + 1
                    hit["x"] += (x - hit["x"]) / k
                    hit["y"] += (y - hit["y"]) / k
                    hit["n"], hit["t"] = k, time.time()
                    if c.get("confident"):
                        hit["angle"], hit["yaw_known"] = c.get("angle"), True

    def cap_overlay(img):
        # DETECT ON THE CLEAN FRAME, DRAW ON THE ANNOTATED ONE. The hooks run at the end
        # of publish(), by which point the grid, the jaw cells and the hand-eye marker
        # have all been drawn into `img` -- dark lines straight across the tube bodies.
        # The cap survives that (it is found by colour) but the AXIS does not: tube_axis
        # segments by contrast, and a grid line through the body splits it into pieces
        # that are no longer elongated. Both tubes reported "orientation unknown" from a
        # view where both bodies were plainly visible, and this was why.
        src = ms.latest_rgb[0]
        clean = img if src is None else cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2BGR)
        # NO FINGERTIP EXCLUSION ON THIS ARM, and that is a measurement not an
        # oversight. The exclusion exists because the X250's jaws wear blue tape that
        # reads as a blue cap. Here the gripper sits at V ~ 44 and the detector's
        # measured gate is V >= 85, so it is already rejected on its own merits -- while
        # the exclusion blinds the detector in a 70px disc around the fingertip, which is
        # exactly where the cap sits once the hand is over it. The descent kept reporting
        # "cap lost" at 13cm for this reason.
        # THE LEFT JAW'S CORNER IS NOT A CAP. Parts of that jaw read blue and passed the
        # detector's gates more than once (a "blue cap" sitting on the gripper). A real
        # cap at the grasp sits between the jaws, well right of this corner.
        caps = [c for c in find_caps(clean)
                if not (c.x < JAW_CORNER_X and c.y > JAW_CORNER_Y)]
        # NOTHING ON OR BY A RACK IS A TUBE TO PICK. The arm went for a "gold cap" beside
        # the black rack -- a stain -- and tubes already racked read as caps too. The
        # racks' place is known; any cap whose table cast lands there is dropped, so the
        # pick never steers toward a rack. (The drop does not use these detections.)
        try:
            T_now = ms.T_cam_of(ms.observe(False)[0])
            keep = []
            for c in caps:
                pt = ms.ray_to_table((c.x, c.y), T_now)
                if pt is not None and in_rack_zone(float(pt[0]), float(pt[1])):
                    continue
                keep.append(c)
            caps = keep
        except Exception:
            pass
        found = []
        for c in caps:
            # The window scales with the cap: a tube is about six cap-diameters
            # long, so a fixed radius that fits at 25cm crops the body at 15cm.
            ax = tube_axis(clean, (c.x, c.y),
                           r=int(max(90, min(220, 3.2 * max(c.w, c.h)))))
            found.append({"colour": c.colour, "x": c.x, "y": c.y, "area": c.area,
                          "bbox": list(c.bbox),
                          "angle": None if ax is None else ax.angle_deg,
                          "elong": None if ax is None else ax.elongation,
                          "axis_len": None if ax is None else ax.length,
                          # The BODY's centroid, not the cap's. A tube extends to one
                          # side of its cap, so a line centred on the cap runs half its
                          # length into empty space above the tube and reads as pointing
                          # somewhere it is not.
                          "axis_cx": None if ax is None else ax.centre[0],
                          "axis_cy": None if ax is None else ax.centre[1],
                          "confident": bool(ax is not None and ax.is_confident)})
        ms.LAST_CAPS[0] = found
        ms.LAST_CAPS[1] = time.time()
        try:
            _map_observe()
        except Exception:
            pass
        if focus["colour"] is not None and found:
            mine = [f for f in found if f["colour"] == focus["colour"]]
            if mine:
                ref = focus["uv"]
                tgt = (min(mine, key=lambda f: math.hypot(f["x"] - ref[0], f["y"] - ref[1]))
                       if ref is not None else max(mine, key=lambda f: f["area"]))
                focus["uv"] = (tgt["x"], tgt["y"])
                found = [tgt]
                caps = [c for c in caps if abs(c.x - tgt["x"]) < 1 and abs(c.y - tgt["y"]) < 1]
            else:
                found, caps = [], []
        if caps:
            img[:, :] = draw_caps(img, caps)
            for f in found:
                if f["angle"] is None:
                    continue
                a = math.radians(f["angle"])
                half = f["axis_len"] / 2.0
                col = {"green": (60, 220, 90), "red": (60, 60, 230)}.get(f["colour"], (235, 170, 60))
                ax_cx = f["axis_cx"] if f["axis_cx"] is not None else f["x"]
                ax_cy = f["axis_cy"] if f["axis_cy"] is not None else f["y"]
                p0 = (int(ax_cx - half * math.cos(a)), int(ax_cy - half * math.sin(a)))
                p1 = (int(ax_cx + half * math.cos(a)), int(ax_cy + half * math.sin(a)))
                cv2.line(img, p0, p1, col, 2)
                cv2.putText(img, f"{f['angle']:+.0f}d", (int(f["x"]) + 10, int(f["y"]) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
            # THE LINE FROM THE CAP TO THE GRIP CENTRE, which is what pick.py's own UI
            # draws and what its comment calls "the grid the operator reasons in: cap in
            # a cell, jaws in a cell, make them match". It is a DISPLAY, not a control
            # input -- only its HORIZONTAL component steers anything, because the
            # vertical is the one the camera cannot decide (tilting the wrist moves the
            # cap up the frame while the arm is physically closing in). Drawing it makes
            # the remaining error visible instead of only logged.
            try:
                jc = ms.jaw_frame().centre_uv
                jcx, jcy = int(jc[0]), int(jc[1])
                for f in found:
                    cx, cy = int(f["x"]), int(f["y"])
                    col = {"green": (60, 220, 90), "red": (60, 60, 230)}.get(f["colour"], (235, 170, 60))
                    cv2.line(img, (cx, cy), (jcx, jcy), col, 1, cv2.LINE_AA)
                    # the horizontal part is the bit that actually steers: draw it solid
                    cv2.line(img, (cx, cy), (jcx, cy), (0, 255, 255), 2)
                    cv2.putText(img, f"dx {jcx - cx:+d}", ((cx + jcx) // 2 - 22, cy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                                cv2.LINE_AA)
                cv2.drawMarker(img, (jcx, jcy), (255, 120, 255), cv2.MARKER_TILTED_CROSS,
                               16, 2)
            except Exception:
                pass

    def tsay(msg):
        ms.say(msg)
        with lock:
            tstate["log"].append({"t": time.strftime("%H:%M:%S"), "m": str(msg)[:220]})
            del tstate["log"][:-200]

    def tphase(name, note=""):
        with lock:
            tstate["phase"], tstate["note"] = name, note
        tsay(f"[{name}] {note}" if note else f"[{name}]")

    # ---- gripper ---------------------------------------------------------------
    def sample_idle():
        """Measure the gripper's idle draw. It drifts, so it is measured, never assumed."""
        vals = [c for c in (ms.gripper_current() for _ in range(5)) if c is not None]
        if vals:
            with lock:
                tstate["idle_current"] = float(np.mean(vals))
        return tstate["idle_current"]

    def grip_verdict():
        """The shared verdict, plus the server's own carry flag.

        Two sources on purpose. The CurrentRise reading is the raw sensor; `carry["held"]`
        is the mission server's considered state, which already has the vision model's
        one-way override applied to it (see _log_grasp_verdict there). When they disagree
        the carry flag wins, because it is the one that has heard from the camera.
        """
        cur = ms.gripper_current()
        with lock:
            idle = tstate["idle_current"]
        with ms.lock:
            carried = bool(ms.carry["held"])
        if cur is None:
            return carried, "gripper current unreadable; using the server's carry flag"
        d = reconcile(CurrentRise(idle=idle,
                                  delta=ms.ARM.gripper.contact_current_delta).verdict(cur))
        if d.held != carried:
            return carried, (f"{d.detail}; the server's carry flag says "
                             f"{'HELD' if carried else 'EMPTY'} and wins — it has the "
                             f"camera's verdict applied")
        return carried, d.detail

    # ---- the map ---------------------------------------------------------------
    def tubes():
        """Every tube the map knows about, seen or remembered.

        NOT ONLY WHAT IS IN FRAME. The first version listed the current view and nothing
        else, so a tube vanished from the map the moment the camera looked away -- which
        makes picking several impossible, because approaching the first one loses the
        second. Entries persist, carry how many looks went into them, and say whether
        they were seen recently or are being remembered.
        """
        now = time.time()
        with lock:
            out = []
            for e in sorted(tube_map.values(), key=lambda v: v["id"]):
                if e.get("picked"):
                    continue
                out.append({
                    "id": e["id"], "colour": e["colour"],
                    "x": round(e["x"], 4), "y": round(e["y"], 4), "z": 0.0,
                    "held": False, "rack": None, "hole": None,
                    "source": "seen" if now - e["t"] < MAP_FRESH_S else "mapped",
                    "d": TUBE_D_M, "l": TUBE_L_M,
                    "standing": False if e["yaw_known"] else None,
                    "yaw": (math.degrees(math.atan2(e["top_dir"][1], e["top_dir"][0]))
                            if e.get("top_dir") is not None else float(e["angle"] or 0.0)),
                    "yaw_known": bool(e["yaw_known"] or e.get("top_dir") is not None),
                    "n": e["n"], "age": round(now - e["t"], 1),
                    "focus": bool(focus["xy"] is not None and e["colour"] == focus["colour"]
                                  and math.hypot(e["x"] - focus["xy"][0],
                                                 e["y"] - focus["xy"][1]) < 0.03),
                    "label": f"{e['colour']} tube"})
        return out

    def _colour_of(label):
        for c in ("green", "blue", "red", "gold", "yellow", "orange"):
            if c in label.lower():
                return "gold" if c in ("yellow", "orange") else c
        return "blue"

    # ---- the top camera ---------------------------------------------------------
    def _top_caps(colour):
        """Caps of this colour in the top view, as [(u, v)], or None if no frame.

        ON THE TABLE ONLY, with gates of its own. Unmasked, gold returned eighteen blobs
        -- the wooden floor around the table -- and missed the real gold cap, which from
        up here reads S 76-98 against the table's 14-30.
        """
        img = ms._overhead_frame()
        if img is None:
            return None
        x0, y0, x1, y1 = TOP_TABLE_ROI
        roi = img[y0:y1, x0:x1]
        return [(c.x + x0, c.y + y0)
                for c in find_caps(roi, colours=(colour,), min_area=12, max_area=900,
                                   gates=TOP_CAP_GATES)]

    def _go(q, settle=0.25, step=2.0):
        """goto_smooth at the tube mode's pace: bigger steps, shorter settles.

        THE GUARDRAIL lives here, because every pick move goes through here: while a
        pick is running the base may not face further right than PICK_MIN_BEARING_DEG.
        The racks are over there; picking has no business turning toward them, whatever
        the camera thinks it sees. Only the drop, which runs with the guard off, goes.
        """
        q = np.asarray(q, float).copy()
        if pick_guard[0]:
            tp = ms._tip(q)
            bear = math.degrees(math.atan2(tp[1], tp[0]))
            if bear < PICK_MIN_BEARING_DEG:
                q[ms.ARM.pan_joint] = _pan_for_bearing(q, math.radians(PICK_MIN_BEARING_DEG))
                tsay(f"        guardrail: {bear:+.0f}deg is toward the racks — held at "
                     f"{PICK_MIN_BEARING_DEG:+.0f}deg")
        fast = not pick_guard[0]          # the pick keeps its own, proven pace
        ms.goto_smooth(ms._clamp_joints(q),
                       settle=settle * (SETTLE_SCALE if fast else 1.0),
                       step=step * (SPEED_SCALE if fast else 1.0))

    def _go_ik(p, pitch, roll, what, step=1.6, settle=0.25, tol=0.01):
        """Put the fingertip at ``p`` holding ``pitch`` and ``roll``, or raise."""
        q_t, e = ms._ik_hold_pitch(ms.observe(False)[0].astype(float),
                                   np.asarray(p, float), pitch, roll, ret_err=True)
        if e > tol:
            raise RuntimeError(f"cannot reach {what} at ({p[0]*100:+.1f},{p[1]*100:+.1f},"
                               f"{p[2]*100:+.1f})cm, pitch {pitch:.0f} "
                               f"(IK residual {e*100:.1f}cm)")
        _go(q_t, settle=settle, step=step)
        return q_t

    # ---- the upright drop -------------------------------------------------------
    def place_upright(colour, rack_name=None):
        """Stand the held tube up and drop it into a free hole. Returns (ok, tag, detail).

        The operator's sequence: grabbed looking straight down; come up; turn the hand
        to look forward (level) so the tube stands; carry high; hover over the hole;
        adjust by the top camera; go down; open. Then look from above, with the arm out
        of the way, and say whether the hole now holds a cap -- that is the TAG.
        """
        rack = next(r for r in TOP_RACKS
                    if r["name"] == (rack_name or RACK_FOR_COLOUR[colour]))
        used = tstate.setdefault("used_top_holes", {}).setdefault(rack["name"], [])

        # ---- from the LIFT (already 9cm up, still looking down) ------------------
        tphase("STAND", f"standing the {colour} tube up")
        tip0 = ms._tip(ms.observe(False)[0].astype(float))
        bear0 = math.atan2(tip0[1], tip0[0])
        stand = np.array([STAND_R_M * math.cos(bear0), STAND_R_M * math.sin(bear0),
                          STAND_Z_M])
        before = _top_caps(colour) or []

        # ---- LOOK FORWARD: the hand level, the tube vertical ----------------------
        # THE WRIST NEVER TURNS OVER. Either +-90 stands the tube up; take the one
        # nearest where the wrist already is, so ID 5 moves a few degrees, not 180.
        j5_now = float(ms.observe(False)[0][ms.ARM.roll_joint])
        roll = STAND_ROLL_DEG if abs(j5_now - STAND_ROLL_DEG) <= abs(
            j5_now + STAND_ROLL_DEG) else -STAND_ROLL_DEG
        _go_ik(stand, 0.0, roll, "the level hand", step=2.4, settle=0.25)

        def _held_from(after):
            """The held cap: new since `before`, and near where the fingertip is."""
            if not after:
                return None
            tp = ms._tip(ms.observe(False)[0].astype(float))
            pu, pv = xy_to_top_px(float(tp[0]), float(tp[1]))
            new = [c for c in after
                   if all(math.hypot(c[0] - b[0], c[1] - b[1]) > 8.0 for b in before)
                   and math.hypot(c[0] - pu, c[1] - pv) <= TOP_HELD_SEARCH_PX]
            return min(new, key=lambda c: math.hypot(c[0] - pu, c[1] - pv), default=None)

        g_now = float(ms.state.get("gripper") or 0.0)
        if g_now <= GRIP_BLOCKED_PCT:
            raise RuntimeError(f"dropped the tube while standing it up (jaws at {g_now:.1f})")
        after = _top_caps(colour)
        held = _held_from(after)
        if held is None:
            tsay("        cannot see the held cap from the top camera — carrying on "
                 "without the top-camera alignment")
        else:
            tsay(f"        cap up at top-camera ({held[0]:.0f},{held[1]:.0f})px, "
                 f"wrist roll {roll:+.0f}")
        static = [c for c in (after or []) if held is None
                  or math.hypot(c[0] - held[0], c[1] - held[1]) > 8.0]

        # ---- CHOOSE A HOLE: free by bookkeeping and by the top camera -------------
        cands = []
        for k, (u, v) in enumerate(rack["holes_px"]):
            if k in used:
                continue
            if any(math.hypot(u - s_[0], v - s_[1]) < 10.0 for s_ in static):
                continue
            x, y = top_px_to_xy(u, v)
            # only holes the level hand can actually get down to
            _q, e_rel = ms._ik_hold_pitch(ms.observe(False)[0].astype(float),
                                          np.array([x, y, RACK_TOP_Z + TUBE_BELOW_TIP_M
                                                    + DROP_CLEAR_M]), 0.0, roll, ret_err=True)
            if e_rel > 0.01:
                continue
            cands.append((math.hypot(x, y), k, (u, v), (x, y)))
        if not cands:
            raise RuntimeError(f"no free hole left in the {rack['name']}")
        _r, k, hole_px, (hx, hy) = min(cands)
        tsay(f"        target: {rack['name']} hole {k} at ({hx*100:+.1f},{hy*100:+.1f})cm")

        # ---- HOVER HEIGHT: as high as asked, or as high as this hole allows --------
        # Far holes run out of arm before near ones (the silver rack's far corner tops
        # out ~22cm, level): step the extra height down 1cm at a time rather than
        # refuse the hole.
        tphase("CARRY", f"over {rack['name']} hole {k}")
        z_extra = EXTRA_Z_BY_COLOUR.get(colour, 0.0)
        z_base = RACK_TOP_Z + TUBE_BELOW_TIP_M
        while True:
            z_hover = z_base + HOVER_CLEAR_M + z_extra
            q_h, e_h = ms._ik_hold_pitch(ms.observe(False)[0].astype(float),
                                         np.array([hx, hy, z_hover]), 0.0, roll,
                                         ret_err=True)
            if e_h <= 0.01 or z_extra <= 0.0:
                break
            z_extra = max(0.0, z_extra - 0.01)
        if e_h > 0.01:
            raise RuntimeError(f"cannot hover over hole {k} (IK residual {e_h*100:.1f}cm)")
        tsay(f"        hover at {z_hover*100:.0f}cm, release at "
             f"{(z_base + DROP_CLEAR_M)*100:.0f}cm (fingertip height)")

        # ---- HIGH OVER THE RACK, THEN DOWN ----------------------------------------
        # Rise at the stand point, swing, come over the hole at that height, and only
        # then descend -- so the hanging tube passes over the rack instead of into it.
        z_carry = max(CARRY_Z_M, z_hover)
        tip_s = ms._tip(ms.observe(False)[0].astype(float))
        _go_ik(np.array([tip_s[0], tip_s[1], z_carry]), 0.0, roll, "the carry height",
               step=3.0, settle=0.1, tol=0.02)
        q_c, e_c = ms._ik_hold_pitch(ms.observe(False)[0].astype(float),
                                     np.array([hx, hy, z_carry]), 0.0, roll, ret_err=True)
        q_turn = ms.observe(False)[0].astype(float)
        q_turn[ms.ARM.pan_joint] = (q_c if e_c <= 0.02 else q_h)[ms.ARM.pan_joint]
        _go(q_turn, settle=0.1, step=3.0)
        if e_c <= 0.02:
            _go(q_c, settle=0.1, step=3.0)
        _go(q_h, settle=0.25, step=2.4)
        aim = np.array([hx, hy])

        # ---- ADJUST by the top camera: the held cap onto the hole -----------------
        if held is not None:
            tphase("ALIGN", "lining the tube up over the hole from the top camera")
            travel = 0.0
            for s_i in range(TOP_ALIGN_STEPS):
                ms.checkpoint()
                caps = _top_caps(colour) or []
                tp = ms._tip(ms.observe(False)[0].astype(float))
                pu, pv = xy_to_top_px(float(tp[0]), float(tp[1]))
                mine = [c for c in caps
                        if all(math.hypot(c[0] - t_[0], c[1] - t_[1]) > 8.0 for t_ in static)
                        and math.hypot(c[0] - pu, c[1] - pv) <= TOP_HELD_SEARCH_PX]
                if not mine:
                    tsay(f"        align {s_i+1}: lost the cap in the top view — back to "
                         "the hole's own position rather than trust the last nudge")
                    aim = np.array([hx, hy])
                    _go_ik(np.array([hx, hy, z_hover]), 0.0, roll, "the hole",
                           step=1.6, settle=0.3)
                    break
                cap = min(mine, key=lambda c: math.hypot(c[0] - pu, c[1] - pv))
                du, dv = hole_px[0] - cap[0], hole_px[1] - cap[1]
                err = math.hypot(du, dv)
                tsay(f"        align {s_i+1}: cap ({cap[0]:.0f},{cap[1]:.0f}) hole "
                     f"({hole_px[0]:.0f},{hole_px[1]:.0f}) -> {err:.0f}px")
                if err <= TOP_ALIGN_TOL_PX:
                    break
                d = np.array(top_px_delta_to_xy(du, dv))
                d *= TOP_ALIGN_GAIN
                if travel + float(np.linalg.norm(d)) > TOP_ALIGN_MAX_M:
                    tsay(f"        align: would move over {TOP_ALIGN_MAX_M*100:.0f}cm in "
                         "total — stopping (a mis-detection, not a correction)")
                    break
                travel += float(np.linalg.norm(d))
                aim = aim + d
                _go_ik(np.array([aim[0], aim[1], z_hover]), 0.0, roll,
                       "the adjusted hover", step=1.6, settle=0.3)

        # ---- DOWN AND DROP --------------------------------------------------------
        g_now = float(ms.state.get("gripper") or 0.0)
        if g_now <= GRIP_BLOCKED_PCT:
            raise RuntimeError(f"dropped the tube while carrying it (jaws at {g_now:.1f})")
        tphase("DROP", f"into {rack['name']} hole {k}")
        # the colour's extra height is for the carry and the hover, not the release
        _go_ik(np.array([aim[0], aim[1], z_base + DROP_CLEAR_M]),
               0.0, roll, "the release height", step=1.6, settle=0.3)
        ms.send_joints(ms.observe(False)[0], gripper=float(ms.ARM.gripper.place_open_pct))
        time.sleep(0.5)
        ms._set_carry(False)
        with lock:
            used.append(k)
        _go_ik(np.array([aim[0], aim[1], z_hover + 0.03]), 0.0, roll, "back up",
               step=2.8, settle=0.1, tol=0.03)

        # ---- DID IT GO IN? Look with the arm out of the way ----------------------
        _go(np.array(ms.HOME, np.float64), settle=0.2, step=2.8)
        caps = _top_caps(colour)
        if caps is None:
            return (True, "placed (unverified)",
                    f"dropped at {rack['name']} hole {k} — no top view to check")
        near = min((math.hypot(c[0] - hole_px[0], c[1] - hole_px[1]) for c in caps),
                   default=None)
        if near is not None and near <= HOLE_VERIFY_PX:
            tsay(f"        top camera: {colour} cap {near:.0f}px from hole {k} — in")
            return True, "placed", f"in {rack['name']} hole {k}"
        tsay(f"        top camera: no {colour} cap at hole {k} — missed the hole")
        return (False, "missed hole",
                f"dropped at {rack['name']} hole {k}, no cap there after")

    def racks():
        """The racks as ESTIMATED from the top camera (see TOP_RACKS)."""
        out = []
        for r in TOP_RACKS:
            hs = [top_px_to_xy(u, v) for u, v in r["holes_px"]]
            out.append({"name": r["name"], "colour": r["colour"], "yaw": 0.0,
                        "x": round(sum(h[0] for h in hs) / len(hs), 4),
                        "y": round(sum(h[1] for h in hs) / len(hs), 4),
                        "holes": [[round(h[0], 4), round(h[1], 4)] for h in hs]})
        return out


    # ---- the pick: look, go over, square up, put the cap in the grid --------------
    #: Fingertip height at the grasp. A tube lying down puts its centre one radius up,
    #: and the jaws close AROUND it, so the tip wants to arrive level with that centre
    #: rather than on the table. 8mm is the tube's own radius.
    GRASP_Z = 0.010
    #: How many increments the creep takes. Twelve at ~1cm each is a walk, not a dive,
    #: and every one of them re-measures the cap first.
    #: pick.py aims to 150px before approaching. Same here: the point is only to get the
    #: tube off the edge of the picture, not to line it up -- the approach does that.
    AIM_TOL_PX, AIM_STEPS = 150.0, 6
    #: How far down the planned descent before a close is allowed. pick.py's 0.85: image
    #: alignment fixes the bearing and says nothing about height, and it logged a run that
    #: converged to 7px at 41% and closed on air above the tube.
    AT_DEPTH = 0.85
    #: Increments in the approach. Each one looks first.
    #: 12, not 8. Eight increments closed 246px of error down to 84 and ran out -- the
    #: correction is deliberately small and shrinking, so a large starting error simply
    #: needs more of them. With AIM running at the final pitch the error starts far
    #: smaller anyway, and the loop exits the moment it is inside the jaws.
    N_APPROACH = 12
    #: pick.py's, and this rig's own [grip] log agrees with it: across 13 trials the
    #: target read <=67px from the grip centre on every pick that worked.
    GRASP_RADIUS_PX = 70.0

    #: THE GRIP IS PERPENDICULAR TO THE TABLE. The pitch chain sums to a right angle,
    #: so the jaws come straight down and straddle the lying tube. PERP asks for this
    #: first and settles for the steepest angle that still reaches the tube.
    #:
    #: It was briefly a SEARCH over what the CAMERA could see -- try 72, then 60, then
    #: 48, keep the steepest angle the cap was still visible at -- and the arm kept
    #: settling on 36, most of the way to parallel, because the camera is bolted to the
    #: wrist and a steep pitch points it at the bench just in front of the jaws.
    #: Backing the angle off to keep the cap in shot trades away the thing the grasp
    #: needs. The angle is chosen by REACH now, which is a question about the arm rather
    #: than about the view, and the approach keeps the cap in sight by staying shallow
    #: until it is over the tube.
    GRASP_PITCH = 90.0
    #: How close to GRASP_DX_TARGET_PX the cap has to be before the jaws close. A
    #: tolerance on the set point, not a distance from the jaw centre.
    CENTRE_TOL_PX = 28.0
    #: How far short of the cast the approach stops. The hand wants to arrive beside
    #: the tube and stand up over it, not on top of it.
    APPROACH_LEAD_M = 0.02
    #: Close enough, radially, to call the approach done.
    APPROACH_TOL_M = 0.015
    #: Bites of each vertical descent.
    N_DOWN = 8
    #: The hover: high enough to be clear of everything on the bench, low enough that a
    #: centimetre of reach is worth a useful number of pixels.
    HOVER_Z = 0.06
    #: The trim's budget and bites. Generous, because this is where the cast's inward
    #: bias gets paid off -- the tube is routinely a few centimetres further out than
    #: the cast says, and this is the stage that can see that and fix it.
    TRIM_MAX_M, N_TRIM = 0.07, 10
    #: The first bite of a reach, taken to MEASURE pixels-per-centimetre.
    REACH_PROBE_M = 0.015
    #: The most any single solved bite may be. The estimate is good enough to aim with
    #: and not good enough to trust in one go.
    REACH_BITE_MAX_M = 0.025
    #: The most any reach joint may move for one reach. Beyond this the "solution" is
    #: the other elbow configuration, not a reach.
    REACH_MAX_JOINT_JUMP_DEG = 45.0
    #: How near the fingers' landing point the cap must be ALONG the reach, in pixels.
    #: Wider than the across-tolerance because a bite is 2.5cm and the gain near the end
    #: is around 3000px/m, so one bite is roughly 75px of this error.
    TRIM_DY_TOL_PX = 45.0
    #: WHERE dx SHOULD END UP, in the pixels the overlay prints. The operator gave
    #: the number directly: "I just want the dx value to be around +60-100, this should
    #: be the target grid." So that is the set point -- not zero, and not a fraction of
    #: a grid cell that has to be argued about. dx on the overlay is
    #: (jaw centre x - cap x), so a positive target puts the cap to the LEFT of the jaw
    #: centre and the hand to the RIGHT of the tube, which is what was asked for.
    GRASP_DX_TARGET_PX = 80.0
    #: The second pass after the wrist rolls. Small: the arm is already there.
    RETRIM_MAX_M, N_RETRIM = 0.03, 5

    def _cap_now(want_uv=None, colour=None, tries=6):
        """The freshest cap OF THIS COLOUR, nearest ``want_uv``. Waits for the frame loop.

        COLOUR IS NOT OPTIONAL IN PRACTICE, and leaving it out cost a whole pick. The
        first version matched on proximity alone: pick the cap nearest where the target
        was last seen. That is fine while the camera is still and wrong the moment it
        moves, because the arm moving over one tube slides BOTH caps across the image --
        and the other tube can easily end up nearer to the remembered pixel than the one
        being picked. Observed: the fix jumped 5.6cm between the approach and the view
        from above, which is not a tube moving, it is the tracker changing its mind about
        which tube it was looking at.

        The caps are measured by the overlay hook at frame rate, so this does not run a
        second detection pass -- it waits for one it has not already seen.
        """
        for _ in range(tries):
            # TAKE A FRAME. The caps are measured by the overlay hook, and the overlay
            # hook runs inside publish(), and publish() only runs when observe() is
            # called with overlay=True. The pick reads joints with observe(False)
            # everywhere for speed, so across a whole pick the detector was never run
            # ONCE: every grid check found a cap list older than its freshness window
            # and reported "the cap is not in view" while the cap sat in plain sight at
            # the top of the frame. Asking for the overlay here is what makes the grid
            # stage able to see anything at all -- and it puts the boxes in front of the
            # operator at the moment they matter, which is the same call.
            try:
                ms.observe(True)
            except Exception:
                pass
            caps, t = ms.LAST_CAPS[0], ms.LAST_CAPS[1]
            if caps and time.time() - t < 1.2:
                if colour is not None:
                    caps = [c for c in caps if c["colour"] == colour]
                if not caps:
                    time.sleep(0.15)
                    continue
                if want_uv is None:
                    return max(caps, key=lambda c: c["area"])
                return min(caps, key=lambda c: (c["x"] - want_uv[0]) ** 2
                           + (c["y"] - want_uv[1]) ** 2)
            time.sleep(0.06)
        return None

    def _table_xy(uv):
        """Cast a pixel onto the table plane -> base-frame (x, y), or None."""
        try:
            pt = ms.ray_to_table((float(uv[0]), float(uv[1])),
                                 ms.T_cam_of(ms.observe(False)[0]))
        except Exception:
            return None
        return None if pt is None else (float(pt[0]), float(pt[1]))

    def _vision_approach(see, q_goal, gain_base, what, last_uv):
        """Interpolate the joints toward ``q_goal`` while the BASE tracks what it sees.

        THE ONE APPROACH BOTH STAGES USE. Putting a cap between the jaws and putting a
        held tube over a rack hole are the same problem seen twice: a thing in the
        picture has to end up in the jaw cells while the arm follows a planned descent.
        Everything that was learned the hard way lives here once --

          * LOOK BEFORE MOVING. A lost target means HOLD, not carry on: pick.py records
            an arm that descended through five "not in view" steps and finished past its
            target with nothing in the jaws.
          * THE VERTICAL PIXEL ERROR IS NOT AN ERROR. "Extending the arm moves the cap UP
            31.8px per unit, because the wrist camera tilts as the arm reaches: image
            alignment says retract while physical alignment says extend." Measured here
            over one descent, dy went +87 -> +285 while dx stayed inside 51px. Folding dy
            into a distance makes that distance grow monotonically and no threshold on it
            can ever be met. Alignment is HORIZONTAL; the vertical belongs to the plan.
          * FLIP ONLY ON A REAL REGRESSION. pick.py's +4px margin is against its own
            150px-scale errors; at a handful of pixels it fires on detector noise, and it
            did -- "the error grew 1 -> 5px, wrong way, flipping", four times in one
            descent, each flip undoing the last while the aim was fine.
          * NO IK IN THE LOOP. q_goal is solved once by the caller, holding the wrist
            pitch. Re-solving per step lets the wrist wander, and the camera is bolted to
            it, so the relationship between a correction and its effect stops being
            stable.

        ``see`` returns a dict with x, y and bbox, or None. Returns (reached, last_dist).
        """
        q_start = ms.observe(False)[0].astype(float)
        aim = ms.jaw_frame().centre_uv
        misses, last_dist, reached = 0, None, False
        for step in range(N_APPROACH):
            ms.checkpoint()
            a = (step + 1) / N_APPROACH
            t = see()
            if t is None:
                misses += 1
                if last_dist is not None and last_dist < 170 and a >= 0.7:
                    tsay(f"        {what} passed under the jaws at {last_dist:.0f}px, "
                         f"{100*a:.0f}% down — completing the descent")
                    q_fin = q_goal.copy()
                    q_fin[ms.ARM.pan_joint] = ms.observe(False)[0][ms.ARM.pan_joint]
                    _go(ms._clamp_joints(q_fin), settle=0.20, step=2.0)
                    return True, last_dist
                tsay(f"        {step+1:2d}: {what} not in view — HOLDING ({misses}/4)")
                if misses >= 4:
                    break
                time.sleep(0.12)
                continue
            misses = 0
            last_uv[0] = (t["x"], t["y"])
            ex = aim[0] - t["x"]
            ey = aim[1] - t["y"]
            dist = abs(float(ex))
            jg = ms.jaw_frame()
            bb = t.get("bbox")
            corners = ([(bb[0], bb[1]), (bb[2], bb[1]), (bb[0], bb[3]), (bb[2], bb[3])]
                       if bb else [])
            in_grip = (ms.GRID.in_grip((t["x"], t["y"]), jg)
                       or any(ms.GRID.in_grip(k, jg) for k in corners))
            tsay(f"        {step+1:2d}/{N_APPROACH}: "
                 f"{ms.GRID.cell_of((t['x'], t['y']))} -> jaws {ms.GRID.cell_of(aim)}  "
                 f"dx {ex:+.0f} dy {ey:+.0f}  {100*a:.0f}% down"
                 f"{'  IN GRIP' if in_grip else ''}")

            # NO FLIP-ON-WORSE. It existed because a fixed sign could be wrong; the
            # step is ex/gain now, so a correction that overshoots simply comes back.
            last_dist = dist

            if (in_grip or dist < GRASP_RADIUS_PX) and a >= AT_DEPTH:
                tsay(f"        aligned to {dist:.0f}px and {100*a:.0f}% down — done")
                reached = True
                break

            q = q_start + (q_goal - q_start) * a
            q[ms.ARM.roll_joint] = q_start[ms.ARM.roll_joint]
            pan_now = ms.observe(False)[0][ms.ARM.pan_joint]
            if abs(ex) > 30:
                scale = 1.0 + 1.1 * a
                q[ms.ARM.pan_joint] = float(np.clip(
                    pan_now + _pan_step(ex, gain_base, 0.4, 3.2 / scale),
                    ms.J_LO[ms.ARM.pan_joint], ms.J_HI[ms.ARM.pan_joint]))
            else:
                q[ms.ARM.pan_joint] = pan_now
            _go(ms._clamp_joints(q), settle=0.10, step=2.2)

        if not reached and last_dist is not None:
            tsay(f"        finished {last_dist:.0f}px off — continuing anyway; the "
                 f"episode records how far")
        return reached, last_dist

    def _pan_step(ex, gain, lo, hi):
        """Degrees of base that remove a horizontal error of ``ex`` pixels.

        ex is measured as (where the cap should be) - (where it is), so moving the cap
        by -ex is what zeroes it, and the gain says how many pixels a degree moves it:
        the step is ex/gain. The sign falls out of the arithmetic, which is the whole
        point -- a fixed sign multiplied by ABS(ex) is only right while the error keeps
        the sign it was measured with.
        """
        if not gain or abs(gain) < 1e-6:
            return 0.0
        step = ex / gain
        mag = float(np.clip(abs(step), lo, hi))
        return math.copysign(mag, step)

    def _probe_base(see, last_uv, aim_u):
        """Which way the base joint moves what we are watching. pick.py's probe.

        The 18px floor is its, and its comment says why: "A 3-unit probe moved the cap
        less than the +-8px noise floor, so the sign stayed a guess -- and then the
        flip-on-worse rule toggled it at random every step, leaving dx pinned at +100 for
        an entire descent while 'correcting'."
        """
        for probe_q in (7.0, -10.0, 14.0):
            ms.checkpoint()
            t0 = see()
            if t0 is None:
                continue
            before_x = t0["x"]
            q_from = ms.observe(False)[0].astype(float)
            q_try = q_from.copy()
            q_try[ms.ARM.pan_joint] = float(np.clip(
                q_from[ms.ARM.pan_joint] + probe_q,
                ms.J_LO[ms.ARM.pan_joint], ms.J_HI[ms.ARM.pan_joint]))
            _go(ms._clamp_joints(q_try), settle=0.20, step=2.6)
            after = see()
            _go(ms._clamp_joints(q_from), settle=0.20, step=2.6)
            if after is None:
                continue
            moved = after["x"] - before_x
            if abs(moved) < 18.0:
                tsay(f"        base {probe_q:+.0f} moved it only {moved:+.0f}px "
                     f"— under the noise floor")
                continue
            gain = moved / probe_q
            tsay(f"        base gain {gain:+.2f}px/deg")
            return gain
        tsay("        could not measure the base direction")
        return 0.0

    def cap_pick(colour, uv_hint, map_xy=None):
        """Pick a tube by its cap, in pick.py's order: LOOK, AIM, PROBE, APPROACH, GRASP.

        THE ORDER IS THE ALGORITHM, and getting it wrong is what every failed run here
        had in common. Written out, with what each stage is protecting against:

          LOOK     go UP first, retracted, not reached out. From the look pose the whole
                   bench is in frame; from wherever the arm happened to stop, it is not.
          AIM      turn the BASE ONLY until the cap is near the jaws horizontally. Skip
                   this and the approach starts with the tube at the edge of the picture,
                   where the first small move pushes it out of view entirely -- observed,
                   with the cap 183px off at x=623 in a 640-wide frame, lost on step two
                   and held for the remaining four.
          PROBE    measure which way the base moves the cap. A wrong sign hides under
                   detector noise while steering the arm steadily the wrong way.
          APPROACH move toward the tube in small increments, LOOKING BEFORE EACH ONE, and
                   correcting the bearing as it goes. Stop when the cap's box is in the
                   jaw cells and the hand is low enough.
          GRASP    close, and only claim a tube if the evidence supports one.

        WHAT THE IMAGE IS ALLOWED TO MOVE: the bearing, and nothing else. Turning the
        base swings the hand ACROSS the target; correcting the tip in x and y reaches the
        arm OUT, and told to close a few centimetres of image error that drives the arm
        forward past the tube. The radius comes from the fix and the height from the
        profile.
        """
        # ---- LOOK: go up, and see the whole bench ---------------------------------
        tphase("LOOK", "going up to the look pose")
        _go(ms._clamp_joints(np.array(ms.HOME, np.float64)),
                       settle=0.20, step=2.8)
        j5 = float(ms.observe(False)[0][ms.ARM.roll_joint])

        # OPEN THE JAWS NOW, before anything approaches, so the hand arrives ready
        # instead of travelling in with the fingers wherever the last run left them.
        ms.send_joints(ms.observe(False)[0], gripper=GRIP_PREOPEN_PCT)
        time.sleep(0.2)
        tsay(f"        jaws open to {GRIP_PREOPEN_PCT:.0f} for the approach")

        def _cap_at_map():
            """The visible cap of this colour whose cast lands nearest the map position
            (within 8cm) -- so the arm takes the tube the top camera chose, not just
            the biggest one of that colour in the wrist view."""
            # NEAREST, NOT "WITHIN 8cm". From the look pose the wrist cast is off by
            # more than that, so a hard gate rejected every cap ("nothing" x5 for each
            # tube). The ranking still takes the tube the top camera chose when there
            # are several of one colour; the approach then bounds the radius by the map.
            _cap_now(None, colour)
            best = None
            for c in (ms.LAST_CAPS[0] or []):
                if c["colour"] != colour:
                    continue
                xy_c = _table_xy((c["x"], c["y"]))
                dd = 9.9 if xy_c is None else math.hypot(xy_c[0] - map_xy[0],
                                                           xy_c[1] - map_xy[1])
                if best is None or dd < best[0]:
                    best = (dd, c)
            return None if best is None else best[1]

        cap = _cap_now(uv_hint or None, colour) if map_xy is None else None
        if map_xy is not None:
            # THE MAP KNOWS WHERE IT IS -- TURN AND LOOK. The look pose is one fixed
            # view of a bench wider than the camera, so a tube the scan found perfectly
            # well can sit outside it, and the pick died on the spot with "no green cap
            # in view from the look pose" while the map listed it at 22cm. The scan's
            # own position is the answer: face that bearing and look again, then sweep
            # either side of it.
            # The pan angle comes from the IK, not from an assumed sign convention:
            # aim the tip at the mapped spot (at a radius the arm can certainly make --
            # the bearing is the same all along it) and keep only the base joint.
            q_l = ms.observe(False)[0].astype(float)
            here = float(q_l[ms.ARM.pan_joint])
            bear_m = math.atan2(map_xy[1], map_xy[0])
            # THE BASE ANGLE THAT FACES THE MAP BEARING, read off the arm model: the
            # pan whose fingertip bearing matches. This used to ask the IK for a pose
            # there and, when a close-in tube made the IK fail, quietly swept around
            # wherever the base already was -- the map said +26deg and the arm turned
            # to -14, -26, -2: the wrong way.
            pan_want = _pan_for_bearing(q_l, bear_m)
            tsay(f"        map bearing {math.degrees(bear_m):+.0f}deg -> base {pan_want:+.0f}deg")
            for extra in (0.0,):
                ms.checkpoint()
                want = float(np.clip(pan_want + extra,
                                     ms.J_LO[ms.ARM.pan_joint],
                                     ms.J_HI[ms.ARM.pan_joint]))
                q_l[ms.ARM.pan_joint] = want
                _go(ms._clamp_joints(q_l), settle=0.30, step=2.8)
                cap = _cap_at_map()
                tsay(f"        base {want:+.0f}deg "
                     f"(map bearing {math.degrees(bear_m):+.0f}deg): "
                     + ("found it" if cap else "nothing"))
                if cap is not None:
                    break
            if cap is None:
                q_l[ms.ARM.pan_joint] = here
                _go(ms._clamp_joints(q_l), settle=0.25, step=2.8)
        if cap is None:
            raise RuntimeError(f"no {colour} cap in view from the look pose")
        last_uv = [(cap["x"], cap["y"])]
        focus["colour"], focus["uv"] = colour, (cap["x"], cap["y"])
        xy = _table_xy((cap["x"], cap["y"]))
        if xy is None:
            raise RuntimeError("could not cast the cap onto the table plane")
        tsay(f"        {colour} cap at ({cap['x']:.0f},{cap['y']:.0f})px -> "
             f"({xy[0]*100:+.1f},{xy[1]*100:+.1f})cm, r={math.hypot(*xy)*100:.1f}cm")

        AIM = ms.jaw_frame().centre_uv

        # ---- THE WRIST STANDS UP ON THE WAY IN, not before -----------------------
        #
        # The grasp pose is solved at GRASP_PITCH, so the approach interpolates the hand
        # to a right angle as it travels; it arrives perpendicular and it never needed a
        # stage of its own.
        #
        # It HAD one, tipping to 90 in place at the look pose, and the log says what
        # that costs: the camera rides on the wrist, so from up there a 26 -> 90 tip
        # points it at the bench under the jaws and the cap is simply gone --
        #
        #     [AIM]    (nothing: no cap)
        #     [CENTRE] centre 1: cap not in view — closing from here
        #     [GRASP]  settled at 7.8 -> EMPTY
        #
        # -- three tries, every one of them blind from the second stage onward. Tipping
        # while ALSO moving toward the tube keeps it in shot, because the arm is closing
        # the distance the tilt is opening up.
        pitch_hold = GRASP_PITCH

        # ---- PROBE the BASE JOINT, which is what the correction moves --------------
        # pick.py's own comment records why this has to be measured hard: "A 3-unit probe
        # moved the cap less than the +-8px noise floor, so the sign stayed a guess -- and
        # then the flip-on-worse rule toggled it at random every step, leaving dx pinned
        # at +100 for an entire descent while 'correcting'."
        #
        # WHAT IS KEPT IS THE GAIN ITSELF, px of cap motion per degree of base -- not a
        # sign distilled out of it. pick.py reduced it to `sign = +1 if ex * gain > 0`
        # and then stepped ABS(ex) times that sign, which is only correct while the
        # error keeps the sign it had at the probe. It does not. The moment a set point
        # other than zero was introduced the error started NEGATIVE, the sign had been
        # measured against a POSITIVE one, and the correction drove the wrong way on
        # every single step --
        #
        #     dx +8 -15 -42 -67 -93 -122 -154 -175 -204 -229 -254 -284
        #
        # -- twelve bites, monotonically away, "correcting" the whole time. Keeping the
        # gain makes the step ex/gain, which carries the error's sign with it and cannot
        # do this: if the error changes sign, so does the correction.
        tphase("PROBE", "measuring how far the base moves the cap")
        gain_base = 0.0
        for probe_q in (8.0,):
            ms.checkpoint()
            before_x = cap["x"]
            q_from = ms.observe(False)[0].astype(float)
            q_try = q_from.copy()
            q_try[ms.ARM.pan_joint] = float(np.clip(
                q_from[ms.ARM.pan_joint] + probe_q,
                ms.J_LO[ms.ARM.pan_joint], ms.J_HI[ms.ARM.pan_joint]))
            _go(ms._clamp_joints(q_try), settle=0.20, step=2.6)
            after = _cap_now(last_uv[0], colour)
            _go(ms._clamp_joints(q_from), settle=0.20, step=2.6)
            if after is None:
                tsay(f"        base {probe_q:+.0f}: lost the cap during the probe")
                continue
            moved = after["x"] - before_x
            if abs(moved) < 18.0:
                tsay(f"        base {probe_q:+.0f} moved the cap only {moved:+.0f}px "
                     f"— under the noise floor, probing harder")
                continue
            gain_base = moved / probe_q
            tsay(f"        base gain {gain_base:+.2f}px/deg "
                 f"({probe_q:+.0f}deg moved the cap {moved:+.0f}px)")
            break
        if gain_base == 0.0:
            # ONE PROBE, THEN THE KNOWN VALUE. Three probes that each lost the cap was
            # most of the "10 loops". Measured on this rig: -5.1 to -10.4 px/deg.
            gain_base = BASE_GAIN_DEFAULT_PX_PER_DEG
            tsay(f"        probe lost the cap — using the usual {gain_base:+.1f}px/deg")
        cap = _cap_now(last_uv[0], colour) or cap

        # ---- AIM: base only, bring the cap across before approaching --------------
        #
        # AFTER THE PITCH, NOT BEFORE IT. Aiming first and then tilting the wrist throws
        # the aim away: the camera rides on that wrist, so choosing a new pitch moves the
        # cap right across the frame. Measured -- an approach that began with the cap
        # 246px off, because AIM had run at the look pose's 25 degrees and the hand then
        # tipped to 48. The approach spent all eight of its increments clawing that back
        # (246 -> 84px) and ran out before it was inside the jaws.
        #
        # Aiming at the FINAL pitch costs one extra stage and hands the approach an error
        # it can actually finish.
        tphase("AIM", "turning the base to bring the cap across")
        misses_aim = 0
        for k in range(AIM_STEPS):
            ms.checkpoint()
            c = _cap_now(last_uv[0], colour)
            if c is None:
                tsay(f"        aim {k+1}: cap not in view")
                misses_aim += 1
                if misses_aim >= 2:
                    break
                continue
            last_uv[0] = (c["x"], c["y"])
            ex = AIM[0] - c["x"]
            if abs(ex) < AIM_TOL_PX:
                tsay(f"        aimed: {abs(ex):.0f}px < {AIM_TOL_PX:.0f}px")
                break
            if gain_base == 0.0:
                tsay("        no measured base gain — leaving the aim alone")
                break
            q_a = ms.observe(False)[0].astype(float)
            d_pan = _pan_step(ex, gain_base, 1.0, 5.0)
            q_a[ms.ARM.pan_joint] = float(np.clip(
                q_a[ms.ARM.pan_joint] + d_pan,
                ms.J_LO[ms.ARM.pan_joint], ms.J_HI[ms.ARM.pan_joint]))
            _go(ms._clamp_joints(q_a), settle=0.18, step=2.6)
            tsay(f"        aim {k+1}: dx {ex:+.0f}px -> base {d_pan:+.1f}deg")

        # Re-fix now the tube is in front of the camera instead of off to one side: the
        # cast is least accurate down a grazing sightline, which is exactly where it was.
        c = _cap_now(last_uv[0], colour)
        if c is not None:
            xy_a = _table_xy((c["x"], c["y"]))
            if xy_a is not None:
                tsay(f"        re-fixed after aiming: ({xy_a[0]*100:+.1f},"
                     f"{xy_a[1]*100:+.1f})cm")
                xy, cap = xy_a, c
                last_uv[0] = (c["x"], c["y"])

        # A grasp pose used to be solved here and driven to by a joint interpolation
        # that ignored the picture until it arrived. It is gone -- the stages below
        # travel by what the camera measures on each bite -- and this is only the
        # default PERP starts from.
        pitch_hold = GRASP_PITCH

        # ---- THE ONE MOVE THAT MATTERS, as a function ----------------------------
        #
        # "Calculate the IK to put the cap at the bottom of the camera view." That is
        # the job, and this is it: measure how far the cap moves down the frame per
        # centimetre of reach, then solve for the reach that lands it on the jaw cells.
        #
        # MEASURED, NOT PROJECTED. Casting the cap to the table and driving there trusts
        # a hand-eye that reads short, and the arm stops with the cap 350px high.
        # Casting the JAW pixel as well and differencing is worse -- the fingertips sit
        # 13cm above the table, so their pixel projects far down its own sightline and
        # the correction comes out a lunge. One small bite, though, moves the cap a
        # number of pixels that can simply be counted, and that count has no calibration
        # in it at all. It is re-estimated after every bite, because the gain grows as
        # the hand closes in.
        def _grasp_uv():
            """The pixel the fingertips will occupy once they are down at the tube.

            NOT the jaw cells, and the difference is the whole reason the jaws kept
            closing on nothing. The cells are where the fingers appear RIGHT NOW, nine
            centimetres above the bench, and a cap lined up with them is lined up with
            the SIGHTLINE through them -- which carries on down and forward and meets
            the table well beyond where the fingers will land. One run trimmed the cap
            neatly into the cells, "14px off centre", came straight down and closed on
            air at 5.7.

            So the target is the fingertip's own grasp position, (x, y, GRASP_Z),
            projected into the picture through the same hand-eye the casts use. It moves
            up the frame as the hand descends, which is exactly right: it is a point on
            the table, seen from a camera that is getting closer to it.
            """
            q_g = ms.observe(False)[0]
            t_g = ms._tip(q_g)
            try:
                uv = ms.project_base(np.array([t_g[0], t_g[1], GRASP_Z], float),
                                     ms.T_cam_of(q_g))
            except Exception:
                return None
            return None if uv is None else (float(uv[0]), float(uv[1]))

        def _close_in(pitch_now, budget_m, n_max, what="closing in"):
            """Reach until the cap is on the fingers' landing point. (reached, dx_px)."""
            q_h = ms.observe(False)[0].astype(float)
            tip_h = ms._tip(q_h)
            bear = math.atan2(tip_h[1], tip_h[0])
            r_at = float(math.hypot(tip_h[0], tip_h[1]))
            z_at = float(tip_h[2])
            got, dxp, gone, gain, prev = False, None, 0.0, None, None
            for k in range(n_max):
                ms.checkpoint()
                c = _cap_now(last_uv[0], colour)
                if c is None:
                    tsay(f"        {what} {k+1}: cap not in view — stopping here")
                    break
                last_uv[0] = (c["x"], c["y"])
                jg = ms.jaw_frame()
                aim = _grasp_uv() or jg.centre_uv
                # A CELL TO THE RIGHT OF THE TUBE, so the cap ends up in the lower LEFT
                # of the grip rather than dead centre. Dead centre sounds right and is
                # not: the moving finger sweeps in from one side, and a cap sitting on
                # the line it sweeps through gets nudged along instead of captured. Put
                # the tube slightly into the fixed finger's half and the closing jaw
                # arrives against it.
                aim = (aim[0] - GRASP_DX_TARGET_PX, aim[1])
                ex = aim[0] - c["x"]
                dy = aim[1] - c["y"]
                dxp = abs(ex)
                bb = c.get("bbox")
                corners = ([(bb[0], bb[1]), (bb[2], bb[1]), (bb[0], bb[3]), (bb[2], bb[3])]
                           if bb else [])
                inside = (ms.GRID.in_grip((c["x"], c["y"]), jg)
                          or any(ms.GRID.in_grip(k2, jg) for k2 in corners))
                if abs(ex) <= CENTRE_TOL_PX and abs(dy) <= TRIM_DY_TOL_PX:
                    tsay(f"        {what} {k+1}: cap on the fingers' landing point "
                         f"({abs(ex):.0f}px across, {abs(dy):.0f}px along"
                         + (", in the jaw cells" if inside else "")
                         + ") — there")
                    got = True
                    break

                if prev is not None and abs(prev[1]) > 1e-6:
                    moved_px, moved_m = prev[0] - dy, prev[1]
                    g_now = moved_px / moved_m
                    if g_now > 200.0:
                        gain = g_now if gain is None else 0.5 * (gain + g_now)
                    elif not inside:
                        tsay(f"        {what} {k+1}: {moved_m*100:.1f}cm moved the cap "
                             f"only {moved_px:+.0f}px — not closing the gap this way")
                        break

                if gain is None:
                    bite = REACH_PROBE_M
                    why = f"probing {bite*100:.1f}cm to measure pixels per cm"
                else:
                    need = dy / gain
                    bite = float(np.clip(need, 0.0, REACH_BITE_MAX_M))
                    why = (f"{dy:+.0f}px high, {gain:.0f}px/m -> {need*100:.1f}cm to "
                           f"go, taking {bite*100:.1f}cm")
                if bite <= 0.001:
                    tsay(f"        {what} {k+1}: nothing left to reach "
                         f"({dy:+.0f}px along) — there")
                    got = inside or abs(dy) <= TRIM_DY_TOL_PX
                    break
                if gone + bite > budget_m:
                    bite = budget_m - gone
                    if bite <= 0.001:
                        tsay(f"        {what} {k+1}: {gone*100:.0f}cm used up — "
                             f"stopping rather than pushing")
                        break

                q_n = ms.observe(False)[0].astype(float)
                tgt = np.array([(r_at + bite) * math.cos(bear),
                                (r_at + bite) * math.sin(bear), z_at])
                q_s, e_s = ms._ik_hold_pitch(q_n, tgt, pitch_now, j5, ret_err=True)
                if e_s > 0.03:
                    tsay(f"        {what} {k+1}: r={(r_at+bite)*100:.1f}cm is past the "
                         f"arm at {pitch_now:.0f}deg — stopping here")
                    break
                q_s = np.asarray(q_s, float)
                jump = max(abs(float(q_s[i]) - float(q_n[i]))
                           for i in ms.ARM.pitch_chain)
                if jump > REACH_MAX_JOINT_JUMP_DEG:
                    tsay(f"        {what} {k+1}: that only solves by flipping the arm "
                         f"over ({jump:.0f}deg) — stopping here")
                    break
                pan = float(q_n[ms.ARM.pan_joint])
                if abs(ex) > CENTRE_TOL_PX:
                    pan = float(np.clip(
                        pan + _pan_step(ex, gain_base, 0.3, 1.8),
                        ms.J_LO[ms.ARM.pan_joint], ms.J_HI[ms.ARM.pan_joint]))
                q_s[ms.ARM.pan_joint] = pan
                q_s[ms.ARM.roll_joint] = j5
                tsay(f"        {what} {k+1}: cap {ms.GRID.cell_of((c['x'], c['y']))} "
                     f"dx {ex:+.0f} dy {dy:+.0f}px — {why}")
                _go(ms._clamp_joints(q_s), settle=0.12, step=2.0)
                r_at += bite
                gone += bite
                prev = (dy, bite)
            return got, dxp

        # ---- APPROACH: get over the cap WITH IT IN VIEW ---------------------------
        #
        # THE HAND STAYS SHALLOW FOR THIS, so the cap never leaves the picture. Going in
        # already perpendicular points the camera at the bench right under the jaws and
        # a tube any distance out is off the top of the frame before the arm has moved:
        # three runs descended blind that way and closed on air.
        #
        # AND THE IMAGE CANNOT SAY WHEN IT HAS ARRIVED. Measured on this rig, at this
        # angle: one 1.5cm bite outward moved the cap 38px UP the frame, not down. That
        # is not a fluke, it is the geometry -- the camera tilts with the forearm, and
        # pick.py measured the same thing (+31.8px per unit of reach) and concluded the
        # image "cannot decide the descent". A loop waiting for the cap to come down to
        # the jaws at this pitch waits forever.
        #
        # So each signal is used where it is sound: THE BEARING from the picture, which
        # the base closes reliably, and THE DISTANCE from the table cast, re-taken every
        # bite. The cast is biased down a grazing sightline and gets better as the hand
        # gets nearer and looks more steeply down -- so the estimate that matters most,
        # the last one, is also the best one.
        tphase("APPROACH", "moving over the cap, keeping it in view")
        pitch_see = float(sum(ms.observe(False)[0][i] for i in ms.ARM.pitch_chain))
        tsay(f"        approaching at {pitch_see:+.0f}deg, where the cap stays visible")
        reached, last_dist, r_goal = False, None, float(math.hypot(xy[0], xy[1]))
        for k in range(N_APPROACH):
            ms.checkpoint()
            c = _cap_now(last_uv[0], colour)
            if c is None:
                tsay(f"        approach {k+1}: cap not in view — holding here")
                break
            last_uv[0] = (c["x"], c["y"])
            jg = ms.jaw_frame()
            # THE SAME OFFSET THE TRIM USES, and it belongs here most of all: this is
            # the loop that closes the sideways error, and it was centring on the bare
            # jaw pixel while the offset sat in a later stage that often stops early.
            # So the bias was in the code and never on the robot.
            # dx is driven to GRASP_DX_TARGET_PX, not to zero.
            ex = (jg.centre_uv[0] - c["x"]) - GRASP_DX_TARGET_PX
            last_dist = abs(ex)
            xy_c = _table_xy((c["x"], c["y"]))
            if xy_c is not None:
                xy = xy_c
                r_goal = 0.5 * (r_goal + float(math.hypot(xy_c[0], xy_c[1])))
            q_n = ms.observe(False)[0].astype(float)
            tip_n = ms._tip(q_n)
            r_now = float(math.hypot(tip_n[0], tip_n[1]))
            gap = r_goal - APPROACH_LEAD_M - r_now
            # PRINT THE dx THE OVERLAY PRINTS, not the control error. They differ by
            # the set point, and a log that quietly reports a different number than the
            # picture is a log that cannot be checked against the picture.
            tsay(f"        approach {k+1}: cap {ms.GRID.cell_of((c['x'], c['y']))} "
                 f"dx {ex + GRASP_DX_TARGET_PX:+.0f}px "
                 f"(target {GRASP_DX_TARGET_PX:+.0f}, so {ex:+.0f} off), "
                 f"cast {r_goal*100:.1f}cm, hand {r_now*100:.1f}cm "
                 f"-> {gap*100:+.1f}cm to go")
            if gap <= APPROACH_TOL_M and abs(ex) <= CENTRE_TOL_PX:
                tsay(f"        over the tube ({gap*100:+.1f}cm, {abs(ex):.0f}px) "
                     f"— stopping here")
                reached = True
                break

            bite = float(np.clip(gap, 0.0, REACH_BITE_MAX_M))
            bear_n = math.atan2(tip_n[1], tip_n[0])
            q_s = None
            if bite > 0.002:
                tgt = np.array([(r_now + bite) * math.cos(bear_n),
                                (r_now + bite) * math.sin(bear_n), float(tip_n[2])])
                q_t, e_t = ms._ik_hold_pitch(q_n, tgt, pitch_see, j5, ret_err=True)
                if e_t > 0.03:
                    tsay(f"        approach {k+1}: {(r_now+bite)*100:.1f}cm does not solve "
                         f"at {pitch_see:+.0f}deg — holding the radius")
                elif max(abs(float(q_t[i]) - float(q_n[i]))
                         for i in ms.ARM.pitch_chain) > REACH_MAX_JOINT_JUMP_DEG:
                    tsay(f"        approach {k+1}: that only solves by flipping the arm "
                         f"over — going no further out")
                else:
                    q_s = np.asarray(q_t, float)
            if q_s is None:
                q_s = q_n.copy()
            if abs(ex) > CENTRE_TOL_PX:
                q_s[ms.ARM.pan_joint] = float(np.clip(
                    float(q_n[ms.ARM.pan_joint]) + _pan_step(ex, gain_base, 0.4, 2.2),
                    ms.J_LO[ms.ARM.pan_joint], ms.J_HI[ms.ARM.pan_joint]))
            q_s[ms.ARM.roll_joint] = j5
            if np.allclose(q_s, q_n, atol=1e-3):
                tsay(f"        approach {k+1}: nothing left to move — stopping")
                break
            _go(ms._clamp_joints(q_s), settle=0.14, step=2.2)

        # ---- DOWN TO A HOVER, still shallow --------------------------------------
        #
        # HEIGHT BEFORE ANGLE, and the order is not arbitrary. Standing the hand up at
        # the look pose's altitude puts the camera 15cm above the bench, where one
        # centimetre of reach moves the cap 15px -- measured, gain 1550px/m -- and the
        # trim computed that it needed 25cm more reach, which is nonsense the arm cannot
        # act on anyway. Six centimetres up the same bite moves the cap several times as
        # far, the cast down a near-vertical sightline is at its most accurate, and the
        # radius the arm can make at a right angle is larger down there than it is up
        # here. So: come down first, at the angle that keeps the tube in shot.
        def _ik_steep(q_n, p, pitch_lo, pitch_hi):
            """Steepest pitch in [pitch_lo, pitch_hi] that reaches ``p``, from several seeds.

            One seed is why grasps came in at 70deg: the model reaches 90 at the grasp
            height out to 30cm, but not from wherever the solver happened to start.
            """
            seeds = [q_n]
            for sh, el, wf in ((-30, 40, 70), (0, -10, 90), (20, -40, 95), (-10, 20, 80)):
                sd = q_n.copy()
                sd[ms.ARM.pitch_chain[0]], sd[ms.ARM.pitch_chain[1]],                     sd[ms.ARM.pitch_chain[2]] = sh, el, wf
                seeds.append(sd)
            for pitch in np.arange(pitch_hi, pitch_lo - 0.1, -2.5):
                best = None
                for sd in seeds:
                    q_d, e_d = ms._ik_hold_pitch(sd, p, float(pitch), j5, ret_err=True)
                    if e_d <= 0.004 and (best is None or np.abs(q_d - q_n).sum()
                                         < np.abs(best - q_n).sum()):
                        best = np.asarray(q_d, float)
                if best is not None:
                    return best, float(pitch)
            return None, None

        def _down_to(z_want, pitch_now, what, pitch_end=None):
            """Straight down to ``z_want`` at fixed x and y. Returns the height reached.

            With ``pitch_end`` the hand also steepens toward it on the way down, so the
            grasp is taken square to the table even where the hover could not be.
            """
            tip0 = np.asarray(ms._tip(ms.observe(False)[0]), float)
            z0 = float(tip0[2])
            if z0 <= z_want + 0.002:
                tsay(f"        already at {z0*100:+.1f}cm")
                return z0
            tsay(f"        {z0*100:+.1f}cm -> {z_want*100:+.1f}cm at "
                 f"({tip0[0]*100:+.1f},{tip0[1]*100:+.1f})cm")
            for k in range(N_DOWN):
                ms.checkpoint()
                z_k = z0 + (z_want - z0) * ((k + 1) / N_DOWN)
                q_n = ms.observe(False)[0].astype(float)
                if pitch_end is not None:
                    want = pitch_now + (pitch_end - pitch_now) * ((k + 1) / N_DOWN)
                    q_d, got = _ik_steep(q_n, np.array([tip0[0], tip0[1], z_k]),
                                         pitch_now, want)
                    if q_d is not None:
                        _go(ms._clamp_joints(q_d), settle=0.12, step=1.8)
                        continue
                q_d, e_d = ms._ik_hold_pitch(
                    q_n, np.array([tip0[0], tip0[1], z_k]), pitch_now, j5, ret_err=True)
                if e_d > 0.02:
                    tsay(f"        {what} {k+1}: z={z_k*100:+.1f}cm not reachable "
                         f"(residual {e_d*100:.1f}cm) — stopping here")
                    break
                _go(ms._clamp_joints(np.asarray(q_d, float)),
                               settle=0.12, step=1.8)
            q_e = ms.observe(False)[0]
            z_end = float(ms._tip(q_e)[2])
            tsay(f"        at {z_end*100:+.1f}cm, hand at "
                 f"{float(sum(q_e[i] for i in ms.ARM.pitch_chain)):+.0f}deg (90 = square)")
            return z_end

        tphase("DOWN", f"down to a hover {HOVER_Z*100:.0f}cm up, still able to see it")
        _down_to(HOVER_Z, pitch_see, "hover")

        # ---- PERPENDICULAR: stand the hand up OVER the tube ----------------------
        #
        # Holding the fingertip where it is and only changing the angle: the hand is
        # already over the cap, so the fingers stay there while the forearm comes
        # upright.
        tphase("PERP", "standing up over the tube, as square as the reach allows")
        q_u = ms.observe(False)[0].astype(float)
        tip_u = np.asarray(ms._tip(q_u), float)
        bear_u = math.atan2(tip_u[1], tip_u[0])
        r_u = float(math.hypot(tip_u[0], tip_u[1]))
        r_need = max(r_goal, r_u)
        tsay(f"        hand at {r_u*100:.1f}cm, the tube casts to {r_goal*100:.1f}cm")

        # THE ANGLE IS CHOSEN BY WHETHER IT CAN REACH THE TUBE, and this is the fault
        # that has been wasting whole runs. Standing the wrist up ON THE SPOT and only
        # then asking the reach to close the gap gets the refusal every time --
        #
        #     trim 2: r=28.1cm is past the arm at 90deg — stopping here
        #     trim 3: r=30.2cm is past the arm at 85deg — stopping here
        #
        # -- and the arm closes its jaws 390px short of a tube it was never going to
        # touch. A right angle costs radial reach: the forearm points down instead of
        # out, and past about 25cm this arm cannot do both. So the question is asked in
        # the order that matters: for each angle, steepest first, CAN THE HAND GET TO
        # THE TUBE? The first yes wins, and the move is made to that radius at that
        # angle in one go. The log then says plainly how square the grasp will be.
        best = None
        for trial in np.arange(GRASP_PITCH, GRASP_PITCH - 40.0, -5.0):
            for r_try in (r_need, r_need - 0.01, r_need - 0.02):
                if r_try <= r_u - 0.01:
                    continue
                tgt = np.array([r_try * math.cos(bear_u), r_try * math.sin(bear_u),
                                float(tip_u[2])])
                q_t, e_t = ms._ik_hold_pitch(q_u, tgt, float(trial), j5, ret_err=True)
                if e_t > 0.02:
                    continue
                if max(abs(float(q_t[i]) - float(q_u[i]))
                       for i in ms.ARM.pitch_chain) > 70.0:
                    continue
                best = (float(trial), float(r_try), np.asarray(q_t, float))
                break
            if best is not None:
                break

        if best is None:
            # Nothing reaches it. Stand up where we are and let the trim do what it can
            # -- but say so, because this is the arm's limit and not a tuning problem.
            tsay(f"        no angle down to {GRASP_PITCH-35:.0f}deg reaches "
                 f"{r_need*100:.1f}cm — standing up here and trimming by eye")
            pitch_hold = None
            for trial in np.arange(GRASP_PITCH, GRASP_PITCH - 35.0, -5.0):
                q_t, e_t = ms._ik_hold_pitch(q_u, tip_u, float(trial), j5, ret_err=True)
                if e_t <= 0.02:
                    _go(ms._clamp_joints(np.asarray(q_t, float)),
                                   settle=0.28, step=2.2)
                    pitch_hold = float(trial)
                    break
            if pitch_hold is None:
                pitch_hold = float(sum(q_u[i] for i in ms.ARM.pitch_chain))
            tsay(f"        wrist at {pitch_hold:+.0f}deg")
        else:
            pitch_hold, r_set, q_set = best
            q_set[ms.ARM.roll_joint] = j5
            tsay(f"        {pitch_hold:+.0f}deg reaches {r_set*100:.1f}cm"
                 + ("" if pitch_hold >= GRASP_PITCH else
                    f" — {GRASP_PITCH:.0f}deg cannot get out that far")
                 + f", going there ({(r_set - r_u)*100:+.1f}cm out)")
            _go(ms._clamp_joints(q_set), settle=0.28, step=2.2)

        # ---- TRIM: the one place the picture can judge the reach ------------------
        #
        # Right angle, low over the bench: now the camera looks down just ahead of the
        # jaws, so reaching out walks the cap DOWN the frame and arrival is something
        # the picture shows rather than something the cast has to be trusted for. This
        # is the loop that has worked every time it had room -- dy 345 -> 125px and into
        # the jaws.
        tphase("TRIM", "reaching by eye until the cap sits in the jaws")
        seen, d_after = _close_in(pitch_hold, TRIM_MAX_M, N_TRIM, what="trim")
        if d_after is not None:
            last_dist = d_after
            reached = reached and seen if seen is not None else reached

        # ---- DOWN THE LAST BIT ---------------------------------------------------
        tphase("DOWN", "the last few centimetres onto the tube")

        def _body_angle_once(bgr, cx, cy, r_cap):
            """The tube BODY's direction from its cap, in the wrist view (deg, 0-360), or None.

            THE BODY, NOT THE CAP. The cap is about as long as it is wide, so its own
            "axis" is noise that snaps to flat -- the twist read 0, +10, +15 deg run after
            run and rolled the same way every time. The body is the long streak leaving
            the cap. Scored by CONTRAST: brighter along the ray than 35deg either side of
            it, so a bright wooden bench (uniformly bright) scores nothing and a white or
            clear tube on it or on the dark mat scores high. The darker half of each ray
            is what counts, so a gap -- not this tube -- kills it.
            """
            g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (7, 7), 0).astype(float)
            H, W = g.shape

            def ray(a):
                ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                v = [g[int(cy + sa * r), int(cx + ca * r)]
                     for r in range(int(0.9 * r_cap), int(0.9 * r_cap) + 150, 3)
                     if 0 <= int(cy + sa * r) < H and 0 <= int(cx + ca * r) < W]
                return float(np.mean(sorted(v)[:len(v) // 2])) if len(v) >= 15 else None

            rays = {a: ray(a) for a in range(0, 360, 3)}
            best = None
            for a, v in rays.items():
                if v is None:
                    continue
                side = [rays.get((a + d) % 360) for d in (-36, 36)]
                side = [x for x in side if x is not None]
                if not side:
                    continue
                score = v - float(np.mean(side))
                if best is None or score > best[0]:
                    best = (score, a)
            if best is None or best[0] < BODY_MIN_CONTRAST:
                return None
            return float(best[1])

        def _cap_angle_now():
            """The tube's axis in the wrist view (deg, mod 180): median of 3, or None."""
            vals = []
            for _ in range(3):
                c = _cap_now(last_uv[0], colour)
                src = ms.latest_rgb[0]
                if c is None or src is None:
                    continue
                last_uv[0] = (c["x"], c["y"])
                bgr = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2BGR)
                x0, y0, x1, y1 = c["bbox"]
                a = _body_angle_once(bgr, c["x"], c["y"], 0.5 * max(x1 - x0, y1 - y0))
                if a is not None:
                    vals.append(fold(a))
            if len(vals) < 2:
                return None
            base = vals[0]
            folded = [base + fold(v - base) for v in vals]
            if max(folded) - min(folded) > TWIST_MAX_SPREAD_DEG:
                tsay(f"        twist: the body reads {max(folded)-min(folded):.0f}deg apart "
                     f"across looks — not trusting it")
                return None
            return fold(float(np.median(folded)))

        def _wrist_twist():
            nonlocal j5
            tphase("TWIST", "squaring the jaws to the tube, from the wrist camera")
            gain = float(ms.jaw_frame().roll_gain) or 1.0
            a_t = _cap_angle_now()
            if a_t is None:
                tsay("        twist: no clear tube body in the wrist view — leaving the "
                     "wrist where it is")
                return
            err = fold(a_t - (TWIST_JAW_AXIS_DEG + 90.0))
            tsay(f"        twist: tube body at {a_t:+.0f}deg in the wrist view, "
                 f"{err:+.0f}deg off square")
            if abs(err) <= TWIST_TOL_DEG:
                return
            q_before = ms.observe(False)[0].astype(float)
            d_roll = float(np.clip(TWIST_FRACTION * -err / gain, -TWIST_MAX_DEG, TWIST_MAX_DEG))
            if abs(err) > TWIST_AMBIGUOUS_DEG:
                # NEARLY PERPENDICULAR: either way is as short, and a 3deg reading error
                # picked the side -- +43 on one try (wrong), -44 on the next (right).
                # Always the side that worked.
                d_roll = -abs(d_roll)
            q_w = q_before.copy()
            q_w[ms.ARM.roll_joint] = float(np.clip(q_w[ms.ARM.roll_joint] + d_roll,
                                                   ms.J_LO[ms.ARM.roll_joint],
                                                   ms.J_HI[ms.ARM.roll_joint]))
            _go(q_w, settle=0.25, step=2.4)
            # CHECK THE PICTURE TURNED WITH THE WRIST. If the tube did not rotate in the
            # view by about the roll, the reading was not the tube: undo, do not repeat.
            a2 = _cap_angle_now()
            expect = fold(a_t + gain * d_roll)
            if a2 is None or abs(fold(a2 - expect)) > TWIST_CHECK_TOL_DEG:
                tsay(f"        twist: after rolling {d_roll:+.0f}deg the tube reads "
                     f"{'nothing' if a2 is None else format(a2, '+.0f') + 'deg'}, expected "
                     f"{expect:+.0f}deg — the reading was wrong; rolling back")
                _go(q_before, settle=0.25, step=2.4)
                return
            j5 = float(q_w[ms.ARM.roll_joint])
            tsay(f"        rolled the wrist {d_roll:+.0f}deg -> "
                 f"{abs(fold(a2 - (TWIST_JAW_AXIS_DEG + 90.0))):.0f}deg off square")

        def _err_px():
            """Cap minus the projected grasp point, in the wrist view, or None."""
            gu = _grasp_uv()
            c = _cap_now(last_uv[0], colour)
            if gu is None or c is None:
                return None
            return np.array([c["x"] - gu[0], c["y"] - gu[1]])

        # ---- FROM THE TOP, STRAIGHT DOWN ---------------------------------------------
        # Nothing may move the fingertips sideways near the tube. Two things did: the
        # hand STEEPENING while it descended (joint-space moves swing the tips in an
        # arc -- ~1cm sideways on one run) and the TWIST (the tips sit off the roll axis,
        # so rolling sweeps them). So: down to TWIST_Z at the angle already held; square
        # the hand to 90 THERE, tips well clear of the tube; twist; put the tips back
        # over the same spot; then straight down at a fixed angle and close.
        # TWIST FIRST, AT HOVER HEIGHT (the operator: "do the twist before coming down").
        # High up the tips are nowhere near the tube, and rolling there sweeps nothing.
        q_c = ms.observe(False)[0].astype(float)
        tip_h = np.asarray(ms._tip(q_c), float)
        _wrist_twist()
        q_c = ms.observe(False)[0].astype(float)
        p_now = float(sum(q_c[i] for i in ms.ARM.pitch_chain))
        q_back, e_back = ms._ik_hold_pitch(q_c, tip_h, p_now, j5, ret_err=True)
        if e_back <= 0.01:
            _go(np.asarray(q_back, float), settle=0.2, step=1.8)   # tips back over the spot

        # then down to TWIST_Z at that angle, square the hand in place there
        _down_to(TWIST_Z, p_now, "down")
        q_c = ms.observe(False)[0].astype(float)
        tip_c = np.asarray(ms._tip(q_c), float)
        spot = np.array([tip_c[0], tip_c[1], TWIST_Z])
        q_sq, p_sq = _ik_steep(q_c, spot, float(sum(q_c[i] for i in ms.ARM.pitch_chain)),
                               GRASP_PITCH)
        if q_sq is not None:
            q_sq[ms.ARM.roll_joint] = j5
            _go(q_sq, settle=0.25, step=1.8)
            tsay(f"        hand squared to {p_sq:+.0f}deg at {TWIST_Z*100:.0f}cm, over the spot")
        p_now = float(sum(ms.observe(False)[0][i] for i in ms.ARM.pitch_chain))

        # straight down, the angle held -- no arc, no sweep
        _down_to(GRASP_Z, p_now, "down")

        # ---- GRASP ----------------------------------------------------------------
        tphase("GRASP", "closing across the tube")
        # OPEN A LITTLE, NOT ALL THE WAY. A 16mm tube needs the jaws barely parted --
        # they rest at 6.5 on air and at 13-21 holding one -- so the 95 they were opened
        # to was 70 units of travel spent before the fingers were anywhere near the
        # tube, and every one of those units was a chance to knock it. Opening to
        # GRIP_PREOPEN_PCT leaves clear room around the tube and starts the close where
        # the work actually is.
        ms.send_joints(ms.observe(False)[0], gripper=GRIP_PREOPEN_PCT)
        time.sleep(0.25)
        tsay(f"        jaws opened to {GRIP_PREOPEN_PCT:.0f} (wide is "
             f"{ms.ARM.gripper.open_pct:.0f}) — closing from there on the torque")
        held_i, idle = ms.close_with_current(step=3.0, delay=0.09,
                                             ignore_above_pct=GRIP_TRUST_BELOW_PCT,
                                             from_pct=GRIP_PREOPEN_PCT)

        # WHERE THE JAWS COME TO REST, because the current says nothing on this arm.
        # gripper_current() reads 0 here run after run -- the servo either does not
        # report it or reports it too small to use against an 8-count threshold -- which
        # makes the contact test compare against nothing and call any draw a grasp. That
        # is how a run reported SUCCESS with the cap 218px away.
        #
        # The position is the signal the X250 uses for exactly this reason, and it is
        # available here: jaws closed on air settle at the closed stop, and jaws closed
        # on a 16mm tube cannot. `settled` waits for motion to stop first, because
        # mid-close the jaws pass THROUGH the holding band on their way shut.
        pos = settled(lambda: float(ms.state.get("gripper") or 0.0),
                      tol=0.4, timeout=1.5, dt=0.06)
        held = (GRIP_BLOCKED_PCT < pos < GRIP_JAMMED_PCT) or (held_i and pos > GRIP_AIR_PCT + 1.0 and pos < GRIP_JAMMED_PCT)
        verdict = ("HOLDING" if held else
                   "NEVER CLOSED" if pos >= GRIP_JAMMED_PCT else "EMPTY")
        tsay(f"        gripper settled at {pos:.1f} "
             f"(air closes to {GRIP_AIR_PCT:.1f}, held if it stops above "
             f"{GRIP_BLOCKED_PCT:.1f} and under {GRIP_JAMMED_PCT:.1f}) -> {verdict}"
             f"   [current said {'contact' if held_i else 'nothing'}, "
             f"idle {idle:.1f}]")

        # NEVER SAW IT, NEVER ALIGNED IT. A run whose approach could not find the cap
        # on a single step has closed the jaws wherever it happened to be standing. One
        # such run reported SUCCESS on a gripper reading of 15.1 -- it had grabbed
        # something, by luck, and there was no evidence at all that it was this tube.
        if last_dist is None:
            # ADVISORY. This refused grips the jaws plainly had (7.7, current up) and
            # opened on them -- the tube dropped straight back down. Held is held.
            tsay(f"        (the approach never saw the {colour} cap — keeping what the "
                 f"jaws hold anyway)")
        if last_dist > 2.0 * GRASP_RADIUS_PX:
            # ADVISORY ONLY. This used to refuse the grasp, and it refused real ones:
            # the jaws blocked at 7.6 with the current up, and the run was thrown away
            # because the CAP was 183px off -- the grip was on the tube's body.
            tsay(f"        (the cap was last seen {last_dist:.0f}px from the grip — "
                 f"holding the body, not the cap end)")
        if not held and pos >= GRIP_JAMMED_PCT:
            raise RuntimeError(
                f"the jaws never closed — they stopped at {pos:.1f}, nearly open "
                f"(a tube holds between {GRIP_HOLDING_PCT:.1f} and "
                f"{GRIP_JAMMED_PCT:.1f}); a false contact fired the squeeze early")
        if not held:
            raise RuntimeError(
                f"closed on nothing — the jaws settled at {pos:.1f}, at the air stop "
                f"({GRIP_AIR_PCT:.1f})")
        ms._set_carry(True, label=f"{colour} tube", h_m=TUBE_D_M)
        tphase("LIFT", "lifting clear")
        q_g = ms.observe(False)[0]
        tip = ms._tip(q_g)
        pitch_g = float(sum(q_g[i] for i in ms.ARM.pitch_chain))
        ms._move_tip(np.array([tip[0], tip[1], tip[2] + 0.09]), pitch_g, j5,
                     settle=0.20, step=1.6)

        # TWO TUBES? Wider than one tube allows, or two caps sitting at the jaws.
        _cap_now(None, None)
        jc = ms.jaw_frame().centre_uv
        at_jaws = [c for c in (ms.LAST_CAPS[0] or [])
                   if math.hypot(c["x"] - jc[0], c["y"] - jc[1]) <= TWO_CAPS_PX
                   and c.get("area", 0) >= TWO_CAPS_MIN_AREA]
        if pos >= GRIP_TWO_PCT:           # width only: the cap count gave false alarms
            tsay(f"        grabbed two? jaws at {pos:.1f} (one tube stops under "
                 f"{GRIP_TWO_PCT:.0f}), {len(at_jaws)} caps at the jaws — putting them back")
            ms._move_tip(np.array([tip[0], tip[1], tip[2] + 0.01]), pitch_g, j5,
                         settle=0.2, step=1.6)
            ms.send_joints(ms.observe(False)[0], gripper=float(ms.ARM.gripper.open_pct))
            time.sleep(0.4)
            ms._set_carry(False)
            ms._move_tip(np.array([tip[0], tip[1], tip[2] + 0.09]), pitch_g, j5,
                         settle=0.2, step=2.0)
            raise RuntimeError(f"grabbed two tubes (jaws at {pos:.1f}, "
                               f"{len(at_jaws)} caps at the jaws)")
        off = "alignment unknown" if last_dist is None else f"{last_dist:.0f}px off"
        return f"holding the {colour} tube ({off} at the close)"

    def _fold(deg, period=180.0):
        return ((float(deg) + period / 2.0) % period) - period / 2.0

    # ---- the run ---------------------------------------------------------------
    pick_ctx = {}

    def _tag_of(e):
        """A short failure tag from an exception, for the results list and the UI."""
        t = f"{type(e).__name__}: {e}".lower()
        if "stopped by user" in t or t.startswith("abort"):
            return "stopped"
        if "cap in view" in t or "not on the map" in t:
            return "not found"
        if "while standing" in t or "while carrying" in t:
            return "dropped in carry"
        if "grabbed two" in t:
            return "grabbed two"
        if "closed on nothing" in t or "never saw" in t or "never closed" in t:
            return "missed grasp"
        if "cap in the box" in t:
            return "not centred"
        if "no free hole" in t:
            return "rack full"
        if "cannot reach" in t or "cannot hover" in t or "not reachable" in t:
            return "unreachable"
        return "error"

    def _record(colour, ok, tag, detail, xy=None):
        """One line in the results list the UI shows, with its tag."""
        with lock:
            tstate.setdefault("results", []).append(
                {"t": time.strftime("%H:%M:%S"), "colour": colour, "ok": bool(ok),
                 "tag": tag, "detail": str(detail)[:160],
                 "x": None if xy is None else round(float(xy[0]), 3),
                 "y": None if xy is None else round(float(xy[1]), 3)})
            del tstate["results"][:-60]
        tsay(f"[{tag.upper()}] {colour}: {detail}")

    def _pick_and_place(colour, uv_hint, map_xy, rack_name, label):
        """One tube, start to finish. Returns (ok, tag). Never raises."""
        pick_ctx.clear()
        focus["colour"], focus["uv"], focus["xy"] = colour, None, map_xy
        if map_xy is not None:
            with lock:
                near = [e for e in tube_map.values() if e["colour"] == colour
                        and math.hypot(e["x"] - map_xy[0], e["y"] - map_xy[1]) < 0.03]
            if near and near[0].get("top_dir") is not None:
                pick_ctx["tube_dir"] = (near[0]["top_dir"], near[0]["top_angle"])
        ep = episodes.start("tube_pick", label, arm=ms.ARM.name, simulated=False,
                            dest=rack_name)
        try:
            tphase("PICK", label)
            if map_xy is not None and math.degrees(math.atan2(map_xy[1], map_xy[0]))                     < PICK_MIN_BEARING_DEG:
                raise RuntimeError(f"not reachable: the tube is past the guardrail, toward "
                                   f"the racks ({math.degrees(math.atan2(map_xy[1], map_xy[0])):+.0f}deg)")
            pick_guard[0] = True
            try:
                detail = cap_pick(colour, uv_hint, map_xy)
            finally:
                pick_guard[0] = False
            if rack_name is None:
                ok, tag, d = True, "picked", detail
            else:
                ok, tag, d = place_upright(colour, rack_name)
        except Exception as e:
            ok, tag, d = False, _tag_of(e), f"{type(e).__name__}: {e}"
            try:
                ms.send_joints(ms.observe(False)[0], gripper=float(ms.ARM.gripper.open_pct))
                ms._set_carry(False)
            except Exception:
                pass
        focus["colour"], focus["uv"], focus["xy"] = None, None, None
        episodes.end(ep, ok, f"[{tag}] {d}")
        _record(colour, ok, tag, d, map_xy)
        return ok, tag

    def _start_job(target):
        """Claim the arm and run ``target`` in the background, releasing it after."""
        if not ms.claim_arm():
            return jsonify(ok=False, error="the arm is busy"), 409
        ms.stop_flag.clear()
        with lock:
            tstate["running"] = True

        def go():
            try:
                target()
            except Exception as e:
                tphase("FAILED", f"{type(e).__name__}: {e}")
            finally:
                ms.release_arm()
                with lock:
                    tstate["running"] = False

        threading.Thread(target=go, daemon=True).start()
        return jsonify(ok=True)

    def _pan_for_bearing(q, bearing_rad):
        """The base angle whose fingertip faces ``bearing_rad``, read off the arm model.

        On this arm the base angle runs opposite to the bearing (base -33 faces +26deg),
        which is exactly the kind of sign that must not be assumed.
        """
        best, best_err = float(q[ms.ARM.pan_joint]), None
        for pan in np.arange(ms.J_LO[ms.ARM.pan_joint], ms.J_HI[ms.ARM.pan_joint], 1.0):
            qq = np.asarray(q, float).copy()
            qq[ms.ARM.pan_joint] = pan
            tp = ms._tip(qq)
            err = abs(_fold(math.degrees(math.atan2(tp[1], tp[0]) - bearing_rad), 360.0))
            if best_err is None or err < best_err:
                best, best_err = float(pan), err
        return best

    def _scan_now():
        """MAP ONCE, WITH THE WRIST CAMERA: sweep the base across the bench from the look
        pose, folding every cap seen into the map, then stop mapping. Returns the count.
        """
        with lock:
            tube_map.clear()
            next_id[0] = 1
        q = np.array(ms.HOME, np.float64)
        _go(q, settle=0.2, step=2.8)
        mapping[0] = True
        try:
            wf = ms.ARM.pitch_chain[-1]              # the wrist's own pitch joint
            wf0 = float(q[wf])
            for n, bear in enumerate(SCAN_BEARINGS_DEG):
                ms.checkpoint()
                q[ms.ARM.pan_joint] = _pan_for_bearing(q, math.radians(bear))
                # AND LOOK DOWN: tubes close to the base sit below the look pose's view.
                # Alternate the tilt order so the wrist does not swing back each stop.
                tilts = SCAN_TILTS_DEG if n % 2 == 0 else tuple(reversed(SCAN_TILTS_DEG))
                for tilt in tilts:
                    q[wf] = float(np.clip(wf0 + tilt, ms.J_LO[wf], ms.J_HI[wf]))
                    _go(q, settle=0.25, step=3.0)
                    for _ in range(4):               # a few fresh frames per stop
                        _cap_now(None, None)
                        time.sleep(0.08)
        finally:
            mapping[0] = False
        _go(np.array(ms.HOME, np.float64), settle=0.2, step=3.0)
        n = len(tubes())
        tsay(f"        wrist scan: {n} tube(s) mapped")
        return n

    # ---- routes ----------------------------------------------------------------
    _urdf = [None]

    @app.route("/urdf")
    def r_urdf():
        if _urdf[0] is None:
            try:
                _urdf[0] = [{"name": n,
                             "v": [round(float(x), 4) for x in V.ravel()],
                             "f": [int(i) for i in F.ravel()]}
                            for n, (V, F) in link_visuals(
                                ms.ARM.urdf_path, mesh_dir=ms.ARM.mesh_path).items()]
                tsay(f"3D: {sum(len(l['f']) // 3 for l in _urdf[0])} triangles of "
                     f"{ms.ARM.name}")
            except Exception as e:
                tsay(f"3D: visuals failed: {type(e).__name__}: {e}")
                _urdf[0] = []
        return jsonify(links=_urdf[0], arm=ms.ARM.name)

    @app.route("/geom")
    def r_geom():
        xf, tip = {}, None
        try:
            q = np.asarray(ms.observe(False)[0], np.float64).copy()
            q[4] += ms.WRIST_RENDER_OFFSET        # display-only, as the admin view does
            for name, T in ms.kin.get_link_transforms_chain(q):
                xf[name] = [round(float(v), 5) for v in np.asarray(T, np.float64).ravel()]
            Tg = dict(ms.kin.get_link_transforms_chain(q)).get("gripper_link")
            if Tg is not None:
                xf["moving_jaw_so101_v1_link"] = [
                    round(float(v), 5)
                    for v in (np.asarray(Tg, np.float64) @ ms.JAW_T).ravel()]
            tip = [round(float(v), 4)
                   for v in np.asarray(ms.kin.forward_kinematics(q))[:3, 3]]
        except Exception as e:
            logger.debug("tube /geom: %s", e)
        with lock:
            ph, note = tstate["phase"], tstate["note"]
        g = ms.ARM.gripper
        with ms.lock:
            grip_pct = float(ms.state.get("gripper") or g.open_pct)
        opening = float(np.clip((grip_pct - g.closed_pct) /
                                max(g.open_pct - g.closed_pct, 1e-6), 0.0, 1.0))
        caps = list(ms.LAST_CAPS[0]) if time.time() - ms.LAST_CAPS[1] < 2.0 else []
        return jsonify(arm=ms.ARM.name, simulated=False, tip=tip, xf=xf,
                       opening=opening, tubes=tubes(), racks=racks(), caps=caps,
                       phase=ph, note=note,
                       joints=[round(float(v), 2) for v in ms.observe(False)[0]],
                       joint_names=list(ms.ARM.joint_names))

    @app.route("/state")
    def r_state():
        held, detail = grip_verdict()
        with lock:
            s = {k: tstate[k] for k in ("phase", "note", "running")}
            s["log"] = list(tstate["log"])[-60:]
        with ms.lock:
            s["gripper"] = round(float(ms.state.get("gripper") or 0.0), 1)
        with lock:
            res = list(tstate.get("results", []))
        tags = {}
        for r in res:
            tags[r["tag"]] = tags.get(r["tag"], 0) + 1
        s.update(arm=ms.ARM.name, simulated=False, held=held, grip_detail=detail,
                 episodes=episodes.tally("tube_pick"), target=None, dest=None,
                 results=res[-30:], tags=tags, log=list(tstate["log"])[-150:])
        return jsonify(s)

    @app.route("/stream")
    def r_stream():
        return ms.app.view_functions["stream"]()

    @app.route("/scan", methods=["POST"])
    def r_scan():
        """Map every tube, once, with a wrist-camera sweep."""
        def job():
            tphase("SCAN", "mapping the bench once with the wrist camera")
            n = _scan_now()
            tphase("IDLE", f"{n} tube{'' if n == 1 else 's'} on the map")
        return _start_job(job)

    @app.route("/pick", methods=["POST"])
    def r_pick():
        """Pick one mapped tube; with ``rack`` set, stand it up and drop it in that rack."""
        d = request.get_json(silent=True) or request.form or {}
        found = [t for t in tubes() if str(t["id"]) == str(d.get("tube"))]
        if not found:
            return jsonify(ok=False, error=f"tube {d.get('tube')} is not on the map"), 404
        t = found[0]
        rack_name = d.get("rack") or None
        if rack_name == "auto":
            rack_name = RACK_FOR_COLOUR.get(t["colour"])
        if rack_name is not None and rack_name not in [r["name"] for r in TOP_RACKS]:
            return jsonify(ok=False, error=f"no rack called {rack_name!r}"), 400
        xy = (float(t["x"]), float(t["y"]))

        def job():
            sample_idle()
            ok, tag = _pick_and_place(t["colour"], None, xy, rack_name, t["label"])
            with lock:
                if ok and t["id"] in tube_map:
                    tube_map[t["id"]]["picked"] = True
            _go(np.array(ms.HOME, np.float64), settle=0.2, step=2.8)
            tphase("DONE" if ok else "FAILED", tag)
        return _start_job(job)

    @app.route("/pickall", methods=["POST"])
    def r_pickall():
        """Every tube into its colour's rack, from ONE wrist-camera scan.

        MAP ONCE, THEN TAP TAP. One wrist-camera sweep maps every tube; the arm then
        turns straight to each mapped bearing in order, one direction across the bench,
        without searching left and right. A spot that fails twice comes off the list.
        """
        def job():
            sample_idle()
            tphase("SCAN", "mapping the bench once with the wrist camera")
            _scan_now()
            tally, fails = {}, []
            for _ in range(40):
                ms.checkpoint()
                todo = [t for t in tubes()
                        if sum(1 for f in fails if f[0] == t["colour"] and
                               math.hypot(f[1] - t["x"], f[2] - t["y"]) < 0.03) < 2]
                if not todo:
                    break
                todo.sort(key=lambda t: math.atan2(t["y"], t["x"]))   # one direction
                t = todo[0]
                left = len(todo)
                tphase("PICK", f"{t['colour']} tube at ({t['x']*100:+.0f},{t['y']*100:+.0f})cm"
                               f" — {left} left on the mat")
                ok, tag = _pick_and_place(t["colour"], None, (t["x"], t["y"]),
                                          RACK_FOR_COLOUR.get(t["colour"]), t["label"])
                tally[tag] = tally.get(tag, 0) + 1
                if tag == "stopped":
                    break
                if not ok:
                    fails.append((t["colour"], t["x"], t["y"]))
                # ONE SCAN, AT THE START (the operator's call): each tube is taken off
                # the list once it has been tried twice or placed -- no re-scan.
                with lock:
                    if t["id"] in tube_map and (ok or sum(
                            1 for f in fails if f[0] == t["colour"] and math.hypot(
                                f[1] - t["x"], f[2] - t["y"]) < 0.03) >= 2):
                        tube_map[t["id"]]["picked"] = True
                _go(np.array(ms.HOME, np.float64), settle=0.2, step=2.8)
            summary = ", ".join(f"{v} {k}" for k, v in sorted(tally.items())) or "nothing to do"
            placed = tally.get("placed", 0) + tally.get("placed (unverified)", 0)
            tphase("DONE" if placed and placed == sum(tally.values()) else "PARTIAL", summary)
        return _start_job(job)

    @app.route("/grip", methods=["POST"])
    def r_grip():
        """Set the jaws only, arm held where it is. POST ?pct=45 (open) / ?pct=0 (close)."""
        pct = float(request.args.get("pct", 45))
        ms.send_joints(ms.observe(False)[0], gripper=pct)
        return jsonify(ok=True, pct=pct)

    @app.route("/droptest", methods=["POST"])
    def r_droptest():
        """Run ONLY the upright drop, on a tube already in the jaws. POST ?colour=blue"""
        colour = (request.args.get("colour") or "").strip()
        if colour not in RACK_FOR_COLOUR:
            return jsonify(ok=False, error=f"colour must be one of {list(RACK_FOR_COLOUR)}"), 400

        def job():
            tphase("DROPTEST", f"upright drop only, {colour} tube in the jaws")
            try:
                ok, tag, d = place_upright(colour)
            except Exception as e:
                ok, tag, d = False, _tag_of(e), f"{type(e).__name__}: {e}"
            _record(colour, ok, tag, d)
            tphase("DONE" if ok else "FAILED", tag)
        return _start_job(job)

    @app.route("/clearmap", methods=["POST"])
    def r_clearmap():
        with lock:
            tube_map.clear()
            next_id[0] = 1
            tstate["used_top_holes"] = {}
            tstate["results"] = []
        tsay("map, used holes and results cleared")
        return jsonify(ok=True)

    @app.route("/stop", methods=["POST"])
    def r_stop():
        return ms.app.view_functions["stop"]()

    @app.route("/reset", methods=["POST"])
    def r_reset():
        with lock:
            tstate["running"] = False
        tphase("IDLE", "ready")
        return jsonify(ok=True)

    @app.route("/episodes")
    def r_episodes():
        return jsonify(tally=episodes.tally("tube_pick"),
                       recent=episodes.records("tube_pick")[-25:])

    @app.route("/topstream")
    def r_topstream():
        """The top camera, zoomed to the work area, with every estimated hole marked."""
        import requests as _rq
        from flask import Response

        def gen():
            try:
                s = _rq.Session()
                s.post(ms.CAMSURV[0] + "/", data={"password": ms.CAMSURV[1]}, timeout=5)
                r = s.get(ms.CAMSURV[0] + "/stream/" + ms.CAMSURV_STREAM,
                          stream=True, timeout=10)
                buf = b""
                for chunk in r.iter_content(chunk_size=16384):
                    buf += chunk
                    a = buf.find(b"\xff\xd8")
                    b = buf.find(b"\xff\xd9", a + 2) if a != -1 else -1
                    if b == -1:
                        continue
                    img = cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8),
                                       cv2.IMREAD_COLOR)
                    buf = buf[b + 2:]
                    if img is None:
                        continue
                    for rk in TOP_RACKS:
                        bgr = tuple(int(rk["colour"][i:i + 2], 16) for i in (5, 3, 1))
                        for k, (u, v) in enumerate(rk["holes_px"]):
                            cv2.circle(img, (int(u), int(v)), 8, bgr, 1, cv2.LINE_AA)
                            cv2.putText(img, str(k), (int(u) - 4, int(v) + 3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, bgr, 1,
                                        cv2.LINE_AA)
                    if ms.TOP_ROI is not None:
                        x0, y0, x1, y1 = ms.TOP_ROI
                        img = cv2.resize(img[y0:y1, x0:x1], (1280, int(
                            1280 * (y1 - y0) / (x1 - x0))), interpolation=cv2.INTER_LINEAR)
                    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                               + jpg.tobytes() + b"\r\n")
            except Exception:
                return
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/")
    def index():
        # NO-CACHE, because a stale page is indistinguishable from a broken one. A fix to
        # the camera panel was made and the browser kept showing "this rig has no camera"
        # over a working stream, which reads as the fix not working.
        resp = send_from_directory(TUBE_UI, "tube.html")
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp

    @app.route("/ui/<path:name>")
    def ui_asset(name):
        if "/" in name or "\\" in name or name.startswith("."):
            return ("no", 404)
        return send_from_directory(TUBE_UI, name)

    # Sample the gripper's idle draw once at startup so the verdict shown before the
    # first pick means something. It is re-sampled at every pick because it drifts with
    # temperature and with whatever the jaws are already holding.
    try:
        sample_idle()
    except Exception as e:
        logger.debug("tube: could not sample idle current: %s", e)

    ms.OVERLAY_HOOKS.append(cap_overlay)

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, threaded=True,
                               use_reloader=False),
        daemon=True).start()
    ms.say(f"tube UI: http://127.0.0.1:{port}/  (SO-101, in this process)")
