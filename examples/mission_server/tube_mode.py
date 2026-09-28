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
bench. `RACK_XY` is a CONFIGURED position, not a surveyed one, and the UI labels it
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
GRIP_AIR_PCT = 6.5
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
TWIST_TOL_DEG = 12.0

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

RACK_XY = [0.24, -0.14]
RACK_PITCH_M = 0.022
RACK_NX, RACK_NY = 3, 2


def _hole_grid(x, y):
    return [(x + (i - (RACK_NX - 1) / 2) * RACK_PITCH_M,
             y + (j - (RACK_NY - 1) / 2) * RACK_PITCH_M)
            for j in range(RACK_NY) for i in range(RACK_NX)]


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


#: Where tubes get dropped, base-frame metres. To the RIGHT of the workspace, laid out
#: in a row so several can be put down without stacking them on each other.
DROP_X, DROP_Y0, DROP_PITCH_M = 0.24, -0.13, 0.035
DROP_SLOTS = 5

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
    from rax.manipulation.attempt import with_retries
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
              "idle_current": 0.0, "used_holes": []}
    episodes = EpisodeLog(
        os.path.join(HERE, "tube_episodes.jsonl"),
        probe=lambda: {"joints": [round(float(v), 1) for v in ms.observe(False)[0]]},
        note=lambda m: tsay(m))

    # ---- caps, found every frame and drawn on the FPV overlay -------------------
    # THE BOXES THE OPERATOR ASKED FOR, and they are not only decoration: the same
    # detection is what the grid aim steers on. Registered as a hook so mission_server
    # never imports this module -- the dependency points one way.
    def _map_observe():
        """Fold the caps in view into the map. Called wherever the arm is looking."""
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
        caps = find_caps(clean)
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
        if caps:
            img[:, :] = draw_caps(img, caps)
            for f in found:
                if f["angle"] is None:
                    continue
                a = math.radians(f["angle"])
                half = f["axis_len"] / 2.0
                col = (60, 220, 90) if f["colour"] == "green" else (235, 170, 60)
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
                    col = (60, 220, 90) if f["colour"] == "green" else (235, 170, 60)
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
                    "yaw": float(e["angle"] or 0.0),
                    "yaw_known": bool(e["yaw_known"]),
                    "n": e["n"], "age": round(now - e["t"], 1),
                    "label": f"{e['colour']} tube"})
        return out

    def _colour_of(label):
        for c in ("green", "blue", "red", "gold", "yellow", "orange"):
            if c in label.lower():
                return "gold" if c in ("yellow", "orange") else c
        return "blue"

    def drop_xy(slot):
        return (DROP_X, DROP_Y0 - slot * DROP_PITCH_M)

    def place_right(colour, slot):
        """Put the held tube down on the right — BY EYE, the same way it was picked up.

        SAME STRATEGY, DIFFERENT TARGET. A cap going between the jaws and a held tube
        going over a rack hole are the same problem: something in the picture has to end
        up in the jaw cells while the arm follows a planned descent. So this runs the
        very same `_vision_approach`, with the hole detector supplying the target instead
        of the cap detector.

        The holes come from `rax.perception.rack_holes`, which finds them by Hough
        circles and decides FREE by darkness -- an empty hole looks into the rack's own
        shadow and reads near-black, while one with a tube in it shows the cap. That
        needs no colour model and does not care which cap is already in the way.

        FALLS BACK TO THE BLIND DROP, and says so. If no rack is visible the arm still
        has to put the tube down somewhere, so it goes to the configured slot -- but the
        log distinguishes the two, because one of them is a measurement and the other is
        a guess about where a rack was assumed to be.
        """
        from rax.perception.rack_holes import find_holes, pick_free_hole

        x, y = drop_xy(slot)
        tphase("CARRY", f"carrying the {colour} tube to the right")
        q_now = ms.observe(False)[0].astype(float)
        pitch_hold = float(sum(q_now[i] for i in ms.ARM.pitch_chain))
        j5_now = float(q_now[ms.ARM.roll_joint])

        # OVER THE DROP AREA FIRST, high, and at whatever radius the arm can make on
        # that bearing: a run that had the tube properly in the jaws threw "cannot carry
        # to (+24,-16)cm (IK residual 5.0cm)" and dropped a good pick on the floor of
        # the log. The bearing is the part that matters; the hole detector takes it
        # from there.
        bear_d = math.atan2(y, x)
        r_d = float(math.hypot(x, y))
        q_over = None
        while r_d > 0.12:
            q_t, e_t = ms._ik_hold_pitch(
                q_now, np.array([r_d * math.cos(bear_d), r_d * math.sin(bear_d), 0.12]),
                pitch_hold, j5_now, ret_err=True)
            if e_t <= 0.03:
                q_over = q_t
                if r_d < math.hypot(x, y) - 1e-6:
                    tsay(f"        {math.hypot(x,y)*100:.0f}cm is past the arm's reach "
                         f"holding {pitch_hold:+.0f}deg — going out to {r_d*100:.0f}cm "
                         f"on the same bearing")
                x, y = r_d * math.cos(bear_d), r_d * math.sin(bear_d)
                break
            r_d -= 0.02
        if q_over is None:
            raise RuntimeError(f"nothing on the bearing to "
                               f"({x*100:+.0f},{y*100:+.0f})cm is reachable while "
                               f"holding {pitch_hold:+.0f}deg")
        ms.goto_smooth(ms._clamp_joints(np.asarray(q_over, float)), settle=0.25, step=2.6)

        def see_hole():
            src = ms.latest_rgb[0]
            if src is None:
                return None
            bgr = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2BGR)
            holes = find_holes(bgr)
            h = pick_free_hole(holes, prefer=ms.jaw_frame().centre_uv)
            if h is None:
                return None
            return {"x": float(h.x), "y": float(h.y),
                    "bbox": (h.x - h.r, h.y - h.r, h.x + h.r, h.y + h.r)}

        ms.observe(True)
        target = see_hole()
        if target is None:
            tsay("        no free rack hole in view — putting it down at the "
                 "configured slot instead (a guess, not a measurement)")
            q_down, e_d = ms._ik_hold_pitch(ms.observe(False)[0].astype(float),
                                            np.array([x, y, GRASP_Z + 0.012]),
                                            pitch_hold, j5_now, ret_err=True)
            if e_d <= 0.03:
                ms.goto_smooth(ms._clamp_joints(np.asarray(q_down, float)),
                               settle=0.20, step=2.0)
        else:
            tphase("DROP", f"lining the tube up with a free hole, slot {slot}")
            hole_uv = [(target["x"], target["y"])]
            aim_u = ms.jaw_frame().centre_uv[0]
            gain_hole = _probe_base(see_hole, hole_uv, aim_u)
            pt = _table_xy((target["x"], target["y"])) or (x, y)
            q_goal, e_g = ms._ik_hold_pitch(ms.observe(False)[0].astype(float),
                                            np.array([pt[0], pt[1], GRASP_Z + 0.030]),
                                            pitch_hold, j5_now, ret_err=True)
            if e_g > 0.03:
                raise RuntimeError(f"cannot reach the hole (residual {e_g*100:.1f}cm)")
            _vision_approach(see_hole, ms._clamp_joints(np.asarray(q_goal, float)),
                             gain_hole, "the hole", hole_uv)

        ms.send_joints(ms.observe(False)[0], gripper=float(ms.ARM.gripper.place_open_pct))
        time.sleep(0.4)
        ms._set_carry(False)
        ms.goto_smooth(ms._clamp_joints(np.asarray(q_over, float)), settle=0.20, step=2.6)
        tsay(f"        released over slot {slot}")
        return f"dropped at slot {slot}"

    def racks():
        """The drop row, drawn as 'holes' so the existing viewer shows the slots."""
        return [{"name": "drop", "x": DROP_X,
                 "y": DROP_Y0 - (DROP_SLOTS - 1) * DROP_PITCH_M / 2.0,
                 "yaw": 0.0, "colour": "#8a93a0",
                 "holes": [[round(drop_xy(i)[0], 4), round(drop_xy(i)[1], 4)]
                           for i in range(DROP_SLOTS)]}]


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
                    ms.goto_smooth(ms._clamp_joints(q_fin), settle=0.20, step=2.0)
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
            ms.goto_smooth(ms._clamp_joints(q), settle=0.10, step=2.2)

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
            ms.goto_smooth(ms._clamp_joints(q_try), settle=0.20, step=2.6)
            after = see()
            ms.goto_smooth(ms._clamp_joints(q_from), settle=0.20, step=2.6)
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
        ms.goto_smooth(ms._clamp_joints(np.array(ms.HOME, np.float64)),
                       settle=0.20, step=2.8)
        j5 = float(ms.observe(False)[0][ms.ARM.roll_joint])

        # OPEN THE JAWS NOW, before anything approaches, so the hand arrives ready
        # instead of travelling in with the fingers wherever the last run left them.
        ms.send_joints(ms.observe(False)[0], gripper=GRIP_PREOPEN_PCT)
        time.sleep(0.2)
        tsay(f"        jaws open to {GRIP_PREOPEN_PCT:.0f} for the approach")

        cap = _cap_now(uv_hint or None, colour)
        if cap is None and map_xy is not None:
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
            tip_l = ms._tip(q_l)
            pitch_l = float(sum(q_l[i] for i in ms.ARM.pitch_chain))
            bear_m = math.atan2(map_xy[1], map_xy[0])
            r_m = min(float(math.hypot(map_xy[0], map_xy[1])), 0.20)
            q_a, e_a = ms._ik_hold_pitch(
                q_l, np.array([r_m * math.cos(bear_m), r_m * math.sin(bear_m),
                               float(tip_l[2])]), pitch_l, j5, ret_err=True)
            if e_a > 0.05:
                tsay(f"        the map puts it at {math.degrees(bear_m):+.0f}deg but "
                     f"no pose faces there — sweeping from here instead")
                pan_want = here
            else:
                pan_want = float(q_a[ms.ARM.pan_joint])
            for extra in (0.0, -18.0, 18.0, -36.0, 36.0):
                ms.checkpoint()
                want = float(np.clip(pan_want + extra,
                                     ms.J_LO[ms.ARM.pan_joint],
                                     ms.J_HI[ms.ARM.pan_joint]))
                q_l[ms.ARM.pan_joint] = want
                ms.goto_smooth(ms._clamp_joints(q_l), settle=0.30, step=2.8)
                cap = _cap_now(None, colour)
                tsay(f"        base {want:+.0f}deg "
                     f"(map bearing {math.degrees(bear_m):+.0f}deg): "
                     + ("found it" if cap else "nothing"))
                if cap is not None:
                    break
            if cap is None:
                q_l[ms.ARM.pan_joint] = here
                ms.goto_smooth(ms._clamp_joints(q_l), settle=0.25, step=2.8)
        if cap is None:
            raise RuntimeError(f"no {colour} cap in view from the look pose")
        last_uv = [(cap["x"], cap["y"])]
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
        for probe_q in (7.0, -10.0, 14.0):
            ms.checkpoint()
            before_x = cap["x"]
            q_from = ms.observe(False)[0].astype(float)
            q_try = q_from.copy()
            q_try[ms.ARM.pan_joint] = float(np.clip(
                q_from[ms.ARM.pan_joint] + probe_q,
                ms.J_LO[ms.ARM.pan_joint], ms.J_HI[ms.ARM.pan_joint]))
            ms.goto_smooth(ms._clamp_joints(q_try), settle=0.20, step=2.6)
            after = _cap_now(last_uv[0], colour)
            ms.goto_smooth(ms._clamp_joints(q_from), settle=0.20, step=2.6)
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
            tsay("        could not measure the base direction — descending without "
                 "a horizontal correction")
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
        for k in range(AIM_STEPS):
            ms.checkpoint()
            c = _cap_now(last_uv[0], colour)
            if c is None:
                tsay(f"        aim {k+1}: cap not in view — holding")
                time.sleep(0.15)
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
            ms.goto_smooth(ms._clamp_joints(q_a), settle=0.18, step=2.6)
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
                ms.goto_smooth(ms._clamp_joints(q_s), settle=0.12, step=2.0)
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
                    tsay(f"        approach {k+1}: {(r_now+bite)*100:.1f}cm is past the "
                         f"arm at {pitch_see:+.0f}deg — going no further out")
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
            ms.goto_smooth(ms._clamp_joints(q_s), settle=0.14, step=2.2)

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
        def _down_to(z_want, pitch_now, what):
            """Straight down to ``z_want`` at fixed x and y. Returns the height reached."""
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
                q_d, e_d = ms._ik_hold_pitch(
                    q_n, np.array([tip0[0], tip0[1], z_k]), pitch_now, j5, ret_err=True)
                if e_d > 0.02:
                    tsay(f"        {what} {k+1}: z={z_k*100:+.1f}cm not reachable "
                         f"(residual {e_d*100:.1f}cm) — stopping here")
                    break
                ms.goto_smooth(ms._clamp_joints(np.asarray(q_d, float)),
                               settle=0.12, step=1.8)
            z_end = float(ms._tip(ms.observe(False)[0])[2])
            tsay(f"        at {z_end*100:+.1f}cm")
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
                    ms.goto_smooth(ms._clamp_joints(np.asarray(q_t, float)),
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
            ms.goto_smooth(ms._clamp_joints(q_set), settle=0.28, step=2.2)

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

        # ---- TWIST: close ACROSS the tube, not along it ---------------------------
        #
        # A parallel gripper has to meet a cylinder side-on. If the tube lies at an angle
        # to the jaw line the fingers hit it at a tangent, roll it away, and close on the
        # gap where it used to be.
        #
        # The detector already measures the axis -- `tube_axis` fits the body's
        # silhouette and every cap in LAST_CAPS carries its `angle` and `elong`. So the
        # error is the tube's angle against the jaw line plus a right angle, and the
        # wrist roll turns through it. On this arm the camera rides past the roll joint,
        # so rolling turns the picture as well: `JawFrame.roll_gain` is the measured
        # degrees of image rotation per degree of roll (+1.00 here), which is what makes
        # the correction a division rather than a guess.
        #
        # Done BEFORE the final reach, while the jaws are still clear of the tube --
        # turning the wrist with the fingers already around it would sweep it aside.
        def _axis_now():
            """The tube's angle in the picture: median of three reads, or None.

            THREE READS, NOT ONE. A single read produced "rolled -4.5deg -> 70deg out
            of square": the angle was noise, the gain computed from it was nonsense, and
            the wrist moved on both.

            AND A LOWER ELONGATION GATE THAN A DISTANT TUBE NEEDS. Standing over the
            tube the camera sees it foreshortened -- measured 1.9, 2.0, 2.1 at exactly
            the moment the twist matters -- so a threshold of 3.0 skipped the stage run
            after run in silence. The spread check carries the weight instead.
            """
            vals, last = [], None
            for _ in range(3):
                c = _cap_now(last_uv[0], colour)
                last = c if c is not None else last
                if c is None or c.get("angle") is None:
                    continue
                if float(c.get("elong") or 0.0) < TWIST_MIN_ELONGATION:
                    continue
                vals.append(float(c["angle"]))
                time.sleep(0.05)
            if not vals:
                return None, last
            base = vals[0]
            folded = [base + fold(v - base) for v in vals]
            spread = float(max(folded) - min(folded))
            if len(vals) >= 2 and spread > TWIST_MAX_SPREAD_DEG:
                tsay(f"        the tube's angle reads {spread:.0f}deg apart across "
                     f"{len(vals)} looks — too unsteady to roll on")
                return None, last
            return fold(float(np.median(folded))), last

        a0, c_t = _axis_now()
        if a0 is None:
            el = None if c_t is None else c_t.get("elong")
            tsay(f"        no usable tube axis (elongation "
                 f"{'none' if el is None else format(el, '.1f')}, needs "
                 f"{TWIST_MIN_ELONGATION:.1f}) — leaving the wrist where it is")
        else:
            jaw_sq = float(ms.jaw_frame().axis_deg) + 90.0
            err = fold(a0 - jaw_sq)
            if abs(err) <= TWIST_TOL_DEG:
                tsay(f"        tube is already square to the jaws ({abs(err):.0f}deg)")
            else:
                tphase("TWIST", "turning the wrist so the jaws close across the tube")
                tsay(f"        tube lies at {a0:+.0f}deg, square to the jaws is "
                     f"{fold(jaw_sq):+.0f}deg — {err:+.0f}deg out")

                # MEASURE WHICH WAY THE ROLL TURNS THE PICTURE. The stored roll_gain is
                # +1.00 and acting on it turned the tube the wrong way on this rig, so
                # the direction is taken from the arm: roll a known amount, watch the
                # angle, divide.
                q_w0 = ms.observe(False)[0].astype(float)
                q_probe = q_w0.copy()
                q_probe[ms.ARM.roll_joint] = float(np.clip(
                    q_w0[ms.ARM.roll_joint] + TWIST_PROBE_DEG,
                    ms.J_LO[ms.ARM.roll_joint], ms.J_HI[ms.ARM.roll_joint]))
                moved = q_probe[ms.ARM.roll_joint] - q_w0[ms.ARM.roll_joint]
                ms.goto_smooth(ms._clamp_joints(q_probe), settle=0.25, step=2.4)
                a1, _c1 = _axis_now()
                if a1 is None or abs(moved) < 1.0:
                    tsay("        lost the tube axis during the probe — rolling back")
                    ms.goto_smooth(ms._clamp_joints(q_w0), settle=0.25, step=2.4)
                else:
                    gain = fold(a1 - a0) / moved
                    tsay(f"        roll {moved:+.0f}deg turned the tube "
                         f"{fold(a1 - a0):+.0f}deg -> gain {gain:+.2f}deg/deg")
                    if abs(gain) < 0.3:
                        tsay("        the roll barely turns the picture — leaving the "
                             "wrist alone")
                        ms.goto_smooth(ms._clamp_joints(q_w0), settle=0.25, step=2.4)
                    else:
                        err1 = fold(a1 - jaw_sq)
                        # MINUS. The measured gain is d(tube angle)/d(roll), so to
                        # remove an error of err1 the wrist turns -err1/gain. The
                        # opposite sign was tried on the operator's report and turned
                        # the wrist the wrong way; this is the direction that was right
                        # before. The check below undoes it either way if it does not
                        # actually improve the squareness.
                        d_roll = float(np.clip(-err1 / gain, -TUBE_MAX_ROLL_DEG,
                                               TUBE_MAX_ROLL_DEG))
                        q_w = ms.observe(False)[0].astype(float)
                        q_w[ms.ARM.roll_joint] = float(np.clip(
                            q_w[ms.ARM.roll_joint] + d_roll,
                            ms.J_LO[ms.ARM.roll_joint], ms.J_HI[ms.ARM.roll_joint]))
                        ms.goto_smooth(ms._clamp_joints(q_w), settle=0.25, step=2.4)
                        a2, c2 = _axis_now()
                        if a2 is not None:
                            e2 = fold(a2 - (float(ms.jaw_frame().axis_deg) + 90.0))
                            tsay(f"        rolled {d_roll:+.1f}deg -> {abs(e2):.0f}deg "
                                 f"out of square (was {abs(err1):.0f}deg)")
                            if c2 is not None:
                                last_uv[0] = (c2["x"], c2["y"])
                            if abs(e2) > abs(err1) + 5.0:
                                tsay("        that is worse — rolling back")
                                ms.goto_smooth(ms._clamp_joints(q_w0),
                                               settle=0.25, step=2.4)
                        j5 = float(ms.observe(False)[0][ms.ARM.roll_joint])

        if _grasp_uv() is not None:
            tphase("TRIM", "re-checking the aim after the twist")
            _s2, d2 = _close_in(pitch_hold, RETRIM_MAX_M, N_RETRIM, what="re-trim")
            if d2 is not None:
                last_dist = d2

        # ---- DOWN THE LAST BIT ---------------------------------------------------
        tphase("DOWN", "the last few centimetres onto the tube")
        _down_to(GRASP_Z, pitch_hold, "down")

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
        held = GRIP_HOLDING_PCT < pos < GRIP_JAMMED_PCT
        verdict = ("HOLDING" if held else
                   "NEVER CLOSED" if pos >= GRIP_JAMMED_PCT else "EMPTY")
        tsay(f"        gripper settled at {pos:.1f} "
             f"(air closes to {GRIP_AIR_PCT:.1f}, a tube holds between "
             f"{GRIP_HOLDING_PCT:.1f} and {GRIP_JAMMED_PCT:.1f}) -> {verdict}"
             f"   [current said {'contact' if held_i else 'nothing'}, "
             f"idle {idle:.1f}]")

        # NEVER SAW IT, NEVER ALIGNED IT. A run whose approach could not find the cap
        # on a single step has closed the jaws wherever it happened to be standing. One
        # such run reported SUCCESS on a gripper reading of 15.1 -- it had grabbed
        # something, by luck, and there was no evidence at all that it was this tube.
        if last_dist is None:
            raise RuntimeError(
                f"the approach never saw the {colour} cap, so the jaws closed on "
                f"whatever was in front of them — not claiming this")
        if last_dist > 2.0 * GRASP_RADIUS_PX:
            raise RuntimeError(
                f"the jaws closed, but the cap was last seen {last_dist:.0f}px away — "
                f"further than {2.0*GRASP_RADIUS_PX:.0f}px, so whatever is between them "
                f"is not this tube")
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
        tip = ms._tip(ms.observe(False)[0])
        ms._move_tip(np.array([tip[0], tip[1], tip[2] + 0.09]), pitch_hold, j5,
                     settle=0.20, step=1.6)
        off = "alignment unknown" if last_dist is None else f"{last_dist:.0f}px off"
        return f"holding the {colour} tube ({off} at the close)"

    def _fold(deg, period=180.0):
        return ((float(deg) + period / 2.0) % period) - period / 2.0

    # ---- the run ---------------------------------------------------------------
    def run(label, dest_hole, colour, uv_hint, map_xy=None):
        ep = episodes.start("tube_pick", label, arm=ms.ARM.name, simulated=False,
                            dest=("rack hole %d" % dest_hole) if dest_hole is not None
                            else None)

        def action(n):
            tphase("PICK", f"{label}, attempt {n}")
            detail = cap_pick(colour, uv_hint, map_xy)
            if "nothing" in detail:
                raise RuntimeError(detail)
            if dest_hole is None:
                return detail
            hx, hy = _hole_grid(*RACK_XY)[dest_hole]
            tphase("PLACE", f"into rack hole {dest_hole} at ({hx:.3f}, {hy:.3f})")
            ms.place_at(xy=(hx, hy))
            with lock:
                if dest_hole not in tstate["used_holes"]:
                    tstate["used_holes"].append(dest_hole)
            return f"placed at rack hole {dest_hole}"

        def verify():
            held, detail = grip_verdict()
            if dest_hole is None:
                return held, detail
            # A PLACE IS VERIFIED BY THE JAWS BEING EMPTY, which is the opposite of a
            # pick — the tube is supposed to be in the rack. Checking "held" here is the
            # bug the tube server's first end-to-end run walked into.
            return (not held), (f"released over rack hole {dest_hole}" if not held
                                else f"still holding — the release did not happen: {detail}")

        def between(n):
            tphase("RETRY", f"attempt {n}: back to the look pose and re-acquire")
            try:
                ms.goto_smooth(ms._clamp_joints(np.array(ms.HOME, np.float64)),
                               settle=0.3, step=1.5)
            except Exception as e:
                tsay(f"  (could not re-home: {type(e).__name__}: {e})")

        try:
            out = with_retries(action, verify, tries=3, between=between,
                               settle_s=ms.GRASP_CHECK_WAIT_S, poll_s=0.25,
                               record=lambda n, ok, d: episodes.attempt(ep, n, ok, d),
                               checkpoint=ms.checkpoint)
            episodes.end(ep, out.ok, out.detail)
            tphase("DONE" if out.ok else "FAILED", out.detail)
        except Exception as e:
            episodes.end(ep, False, f"{type(e).__name__}: {e}")
            tphase("FAILED", f"{type(e).__name__}: {e}")
        finally:
            # attempt_pick re-arms state["running"] on its way out, so the latch is ours
            # to drop. Leaving it set makes the whole server look busy forever.
            ms.release_arm()
            with lock:
                tstate["running"] = False

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
        s.update(arm=ms.ARM.name, simulated=False, held=held, grip_detail=detail,
                 episodes=episodes.tally("tube_pick"), target=None, dest=None)
        return jsonify(s)

    @app.route("/stream")
    def r_stream():
        return ms.app.view_functions["stream"]()

    @app.route("/scan", methods=["POST"])
    def r_scan():
        """Point the detector at tubes and sweep for them."""
        def go():
            tphase("SCAN", f"looking for tubes — query '{TUBE_QUERY}'")
            try:
                ms._apply_query_now(TUBE_QUERY)
                # broad=False IS THE POINT. A broad scan replaces the query with the
                # whole tabletop vocabulary, which is how the first tube scan here mapped
                # ten objects and not one tube: it found them, then had to name them out
                # of a 29-word list that contained no tube, so three came back
                # "toothbrush" and three "pen". The label picks the size prior, and
                # toothbrush's is 3.3x a tube's, so every one of them would have been
                # ranged far past where it actually was. A tube run scans for tubes.
                ms.scan_2d(broad=False)
                n = len(tubes())
                tphase("IDLE", f"{n} tube{'' if n == 1 else 's'} on the map")
            except Exception as e:
                tphase("FAILED", f"scan: {type(e).__name__}: {e}")
            finally:
                ms.release_arm()
                with lock:
                    tstate["running"] = False
        if not ms.claim_arm():
            return jsonify(ok=False, error="the arm is busy"), 409
        ms.stop_flag.clear()
        with lock:
            tstate["running"] = True
        threading.Thread(target=go, daemon=True).start()
        return jsonify(ok=True)

    @app.route("/pick", methods=["POST"])
    def r_pick():
        d = request.get_json(silent=True) or request.form or {}
        tag = d.get("tube")
        hole = d.get("hole")
        hole = int(hole) if hole not in (None, "", "null") else None
        found = [t for t in tubes() if str(t["id"]) == str(tag)]
        if not found:
            return jsonify(ok=False, error=f"tube {tag} is not on the map"), 404
        label = found[0]["label"]
        # CLAIM THE ARM THE WAY /start DOES, and clear the stop flag the way run_mission
        # does. Neither was done in the first version and both bit:
        #
        #  * without the claim, `state["running"]` was left latched True after the run --
        #    the server then looked permanently busy to every other route.
        #  * without clearing the stop flag, a pick inherits whatever set it last. The
        #    first real tube pick died with "Abort: stopped by user" seconds in, because
        #    _guest_cleanup sets the flag for 0.4s when a guest session expires and the
        #    guest tunnel is live on this rig.
        if not ms.claim_arm():
            return jsonify(ok=False, error="the arm is busy"), 409
        ms.stop_flag.clear()
        with lock:
            tstate["running"] = True
        sample_idle()
        colour = found[0]["colour"]
        uv_hint = tuple(found[0].get("uv") or (0, 0))
        mx, my = found[0].get("x"), found[0].get("y")
        map_xy = None if mx is None or my is None else (float(mx), float(my))
        threading.Thread(target=run, args=(label, hole, colour, uv_hint, map_xy),
                         daemon=True).start()
        return jsonify(ok=True, label=label, colour=colour, hole=hole)

    @app.route("/pickall", methods=["POST"])
    def r_pickall():
        """Pick every tube on the map and put them down in the drop row on the right.

        WORKS OFF THE MAP, NOT OFF THE VIEW, which is the whole reason the map exists.
        Approaching the first tube points the camera at it and away from the others; a
        list rebuilt from the current frame would lose them exactly when it needed them.
        The map remembers roughly where each one was, the arm goes back up to the look
        pose between tubes, and the remembered position is good enough to start the next
        approach -- which then re-measures anyway.
        """
        if not ms.claim_arm():
            return jsonify(ok=False, error="the arm is busy"), 409
        ms.stop_flag.clear()
        with lock:
            tstate["running"] = True

        def go():
            done, failed = 0, 0
            try:
                # One scan first, from the look pose, so the map has everything before
                # the arm starts moving and changing what it can see.
                tphase("SCAN", "looking over the bench before starting")
                ms.goto_smooth(ms._clamp_joints(np.array(ms.HOME, np.float64)),
                               settle=0.25, step=2.8)
                for _ in range(6):
                    ms.checkpoint()
                    _cap_now(None, None)
                    time.sleep(0.12)
                todo = [t for t in tubes()]
                names = ", ".join("#%d %s" % (t["id"], t["colour"]) for t in todo)
                tsay(f"        {len(todo)} tube(s) on the map: {names}")
                for slot, t in enumerate(todo):
                    ms.checkpoint()
                    if slot >= DROP_SLOTS:
                        tsay(f"        the drop row holds {DROP_SLOTS}; stopping there")
                        break
                    tphase("PICK", f"tube #{t['id']} ({t['colour']}), "
                                   f"{slot+1} of {len(todo)}")
                    ep = episodes.start("tube_pick", f"{t['colour']} tube",
                                        arm=ms.ARM.name, simulated=False, slot=slot)
                    try:
                        detail = cap_pick(t["colour"], tuple(t.get("uv") or ()) or None)
                        detail = place_right(t["colour"], slot)
                        with lock:
                            if t["id"] in tube_map:
                                tube_map[t["id"]]["picked"] = True
                        done += 1
                        episodes.end(ep, True, detail)
                    except Exception as e:
                        failed += 1
                        tsay(f"        tube #{t['id']}: {type(e).__name__}: {e}")
                        episodes.end(ep, False, f"{type(e).__name__}: {e}")
                        try:
                            ms.send_joints(ms.observe(False)[0],
                                           gripper=float(ms.ARM.gripper.open_pct))
                        except Exception:
                            pass
                    # back up to the look pose before the next one, so the map's
                    # remembered position is approached from the same vantage it was
                    # measured from.
                    ms.goto_smooth(ms._clamp_joints(np.array(ms.HOME, np.float64)),
                                   settle=0.20, step=2.8)
                tphase("DONE" if failed == 0 else "PARTIAL",
                       f"{done} placed, {failed} failed")
            except Exception as e:
                tphase("FAILED", f"{type(e).__name__}: {e}")
            finally:
                ms.release_arm()
                with lock:
                    tstate["running"] = False

        threading.Thread(target=go, daemon=True).start()
        return jsonify(ok=True)

    @app.route("/clearmap", methods=["POST"])
    def r_clearmap():
        with lock:
            tube_map.clear()
            next_id[0] = 1
        tsay("map cleared")
        return jsonify(ok=True)

    @app.route("/rack", methods=["POST"])
    def r_rack():
        d = request.get_json(silent=True) or request.form or {}
        RACK_XY[0], RACK_XY[1] = float(d.get("x", RACK_XY[0])), float(d.get("y", RACK_XY[1]))
        with lock:
            tstate["used_holes"] = []
        tsay(f"rack moved to ({RACK_XY[0]:.3f}, {RACK_XY[1]:.3f}) — configured, not measured")
        return jsonify(ok=True, x=RACK_XY[0], y=RACK_XY[1])

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
