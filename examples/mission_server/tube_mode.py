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
TUBE_MAX_ROLL_DEG = 20.0

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

#: How elongated the silhouette must be before its angle is allowed to move the wrist.
#: `is_confident` is 2.0, which is enough to SAY which way a tube lies; acting on it
#: deserves more. The reading that drove the 70-degree roll sat at 2.3.
TUBE_ROLL_MIN_ELONGATION = 3.0

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
    from rax.perception.tube_caps import find_caps, tube_axis
    from rax.robots.urdf_visuals import link_visuals

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

        # over the drop area first, high, so the rack comes into view
        q_over, e_over = ms._ik_hold_pitch(q_now, np.array([x, y, 0.12]),
                                           pitch_hold, j5_now, ret_err=True)
        if e_over > 0.03:
            raise RuntimeError(f"cannot carry to ({x*100:+.0f},{y*100:+.0f})cm "
                               f"(IK residual {e_over*100:.1f}cm)")
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
            sign = _probe_base(see_hole, hole_uv, aim_u)
            pt = _table_xy((target["x"], target["y"])) or (x, y)
            q_goal, e_g = ms._ik_hold_pitch(ms.observe(False)[0].astype(float),
                                            np.array([pt[0], pt[1], GRASP_Z + 0.030]),
                                            pitch_hold, j5_now, ret_err=True)
            if e_g > 0.03:
                raise RuntimeError(f"cannot reach the hole (residual {e_g*100:.1f}cm)")
            _vision_approach(see_hole, ms._clamp_joints(np.asarray(q_goal, float)),
                             sign, "the hole", hole_uv)

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
    #: Bites of forward reach taken after the interpolation ends, if the cap still is not
    #: in the cells. Small, and re-looked between each.
    CREEP_ON_STEP_M, N_CREEP_ON, CREEP_ON_MAX_M = 0.005, 14, 0.07
    #: pick.py's, and this rig's own [grip] log agrees with it: across 13 trials the
    #: target read <=67px from the grip centre on every pick that worked.
    GRASP_RADIUS_PX = 70.0

    #: Grasp pitch. Steep, because a lying tube is approached from directly above: the
    #: jaws have to straddle a 16mm cylinder, and a shallow wrist puts one finger into
    #: the table before the other reaches the far side.
    #: The hand comes STRAIGHT DOWN onto the tube: the pitch chain sums to 90 degrees,
    #: so the gripper's approach axis is perpendicular to the table.
    #:
    #: It is set ONCE, before the approach, and held for the rest of the pick. Both
    #: halves of that matter and each was got wrong separately. Holding a pitch the arm
    #: merely happened to be at (25 degrees, from the look pose) reaches for a lying tube
    #: at a slant. But CHANGING it during the approach is worse: the camera is bolted to
    #: the wrist, so tilting it sweeps the whole picture upward while the jaw cells sit at
    #: the bottom, and the cap climbs away from the jaws on every step -- measured, dy
    #: going +87 -> +285 over one descent while the horizontal error stayed inside 51px.
    #: Candidate wrist pitches, steepest first. The hand wants to come down on the tube
    #: rather than reach at it, so steeper is better -- but only while the camera can
    #: still SEE the cap, and the camera is bolted to the same wrist.
    #:
    #: 90 was tried on the operator's instruction and is too far: the wrist bends right
    #: in, and the log is unambiguous about what that costs --
    #:
    #:     base  +7: lost the cap during the probe
    #:     base -10: lost the cap during the probe
    #:     base +14: lost the cap during the probe
    #:     1-4: the blue cap not in view -- HOLDING
    #:
    #: The arm then closed blind and got something by luck. So the pitch is not a
    #: constant any more: each candidate is tried and the steepest one that leaves the
    #: cap comfortably inside the frame wins. "Comfortably" matters -- a cap clinging to
    #: the frame edge survives the tilt and is gone the moment the base moves to probe.
    GRASP_PITCH_CANDIDATES = (72.0, 60.0, 48.0, 36.0)
    #: How far a cap must sit from the frame edge to count as safely in view, px.
    PITCH_EDGE_MARGIN_PX = 60.0

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

    def _vision_approach(see, q_goal, sign_base, what, last_uv):
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
        misses, last_dist, last_ex, reached = 0, None, 0.0, False
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

            if (last_dist is not None and sign_base != 0.0
                    and abs(ex) > 30.0 and abs(ex) > abs(last_ex) + 15.0):
                sign_base *= -1.0
                tsay(f"        the error grew {abs(last_ex):.0f} -> {abs(ex):.0f}px "
                     f"— wrong way, flipping to sign {sign_base:+.0f}")
            last_ex, last_dist = ex, dist

            if (in_grip or dist < GRASP_RADIUS_PX) and a >= AT_DEPTH:
                tsay(f"        aligned to {dist:.0f}px and {100*a:.0f}% down — done")
                reached = True
                break

            q = q_start + (q_goal - q_start) * a
            q[ms.ARM.roll_joint] = q_start[ms.ARM.roll_joint]
            pan_now = ms.observe(False)[0][ms.ARM.pan_joint]
            if abs(ex) > 30 and sign_base != 0.0:
                scale = 1.0 + 1.1 * a
                d_pan = sign_base * float(np.clip(abs(ex) / (14.0 * scale),
                                                  0.4, 3.2 / scale))
                q[ms.ARM.pan_joint] = float(np.clip(
                    pan_now + d_pan, ms.J_LO[ms.ARM.pan_joint],
                    ms.J_HI[ms.ARM.pan_joint]))
            else:
                q[ms.ARM.pan_joint] = pan_now
            ms.goto_smooth(ms._clamp_joints(q), settle=0.10, step=2.2)

        if not reached and last_dist is not None:
            tsay(f"        finished {last_dist:.0f}px off — continuing anyway; the "
                 f"episode records how far")
        return reached, last_dist

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
            sign = 1.0 if ((aim_u - before_x) * gain) > 0 else -1.0
            tsay(f"        base gain {gain:+.2f}px/deg -> sign {sign:+.0f}")
            return sign
        tsay("        could not measure the base direction")
        return 0.0

    def cap_pick(colour, uv_hint):
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

        cap = _cap_now(uv_hint or None, colour)
        if cap is None:
            raise RuntimeError(f"no {colour} cap in view from the look pose")
        last_uv = [(cap["x"], cap["y"])]
        xy = _table_xy((cap["x"], cap["y"]))
        if xy is None:
            raise RuntimeError("could not cast the cap onto the table plane")
        tsay(f"        {colour} cap at ({cap['x']:.0f},{cap['y']:.0f})px -> "
             f"({xy[0]*100:+.1f},{xy[1]*100:+.1f})cm, r={math.hypot(*xy)*100:.1f}cm")

        AIM = ms.jaw_frame().centre_uv

        # ---- PITCH: straight down, set once, before anything approaches -----------
        AIM = ms.jaw_frame().centre_uv
        tphase("PITCH", "choosing the steepest wrist angle that still sees the cap")
        q_p = ms.observe(False)[0].astype(float)
        pitch_before = float(sum(q_p[i] for i in ms.ARM.pitch_chain))
        tip_p = np.asarray(ms._tip(q_p), float)
        fw, fh = ms.CAM_FRAME_WH if hasattr(ms, "CAM_FRAME_WH") else (640, 480)

        pitch_hold, c_after = None, None
        for trial in GRASP_PITCH_CANDIDATES:
            ms.checkpoint()
            q_t, e_t = ms._ik_hold_pitch(q_p, tip_p, trial, j5, ret_err=True)
            if e_t > 0.03:
                tsay(f"        {trial:+.0f}deg: unreachable here "
                     f"(residual {e_t*100:.1f}cm)")
                continue
            ms.goto_smooth(ms._clamp_joints(np.asarray(q_t, float)),
                           settle=0.25, step=2.2)
            c = _cap_now(last_uv[0], colour)
            if c is None:
                tsay(f"        {trial:+.0f}deg: the cap is out of frame")
                continue
            edge = min(c["x"], fw - c["x"], c["y"], fh - c["y"])
            if edge < PITCH_EDGE_MARGIN_PX:
                tsay(f"        {trial:+.0f}deg: the cap is only {edge:.0f}px from the "
                     f"frame edge — too close, it will be lost on the first base move")
                continue
            pitch_hold, c_after = trial, c
            tsay(f"        {trial:+.0f}deg: cap at ({c['x']:.0f},{c['y']:.0f})px, "
                 f"{edge:.0f}px clear of the edge — taking it")
            break

        if pitch_hold is None:
            raise RuntimeError(
                f"no wrist pitch between {GRASP_PITCH_CANDIDATES[-1]:.0f} and "
                f"{GRASP_PITCH_CANDIDATES[0]:.0f}deg keeps the {colour} cap in view")
        tsay(f"        wrist {pitch_before:+.0f} -> {pitch_hold:+.0f}deg, held from here")
        last_uv[0] = (c_after["x"], c_after["y"])
        cap = c_after
        xy_p = _table_xy((c_after["x"], c_after["y"]))
        if xy_p is not None:
            xy = xy_p
            tsay(f"        re-fixed looking down: ({xy[0]*100:+.1f},{xy[1]*100:+.1f})cm")

        # ---- PROBE the BASE JOINT, which is what the correction moves --------------
        # pick.py's own comment records why this has to be measured hard: "A 3-unit probe
        # moved the cap less than the +-8px noise floor, so the sign stayed a guess -- and
        # then the flip-on-worse rule toggled it at random every step, leaving dx pinned
        # at +100 for an entire descent while 'correcting'." So the threshold is 18px, not
        # 8, and the sign folds in the error direction the way pick.py computes it:
        #
        #     gain = moved / probe          sign = +1 if ex * gain > 0 else -1
        #
        # and the step then uses ABS(ex) times that sign.
        tphase("PROBE", "measuring which way the base moves the cap")
        sign_base = 0.0
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
            gain = moved / probe_q
            ex0 = AIM[0] - before_x
            sign_base = 1.0 if (ex0 * gain) > 0 else -1.0
            tsay(f"        base gain {gain:+.2f}px/deg, error {ex0:+.0f}px "
                 f"-> correcting with sign {sign_base:+.0f}")
            break
        if sign_base == 0.0:
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
            if sign_base == 0.0:
                tsay("        no measured base direction — leaving the aim alone")
                break
            q_a = ms.observe(False)[0].astype(float)
            d_pan = sign_base * float(np.clip(abs(ex) / 14.0, 1.0, 5.0))
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

        # ---- THE GRASP POSE, solved ONCE, in joint space --------------------------
        #
        # NO CARTESIAN IK INSIDE THE LOOP, and that is the whole point of this rewrite.
        # pick.py interpolates JOINT poses toward its demonstrated grasp and adds a base
        # correction on top; it never solves IK during the descent. Every version of this
        # that used _move_tip per step re-solved all five joints each increment, so the
        # camera's orientation was not determined by what was commanded -- which makes
        # the relationship between "the correction I applied" and "which way the cap
        # moved" unstable, and that is why the sign appeared to behave at random.
        #
        # The SO-101 has no demonstrated grasp pose, so one is SOLVED, once, here. After
        # this the approach is pure joint interpolation plus a base nudge, exactly as
        # pick.py does it.
        tphase("PLAN", "solving the grasp pose once, holding the wrist where it is")
        q_start = ms.observe(False)[0].astype(float)

        # THE WRIST PITCH DOES NOT CHANGE. It is read from the arm here and held for the
        # whole pick -- the approach, the grasp and the lift.
        #
        # This is the fault that produced every "it is going the opposite way". A grasp
        # pitch of 78 degrees was being commanded, tipping the wrist far over from the
        # ~25 it sits at, and THE CAMERA IS BOLTED TO THAT WRIST: tilting it sweeps the
        # entire picture upward, and the jaw cells are at the bottom of the picture. So
        # the cap climbed away from the jaws on every step while the approach was
        # supposedly closing on it. Measured, one descent:
        #
        #     dy: +87 128 165 194 214 233 264 285      away, monotonically
        #
        # Solving with an unweighted orientation was no better: it let the IK choose
        # whatever wrist angle was convenient for each target, so the pitch still moved,
        # just unpredictably. `_ik_hold_pitch` pins it.
        tsay(f"        holding the wrist at {pitch_hold:+.1f}deg for the whole pick")

        q_grasp, err = ms._ik_hold_pitch(q_start, np.array([xy[0], xy[1], GRASP_Z]),
                                         pitch_hold, j5, ret_err=True)
        q_grasp = ms._clamp_joints(np.asarray(q_grasp, float))
        if err > 0.02:
            raise RuntimeError(
                f"no grasp pose for ({xy[0]*100:+.1f},{xy[1]*100:+.1f})cm at "
                f"z={GRASP_Z*100:.1f}cm holding {pitch_hold:.0f}deg — "
                f"IK residual {err*100:.1f}cm")
        pitch_end = float(sum(q_grasp[i] for i in ms.ARM.pitch_chain))
        tsay(f"        grasp pose {np.round(q_grasp,1).tolist()}, "
             f"wrist ends at {pitch_end:+.1f}deg ({err*100:.1f}cm residual)")

        reached, last_dist = _vision_approach(
            lambda: _cap_now(last_uv[0], colour), q_grasp, sign_base,
            f"the {colour} cap", last_uv)

        # ---- GRASP ----------------------------------------------------------------
        tphase("GRASP", "closing across the tube")
        held_i, idle = ms.close_with_current(step=4.0, delay=0.08)

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
        held = pos > GRIP_HOLDING_PCT
        tsay(f"        gripper settled at {pos:.1f} "
             f"(air closes to {GRIP_AIR_PCT:.1f}, holding is above "
             f"{GRIP_HOLDING_PCT:.1f}) -> {'HOLDING' if held else 'EMPTY'}"
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
    def run(label, dest_hole, colour, uv_hint):
        ep = episodes.start("tube_pick", label, arm=ms.ARM.name, simulated=False,
                            dest=("rack hole %d" % dest_hole) if dest_hole is not None
                            else None)

        def action(n):
            tphase("PICK", f"{label}, attempt {n}")
            detail = cap_pick(colour, uv_hint)
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
        threading.Thread(target=run, args=(label, hole, colour, uv_hint),
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
