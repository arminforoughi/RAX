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


def start(ms, port: int = 8486) -> None:
    """Serve the tube UI on ``port``, backed by mission-server module ``ms``."""
    from rax.manipulation.attempt import with_retries
    from rax.manipulation.episodes import EpisodeLog
    from rax.manipulation.grip import CurrentRise, reconcile
    from rax.perception.tube_caps import draw as draw_caps
    from rax.perception.tube_caps import find_caps, tube_axis
    from rax.robots.urdf_visuals import link_visuals

    app = Flask("tube_mode", static_folder=None)
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
        """Every tube in the CURRENT VIEW, mapped onto the table plane.

        NO YOLO, AND NO SWEEP. The detector's part in this is over: caps are found by
        colour (measured on this camera), the body's axis by contrast, and the position
        by casting the cap's pixel onto the table plane through the calibrated hand-eye
        transform -- which reprojects the fingertip to 0 px on this rig, so the cast is
        as good as the plane height.

        That replaces a 29-class vocabulary, a prompt, and a base sweep with one frame.
        It is also the only version that has worked here: pointed at tubes, the broad
        scan mapped ten objects and no tubes (three "toothbrush", three "pen"), and the
        targeted scan found one of the two tubes on the bench. The cap detector finds
        both, every frame.

        WHAT IS STILL HONEST ABOUT IT. Only what the camera can see right now is
        listed -- there is no memory and nothing is carried over, so a tube that leaves
        the view leaves the map rather than lingering as a stale fix. And `yaw_known`
        follows the axis measurement's own confidence: a tube seen end-on is barely
        elongated and its angle means little, so it is reported as unknown rather than
        as whatever the fit returned.
        """
        caps, t = ms.LAST_CAPS[0], ms.LAST_CAPS[1]
        if not caps or time.time() - t > 2.5:
            return []
        try:
            q = ms.observe(False)[0]
            T = ms.T_cam_of(q)
        except Exception:
            return []
        with ms.lock:
            held_label = ms.carry["label"] if ms.carry["held"] else None

        out = []
        for i, c in enumerate(caps, start=1):
            try:
                p = ms.ray_to_table((c["x"], c["y"]), T)
            except Exception:
                continue
            if p is None:
                continue
            x, y = float(p[0]), float(p[1])
            r = math.hypot(x, y)
            # A cast that lands outside the arm's own workspace is a broken solve, not a
            # distant tube -- the same plausibility bound the rest of the stack uses.
            if not (ms.ARM.reach_min_m <= r <= ms.ARM.reach_max_m):
                continue
            held = held_label is not None and c["colour"] in str(held_label)
            out.append({"id": i, "colour": c["colour"],
                        "x": round(x, 4), "y": round(y, 4), "z": 0.0,
                        "held": bool(held), "rack": None, "hole": None,
                        "source": "seen",
                        "d": TUBE_D_M, "l": TUBE_L_M,
                        # THREE ANSWERS, from the tube's own silhouette. A confident
                        # elongated body is a tube LYING at a measured angle. No
                        # elongated body is UNKNOWN, not "standing": a tube upright in a
                        # rack and one pointing end-on at the camera present the same
                        # circle, and nothing here can tell them apart. Reporting either
                        # as a fact is the lie the object map is careful not to tell
                        # with its own `yaw_known`.
                        "standing": False if c["confident"] else None,
                        "yaw": float(c["angle"] or 0.0),
                        "yaw_known": bool(c["confident"]),
                        "uv": [round(c["x"], 1), round(c["y"], 1)],
                        "label": f"{c['colour']} tube"})
        return out

    def _colour_of(label):
        for c in ("green", "blue", "red", "gold", "yellow", "orange"):
            if c in label.lower():
                return "gold" if c in ("yellow", "orange") else c
        return "blue"

    def racks():
        holes = _hole_grid(*RACK_XY)
        return [{"name": "tube", "x": RACK_XY[0], "y": RACK_XY[1], "yaw": 0.0,
                 "colour": "#8a93a0",
                 "holes": [[round(h[0], 4), round(h[1], 4)] for h in holes]}]


    # ---- the pick: look, go over, square up, put the cap in the grid --------------
    #: Fingertip height at the grasp. A tube lying down puts its centre one radius up,
    #: and the jaws close AROUND it, so the tip wants to arrive level with that centre
    #: rather than on the table. 8mm is the tube's own radius.
    GRASP_Z = 0.010
    #: How many increments the creep takes. Twelve at ~1cm each is a walk, not a dive,
    #: and every one of them re-measures the cap first.
    #: pick.py aims to 150px before approaching. Same here: the point is only to get the
    #: tube off the edge of the picture, not to line it up -- the approach does that.
    AIM_TOL_PX, AIM_STEPS = 150.0, 8
    #: Increments in the approach. Each one looks first.
    N_APPROACH = 10
    #: pick.py's, and this rig's own [grip] log agrees with it: across 13 trials the
    #: target read <=67px from the grip centre on every pick that worked.
    GRASP_RADIUS_PX = 70.0

    #: Grasp pitch. Steep, because a lying tube is approached from directly above: the
    #: jaws have to straddle a 16mm cylinder, and a shallow wrist puts one finger into
    #: the table before the other reaches the far side.
    GRASP_PITCH = 78.0

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
            time.sleep(0.15)
        return None

    def _table_xy(uv):
        """Cast a pixel onto the table plane -> base-frame (x, y), or None."""
        try:
            pt = ms.ray_to_table((float(uv[0]), float(uv[1])),
                                 ms.T_cam_of(ms.observe(False)[0]))
        except Exception:
            return None
        return None if pt is None else (float(pt[0]), float(pt[1]))

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
                       settle=0.30, step=2.0)
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

        # ---- PROBE: which way does the base move the cap? -------------------------
        tphase("PROBE", "measuring which way the base moves the cap")
        sign_base = 0.0
        q0 = ms.observe(False)[0].astype(float)
        u0 = cap["x"]
        for nudge in (4.0, -7.0, 10.0):
            ms.checkpoint()
            q = q0.copy()
            q[ms.ARM.pan_joint] = float(np.clip(q0[ms.ARM.pan_joint] + nudge,
                                                ms.J_LO[ms.ARM.pan_joint],
                                                ms.J_HI[ms.ARM.pan_joint]))
            ms.goto_smooth(ms._clamp_joints(q), settle=0.25, step=2.0)
            c = _cap_now(last_uv[0], colour)
            if c is not None and abs(c["x"] - u0) >= 8.0:
                moved = c["x"] - u0
                sign_base = 1.0 if (moved / nudge) > 0 else -1.0
                tsay(f"        base {nudge:+.0f}deg moved the cap {moved:+.0f}px "
                     f"-> sign {sign_base:+.0f}")
                break
            tsay(f"        base {nudge:+.0f}deg moved the cap under the noise floor")
        ms.goto_smooth(ms._clamp_joints(q0), settle=0.25, step=2.0)
        if sign_base == 0.0:
            raise RuntimeError("could not measure which way the base moves the cap")

        # ---- AIM: base only, until the cap is near the jaws horizontally ----------
        tphase("AIM", "turning the base to bring the cap across")
        for n in range(AIM_STEPS):
            ms.checkpoint()
            c = _cap_now(last_uv[0], colour)
            if c is None:
                tsay(f"        aim {n+1}: cap not in view — holding")
                time.sleep(0.25)
                continue
            last_uv[0] = (c["x"], c["y"])
            ex = AIM[0] - c["x"]
            tsay(f"        aim {n+1}: cap x={c['x']:.0f} vs jaws x={AIM[0]:.0f} "
                 f"-> dx {ex:+.0f}px")
            if abs(ex) < AIM_TOL_PX:
                tsay(f"        aimed, {abs(ex):.0f}px < {AIM_TOL_PX:.0f}px")
                break
            q = ms.observe(False)[0].astype(float)
            d_pan = math.copysign(min(abs(ex) / 14.0, 6.0), ex * sign_base)
            q[ms.ARM.pan_joint] = float(np.clip(q[ms.ARM.pan_joint] + d_pan,
                                                ms.J_LO[ms.ARM.pan_joint],
                                                ms.J_HI[ms.ARM.pan_joint]))
            ms.goto_smooth(ms._clamp_joints(q), settle=0.20, step=2.0)

        # Re-fix now that the tube is in front of the camera rather than off to one side.
        c = _cap_now(last_uv[0], colour)
        if c is not None:
            xy2 = _table_xy((c["x"], c["y"]))
            if xy2 is not None:
                tsay(f"        re-fixed after aiming: ({xy2[0]*100:+.1f},"
                     f"{xy2[1]*100:+.1f})cm, r={math.hypot(*xy2)*100:.1f}cm")
                xy, cap = xy2, c
                last_uv[0] = (c["x"], c["y"])

        # ---- SQUARE: a nudge, only on a confident axis ----------------------------
        if TUBE_YAW_ALIGN and cap.get("confident") and \
                (cap.get("elong") or 0.0) >= TUBE_ROLL_MIN_ELONGATION:
            jg = ms.jaw_frame()
            err = _fold(cap["angle"] - (jg.axis_deg + 90.0))
            gain = float(getattr(jg, "roll_gain", 1.0)) or 1.0
            if abs(err) >= float(ms.CFG.yaw_deadband_deg):
                tphase("SQUARE", "turning the jaws across the tube")
                step = float(np.clip(err / gain, -TUBE_MAX_ROLL_DEG, TUBE_MAX_ROLL_DEG))
                lo, hi = ms.J_LO[ms.ARM.roll_joint], ms.J_HI[ms.ARM.roll_joint]
                j5 = float(np.clip(j5 + step, lo, hi))
                tsay(f"        tube {cap['angle']:+.0f}deg, jaws {jg.axis_deg:+.0f}deg "
                     f"-> rolling {step:+.0f}deg")
                q = ms.observe(False)[0].astype(float)
                q[ms.ARM.roll_joint] = j5
                ms.goto_smooth(ms._clamp_joints(q), settle=0.30, step=2.0)

        # ---- APPROACH: move toward it, looking before every step ------------------
        tphase("APPROACH", "moving toward the tube, box into the jaw cells")
        tip0 = ms._tip(ms.observe(False)[0])
        r0, z0 = float(math.hypot(tip0[0], tip0[1])), float(tip0[2])
        r_goal = float(math.hypot(xy[0], xy[1]))
        bearing = math.degrees(math.atan2(xy[1], xy[0]))
        tsay(f"        r {r0*100:.1f} -> {r_goal*100:.1f}cm, "
             f"z {z0*100:.1f} -> {GRASP_Z*100:.1f}cm in {N_APPROACH} steps")

        misses, last_dist, reached = 0, None, False
        for step in range(N_APPROACH):
            ms.checkpoint()
            a = (step + 1) / N_APPROACH
            # LOOK FIRST. The other order advances an increment and only then checks, so
            # a lost detection keeps reaching: pick.py records an arm that carried on
            # through five "not in view" steps and finished forward of the tube.
            c = _cap_now(last_uv[0], colour)
            z_is = float(ms._tip(ms.observe(False)[0])[2])
            if c is None:
                misses += 1
                if last_dist is not None and last_dist < 170 and z_is <= GRASP_Z + 0.03:
                    tsay(f"        the cap passed under the jaws at {last_dist:.0f}px, "
                         f"z={z_is*100:.1f}cm — finishing")
                    reached = True
                    break
                tsay(f"        {step+1:2d}: cap not in view — HOLDING ({misses}/4)")
                if misses >= 4:
                    break
                time.sleep(0.25)
                continue
            misses = 0
            last_uv[0] = (c["x"], c["y"])
            ex, ey = AIM[0] - c["x"], AIM[1] - c["y"]
            dist = float(math.hypot(ex, ey))
            last_dist = dist
            jg = ms.jaw_frame()
            # THE BOX, not just the centroid: the cap is 40px across and the question is
            # whether it is BETWEEN the fingers, which a box can answer and a point
            # cannot when it sits on a cell boundary.
            corners = [(c["bbox"][0], c["bbox"][1]), (c["bbox"][2], c["bbox"][1]),
                       (c["bbox"][0], c["bbox"][3]), (c["bbox"][2], c["bbox"][3])]
            in_grip = (ms.GRID.in_grip((c["x"], c["y"]), jg)
                       or any(ms.GRID.in_grip(k, jg) for k in corners))
            tsay(f"        {step+1:2d}/{N_APPROACH}: box "
                 f"{ms.GRID.cell_of((c['x'], c['y']))} -> jaws {ms.GRID.cell_of(AIM)}  "
                 f"dx {ex:+.0f} dy {ey:+.0f} d {dist:.0f}px  z {z_is*100:.1f}cm"
                 f"{'  IN GRIP' if in_grip else ''}")

            if in_grip and z_is <= GRASP_Z + 0.02:
                tsay(f"        box is in the jaw cells and the hand is down at "
                     f"{z_is*100:.1f}cm — closing")
                reached = True
                break

            # bearing only, on pick.py's shrinking gain
            scale = 1.0 + 1.1 * a
            if abs(ex) > 30:
                mag = float(np.clip(abs(ex) / (14.0 * scale), 0.4, 3.2 / scale))
                bearing += math.copysign(mag, ex * sign_base)

            r = min(r0 + (r_goal - r0) * a, r_goal)
            z = z0 + (GRASP_Z - z0) * a
            tgt = np.array([r * math.cos(math.radians(bearing)),
                            r * math.sin(math.radians(bearing)), z])
            if ms._move_tip(tgt, GRASP_PITCH, j5, settle=0.14, step=1.0) is None:
                tsay(f"        r={r*100:.0f}cm z={z*100:.1f}cm unreachable — "
                     f"closing from here")
                break

        # ---- GRASP ----------------------------------------------------------------
        tphase("GRASP", "closing across the tube")
        held, idle = ms.close_with_current(step=4.0, delay=0.08)
        tsay(f"        {'CONTACT' if held else 'NO CONTACT'} (idle current {idle:.0f})")

        # DO NOT CLAIM A TUBE THE EVIDENCE DOES NOT SUPPORT. A previous run reported
        # SUCCESS with the cap 218px from the jaws: the idle current had read 0, so the
        # contact test compared against nothing and any draw at all tripped it. Two
        # guards, and each catches what the other misses.
        if idle <= 0.5:
            raise RuntimeError(
                f"the gripper's idle current read {idle:.1f}, so the contact test had "
                f"nothing to compare against — refusing to call this a grasp")
        if not held:
            raise RuntimeError("closed on nothing")
        if last_dist is not None and last_dist > 2.0 * GRASP_RADIUS_PX:
            raise RuntimeError(
                f"the jaws closed, but the cap was last seen {last_dist:.0f}px away — "
                f"further than {2.0*GRASP_RADIUS_PX:.0f}px, so whatever is between them "
                f"is not this tube")

        ms._set_carry(True, label=f"{colour} tube", h_m=TUBE_D_M)
        tphase("LIFT", "lifting clear")
        tip = ms._tip(ms.observe(False)[0])
        ms._move_tip(np.array([tip[0], tip[1], z0]), GRASP_PITCH, j5,
                     settle=0.22, step=1.2)
        return f"holding the {colour} tube ({last_dist:.0f}px at the close)"

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
