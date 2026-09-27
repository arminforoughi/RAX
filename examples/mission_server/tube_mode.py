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
    #: Fingertip height while travelling over the table, metres.
    HOVER_Z = 0.13
    #: Fingertip height at the grasp. A tube lying down puts its centre one radius up,
    #: and the jaws close AROUND it, so the tip wants to arrive level with that centre
    #: rather than on the table. 8mm is the tube's own radius.
    GRASP_Z = 0.010
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
        """Pick a tube by its cap. Four stages, and no detector network anywhere.

        LOOK -> OVER -> SQUARE -> GRID, which is the sequence the operator asked for and
        also the one pick.py uses on the X250. What it replaces is a base sweep that
        searched for a YOLO label: on this bench that mapped ten objects and no tubes
        with the broad vocabulary, found one of two with a targeted query, and spent a
        minute of arm motion doing it. The cap detector sees both tubes in the frame the
        arm is already looking at.

        EVERY CORRECTION IS METRIC, VIA THE TABLE PLANE, and that is what makes the grid
        stage exact rather than a tuned gain. The cap's pixel and the grip centre's pixel
        are both cast onto the table through the calibrated hand-eye transform, and the
        difference between those two points IS the distance the tip must move. No pixels
        per centimetre to measure, no gain to drift: the same transform that reprojects
        this rig's fingertip to 0px does the whole job.
        """
        q0 = ms.observe(False)[0].astype(float)
        j5 = float(q0[ms.ARM.roll_joint])

        # ---- LOOK -----------------------------------------------------------------
        tphase("LOOK", f"finding the {colour} cap")
        cap = _cap_now(uv_hint, colour)
        if cap is None:
            raise RuntimeError(f"no {colour} cap in view")
        xy = _table_xy((cap["x"], cap["y"]))
        if xy is None:
            raise RuntimeError("could not cast the cap onto the table plane")
        tsay(f"        cap at ({cap['x']:.0f},{cap['y']:.0f})px -> "
             f"({xy[0]*100:+.1f},{xy[1]*100:+.1f})cm, r={math.hypot(*xy)*100:.1f}cm")

        # ---- OVER -----------------------------------------------------------------
        tphase("OVER", f"moving over the tube at r={math.hypot(*xy)*100:.0f}cm")
        if ms._move_tip(np.array([xy[0], xy[1], HOVER_Z]), GRASP_PITCH, j5,
                        settle=0.20, step=1.6) is None:
            raise RuntimeError(f"cannot reach over ({xy[0]*100:+.0f},{xy[1]*100:+.0f})cm")

        # Re-measure from directly above. The first fix was taken from a shallow angle
        # where the sightline grazes the table and range error is amplified; this one
        # looks straight down at it.
        cap2 = _cap_now((cap["x"], cap["y"]), colour)
        xy2 = _table_xy((cap2["x"], cap2["y"])) if cap2 else None
        if xy2 is not None:
            moved = math.hypot(xy2[0] - xy[0], xy2[1] - xy[1])
            # A SECOND LOOK THAT DISAGREES BY A LOT IS NOT A REFINEMENT. The same
            # judgement mission_server makes with FIX_REFINE_MAX_M: past some distance
            # the two measurements are not of the same thing, and believing the second
            # one silently walks the arm to wherever the mistake was. The view from
            # above IS the better geometry -- the first fix is taken down a grazing
            # sightline where range error is amplified -- so a small correction is
            # trusted and a large one is reported and refused.
            if moved <= FIX_REFINE_MAX_M:
                tsay(f"        from above: ({xy2[0]*100:+.1f},{xy2[1]*100:+.1f})cm "
                     f"({moved*100:.1f}cm refinement)")
                xy, cap = xy2, cap2
            else:
                tsay(f"        from above the {colour} cap maps to "
                     f"({xy2[0]*100:+.1f},{xy2[1]*100:+.1f})cm, {moved*100:.1f}cm from "
                     f"the first fix — too far to be the same tube. Keeping the first.")

        # ---- APPROACH: pick.py's logic, which is already in this branch ----------
        #
        # THE IMAGE CORRECTS THE BEARING AND NOTHING ELSE. This is the part I kept
        # rewriting from first principles and kept getting wrong. pick.py drives the
        # descent like this:
        #
        #     if abs(ex) > 30: base += sign * clip(abs(ex)/(14*scale), 0.4, 3.2/scale)
        #
        # A base nudge of between 0.4 and 3.2 degrees, shrinking as the arm gets lower,
        # and the VERTICAL pixel error `ey` is computed and deliberately never used to
        # move anything. Depth follows a fixed profile; vision only fixes which way the
        # arm is pointing.
        #
        # That is why it does not overshoot and my metric version did. Rotating the base
        # SWINGS the hand across the target; correcting the tip in x and y, which is what
        # I was doing from a table-plane cast, REACHES the arm out. Told to close a few
        # centimetres of image error, the first extends the arm forward into the bench,
        # and it does it on the strength of a cast whose error grows exactly where the
        # sightline is shallow.
        #
        # So: the radius is set once, by the fix taken over the tube, and afterwards only
        # the base turns and only z comes down.
        AIM = ms.jaw_frame().centre_uv
        GRASP_RADIUS_PX = 70.0     # pick.py's, and the [grip] log agrees: <=67px picked
        AT_DEPTH = 0.85            # ...and it must also be down, or it closes on air

        # THE SIGN OF THE BASE CORRECTION IS MEASURED, NOT ASSUMED. Which way the cap
        # slides when the base turns depends on how the camera is mounted, and a wrong
        # sign hides under detector noise while driving the arm steadily the wrong way.
        # pick.py probes for it with +7/-10/+14 until the cap moves clear of noise.
        tphase("PROBE", "measuring which way the base moves the cap")
        sign_base, probe_px = 0.0, 0.0
        q_probe = ms.observe(False)[0].astype(float)
        u0 = cap["x"]
        for nudge in (3.0, -5.0, 7.0):
            ms.checkpoint()
            q = q_probe.copy()
            q[ms.ARM.pan_joint] = float(np.clip(q_probe[ms.ARM.pan_joint] + nudge,
                                                ms.J_LO[ms.ARM.pan_joint],
                                                ms.J_HI[ms.ARM.pan_joint]))
            ms.goto_smooth(ms._clamp_joints(q), settle=0.25, step=2.0)
            c = _cap_now(last_uv[0], colour)
            if c is not None:
                moved = c["x"] - u0
                if abs(moved) >= 8.0:
                    sign_base = 1.0 if (moved / nudge) > 0 else -1.0
                    probe_px = abs(moved)
                    tsay(f"        base {nudge:+.0f}deg moved the cap {moved:+.0f}px "
                         f"-> correcting with sign {sign_base:+.0f}")
                    break
            tsay(f"        base {nudge:+.0f}deg moved the cap under the noise floor")
        ms.goto_smooth(ms._clamp_joints(q_probe), settle=0.25, step=2.0)
        cap = _cap_now(last_uv[0], colour) or cap
        if sign_base == 0.0:
            tsay("        could not measure the base direction — descending without "
                 "a bearing correction")

        # ---- SQUARE, a nudge only -------------------------------------------------
        tphase("SQUARE", "turning the jaws across the tube")
        if not TUBE_YAW_ALIGN:
            tsay("        tube yaw alignment is off — holding the wrist where it is")
        elif cap.get("angle") is None or not cap.get("confident"):
            tsay("        the tube's angle is not confidently measured — "
                 "holding the wrist rather than guessing")
        elif (cap["elong"] or 0.0) < TUBE_ROLL_MIN_ELONGATION:
            tsay(f"        elongation {cap['elong']:.1f} is under "
                 f"{TUBE_ROLL_MIN_ELONGATION:.1f} — not worth rolling on")
        else:
            jg = ms.jaw_frame()
            err = _fold(cap["angle"] - (jg.axis_deg + 90.0))
            gain = float(getattr(jg, "roll_gain", 1.0)) or 1.0
            tsay(f"        axis {cap['angle']:+.0f}deg (elongation {cap['elong']:.1f}), "
                 f"jaws {jg.axis_deg:+.0f}deg -> {err:+.0f}deg out of square")
            if abs(err) < float(ms.CFG.yaw_deadband_deg):
                tsay(f"        square within {err:+.1f}deg — leaving the wrist")
            else:
                step = float(np.clip(err / gain, -TUBE_MAX_ROLL_DEG, TUBE_MAX_ROLL_DEG))
                lo, hi = ms.J_LO[ms.ARM.roll_joint], ms.J_HI[ms.ARM.roll_joint]
                j5_new = float(np.clip(j5 + step, lo, hi))
                tsay(f"        rolling {step:+.0f}deg ({j5:+.0f} -> {j5_new:+.0f})")
                q = ms.observe(False)[0].astype(float)
                q[ms.ARM.roll_joint] = j5_new
                ms.goto_smooth(ms._clamp_joints(q), settle=0.30, step=2.0)
                j5 = j5_new

        # ---- DESCEND: look, nudge the base, drop one increment --------------------
        tphase("APPROACH", "lining the cap up with the jaws while coming down")
        # POLAR, NOT CARTESIAN, AND THIS IS NOT A STYLE CHOICE. The descent target has to
        # be expressed as (radius, bearing, height) because vision owns the BEARING and
        # kinematics owns the other two. Commanding a Cartesian point instead solves IK
        # for every joint including the pan, which silently undoes the base correction
        # made a moment earlier -- the two would fight each other every step, and the
        # image error would never close no matter how right the nudge was.
        #
        # pick.py sidesteps this by commanding joints directly and never doing Cartesian
        # IK during the descent. Same idea here: the radius is frozen at the fix taken
        # over the tube, the height follows the profile, and the bearing is the one thing
        # the camera is allowed to move.
        z0 = float(ms._tip(ms.observe(False)[0])[2])
        radius = math.hypot(xy[0], xy[1])
        bearing = math.degrees(math.atan2(xy[1], xy[0]))
        tsay(f"        radius {radius*100:.1f}cm frozen; bearing {bearing:+.1f}deg is "
             f"what the camera corrects")
        n_steps, misses, last_dist, reached = 12, 0, None, False
        for step in range(n_steps):
            ms.checkpoint()
            # LOOK FIRST, THEN MOVE. The other order descends an increment and only then
            # checks, so a lost detection keeps reaching -- pick.py's comment records an
            # arm that carried on down through five "not in view" steps and finished well
            # forward of the tube with nothing in the jaws.
            c = _cap_now(last_uv[0], colour)
            a = ((step + 1) / n_steps)
            if c is None:
                misses += 1
                # Losing it CLOSE is how an approach ends: the tube passes under the
                # jaws. Losing it far away is a real failure and must not be driven
                # through.
                if last_dist is not None and last_dist < 170 and a >= 0.7:
                    tsay(f"        the cap passed under the jaws at {last_dist:.0f}px, "
                         f"{100*a:.0f}% down — completing the descent")
                    reached = True
                    break
                tsay(f"        {step+1:2d}: cap not in view — holding, not descending "
                     f"({misses}/4)")
                if misses >= 4:
                    tsay(f"        lost the {colour} cap while still "
                         f"{'%.0fpx away' % last_dist if last_dist else 'far off'}")
                    break
                time.sleep(0.25)
                continue
            misses = 0
            last_uv[0] = (c["x"], c["y"])
            ex, ey = AIM[0] - c["x"], AIM[1] - c["y"]
            dist = float(math.hypot(ex, ey))
            last_dist = dist
            tsay(f"        {step+1:2d}/{n_steps}: cap {ms.GRID.cell_of((c['x'], c['y']))} "
                 f"-> jaws {ms.GRID.cell_of(AIM)}  dx {ex:+.0f} dy {ey:+.0f} "
                 f"dist {dist:.0f}px")

            # CLOSING NEEDS BOTH: lined up in the picture AND down at tube height. Image
            # alignment fixes the bearing and says nothing about depth; pick.py logged a
            # run that converged to 7px at 41% descent and closed on air above the tube.
            if dist < GRASP_RADIUS_PX and a >= AT_DEPTH:
                tsay(f"        between the jaws ({dist:.0f}px) and at depth "
                     f"({100*a:.0f}%) — closing")
                reached = True
                break
            if dist < GRASP_RADIUS_PX:
                tsay(f"        lined up ({dist:.0f}px) but only {100*a:.0f}% down — "
                     f"continuing")

            # ONE OBSERVATION, ONE MOVE. The base and the descent are independent, so
            # they go in a single command rather than two look-move round trips.
            # pick.py's gain, unchanged: between 0.4 and 3.2 degrees of bearing, and
            # shrinking as the arm gets lower, because the same pixel error means less
            # distance the closer the camera is.
            scale = 1.0 + 1.1 * a
            if abs(ex) > 30 and sign_base != 0.0:
                mag = float(np.clip(abs(ex) / (14.0 * scale), 0.4, 3.2 / scale))
                bearing += math.copysign(mag, ex * sign_base)
                tsay(f"        bearing {math.copysign(mag, ex*sign_base):+.1f}deg "
                     f"-> {bearing:+.1f}deg")

            z = z0 + (GRASP_Z - z0) * a
            tgt = np.array([radius * math.cos(math.radians(bearing)),
                            radius * math.sin(math.radians(bearing)), z])
            if ms._move_tip(tgt, GRASP_PITCH, j5, settle=0.14, step=1.0) is None:
                tsay(f"        z={z*100:.1f}cm at r={radius*100:.0f}cm unreachable — "
                     f"closing from here")
                break

        if not reached and last_dist is not None and last_dist >= GRASP_RADIUS_PX:
            tsay(f"        finished the descent {last_dist:.0f}px from the jaws — "
                 f"closing anyway, and the episode records how far off it was")

        # ---- CLOSE ----------------------------------------------------------------
        tphase("GRASP", "closing across the tube")
        held, idle = ms.close_with_current(step=4.0, delay=0.08)
        tsay(f"        {'CONTACT' if held else 'NO CONTACT'} (idle current {idle:.0f})")
        if not held:
            return "closed on nothing"
        ms._set_carry(True, label=f"{colour} tube", h_m=TUBE_D_M)
        tphase("LIFT", "lifting clear")
        tip = ms._tip(ms.observe(False)[0])
        ms._move_tip(np.array([tip[0], tip[1], HOVER_Z]), GRASP_PITCH, j5,
                     settle=0.22, step=1.2)
        return f"holding the {colour} tube"


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
