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
import os
import threading
import time

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
RACK_XY = [0.24, -0.14]
RACK_PITCH_M = 0.022
RACK_NX, RACK_NY = 3, 2


def _hole_grid(x, y):
    return [(x + (i - (RACK_NX - 1) / 2) * RACK_PITCH_M,
             y + (j - (RACK_NY - 1) / 2) * RACK_PITCH_M)
            for j in range(RACK_NY) for i in range(RACK_NX)]


#: A lab tube's nominal shape, from `rax.perception.object_priors`: 16 mm across,
#: 100 mm long. The orientation test compares a measurement against these.
TUBE_D_M, TUBE_L_M = 0.016, 0.100

#: How far a measured dimension may be from the nominal one and still count as a match.
#: Generous, because this rig's height measurement is known to read low (see
#: _dest_geometry in mission_server) -- the test is meant to reject measurements that
#: match NEITHER pose, not to grade the ones that do.
_FIT = 0.45


def _orientation(w_m: float, d_m: float, h_m: float):
    """True = upright, False = lying, None = the measurement says neither.

    See the caller for why "neither" is a necessary third answer and not a cop-out.
    """
    if max(w_m, d_m, h_m) <= 0.0:
        return None

    def near(value, want):
        return abs(value - want) <= _FIT * want

    foot_long, foot_short = max(w_m, d_m), min(w_m, d_m)
    # Upright: tall as the tube is long, on a small roughly-round footprint.
    if near(h_m, TUBE_L_M) and near(foot_long, TUBE_D_M):
        return True
    # Lying: as long as the tube across the table, only a diameter high.
    if near(foot_long, TUBE_L_M) and near(h_m, TUBE_D_M):
        return False
    return None


def start(ms, port: int = 8486) -> None:
    """Serve the tube UI on ``port``, backed by mission-server module ``ms``."""
    from rax.manipulation.attempt import with_retries
    from rax.manipulation.episodes import EpisodeLog
    from rax.manipulation.grip import CurrentRise, reconcile
    from rax.perception.tube_caps import draw as draw_caps
    from rax.perception.tube_caps import find_caps
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
        caps = find_caps(img, exclude=[ms.HAND_UV], exclude_r=70)
        ms.LAST_CAPS[0] = [{"colour": c.colour, "x": c.x, "y": c.y, "area": c.area,
                            "bbox": list(c.bbox)} for c in caps]
        ms.LAST_CAPS[1] = time.time()
        if caps:
            img[:, :] = draw_caps(img, caps)

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
        out = []
        for o in ms.world2d_snapshot():
            label = str(o.get("label", ""))
            if not any(w in label.lower() for w in TUBE_WORDS):
                continue
            # WHICH WAY IS IT LYING? THREE ANSWERS, AND ONE OF THEM IS "DON'T KNOW".
            #
            # The obvious test is the measured height: a 100 mm tube standing reads
            # ~100 mm tall, one on its side reads ~16 mm, its diameter. That test was
            # tried here and it QUIETLY LIED. The first tube this rig mapped measured
            # w/d/h = 9/13/41 mm, which the height test called "standing" -- while the
            # wrist camera plainly showed it lying on the turntable.
            #
            # The reason is that 9/13/41 is not the measurement of a tube in ANY pose.
            # Standing, a tube is about 16x16x100; lying, about 100x16x16. This is
            # neither, so the silhouette solve had measured a fragment -- most likely
            # just the cap. Feeding a number that shape into a two-way test does not
            # produce a wrong answer honestly labelled, it produces a confident one.
            #
            # So the test is run against BOTH hypotheses and has to positively match one:
            # standing means the height is near the tube's length; lying means the
            # footprint's long side is. Matching neither returns None -- orientation
            # unknown -- and the viewer then draws a dashed footprint ring instead of a
            # tube pointing somewhere. That is the same discipline the object map applies
            # with `yaw_known`, which it sets only when the solve really produced an
            # angle: "drawing that as a definite orientation is a lie".
            w_m, d_m, h_m = (float(o.get("w_m", 0.0)), float(o.get("d_m", 0.0)),
                             float(o.get("h_m", 0.0)))
            standing = _orientation(w_m, d_m, h_m)
            out.append({"id": o.get("tag"), "colour": _colour_of(label),
                        "x": round(float(o["x"]), 4), "y": round(float(o["y"]), 4),
                        "z": 0.0, "held": False, "rack": None, "hole": None,
                        "source": "seen" if o.get("age", 99) < 8 else "mapped",
                        # WHICH DIMENSIONS TO DRAW. When the orientation classified, the
                        # measurement matched a tube and can be drawn as measured. When it
                        # did not, the measurement is a fragment (the 9/13/41 mm case is a
                        # cap, not a tube) and drawing it would render a stub that is
                        # neither the object nor honestly vague -- so fall back to the
                        # nominal tube and let the dashed footprint carry the uncertainty.
                        "d": round(min(w_m, d_m, h_m), 4) if standing is not None
                             else TUBE_D_M,
                        "l": round(max(w_m, d_m, h_m), 4) if standing is not None
                             else TUBE_L_M,
                        # None -> the viewer draws an uncertainty footprint, not a pose
                        "standing": standing,
                        "yaw": float(o.get("yaw", 0.0)),
                        # An orientation we could not classify is not a known yaw either,
                        # however confident the map's own silhouette solve was about the
                        # angle of whatever fragment it measured.
                        "yaw_known": bool(o.get("yaw_known", False)) and standing is not None,
                        "measured": bool(o.get("measured", False)),
                        "w": w_m, "h_m": h_m, "age": o.get("age"),
                        "label": label})
        with ms.lock:
            held_label = ms.carry["label"] if ms.carry["held"] else None
        if held_label:
            for t in out:
                if t["label"] == held_label:
                    # Upright in the jaws: the gripper closes across the tube and lifts
                    # it, so for once the orientation is known without measuring.
                    t.update(held=True, standing=True, yaw_known=True)
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

    # ---- the run ---------------------------------------------------------------
    def run(label, dest_hole):
        ep = episodes.start("tube_pick", label, arm=ms.ARM.name, simulated=False,
                            dest=("rack hole %d" % dest_hole) if dest_hole is not None
                            else None)

        def action(n):
            tphase("PICK", f"'{label}', attempt {n} — the server's own pick")
            ok, detail = ms.attempt_pick(label, tries=1)
            if not ok:
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
            tphase("RETRY", f"attempt {n}: back to home and re-acquire")
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
        threading.Thread(target=run, args=(label, hole), daemon=True).start()
        return jsonify(ok=True, label=label, hole=hole)

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
