"""Tube sorting on the SO-101: map the mat, pick each tube, stand it up in its rack.

The pick and the place are the generic ones in :mod:`rax.pick`; this module adds what is
specific to this bench: which caps count, where the racks are (from the overhead
camera), which rack each colour goes to, and the UI.

It runs inside the mission server's process, because that process owns the serial bus
and the wrist camera, and serves the UI on its own port.
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
from so101_arm import So101Arm

from rax.pick import ColourTarget, PickConfig, pick, place, scan
from rax.pick.arm import bearing_of, move_to, solve

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
TUBE_UI = os.path.join(os.path.dirname(HERE), "tube_server", "ui")

# ---- the tubes ----------------------------------------------------------------------
#: Caps are found by colour. The left jaw's corner of the wrist view reads blue.
TUBE = ColourTarget(name="tube", colours=("green", "blue", "red"),
                    ignore=((0.0, 340.0, 200.0, 1e4),), grasp_z=0.010, open_pct=45.0)
TUBE_D_M, TUBE_L_M = 0.016, 0.100

#: The pick, as tuned on this rig. Everything not named here is the library default.
RIG = PickConfig(
    default_gain=-7.5,      # base px/deg measured here: -5.1 .. -10.4
    aim_offset_px=80.0,     # the operator's set point: the hand lands on the tube's right
    twist_fraction=0.5,     # the operator: "do half of the angle"
    twist_max=45.0,
    twist_ambiguous=70.0,   # nearly perpendicular: roll negative, what worked here
)

#: Jaw stop that means something is held (percent open): below it, the jaws shut on air.
GRIP_BLOCKED_PCT = 3.5
#: While picking, the base never faces further right than this (deg): the racks.
PICK_MIN_BEARING_DEG = -15.0
#: The wrist scan's stops, and the image row below which caps are the gripper's own.
SCAN_BEARINGS_DEG = (40.0, 20.0, 0.0)
SCAN_TILTS_DEG = (0.0, 25.0)
MAP_MAX_Y_PX = 330.0

# ---- the overhead camera and the racks ----------------------------------------------
#: Top camera pixel <-> table, a similarity fitted by eye off two fingertip positions
#: (2026-09-27). Good to about 2cm. A 3-point refit made the drops worse and was reverted.
TOP_ORIGIN_PX = (557.0, 185.0)
TOP_ORIGIN_XY = (0.192, 0.001)
TOP_PX_PER_M = 960.0
TOP_EX = (0.977, 0.203)        # robot +x, as a unit vector in the image
TOP_EY_DOWN = (-0.203, 0.977)  # robot -y, as a unit vector in the image


def top_px_to_xy(u, v):
    du, dv = u - TOP_ORIGIN_PX[0], v - TOP_ORIGIN_PX[1]
    return (TOP_ORIGIN_XY[0] + (du * TOP_EX[0] + dv * TOP_EX[1]) / TOP_PX_PER_M,
            TOP_ORIGIN_XY[1] - (du * TOP_EY_DOWN[0] + dv * TOP_EY_DOWN[1]) / TOP_PX_PER_M)


def top_px_delta_to_xy(du, dv):
    return ((du * TOP_EX[0] + dv * TOP_EX[1]) / TOP_PX_PER_M,
            -(du * TOP_EY_DOWN[0] + dv * TOP_EY_DOWN[1]) / TOP_PX_PER_M)


def xy_to_top_px(x, y):
    dx, dy = (x - TOP_ORIGIN_XY[0]) * TOP_PX_PER_M, -(y - TOP_ORIGIN_XY[1]) * TOP_PX_PER_M
    return (TOP_ORIGIN_PX[0] + dx * TOP_EX[0] + dy * TOP_EY_DOWN[0],
            TOP_ORIGIN_PX[1] + dx * TOP_EX[1] + dy * TOP_EY_DOWN[1])


#: Rack holes as pixels in the top view (Hough, top faces only, 2026-09-27 19:50).
TOP_RACKS = [
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
#: Which rack each cap colour goes to -- the operator's rule.
RACK_FOR_COLOUR = {"blue": "black rack (est)", "green": "black rack (est)",
                   "red": "silver rack (est)"}
#: A cap this close to a rack's holes is already racked, not one to pick.
RACK_EXCLUDE_M = 0.04

TOP_TABLE_ROI = (370, 40, 1040, 600)
TOP_CAP_GATES = {"green": (90, 70), "blue": (90, 70), "gold": (55, 90), "red": (100, 60)}
TOP_HELD_SEARCH_PX = 70.0     # the held cap is within this of the fingertip's own pixel
TOP_ALIGN_TOL_PX = 6.0
HOLE_VERIFY_PX = 40.0         # a cap this near the hole afterwards means it went in

# ---- the upright drop -------------------------------------------------------------
#: Held from above, the tube lies across the jaws; with the hand level (pitch 0) and the
#: wrist at roll +-90 it stands vertical.
STAND_ROLL_DEG = 90.0
STAND_R_M, STAND_Z_M = 0.26, 0.16   # where it is stood up (solves level at either roll)
RACK_TOP_Z = 0.055                  # rack top above the table (estimated)
TUBE_BELOW_TIP_M = 0.085            # tube hanging below the fingertips once standing
HOVER_CLEAR_M = 0.05                # tube bottom over the rack while lining up
DROP_CLEAR_M = 0.01                 # ...and at the release
EXTRA_Z_BY_COLOUR = {"red": 0.06}   # carried higher, on request
CARRY_Z_M = 0.25


def in_rack_zone(xy, margin=RACK_EXCLUDE_M):
    x, y = xy
    for r in TOP_RACKS:
        hs = [top_px_to_xy(u, v) for u, v in r["holes_px"]]
        xs, ys = [h[0] for h in hs], [h[1] for h in hs]
        if min(xs) - margin <= x <= max(xs) + margin and min(ys) - margin <= y <= max(ys) + margin:
            return True
    return False


def tag_of(e: Exception) -> str:
    """A short failure tag for the results list."""
    t = str(e).lower()
    for words, tag in ((("stopped by user",), "stopped"),
                       (("in view", "not on the map"), "not found"),
                       (("dropped",), "dropped in carry"),
                       (("grabbed two",), "grabbed two"),
                       (("closed on nothing", "never closed"), "missed grasp"),
                       (("no free hole",), "rack full"),
                       (("cannot reach", "cannot hover", "not reachable"), "unreachable")):
        if any(w in t for w in words):
            return tag
    return "error"


def start(ms, port: int = 8486) -> None:
    """Serve the tube UI on ``port``, driving the arm of mission-server module ``ms``."""
    from rax.manipulation.episodes import EpisodeLog
    from rax.perception.tube_caps import draw as draw_caps
    from rax.perception.tube_caps import find_caps
    from rax.robots.urdf_visuals import link_visuals

    try:
        ms.DRAW_YOLO_BOXES[0] = False       # caps only in this mode
    except Exception:
        pass

    app = Flask("tube_mode", static_folder=None)
    lock = threading.Lock()
    tstate = {"phase": "IDLE", "note": "ready", "running": False, "log": [], "results": [],
              "used_holes": {}}
    tube_map: dict[int, dict] = {}    # id -> {id, colour, x, y, n, t, picked}
    focus = {"colour": None, "xy": None}

    def tsay(msg):
        ms.say(msg)
        with lock:
            tstate["log"].append({"t": time.strftime("%H:%M:%S"), "m": str(msg)[:220]})
            del tstate["log"][:-200]

    def tphase(name, note=""):
        with lock:
            tstate["phase"], tstate["note"] = name, note
        tsay(f"[{name}] {note}" if note else f"[{name}]")

    arm = So101Arm(ms, log=tsay, phase=tphase)
    episodes = EpisodeLog(
        os.path.join(HERE, "tube_episodes.jsonl"),
        probe=lambda: {"joints": [round(float(v), 1) for v in ms.observe(False)[0]]},
        note=tsay)

    def holding():
        return float(ms.state.get("gripper") or 0.0) > GRIP_BLOCKED_PCT

    # ---- the wrist view: caps boxed, the target joined to the grip ------------------
    def cap_overlay(img):
        src = ms.latest_rgb[0]
        clean = img if src is None else cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2BGR)
        dets = TUBE.detect(clean)
        try:
            q = arm.joints()
            dets = [d for d in dets if not in_rack_zone(arm.cast((d.u, d.v), q) or (9, 9))]
        except Exception:
            pass
        if focus["colour"] is not None:
            dets = [d for d in dets if d.label == focus["colour"]]
        ms.LAST_CAPS[0] = [{"colour": d.label, "x": d.u, "y": d.v, "area": d.area,
                            "bbox": list(d.box)} for d in dets]
        ms.LAST_CAPS[1] = time.time()
        if not dets:
            return
        img[:, :] = draw_caps(img, [c for c in find_caps(clean, colours=TUBE.colours)
                                    if any(abs(c.x - d.u) < 1 and abs(c.y - d.v) < 1
                                           for d in dets)])
        ju, jv = (int(v) for v in arm.jaw_uv)
        for d in dets:
            cv2.line(img, (int(d.u), int(d.v)), (ju, jv), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(img, (ju, jv), (255, 120, 255), cv2.MARKER_TILTED_CROSS, 16, 2)

    # ---- the map ---------------------------------------------------------------
    def tubes():
        now = time.time()
        with lock:
            return [{"id": e["id"], "colour": e["colour"], "x": round(e["x"], 4),
                     "y": round(e["y"], 4), "z": 0.0, "held": False, "rack": None,
                     "hole": None, "source": "mapped", "d": TUBE_D_M, "l": TUBE_L_M,
                     "standing": None, "yaw": 0.0, "yaw_known": False, "n": e["n"],
                     "age": round(now - e["t"], 1), "label": f"{e['colour']} tube",
                     "focus": bool(focus["xy"] is not None and e["colour"] == focus["colour"]
                                   and math.hypot(e["x"] - focus["xy"][0],
                                                  e["y"] - focus["xy"][1]) < 0.03)}
                    for e in sorted(tube_map.values(), key=lambda v: v["id"])
                    if not e["picked"]]

    def do_scan():
        found = scan(arm, TUBE, SCAN_BEARINGS_DEG, SCAN_TILTS_DEG, max_v=MAP_MAX_Y_PX,
                     avoid=in_rack_zone, reach=(ms.ARM.reach_min_m, ms.ARM.reach_max_m))
        with lock:
            tube_map.clear()
            for i, f in enumerate(found, 1):
                tube_map[i] = {"id": i, "colour": f.label, "x": f.x, "y": f.y, "n": f.n,
                               "t": time.time(), "picked": False}
        tsay(f"        wrist scan: {len(found)} tube(s) mapped")
        return len(found)

    def racks():
        out = []
        for r in TOP_RACKS:
            hs = [top_px_to_xy(u, v) for u, v in r["holes_px"]]
            out.append({"name": r["name"], "colour": r["colour"], "yaw": 0.0,
                        "x": round(sum(h[0] for h in hs) / len(hs), 4),
                        "y": round(sum(h[1] for h in hs) / len(hs), 4),
                        "holes": [[round(h[0], 4), round(h[1], 4)] for h in hs]})
        return out

    # ---- the overhead camera ------------------------------------------------------
    def top_caps(colour):
        """Caps of this colour in the top view, as [(u, v)], or None without a frame."""
        img = ms._overhead_frame()
        if img is None:
            return None
        x0, y0, x1, y1 = TOP_TABLE_ROI
        return [(c.x + x0, c.y + y0)
                for c in find_caps(img[y0:y1, x0:x1], colours=(colour,), min_area=12,
                                   max_area=900, gates=TOP_CAP_GATES)]

    def held_cap(colour, static):
        """The held cap in the top view: not one of ``static``, near the fingertip's pixel."""
        caps = top_caps(colour) or []
        tip = arm.tip(arm.joints())
        pu, pv = xy_to_top_px(float(tip[0]), float(tip[1]))
        mine = [c for c in caps
                if all(math.hypot(c[0] - s[0], c[1] - s[1]) > 8.0 for s in static)
                and math.hypot(c[0] - pu, c[1] - pv) <= TOP_HELD_SEARCH_PX]
        return min(mine, key=lambda c: math.hypot(c[0] - pu, c[1] - pv), default=None)

    # ---- the upright drop -------------------------------------------------------
    def place_upright(colour, rack_name=None):
        """Stand the held tube up and drop it in a free hole. Returns (ok, tag, detail)."""
        rack = next(r for r in TOP_RACKS
                    if r["name"] == (rack_name or RACK_FOR_COLOUR[colour]))
        used = tstate["used_holes"].setdefault(rack["name"], [])

        tphase("STAND", f"standing the {colour} tube up")
        b0 = bearing_of(arm.tip(arm.joints()))
        before = top_caps(colour) or []
        j5 = float(arm.joints()[arm.roll])     # the nearer of +-90: the wrist never turns over
        roll = STAND_ROLL_DEG if abs(j5 - STAND_ROLL_DEG) <= abs(j5 + STAND_ROLL_DEG) \
            else -STAND_ROLL_DEG
        move_to(arm, (STAND_R_M * math.cos(b0), STAND_R_M * math.sin(b0), STAND_Z_M),
                0.0, roll, "the level hand", speed=1.2)
        if not holding():
            raise RuntimeError("dropped the tube while standing it up")
        held = held_cap(colour, before)
        static = [c for c in (top_caps(colour) or [])
                  if held is None or math.hypot(c[0] - held[0], c[1] - held[1]) > 8.0]

        # a free hole: not used, not seen occupied, and reachable with the hand level
        z_base = RACK_TOP_Z + TUBE_BELOW_TIP_M
        cands = []
        for k, (u, v) in enumerate(rack["holes_px"]):
            if k in used or any(math.hypot(u - s[0], v - s[1]) < 10.0 for s in static):
                continue
            x, y = top_px_to_xy(u, v)
            if solve(arm, (x, y, z_base + DROP_CLEAR_M), 0.0, roll) is not None:
                cands.append((math.hypot(x, y), k, (u, v), (x, y)))
        if not cands:
            raise RuntimeError(f"no free hole left in the {rack['name']}")
        _, k, hole_px, (hx, hy) = min(cands)

        # hover as high as asked, or as high as this hole allows
        z_extra = EXTRA_Z_BY_COLOUR.get(colour, 0.0)
        while z_extra > 0 and solve(arm, (hx, hy, z_base + HOVER_CLEAR_M + z_extra),
                                    0.0, roll) is None:
            z_extra = max(0.0, z_extra - 0.01)
        z_hover = z_base + HOVER_CLEAR_M + z_extra
        tphase("CARRY", f"over {rack['name']} hole {k}")

        def align():
            cap = held_cap(colour, static)
            if cap is None:
                return None
            du, dv = hole_px[0] - cap[0], hole_px[1] - cap[1]
            tsay(f"        align: cap {math.hypot(du, dv):.0f}px from the hole")
            return (0.0, 0.0) if math.hypot(du, dv) <= TOP_ALIGN_TOL_PX \
                else top_px_delta_to_xy(du, dv)

        place(arm, (hx, hy), release_z=z_base + DROP_CLEAR_M, pitch=0.0, roll=roll,
              hover_z=z_hover, carry_z=CARRY_Z_M, align=align if held else None,
              holding=holding)
        with lock:
            used.append(k)

        # did it go in? look from above with the arm out of the way
        arm.move(arm.home, speed=1.4, settle=0.2)
        caps = top_caps(colour)
        if caps is None:
            return True, "placed (unverified)", f"{rack['name']} hole {k}, no top view"
        near = min((math.hypot(c[0] - hole_px[0], c[1] - hole_px[1]) for c in caps),
                   default=None)
        if near is not None and near <= HOLE_VERIFY_PX:
            return True, "placed", f"in {rack['name']} hole {k}"
        return False, "missed hole", f"dropped at {rack['name']} hole {k}, no cap there after"

    # ---- one tube, start to finish ---------------------------------------------------
    def pick_and_place(t, rack_name):
        """Returns (ok, tag). Never raises."""
        colour, xy = t["colour"], (t["x"], t["y"])
        focus["colour"], focus["xy"] = colour, xy
        ep = episodes.start("tube_pick", t["label"], arm=ms.ARM.name, simulated=False,
                            dest=rack_name)
        try:
            if math.degrees(bearing_of(xy)) < PICK_MIN_BEARING_DEG:
                raise RuntimeError("not reachable: past the guardrail, toward the racks")
            arm.min_bearing_deg = PICK_MIN_BEARING_DEG
            try:
                res = pick(arm, TUBE, near_xy=xy, label=colour, cfg=RIG, avoid=in_rack_zone)
            finally:
                arm.min_bearing_deg = None
            ms._set_carry(True, label=f"{colour} tube", h_m=TUBE_D_M)
            if rack_name is None:
                ok, tag, detail = True, "picked", f"jaws at {res.grip_pct:.1f}"
            else:
                ok, tag, detail = place_upright(colour, rack_name)
        except Exception as e:
            ok, tag, detail = False, tag_of(e), f"{type(e).__name__}: {e}"
            try:
                arm.release()
            except Exception:
                pass
        focus["colour"], focus["xy"] = None, None
        episodes.end(ep, ok, f"[{tag}] {detail}")
        with lock:
            tstate["results"].append({"t": time.strftime("%H:%M:%S"), "colour": colour,
                                      "ok": bool(ok), "tag": tag, "detail": str(detail)[:160],
                                      "x": round(xy[0], 3), "y": round(xy[1], 3)})
            del tstate["results"][:-60]
            if ok and t["id"] in tube_map:
                tube_map[t["id"]]["picked"] = True
        tsay(f"[{tag.upper()}] {colour}: {detail}")
        return ok, tag

    def start_job(target):
        """Claim the arm and run ``target`` in the background."""
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

    # ---- routes ----------------------------------------------------------------
    urdf = [None]

    @app.route("/urdf")
    def r_urdf():
        if urdf[0] is None:
            try:
                urdf[0] = [{"name": n, "v": [round(float(x), 4) for x in V.ravel()],
                            "f": [int(i) for i in F.ravel()]}
                           for n, (V, F) in link_visuals(
                               ms.ARM.urdf_path, mesh_dir=ms.ARM.mesh_path).items()]
            except Exception as e:
                tsay(f"3D: visuals failed: {type(e).__name__}: {e}")
                urdf[0] = []
        return jsonify(links=urdf[0], arm=ms.ARM.name)

    @app.route("/geom")
    def r_geom():
        xf, tip = {}, None
        try:
            q = np.asarray(ms.observe(False)[0], np.float64).copy()
            q[4] += ms.WRIST_RENDER_OFFSET        # display only
            chain = dict(ms.kin.get_link_transforms_chain(q))
            xf = {n: [round(float(v), 5) for v in np.asarray(T, np.float64).ravel()]
                  for n, T in chain.items()}
            if "gripper_link" in chain:
                xf["moving_jaw_so101_v1_link"] = [
                    round(float(v), 5)
                    for v in (np.asarray(chain["gripper_link"], np.float64) @ ms.JAW_T).ravel()]
            tip = [round(float(v), 4) for v in np.asarray(ms.kin.forward_kinematics(q))[:3, 3]]
        except Exception as e:
            logger.debug("tube /geom: %s", e)
        g = ms.ARM.gripper
        grip = float(ms.state.get("gripper") or g.open_pct)
        with lock:
            ph, note = tstate["phase"], tstate["note"]
        caps = list(ms.LAST_CAPS[0]) if time.time() - ms.LAST_CAPS[1] < 2.0 else []
        return jsonify(arm=ms.ARM.name, simulated=False, tip=tip, xf=xf,
                       opening=float(np.clip((grip - g.closed_pct)
                                             / max(g.open_pct - g.closed_pct, 1e-6), 0, 1)),
                       tubes=tubes(), racks=racks(), caps=caps, phase=ph, note=note,
                       joints=[round(float(v), 2) for v in ms.observe(False)[0]],
                       joint_names=list(ms.ARM.joint_names))

    @app.route("/state")
    def r_state():
        with ms.lock:
            held = bool(ms.carry["held"])
        with lock:
            res = list(tstate["results"])
            s = {k: tstate[k] for k in ("phase", "note", "running")}
            s["log"] = list(tstate["log"])[-150:]
        tags = {}
        for r in res:
            tags[r["tag"]] = tags.get(r["tag"], 0) + 1
        s.update(arm=ms.ARM.name, simulated=False, held=held, grip_detail="",
                 gripper=round(float(ms.state.get("gripper") or 0.0), 1),
                 episodes=episodes.tally("tube_pick"), target=None, dest=None,
                 results=res[-30:], tags=tags)
        return jsonify(s)

    @app.route("/stream")
    def r_stream():
        return ms.app.view_functions["stream"]()

    @app.route("/scan", methods=["POST"])
    def r_scan():
        def job():
            tphase("SCAN", "mapping the mat with the wrist camera")
            n = do_scan()
            tphase("IDLE", f"{n} tube{'' if n == 1 else 's'} on the map")
        return start_job(job)

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

        def job():
            ok, tag = pick_and_place(t, rack_name)
            arm.move(arm.home, speed=1.4, settle=0.2)
            tphase("DONE" if ok else "FAILED", tag)
        return start_job(job)

    @app.route("/pickall", methods=["POST"])
    def r_pickall():
        """Map once, then every tube into its colour's rack, sweeping one way across the
        mat. A tube that fails twice comes off the list."""
        def job():
            tphase("SCAN", "mapping the mat with the wrist camera")
            do_scan()
            tally, fails = {}, {}
            while True:
                ms.checkpoint()
                todo = sorted((t for t in tubes() if fails.get(t["id"], 0) < 2),
                              key=lambda t: math.atan2(t["y"], t["x"]))
                if not todo:
                    break
                t = todo[0]
                tphase("PICK", f"{t['colour']} tube at ({t['x']*100:+.0f},{t['y']*100:+.0f})cm"
                               f" — {len(todo)} left")
                ok, tag = pick_and_place(t, RACK_FOR_COLOUR.get(t["colour"]))
                tally[tag] = tally.get(tag, 0) + 1
                if tag == "stopped":
                    break
                if not ok:
                    fails[t["id"]] = fails.get(t["id"], 0) + 1
                arm.move(arm.home, speed=1.4, settle=0.2)
            summary = ", ".join(f"{v} {k}" for k, v in sorted(tally.items())) or "nothing to do"
            placed = tally.get("placed", 0) + tally.get("placed (unverified)", 0)
            tphase("DONE" if placed and placed == sum(tally.values()) else "PARTIAL", summary)
        return start_job(job)

    @app.route("/grip", methods=["POST"])
    def r_grip():
        """Set the jaws only: ?pct=45 opens, ?pct=0 closes."""
        pct = float(request.args.get("pct", 45))
        ms.send_joints(ms.observe(False)[0], gripper=pct)
        return jsonify(ok=True, pct=pct)

    @app.route("/droptest", methods=["POST"])
    def r_droptest():
        """Only the upright drop, on a tube already in the jaws: ?colour=blue"""
        colour = (request.args.get("colour") or "").strip()
        if colour not in RACK_FOR_COLOUR:
            return jsonify(ok=False, error=f"colour must be one of {list(RACK_FOR_COLOUR)}"), 400

        def job():
            tphase("DROPTEST", f"upright drop only, {colour} tube in the jaws")
            try:
                ok, tag, d = place_upright(colour)
            except Exception as e:
                ok, tag, d = False, tag_of(e), f"{type(e).__name__}: {e}"
            tsay(f"[{tag.upper()}] {colour}: {d}")
            tphase("DONE" if ok else "FAILED", tag)
        return start_job(job)

    @app.route("/clearmap", methods=["POST"])
    def r_clearmap():
        with lock:
            tube_map.clear()
            tstate["used_holes"] = {}
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
        """The top camera, zoomed to the work area, with every rack hole marked."""
        import requests
        from flask import Response

        def gen():
            try:
                s = requests.Session()
                s.post(ms.CAMSURV[0] + "/", data={"password": ms.CAMSURV[1]}, timeout=5)
                r = s.get(ms.CAMSURV[0] + "/stream/" + ms.CAMSURV_STREAM, stream=True,
                          timeout=10)
                buf = b""
                for chunk in r.iter_content(chunk_size=16384):
                    buf += chunk
                    a = buf.find(b"\xff\xd8")
                    b = buf.find(b"\xff\xd9", a + 2) if a != -1 else -1
                    if b == -1:
                        continue
                    img = cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR)
                    buf = buf[b + 2:]
                    if img is None:
                        continue
                    for rk in TOP_RACKS:
                        bgr = tuple(int(rk["colour"][i:i + 2], 16) for i in (5, 3, 1))
                        for k, (u, v) in enumerate(rk["holes_px"]):
                            cv2.circle(img, (int(u), int(v)), 8, bgr, 1, cv2.LINE_AA)
                            cv2.putText(img, str(k), (int(u) - 4, int(v) + 3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, bgr, 1, cv2.LINE_AA)
                    if ms.TOP_ROI is not None:
                        x0, y0, x1, y1 = ms.TOP_ROI
                        img = cv2.resize(img[y0:y1, x0:x1],
                                         (1280, int(1280 * (y1 - y0) / (x1 - x0))))
                    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            except Exception:
                return
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/")
    def index():
        resp = send_from_directory(TUBE_UI, "tube.html")
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    @app.route("/ui/<path:name>")
    def ui_asset(name):
        if "/" in name or "\\" in name or name.startswith("."):
            return ("no", 404)
        return send_from_directory(TUBE_UI, name)

    ms.OVERLAY_HOOKS.append(cap_overlay)
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=port, threaded=True,
                                            use_reloader=False), daemon=True).start()
    ms.say(f"tube UI: http://127.0.0.1:{port}/  (SO-101, in this process)")
