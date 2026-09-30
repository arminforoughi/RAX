"""Tube sorting: map the mat, pick each tube, stand it up in its rack.

A UI on the shared rig (examples/rig_app.py), served on :8486 by examples/server.py
next to the general pick UI on :8484.

The pick and the place are the generic ones in :mod:`rax.pick`; this file adds what is
specific to this bench: which caps count, where the racks are (from an overhead
camera), which rack each colour goes to, the upright drop, and the UI.

Optional overhead camera (a CamSurv MJPEG server): RAX_CAMSURV_URL,
RAX_CAMSURV_PASSWORD, RAX_CAMSURV_STREAM. Without it the drop goes to the hole's
estimated position and is not verified.
"""

from __future__ import annotations

import math
import os
import sys
import time

import cv2
import numpy as np
from flask import Response, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rig_app import App, overhead_frame  # noqa: E402

from rax.perception.tube_caps import find_caps  # noqa: E402
from rax.pick import ColourTarget, PickConfig, pick, place, scan  # noqa: E402
from rax.pick.arm import bearing_of, move_to, solve  # noqa: E402
from rax.pick.episodes import EpisodeLog  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(HERE, "ui")

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
    target_v=405.0,         # reach until the cap is in the grip cells (the grid's bottom
    along_tol_px=18.0,      # row, 384-480px): the operator's "put it in the grid"
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
TOP_ROI = (164, 80, 1017, 560)   # the work area, shown zoomed on /topstream

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
#: Pace of everything but the pick (scan, carry, drop, going home).
FAST = 1.35

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


def top_caps(colour):
    """Caps of this colour in the top view, as [(u, v)], or None without a frame."""
    img = overhead_frame()
    if img is None:
        return None
    x0, y0, x1, y1 = TOP_TABLE_ROI
    return [(c.x + x0, c.y + y0)
            for c in find_caps(img[y0:y1, x0:x1], colours=(colour,), min_area=12,
                               max_area=900, gates=TOP_CAP_GATES)]


class TubeApp(App):
    """The state of one sorting session: the arm, the map, the racks and the results."""

    name = "tube"

    def __init__(self, rig):
        super().__init__(rig)
        arm = self.arm
        self.held = False
        self.used_holes: dict[str, list[int]] = {}
        self.tube_map: dict[int, dict] = {}
        self.focus = {"colour": None, "xy": None}
        self.caps: list[dict] = []
        self.episodes = EpisodeLog(os.path.join(HERE, "episodes.jsonl"),
                                   probe=lambda: {"joints": [round(float(v), 1) for v in arm.q]},
                                   note=self.say)
        add_routes(self)

    def start_job(self, target):
        """Tube jobs move at the tube pace (the pick inside keeps its own)."""
        def run():
            self.arm.pace = FAST
            try:
                target()
            finally:
                self.arm.pace = 1.0
        return super().start_job(run)

    def holding(self):
        return self.arm.gripper_pct > GRIP_BLOCKED_PCT

    # ---- the wrist view -----------------------------------------------------------
    def draw(self, img, q):
        """The grid, the grip cells (between the two fingertips) and the caps."""
        clean = img.copy()
        self.draw_grid(img)
        h, w = img.shape[:2]
        (fu, fv), (ju, jv) = self.arm.p.gripper.hand_uv, self.arm.jaw_uv
        mu, mv = 2 * ju - fu, 2 * jv - fv            # the other fingertip
        cells = {(min(5, int((fu + (mu - fu) * t) * 6 / w)), min(4, int((fv + (mv - fv) * t) * 5 / h)))
                 for t in np.linspace(0, 1, 21)}
        for c, r in cells:
            cv2.rectangle(img, (c * w // 6, r * h // 5), ((c + 1) * w // 6, (r + 1) * h // 5),
                          (0, 220, 220), 1)
        dets = TUBE.detect(clean)
        dets = [d for d in dets if not in_rack_zone(self.arm.cast((d.u, d.v), q) or (9, 9))]
        if self.focus["colour"] is not None:
            dets = [d for d in dets if d.label == self.focus["colour"]]
        self.caps = [{"colour": d.label, "x": d.u, "y": d.v, "area": d.area,
                      "bbox": list(d.box)} for d in dets]
        for d in dets:
            x0, y0, x1, y1 = (int(v) for v in d.box)
            cv2.rectangle(img, (x0, y0), (x1, y1), (60, 220, 90), 2)
            cv2.putText(img, f"{d.label} cap", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (60, 220, 90), 1, cv2.LINE_AA)
            cv2.line(img, (int(d.u), int(d.v)), (int(ju), int(jv)), (0, 255, 255), 1,
                     cv2.LINE_AA)

    # ---- the map --------------------------------------------------------------------
    def tubes(self):
        now, f = time.time(), self.focus
        with self.rig.lock:
            return [{"id": e["id"], "colour": e["colour"], "x": round(e["x"], 4),
                     "y": round(e["y"], 4), "z": 0.0, "held": False, "rack": None,
                     "hole": None, "source": "mapped", "d": TUBE_D_M, "l": TUBE_L_M,
                     "standing": None, "yaw": 0.0, "yaw_known": False, "n": e["n"],
                     "age": round(now - e["t"], 1), "label": f"{e['colour']} tube",
                     "focus": bool(f["xy"] is not None and e["colour"] == f["colour"]
                                   and math.hypot(e["x"] - f["xy"][0], e["y"] - f["xy"][1]) < 0.03)}
                    for e in sorted(self.tube_map.values(), key=lambda v: v["id"])
                    if not e["picked"]]

    def do_scan(self):
        found = scan(self.arm, TUBE, SCAN_BEARINGS_DEG, SCAN_TILTS_DEG, max_v=MAP_MAX_Y_PX,
                     avoid=in_rack_zone,
                     reach=(self.arm.p.reach_min_m, self.arm.p.reach_max_m))
        with self.rig.lock:
            self.tube_map = {i: {"id": i, "colour": f.label, "x": f.x, "y": f.y, "n": f.n,
                                 "t": time.time(), "picked": False}
                             for i, f in enumerate(found, 1)}
        self.say(f"        wrist scan: {len(found)} tube(s) mapped")
        return len(found)

    @staticmethod
    def racks():
        out = []
        for r in TOP_RACKS:
            hs = [top_px_to_xy(u, v) for u, v in r["holes_px"]]
            out.append({"name": r["name"], "colour": r["colour"], "yaw": 0.0,
                        "x": round(sum(h[0] for h in hs) / len(hs), 4),
                        "y": round(sum(h[1] for h in hs) / len(hs), 4),
                        "holes": [[round(h[0], 4), round(h[1], 4)] for h in hs]})
        return out

    # ---- the upright drop -------------------------------------------------------
    def held_cap(self, colour, static):
        """The held cap in the top view: not one of ``static``, near the fingertip's pixel."""
        caps = top_caps(colour) or []
        tip = self.arm.tip(self.arm.joints())
        pu, pv = xy_to_top_px(float(tip[0]), float(tip[1]))
        mine = [c for c in caps
                if all(math.hypot(c[0] - s[0], c[1] - s[1]) > 8.0 for s in static)
                and math.hypot(c[0] - pu, c[1] - pv) <= TOP_HELD_SEARCH_PX]
        return min(mine, key=lambda c: math.hypot(c[0] - pu, c[1] - pv), default=None)

    def place_upright(self, colour, rack_name=None):
        """Stand the held tube up and drop it in a free hole. Returns (ok, tag, detail)."""
        arm = self.arm
        rack = next(r for r in TOP_RACKS if r["name"] == (rack_name or RACK_FOR_COLOUR[colour]))
        used = self.used_holes.setdefault(rack["name"], [])

        self.phase("STAND", f"standing the {colour} tube up")
        b0 = bearing_of(arm.tip(arm.joints()))
        before = top_caps(colour) or []
        j5 = float(arm.joints()[arm.roll])     # the nearer of +-90: the wrist never turns over
        roll = STAND_ROLL_DEG if abs(j5 - STAND_ROLL_DEG) <= abs(j5 + STAND_ROLL_DEG) \
            else -STAND_ROLL_DEG
        move_to(arm, (STAND_R_M * math.cos(b0), STAND_R_M * math.sin(b0), STAND_Z_M),
                0.0, roll, "the level hand", speed=1.2)
        if not self.holding():
            raise RuntimeError("dropped the tube while standing it up")
        held = self.held_cap(colour, before)
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
        self.phase("CARRY", f"over {rack['name']} hole {k}")

        def align():
            cap = self.held_cap(colour, static)
            if cap is None:
                return None
            du, dv = hole_px[0] - cap[0], hole_px[1] - cap[1]
            self.say(f"        align: cap {math.hypot(du, dv):.0f}px from the hole")
            return (0.0, 0.0) if math.hypot(du, dv) <= TOP_ALIGN_TOL_PX \
                else top_px_delta_to_xy(du, dv)

        place(arm, (hx, hy), release_z=z_base + DROP_CLEAR_M, pitch=0.0, roll=roll,
              hover_z=z_base + HOVER_CLEAR_M + z_extra, carry_z=CARRY_Z_M,
              align=align if held else None, holding=self.holding)
        self.held = False
        used.append(k)

        # did it go in? look from above with the arm out of the way
        arm.move(arm.home, speed=1.4, settle=0.2)
        caps = top_caps(colour)
        if caps is None:
            return True, "placed (unverified)", f"{rack['name']} hole {k}, no top view"
        near = min((math.hypot(c[0] - hole_px[0], c[1] - hole_px[1]) for c in caps), default=None)
        if near is not None and near <= HOLE_VERIFY_PX:
            return True, "placed", f"in {rack['name']} hole {k}"
        return False, "missed hole", f"dropped at {rack['name']} hole {k}, no cap there after"

    # ---- one tube, start to finish ---------------------------------------------------
    def pick_and_place(self, t, rack_name):
        """Returns (ok, tag). Never raises."""
        arm = self.arm
        colour, xy = t["colour"], (t["x"], t["y"])
        self.focus.update(colour=colour, xy=xy)
        ep = self.episodes.start("tube_pick", t["label"], arm=self.arm.p.name, simulated=False,
                                 dest=rack_name)
        try:
            if math.degrees(bearing_of(xy)) < PICK_MIN_BEARING_DEG:
                raise RuntimeError("not reachable: past the guardrail, toward the racks")
            arm.min_bearing_deg, arm.pace = PICK_MIN_BEARING_DEG, 1.0   # the pick's own pace
            try:
                res = pick(arm, TUBE, near_xy=xy, label=colour, cfg=RIG, avoid=in_rack_zone)
            finally:
                arm.min_bearing_deg, arm.pace = None, FAST
            self.held = True
            if rack_name is None:
                ok, tag, detail = True, "picked", f"jaws at {res.grip_pct:.1f}"
            else:
                ok, tag, detail = self.place_upright(colour, rack_name)
        except Exception as e:
            ok, tag, detail = False, tag_of(e), f"{type(e).__name__}: {e}"
            try:
                arm.release()
                self.held = False
            except Exception:
                pass
        self.focus.update(colour=None, xy=None)
        self.episodes.end(ep, ok, f"[{tag}] {detail}")
        with self.rig.lock:
            self.rig.results.append({"t": time.strftime("%H:%M:%S"), "colour": colour,
                                 "ok": bool(ok), "tag": tag, "detail": str(detail)[:160],
                                 "x": round(xy[0], 3), "y": round(xy[1], 3)})
            del self.rig.results[:-60]
            if ok and t["id"] in self.tube_map:
                self.tube_map[t["id"]]["picked"] = True
        self.say(f"[{tag.upper()}] {colour}: {detail}")
        return ok, tag

    def pick_all(self):
        """Map once, then every tube into its colour's rack, sweeping one way across the
        mat. A tube that fails twice comes off the list."""
        self.phase("SCAN", "mapping the mat with the wrist camera")
        self.do_scan()
        tally, fails = {}, {}
        while True:
            self.arm.checkpoint()
            todo = sorted((t for t in self.tubes() if fails.get(t["id"], 0) < 2),
                          key=lambda t: math.atan2(t["y"], t["x"]))
            if not todo:
                break
            t = todo[0]
            self.phase("PICK", f"{t['colour']} tube at ({t['x']*100:+.0f},{t['y']*100:+.0f})cm"
                               f" — {len(todo)} left")
            ok, tag = self.pick_and_place(t, RACK_FOR_COLOUR.get(t["colour"]))
            tally[tag] = tally.get(tag, 0) + 1
            if tag == "stopped":
                break
            if not ok:
                fails[t["id"]] = fails.get(t["id"], 0) + 1
            self.arm.move(self.arm.home, speed=1.4, settle=0.2)
        summary = ", ".join(f"{v} {k}" for k, v in sorted(tally.items())) or "nothing to do"
        placed = tally.get("placed", 0) + tally.get("placed (unverified)", 0)
        self.phase("DONE" if placed and placed == sum(tally.values()) else "PARTIAL", summary)

def add_routes(tube: TubeApp) -> None:
    app, arm = tube.app, tube.arm
    @app.route("/geom")
    def r_geom():
        g = arm.p.gripper
        return jsonify(tube.geom(dict(
            simulated=False, tubes=tube.tubes(), racks=tube.racks(), caps=tube.caps,
            phase=tube.rig.phase_name, note=tube.rig.note,
            opening=float(np.clip((arm.gripper_pct - g.closed_pct)
                                  / (g.open_pct - g.closed_pct), 0, 1)))))

    @app.route("/state")
    def r_state():
        rig = tube.rig
        with rig.lock:
            res = list(rig.results)
            log = list(rig.log)[-150:]
        tags = {}
        for r in res:
            tags[r["tag"]] = tags.get(r["tag"], 0) + 1
        return jsonify(arm=arm.p.name, simulated=False, phase=rig.phase_name, note=rig.note,
                       running=rig.running, held=tube.held, grip_detail="",
                       gripper=round(arm.gripper_pct, 1), log=log,
                       episodes=tube.episodes.tally("tube_pick"), target=None, dest=None,
                       results=res[-30:], tags=tags)

    @app.route("/topstream")
    def r_topstream():
        """The overhead camera, zoomed to the work area, with every rack hole marked."""
        def gen():
            while True:
                img = overhead_frame()
                if img is None:
                    return
                for rk in TOP_RACKS:
                    bgr = tuple(int(rk["colour"][i:i + 2], 16) for i in (5, 3, 1))
                    for k, (u, v) in enumerate(rk["holes_px"]):
                        cv2.circle(img, (int(u), int(v)), 8, bgr, 1, cv2.LINE_AA)
                        cv2.putText(img, str(k), (int(u) - 4, int(v) + 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, bgr, 1, cv2.LINE_AA)
                x0, y0, x1, y1 = TOP_ROI
                img = cv2.resize(img[y0:y1, x0:x1], (1280, int(1280 * (y1 - y0) / (x1 - x0))))
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/scan", methods=["POST"])
    def r_scan():
        def job():
            tube.phase("SCAN", "mapping the mat with the wrist camera")
            n = tube.do_scan()
            tube.phase("IDLE", f"{n} tube{'' if n == 1 else 's'} on the map")
        return tube.start_job(job)

    @app.route("/pick", methods=["POST"])
    def r_pick():
        """Pick one mapped tube; with ``rack`` set, stand it up and drop it in that rack."""
        d = request.get_json(silent=True) or request.form or {}
        found = [t for t in tube.tubes() if str(t["id"]) == str(d.get("tube"))]
        if not found:
            return jsonify(ok=False, error=f"tube {d.get('tube')} is not on the map"), 404
        t = found[0]
        rack_name = d.get("rack") or None
        if rack_name == "auto":
            rack_name = RACK_FOR_COLOUR.get(t["colour"])
        if rack_name is not None and rack_name not in [r["name"] for r in TOP_RACKS]:
            return jsonify(ok=False, error=f"no rack called {rack_name!r}"), 400

        def job():
            ok, tag = tube.pick_and_place(t, rack_name)
            arm.move(arm.home, speed=1.4, settle=0.2)
            tube.phase("DONE" if ok else "FAILED", tag)
        return tube.start_job(job)

    @app.route("/pickall", methods=["POST"])
    def r_pickall():
        return tube.start_job(tube.pick_all)

    @app.route("/droptest", methods=["POST"])
    def r_droptest():
        """Only the upright drop, on a tube already in the jaws: ?colour=blue"""
        colour = (request.args.get("colour") or "").strip()
        if colour not in RACK_FOR_COLOUR:
            return jsonify(ok=False, error=f"colour must be one of {list(RACK_FOR_COLOUR)}"), 400

        def job():
            try:
                ok, tag, d = tube.place_upright(colour)
            except Exception as e:
                ok, tag, d = False, tag_of(e), f"{type(e).__name__}: {e}"
            tube.say(f"[{tag.upper()}] {colour}: {d}")
            tube.phase("DONE" if ok else "FAILED", tag)
        return tube.start_job(job)

    @app.route("/grip", methods=["POST"])
    def r_grip():
        """Set the jaws only: ?pct=45 opens, ?pct=0 closes."""
        pct = float(request.args.get("pct", 45))
        return tube.start_job(lambda: arm.grip(pct))

    @app.route("/clearmap", methods=["POST"])
    def r_clearmap():
        with tube.rig.lock:
            tube.tube_map, tube.used_holes = {}, {}
            tube.rig.results.clear()
        tube.say("map, used holes and results cleared")
        return jsonify(ok=True)

    @app.route("/episodes")
    def r_episodes():
        return jsonify(tally=tube.episodes.tally("tube_pick"),
                       recent=tube.episodes.records("tube_pick")[-25:])

    @app.route("/")
    def index():
        resp = send_from_directory(UI, "tube.html")
        resp.headers["Cache-Control"] = "no-store"
        return resp
