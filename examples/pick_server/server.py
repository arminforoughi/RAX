"""The general pick UI: name an object, map the table, pick it, place it.

A UI on the shared rig (examples/rig_app.py), served on :8484 by examples/server.py.
Objects are found by YOLO-World from the query (``pip install "rax[detect]"``); the
pick is :func:`rax.pick.pick`, the same on every arm.
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading
import time

import cv2
from flask import jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rig_app import App  # noqa: E402

from rax.perception.handeye import calibrate  # noqa: E402
from rax.pick import SIDE, TOP, PromptTarget, StickyTarget, pick, place, scan  # noqa: E402
from rax.pick.arm import pitch_of, steepest  # noqa: E402
from rax.pick.episodes import EpisodeLog  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(HERE, "ui")

#: Query presets, as the old panel had them.
PRESETS = {
    "cubes": "red cube, green cube",
    "table": "cup, bottle, block, pen, tube, box, ball, toy",
    "coco": ("person, bicycle, car, bottle, wine glass, cup, fork, knife, spoon, bowl, "
             "banana, apple, orange, carrot, book, clock, vase, scissors, teddy bear, "
             "toothbrush, cell phone, mouse, remote, keyboard"),
}


class PickApp(App):
    name = "pick"

    def __init__(self, rig):
        super().__init__(rig)
        self.target = self.make_target(os.environ.get("RAX_PROMPT", "red cube, green cube"))
        self.cfg = None                        # the arm's own tuning (PickConfig.for_arm)
        self.objects: list[dict] = []          # the 2D map
        self.dets: list = []                   # latest detections, for the view
        self.held: str | None = None
        self.episodes = EpisodeLog(os.path.join(HERE, "episodes.jsonl"),
                                   probe=lambda: {"joints": [round(float(v), 1)
                                                             for v in self.arm.q]},
                                   note=self.say)
        self._routes()

    @staticmethod
    def make_target(prompt):
        """YOLO-World finds it; OpenCV holds on to it when YOLO drops out for a moment."""
        return StickyTarget(PromptTarget(prompt=prompt, grasp_z=0.02))

    # ---- detection runs on its own thread: the model is slower than the camera ------
    def detect_loop(self):
        import traceback
        n, t_log = 0, 0.0
        while True:
            rgb = self.arm.rgb
            if rgb is not None:
                try:
                    t0 = time.time()
                    self.dets = self.target.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    n += 1
                    if n == 1 or time.time() - t_log > 60:
                        t_log = time.time()
                        self.say(f"detector: '{self.target.prompt}' {time.time() - t0:.2f}s, "
                                 f"{len(self.dets)} in view")
                except Exception as e:
                    self.say(f"detector: {type(e).__name__}: {e}")
                    print(traceback.format_exc(), flush=True)
                    time.sleep(5.0)
            time.sleep(0.4)

    def draw(self, img, q):
        self.draw_grid(img)
        ju, jv = (int(v) for v in self.arm.jaw_uv)
        for d in self.dets:
            x0, y0, x1, y1 = (int(v) for v in d.box)
            # green: YOLO; magenta: found by its colour; orange: held by the tracker
            col = {"tracked": (0, 160, 255), "colour": (255, 80, 220)}.get(d.source,
                                                                         (60, 220, 90))
            tag = {"tracked": " (held)", "colour": " (colour)"}.get(d.source, "")
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            cv2.putText(img, d.label + tag, (x0, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            cv2.line(img, (int(d.u), int(d.v)), (ju, jv), (0, 255, 255), 1, cv2.LINE_AA)

    # ---- the map --------------------------------------------------------------------
    def map2d(self):
        with self.rig.lock:
            return [{"tag": o["id"], "label": o["label"], "x": o["x"], "y": o["y"], "n": o["n"],
                     "w_m": 0.03, "d_m": 0.03, "h_m": 0.03, "size": 0.03, "yaw": 0.0,
                     "yaw_known": False, "measured": False, "shape": "object",
                     "r_cm": round(math.hypot(o["x"], o["y"]) * 100)} for o in self.objects]

    def find(self, tag=None, word=None):
        for o in self.objects:
            if (tag is not None and str(o["id"]) == str(tag)) or \
                    (word is not None and word.lower() in o["label"].lower()):
                return o
        return None

    # ---- jobs ----------------------------------------------------------------------
    def do_scan(self):
        self.phase("SCAN", f"looking for '{self.target.prompt}'")
        found = scan(self.arm, self.target,
                     reach=(self.arm.p.reach_min_m, self.arm.p.reach_max_m))
        with self.rig.lock:
            self.objects = [{"id": i, "label": f.label, "x": round(f.x, 4),
                             "y": round(f.y, 4), "n": f.n} for i, f in enumerate(found, 1)]
        self.phase("IDLE", f"{len(found)} object(s) on the map")

    def do_pick(self, obj=None, grasp=TOP):
        """Pick ``obj`` from the map, or whatever of the query is in view. True if held."""
        label = obj["label"] if obj else self.target.prompt
        near = (obj["x"], obj["y"]) if obj else None
        ep = self.episodes.start("pick", label, arm=self.arm.p.name, simulated=False)
        try:
            res = pick(self.arm, self.target, near_xy=near, grasp=grasp, cfg=self.cfg,
                       label=obj["label"] if obj else None)
            ok, tag, detail = True, "picked", f"jaws at {res.grip_pct:.1f}"
            self.held = label
            if obj:
                with self.rig.lock:
                    self.objects = [o for o in self.objects if o["id"] != obj["id"]]
        except Exception as e:
            ok, tag, detail = False, "failed", f"{type(e).__name__}: {e}"
            try:
                self.arm.release()
            except Exception:
                pass
        self.episodes.end(ep, ok, f"[{tag}] {detail}")
        self.record(label, ok, tag, detail, near)
        return ok

    def do_place(self, dest, grasp=TOP):
        """Put the held object at ``dest``: a map object (stack on it) or an (x, y)."""
        if isinstance(dest, dict):
            xy, z = (dest["x"], dest["y"]), self.target.grasp_z * 2 + 0.04   # on top of it
        else:
            xy, z = dest, self.target.grasp_z + 0.02
        roll = float(self.arm.joints()[self.arm.roll])
        at = place(self.arm, xy, release_z=z, pitch=grasp.pitch, roll=roll)
        self.record(self.held or "object", True, "placed",
                    f"at ({at[0]*100:+.0f},{at[1]*100:+.0f})cm", xy)
        self.held = None

    def do_task(self, steps, grasp):
        """'green on red, blue on green': pick each object from the map, stack it."""
        for what, where in steps:
            self.arm.checkpoint()
            obj, dest = self.find(word=what), self.find(word=where)
            if obj is None or dest is None:
                self.phase("FAILED", f"'{what}' or '{where}' is not on the map")
                return
            self.phase("TASK", f"{what} on {where}")
            if not self.do_pick(obj, grasp):
                self.phase("FAILED", f"could not pick {what}")
                return
            self.do_place(dest, grasp)
        self.arm.move(self.arm.home, speed=1.2)
        self.phase("DONE", "task finished")

    def goto(self, obj):
        """Hover over a mapped object, hand pointing down."""
        q, p = steepest(self.arm, (obj["x"], obj["y"], 0.10), 90.0, 45.0,
                        float(self.arm.joints()[self.arm.roll]))
        if q is None:
            raise RuntimeError("cannot reach above it")
        self.arm.move(q, speed=1.0)
        self.phase("IDLE", f"over {obj['label']} at {p:.0f}deg")

    # ---- routes --------------------------------------------------------------------
    def _routes(self):
        app, arm, rig = self.app, self.arm, self.rig

        def grasp_arg():
            return SIDE if request.args.get("grasp") == "side" else TOP

        def dest_arg():
            if request.args.get("tag") is not None:
                return self.find(tag=request.args["tag"])
            if request.args.get("x") is not None:
                return (float(request.args["x"]) / 100, float(request.args["y"]) / 100)
            return None

        @app.route("/")
        def index():
            resp = send_from_directory(UI, "admin.html")
            resp.headers["Cache-Control"] = "no-store"
            return resp

        @app.route("/ui/<name>")
        def ui_asset(name):
            return send_from_directory(UI, name)

        @app.route("/status")
        def r_status():
            q = arm.q
            tip = arm.tip(q)
            with rig.lock:
                log = [f"{entry['t']} {entry['m']}" for entry in rig.log[-150:]]
            return jsonify(arm=arm.p.name, phase=rig.phase_name, detail=rig.note,
                           running=rig.running, t0=rig.t0, query=self.target.prompt,
                           joints=[round(float(v), 1) for v in q],
                           gripper=round(arm.gripper_pct, 1),
                           jog_xyz=[round(float(v), 3) for v in tip],
                           pitch=round(pitch_of(arm, q)), carry=self.held,
                           relaxed=arm.relaxed, has_handeye=arm.has_handeye, log=log,
                           detections=[{"label": d.label, "u": round(d.u), "v": round(d.v),
                                        "source": d.source} for d in self.dets])

        @app.route("/geom")
        def r_geom():
            objs2d = [{"x": o["x"], "y": o["y"], "w": 0.03, "d": 0.03, "h": 0.03, "yaw": 0,
                       "label": o["label"], "tag": o["tag"]} for o in self.map2d()]
            return jsonify(self.geom({"objs2d": objs2d}))

        @app.route("/setquery", methods=["POST"])
        def r_setquery():
            q = (request.args.get("q") or "").strip()
            if not q:
                return jsonify(ok=False)
            self.target = self.make_target(q)
            self.say(f"query: '{q}'")
            return jsonify(ok=True)

        @app.route("/preset")
        def r_preset():
            return jsonify(q=PRESETS.get(request.args.get("name", ""), ""))

        @app.route("/start", methods=["POST"])
        def r_start():
            g = grasp_arg()

            def job():
                ok = self.do_pick(None, g)
                arm.move(arm.home, speed=1.2)
                self.phase("DONE" if ok else "FAILED", "holding it" if ok else "missed")
            return self.start_job(job)

        @app.route("/survey", methods=["POST"])
        def r_survey():
            """Where is the query object in view? Reports it, moves nothing."""
            dets = list(self.dets)
            if not dets:
                self.say("locate: nothing of the query in view")
            for d in dets:
                xy = arm.cast((d.u, d.v), arm.q)
                self.say(f"locate: {d.label} at ({d.u:.0f},{d.v:.0f})px"
                         + ("" if xy is None else f" -> ({xy[0]*100:+.1f},{xy[1]*100:+.1f})cm"))
            return jsonify(ok=True)

        @app.route("/scan2d", methods=["POST"])
        def r_scan2d():
            return self.start_job(self.do_scan)

        @app.route("/map2d")
        def r_map2d():
            return jsonify(objs=self.map2d())

        @app.route("/clearmap2d", methods=["POST"])
        def r_clearmap2d():
            with rig.lock:
                self.objects = []
            return jsonify(ok=True)

        @app.route("/goto2d", methods=["POST"])
        def r_goto2d():
            obj = self.find(tag=request.args.get("tag"))
            if obj is None:
                return jsonify(ok=False, reason="not on the map"), 404
            return self.start_job(lambda: self.goto(obj))

        @app.route("/place", methods=["POST"])
        def r_place():
            dest, g = dest_arg(), grasp_arg()
            if dest is None:
                return jsonify(ok=False, reason="where? click an object or the table")

            def job():
                self.do_place(dest, g)
                arm.move(arm.home, speed=1.2)
            return self.start_job(job)

        @app.route("/pickplace", methods=["POST"])
        def r_pickplace():
            dest, g = dest_arg(), grasp_arg()
            if dest is None:
                return jsonify(ok=False, reason="where? click an object or the table")

            def job():
                if self.do_pick(None, g):
                    self.do_place(dest, g)
                arm.move(arm.home, speed=1.2)
            return self.start_job(job)

        @app.route("/task", methods=["POST"])
        def r_task():
            text = request.args.get("q", "")
            steps = [tuple(p.strip() for p in m) for m in
                     re.findall(r"\s*([\w ]+?)\s+on\s+([\w ]+?)\s*(?:,|$)", text)]
            if not steps:
                return jsonify(ok=False, reason="write it as: green on red, blue on green")
            g = grasp_arg()
            return self.start_job(lambda: self.do_task(steps, g))

        @app.route("/calibrate", methods=["POST"])
        def r_calibrate():
            """Fit the wrist camera's pose from one still object of the query, in view.
            Saved per arm and loaded automatically from then on."""
            path = os.path.join(os.path.dirname(HERE), f"handeye_{arm.p.name}.json")

            def job():
                fit = calibrate(arm, self.target, save_to=path)
                self.phase("DONE" if fit.converged else "FAILED",
                           "camera calibrated" if fit.converged else fit.reason)
            return self.start_job(job)

    def start(self):
        threading.Thread(target=self.detect_loop, daemon=True).start()
