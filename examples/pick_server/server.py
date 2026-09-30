"""Pick anything you can name: type a prompt, map the table, pick, place.

    python examples/pick_server/server.py --arm so101 --port COM4      # UI on :8484
    python examples/pick_server/server.py --arm x250  --port COM5 --camera 1

Objects are found by YOLO-World from the prompt (``pip install "rax[detect]"``); the
pick is :func:`rax.pick.pick`, the same on every arm.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import cv2
from flask import jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rig_app import RigApp, arm_from_args  # noqa: E402

from rax.perception.handeye import calibrate  # noqa: E402
from rax.pick import SIDE, TOP, PickConfig, PromptTarget, pick, place, scan  # noqa: E402
from rax.pick.episodes import EpisodeLog  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class PickApp(RigApp):
    def __init__(self, arm):
        super().__init__(arm, "pick_server")
        self.target = PromptTarget(prompt=os.environ.get("RAX_PROMPT", "cup"), grasp_z=0.02)
        self.cfg = PickConfig()
        self.objects: list[dict] = []          # the map: {id, label, x, y, n}
        self.dets: list = []                   # latest detections, for the view
        self.episodes = EpisodeLog(os.path.join(HERE, "episodes.jsonl"),
                                   probe=lambda: {"joints": [round(float(v), 1)
                                                             for v in arm.q]},
                                   note=self.say)
        self._pick_routes()

    # ---- detection runs in its own thread: the model is slower than the camera ------
    def detect_loop(self):
        while True:
            rgb = self.arm.rgb
            if rgb is not None:
                try:
                    self.dets = self.target.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    self.say(f"detector: {type(e).__name__}: {e}")
                    time.sleep(5.0)
            time.sleep(0.4)

    def draw(self, img, q):
        self.draw_grid(img)
        ju, jv = (int(v) for v in self.arm.jaw_uv)
        for d in self.dets:
            x0, y0, x1, y1 = (int(v) for v in d.box)
            cv2.rectangle(img, (x0, y0), (x1, y1), (60, 220, 90), 2)
            cv2.putText(img, d.label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (60, 220, 90), 1, cv2.LINE_AA)
            cv2.line(img, (int(d.u), int(d.v)), (ju, jv), (0, 255, 255), 1, cv2.LINE_AA)

    # ---- jobs ----------------------------------------------------------------------
    def do_scan(self):
        self.phase("SCAN", f"looking for '{self.target.prompt}'")
        found = scan(self.arm, self.target,
                     reach=(self.arm.p.reach_min_m, self.arm.p.reach_max_m))
        with self.lock:
            self.objects = [{"id": i, "label": f.label, "x": round(f.x, 4),
                             "y": round(f.y, 4), "n": f.n} for i, f in enumerate(found, 1)]
        self.phase("IDLE", f"{len(found)} '{self.target.prompt}' on the map")

    def do_pick(self, obj, grasp, place_xy):
        label = obj["label"] if obj else self.target.prompt
        near = (obj["x"], obj["y"]) if obj else None
        ep = self.episodes.start("pick", label, arm=self.arm.p.name, simulated=False)
        try:
            res = pick(self.arm, self.target, near_xy=near, grasp=grasp, cfg=self.cfg)
            detail = f"jaws at {res.grip_pct:.1f}"
            if place_xy is not None:
                at = place(self.arm, place_xy, release_z=self.target.grasp_z + 0.03,
                           pitch=grasp.pitch, roll=float(self.arm.joints()[self.arm.roll]))
                detail += f", placed at ({at[0]*100:+.0f},{at[1]*100:+.0f})cm"
            ok, tag = True, "picked" if place_xy is None else "placed"
        except Exception as e:
            ok, tag, detail = False, "failed", f"{type(e).__name__}: {e}"
            try:
                self.arm.release()
            except Exception:
                pass
        self.episodes.end(ep, ok, f"[{tag}] {detail}")
        self.record(label, ok, tag, detail, near)
        if ok and obj:
            with self.lock:
                self.objects = [o for o in self.objects if o["id"] != obj["id"]]
        self.arm.move(self.arm.home, speed=1.2)
        self.phase("DONE" if ok else "FAILED", tag)

    # ---- routes --------------------------------------------------------------------
    def _pick_routes(self):
        app = self.app

        @app.route("/")
        def index():
            resp = send_from_directory(os.path.join(HERE, "ui"), "index.html")
            resp.headers["Cache-Control"] = "no-store"
            return resp

        @app.route("/state")
        def r_state():
            with self.lock:
                return jsonify(arm=self.arm.p.name, phase=self.phase_name, note=self.note,
                               running=self.running, prompt=self.target.prompt,
                               objects=list(self.objects), log=list(self.log)[-120:],
                               results=list(self.results)[-30:],
                               has_handeye=self.arm.has_handeye,
                               joints=[round(float(v), 1) for v in self.arm.q],
                               gripper=round(self.arm.gripper_pct, 1))

        @app.route("/prompt", methods=["POST"])
        def r_prompt():
            text = ((request.get_json(silent=True) or {}).get("prompt") or "").strip()
            if not text:
                return jsonify(ok=False, error="empty prompt"), 400
            self.target = PromptTarget(prompt=text, grasp_z=self.target.grasp_z)
            with self.lock:
                self.objects = []
            self.say(f"looking for '{text}' now")
            return jsonify(ok=True)

        @app.route("/scan", methods=["POST"])
        def r_scan():
            return self.start_job(self.do_scan)

        @app.route("/calibrate", methods=["POST"])
        def r_calibrate():
            """Fit the wrist camera's pose from one still object of the current prompt,
            in view. Saved per arm and loaded automatically from then on."""
            path = os.path.join(HERE, f"handeye_{self.arm.p.name}.json")

            def job():
                fit = calibrate(self.arm, self.target, save_to=path)
                self.phase("DONE" if fit.converged else "FAILED",
                           f"hand-eye {'saved to ' + os.path.basename(path) if fit.converged else fit.reason}")
            return self.start_job(job)

        @app.route("/pick", methods=["POST"])
        def r_pick():
            """{"id": n} picks a mapped object; no id picks what is in view.
            {"grasp": "top"|"side"}, {"place": [x, y]} (metres) to put it down."""
            d = request.get_json(silent=True) or {}
            obj = next((o for o in self.objects if str(o["id"]) == str(d.get("id"))), None)
            if d.get("id") is not None and obj is None:
                return jsonify(ok=False, error="no such object on the map"), 404
            grasp = SIDE if d.get("grasp") == "side" else TOP
            where = d.get("place")
            place_xy = (float(where[0]), float(where[1])) if where else None
            return self.start_job(lambda: self.do_pick(obj, grasp, place_xy))


def main():
    a, arm = arm_from_args(__doc__.split("\n\n")[0], 8484, HERE)
    app = PickApp(arm)
    threading.Thread(target=app.detect_loop, daemon=True).start()
    app.run(a.http)


if __name__ == "__main__":
    main()
