"""What every robot server here needs: a log, one job at a time, a live view, stop.

Both servers (``pick_server`` for any object, ``tube_sorting`` for tubes) build on
:class:`RigApp` and add their own routes.
"""

from __future__ import annotations

import argparse
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify

from rax.robots import ARMS, make_arm


def arm_from_args(description: str, http_default: int, here: str):
    """Parse the common flags and build (not connect) the arm."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--arm", default=os.environ.get("RAX_ARM", "so101"), choices=ARMS)
    ap.add_argument("--port", default=os.environ.get("RAX_ARM_PORT", ""),
                    help="the arm's serial port (COM4, /dev/ttyACM0, ...)")
    ap.add_argument("--camera", type=int, default=0, help="wrist camera index (x250)")
    ap.add_argument("--http", type=int, default=http_default, help="the UI's port")
    a, _ = ap.parse_known_args()
    handeye = os.path.join(here, f"handeye_{a.arm}.json")      # your own fit, if any
    if not os.path.exists(handeye):
        handeye = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"handeye_{a.arm}.example.json")
    kw = {"camera_index": a.camera} if a.arm == "x250" else {}
    return a, make_arm(a.arm, a.port, handeye_file=handeye, **kw)


class RigApp:
    """One arm, one job at a time, a rolling log and a rendered wrist view."""

    def __init__(self, arm, name: str):
        self.arm = arm
        self.lock = threading.Lock()
        self.phase_name, self.note = "IDLE", "ready"
        self.running = False
        self.log: list[dict] = []
        self.results: list[dict] = []
        self.jpeg = None
        self.app = Flask(name, static_folder=None)
        arm.log_fn, arm.phase_fn = self.say, self.phase
        self._routes()

    # ---- log --------------------------------------------------------------------
    def say(self, msg):
        print(msg, flush=True)
        with self.lock:
            self.log.append({"t": time.strftime("%H:%M:%S"), "m": str(msg)[:220]})
            del self.log[:-200]

    def phase(self, name, note=""):
        with self.lock:
            self.phase_name, self.note = name, note
        self.say(f"[{name}] {note}" if note else f"[{name}]")

    def record(self, label, ok, tag, detail, xy=None):
        with self.lock:
            self.results.append({"t": time.strftime("%H:%M:%S"), "colour": label,
                                 "label": label, "ok": bool(ok), "tag": tag,
                                 "detail": str(detail)[:160],
                                 "x": None if xy is None else round(float(xy[0]), 3),
                                 "y": None if xy is None else round(float(xy[1]), 3)})
            del self.results[:-60]
        self.say(f"[{tag.upper()}] {label}: {detail}")

    # ---- jobs -------------------------------------------------------------------
    def start_job(self, target):
        """Run ``target`` on the arm in the background, one job at a time."""
        with self.lock:
            if self.running:
                return jsonify(ok=False, error="the arm is busy"), 409
            self.running = True
        self.arm.stop_flag.clear()

        def go():
            try:
                target()
            except Exception as e:
                self.phase("FAILED", f"{type(e).__name__}: {e}")
            finally:
                with self.lock:
                    self.running = False
        threading.Thread(target=go, daemon=True).start()
        return jsonify(ok=True)

    # ---- the wrist view -----------------------------------------------------------
    def draw(self, bgr, q):
        """Draw on the wrist frame. Subclasses add their detections."""

    def draw_grid(self, img):
        h, w = img.shape[:2]
        for c in range(1, 6):
            cv2.line(img, (c * w // 6, 0), (c * w // 6, h), (80, 80, 80), 1)
        for r in range(1, 5):
            cv2.line(img, (0, r * h // 5), (w, r * h // 5), (80, 80, 80), 1)
        ju, jv = (int(v) for v in self.arm.jaw_uv)
        cv2.drawMarker(img, (ju, jv), (255, 120, 255), cv2.MARKER_TILTED_CROSS, 16, 2)

    def camera_loop(self):
        """Keep the view fresh: render the latest frame, and read one when idle."""
        last = None
        while True:
            try:
                if not self.running:
                    self.arm.observe(check_stop=False)
                rgb = self.arm.rgb
                if rgb is not None and rgb is not last:
                    last = rgb
                    img = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
                    self.draw(img, self.arm.q)
                    cv2.putText(img, self.phase_name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (90, 255, 90), 2, cv2.LINE_AA)
                    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        self.jpeg = jpg.tobytes()
            except Exception:
                pass
            time.sleep(0.1)

    # ---- common routes ------------------------------------------------------------
    def _routes(self):
        app, arm = self.app, self.arm

        @app.route("/stream")
        def r_stream():
            def gen():
                while True:
                    if self.jpeg is not None:
                        yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + self.jpeg + b"\r\n"
                    time.sleep(0.1)
            return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

        @app.route("/stop", methods=["POST"])
        def r_stop():
            arm.stop_flag.set()
            self.say("STOP requested")
            return jsonify(ok=True)

        @app.route("/reset", methods=["POST"])
        def r_reset():
            arm.stop_flag.clear()
            self.phase("IDLE", "ready")
            return jsonify(ok=True)

        @app.route("/home", methods=["POST"])
        def r_home():
            return self.start_job(lambda: arm.move(arm.home, speed=1.2))

        @app.route("/shutdown", methods=["POST"])
        def r_shutdown():
            """Release the camera and the bus, then exit. Use this, not a process kill:
            a killed process can leave the camera claimed by nobody."""
            def bye():
                time.sleep(0.3)
                arm.disconnect()
                os._exit(0)
            threading.Thread(target=bye, daemon=True).start()
            return jsonify(ok=True)

    def run(self, http: int):
        import atexit
        self.say(f"connecting the {self.arm.p.name} on {self.arm.port} ...")
        self.arm.connect()
        atexit.register(self.arm.disconnect)
        threading.Thread(target=self.camera_loop, daemon=True).start()
        self.say(f"UI: http://127.0.0.1:{http}/")
        self.app.run(host="0.0.0.0", port=http, threaded=True)
