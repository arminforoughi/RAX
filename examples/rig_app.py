"""One arm, shared by every UI in the process: a log, one job at a time, the live view,
teleop, stop, and the optional room camera.

Each UI (the general pick on :8484, tube sorting on :8486) is an :class:`App` with its
own Flask routes on top of the same :class:`Rig`, the way the tube UI used to live
inside the mission server. The arm is opened once; jobs from either UI queue on the
same lock, so two pages can never drive it at the same time.
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, request

from rax.robots import ARMS, make_arm
from rax.robots.urdf_visuals import link_visuals

HERE = os.path.dirname(os.path.abspath(__file__))

#: The optional overhead ("room") camera, a CamSurv MJPEG server.
CAMSURV_URL = os.environ.get("RAX_CAMSURV_URL", "http://127.0.0.1:5000")
CAMSURV_PASSWORD = os.environ.get("RAX_CAMSURV_PASSWORD", "")
CAMSURV_STREAM = os.environ.get("RAX_CAMSURV_STREAM", "0")

#: The SO-101's moving jaw is not in the FK chain: URDF joint `gripper`, on gripper_link.
_c, _s = math.cos(1.5708), math.sin(1.5708)
JAW_T = np.array([[1, 0, 0, 0.0202], [0, _c, -_s, 0.0188], [0, _s, _c, -0.0234],
                  [0, 0, 0, 1.0]])

#: Teleop speeds: m/s for reach and height, deg/s for base, tilt and roll.
JOG_SPEED, JOG_AZIM, JOG_WRIST, JOG_DT = 0.05, 14.0, 45.0, 0.05


def overhead_frame():
    """One frame from the overhead camera, or None (none there, or streaming black)."""
    try:
        s = requests.Session()
        s.post(CAMSURV_URL + "/", data={"password": CAMSURV_PASSWORD}, timeout=4)
        r = s.get(CAMSURV_URL + "/stream/" + CAMSURV_STREAM, stream=True, timeout=8)
        buf = b""
        for chunk in r.iter_content(4096):
            buf += chunk
            a, b = buf.find(b"\xff\xd8"), buf.find(b"\xff\xd9", 2)
            if a != -1 and b != -1:
                r.close()
                img = cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR)
                return None if img is None or float(img.mean()) < 5.0 else img
        r.close()
    except Exception:
        pass
    return None


def arm_from_args(description: str):
    """Parse the common flags and build (not connect) the arm."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--arm", default=os.environ.get("RAX_ARM", "so101"), choices=ARMS)
    ap.add_argument("--port", default=os.environ.get("RAX_ARM_PORT", ""),
                    help="the arm's serial port (COM4, /dev/ttyACM0, ...)")
    ap.add_argument("--camera", type=int, default=0, help="wrist camera index (x250)")
    ap.add_argument("--no-tubes", action="store_true", help="do not serve the tube UI")
    a, _ = ap.parse_known_args()
    handeye = os.path.join(HERE, f"handeye_{a.arm}.json")        # your own fit, if any
    if not os.path.exists(handeye):
        handeye = os.path.join(HERE, f"handeye_{a.arm}.example.json")
    kw = {"camera_index": a.camera} if a.arm == "x250" else {}
    return a, make_arm(a.arm, a.port, handeye_file=handeye, **kw)


class Rig:
    """The arm and everything the UIs share about it."""

    def __init__(self, arm):
        self.arm = arm
        self.lock = threading.Lock()
        self.phase_name, self.note = "IDLE", "ready"
        self.running = False
        self.t0 = 0.0
        self.log: list[dict] = []
        self.results: list[dict] = []
        self.jog_held: set[str] = set()
        self.jog_vec = {"r": 0.0, "th": 0.0, "z": 0.0, "t": 0.0}
        arm.log_fn, arm.phase_fn = self.say, self.phase

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

    def start_job(self, target):
        """Run ``target`` on the arm in the background, one job at a time, from any UI."""
        with self.lock:
            if self.running:
                return jsonify(ok=False, error="the arm is busy"), 409
            self.running, self.t0 = True, time.time()
        self.jog_held.clear()
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

    # ---- background loops -------------------------------------------------------
    def camera_loop(self):
        """Keep the frame fresh while nothing else is reading the arm."""
        while True:
            try:
                if not self.running and not self._jogging():
                    self.arm.observe(check_stop=False)
            except Exception:
                pass
            time.sleep(0.1)

    def _jogging(self):
        return bool(self.jog_held) or time.time() - self.jog_vec["t"] < 0.4

    def jog_loop(self):
        """Polar teleop: reach, base, height, tilt, roll, holding the hand's angle."""
        from rax.pick.arm import pitch_of
        arm = self.arm
        while True:
            time.sleep(JOG_DT)
            if self.running or not self._jogging():
                continue
            try:
                h, v = set(self.jog_held), dict(self.jog_vec)
                live = time.time() - v["t"] < 0.4
                dr = JOG_SPEED * (("fwd" in h) - ("back" in h) + (v["r"] if live else 0))
                dth = JOG_AZIM * (("left" in h) - ("right" in h) + (v["th"] if live else 0))
                dz = JOG_SPEED * (("up" in h) - ("down" in h) + (v["z"] if live else 0))
                dp = JOG_WRIST * (("pitch_dn" in h) - ("pitch_up" in h))
                dro = JOG_WRIST * (("roll_cw" in h) - ("roll_ccw" in h))
                q = arm.observe(check_stop=False)
                tip = arm.tip(q)
                r = math.hypot(tip[0], tip[1]) + dr * JOG_DT
                th = math.atan2(tip[1], tip[0]) + math.radians(dth * JOG_DT)
                z = max(0.0, float(tip[2]) + dz * JOG_DT)
                pitch = pitch_of(arm, q) + dp * JOG_DT
                roll = float(q[arm.roll]) + dro * JOG_DT
                q_new, err = arm.ik(q, np.array([r * math.cos(th), r * math.sin(th), z]),
                                    pitch, roll)
                if err < 0.01:
                    step = np.clip(np.asarray(q_new) - q, -3.0, 3.0)
                    arm.send(np.clip(q + step, arm.lo, arm.hi))
            except Exception as e:
                self.say(f"jog: {type(e).__name__}: {e}")
                self.jog_held.clear()
                time.sleep(0.5)


class App:
    """One UI on the shared rig: its own Flask app, the rig's arm, log and job lock."""

    name = "app"

    def __init__(self, rig: Rig):
        self.rig, self.arm = rig, rig.arm
        self.app = Flask(self.name, static_folder=None)
        self._common_routes()

    # the rig's log and job runner, reachable as self.say(...) etc.
    def say(self, msg):
        self.rig.say(msg)

    def phase(self, name, note=""):
        self.rig.phase(name, note)

    def record(self, *a, **kw):
        self.rig.record(*a, **kw)

    def start_job(self, target):
        return self.rig.start_job(target)

    # ---- the wrist view: each UI draws its own overlay ---------------------------
    def draw(self, img, q):
        """Draw this UI's overlay on the wrist frame (BGR, in place)."""

    def draw_grid(self, img):
        """The 6x5 grid, the grip cells (yellow: between the two fingertips) and the
        grip centre. The pick is done when the object sits in the yellow cells."""
        h, w = img.shape[:2]
        for c in range(1, 6):
            cv2.line(img, (c * w // 6, 0), (c * w // 6, h), (80, 80, 80), 1)
        for r in range(1, 5):
            cv2.line(img, (0, r * h // 5), (w, r * h // 5), (80, 80, 80), 1)
        (fu, fv), (ju, jv) = self.arm.tip_uv, self.arm.jaw_uv
        mu, mv = 2 * ju - fu, 2 * jv - fv            # the other fingertip
        cells = {(min(5, max(0, int((fu + (mu - fu) * t) * 6 / w))),
                  min(4, max(0, int((fv + (mv - fv) * t) * 5 / h))))
                 for t in np.linspace(0, 1, 21)}
        for c, r in cells:
            cv2.rectangle(img, (c * w // 6, r * h // 5), ((c + 1) * w // 6, (r + 1) * h // 5),
                          (0, 220, 220), 1)
        ju, jv = (int(v) for v in self.arm.jaw_uv)
        cv2.drawMarker(img, (ju, jv), (255, 120, 255), cv2.MARKER_TILTED_CROSS, 16, 2)

    def render(self):
        rgb = self.arm.rgb
        if rgb is None:
            return None
        img = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        try:
            self.draw(img, self.arm.q)
        except Exception:
            pass
        cv2.putText(img, self.rig.phase_name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (90, 255, 90), 2, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpg.tobytes() if ok else None

    # ---- routes every UI has ------------------------------------------------------
    def _common_routes(self):
        app, arm, rig = self.app, self.arm, self.rig
        urdf = [None]

        @app.route("/stream")
        def r_stream():
            def gen():
                last = None
                while True:
                    if arm.rgb is not None and arm.rgb is not last:
                        last = arm.rgb
                        jpg = self.render()
                        if jpg:
                            yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    time.sleep(0.08)
            return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

        @app.route("/stream2")
        def r_stream2():
            """The overhead camera, if there is one."""
            def gen():
                while True:
                    img = overhead_frame()
                    if img is None:
                        return
                    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

        @app.route("/camstatus")
        def r_camstatus():
            return jsonify(room=overhead_frame() is not None)

        @app.route("/urdf")
        def r_urdf():
            if urdf[0] is None:
                urdf[0] = [{"name": n, "v": [round(float(x), 4) for x in V.ravel()],
                            "f": [int(i) for i in F.ravel()]}
                           for n, (V, F) in link_visuals(arm.p.urdf_path,
                                                         mesh_dir=arm.p.mesh_path).items()]
            return jsonify(links=urdf[0], arm=arm.p.name)

        @app.route("/stop", methods=["POST"])
        def r_stop():
            arm.stop_flag.set()
            rig.jog_held.clear()
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

        @app.route("/relax", methods=["POST"])
        def r_relax():
            return self.start_job(arm.relax)

        @app.route("/jogpress", methods=["POST"])
        def r_jogpress():
            d = request.args.get("dir", "")
            if rig.running:
                return jsonify(ok=False, reason="the arm is busy")
            if d in ("open", "close"):
                g = arm.p.gripper
                pct = g.open_pct if d == "open" else g.closed_pct
                threading.Thread(target=lambda: arm.grip(pct), daemon=True).start()
                return jsonify(ok=True, grip=f"gripper {d}")
            rig.jog_held.add(d)
            return jsonify(ok=True)

        @app.route("/jogrelease", methods=["POST"])
        def r_jogrelease():
            d = request.args.get("dir", "")
            if d == "all":
                rig.jog_held.clear()
            else:
                rig.jog_held.discard(d)
            return jsonify(ok=True)

        @app.route("/jogvec", methods=["POST"])
        def r_jogvec():
            rig.jog_vec = {k: float(request.args.get(k, 0.0)) for k in ("r", "th", "z")}
            rig.jog_vec["t"] = time.time()
            return jsonify(ok=True)

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

    def geom(self, extra=None):
        """Every URDF link's 4x4 pose, for the 3D view."""
        arm = self.arm
        q = arm.q.copy()
        q[arm.roll] += arm.p.gripper.render_offset_deg        # display only
        chain = list(arm.kin.get_link_transforms_chain(q))
        xf = {n: [round(float(v), 5) for v in np.asarray(T, float).ravel()] for n, T in chain}
        tg = dict(chain).get("gripper_link")
        if tg is not None and arm.p.name == "so101":
            xf["moving_jaw_so101_v1_link"] = [round(float(v), 5)
                                              for v in (np.asarray(tg) @ JAW_T).ravel()]
        links = [[round(float(T[i, 3]), 4) for i in range(3)] for _n, T in chain]
        tip = [round(float(v), 4) for v in arm.tip(arm.q)]
        out = dict(xf=xf, links=links, ee=tip, tip=tip, arm=arm.p.name,
                   joints=[round(float(v), 2) for v in arm.q], joint_names=arm.motors)
        out.update(extra or {})
        return out


def serve(rig: Rig, apps: list[tuple[App, int]]):
    """Connect the arm once, start the shared loops, and serve every UI."""
    import atexit
    arm = rig.arm
    rig.say(f"connecting the {arm.p.name} on {arm.port} ...")
    arm.connect()
    atexit.register(arm.disconnect)
    threading.Thread(target=rig.camera_loop, daemon=True).start()
    threading.Thread(target=rig.jog_loop, daemon=True).start()
    for app, port in apps[1:]:
        threading.Thread(target=lambda a=app, p=port: a.app.run(
            host="0.0.0.0", port=p, threaded=True, use_reloader=False), daemon=True).start()
    for app, port in apps:
        rig.say(f"{app.name} UI: http://127.0.0.1:{port}/")
    first, port = apps[0]
    first.app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
