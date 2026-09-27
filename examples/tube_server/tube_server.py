"""Tube pick-and-place, one server, any arm, with a 3D view of the arm and the tubes.

WHAT THIS REPLACES. The X250's tube pick lived in `examples/vla/pick/pick.py`: 878 lines
of hand-written controller with its own retry loop, its own grasp check, its own episode
logging, its own OpenCV window, and no way to see what the arm thought the world looked
like. The SO-101's lived in `examples/mission_server/mission_server.py`: 9,682 lines with
a second copy of every one of those. This is the third implementation of neither -- it is
the two of them with the shared parts taken out and used:

    rax.manipulation.approach.visual_servo   the approach        (already shared)
    rax.manipulation.approach.jacobian       what moves what     (already shared)
    rax.manipulation.grip                    did we get it       (extracted here)
    rax.manipulation.attempt                 try again           (extracted here)
    rax.manipulation.episodes                what happened       (extracted here)
    rax.perception.rack_holes                where to put it     (extracted here)
    rax.robots.profiles.x250 / so101         what the arm IS     (added here)

So the arm-specific surface left in this file is a `TubeRig` (see rig.py) and nothing
else. Adding a third arm means describing it, not reimplementing a pick.

THE 3D VIEW IS DRIVEN ENTIRELY BY THE ARM'S DESCRIPTION. /urdf tessellates whatever the
profile's URDF says -- STL meshes for the SO-101, boxes and cylinders for the X250 -- and
/geom streams a 4x4 per link from the same forward kinematics the rest of the stack uses.
Neither route mentions either arm. That was already true of the mission server's viewer;
what was missing was a second arm to prove it.

READ THE X250 URDF'S HEADER BEFORE JUDGING A DISTANCE OFF THIS VIEW. Its joint structure
and travel are measured; its link lengths are nominal. The arm moves correctly and is
drawn approximately.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
import time

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

from rax.manipulation.attempt import with_retries
from rax.manipulation.episodes import EpisodeLog
from rax.manipulation.grip import reconcile, settled
from rax.robots.urdf_visuals import link_visuals

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(HERE, "ui")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("tube")

app = Flask(__name__, static_folder=None)

# ---------------------------------------------------------------------------------
# state. One lock, and it guards only the dict -- never held across a robot move.
# ---------------------------------------------------------------------------------
lock = threading.Lock()
state = {
    "phase": "IDLE",
    "note": "",
    "running": False,
    "stop": False,
    "target": None,
    "dest": None,
    "log": [],
}
RIG = [None]
KIN = [None]
EPISODES = [None]


class Stopped(RuntimeError):
    """Raised out of checkpoint() when the Stop button has been pressed."""


def say(msg: str) -> None:
    logger.info(msg)
    with lock:
        state["log"].append({"t": time.strftime("%H:%M:%S"), "m": str(msg)[:220]})
        del state["log"][:-200]


def phase(name: str, note: str = "") -> None:
    with lock:
        state["phase"], state["note"] = name, note
    say(f"[{name}] {note}" if note else f"[{name}]")


def checkpoint() -> None:
    with lock:
        if state["stop"]:
            raise Stopped("stopped by the operator")


def kin():
    if KIN[0] is None:
        from rax.manipulation.arms.urdf_kinematics import UrdfKinematics
        p = RIG[0].profile
        KIN[0] = UrdfKinematics(p.urdf_path, ee_frame=p.ee_frame,
                                joint_names=list(p.joint_names))
    return KIN[0]


# ---------------------------------------------------------------------------------
# the pick, expressed once, over the shared pieces
# ---------------------------------------------------------------------------------
def grip_decision(camera=None):
    """Read the gripper, settle it, and reconcile with a camera verdict if there is one.

    `settled` is not optional: mid-close the jaws pass THROUGH the held band on their way
    shut, so an early read reports a grasp that has not happened yet.
    """
    rig = RIG[0]
    value = settled(rig.gripper, tol=0.05, timeout=1.5, dt=0.05)
    return reconcile(rig.grip_sensor().verdict(value), camera)


def sim_pick(tube_id: int, dest_rack: str | None) -> str:
    """The sim's pick: aim, close, carry, release — through the real kinematics.

    Deliberately able to MISS (see SimRig.grasp): a UI developed against a rig that
    always succeeds never shows its own failure path, and the failure path is the one
    an operator needs to read.
    """
    rig = RIG[0]
    t = rig.tube(tube_id)
    if t is None:
        raise RuntimeError(f"no tube {tube_id}")

    phase("AIM", f"lining up on the {t.colour} tube at r={math.hypot(t.x, t.y):.2f}m")
    _sim_move_tip((t.x, t.y, 0.10))
    checkpoint()

    phase("APPROACH", "descending onto the cap")
    _sim_move_tip((t.x, t.y, 0.015))
    checkpoint()

    phase("GRASP", "closing")
    ok = rig.grasp(tube_id)
    if not ok:
        return "closed on nothing"

    phase("LIFT", "checking the grip as it comes up")
    _sim_move_tip((t.x, t.y, 0.14))
    if dest_rack:
        free = rig.free_holes(dest_rack)
        if not free:
            return f"held, but the {dest_rack} rack is full"
        hole = free[0]
        rack = next(r for r in rig.racks() if r.name == dest_rack)
        hx, hy = rack.hole_xy(hole)
        phase("CARRY", f"to the {dest_rack} rack, hole {hole}")
        _sim_move_tip((hx, hy, 0.16))
        _sim_move_tip((hx, hy, 0.06))
        phase("RELEASE", f"into hole {hole}")
        rig.release_into(dest_rack, hole)
        _sim_move_tip((hx, hy, 0.16))
        return f"placed in {dest_rack} hole {hole}"
    return "held"


def _sim_move_tip(p, steps: int = 14):
    """Move the sim's tip to a base-frame point by IK, in visible increments.

    Increments so the 3D view shows a MOTION rather than a teleport -- the whole reason
    the view exists is to watch an approach happen.
    """
    rig = RIG[0]
    k = kin()
    q = rig.joints_deg()
    start = np.asarray(k.forward_kinematics(q))[:3, 3]
    target = np.asarray(p, np.float64)
    for i in range(1, steps + 1):
        checkpoint()
        want = start + (target - start) * (i / steps)
        q_new, err = _ik(k, want, q)
        if err > 0.03:
            # Do not force it. An unreachable waypoint is information — the sim's whole
            # value is that it fails where the description says the arm cannot go.
            say(f"        (sim) cannot reach {np.round(want, 3).tolist()} — "
                f"off by {err * 100:.1f}cm, stopping the move there")
            break
        rig.set_joints(q_new)
        q = q_new
        rig.tip_xyz()          # drags a held tube along
        time.sleep(0.035)


def _ik(k, p_tgt, q_seed):
    """POSITION-ONLY IK, RE-SEEDED on failure. Returns (q, error_m).

    Orientation is deliberately unweighted. Where the gripper points during a tube
    approach is decided by the demonstrated wrist and tool angles, not by the solver --
    the same reason the servo does not drive those two axes. Asking the IK to hit a full
    pose would spend the arm's freedom on an orientation nobody specified and fail on
    poses that are perfectly reachable.

    RE-SEEDING IS NOT BELT AND BRACES, and the sim found the case that proves it. Walking
    a straight Cartesian line from the look pose to a tube at (0.24, +0.07), the solver
    tracked twelve waypoints at 0.00 cm, then missed the thirteenth by 0.07 cm and the
    last by 2.06 cm -- while that final point solves EXACTLY when seeded from the look
    pose directly. Nothing was out of reach; the chain had walked into a branch it could
    not finish from, which is the elbow-flip dead band the SO-101 profile already carries
    `ik_seeds` for. The X250 has none characterised, so the fallback seeds are its own
    named poses: they are known-good configurations, which is the property a seed needs.

    Reported failures stay failures. If no seed solves it, the caller gets the error and
    stops -- a sim that quietly teleports past an unreachable waypoint would be lying
    about exactly the thing it exists to show.
    """
    T = np.eye(4)
    T[:3, 3] = np.asarray(p_tgt, np.float64)
    prof = RIG[0].profile

    def attempt(seed):
        q = np.asarray(k.inverse_kinematics(np.asarray(seed, np.float64), T,
                                            orientation_weight=0.0), np.float64)
        reached = np.asarray(k.forward_kinematics(q))[:3, 3]
        return q, float(np.linalg.norm(reached - np.asarray(p_tgt, np.float64)))

    best_q, best_e = attempt(q_seed)
    if best_e <= 0.003:
        return best_q, best_e

    seeds = [prof.home_deg, prof.view_deg]
    for s in prof.ik_seeds:
        # A profile seed may leave joints as None, meaning "keep the caller's value".
        seeds.append([float(v) if v is not None else float(q_seed[i])
                      for i, v in enumerate(s)])
    for s in seeds:
        if not s:
            continue
        q, e = attempt(np.asarray(s, np.float64))
        if e < best_e:
            best_q, best_e = q, e
        if best_e <= 0.003:
            break
    return best_q, best_e


def run_pick(tube_id: int, dest_rack: str | None, tries: int = 3):
    """The whole job, over the SHARED retry loop and the SHARED episode log."""
    rig = RIG[0]
    ep = EPISODES[0].start("tube_pick", f"tube {tube_id}",
                           arm=rig.profile.name, simulated=rig.simulated,
                           dest=dest_rack)

    def action(n):
        if rig.simulated:
            return sim_pick(tube_id, dest_rack)
        return hardware_pick(tube_id, dest_rack)

    def verify():
        """Check WHAT WAS ASKED FOR, which is not always "are the jaws full".

        This is the bug the first end-to-end place walked into, and it is worth spelling
        out because it is the same mistake as reading "PICK SUCCESS" off a log. If the job
        was pick-and-place, then by the time it has finished the jaws are SUPPOSED to be
        empty — the tube is in the rack. Verifying the gripper there asks the wrong
        question, and on the X250's position sensor it gets a confidently wrong answer,
        because the place-open position reads far above the holding threshold. The run
        reported DONE, and the reason it gave was "gripper settled at 50.00, holding".

        So: a place is verified by the tube being in the rack, and a pick-and-hold is
        verified by the grip. Each job is checked against its own goal.
        """
        if dest_rack:
            t = next((x for x in rig.tubes() if x.id == tube_id), None)
            if t is None:
                return False, f"tube {tube_id} is no longer in the map"
            if t.rack == dest_rack:
                return True, (f"tube {tube_id} is in the {dest_rack} rack, hole {t.hole}"
                              f" ({t.source})")
            where = "still in the jaws" if t.held else (
                f"in the {t.rack} rack" if t.rack else "loose on the bench")
            return False, f"tube {tube_id} did not reach the {dest_rack} rack — {where}"
        d = grip_decision()
        return d.held, d.detail

    def between(n):
        phase("RETRY", f"attempt {n}: back to the look pose and re-acquire")
        if rig.simulated:
            rig.set_joints(rig.profile.home_deg)
            rig.set_gripper(rig.profile.gripper.open_pct)

    try:
        out = with_retries(action, verify, tries=tries, between=between,
                           settle_s=2.0, poll_s=0.25,
                           record=lambda n, ok, d: EPISODES[0].attempt(ep, n, ok, d),
                           checkpoint=checkpoint)
        EPISODES[0].end(ep, out.ok, out.detail)
        phase("DONE" if out.ok else "FAILED", out.detail)
        return out
    except Stopped as e:
        EPISODES[0].end(ep, False, str(e))
        phase("STOPPED", str(e))
        raise
    finally:
        with lock:
            state["running"] = False


def hardware_pick(tube_id: int, dest_rack: str | None) -> str:
    """The real pick: the SHARED visual servo, through the rig's ServoArm.

    NOT YET EXERCISED ON HARDWARE. The X250 is on COM5, which does not enumerate right
    now, and the SO-101's camera is owned by the mission server's process. So this path
    is written against the same seam the tests cover and is honestly labelled rather
    than claimed to work: the moment either arm is on the bus it is one run from being
    either right or debuggable, and that is a better place to be than a second
    hand-written controller.
    """
    from rax.manipulation.approach.jacobian import measure_jacobian
    from rax.manipulation.approach.visual_servo import ServoConfig, begin, step, ARRIVED

    rig = RIG[0]
    if not hasattr(rig, "servo_arm"):
        raise RuntimeError(f"{rig.profile.name} has no servo adapter wired up yet")
    tubes = rig.tubes()
    t = next((x for x in tubes if x.id == tube_id), None)
    colour = t.colour if t else "green"

    phase("MEASURE", f"probing what each joint does to the picture ({colour} cap)")
    arm = rig.servo_arm(colour)
    jac = measure_jacobian(arm, log=say)
    say(f"jacobian: {jac.describe()}")

    phase("APPROACH", "closing on the cap")
    st, cfg = begin(jac), ServoConfig()
    for _ in range(60):
        checkpoint()
        st, cmd = step(st, arm.sense(), cfg)
        if cmd.kind == ARRIVED:
            break
        if getattr(cmd, "dq", None) is not None:
            arm.apply(arm.actuators() + cmd.dq)

    phase("GRASP", "closing")
    # The close itself is the arm's own: the gripper is the one thing that is genuinely
    # per-arm here (different sensor, different units, different safe stall behaviour).
    raise NotImplementedError(
        "the hardware close is not wired up for this rig yet — connect the arm and "
        "implement TubeRig.close() for it")


# ---------------------------------------------------------------------------------
# routes: description and geometry. NEITHER MENTIONS AN ARM.
# ---------------------------------------------------------------------------------
_urdf_cache = [None]


@app.route("/urdf")
def r_urdf():
    """Link visual geometry in link-local coords, tessellated from the profile's URDF.

    Works for an STL-mesh URDF and a primitive one alike — which is the whole reason
    `rax.robots.urdf_visuals` exists rather than lerobot's mesh-only loader.
    """
    if _urdf_cache[0] is None:
        p = RIG[0].profile
        try:
            vis = link_visuals(p.urdf_path, mesh_dir=p.mesh_path)
            _urdf_cache[0] = [
                {"name": n,
                 "v": [round(float(x), 5) for x in V.ravel()],
                 "f": [int(i) for i in F.ravel()]}
                for n, (V, F) in vis.items()]
            say(f"3D: {sum(len(l['f']) // 3 for l in _urdf_cache[0])} triangles across "
                f"{len(_urdf_cache[0])} links of {p.name}")
        except Exception as e:
            say(f"3D: could not load {p.name}'s visuals: {type(e).__name__}: {e}")
            _urdf_cache[0] = []
    return jsonify(links=_urdf_cache[0], arm=RIG[0].profile.name)


@app.route("/geom")
def r_geom():
    """Live geometry: a 4x4 per link, the tip, the tubes and the racks."""
    rig = RIG[0]
    xf, tip = {}, None
    try:
        q = rig.joints_deg()
        k = kin()
        for name, T in k.get_link_transforms_chain(q):
            xf[name] = [round(float(v), 5) for v in np.asarray(T, np.float64).ravel()]
        # The fingers are off the FK chain (one actuator drives both), so the viewer is
        # handed the hand's pose and the current opening and composes them itself —
        # exactly what the SO-101 viewer already does for its moving jaw.
        Tg = xf.get("gripper_link")
        tip = [round(float(v), 4)
               for v in np.asarray(k.forward_kinematics(q))[:3, 3]]
    except Exception as e:
        say(f"3D: geometry unavailable ({type(e).__name__}: {e})")

    g = rig.profile.gripper
    span = max(g.open_pct - g.closed_pct, 1e-6)
    opening = float(np.clip((rig.gripper() - g.closed_pct) / span, 0.0, 1.0))

    with lock:
        ph, note = state["phase"], state["note"]
    return jsonify(
        arm=rig.profile.name, simulated=bool(rig.simulated),
        joints=[round(float(v), 2) for v in rig.joints_deg()],
        joint_names=list(rig.profile.joint_names),
        xf=xf, tip=tip, opening=opening,
        tubes=[t.as_json() for t in rig.tubes()],
        racks=[r.as_json() for r in rig.racks()],
        phase=ph, note=note)


@app.route("/state")
def r_state():
    rig = RIG[0]
    with lock:
        s = {k: state[k] for k in ("phase", "note", "running", "target", "dest")}
        s["log"] = list(state["log"])[-60:]
    d = grip_decision()
    s.update(arm=rig.profile.name, simulated=bool(rig.simulated),
             gripper=round(rig.gripper(), 2), held=d.held, grip_detail=d.detail,
             episodes=EPISODES[0].tally("tube_pick"))
    return jsonify(s)


@app.route("/stream")
def r_stream():
    """Wrist MJPEG. 404s cleanly when the rig has no camera, so the UI can hide it."""
    import cv2
    if RIG[0].frame() is None:
        return ("no camera on this rig", 404)

    def gen():
        while True:
            f = RIG[0].frame()
            if f is not None:
                ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/pick", methods=["POST"])
def r_pick():
    d = request.get_json(silent=True) or request.form or {}
    tube_id = int(d.get("tube", 1))
    dest = d.get("rack") or None
    with lock:
        if state["running"]:
            return jsonify(ok=False, error="already running"), 409
        state.update(running=True, stop=False, target=tube_id, dest=dest)

    threading.Thread(target=lambda: _guard(run_pick, tube_id, dest),
                     daemon=True).start()
    return jsonify(ok=True, tube=tube_id, rack=dest)


def _guard(fn, *a):
    try:
        fn(*a)
    except Stopped:
        pass
    except Exception as e:
        say(f"!! {type(e).__name__}: {e}")
        phase("FAILED", f"{type(e).__name__}: {e}")
    finally:
        with lock:
            state["running"] = False


@app.route("/stop", methods=["POST"])
def r_stop():
    with lock:
        state["stop"] = True
    say("STOP requested")
    return jsonify(ok=True)


@app.route("/reset", methods=["POST"])
def r_reset():
    rig = RIG[0]
    with lock:
        state.update(stop=False, running=False)
    if rig.simulated:
        rig.set_joints(rig.profile.home_deg)
        rig.set_gripper(rig.profile.gripper.open_pct)
    phase("IDLE", "back at the look pose")
    return jsonify(ok=True)


@app.route("/episodes")
def r_episodes():
    return jsonify(tally=EPISODES[0].tally("tube_pick"),
                   recent=EPISODES[0].records("tube_pick")[-25:])


@app.route("/")
def index():
    return send_from_directory(UI, "tube.html")


@app.route("/ui/<path:name>")
def ui_asset(name):
    if "/" in name or "\\" in name or name.startswith("."):
        return ("no", 404)
    return send_from_directory(UI, name)


# ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default="sim", choices=("sim", "x250", "so101"),
                    help="which rig to drive. 'sim' needs no hardware.")
    ap.add_argument("--profile", default="x250",
                    help="which arm the simulator should BE (sim only)")
    ap.add_argument("--port", type=int, default=8486)
    ap.add_argument("--serial", default=os.environ.get("RAX_X250_PORT", "COM5"))
    ap.add_argument("--wrist-cam", type=int, default=0)
    ap.add_argument("--episodes", default=os.path.join(HERE, "episodes.jsonl"))
    a = ap.parse_args()

    from rig import SimRig, X250Rig       # noqa: E402  (same directory)

    if a.arm == "sim":
        RIG[0] = SimRig(a.profile)
        say(f"simulating the {a.profile} — NO HARDWARE. Kinematics are real; the arm's "
            f"responses and the tube positions are not.")
    elif a.arm == "x250":
        import sys
        sys.path.insert(0, os.path.join(HERE, "..", "vla", "pick"))
        from x250_driver import X250Follower, X250FollowerConfig
        try:
            from lerobot.cameras.opencv import OpenCVCameraConfig
            cams = {"wrist": OpenCVCameraConfig(index_or_path=a.wrist_cam,
                                                width=640, height=480, fps=30)}
        except Exception:
            cams = {}
            say("no lerobot camera config available — running without the wrist view")
        bot = X250Follower(X250FollowerConfig(port=a.serial, cameras=cams))
        bot.connect()
        RIG[0] = X250Rig(bot)
        say(f"X250 connected on {a.serial}")
    else:
        raise SystemExit(
            "the SO-101's camera and bus are owned by the mission server's process, so "
            "--arm so101 has to be constructed inside it (see So101Rig's docstring). "
            "Run --arm sim --profile so101 to work on the UI against its description.")

    EPISODES[0] = EpisodeLog(
        a.episodes,
        probe=lambda: {"tip_cm": [round(float(v) * 100, 1)
                                  for v in np.asarray(
                                      kin().forward_kinematics(RIG[0].joints_deg()))[:3, 3]]},
        note=say)

    p = RIG[0].profile
    say(f"arm       {p.name}  ({p.n_joints} joints: {', '.join(p.joint_names)})")
    say(f"urdf      {p.urdf_path}")
    say(f"episodes  {a.episodes}")
    say(f"open      http://127.0.0.1:{a.port}/")
    phase("IDLE", "ready")
    app.run(host="0.0.0.0", port=a.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
