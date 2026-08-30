"""LiveKit + Gemini 3.1 gaze-controlled robot agent.

User's webcam/mic arrives via a LiveKit room → Gemini 3.1 Flash Audio sees
the user's view and hears their voice → the agent calls ``gaze_robot()`` →
the SO-101 arm's GazeEngine runs (SEARCH → APPROACH → GRASP) → the arm's
OAK-D left-camera stream is published back into the same LiveKit room so
the user can watch the robot's eye view in real-time.

Environment variables (required):
    LIVEKIT_URL         wss://your-project.livekit.cloud
    LIVEKIT_API_KEY     your LiveKit API key
    LIVEKIT_API_SECRET  your LiveKit API secret
    GOOGLE_API_KEY      Google / Gemini API key

Hardware env variables (optional, override built-in defaults):
    ROBOT_PORT          serial port for the SO-101 (default: auto-detect)
    ROBOT_URDF          path to the SO-101 URDF
    GRIPPER_CAM_TF      eye-in-hand extrinsic as 'x,y,z,rx,ry,rz'
    GAZE_APPROACH       approach style: angled | topdown | horizontal
    GAZE_DETECTOR       yolo | color_blob | blob | auto
    GAZE_STEREO         sgbm | raft | foundation | auto
    ROBOT_MOCK          1 = use mock arm (no hardware required)

Run::
    cd /path/to/RAX
    PYTHONPATH=. python agents/livekit_gaze_agent.py dev
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Ensure the RAX repo root is on the path when this file is run directly.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv

load_dotenv(".env.local")

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, RunContext, function_tool
from livekit.plugins import google

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

REALTIME_MODEL   = "gemini-2.5-flash-native-audio-preview-12-2025"  # plugin default
VISION_MODEL     = "gemini-2.5-flash"
ROBOT_PORT       = os.environ.get("ROBOT_PORT", "")
ROBOT_URDF       = os.environ.get("ROBOT_URDF", "SO101/so101_new_calib.urdf")
GRIPPER_CAM_TF   = os.environ.get("GRIPPER_CAM_TF", "0.04,0,0.09,-0.2690,0.2824,-1.6014")
GAZE_APPROACH    = os.environ.get("GAZE_APPROACH", "angled")
GAZE_DETECTOR    = os.environ.get("GAZE_DETECTOR", "auto")
GAZE_STEREO      = os.environ.get("GAZE_STEREO", "sgbm")
USE_MOCK         = os.environ.get("ROBOT_MOCK", "0").strip() in ("1", "true", "yes")
USE_RERUN        = os.environ.get("ROBOT_RERUN", "1").strip() in ("1", "true", "yes")

PERSONA = """You are a gaze-controlled robot arm assistant at the AI Engineer World's Fair Hackathon 2026.

You see through the user's webcam AND can look through the robot's own camera on demand.
When the user asks what the robot sees, what's on the table, or where an object is — call look().

Your tools:
- look: capture the robot's camera and answer a question about what it sees.
- gaze_robot: command the SO-101 arm to find and grasp a named object. The robot stays on standby between commands — no reload needed.
- drop_object: open the gripper to release whatever the robot is holding.
- stop_robot: immediately stop the arm.
- robot_status: ask what the arm is currently doing.

Workflow:
1. If the user asks what you see → call look("what do you see?").
2. If they describe or point at an object → confirm, then call gaze_robot("object name").
3. Narrate the robot state as it searches → approaches → grasps.
4. After grasping, you can call drop_object() when asked to release.
5. Celebrate success or diagnose failure.

Keep responses short and energetic — this is a live hackathon demo.
Never invent visual details you cannot actually see — use look() to check."""

# ---------------------------------------------------------------------------
# GazeRunner — manages GazeEngine in a background thread
# ---------------------------------------------------------------------------

LEROBOT_PYTHON  = "/home/labot/.venv/lerobot/bin/python"
LEROBOT_SERVER  = str(Path(__file__).resolve().parent / "lero_server.py")
LEROBOT_CONFIG  = "/mnt/c/Users/labot/Documents/lerobot/configs/gaze_engine_zmq.yaml"
LEROBOT_CWD     = "/mnt/c/Users/labot/Documents/lerobot"


class _PersistentGaze:
    """Persistent lerobot gaze server: loads YOLO+robot ONCE, accepts queries via stdin.

    lero_server.py starts at agent launch. The slow YOLO/robot init (~20 s) happens
    in the background so by the time the user says "pick up X" everything is warm.
    Subsequent queries send a single stdin line — no reload, ~1 s latency.
    """

    def __init__(self) -> None:
        self.state        = "IDLE"
        self.query: str | None = None
        self.final_state: str | None = None
        self.log: list[str] = []
        self.on_state_change: "callable | None" = None
        self._proc: "subprocess.Popen | None" = None
        self._ready       = False   # True once "READY" seen from server stdout
        self._busy        = False   # True while a query is running
        self._starting    = False   # True while a launch is in flight (prevents double-spawn)
        self._pending: str | None = None   # query queued to run as soon as READY
        self._start_time: float | None = None   # perf_counter when launch began
        self._lock        = threading.Lock()
        self._arm         = None    # kept None; camera stream falls back to blank

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch lero_server.py in the background; it pre-loads YOLO+robot."""
        with self._lock:
            if self._starting:
                return   # a launch is already in flight
            self._starting = True
        t = threading.Thread(target=self._launch_server, daemon=True)
        t.start()

    def _ensure_running(self) -> None:
        """(Re)start the daemon if it isn't alive — e.g. after a crash (OFFLINE)."""
        with self._lock:
            proc = self._proc
            starting = self._starting
        if starting:
            return
        if proc is not None and proc.poll() is None:
            return   # already running
        self._note("daemon not running — (re)starting")
        self.start()

    def shutdown(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def request_pick(self, query: str) -> str:
        """Pick now if ready; otherwise queue the query and make sure the daemon is coming up.

        Returns one of:
          "picking" — sent to a ready arm, task started
          "queued"  — arm still loading (or crashed); will auto-run on READY
          "busy"    — arm ready but mid-task; caller should say "stop" first
        """
        with self._lock:
            proc  = self._proc
            ready = self._ready
            busy  = self._busy

        # Daemon down / crashed → queue and (re)launch.
        if proc is None or proc.poll() is not None:
            with self._lock:
                self._pending = query
            self._ensure_running()
            return "queued"

        # Still loading → queue; _launch_server dispatches it when READY.
        if not ready:
            with self._lock:
                self._pending = query
            return "queued"

        if busy:
            return "busy"

        return "picking" if self.pick(query) else "busy"

    def pick(self, query: str) -> bool:
        """Send a query to the running server. Returns False if busy or not ready."""
        with self._lock:
            proc = self._proc
            busy = self._busy
            ready = self._ready

        if proc is None or proc.poll() is not None:
            self._note("server not running — pick() ignored")
            return False
        if busy:
            return False   # currently executing a task

        self.query       = query
        self.final_state = None
        with self._lock:
            self._busy = True

        try:
            proc.stdin.write(query + "\n")
            proc.stdin.flush()
        except Exception as exc:
            self._note(f"stdin write failed: {exc}")
            with self._lock:
                self._busy = False
            return False

        self.state = "SEARCH"
        if self.on_state_change:
            self.on_state_change("SEARCH")
        return True

    def send_raw(self, cmd: str) -> bool:
        """Send a raw command line to the server stdin. Returns False if server is down or busy."""
        with self._lock:
            proc = self._proc
            busy = self._busy
        if proc is None or proc.poll() is not None:
            return False
        if busy:
            return False
        with self._lock:
            self._busy = True
        try:
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()
            return True
        except Exception:
            with self._lock:
                self._busy = False
            return False

    def send_jog(self, cmd: str) -> bool:
        """Fire-and-forget stdin line for manual jog / gripper.

        Unlike send_raw it does NOT set the busy latch — jog produces no
        protocol reply, so a latch would stick and block every later jog. Still
        refuses while an autonomous task owns the arm (busy True)."""
        with self._lock:
            proc = self._proc
            busy = self._busy
        if proc is None or proc.poll() is not None or busy:
            return False
        try:
            with self._lock:
                proc.stdin.write(cmd + "\n")
                proc.stdin.flush()
            return True
        except Exception:
            return False

    def stop_task(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is not None:
            try:
                proc.stdin.write("STOP\n")
                proc.stdin.flush()
            except Exception:
                pass
        self.state = "STOPPED"
        with self._lock:
            self._busy = False
        if self.on_state_change:
            self.on_state_change("STOPPED")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _note(self, msg: str) -> None:
        logger.info("[LeroGaze] %s", msg)
        self.log.append(msg)

    def _launch_server(self) -> None:
        """Start lero_server.py and stream its stdout, updating state from protocol lines."""
        import subprocess as _sp
        import os as _os

        env = dict(_os.environ)
        env["LERO_CONFIG"] = LEROBOT_CONFIG

        cmd = [LEROBOT_PYTHON, LEROBOT_SERVER]
        self._note(f"starting lero_server: {' '.join(cmd)}")
        self._start_time = time.perf_counter()
        self.state = "LOADING"
        if self.on_state_change:
            self.on_state_change("LOADING")

        try:
            proc = _sp.Popen(
                cmd,
                stdin=_sp.PIPE,
                stdout=_sp.PIPE,
                stderr=_sp.STDOUT,
                text=True,
                cwd=LEROBOT_CWD,
                env=env,
            )
            with self._lock:
                self._proc     = proc
                self._starting = False

            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                self._note(line)
                print(f"[lero-server] {line}", flush=True)

                upper = line.upper()

                if line == "READY":
                    with self._lock:
                        self._ready   = True
                        pending       = self._pending
                        self._pending = None
                    elapsed = (
                        time.perf_counter() - self._start_time
                        if self._start_time else 0.0
                    )
                    self._note(f"READY in {elapsed:.1f}s")
                    self.state = "READY"
                    if self.on_state_change:
                        self.on_state_change("READY")
                    if pending:
                        self._note(f"auto-dispatching queued pick: {pending!r}")
                        self.pick(pending)

                elif upper.startswith("STARTING "):
                    pass  # query echo

                elif upper.startswith("STATE "):
                    kw = line[6:].strip().upper()
                    self.state = kw
                    if self.on_state_change:
                        self.on_state_change(kw)

                elif upper in ("DONE", "DROPPED", "HOME"):
                    self.state = self.final_state = upper
                    with self._lock:
                        self._busy = False
                    if self.on_state_change:
                        self.on_state_change(upper)

                elif upper.startswith("FAILED"):
                    self.state = self.final_state = "FAILED"
                    with self._lock:
                        self._busy = False
                    if self.on_state_change:
                        self.on_state_change("FAILED")

                elif upper == "STOPPED":
                    self.state = "STOPPED"
                    with self._lock:
                        self._busy = False
                    if self.on_state_change:
                        self.on_state_change("STOPPED")

                else:
                    # Parse lerobot log lines for state keywords
                    for kw in ("SEARCH", "APPROACH", "GRASP", "PLACE", "PREPOSITION"):
                        if kw in upper:
                            self.state = kw
                            if self.on_state_change:
                                self.on_state_change(kw)
                            break

            proc.wait()
            rc = proc.returncode
            self._note(f"lero_server exited rc={rc}")
            with self._lock:
                self._ready    = False
                self._busy     = False
                self._starting = False
            self.state = "OFFLINE"
            if self.on_state_change:
                self.on_state_change("OFFLINE")

        except Exception as exc:
            self._note(f"server error: {exc}")
            with self._lock:
                self._ready    = False
                self._starting = False
            self.state = "OFFLINE"
            if self.on_state_change:
                self.on_state_change("OFFLINE")


# ---------------------------------------------------------------------------
# Robot camera → LiveKit video track
# ---------------------------------------------------------------------------

async def _stream_robot_camera(
    room: rtc.Room,
    gaze: "_PersistentGaze",
    fps: int = 15,
) -> None:
    """Publish the robot's camera as a LiveKit video track. Reads from the persistent arm."""
    import numpy as np
    import cv2

    W, H   = 640, 400
    source = rtc.VideoSource(width=W, height=H)
    track  = rtc.LocalVideoTrack.create_video_track("robot-eye", source)
    opts   = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await room.local_participant.publish_track(track, opts)
    logger.info("[robot-cam] published robot-eye video track")

    blank    = np.zeros((H, W, 4), dtype=np.uint8)
    interval = 1.0 / fps

    while True:
        try:
            frame_bgr = await asyncio.to_thread(_grab_zmq_frame)
            if frame_bgr is not None:
                rgba = cv2.cvtColor(cv2.resize(frame_bgr, (W, H)), cv2.COLOR_BGR2RGBA)
            else:
                rgba = blank

            source.capture_frame(rtc.VideoFrame(
                width=W, height=H,
                type=rtc.VideoBufferType.RGBA,
                data=rgba.tobytes(),
            ))
        except Exception as exc:
            logger.debug("[robot-cam] %s", exc)

        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# GazeRobotAgent
# ---------------------------------------------------------------------------

class GazeRobotAgent(Agent):
    """Gemini-powered agent that controls the SO-101 via voice + vision."""

    def __init__(self, room: rtc.Room) -> None:
        super().__init__(instructions=PERSONA)
        self._room  = room
        self._gaze  = _PersistentGaze()   # initialized once, never torn down per-query

    # ---- lifecycle ------------------------------------------------------

    async def on_enter(self) -> None:
        loop = asyncio.get_running_loop()

        def on_state(state: str):
            loop.call_soon_threadsafe(
                lambda s=state: asyncio.ensure_future(
                    _broadcast(self._room, {"type": "state", "state": s})
                )
            )

        self._gaze.on_state_change = on_state
        self._gaze.start()   # launches background thread; arm init happens there
        logger.info("[GazeRobotAgent] persistent gaze engine started")

    def start_camera_stream(self) -> None:
        asyncio.create_task(_stream_robot_camera(self._room, self._gaze))

    # ---- function tools -------------------------------------------------

    @function_tool()
    async def gaze_robot(self, context: RunContext, query: str) -> str:
        """Command the SO-101 arm to find and grasp the named object.

        The gaze engine is always running in the background — this just
        gives it a new target. No reload delay.

        Args:
            query: Natural-language object description, e.g. "red cube" or "blue block".
        """
        result = self._gaze.request_pick(query)
        if result == "busy":
            return f"Arm is busy (state={self._gaze.state}). Say stop first."
        if result == "queued":
            return f"Arm is still warming up (state={self._gaze.state}). I've queued {query!r} — it'll start automatically as soon as the arm is ready."
        return f"Searching for {query!r}. Call robot_status() to track progress."

    @function_tool()
    async def drop_object(self, context: RunContext) -> str:
        """Open the gripper to release whatever the robot is holding. Robot stays on standby."""
        if not self._gaze._ready:
            return "Arm not ready."
        ok = self._gaze.send_raw("DROP")
        return "Dropping object — gripper opening." if ok else f"Busy (state={self._gaze.state})."

    @function_tool()
    async def stop_robot(self, context: RunContext) -> str:
        """Stop the current gaze task. Arm stays connected and ready for the next command."""
        self._gaze.stop_task()
        return f"Stop sent (state was {self._gaze.state})."

    @function_tool()
    async def robot_status(self, context: RunContext) -> str:
        """Return the current gaze engine state and recent log."""
        g = self._gaze
        recent = "\n".join(g.log[-5:]) if g.log else "(no log yet)"
        return f"Query: {g.query!r}\nState: {g.state}\nFinal: {g.final_state or 'running'}\n{recent}"

    @function_tool()
    async def look(self, context: RunContext, question: str = "What do you see?") -> str:
        """Look through the robot's camera and answer a question about the scene."""
        return await _look(self, question)


# ---------------------------------------------------------------------------
# Shared look() implementation (used by function tool AND direct text dispatch)
# ---------------------------------------------------------------------------

ZMQ_CAM_HOST = os.environ.get("ZMQ_CAM_HOST", "172.17.240.1")
ZMQ_CAM_PORT = int(os.environ.get("ZMQ_CAM_PORT", "5555"))


def _grab_zmq_frame():
    """Grab one BGR frame from the OAK-D ZMQ PUB server (runs in a thread)."""
    try:
        import zmq, numpy as np, cv2, json, base64
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 3000)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.connect(f"tcp://{ZMQ_CAM_HOST}:{ZMQ_CAM_PORT}")
        raw = sock.recv_string()
        sock.close()
        msg = json.loads(raw)
        buf = base64.b64decode(msg["images"]["color"])
        arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


async def _look(agent: "GazeRobotAgent", question: str = "What do you see?") -> str:
    import cv2

    frame_bgr = await asyncio.to_thread(_grab_zmq_frame)
    if frame_bgr is None:
        return "Robot camera not active — ZMQ camera server unreachable."

    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    try:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        resp = await asyncio.to_thread(
            client.models.generate_content,
            model=VISION_MODEL,
            contents=[
                genai_types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"),
                question,
            ],
        )
        return resp.text.strip()
    except Exception as e:
        return f"Vision query failed: {e}"


# ---------------------------------------------------------------------------
# LiveKit session wiring
# ---------------------------------------------------------------------------

server = AgentServer()


async def _broadcast_state(room: rtc.Room, state: str) -> None:
    """Push arm state to all browser participants as a JSON data message."""
    try:
        payload = json.dumps({"type": "state", "state": state}).encode()
        await room.local_participant.publish_data(payload, reliable=True)
    except Exception:
        pass


@server.rtc_session(agent_name="rax-gaze-agent")
async def entrypoint(ctx: agents.JobContext) -> None:
    print(f"\n>>> [RAX] job received for room: {ctx.room.name}", flush=True)

    robot_agent = GazeRobotAgent(room=ctx.room)
    loop = asyncio.get_running_loop()

    # Register data_received BEFORE connecting so we never miss a message
    @ctx.room.on("data_received")
    def on_data(packet: rtc.DataPacket):
        try:
            msg = json.loads(packet.data.decode())
        except Exception:
            return
        mtype = msg.get("type")

        # ── Manual teleop from the phone UI (fast, silent, no TTS) ──
        if mtype == "jog":
            try:
                f = float(msg.get("f", 0.0))
                r = float(msg.get("r", 0.0))
                u = float(msg.get("u", 0.0))
            except (TypeError, ValueError):
                return
            robot_agent._gaze.send_jog(f"MOVE {f:.4f} {r:.4f} {u:.4f}")
            return
        if mtype == "grip":
            robot_agent._gaze.send_jog("GRIP OPEN" if msg.get("open") else "GRIP CLOSE")
            return
        if mtype == "home":
            robot_agent._gaze.send_raw("HOME")
            return

        if mtype != "command":
            return
        text = msg.get("text", "").strip()
        if not text:
            return
        print(f">>> [RAX] text command: {text!r}", flush=True)
        loop.create_task(_dispatch_text(robot_agent, session, ctx.room, text))

    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model=REALTIME_MODEL,
            voice="Aoede",
        ),
    )

    # start session first (hackathon-starter pattern), then connect
    await session.start(room=ctx.room, agent=robot_agent)
    await ctx.connect()
    robot_agent.start_camera_stream()  # safe now — room is connected
    print(f">>> [RAX] connected. Participants: {list(ctx.room.remote_participants.keys())}", flush=True)

    arm_desc = "mock arm" if USE_MOCK else f"SO-101 on {ROBOT_PORT or 'auto-detect'}"
    greeting = (
        f"RAX gaze robot online. Running {arm_desc}. "
        f"Loading YOLO and connecting arm in the background — "
        f"tell me what to pick up and I'll be ready in about 20 seconds."
    )
    await _say(session, ctx.room, greeting)


_PICK_WORDS  = {"pick", "grab", "get", "take", "grasp", "fetch", "lift"}
_DROP_WORDS  = {"drop", "release", "put", "place", "let"}
_LOOK_WORDS  = {"see", "look", "what", "describe", "show", "view", "scene", "there", "visible"}
_STOP_WORDS  = {"stop", "halt", "cancel", "freeze", "abort"}
_STATUS_WORDS = {"status", "state", "doing", "happening", "progress"}


async def _dispatch_text(
    agent: "GazeRobotAgent",
    session: "AgentSession",
    room: rtc.Room,
    text: str,
) -> None:
    """Parse a browser text command and dispatch to gaze engine or look()."""
    print(f">>> [dispatch] {text!r}", flush=True)
    words = set(text.lower().split())

    # ── STOP ──
    if words & _STOP_WORDS:
        agent._gaze.stop_task()
        await _say(session, room, "Stopping the arm.")
        return

    # ── DROP / RELEASE ──
    if words & _DROP_WORDS and not (words & _PICK_WORDS):
        ok = agent._gaze.send_raw("DROP")
        if ok:
            await _say(session, room, "Releasing — opening gripper.")
        else:
            await _say(session, room, f"Can't drop right now (state={agent._gaze.state}).")
        return

    # ── STATUS ──
    if words & _STATUS_WORDS:
        g = agent._gaze
        msg = f"State is {g.state}" + (f", query is {g.query!r}." if g.query else ".")
        await _say(session, room, msg)
        return

    # ── LOOK / DESCRIBE ──
    if words & _LOOK_WORDS and not (words & _PICK_WORDS):
        await _broadcast(room, {"type": "log", "text": f"Looking… ({text})"})
        result = await _look(agent, text)
        await _broadcast(room, {"type": "log", "text": f"Vision: {result[:200]}"})
        await _say(session, room, result)
        return

    # ── PICK UP / GRASP ──
    query = text
    for w in sorted(_PICK_WORDS, key=len, reverse=True):
        for phrase in (f"{w} up the ", f"{w} up a ", f"{w} up ", f"{w} the ", f"{w} a ", f"{w} "):
            if phrase in text.lower():
                query = text.lower().split(phrase, 1)[-1].strip()
                break

    if not query or query == text.lower():
        query = text

    logger.info("[dispatch] gaze_robot query=%r", query)
    result = agent._gaze.request_pick(query)

    if result == "busy":
        await _say(session, room, f"Already busy (state={agent._gaze.state}). Say stop first.")
        return

    if result == "queued":
        await _broadcast(room, {"type": "log", "text": f"Queued {query!r} — arm warming up ({agent._gaze.state})"})
        await _say(session, room, f"Warming up the arm — I'll grab the {query} as soon as it's ready.")
        return

    await _broadcast(room, {"type": "log", "text": f"Gaze engine: {query!r}"})
    await _say(session, room, f"On it. Searching for {query}.")


async def _broadcast(room: rtc.Room, msg: dict) -> None:
    try:
        await room.local_participant.publish_data(json.dumps(msg).encode(), reliable=True)
    except Exception as e:
        logger.debug("[broadcast] %s", e)


async def _say(session: "AgentSession", room: rtc.Room, text: str) -> None:
    """Speak text via Gemini if running, otherwise send as browser log."""
    print(f">>> [say] {text!r}", flush=True)
    try:
        await session.say(text)
    except Exception:
        await _broadcast(room, {"type": "reply", "text": text})




if __name__ == "__main__":
    agents.cli.run_app(server)
