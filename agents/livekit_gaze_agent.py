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
- gaze_robot: command the SO-101 arm to find and grasp a named object.
- stop_robot: immediately stop the arm.
- robot_status: ask what the arm is currently doing.

Workflow:
1. If the user asks what you see → call look("what do you see?").
2. If they describe or point at an object → confirm, then call gaze_robot("object name").
3. Narrate the robot state as it searches → approaches → grasps.
4. Celebrate success or diagnose failure.

Keep responses short and energetic — this is a live hackathon demo.
Never invent visual details you cannot actually see — use look() to check."""

# ---------------------------------------------------------------------------
# GazeRunner — manages GazeEngine in a background thread
# ---------------------------------------------------------------------------

class _PersistentGaze:
    """Always-on gaze engine: initializes hardware once, accepts queries at runtime.

    Architecture:
      - Background thread initializes arm + camera + Rerun once on start().
      - pick(query) queues a new task; a fresh CloudTracker + GazeEngine is
        created per query (cheap, ~50 ms) while the arm stays connected.
      - Rerun streams continuously — even while idle the camera image updates.
    """

    LOOP_HZ  = 20.0
    MAX_TICKS = 600   # 30 s at 20 Hz

    def __init__(self) -> None:
        self.state       = "INIT"
        self.query: str | None = None
        self.final_state: str | None = None
        self.log: list[str] = []
        self.on_state_change: "callable | None" = None

        self._shutdown   = threading.Event()
        self._task_stop  = threading.Event()
        self._task_ready = threading.Event()
        self._lock       = threading.Lock()
        self._pending_query: str | None = None

        # initialized in _setup()
        self._arm   = None
        self._kin   = None
        self._viz   = None
        self._cfg   = None
        self._detector     = None
        self._mask_tracker = None
        self._stereo       = None
        self._idle_cloud   = None   # empty cloud for idle Rerun ticks

        self._thread: threading.Thread | None = None

    # ── public API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn background thread; returns immediately."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="persistent-gaze")
        self._thread.start()

    def pick(self, query: str) -> bool:
        """Queue a pick task. Returns False if arm is currently mid-task."""
        with self._lock:
            busy = self.state not in ("IDLE", "DONE", "FAILED", "STOPPED", "INIT")
            if busy:
                return False
            self._pending_query = query
            self._task_stop.clear()
            self._task_ready.set()
        return True

    def stop_task(self) -> None:
        self._task_stop.set()

    def shutdown(self) -> None:
        self._shutdown.set()
        self._task_stop.set()
        self._task_ready.set()

    # ── internals ───────────────────────────────────────────────────────────

    def _note(self, msg: str) -> None:
        logger.info("[PersistentGaze] %s", msg)
        self.log.append(msg)

    def _run(self) -> None:
        try:
            self._setup()
        except Exception as exc:
            logger.exception("[PersistentGaze] setup failed: %s", exc)
            self.state = "FAILED"
            if self.on_state_change:
                self.on_state_change("FAILED")
            return

        self._note("arm ready — waiting for queries")
        self.state = "IDLE"
        if self.on_state_change:
            self.on_state_change("IDLE")

        dt = 1.0 / self.LOOP_HZ
        while not self._shutdown.is_set():
            # Wait up to 50 ms then do an idle Rerun tick
            self._task_ready.wait(timeout=0.05)
            if self._shutdown.is_set():
                break

            with self._lock:
                query = self._pending_query
                if query:
                    self._pending_query = None
                    self._task_ready.clear()

            if query:
                self.query = query
                self._run_task(query, dt)
                self.state = "IDLE"
                if self.on_state_change:
                    self.on_state_change("IDLE")
            else:
                self._idle_tick()

        self._teardown()

    def _idle_tick(self) -> None:
        """Stream camera to Rerun while waiting for a command."""
        if self._viz is None or self._arm is None:
            return
        try:
            obs = self._arm.get_observation()
            self._viz.log(obs, self._idle_cloud, "IDLE", kin=self._kin)
        except Exception:
            pass
        time.sleep(0.05)

    def _run_task(self, query: str, dt: float) -> None:
        from perception.depth_cloud import CloudTracker, PointCloudStream
        from manipulation.arms.gaze_engine import GazeEngine, DONE, FAILED

        self._note(f"starting task query={query!r}")
        stream = PointCloudStream()
        cloud  = CloudTracker(
            self._detector, self._mask_tracker, self._stereo, query,
            detect_every=1, stream=stream,
        )
        engine = GazeEngine(self._arm, self._kin, cloud, self._cfg, cartesian=USE_MOCK)

        prev = None
        for _ in range(self.MAX_TICKS):
            if self._task_stop.is_set() or self._shutdown.is_set():
                self._note("task stopped")
                self.state = "STOPPED"
                self.final_state = "STOPPED"
                if self.on_state_change:
                    self.on_state_change("STOPPED")
                return

            s = engine.step(dt)
            self.state = s
            if s != prev:
                self._note(f"→ {s}")
                if self.on_state_change:
                    self.on_state_change(s)
                prev = s

            if self._viz is not None and engine.last_obs is not None:
                try:
                    focus = cloud.focus_track()
                    u_aim = v_aim = None
                    if focus is not None:
                        u_aim, v_aim = engine._gaze_uv(focus, engine.last_obs)
                    self._viz.log(
                        engine.last_obs, cloud, s,
                        kin=self._kin,
                        depth_m=engine.range_m,
                        u_aim=u_aim, v_aim=v_aim,
                        aim_v_offset_px=engine._aim_v_now,
                    )
                except Exception as e:
                    logger.debug("[rerun] %s", e)

            if s in (DONE, FAILED):
                self.final_state = s
                self._note(f"finished: {s}")
                return

            time.sleep(dt)

        self.final_state = self.state
        self._note(f"timed out in state={self.state}")

    def _setup(self) -> None:
        import numpy as np
        from models.depth import make_stereo
        from models.detection import make_detector, make_mask_tracker
        from perception.depth_cloud import CloudTracker, PointCloudStream
        from manipulation.arms.gaze_engine import GazeConfig

        if USE_RERUN:
            from perception.depth_cloud.rerun_viz import RerunViz
            self._viz = RerunViz.try_create(session="rax_gaze_agent", spawn=True)

        if USE_MOCK:
            from manipulation.arms.mock_arm import WORLD_UP, MockArm
            from manipulation.arms.kinematics import CartesianKinematics
            self._arm    = MockArm()
            self._kin    = CartesianKinematics()
            T_ee_cam     = np.eye(4)
            world_up     = WORLD_UP
            det_backend  = "color_blob"
            mask_backend = "ellipse"
            stereo_backend = "sgbm"
        else:
            from robots.arms.lerobot_so101.driver import So101Arm
            self._arm    = So101Arm(
                port=ROBOT_PORT or self._auto_detect_port(),
                urdf=ROBOT_URDF,
                gripper_camera_tf=GRIPPER_CAM_TF,
            )
            self._kin    = self._arm.kin
            T_ee_cam     = self._arm.T_ee_cam
            world_up     = np.array([0.0, 0.0, 1.0])
            det_backend  = GAZE_DETECTOR if GAZE_DETECTOR != "auto" else "yolo"
            mask_backend = "auto"
            stereo_backend = GAZE_STEREO

        self._detector     = make_detector(det_backend)
        self._mask_tracker = make_mask_tracker(mask_backend)
        self._stereo       = make_stereo(stereo_backend)

        self._cfg = GazeConfig(
            T_ee_cam=T_ee_cam,
            world_up=world_up,
            approach_style=GAZE_APPROACH,
            gaze_kp_pan=0.4,
            gaze_kp_tilt=0.5,
            approach_kp=1.8,
            max_lin_vel_m_s=0.14,
            max_joint_step_deg=7.0,
            center_tol_px=80.0,
            search_yaw_amp_deg=28.0,
            search_period_s=12.0,
            search_timeout_s=60.0,
            pregrasp_standoff_m=0.10,
            approach_close_m=0.125,
            approach_close_step_m=0.010,
        )

        # Empty cloud used for idle Rerun ticks (shows camera with no detections)
        stream = PointCloudStream()
        self._idle_cloud = CloudTracker(
            self._detector, self._mask_tracker, self._stereo, "",
            detect_every=999999, stream=stream,
        )

    def _teardown(self) -> None:
        arm = self._arm
        self._arm = None
        if arm is not None and not USE_MOCK:
            try:
                arm.disconnect()
            except Exception:
                pass

    @staticmethod
    def _auto_detect_port() -> str:
        import glob as _glob
        for pat in ("/dev/ttyACM*", "/dev/ttyUSB*", "/dev/ttyS[2-9]"):
            cands = sorted(_glob.glob(pat))
            if cands:
                return cands[0]
        raise RuntimeError("No serial port found for SO-101. Set ROBOT_PORT=/dev/ttyACMx")


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
            frame_bgr = None
            arm = gaze._arm
            if arm is not None and not USE_MOCK:
                try:
                    frame_bgr = arm.latest_left_bgr()
                except Exception:
                    pass

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
        ok = self._gaze.pick(query)
        if not ok:
            return f"Arm is busy (state={self._gaze.state}). Say stop first."
        return f"Searching for {query!r}. Call robot_status() to track progress."

    @function_tool()
    async def stop_robot(self, context: RunContext) -> str:
        """Stop the current gaze task. Arm stays connected and ready."""
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

async def _look(agent: "GazeRobotAgent", question: str = "What do you see?") -> str:
    import cv2

    frame_bgr = None
    arm = agent._gaze._arm
    if arm is not None:
        try:
            if USE_MOCK:
                obs = arm.get_observation()
                frame_bgr = cv2.cvtColor(obs.left, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = arm.latest_left_bgr()
        except Exception:
            pass

    if frame_bgr is None and USE_MOCK:
        try:
            from manipulation.arms.mock_arm import MockArm
            obs = MockArm().get_observation()
            frame_bgr = cv2.cvtColor(obs.left, cv2.COLOR_RGB2BGR)
        except Exception as e:
            return f"Could not capture frame: {e}"

    if frame_bgr is None:
        return "Robot camera not active — arm may still be initializing."

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
        if msg.get("type") != "command":
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
    greeting = f"RAX gaze robot online. Running {arm_desc}. Tell me what to pick up, or ask what I see."
    await _say(session, ctx.room, greeting)


_PICK_WORDS  = {"pick", "grab", "get", "take", "grasp", "fetch", "lift"}
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
    ok = agent._gaze.pick(query)
    if not ok:
        await _say(session, room, f"Already busy (state={agent._gaze.state}). Say stop first.")
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
