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

REALTIME_MODEL   = "gemini-3.1-flash-audio-eap"
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

class _GazeRunner:
    """Runs a single gaze-pick task synchronously in a worker thread.

    Thread-safe state is exposed via ``state``, ``final_state``, and ``log``.
    """

    def __init__(
        self,
        query: str,
        approach_style: str = GAZE_APPROACH,
        max_ticks: int = 600,
        loop_hz: float = 20.0,
        on_state_change: "callable | None" = None,
    ) -> None:
        self.query          = query
        self.approach_style = approach_style
        self.max_ticks      = max_ticks
        self.loop_hz        = loop_hz
        self.on_state_change = on_state_change  # called(state) on each transition

        self.state       = "INIT"
        self.final_state: str | None = None
        self.log: list[str] = []

        self._stop = threading.Event()
        self._arm  = None   # set inside the thread
        self._viz  = None   # RerunViz if available

    # -- public API -------------------------------------------------------

    def run_until_done(self) -> None:
        """Blocking: open hardware, run engine, close hardware. Call in a thread."""
        try:
            self._setup()
            self._loop()
        except Exception as exc:
            logger.exception("[GazeRunner] fatal: %s", exc)
            self._note(f"FAILED: {exc}")
            self.state = self.final_state = "FAILED"
        finally:
            self._teardown()

    def stop(self) -> None:
        """Signal the engine to stop after the current tick."""
        self._stop.set()

    def stopped(self) -> bool:
        return self._stop.is_set()

    # -- internals --------------------------------------------------------

    def _note(self, msg: str) -> None:
        logger.info("[GazeRunner] %s", msg)
        self.log.append(msg)

    def _setup(self) -> None:
        from models.depth import make_stereo
        from models.detection import make_detector, make_mask_tracker
        from perception.depth_cloud import CloudTracker, PointCloudStream
        from manipulation.arms.gaze_engine import GazeConfig, GazeEngine, DONE, FAILED
        if USE_RERUN:
            from perception.depth_cloud.rerun_viz import RerunViz
            self._viz = RerunViz.try_create(session="rax_gaze_agent", spawn=True)

        self._note(f"opening arm  mock={USE_MOCK}  query={self.query!r}")

        if USE_MOCK:
            from manipulation.arms.mock_arm import WORLD_UP, MockArm
            from manipulation.arms.kinematics import CartesianKinematics
            import numpy as np

            self._arm = MockArm()
            kin       = CartesianKinematics()
            T_ee_cam  = np.eye(4)
            world_up  = WORLD_UP
        else:
            from robots.arms.lerobot_so101.driver import So101Arm
            self._arm = So101Arm(
                port=ROBOT_PORT or self._auto_detect_port(),
                urdf=ROBOT_URDF,
                gripper_camera_tf=GRIPPER_CAM_TF,
            )
            kin      = self._arm.kin
            T_ee_cam = self._arm.T_ee_cam
            import numpy as np
            world_up = np.array([0.0, 0.0, 1.0])

        # detector
        det_backend = GAZE_DETECTOR
        if det_backend == "auto":
            from models.detection.prompt_detector import _HSV_RANGES
            words = self.query.lower().split()
            det_backend = "color_blob" if any(w in _HSV_RANGES for w in words) else "blob"
        if USE_MOCK:
            det_backend = "color_blob"

        detector     = make_detector(det_backend)
        mask_tracker = make_mask_tracker("ellipse" if USE_MOCK else "auto")
        stereo       = make_stereo("sgbm" if USE_MOCK else GAZE_STEREO)

        stream = PointCloudStream()
        cloud  = CloudTracker(
            detector, mask_tracker, stereo, self.query,
            detect_every=5, stream=stream,
        )

        cfg = GazeConfig(
            T_ee_cam=T_ee_cam,
            world_up=world_up,
            approach_style=self.approach_style,
        )
        self._cloud  = cloud
        self._engine = GazeEngine(
            self._arm, kin, cloud, cfg,
            cartesian=USE_MOCK,
        )
        self._DONE   = "DONE"
        self._FAILED = "FAILED"

    def _loop(self) -> None:
        dt     = 1.0 / max(1.0, self.loop_hz)
        engine = self._engine

        for _ in range(self.max_ticks):
            if self._stop.is_set():
                self._note("stopped by user")
                self.final_state = "STOPPED"
                if self.on_state_change:
                    self.on_state_change("STOPPED")
                return

            prev = self.state
            s    = engine.step(dt)
            self.state = s

            if s != prev:
                self._note(f"→ {s}")
                if self.on_state_change:
                    self.on_state_change(s)

            # Stream to Rerun viewer
            if self._viz is not None:
                try:
                    obs   = self._arm.get_observation()
                    u_aim = getattr(engine, '_u_aim', None)
                    v_aim = getattr(engine, '_v_aim', None)
                    self._viz.log(obs, self._cloud, state=s, u_aim=u_aim, v_aim=v_aim)
                except Exception:
                    pass

            if s in (self._DONE, self._FAILED):
                break

            time.sleep(dt)

        self.final_state = self.state
        if self.on_state_change:
            self.on_state_change(self.final_state)
        self._note(f"finished: {self.final_state}")

    def _teardown(self) -> None:
        arm = self._arm
        if arm is None:
            return
        try:
            if USE_MOCK:
                pass
            else:
                arm.disconnect()
        except Exception:
            pass
        self._arm = None

    @staticmethod
    def _auto_detect_port() -> str:
        import glob as _glob
        for pat in ("/dev/ttyACM*", "/dev/ttyUSB*", "/dev/ttyS[2-9]"):
            cands = sorted(_glob.glob(pat))
            if cands:
                return cands[0]
        raise RuntimeError(
            "No serial port found for SO-101. Set ROBOT_PORT=/dev/ttyACMx"
        )


# ---------------------------------------------------------------------------
# Robot camera → LiveKit video track
# ---------------------------------------------------------------------------

async def _stream_robot_camera(
    room: rtc.Room,
    runner_ref: list[_GazeRunner | None],
    fps: int = 15,
) -> None:
    """Continuously publish the robot's OAK-D left frame as a LiveKit video track.

    Reads frames directly from the arm that the current GazeRunner holds.
    Falls back to a black 640×400 frame when the arm is idle.
    """
    import numpy as np
    import cv2

    W, H   = 640, 400
    source = rtc.VideoSource(width=W, height=H)
    track  = rtc.LocalVideoTrack.create_video_track("robot-eye", source)
    opts   = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await room.local_participant.publish_track(track, opts)
    logger.info("[robot-cam] published robot-eye video track")

    blank = np.zeros((H, W, 4), dtype=np.uint8)  # RGBA black

    interval = 1.0 / fps
    while True:
        try:
            runner = runner_ref[0]
            frame_bgr = None
            if runner is not None and not USE_MOCK:
                arm = runner._arm
                if arm is not None:
                    try:
                        frame_bgr = arm.latest_left_bgr()
                    except Exception:
                        pass

            if frame_bgr is not None:
                frame_resized = cv2.resize(frame_bgr, (W, H))
                rgba = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGBA)
            else:
                rgba = blank

            vf = rtc.VideoFrame(
                width=W, height=H,
                type=rtc.VideoBufferType.RGBA,
                data=rgba.tobytes(),
            )
            source.capture_frame(vf)
        except Exception as exc:
            logger.debug("[robot-cam] frame error: %s", exc)

        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# GazeRobotAgent
# ---------------------------------------------------------------------------

class GazeRobotAgent(Agent):
    """Gemini-powered agent that controls the SO-101 via voice + vision."""

    def __init__(self, room: rtc.Room) -> None:
        super().__init__(instructions=PERSONA)
        self._room       = room
        self._runner: _GazeRunner | None = None
        self._runner_ref: list[_GazeRunner | None] = [None]  # shared with camera streamer
        self._executor   = None   # ThreadPoolExecutor set in on_session_start

    # ---- lifecycle ------------------------------------------------------

    async def on_enter(self) -> None:
        import concurrent.futures
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gaze")
        asyncio.create_task(_stream_robot_camera(self._room, self._runner_ref))
        logger.info("[GazeRobotAgent] entered")

    # ---- function tools -------------------------------------------------

    @function_tool()
    async def gaze_robot(
        self,
        context: RunContext,
        query: str,
        approach: str = GAZE_APPROACH,
    ) -> str:
        """Command the SO-101 arm to find and grasp the named object.

        Starts the gaze engine in the background and returns immediately.
        The arm will search, approach, and grasp autonomously.

        Args:
            query: Natural-language object description, e.g. "red cube" or "blue block".
            approach: How the gripper comes in — "angled" (default), "topdown", or "horizontal".
        """
        if self._runner is not None and self._runner.state not in ("DONE", "FAILED", "STOPPED", "INIT"):
            return (
                f"Arm is already running (state={self._runner.state}, query={self._runner.query!r}). "
                "Call stop_robot first."
            )

        runner = _GazeRunner(query=query, approach_style=approach)
        self._runner             = runner
        self._runner_ref[0]      = runner

        loop = asyncio.get_running_loop()
        loop.run_in_executor(self._executor, runner.run_until_done)

        return (
            f"Gaze engine started — searching for {query!r} "
            f"(approach={approach}). Call robot_status() to check progress."
        )

    @function_tool()
    async def stop_robot(self, context: RunContext) -> str:
        """Stop the robot arm immediately and cancel the current gaze task."""
        if self._runner is None:
            return "Robot is already idle."
        self._runner.stop()
        state = self._runner.state
        self._runner         = None
        self._runner_ref[0]  = None
        return f"Stop signal sent (was in state={state})."

    @function_tool()
    async def robot_status(self, context: RunContext) -> str:
        """Return the current state of the gaze engine and recent log messages."""
        if self._runner is None:
            return "Robot is idle (no active task)."
        r = self._runner
        recent = "\n".join(r.log[-5:]) if r.log else "(no log yet)"
        return (
            f"Query: {r.query!r}\n"
            f"State: {r.state}\n"
            f"Final: {r.final_state or 'running'}\n"
            f"Recent log:\n{recent}"
        )

    @function_tool()
    async def look(self, context: RunContext, question: str = "What do you see?") -> str:
        """Look through the robot's camera and answer a question about the scene.

        Captures the current robot camera frame and asks Gemini Flash to describe it.
        Use this when the user asks what the robot sees, what's on the table, where
        an object is, or any question about the robot's point of view.

        Args:
            question: The question to answer about the robot's view.
        """
        import cv2
        import base64
        import numpy as np

        # Grab a frame from the active runner or directly from mock arm
        frame_bgr = None
        runner = self._runner
        if runner is not None and runner._arm is not None:
            try:
                if USE_MOCK:
                    obs = runner._arm.get_observation()
                    frame_bgr = obs.left
                    if frame_bgr.shape[2] == 3:
                        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = runner._arm.latest_left_bgr()
            except Exception:
                pass

        if frame_bgr is None:
            # Spin up a fresh mock arm just to get a frame
            if USE_MOCK:
                try:
                    from manipulation.arms.mock_arm import MockArm
                    _tmp = MockArm()
                    obs = _tmp.get_observation()
                    frame_bgr = obs.left
                    if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
                        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    return f"Could not capture robot camera frame: {e}"
            else:
                return "Robot camera not active — start the arm first or use ROBOT_MOCK=1."

        # Encode as JPEG for the vision model
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode()

        # Ask Gemini Flash to describe the scene
        try:
            from google import genai
            from google.genai import types as genai_types
            client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.0-flash",
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
    robot_agent = GazeRobotAgent(room=ctx.room)

    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model=REALTIME_MODEL,
            voice="Aoede",
        ),
    )

    await session.start(room=ctx.room, agent=robot_agent)
    await ctx.connect()

    # Listen for text messages from the browser demo UI
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
        logger.info("[entrypoint] browser text: %r", text)
        # Route to Gemini as a user turn — it decides whether to call a tool or just answer
        asyncio.get_running_loop().create_task(
            session.generate_reply(user_input=text)
        )

    try:
        arm_desc = "mock arm (ROBOT_MOCK=1)" if USE_MOCK else f"SO-101 on {ROBOT_PORT or 'auto-detect'}"
        await session.generate_reply(
            instructions=(
                f"Greet the user. Tell them you control a robot arm ({arm_desc}) "
                "and can pick up objects they describe. Ask what they'd like you to grab."
            )
        )
    except Exception as exc:
        logger.warning("[entrypoint] greeting failed: %s", exc)




if __name__ == "__main__":
    agents.cli.run_app(server)
