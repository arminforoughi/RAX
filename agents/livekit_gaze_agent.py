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

PERSONA = """You are a gaze-controlled robot arm assistant at the AI Engineer World's Fair Hackathon 2026.

You see through the user's webcam. When they look at or describe an object on the table
in front of the robot arm, you can command the arm to pick it up.

Your tools:
- gaze_robot: command the SO-101 arm to find and grasp a named object.
- stop_robot: immediately stop the arm.
- robot_status: ask what the arm is currently doing.

Workflow:
1. Watch the user's camera. If they point at or describe an object, ask to confirm.
2. Call gaze_robot with the object name (e.g. "red cube", "blue block").
3. Narrate the robot state as it searches → approaches → grasps.
4. Celebrate when it succeeds or diagnose and retry if it fails.

Keep responses short and energetic — this is a live hackathon demo.
Never invent visual details you cannot actually see."""

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
    ) -> None:
        self.query          = query
        self.approach_style = approach_style
        self.max_ticks      = max_ticks
        self.loop_hz        = loop_hz

        self.state       = "INIT"
        self.final_state: str | None = None
        self.log: list[str] = []

        self._stop = threading.Event()
        self._arm  = None   # set inside the thread

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
                return

            prev = self.state
            s    = engine.step(dt)
            self.state = s

            if s != prev:
                self._note(f"→ {s}")

            if s in (self._DONE, self._FAILED):
                break

            time.sleep(dt)

        self.final_state = self.state
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


# ---------------------------------------------------------------------------
# LiveKit session wiring
# ---------------------------------------------------------------------------

server = AgentServer()


@server.rtc_session(agent_name="rax-gaze-agent")
async def entrypoint(ctx: agents.JobContext) -> None:
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model=REALTIME_MODEL,
            voice="Aoede",
        ),
    )

    await session.start(
        room=ctx.room,
        agent=GazeRobotAgent(room=ctx.room),
    )
    await ctx.connect()

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
