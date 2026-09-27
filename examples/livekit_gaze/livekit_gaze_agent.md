# LiveKit Gaze Agent

Voice-controlled robot arm powered by **Gemini 3.1 Flash Audio** and **LiveKit Agents 1.6.4**.
A user speaks or points their webcam at an object — the agent understands, commands the
SO-101 arm via the [GazeEngine](manipulation_arms.md), and streams the robot's camera view
back into the same LiveKit room in real time.

Built for the **AI Engineer World's Fair Hackathon 2026** (San Francisco, June 27–28).

---

## Architecture

```
Browser (webcam + mic)
        │  WebRTC via LiveKit Cloud
        ▼
┌────────────────────────────────────────┐
│  agents/livekit_gaze_agent.py          │
│                                        │
│  Gemini 3.1 Flash Audio                │
│  ├─ sees user's webcam                 │
│  ├─ hears user's voice                 │
│  └─ function tools ──────────────────► GazeEngine (background thread)
│                                        │   SEARCH → APPROACH → GRASP
│  robot-eye track ◄────────────────────  OAK-D left frame (15 fps)
└────────────────────────────────────────┘
```

### Component Map

| Component | File | Role |
|-----------|------|------|
| `GazeRobotAgent` | `agents/livekit_gaze_agent.py` | Gemini agent with 3 function tools |
| `_GazeRunner` | same | Thread wrapper around `GazeEngine` |
| `_stream_robot_camera` | same | Async task publishing OAK-D frames as `robot-eye` |
| `GazeEngine` | `manipulation/arms/gaze_engine.py` | Visual servo state machine |
| `So101Arm` | `robots/arms/lerobot_so101/driver.py` | Real hardware driver |
| `MockArm` | `manipulation/arms/mock_arm.py` | Dev harness, no hardware needed |

---

## Prerequisites

**Python environment** — the lerobot venv or any Python 3.10+ env with:

```bash
pip install -r agents/requirements_livekit.txt
```

**Credentials** — copy `.env.local.example` and fill in your keys:

```bash
cp .env.local.example .env.local   # then edit
```

| Variable | Where to get it |
|----------|----------------|
| `LIVEKIT_URL` | [LiveKit Cloud](https://cloud.livekit.io) → your project → Settings |
| `LIVEKIT_API_KEY` | same page |
| `LIVEKIT_API_SECRET` | same page |
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) |

**Hardware (optional)** — SO-101 arm + OAK-D camera connected via USB/usbipd.  
Use `ROBOT_MOCK=1` to skip hardware entirely.

---

## Running

### Mock arm (no hardware)

```bash
ROBOT_MOCK=1 ./run_livekit_gaze.sh dev
```

### Real SO-101 (auto-detect port)

```bash
./run_livekit_gaze.sh dev
```

### Real SO-101 with explicit port

```bash
ROBOT_PORT=/dev/ttyACM0 ./run_livekit_gaze.sh dev
```

The `dev` argument starts the agent in development mode — it connects to your LiveKit project
and waits for a participant to join.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIVEKIT_URL` | *(required)* | `wss://your-project.livekit.cloud` |
| `LIVEKIT_API_KEY` | *(required)* | LiveKit API key |
| `LIVEKIT_API_SECRET` | *(required)* | LiveKit API secret |
| `GOOGLE_API_KEY` | *(required)* | Google / Gemini API key |
| `ROBOT_MOCK` | `0` | Set to `1` to use `MockArm` instead of real SO-101 |
| `ROBOT_PORT` | auto-detect | Serial port, e.g. `/dev/ttyACM0` |
| `ROBOT_URDF` | `SO101/so101_new_calib.urdf` | Path to the SO-101 URDF |
| `GRIPPER_CAM_TF` | `0.04,0,0.09,...` | Eye-in-hand extrinsic (x,y,z,rx,ry,rz) |
| `GAZE_APPROACH` | `angled` | Approach style: `angled` / `topdown` / `horizontal` |
| `GAZE_DETECTOR` | `auto` | Detector: `yolo` / `color_blob` / `blob` / `auto` |
| `GAZE_STEREO` | `sgbm` | Stereo backend: `sgbm` / `raft` / `foundation` / `auto` |

---

## Agent Function Tools

The `GazeRobotAgent` exposes three tools to Gemini:

### `gaze_robot(query, approach)`

Starts the gaze engine in a background thread. Returns immediately; the arm runs
autonomously through SEARCH → APPROACH → GRASP.

```
query    — e.g. "red cube", "blue block", "water bottle"
approach — "angled" (default), "topdown", "horizontal"
```

### `stop_robot()`

Signals the gaze engine to stop after the current tick and clears the runner.

### `robot_status()`

Returns the current state machine state + last 5 log lines.

```
Query: "red cube"
State: APPROACH
Final: running
Recent log:
  → SEARCH
  → APPROACH
```

---

## Robot Camera Stream

While a `_GazeRunner` is active, the OAK-D's left rectified frame is published into
the LiveKit room as a video track named **`robot-eye`** at 15 fps (640×400 px).

When the arm is idle (no active task or mock mode), a black placeholder frame is sent
so the track stays alive in the room.

Clients can subscribe to this track by name using the LiveKit client SDK.

---

## Frontend

Point any LiveKit-compatible client at your project room. The [gemini-hacker-starter](https://github.com/livekit-examples/gemini-hacker-starter) frontend works out of the box:

```bash
cd ../gemini-hacker-starter/frontend
pnpm install && pnpm dev
# open http://localhost:3000
```

Make sure to use the same `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`.

---

## WSL + usbipd Setup (Windows)

The SO-101 serial port needs to be forwarded from Windows into WSL:

```powershell
# In an elevated PowerShell prompt on Windows:
usbipd list                          # find the SO-101 (CP2102 or CH340)
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

Then in WSL:

```bash
ls /dev/ttyACM*    # should appear
```

The launch script auto-detects the port; or set `ROBOT_PORT=/dev/ttyACM0` explicitly.

---

## Troubleshooting

**`ValueError: ws_url is required`**
→ `.env.local` is missing or has placeholder values. Fill in your real LiveKit credentials.

**`No serial port found for SO-101`**
→ Run with `ROBOT_MOCK=1` or attach the USB device via `usbipd`.

**`ImportError: cannot import name 'agents' from 'livekit'`**
→ `livekit-agents` is not installed. Run:
```bash
pip install 'livekit-agents[google]>=0.13'
```

**Arm starts but never reaches GRASP**
→ Check `robot_status()` in the Gemini conversation. Common causes:
- Object not visible in OAK-D frame — reposition camera or object.
- Detector falling back to blob (no colour keyword in query) — add a colour: "red cube".
- Approach clearance too tight — try `GAZE_APPROACH=topdown`.
