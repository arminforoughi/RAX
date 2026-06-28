# Manipulation — Arms & Gaze Engine

Visual-servo grasping for the SO-101 arm: detect an object by name, track its 3D
centroid in the robot's base frame, approach it from any direction, and close the
gripper. The entire pipeline runs with **no GPU and no model weights** via graceful
fallbacks, so development needs only a laptop.

---

## Pipeline

```
OAK-D stereo pair (left + right rectified)
        │
        ├─ PromptDetector ────► bounding boxes + class labels
        │   yolo-world (GPU)      (open-vocab: "red cube", "water bottle", ...)
        │   color_blob (CPU)      (HSV fallback for colour queries)
        │
        ├─ MaskTracker ──────► per-object soft masks
        │   SAM2 (GPU)
        │   ellipse (CPU)
        │
        └─ StereoDepth ──────► dense disparity → depth
            RAFT-Stereo (GPU)
            FoundationStereo (GPU)
            SGBM (CPU)

        ↓  CloudTracker
        per-object 3D point clouds in base frame
        budgeted: focused object every tick, others round-robin

        ↓  GazeEngine
        SEARCH → TRACK → APPROACH → GRASP → PLACE → DONE / FAILED

        ↓  ArmInterface
        MockArm  (synthetic scene, no hardware)
        So101Arm (real SO-101 + OAK-D via lerobot)
```

---

## Module Reference

### `manipulation/arms/gaze_engine.py` — `GazeEngine`

State machine that drives the arm:

| State | What happens |
|-------|-------------|
| `SEARCH` | Pan the arm looking for the target object |
| `TRACK` | Hold on the object; build up point-cloud estimate |
| `APPROACH` | Move EE toward the target using approach style |
| `GRASP` | Lower gripper and close |
| `PLACE` | Move to place-on target and open gripper |
| `DONE` | Grasp (and optional place) succeeded |
| `FAILED` | Timed out or lost track |

**Key params** (via `GazeConfig`):

| Field | Default | Description |
|-------|---------|-------------|
| `approach_style` | `angled` | `angled` / `topdown` / `horizontal` |
| `T_ee_cam` | eye identity | Eye-in-hand transform (4×4 homogeneous) |
| `world_up` | `[0,0,1]` | World up vector for orbit math |
| `gaze_kp_tilt` | `0.4` | Proportional gain on tilt axis |
| `max_search_ticks` | `300` | Steps before declaring FAILED |

### `manipulation/arms/arm_interface.py` — `ArmInterface`

Protocol (structural typing) every arm driver must satisfy:

```python
def get_observation() -> Observation   # stereo frames, joint angles, gripper
def send_joint_targets(q_deg)          # 5-DOF position targets in degrees
def set_gripper(pct)                   # 0 = open, 100 = closed
def read_gripper_current() -> float    # mA, for grasp-contact detection
```

### `manipulation/arms/mock_arm.py` — `MockArm`

Synthetic scene: red, green, and blue spheres ~0.4 m in front of a virtual
eye-in-hand camera. No hardware, no display — state logs are the output.

```python
from manipulation.arms.mock_arm import MockArm, WORLD_UP
arm = MockArm()
obs = arm.get_observation()   # left/right frames with coloured blobs
```

### `manipulation/arms/kinematics.py`

| Class | Backend | Use |
|-------|---------|-----|
| `PlacoKinematics` | lerobot's placo FK/IK | Real SO-101 |
| `CartesianKinematics` | pure maths | MockArm / dev |

### `manipulation/arms/lerobot_so101/` — RAX-native CLI

Drop-in replacement for the `lerobot-gaze-engine` console script, with all the
same flags. Run directly as a module:

```bash
python -m manipulation.arms.lerobot_so101 \
  --robot.port /dev/ttyACM0 \
  --urdf SO101/so101_new_calib.urdf \
  --query "red cube" \
  --gripper-camera-tf "0.04,0,0.09,-0.2690,0.2824,-1.6014" \
  --display-data
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--robot.port` | *(required)* | SO-101 serial port |
| `--query` | `red box` | Object to grasp |
| `--urdf` | `SO101/so101_new_calib.urdf` | URDF path |
| `--gripper-camera-tf` | see above | Eye-in-hand extrinsic |
| `--approach-style` | `angled` | `angled` / `topdown` / `horizontal` |
| `--model-path` | *(none)* | YOLO-World weights path |
| `--display-data` | off | Live OpenCV window |
| `--display-sim3d` | off | Rerun 3D viewer |
| `--live-control-keypress` | off | WASD keyboard nudge |

### `robots/arms/lerobot_so101/driver.py` — `So101Arm`

Real-hardware `ArmInterface` wrapping lerobot's `make_robot_from_config`:

- Opens the SO-101 follower via `SOFollowerRobotConfig(port=port)`
- Opens OAK-D via `OAKDCameraConfig(export_stereo_rectified=True)` — raw rectified
  stereo frames (no firmware depth)
- `latest_left_bgr()` — non-blocking peek at the current left frame for video streaming

---

## Quick Start

### No hardware (mock)

```bash
python -m manipulation.arms.run_gaze --backend mock --query "red cube"
```

### With SO-101 + OAK-D

```bash
python -m manipulation.arms.lerobot_so101 \
  --robot.port /dev/ttyACM0 \
  --query "red cube"
```

### Via LiveKit agent (voice-controlled)

```bash
ROBOT_MOCK=1 ./run_livekit_gaze.sh dev   # mock
./run_livekit_gaze.sh dev                 # real arm
```

See [livekit_gaze_agent.md](livekit_gaze_agent.md) for the full agent docs.

---

## Stereo Depth Backends

| Backend | Flag | Requires | Speed |
|---------|------|----------|-------|
| SGBM | `sgbm` | OpenCV | Fast (CPU) |
| RAFT-Stereo | `raft` | PyTorch + weights | Accurate (GPU) |
| FoundationStereo | `foundation` | PyTorch + weights | Best quality (GPU) |
| auto | `auto` | — | Uses best available |

Set weights via env vars:

```bash
# RAFT-Stereo
export RAFT_STEREO_REPO=/path/to/RAFT-Stereo
export RAFT_STEREO_CKPT=/path/to/raftstereo-sceneflow.pth

# FoundationStereo
export FOUNDATION_STEREO_REPO=/path/to/FoundationStereo
export FOUNDATION_STEREO_CKPT=/path/to/model.pth
```

## Detection Backends

| Backend | Flag | Requires | Notes |
|---------|------|----------|-------|
| YOLO-World | `yolo` | ultralytics weights | Open-vocab, GPU |
| ColorBlob | `color_blob` | nothing | HSV; colour keyword in query |
| Blob | `blob` | nothing | Foreground segmentation |
| auto | `auto` | — | color_blob if colour word found, else blob |

```bash
export YOLO_WORLD_MODEL=/path/to/yolov8s-worldv2.pt   # optional
```
