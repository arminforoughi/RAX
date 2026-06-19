#!/usr/bin/env bash
# Launch the RAX gaze engine on the real SO-101 + OAK-D: search -> locate ->
# focus/track -> approach -> grasp, with the Rerun viewer (camera + 3D point cloud).
#
# Usage:
#   ./run_gaze_engine.sh                         # grab the "cube" (default)
#   ./run_gaze_engine.sh --query "red cube"      # colour blob if query has a colour word
#   ./run_gaze_engine.sh --place-on "cube"       # grab, then place on another object
#   ./run_gaze_engine.sh --no-move               # perception + Rerun only, arm never moves
#   ./run_gaze_engine.sh --pan-sign 1            # flip if arm pans the wrong way
#   QUERY="red cube" ./run_gaze_engine.sh        # override via env var (unless --query passed)
#
# Any extra flags are passed straight through to manipulation.arms.run_gaze.
# Override defaults with env vars: PORT, QUERY, DETECTOR, STEREO, MAX_DISP, MAX_TICKS, RERUN=0.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# --- Python: prefer the isolated arm venv (depthai + placo + feetech, no glog clash) ---
PY="$REPO_DIR/.venv-arm/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "!! venv not found at $PY"
  echo "   Create it once with:"
  echo "     python3 -m venv .venv-arm && .venv-arm/bin/pip install -U pip"
  echo "     .venv-arm/bin/pip install -e \"/Users/armin/Documents/lerobot[feetech,kinematics,oakd,yolo-world]\" feetech-servo-sdk"
  exit 1
fi

# --- Auto-detect the SO-101 serial port if not provided ---
if [[ -z "${PORT:-}" ]]; then
  PORT="$(ls /dev/tty.usbmodem* 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${PORT:-}" ]]; then
  echo "!! No SO-101 serial port found (no /dev/tty.usbmodem*). Set PORT=/dev/tty.usbmodemXXXX"
  exit 1
fi

# --- Defaults tuned for the SO-101 + OAK-D close-range pick ---
QUERY="${QUERY:-cube}"
APPROACH="${APPROACH:-topdown}" # how the gripper comes in: topdown | horizontal | angled
DETECTOR="${DETECTOR:-auto}"   # auto: colour word -> color_blob, else foreground blob
STEREO="${STEREO:-sgbm}"       # raft/foundation if you have weights; sgbm always works
MAX_DISP="${MAX_DISP:-384}"    # close objects need a big disparity range (~18cm => ~340px)
MAX_TICKS="${MAX_TICKS:-400}"
RERUN_FLAG=""
[[ "${RERUN:-1}" == "1" ]] && RERUN_FLAG="--rerun"

# If the user passed --query on the CLI, do not override it with QUERY= from the env.
HAS_QUERY=false
for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == --query ]]; then HAS_QUERY=true; break; fi
done

echo ">>> gaze engine: port=$PORT detector=$DETECTOR stereo=$STEREO max_disp=$MAX_DISP"
echo ">>> The arm WILL move (unless you passed --no-move). Keep the workspace clear; Ctrl-C to stop."

CMD=( "$PY" -m manipulation.arms.run_gaze
  --backend so101
  --port "$PORT"
  --detector "$DETECTOR"
  --stereo "$STEREO"
  --max-disp "$MAX_DISP"
  --max-ticks "$MAX_TICKS"
  --grasp-range 0.10
  --final-advance 0.02
  --aim-v-offset 120
  --approach-style "$APPROACH"
)
[[ -n "$RERUN_FLAG" ]] && CMD+=( "$RERUN_FLAG" )
if [[ "$HAS_QUERY" == false ]]; then
  CMD+=( --query "$QUERY" )
  echo ">>> query='$QUERY'"
else
  echo ">>> query from CLI args"
fi
CMD+=( "$@" )

exec "${CMD[@]}"
