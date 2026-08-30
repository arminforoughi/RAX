#!/usr/bin/env bash
# Launch the RAX LiveKit + Gemini 3.1 gaze-controlled robot agent.
#
# The agent connects to a LiveKit room, accepts the user's webcam + voice via
# Gemini 3.1 Flash Audio, and commands the SO-101 arm via the GazeEngine.
# The robot's OAK-D camera streams back to the LiveKit room in real time.
#
# Usage:
#   ./run_livekit_gaze.sh             # real SO-101 (auto-detect port)
#   ROBOT_MOCK=1 ./run_livekit_gaze.sh  # mock arm — no hardware required
#   ROBOT_PORT=/dev/ttyACM0 ./run_livekit_gaze.sh
#
# Credentials — set in .env.local or export directly:
#   LIVEKIT_URL        wss://your-project.livekit.cloud
#   LIVEKIT_API_KEY    your LiveKit key
#   LIVEKIT_API_SECRET your LiveKit secret
#   GOOGLE_API_KEY     your Gemini / Google AI key
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Use the RAX arm venv if present; else fall back to whatever python3 is active.
PY="${VENV_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for cand in \
    "$REPO_DIR/.venv-arm/bin/python" \
    "$REPO_DIR/.venv/bin/python" \
    "$(which python3)"; do
    [[ -x "$cand" ]] && PY="$cand" && break
  done
fi
[[ -z "$PY" ]] && { echo "!! No Python found. Set VENV_PYTHON=..."; exit 1; }

# Auto-detect SO-101 port in WSL (usbipd → /dev/ttyACM*; COM → /dev/ttyS*).
if [[ -z "${ROBOT_PORT:-}" && "${ROBOT_MOCK:-0}" != "1" ]]; then
  for _p in /dev/ttyACM{0,1,2} /dev/ttyUSB{0,1,2} /dev/ttyS{2,3,4,5,1,6,7,0}; do
    [[ -e "$_p" ]] || continue
    _r=$(python3 - "$_p" <<'PYEOF' 2>&1
import serial, sys
try:
    s = serial.Serial(sys.argv[1], 1000000, timeout=0.1); s.close(); print("ok")
except PermissionError: print("perm")
except: print("no")
PYEOF
    )
    [[ "$_r" == "ok" ]] && export ROBOT_PORT="$_p" && echo "Auto-detected SO-101 on $ROBOT_PORT" && break
  done
  if [[ -z "${ROBOT_PORT:-}" ]]; then
    echo "!! No SO-101 port found — set ROBOT_PORT or use ROBOT_MOCK=1 for simulation"
    exit 1
  fi
fi

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

DEMO_PORT="${DEMO_PORT:-8888}"

echo ">>> LiveKit gaze agent"
echo "    python  : $PY"
echo "    mock    : ${ROBOT_MOCK:-0}"
[[ -n "${ROBOT_PORT:-}" ]] && echo "    port    : $ROBOT_PORT"
echo "    approach: ${GAZE_APPROACH:-angled}"
echo "    rerun   : ${ROBOT_RERUN:-1}"
echo ""
echo "    Demo UI : http://localhost:$DEMO_PORT"
echo ""

# Start the demo web server in the background (serves demo.html + mints tokens)
"$PY" "$REPO_DIR/demo_server.py" --port "$DEMO_PORT" &
DEMO_PID=$!
trap "kill $DEMO_PID 2>/dev/null" EXIT

exec "$PY" -m agents.livekit_gaze_agent "$@"
