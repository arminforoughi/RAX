#!/usr/bin/env bash
# Start everything: LiveKit agent + demo UI server
# Usage: bash start.sh
cd "$(dirname "$0")"

PYTHON=/home/labot/.venv/lerobot/bin/python

# --- Ensure the SO-101 arm (CH343 USB serial) is attached to WSL2 ------------
# WSL2 forgets USB attachments across reboot/replug. If /dev/ttyLEROBOT is
# missing, forward the CH343 adapter (VID:PID 1a86:55d3) in from Windows.
# The udev rule 90-lerobot.rules then recreates the /dev/ttyLEROBOT symlink.
if [ ! -e /dev/ttyLEROBOT ]; then
  echo "Arm not attached — forwarding CH343 (busid 5-3) into WSL via usbipd..."
  BUSID=$(powershell.exe -NoProfile -Command \
    "(usbipd list | Select-String '1a86:55d3').ToString().Split(' ')[0]" 2>/dev/null | tr -d '\r')
  BUSID=${BUSID:-5-3}
  powershell.exe -NoProfile -Command "usbipd attach --wsl --busid $BUSID" 2>&1 | sed 's/^/  [usbipd] /'
  # give udev a moment to create the symlink
  for _ in 1 2 3 4 5; do [ -e /dev/ttyLEROBOT ] && break; sleep 0.5; done
  if [ -e /dev/ttyLEROBOT ]; then
    echo "  Arm attached: /dev/ttyLEROBOT -> $(readlink -f /dev/ttyLEROBOT)"
  else
    echo "  WARNING: /dev/ttyLEROBOT still missing — arm connect will fail."
    echo "           Check 'usbipd list' on Windows; adapter may be unplugged."
  fi
fi

echo "Starting LiveKit agent worker..."
PYTHONPATH=. $PYTHON agents/livekit_gaze_agent.py start &
AGENT_PID=$!

echo "Starting demo UI on http://localhost:8888 ..."
PYTHONPATH=. $PYTHON demo_server.py &
SERVER_PID=$!

echo ""
echo "  Open http://localhost:8888 in your browser"
echo "  Press Ctrl+C to stop everything"
echo ""

trap "kill $AGENT_PID $SERVER_PID 2>/dev/null" EXIT INT TERM
wait
