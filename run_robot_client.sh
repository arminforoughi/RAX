#!/usr/bin/env bash
# Run robot_client with Booster ROS2 on PYTHONPATH so fight mode can publish LowCmd on joint_ctrl.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -n "${BOOSTER_ROS2_INSTALL:-}" && -f "${BOOSTER_ROS2_INSTALL}/setup.bash" ]]; then
  # shellcheck source=/dev/null
  source "${BOOSTER_ROS2_INSTALL}/setup.bash"
else
  for setup in \
    "/opt/booster/BoosterRos2Interface/install/setup.bash" \
    "/opt/booster/ros2_ws/install/setup.bash" \
    "${HOME}/booster_ws/install/setup.bash" \
    "${HOME}/ros2_ws/install/setup.bash"
  do
    if [[ -f "$setup" ]]; then
      # shellcheck source=/dev/null
      source "$setup"
      break
    fi
  done
fi

PY=python3
if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  PY="${SCRIPT_DIR}/.venv/bin/python"
fi

exec "$PY" "${SCRIPT_DIR}/robot_client.py" "$@"
