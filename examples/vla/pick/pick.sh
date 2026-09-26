#!/bin/bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib
cd ~/lerobot_trossen
exec uv run --no-sync python3 ~/smolvla_runs/pick.py "$@"
