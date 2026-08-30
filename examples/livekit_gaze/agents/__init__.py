"""Agents — smart models that understand the outside world and command action.

Agents fuse perception (vision, depth, faces) with a reasoning model
(e.g. Gemini Live) to interpret natural-language intent and dispatch
high- and low-level robot commands.

Contents:
    gemini_live_camera.py   Gemini Live voice + vision stream (ROS2 + YOLO).
    livekit_gaze_agent.py   LiveKit + Gemini 3.1 Flash Audio → GazeEngine → SO-101.
                            Gaze-controlled robot: user describes/looks at object,
                            agent picks it up. Run with ./run_livekit_gaze.sh.
"""
