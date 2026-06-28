"""Booster K1 humanoid driver.

Contents:
    robot_client.py          Thin client: ROS2 camera/depth + mic streaming,
                             SDK motion execution, dance choreography, audio
                             playback. Talks to exchange/server.py.
    gemini_robot_control.py  All-in-one on-robot mode: runs models + agent +
                             control locally (simpler setup, slower).
"""
