"""LeRobot SO-101 arm driver.

``driver.So101Arm`` implements :class:`manipulation.arms.ArmInterface` on top of
lerobot (``make_robot_from_config``) + an OAK-D in ``export_stereo_rectified``
mode, so the gaze engine's locate -> approach -> grasp / place loop runs on real
hardware. Imported lazily (it pulls in lerobot + the robot SDK); the perception
stack and the mock harness do not need it.
"""
