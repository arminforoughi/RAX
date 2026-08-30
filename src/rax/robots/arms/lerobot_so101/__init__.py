"""LeRobot SO-101 — the driver and its CLI, together.

``driver.So101Arm`` implements :class:`rax.manipulation.arms.ArmInterface` on top of
lerobot (``make_robot_from_config``) plus an OAK-D in ``export_stereo_rectified`` mode,
so the gaze engine's locate -> approach -> grasp / place loop runs on real hardware.

``entrypoint.main`` is the SO-101-specific CLI, kept as a drop-in replacement for the
``lerobot-gaze-engine`` console script — it mirrors that script's argument names so
existing shell scripts keep working::

    python -m rax.robots.arms.lerobot_so101 [flags]

Both live under ``robots/`` rather than ``manipulation/`` because both are about ONE
vendor's arm. ``manipulation/`` is the part that must not know which robot it is
driving; a vendor CLI sitting there was a seam violation even though nothing imported
across it.

Everything here is imported lazily — it pulls in lerobot and the robot SDK, and the
perception stack, the mock harness and ``python -m rax.grasp --arm mock`` must all run
without them.
"""

from rax.robots.arms.lerobot_so101.entrypoint import main

__all__ = ["main"]
