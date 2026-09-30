"""Robots: one driver per arm, all implementing :class:`rax.pick.arm.Arm`.

    arm = make_arm("so101", port="COM4")      # or "x250"
    arm.connect()
"""

from rax.robots.base import Stopped, WristCameraArm

ARMS = ("so101", "x250")


def make_arm(name: str, port: str, handeye_file: str | None = None, log=print, **kw):
    """The driver for arm ``name`` (see ``ARMS``). Nothing is opened until ``connect()``."""
    if name == "so101":
        from rax.robots.so101 import So101
        return So101(port, handeye_file, log)
    if name == "x250":
        from rax.robots.x250 import X250
        return X250(port, handeye_file, log, **kw)
    raise ValueError(f"unknown arm {name!r}; try one of {ARMS}")


__all__ = ["make_arm", "ARMS", "WristCameraArm", "Stopped"]
