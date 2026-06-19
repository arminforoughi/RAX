"""Current-sensed grasp + release, on top of the :class:`ArmInterface` seam.

Ported and condensed from lerobot ``grasp_close.py``, but routed through the arm
interface (``set_gripper`` / ``read_gripper_current``) instead of poking the motor
bus directly — so the same logic drives the mock arm and a real SO-101.

``close_with_current`` inches the gripper shut, watching the motor current rise;
contact is when the current jumps past ``contact_delta_counts``. ``release`` just
opens. Cartesian motion (inch-forward, lift, descend) is handled by the gaze
engine, which owns IK — this module only owns the gripper.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from manipulation.arms.arm_interface import ArmInterface

logger = logging.getLogger(__name__)


@dataclass
class GraspConfig:
    open_pct: float = 100.0
    close_pct: float = 0.0
    close_step_pct: float = 6.0
    step_wait_s: float = 0.05
    contact_delta_counts: float = 40.0
    idle_sample_s: float = 0.2
    hold_s: float = 0.2


@dataclass
class GraspResult:
    contact: bool
    reason: str
    grip_pct_at_contact: float | None
    mean_hold_delta: float


def _sample_current(arm: ArmInterface, seconds: float) -> float:
    end = time.time() + max(0.0, seconds)
    samples: list[float] = []
    while time.time() < end:
        v = arm.read_gripper_current()
        if v is not None:
            samples.append(float(v))
        time.sleep(0.02)
    return float(np.mean(samples)) if samples else 0.0


def close_with_current(arm: ArmInterface, cfg: GraspConfig) -> GraspResult:
    """Step the gripper closed until the motor current spikes (object contact)."""
    i_idle = _sample_current(arm, cfg.idle_sample_s)
    pct = float(cfg.open_pct)
    target = float(cfg.close_pct)
    step = float(cfg.close_step_pct)

    contact, reason, grip_at_contact = False, "fully_closed", None
    while pct > target:
        pct = max(target, pct - step)
        arm.set_gripper(pct)
        time.sleep(max(0.0, cfg.step_wait_s))
        cur = arm.read_gripper_current()
        if cur is not None and abs(float(cur) - i_idle) >= cfg.contact_delta_counts:
            contact, reason, grip_at_contact = True, f"delta_current={cur - i_idle:.0f}", pct
            break

    mean_hold = abs(_sample_current(arm, cfg.hold_s) - i_idle)
    logger.info("[grasp] %s (grip=%.0f%%, hold_delta=%.0f)", reason, pct, mean_hold)
    return GraspResult(contact, reason, grip_at_contact, mean_hold)


def release(arm: ArmInterface, cfg: GraspConfig) -> None:
    """Open the gripper to drop the held object."""
    arm.set_gripper(cfg.open_pct)
    time.sleep(max(0.0, cfg.step_wait_s))
