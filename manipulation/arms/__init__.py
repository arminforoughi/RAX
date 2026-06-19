"""Arm manipulation skills, shared across arm platforms (ALOHA, SO-101).

Public surface:
    ArmInterface / Observation     hardware seam (stereo + joints + IK in/out)
    Kinematics / PlacoKinematics / CartesianKinematics   FK/IK seam + backends
    GazeEngine / GazeConfig        gaze-first locate -> approach -> grasp / place
    GraspConfig                    current-sensed grasp tuning
    MockArm                        synthetic stereo arm for the dev harness
"""

from __future__ import annotations

from manipulation.arms.arm_interface import ArmInterface, Observation
from manipulation.arms.gaze_engine import GazeConfig, GazeEngine
from manipulation.arms.grasp import GraspConfig
from manipulation.arms.kinematics import CartesianKinematics, Kinematics, PlacoKinematics

__all__ = [
    "ArmInterface",
    "Observation",
    "Kinematics",
    "PlacoKinematics",
    "CartesianKinematics",
    "GazeEngine",
    "GazeConfig",
    "GraspConfig",
]
