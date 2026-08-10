"""Arm manipulation skills, shared across arm platforms (ALOHA, SO-101).

Public surface:
    ArmInterface / Observation     hardware seam (stereo + joints + IK in/out)
    Kinematics / PlacoKinematics / CartesianKinematics   FK/IK seam + backends
    IkStrategy / PitchHoldIK / PoseIK   target point + tool angle -> joint angles
    MotionLimits / quintic_waypoints    smooth transit trajectories
    analyze_workspace / WorkspaceMap    derive an arm's reach, pitch table and
                                        IK seeds from its own kinematics
    GazeEngine / GazeConfig        gaze-first locate -> approach -> grasp / place
    GraspConfig                    current-sensed grasp tuning
    MockArm                        synthetic stereo arm for the dev harness

Which IK strategy an arm uses is declared by its profile, not chosen here — see
robots/profiles/ and docs/adding_an_arm.md.
"""

from __future__ import annotations

from manipulation.arms.arm_interface import ArmInterface, Observation
from manipulation.arms.gaze_engine import GazeConfig, GazeEngine
from manipulation.arms.grasp import GraspConfig
from manipulation.arms.ik_strategy import IkStrategy, PitchHoldIK, PoseIK, make_ik
from manipulation.arms.kinematics import CartesianKinematics, Kinematics, PlacoKinematics
from manipulation.arms.motion import MotionLimits, quintic_waypoints, rate_limit
from manipulation.arms.workspace import WorkspaceMap, analyze_workspace

__all__ = [
    "ArmInterface",
    "Observation",
    "Kinematics",
    "PlacoKinematics",
    "CartesianKinematics",
    "IkStrategy",
    "PitchHoldIK",
    "PoseIK",
    "make_ik",
    "MotionLimits",
    "quintic_waypoints",
    "rate_limit",
    "GazeEngine",
    "GazeConfig",
    "GraspConfig",
    "WorkspaceMap",
    "analyze_workspace",
]
