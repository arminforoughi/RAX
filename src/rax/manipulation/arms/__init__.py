"""Arm manipulation skills, shared across arm platforms (ALOHA, SO-101).

Public surface:
    ArmInterface / Observation     arm seam (joints + gripper in/out)
    ArmState / CameralessArm       an arm with no camera of its own
    Rig / geometry_for             join any arm to any camera (wrist or head mount)
    Kinematics seam + backends: make_kinematics() picks placo or the pure-numpy
    UrdfKinematics; CartesianKinematics needs no URDF at all
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

from rax.manipulation.arms.arm_interface import (
    ArmInterface,
    ArmState,
    CameralessArm,
    Observation,
)
from rax.manipulation.arms.gaze_engine import GazeConfig, GazeEngine
from rax.manipulation.arms.grasp import GraspConfig
from rax.manipulation.arms.ik_strategy import IkStrategy, PitchHoldIK, PoseIK, make_ik
from rax.manipulation.arms.kinematics import (
    CartesianKinematics,
    Kinematics,
    PlacoKinematics,
    make_kinematics,
)
from rax.manipulation.arms.motion import MotionLimits, quintic_waypoints, rate_limit
from rax.manipulation.arms.rig import Rig, geometry_for
from rax.manipulation.arms.urdf_kinematics import UrdfKinematics
from rax.manipulation.arms.workspace import WorkspaceMap, analyze_workspace

__all__ = [
    "ArmInterface",
    "ArmState",
    "CameralessArm",
    "Observation",
    "Rig",
    "geometry_for",
    "Kinematics",
    "PlacoKinematics",
    "UrdfKinematics",
    "make_kinematics",
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
