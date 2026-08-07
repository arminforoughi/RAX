"""A synthetic arm profile that exercises the OTHER branch of every seam.

The SO-101 is a parallel-pitch arm with a wrist camera, so running only that leaves
the generic-6DOF IK and the fixed-camera geometry as code nobody executes until the
day someone plugs in a different robot — which is the worst moment to find out.

This profile is the opposite choice at every seam:

    SO-101                          mock
    ------------------------------  ------------------------------
    ik = "pitch_hold"               ik = "pose"
    camera mount = eye_in_hand      camera mount = fixed
    5 revolute joints, real URDF    6 "joints" that ARE the EE pose
    lerobot + a serial port         no hardware at all

The kinematics are :class:`manipulation.arms.kinematics.CartesianKinematics`, whose
six joints are literally ``(x, y, z, rx, ry, rz)``. That needs no URDF and no placo, so
:func:`robots.profiles.load_profile` cannot read limits from a file — they are declared
here instead, which is also the escape hatch a real arm uses when its URDF omits them.
"""

from __future__ import annotations

import numpy as np

from robots.profiles import ArmProfile, CameraProfile, GripperProfile

#: The EE pose components, in the order CartesianKinematics expects.
JOINT_NAMES = ("x", "y", "z", "rx", "ry", "rz")

#: Metres for the translation components, degrees for the rotation vector. Generous,
#: because this is a simulator: the point is to exercise the code paths, not to model
#: a workspace.
LIMITS = (
    np.array([-1.0, -1.0, -1.0, -180.0, -180.0, -180.0]),
    np.array([+1.0, +1.0, +1.0, +180.0, +180.0, +180.0]),
)

#: 60 cm above the base looking straight down, in the OpenCV optical convention
#: (+Z along the view direction, +Y image down). As a "x,y,z,rx,ry,rz" rotation vector,
#: a pi rotation about X flips Y and Z to point the camera at the table.
OVERHEAD_TF = "0,0,0.60,3.14159265,0,0"

PROFILE = ArmProfile(
    name="mock",
    urdf="",                       # CartesianKinematics needs no robot model
    ee_frame="ee",
    joint_names=JOINT_NAMES,
    port="",

    ik="pose",                     # the branch the SO-101 never takes
    pan_joint=None,
    pitch_chain=(),
    roll_joint=None,
    ik_seeds=(),

    home_deg=(0.20, 0.0, 0.15, 0.0, 0.0, 0.0),
    view_deg=(0.20, 0.0, 0.20, 0.0, 0.0, 0.0),
    survey_tilt_deg=None,

    joint_rate_max_dps=25.0,
    table_z_m=0.0,
    reach_min_m=0.05,
    reach_max_m=0.60,

    gripper=GripperProfile(
        open_pct=95.0,
        closed_pct=2.0,
        place_open_pct=60.0,
        contact_current_delta=8.0,
        tip_offset_m=0.0,
        hand_uv=None,              # a fixed camera has no constant fingertip pixel
    ),
    camera=CameraProfile(
        mount="fixed",             # the branch the SO-101 never takes
        extrinsics=OVERHEAD_TF,
        calibration_file=None,
        width=640,
        height=480,
        use_depth=False,
        intrinsics_fallback=(517.0, 517.0, 320.0, 240.0),
    ),
    bus=None,                      # no servo bus to clear overloads on

    limits_deg=LIMITS,             # declared, since there is no URDF to read
)
