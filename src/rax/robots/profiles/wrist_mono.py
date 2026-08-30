"""A webcam taped to the gripper: the last rig combination, and the hardest one.

Mount and sensor are independent choices, so there are six combinations of
``{eye_in_hand, fixed} x {stereo, rgbd, mono}``. Five of them have always worked. This
one — a camera on the *hand* with no depth of any kind — did not, and the failure was
silent: :class:`~rax.manipulation.arms.gaze_engine.GazeEngine` took range from the
point cloud, a mono rig produces no cloud, and the approach ran forever on a hardcoded
0.25 m placeholder while reporting that it was working.

The fix is in the engine, not here: it falls back to apparent-size ranging
(``z = fx * W / w_px``) when the rig has no depth source. This profile exists so that
fix is exercised on every commit rather than believed.

Frame convention differs from :mod:`~rax.robots.profiles.head_mono` and is worth
stating, because getting it wrong silently inverts the servo: this profile is built on
:class:`~rax.manipulation.arms.kinematics.CartesianKinematics`, whose six "joints" ARE
the end-effector pose, and whose base frame follows the OpenCV camera convention
(X right, Y down, Z forward). World-up is therefore ``(0, -1, 0)``, not ``(0, 0, 1)``.
"""

from __future__ import annotations

import numpy as np

from rax.robots.profiles import ArmProfile, CameraProfile, GripperProfile

#: EE pose components, in the order CartesianKinematics expects.
JOINT_NAMES = ("x", "y", "z", "rx", "ry", "rz")

LIMITS = (
    np.array([-1.0, -1.0, -1.0, -180.0, -180.0, -180.0]),
    np.array([+1.0, +1.0, +1.0, +180.0, +180.0, +180.0]),
)

#: The camera IS the end effector: T_ee_cam is the identity. A real wrist mount has a
#: few centimetres of offset and a downward pitch; fit it with
#: :mod:`rax.perception.handeye` rather than measuring it with a ruler.
WRIST_TF = "0,0,0,0,0,0"

#: A 640x480 webcam at roughly 62 degrees horizontal FOV.
WEBCAM_INTRINSICS = (525.0, 525.0, 320.0, 240.0)

PROFILE = ArmProfile(
    name="wrist_mono",
    urdf="",
    ee_frame="ee",
    joint_names=JOINT_NAMES,
    port="",

    ik="pose",
    pan_joint=None,
    pitch_chain=(),
    roll_joint=None,
    ik_seeds=(),

    # Backed off along -Z so the scene at z ~ 0.4 m starts in view and in front.
    home_deg=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    view_deg=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    survey_tilt_deg=None,

    joint_rate_max_dps=25.0,
    table_z_m=0.0,
    reach_min_m=0.05,
    reach_max_m=0.80,

    gripper=GripperProfile(
        open_pct=95.0,
        closed_pct=2.0,
        place_open_pct=60.0,
        contact_current_delta=8.0,
        tip_offset_m=0.0,
        # An eye-in-hand rig DOES have a constant fingertip pixel, but it has to be
        # measured on the real mount — the predicted-vs-measured gap is the live
        # read-out of hand-eye error, and inventing a value here would zero it out.
        hand_uv=None,
    ),
    camera=CameraProfile(
        kind="mono",               # no depth of any kind
        mount="eye_in_hand",       # on the hand, so the view moves with every command
        extrinsics=WRIST_TF,
        calibration_file=None,
        width=640,
        height=480,
        fps=30,
        use_depth=False,
        intrinsics_fallback=WEBCAM_INTRINSICS,
    ),
    bus=None,

    limits_deg=LIMITS,
)
