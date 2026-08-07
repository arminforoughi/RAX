"""The SO-101 + OAK-D profile.

Every value here was a module-level constant in ``stack_mission2.py``, and most were
measured rather than chosen — the comments say which, and where the measurement came
from. Do not round or "tidy" them.

Joint order is the lerobot ``ARM_MOTORS`` order, which is also the order the URDF's
limits and every ``q[i]`` in the stack are indexed by::

    0 shoulder_pan   1 shoulder_lift   2 elbow_flex   3 wrist_flex   4 wrist_roll
"""

from __future__ import annotations

from robots.profiles import ArmProfile, BusProfile, CameraProfile, GripperProfile

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Captured from the physically-correct folded pose (2026-07-20).
HOME_DEG = (-14.1, -99.1, 90.8, 33.2, -4.7)
# Wrist twisted 90 deg so the jaws are square to a table cube.
GRASP_ROLL_DEG = 90.0
VIEW_DEG = (5.0, 37.1, 48.1, -40.4, GRASP_ROLL_DEG)

GRIPPER = GripperProfile(
    joint_name="gripper",
    open_pct=95.0,
    closed_pct=2.0,
    place_open_pct=62.0,        # releases without flicking the object
    close_step_pct=5.0,
    close_delay_s=0.05,
    contact_current_delta=8.0,  # 1.8 let the cube slip during transit
    squeeze_extra_pct=14.0,
    relax_on_miss_pct=40.0,
    # MEASURED FROM THE URDF + JAW MESH (2026-07-13), do not guess this:
    # moving_jaw_so101_v1.stl, in gripper_frame_link coords, spans Z -84.7..+7.3 mm.
    # The jaws hinge ~80 mm BEHIND gripper_frame_link and the fingertips reach only
    # +7 mm past it, so gripper_frame_link IS the fingertip / grasp centre and FK
    # already returns the tip. (lookat_engine's 0.10 does NOT transfer to this FK —
    # applying it pushed the "tip" 10 cm out into empty air.)
    tip_offset_m=0.007,
    grasp_roll_deg=GRASP_ROLL_DEG,
    # The URDF's roll zero is rotated from the servo's zero. If the rendered gripper
    # is twisted the OTHER way, flip this sign.
    render_offset_deg=90.0,
    hand_uv=(440.0, 394.0),     # measured via /caltip against the real black fingertip
)

CAMERA = CameraProfile(
    mount="eye_in_hand",
    extrinsics="-0.0503,0.0906,-0.1730,-0.2921,1.0770,-2.1688",
    calibration_file="handeye_tf.json",
    width=640,
    height=480,
    fps=30,
    # The OAK-D's stereo pipeline crashes this camera in the current firmware/driver
    # combination, so the pick stack runs monocular and localizes against the table
    # plane instead. Depth-absent is the normal path here, not a fallback.
    use_depth=False,
    intrinsics_fallback=(517.0, 517.0, 329.5, 231.4),
    # The camera sits ~10 cm BEHIND the fingertips on the real mount. Seeds/bounds the
    # hand-eye fit; it is NOT applied as a correction on its own.
    cam_tip_m=0.10,
)

BUS = BusProfile(
    protocol="feetech",
    baud=1_000_000,
    # All six, not just the gripper: shoulder_lift (id 2) latches after sustained
    # holding, and that is what kills the server at connect.
    motor_ids=(1, 2, 3, 4, 5, 6),
    torque_register=40,
    status_register=56,
)

PROFILE = ArmProfile(
    name="so101",
    urdf="SO101/so101_new_calib.urdf",
    ee_frame="gripper_frame_link",
    joint_names=JOINT_NAMES,
    port="COM4",

    # The pitch joints (lift, elbow, wrist_flex) share a parallel axis, so the
    # gripper's world pitch is EXACTLY their sum, independent of pan and roll
    # (measured; the constant offset is 0). That is what lets wrist_flex be slaved
    # algebraically instead of solved, so the held pitch never drifts.
    ik="pitch_hold",
    pan_joint=0,
    pitch_chain=(1, 2, 3),
    roll_joint=4,
    # Re-seeds for the elbow-flip dead band (mapped 2026-07-13): at r=10-15 cm the arm
    # simply cannot hold a shallow pitch from the caller's seed. None = keep the
    # caller's value for that joint. The last two extend the arm forward for far /
    # shallow-pitch targets.
    ik_seeds=(
        (None, -95.0, 90.0, 30.0, None),
        (None, -30.0, 50.0, 60.0, None),
        (None, -60.0, 20.0, 80.0, None),
        (None, -20.0, 75.0,  0.0, None),
        (None, -10.0, 85.0,  0.0, None),
    ),

    home_deg=HOME_DEG,
    view_deg=VIEW_DEG,
    survey_tilt_deg=HOME_DEG[1:],   # lift, elbow, wrist_flex, wrist_roll — always these

    joint_rate_max_dps=25.0,        # lower = less jerk at stop
    # The base carries the most inertia and causes the visible jump at motion
    # start/stop, so it gets the gentlest limits; the wrist can move faster.
    goto_vmax_dps=(38.0, 55.0, 55.0, 75.0, 90.0),
    goto_amax_dps2=(75.0, 110.0, 110.0, 150.0, 180.0),
    goto_dt_s=0.02,                 # 50 Hz command rate
    # Base-frame height of a table object's CENTRE; the sightline is intersected with
    # this plane to localize. If the grasp stops short/high, raise it a few mm; if it
    # drives into the table, lower it.
    table_z_m=0.02,
    # Nothing that matters is outside the arm's own workspace (~42 cm reach), so a
    # "cube" localized at 92 cm is a broken solve, not a distant object.
    reach_min_m=0.08,
    reach_max_m=0.55,

    gripper=GRIPPER,
    camera=CAMERA,
    bus=BUS,
)
