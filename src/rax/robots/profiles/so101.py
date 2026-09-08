"""The SO-101 + OAK-D profile.

Every value here was a module-level constant in ``stack_mission2.py``, and most were
measured rather than chosen — the comments say which, and where the measurement came
from. Do not round or "tidy" them.

Joint order is the lerobot ``ARM_MOTORS`` order, which is also the order the URDF's
limits and every ``q[i]`` in the stack are indexed by::

    0 shoulder_pan   1 shoulder_lift   2 elbow_flex   3 wrist_flex   4 wrist_roll
"""

from __future__ import annotations

import os

from rax.robots.profiles import ArmProfile, BusProfile, CameraProfile, GripperProfile

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Captured from the physically-correct folded pose (2026-07-20).
# Recaptured 2026-08-13 after motor id 5 (wrist_roll) was physically swapped and
# recalibrated -- the recalibration changed wrist_roll's homing offset, so the old
# degree value no longer pointed at the same physical angle. Read live from the
# resting arm via /status rather than re-measured by hand.
# RESTORED to the original 2026-07-20 values on 2026-08-16, and that is deliberate.
#
# Motors id 4 (wrist_flex) and id 5 (wrist_roll) were physically REPLACED. A
# replacement servo has its own arbitrary encoder zero, so the intermediate
# HOME_DEG values captured on 2026-08-13/14 were readings taken in a BROKEN frame:
# `set_half_turn_homings` had redefined wrist_flex's zero to wherever the joint
# happened to be resting, shifting it 175.3 deg away from the URDF convention.
# Chasing that by eye (three successive "nudge the wrist" captures) moved the arm
# without ever fixing the frame.
#
# The fix was to calibrate the new servo against a PHYSICAL reference instead:
# this arm's gripper pitch is exactly shoulder_lift + elbow_flex + wrist_flex, so
# holding the gripper level and solving wrist_flex = -(lift + elbow) pins the zero
# to the URDF's own convention. Verified: model pitch reads 0.1 deg at physically
# level, travel window now the URDF's full +-95 deg (was clipped to +-79.9).
#
# Because wrist_flex once again REPORTS IN THE URDF CONVENTION, the originally
# tuned pose below is valid again -- it is an angle in that convention, not a raw
# encoder reading, so it survived the motor swap.
#
# CAVEAT: wrist_roll (id 5) was recentered the same arbitrary way on 2026-08-12 and
# has NOT yet been re-fitted to a physical reference, so the -4.7 here is the one
# component still open. See [[rax-known-defects]].
HOME_DEG = (-14.1, -99.1, 90.8, 33.2, -4.7)

# The tilt the SURVEY localizes from -- deliberately NOT HOME_DEG[1:] any more.
# It used to be derived from HOME, which coupled a cosmetic "how does the idle pose
# look" preference to the geometry every range estimate depends on: range error comes
# from how steeply the sightline meets the table, and the footprint measurement is
# ALREADY marginal at the current angle (contact pixels land where the sightline
# barely grazes the plane). On 2026-08-16 HOME's wrist pitch was raised 20 deg for
# viewing; these stay at HOME's ORIGINAL 2026-07-20 tilt, the one the 0.4 cm
# survey repeatability was measured against. Keep the leading value equal to
# HOME's shoulder_lift or `survey_pose_for` starts from a different arm shape.
# 2026-08-26: wrist pitch 33.2 -> 68.2. At 33.2 the camera sat 3 deg ABOVE horizontal
# and a full pan sweep covered 40% of the reachable table -- nothing inside 30cm was
# ever in frame, so near objects were localized from whatever grazing view clipped the
# frame edge, and the range error that comes back is amplified ~8x at that incidence.
# 68.2 puts the camera 27 deg down and covers 100% at every radius from 18 to 42cm.
# Only id4 moves: it is the one joint that re-aims the camera without changing the
# arm's shape, so reach, IK seeds and the approach trims are all untouched.
# Live-tunable at runtime as the `survey_pitch_deg` knob.
SURVEY_TILT_DEG = (-99.1, 90.8, 68.2, -4.7)
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
    urdf="robots/arms/lerobot_so101/SO101/so101_new_calib.urdf",
    # The same directory: it holds assets/*.stl and a robot.urdf (a copy of the calib
    # URDF, kept because lerobot's mesh loader looks for that exact filename).
    # NOTE the directory holds the same URDF twice, under two names, and both are
    # load-bearing: `urdf` names the file the kinematics read, while the Rerun mesh
    # viewer calls robot_urdf_file_in_dir(), which looks for a file literally called
    # robot.urdf. Deleting it as a duplicate costs you the 3D view, and the error
    # ("No robot.urdf in directory") arrives far from here.
    mesh_dir="robots/arms/lerobot_so101/SO101",
    ee_frame="gripper_frame_link",
    joint_names=JOINT_NAMES,
    # Empty by default. A serial port is a fact about the machine the robot is
    # plugged into, not about the SO-101, and a committed "COM4" is wrong for every
    # user but one — quietly, by connecting to whatever else happens to be on that
    # port. Set RAX_ARM_PORT, or pass --port.
    port=os.environ.get("RAX_ARM_PORT", ""),

    # The pitch joints (lift, elbow, wrist_flex) share a parallel axis, so the
    # gripper's world pitch is EXACTLY their sum, independent of pan and roll
    # (measured; the constant offset is 0). That is what lets wrist_flex be slaved
    # algebraically instead of solved, so the held pitch never drifts.
    ik="pitch_hold",
    pan_joint=0,
    pitch_chain=(1, 2, 3),
    roll_joint=4,
    # Re-seed for the elbow-flip dead band: where the solver's natural branch cannot
    # reach a pose that IS reachable, escaping it needs a seed from the other branch.
    # None = keep the caller's value for that joint.
    #
    # DERIVED, not hand-picked. manipulation.arms.workspace probes the (radius, height,
    # pitch) space with the bare solver, records which sampled seeds rescue which cells,
    # and takes a minimal cover. This single seed is the mirrored elbow configuration —
    # exactly what an elbow flip needs — and it replaced five seeds that had accumulated
    # one at a time. Measured over 700 poses spanning all five working heights: identical
    # coverage (350/700 either way, no pose reachable by one set and not the other) and
    # 2.6x faster, because four of the five were redundant.
    #
    # Re-derive rather than adding to it:  analyze_workspace(ik, profile).seeds()
    ik_seeds=(
        (None, 70.0, -67.8, None, None),
    ),

    home_deg=HOME_DEG,
    view_deg=VIEW_DEG,
    survey_tilt_deg=SURVEY_TILT_DEG,  # lift, elbow, wrist_flex, wrist_roll — always these

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
    # Nothing that matters is outside the arm's own workspace, so a "cube" localized
    # at 92 cm is a broken solve, not a distant object. This is a PLAUSIBILITY bound,
    # deliberately looser than what the arm can grasp.
    reach_min_m=0.08,
    reach_max_m=0.55,
    # MEASURED 2026-09-06, IK-verified, not guessed. Two independent probes agree:
    # workspace.analyze_workspace on a 2 cm grid, and a 5 mm scan against the bare
    # solver at 3 mm tolerance. Max fingertip radius at table height (z=2 cm):
    #
    #     pitch   0-15 deg  ->  47-48 cm     <- a flat wrist reaches furthest
    #     pitch  30    deg  ->  44 cm
    #     pitch  45    deg  ->  42 cm
    #     pitch  60    deg  ->  38 cm
    #     pitch  75    deg  ->  34 cm        <- the first grasp pitch tried
    #     pitch  90    deg  ->  30 cm
    #
    # So "how far can it reach" has no single answer: it costs ~17 cm to go from a
    # flat wrist to a vertical one. 0.47 is the best case, used as the hard ceiling;
    # plan_pitch discovers the pitch-specific limit per target by solving for it.
    reach_grasp_max_m=0.47,

    gripper=GRIPPER,
    camera=CAMERA,
    bus=BUS,
)
