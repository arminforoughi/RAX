"""The X250 profile — the arm described as data, so the shared stack can draw and reason
about it.

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR. The X250's pick is driven by
`visual_servo` through `X250ServoArm`, which deliberately needs NO kinematics, no
intrinsics and no hand-eye transform -- that is the whole portability argument, and
nothing in this file is required for a pick to work. This profile exists so the arm can
be DESCRIBED: so `/urdf` and `/geom` can draw it in 3D, so its joints have names and
travel the UI can show, and so an IK-based path is available later without a second
description of the same robot.

Which means: if a number here is wrong, the 3D view is wrong and the pick is unaffected.
That is the correct blast radius for a file full of nominal link lengths.

UNITS. The X250's control path speaks lerobot NORMALISED units end to end, and keeps
doing so. Every degree figure below was converted from this arm's own recorded
normalised values by `x250/normalise.py` -- scale from the arm's calibration, offsets
ASSUMED. Read that module's docstring before trusting an absolute angle.

JOINT ORDER is `x250_driver.MOTOR_IDS` order, which is `poses.json` order and the order
every X250 pose in the repo is written in::

    0 base   1 shoulder_2   2 elbow   3 wrist   4 tool          (+ gripper, off-chain)

THE TOPOLOGY IS THE SO-101's. base pans; shoulder_2, elbow and wrist share a parallel
axis so their angles SUM to the hand's world pitch; tool rolls. That is the same shape
as the SO-101 (pan + 3-long parallel pitch chain + roll), so `ik="pitch_hold"` applies
unchanged -- the slaved-wrist trick that keeps the SO-101's held pitch from drifting is
not SO-101-specific, it is a property of three parallel joints.
"""

from __future__ import annotations

import os

from rax.robots.profiles import ArmProfile, BusProfile, CameraProfile, GripperProfile
from rax.robots.x250_units import load_units

JOINT_NAMES = ("base", "shoulder_2", "elbow", "wrist", "tool")

_U = load_units()


def _deg(pose: dict) -> tuple[float, ...]:
    """A normalised X250 pose (as recorded) -> degrees, in JOINT_NAMES order."""
    return tuple(_U.to_deg(j, pose.get(j, 0.0)) for j in JOINT_NAMES)


# --- the recorded poses, in the normalised units they were demonstrated in ----------
# Straight out of examples/vla/pick/config/poses.json, which is the median over 113
# teleoperated demonstrations. Copied rather than imported: that file is a SNAPSHOT of
# one rig's calibration living under examples/, and src/ must not read out of examples/.
# If the demonstrations are re-recorded, re-copy.
LOOK_NORM = {"base": -2.6, "shoulder_2": -55.0, "elbow": -58.0, "wrist": -6.0,
             "tool": 41.5, "gripper": 54.0}
GRASP_NORM = {"base": -4.371, "shoulder_2": 29.390, "elbow": 30.832, "wrist": -17.118,
              "tool": 38.923, "gripper": 39.130}
LOOK_FAR_NORM = {"base": -2.613, "shoulder_2": -36.084, "elbow": -43.393,
                 "wrist": -12.821, "tool": 41.531, "gripper": 53.544}

#: The joint box the 113 demonstrations visited, in normalised units
#: (examples/vla/pick/config/safe_envelope.json). THIS is what bounds motion, not the
#: URDF's <limit> — the URDF carries the joint's physical travel, which is much wider,
#: and two of its six entries are the Dynamixel default rather than a measurement.
ENVELOPE_NORM = {
    "base": (-40.611, 11.697),
    "shoulder_2": (-100.0, 37.001),
    "elbow": (-100.0, 41.436),
    "wrist": (-27.814, 55.458),
    "tool": (10.942, 77.507),
    "gripper": (30.2, 76.8),
}

HOME_DEG = _deg(LOOK_NORM)
VIEW_DEG = _deg(LOOK_FAR_NORM)

GRIPPER = GripperProfile(
    joint_name="gripper",
    # All in the gripper's own normalised 0..100, which is what the driver commands.
    # MEASURED across the 113 demonstrations, quoted by pick.py: a held tube reads
    # above 31.9 and closing on nothing settles at 30.2, with NO OVERLAP between the
    # two populations. That separation is why a failed grasp on this arm is detected
    # rather than assumed, and it is why these are not round numbers.
    open_pct=54.0,              # the demonstrated `look` opening
    closed_pct=30.2,            # closed on air
    place_open_pct=50.0,        # pick.py verifies a release by gripper > 45
    close_step_pct=4.0,
    close_delay_s=0.06,
    # The X250 verdict is a POSITION reading, not a current rise: the jaws are
    # back-drivable enough that where they come to rest tells you whether something is
    # between them. So the current threshold the SO-101 uses does not apply, and is left
    # at its default unused rather than transcribed as if it had been measured here.
    squeeze_extra_pct=6.0,
    relax_on_miss_pct=54.0,
    # NOT MEASURED on this arm. The URDF puts gripper_frame_link between the jaws by
    # construction, so zero is the honest entry until someone measures the real offset.
    tip_offset_m=0.0,
    grasp_roll_deg=0.0,
    # The wrist camera's own measured grasp pixel, from
    # examples/vla/pick/config/gripper_geometry.json — the one number in this file that
    # was measured on THIS arm's camera, over a whole session, varying +-1.8px.
    hand_uv=(332.0, 407.0),
)

CAMERA = CameraProfile(
    kind="mono",
    mount="eye_in_hand",
    # DELIBERATELY UNCALIBRATED. There is no hand-eye transform for this arm and the
    # pick does not want one: `visual_servo` measures what each joint does to the
    # picture instead (jacobian.py). An extrinsic string invented here would be a
    # number that looks measured and is not — exactly the failure mode that cost this
    # rig two days when the SO-101's hand-eye TF was out by 370px.
    extrinsics="0,0,0,0,0,0",
    calibration_file=None,
    width=640,
    height=480,
    fps=30,
    use_depth=False,
)

BUS = BusProfile(
    protocol="dynamixel",
    baud=1_000_000,
    # MEASURED by broadcast ping: ids 2-5 are model 1020 (XM430-W350), 6-7 are model
    # 1060 (XL430-W250). Id 1 does not answer.
    motor_ids=(2, 3, 4, 5, 6, 7),
    # Protocol 2.0 control table, not Feetech's: torque enable 64, hardware error 70.
    torque_register=64,
    status_register=70,
)

PROFILE = ArmProfile(
    name="x250",
    urdf="robots/x250_model/x250.urdf",
    ee_frame="gripper_frame_link",
    joint_names=JOINT_NAMES,
    # A serial port is a fact about the machine, not the arm — same reasoning as the
    # SO-101 profile. This arm was on COM5 when it last enumerated.
    port=os.environ.get("RAX_X250_PORT", ""),

    ik="pitch_hold",
    pan_joint=0,
    pitch_chain=(1, 2, 3),      # shoulder_2, elbow, wrist share a parallel axis
    roll_joint=4,               # tool
    ik_seeds=(),                # no elbow-flip dead band characterised on this arm

    home_deg=HOME_DEG,
    view_deg=VIEW_DEG,

    # Deliberately slow. Nothing about this arm's dynamics has been characterised here,
    # and the driver already clips every command to max_relative_target=8.0 normalised
    # units, so these only shape transit moves.
    joint_rate_max_dps=20.0,
    table_z_m=0.0,

    # PLAUSIBILITY BOUNDS ONLY, derived from the nominal URDF (tip radius 0.38 m at the
    # zero pose). They are NOT the measured reach this arm has — the SO-101's
    # reach_grasp_max_m came out of a 2 cm workspace sweep, and nothing equivalent has
    # been run here. Anything that decides reachability should solve IK, not read this.
    reach_min_m=0.06,
    reach_max_m=0.50,
    reach_grasp_max_m=0.42,

    # The demonstrated envelope, converted to degrees, IN PLACE OF the URDF's limits.
    # Passing it explicitly is what stops `resolve()` reading the much wider physical
    # travel out of the URDF — two of whose six entries are the Dynamixel default.
    limits_deg=None,            # filled below; see ENVELOPE_DEG

    gripper=GRIPPER,
    camera=CAMERA,
    bus=BUS,
)

# --- limits ------------------------------------------------------------------------
# Built after the fact because it needs the converter, and assigned through `replace`
# so PROFILE stays a frozen dataclass.
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

ENVELOPE_DEG = (
    np.array([_U.to_deg(j, ENVELOPE_NORM[j][0]) for j in JOINT_NAMES], dtype=np.float64),
    np.array([_U.to_deg(j, ENVELOPE_NORM[j][1]) for j in JOINT_NAMES], dtype=np.float64),
)
PROFILE = replace(PROFILE, limits_deg=ENVELOPE_DEG)
