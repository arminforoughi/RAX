"""Recalibrate ONLY wrist_roll (motor id 5) after it was physically changed.

Drives wrist_roll itself (small steps, stall-detected) to discover its true safe
range instead of asking a human to hand-rotate it, then centers on the discovered
midpoint and takes a fresh homing offset there. Leaves the other five motors'
calibration entries untouched.

Note: range_min/range_max are deliberately left at the codebase's established
0..4095 "full turn" convention for wrist_roll (same as so_follower.calibrate()
always writes) -- only homing_offset is updated from the sweep. The discovered
physical extents are printed for the record / sanity-check, not written into
the calibration, because narrowing range_min/max would rescale every existing
wrist_roll normalized-angle command (grasp roll-for-yaw, hand-eye) system-wide,
which is a bigger change than "id 5 got swapped."
"""
import json
import shutil
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

PORT = "COM4"
CALIB_PATH = Path(
    r"C:\Users\labot\.cache\huggingface\lerobot\calibration\robots\so_follower\so101_follower.json"
)

MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

STEP = 15          # raw ticks per commanded step (~1.3 deg)
SETTLE_S = 0.3      # wrist_roll has a soft P-gain (16, deliberately) -- give it time to catch up
MIN_MOVE = 4        # actual physical movement must be at least this since the LAST reading
STALL_STREAK = 4    # consecutive no-movement steps before calling it a limit (not "lag behind goal")
BACKOFF = 25        # ticks to retreat off the stop once found, to relieve load
MAX_TRAVEL = 1900   # safety cap from start in either direction (~167 deg;
                     # URDF limit is ~157/163 deg either side of a centered zero)
CLAMP_LO, CLAMP_HI = 50, 4045  # never command outside this regardless of MAX_TRAVEL


def sweep(bus, start, direction):
    goal = start
    prev_present = start
    stall_streak = 0
    last_good = start
    while abs(goal - start) < MAX_TRAVEL:
        next_goal = goal + direction * STEP
        if next_goal < CLAMP_LO or next_goal > CLAMP_HI:
            print(f"  hit hard clamp bound at goal={next_goal}, stopping")
            return last_good
        goal = next_goal
        bus.write("Goal_Position", "wrist_roll", goal, normalize=False)
        time.sleep(SETTLE_S)
        present = bus.read("Present_Position", "wrist_roll", normalize=False)
        moved = abs(present - prev_present)
        print(f"  goal={goal} present={present} moved={moved}")
        if moved < MIN_MOVE:
            stall_streak += 1
            if stall_streak >= STALL_STREAK:
                print(f"  stalled at present={present} (goal={goal}) -- treating as limit")
                backoff_goal = present - direction * BACKOFF
                bus.write("Goal_Position", "wrist_roll", backoff_goal, normalize=False)
                time.sleep(0.3)
                return present
        else:
            stall_streak = 0
            last_good = present
        prev_present = present
    print("  hit safety travel cap without stalling")
    return last_good


bus = FeetechMotorsBus(port=PORT, motors=MOTORS)
bus.connect()
print("connected to COM4")

bus.write("Operating_Mode", "wrist_roll", OperatingMode.POSITION.value)
bus.enable_torque("wrist_roll")

start = bus.read("Present_Position", "wrist_roll", normalize=False)
print(f"start position: {start}")

print("sweeping +...")
max_pos = sweep(bus, start, +1)
print("returning to start...")
bus.write("Goal_Position", "wrist_roll", start, normalize=False)
time.sleep(0.5)

print("sweeping -...")
min_pos = sweep(bus, start, -1)

if min_pos > max_pos:
    min_pos, max_pos = max_pos, min_pos
span_deg = (max_pos - min_pos) * 360 / 4096
print(f"discovered range: {min_pos} .. {max_pos}  ({max_pos - min_pos} ticks, {span_deg:.1f} deg)")

mid = (min_pos + max_pos) // 2
print(f"centering at midpoint {mid}...")
bus.write("Goal_Position", "wrist_roll", mid, normalize=False)
time.sleep(0.5)

homing = bus.set_half_turn_homings(motors=["wrist_roll"])
new_offset = homing["wrist_roll"]
print(f"wrist_roll homing_offset -> {new_offset}")

backup = CALIB_PATH.with_suffix(f".pre_id5_recal_{int(time.time())}.json")
shutil.copy(CALIB_PATH, backup)
print(f"backed up existing calibration to {backup}")

data = json.loads(CALIB_PATH.read_text())
data["wrist_roll"] = {
    "id": 5,
    "drive_mode": 0,
    "homing_offset": new_offset,
    "range_min": 0,
    "range_max": 4095,
}
CALIB_PATH.write_text(json.dumps(data, indent=4))
print(f"wrote updated calibration to {CALIB_PATH}")

calibration = {
    name: MotorCalibration(**data[name]) for name in MOTORS
}
bus.write_calibration(calibration)
print("pushed full calibration to motors")

bus.disconnect()
print("done - other 5 motors' calibration was left untouched")
print(f"FYI measured physical range was {span_deg:.1f} deg (URDF assumes ~320 deg)")
