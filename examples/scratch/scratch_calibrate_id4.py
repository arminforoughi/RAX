"""Recalibrate wrist_flex (motor id 4) by driving it itself, stall-detected.

Unlike wrist_roll (id 5, a near-full-turn joint whose calibration keeps the
established 0..4095 convention), wrist_flex is a normal bounded pitch joint --
its calibration is SUPPOSED to carry a real measured range_min/range_max (this
is exactly what so_follower.calibrate()'s record_ranges_of_motion does for it
normally, just by hand instead of driven). So this script writes BOTH a fresh
homing_offset AND a fresh range_min/range_max from the sweep.

wrist_flex carries real load (gripper + camera), so it holds position under
torque between steps rather than drooping -- that's what makes a driven sweep
safe here the same way it was for wrist_roll. Leaves the other five motors'
calibration entries untouched.
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

STEP = 12          # raw ticks per commanded step (~1.1 deg) -- a bit gentler than id5's, it's load-bearing
SETTLE_S = 0.3
MIN_MOVE = 4        # actual physical movement must be at least this since the LAST reading
STALL_STREAK = 4    # consecutive no-movement steps before calling it a limit
BACKOFF = 20        # ticks to retreat off the stop once found, to relieve load
MAX_TRAVEL = 1200   # safety cap from start in either direction; old range was ~1817 ticks
                     # total (~900 from its old center), this gives margin either side
# Old file had range 1024..2841 -- clamp comfortably wider than that in case the
# true range shifted a bit, but still bounded against a runaway.
CLAMP_LO, CLAMP_HI = 700, 3200


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
        bus.write("Goal_Position", "wrist_flex", goal, normalize=False)
        time.sleep(SETTLE_S)
        present = bus.read("Present_Position", "wrist_flex", normalize=False)
        moved = abs(present - prev_present)
        print(f"  goal={goal} present={present} moved={moved}")
        if moved < MIN_MOVE:
            stall_streak += 1
            if stall_streak >= STALL_STREAK:
                print(f"  stalled at present={present} (goal={goal}) -- treating as limit")
                backoff_goal = present - direction * BACKOFF
                bus.write("Goal_Position", "wrist_flex", backoff_goal, normalize=False)
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

bus.write("Operating_Mode", "wrist_flex", OperatingMode.POSITION.value)
bus.enable_torque("wrist_flex")

start = bus.read("Present_Position", "wrist_flex", normalize=False)
print(f"start position: {start}")

print("sweeping +...")
max_pos = sweep(bus, start, +1)
print("returning to start...")
bus.write("Goal_Position", "wrist_flex", start, normalize=False)
time.sleep(0.5)

print("sweeping -...")
min_pos = sweep(bus, start, -1)

if min_pos > max_pos:
    min_pos, max_pos = max_pos, min_pos
span_deg = (max_pos - min_pos) * 360 / 4096
print(f"discovered range: {min_pos} .. {max_pos}  ({max_pos - min_pos} ticks, {span_deg:.1f} deg)")

mid = (min_pos + max_pos) // 2
print(f"centering at midpoint {mid}...")
bus.write("Goal_Position", "wrist_flex", mid, normalize=False)
time.sleep(0.5)

homing = bus.set_half_turn_homings(motors=["wrist_flex"])
new_offset = homing["wrist_flex"]
print(f"wrist_flex homing_offset -> {new_offset}")

backup = CALIB_PATH.with_suffix(f".pre_id4_recal_{int(time.time())}.json")
shutil.copy(CALIB_PATH, backup)
print(f"backed up existing calibration to {backup}")

data = json.loads(CALIB_PATH.read_text())
data["wrist_flex"] = {
    "id": 4,
    "drive_mode": 0,
    "homing_offset": new_offset,
    "range_min": min_pos,
    "range_max": max_pos,
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
print(f"FYI measured physical range was {span_deg:.1f} deg (old file range was ~159.7 deg)")
