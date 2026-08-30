"""Hand-guided recalibration of wrist_flex (motor id 4).

Disables torque on wrist_flex ONLY (other 5 motors stay as they are), then
samples Present_Position continuously for RECORD_S seconds while a human moves
the joint through its full range of motion by hand. Re-enables torque at the
end, centers on the midpoint of what it saw, and takes a fresh homing offset
there -- mirrors so_follower.calibrate()'s own record_ranges_of_motion, just
timed instead of Enter-terminated since there's no interactive stdin here.

Leaves the other five motors' calibration entries untouched.
"""
import json
import shutil
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORT = "COM4"
CALIB_PATH = Path(
    r"C:\Users\labot\.cache\huggingface\lerobot\calibration\robots\so_follower\so101_follower.json"
)
RECORD_S = 25

MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

bus = FeetechMotorsBus(port=PORT, motors=MOTORS)
bus.connect()
print("connected to COM4")

bus.disable_torque("wrist_flex")
print(f"wrist_flex torque OFF -- move it through its full range now, recording for {RECORD_S}s...")

start = bus.read("Present_Position", "wrist_flex", normalize=False)
mn, mx = start, start
t0 = time.time()
last_print = 0
while time.time() - t0 < RECORD_S:
    pos = bus.read("Present_Position", "wrist_flex", normalize=False)
    mn = min(mn, pos)
    mx = max(mx, pos)
    if time.time() - last_print > 1.0:
        print(f"  t={time.time()-t0:4.1f}s pos={pos} min={mn} max={mx}")
        last_print = time.time()
    time.sleep(0.05)

print(f"recording done. min={mn} max={mx} span={mx-mn} ticks ({(mx-mn)*360/4096:.1f} deg)")

if mx - mn < 200:
    print("WARNING: span under 200 ticks -- doesn't look like it was actually moved. Aborting, no changes written.")
    bus.enable_torque("wrist_flex")
    bus.disconnect()
    raise SystemExit(1)

mid = (mn + mx) // 2
print(f"moving to midpoint {mid} before re-enabling torque there...")
bus.enable_torque("wrist_flex")
bus.write("Goal_Position", "wrist_flex", mid, normalize=False)
time.sleep(0.8)

homing = bus.set_half_turn_homings(motors=["wrist_flex"])
new_offset = homing["wrist_flex"]
print(f"wrist_flex homing_offset -> {new_offset}")

backup = CALIB_PATH.with_suffix(f".pre_id4_handguided_{int(time.time())}.json")
shutil.copy(CALIB_PATH, backup)
print(f"backed up existing calibration to {backup}")

data = json.loads(CALIB_PATH.read_text())
data["wrist_flex"] = {
    "id": 4,
    "drive_mode": 0,
    "homing_offset": new_offset,
    "range_min": mn,
    "range_max": mx,
}
CALIB_PATH.write_text(json.dumps(data, indent=4))
print(f"wrote updated calibration to {CALIB_PATH}")

calibration = {name: MotorCalibration(**data[name]) for name in MOTORS}
bus.write_calibration(calibration)
print("pushed full calibration to motors")

bus.disconnect()
print("done - other 5 motors' calibration was left untouched")
