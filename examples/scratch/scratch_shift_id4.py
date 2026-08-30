"""Shift wrist_flex's REPORTED angle by a delta, without moving the arm.

Use when the 3D render disagrees with the real gripper angle: the physical joint
stays exactly where it is, but the number it reports (and therefore the URDF/FK
render, and every pitch computation) moves by `delta_deg`.

    python scratch_shift_id4.py +45      # model reads 45 deg MORE (more nose-down)
    python scratch_shift_id4.py -10      # back off 10 deg

Positive delta = the model reports a larger wrist_flex, i.e. more downward pitch,
since gripper pitch = shoulder_lift + elbow_flex + wrist_flex.

Feetech: Present_Position = Actual_Position - Homing_Offset, so raising the
reported value means LOWERING the homing offset. Torque is dropped for the EEPROM
write and the goal is re-synced to the measured position before re-enabling, or the
servo snaps to a stale goal in the old frame (a 45 deg jerk, and exactly the load
that trips the overload latch).
"""
import json
import shutil
import sys
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

CALIB = Path(r"C:\Users\labot\.cache\huggingface\lerobot\calibration\robots"
             r"\so_follower\so101_follower.json")
TICKS_PER_DEG = 4095 / 360.0

delta_deg = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
if not delta_deg:
    print("usage: scratch_shift_id4.py <delta_deg>")
    raise SystemExit(1)

data = json.loads(CALIB.read_text())
MOTORS = {n: Motor(i, "sts3215", MotorNormMode.DEGREES) for n, i in
          [("shoulder_pan", 1), ("shoulder_lift", 2), ("elbow_flex", 3),
           ("wrist_flex", 4), ("wrist_roll", 5)]}
MOTORS["gripper"] = Motor(6, "sts3215", MotorNormMode.RANGE_0_100)
cal = {n: MotorCalibration(**data[n]) for n in MOTORS}

bus = FeetechMotorsBus(port="COM4", motors=MOTORS, calibration=cal)
bus.connect()

j = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
before = bus.sync_read("Present_Position", j, normalize=True)
print(f"before: wrist_flex={before['wrist_flex']:.1f}  "
      f"model pitch={before['shoulder_lift']+before['elbow_flex']+before['wrist_flex']:.1f}")

ho_old = data["wrist_flex"]["homing_offset"]
ho_new = int(round(ho_old - delta_deg * TICKS_PER_DEG))
if abs(ho_new) > 2047:
    print(f"ABORT: homing offset {ho_new} outside encodable range")
    bus.disconnect()
    raise SystemExit(1)

# Will the joint still sit inside its own limit window after the shift?
raw_now = bus.read("Present_Position", "wrist_flex", normalize=False)
raw_after = raw_now + int(round(delta_deg * TICKS_PER_DEG))
lo, hi = data["wrist_flex"]["range_min"], data["wrist_flex"]["range_max"]
print(f"raw {raw_now} -> {raw_after} (window {lo}..{hi})")
if not (lo <= raw_after <= hi):
    print("ABORT: the shift would put the joint outside its travel window")
    bus.disconnect()
    raise SystemExit(1)

bak = CALIB.with_suffix(f".pre_id4_shift_{int(time.time())}.json")
shutil.copy(CALIB, bak)
print(f"backed up -> {bak.name}")

data["wrist_flex"]["homing_offset"] = ho_new
CALIB.write_text(json.dumps(data, indent=4))
print(f"homing_offset {ho_old} -> {ho_new}  (delta {delta_deg:+.1f} deg)")

bus.disable_torque("wrist_flex")
bus.write_calibration({n: MotorCalibration(**data[n]) for n in MOTORS})
# re-sync the goal in the NEW frame before torque returns, so nothing jerks
raw = bus.read("Present_Position", "wrist_flex", normalize=False)
bus.write("Goal_Position", "wrist_flex", raw, normalize=False)
bus.enable_torque("wrist_flex")

after = bus.sync_read("Present_Position", j, normalize=True)
print(f"after : wrist_flex={after['wrist_flex']:.1f}  "
      f"model pitch={after['shoulder_lift']+after['elbow_flex']+after['wrist_flex']:.1f}")
print("arm did NOT move; only the reported number changed")
bus.disconnect()
