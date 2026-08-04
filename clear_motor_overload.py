"""Clear overload latch on the SO-101 motors.

stack_mission2.py was failing with:
    RuntimeError: Failed to read 'Min_Position_Limit' on id_=2 ... Overload error!

This script power-cycles torque on motor IDs 1-6 via the Feetech bus on COM4,
which clears the latched overload flag so the robot can reconnect.
"""
import time

try:
    import scservo_sdk as scs
except ImportError:
    print("scservo_sdk not installed; cannot clear overload")
    raise SystemExit(1)

PORT = "COM4"
BAUD = 1000000
IDS = [1, 2, 3, 4, 5, 6]

ph = scs.PortHandler(PORT)
if not ph.openPort():
    raise RuntimeError(f"Cannot open {PORT}")
ph.setBaudRate(BAUD)
pk = scs.PacketHandler(0)

for motor_id in IDS:
    try:
        pk.write1ByteTxRx(ph, motor_id, 40, 0)  # torque off
        time.sleep(0.2)
        pk.write1ByteTxRx(ph, motor_id, 40, 1)  # torque on
        time.sleep(0.2)
        pos, _, err = pk.read2ByteTxRx(ph, motor_id, 56)
        print(f"motor {motor_id}: pos={pos}, status={err:#04x}")
    except Exception as e:
        print(f"motor {motor_id}: {e}")

ph.closePort()
print("overload clear done")
