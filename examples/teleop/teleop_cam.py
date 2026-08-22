#!/usr/bin/env python3
"""
Camera-frame keyboard teleop for the SO-101 with the eye-in-hand OAK-D.

Live gripper-camera window. You drive the gripper in CAMERA axes, so the
controls match what you see:

    W / S  : forward / back   (camera optical +z / -z  — into / out of the view)
    A / D  : left / right      (camera -x / +x)
    Q / E  : up / down          (world +z / -z — raise / lower the hand)
    space  : toggle gripper open/closed
    [ / ]  : smaller / bigger step
    R      : re-print current pose
    ESC    : quit

The green cross is the calibrated fingertip pixel — put the object there to grab.

Run this only when the mission server is NOT running (both hold COM4).
    python teleop_cam.py
"""
import sys, time
import numpy as np
import cv2

sys.path.insert(0, r"C:\Users\labot\Documents\lerobot\src")

from lerobot.model.kinematics import RobotKinematics
from lerobot.manipulation.visual_servo.gaze_engine import ARM_MOTORS, parse_tf_string
from lerobot.manipulation.visual_servo.cvs_engine import build_look_at_R
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig

LEROBOT = r"C:\Users\labot\Documents\lerobot"
TF = "-0.0503,0.0906,-0.1730,-0.2921,1.0770,-2.1688"
HAND_UV = (440.0, 394.0)          # calibrated fingertip pixel (from /caltip)
START = np.array([-6.7, 37.1, 48.1, -40.4, -28.4])   # a good viewing pose
Z_FLOOR = -0.020                  # don't drive the frame below the table

# ---- connect ----------------------------------------------------------------
print("connecting robot + camera on COM4 ...")
robot = make_robot_from_config(SO101FollowerConfig(
    port="COM4", id="so101_follower",
    cameras={"front": OAKDCameraConfig(fps=30, width=640, height=480, use_depth=False)},
))
robot.connect()
kin = RobotKinematics(LEROBOT + r"\SO101\so101_new_calib.urdf", "gripper_frame_link", ARM_MOTORS)
T_ee_cam = parse_tf_string(TF)
print("connected. driving to start pose ...")


def observe():
    for _ in range(8):
        try:
            obs = robot.get_observation()
            break
        except ConnectionError:
            time.sleep(0.08)
    q = np.array([float(obs[f"{m}.pos"]) for m in ARM_MOTORS])
    rgb = np.asarray(obs["front"])
    gp = float(obs.get("gripper.pos", 50.0))
    return q, rgb, gp


def send(q, gripper=None):
    act = {f"{m}.pos": float(v) for m, v in zip(ARM_MOTORS, q)}
    if gripper is not None:
        act["gripper.pos"] = float(gripper)
    robot.send_action(act)


# Orientation weight for the IK. Keeps the camera pitch/roll steady while the
# arm translates. On this 5-DoF arm yaw is left free (it follows shoulder_pan),
# so this holds the FPV level without over-constraining reach. Tune 0.5–2.0.
ORI_W = 1.0


def move_cam(q, d_cam, d_world_z=0.0):
    """Translate the gripper by d_cam (camera axes) + d_world_z (base z),
    holding the camera at its locked pitch so the first-person view stays level
    (no droop) instead of letting the wrist flop down as the arm reaches."""
    T_ee = np.asarray(kin.forward_kinematics(q))
    R_cam = (T_ee @ T_ee_cam)[:3, :3]
    d_base = R_cam @ np.asarray(d_cam, float)
    d_base[2] += d_world_z

    # Desired optical axis: keep the current heading (yaw follows the arm),
    # but force the locked downward pitch so forward/up moves don't tip the cam.
    z_cur = R_cam[:, 2]
    az = np.arctan2(z_cur[1], z_cur[0])
    z_des = np.array([
        np.cos(PITCH_LOCK) * np.cos(az),
        np.cos(PITCH_LOCK) * np.sin(az),
        -np.sin(PITCH_LOCK),
    ])
    R_goal = build_look_at_R(T_ee[:3, :3], T_ee_cam, z_des)

    T_goal = T_ee.copy()
    T_goal[:3, :3] = R_goal
    T_goal[:3, 3] = T_ee[:3, 3] + d_base
    T_goal[2, 3] = max(T_goal[2, 3], Z_FLOOR)
    return kin.inverse_kinematics(q, T_goal, position_weight=1.0, orientation_weight=ORI_W)


# move to start
q, _, gp = observe()
for _ in range(60):
    step = np.clip(START - q, -3.0, 3.0)
    if np.max(np.abs(step)) < 0.5:
        break
    q = q + step
    send(q)
    time.sleep(0.03)

# Lock the camera pitch at the START pose. Every move afterwards holds this
# downward pitch (angle below horizon of the camera optical axis) so the FPV
# never droops. Re-lock by restarting from a pose you like.
_q_lock = observe()[0]
_z0 = (np.asarray(kin.forward_kinematics(_q_lock)) @ T_ee_cam)[:3, 2]
PITCH_LOCK = float(np.arctan2(-_z0[2], np.hypot(_z0[0], _z0[1])))  # rad below horizon
print(f"camera pitch locked at {np.degrees(PITCH_LOCK):+.1f} deg below horizon")

step_m = 0.006          # metres per keypress
grip_open = True
send(observe()[0], gripper=95.0)
print("\nREADY — click the camera window, then use W/A/S/D  Q/E  space  [ ]  ESC\n")

win = "SO-101 teleop (gripper cam)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

while True:
    q, rgb, gp = observe()
    img = np.ascontiguousarray(rgb[:, :, ::-1])   # RGB->BGR for cv2
    T = np.asarray(kin.forward_kinematics(q))
    p = T[:3, 3]
    cv2.drawMarker(img, (int(HAND_UV[0]), int(HAND_UV[1])), (0, 255, 0),
                   cv2.MARKER_CROSS, 26, 2)
    hud = [f"xyz=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})m  grip={'OPEN' if grip_open else 'SHUT'}",
           f"step={step_m*1000:.0f}mm   W/S fwd  A/D left/right  Q/E up/down  space grip  ESC quit"]
    for i, line in enumerate(hud):
        cv2.putText(img, line, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(win, img)

    k = cv2.waitKey(30) & 0xFF
    if k == 255:
        continue
    try:
        if k == 27:                                   # ESC
            break
        elif k in (ord('w'), ord('W')):
            send(move_cam(q, [0, 0, +step_m]))
        elif k in (ord('s'), ord('S')):
            send(move_cam(q, [0, 0, -step_m]))
        elif k in (ord('a'), ord('A')):
            send(move_cam(q, [-step_m, 0, 0]))
        elif k in (ord('d'), ord('D')):
            send(move_cam(q, [+step_m, 0, 0]))
        elif k in (ord('q'), ord('Q')):
            send(move_cam(q, [0, 0, 0], d_world_z=+step_m))
        elif k in (ord('e'), ord('E')):
            send(move_cam(q, [0, 0, 0], d_world_z=-step_m))
        elif k == ord(' '):
            grip_open = not grip_open
            send(q, gripper=95.0 if grip_open else 5.0)
        elif k == ord('['):
            step_m = max(0.001, step_m - 0.001)
        elif k == ord(']'):
            step_m = min(0.020, step_m + 0.001)
        elif k in (ord('r'), ord('R')):
            print(f"pose xyz=({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f})  joints={np.round(q,1).tolist()}")
    except Exception as ex:
        print("move rejected:", ex)

cv2.destroyAllWindows()
robot.disconnect()
print("teleop ended, robot disconnected.")
