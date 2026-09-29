"""A simulated SO-101 with a wrist camera and tubes lying on the table.

The kinematics, the IK and the camera model are the real ones; only the picture and the
gripper are made up. Enough to run the whole pick closed-loop without hardware:

    from rax.pick import ColourTarget, pick
    from rax.pick.sim import SimArm, Tube
    arm = SimArm([Tube(0.24, 0.06, yaw_deg=30)])
    pick(arm, ColourTarget(), near_xy=(0.24, 0.06))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .arm import pitch_of

#: A hand-eye transform fitted on the real SO-101 (reprojection, 14 views).
SO101_HANDEYE = "-0.0140,-0.0854,-0.0417,-0.8057,-0.0210,0.0627"

#: Cap colours inside the detector's hue windows (H 82, 97, 177).
BGR = {"green": (146, 190, 26), "blue": (190, 152, 26), "red": (42, 26, 190)}


@dataclass
class Tube:
    """A 16 x 100 mm tube lying on the table; (x, y) is its cap."""

    x: float
    y: float
    yaw_deg: float = 0.0
    colour: str = "green"
    held: bool = False

    def points(self):
        a = math.radians(self.yaw_deg)
        cap = np.array([self.x, self.y, 0.008])
        return cap, cap + 0.09 * np.array([math.cos(a), math.sin(a), 0.0])


class SimArm:
    """Implements :class:`rax.pick.arm.Arm` with the SO-101's model and a drawn picture."""

    def __init__(self, tubes=(), profile: str = "so101", handeye: str = SO101_HANDEYE):
        from rax.manipulation.arms.ik_strategy import make_ik
        from rax.manipulation.arms.kinematics import make_kinematics
        from rax.perception.camera_geometry import (
            CameraGeometry,
            EyeInHand,
            intrinsics_from_dict,
            parse_tf,
        )
        from rax.robots.profiles import load_profile

        p = load_profile(profile)
        self.kin = make_kinematics(p.urdf_path, p.ee_frame, list(p.joint_names))
        self._ik = make_ik(self.kin, p)
        self.geom = CameraGeometry(
            intrinsics_from_dict(dict(zip(("fx", "fy", "cx", "cy"),
                                          p.camera.intrinsics_fallback)),
                                 width=p.camera.width, height=p.camera.height),
            EyeInHand(lambda q: self.kin.forward_kinematics(q), parse_tf(handeye)))
        self.size = (p.camera.width, p.camera.height)
        self.pan, self.roll, self.pitch_chain = p.pan_joint, p.roll_joint, tuple(p.pitch_chain)
        self.lo, self.hi = p.limits()
        self.home = np.array(p.home_deg, float)
        self.q = self.home.copy()
        self.jaw_uv = self.geom.tip_pixel(self.q)
        self.jaw_axis_deg = self._closing_axis_deg()
        self.roll_gain = 1.0
        self.tubes = list(tubes)
        self.opening = 50.0
        self.messages: list[str] = []

    # ---- motion and model ----
    def joints(self):
        return self.q.copy()

    def move(self, q, speed=1.0, settle=0.2):
        self.q = np.clip(np.asarray(q, float), self.lo, self.hi)
        for t in self.tubes:
            if t.held:
                tip = self.tip(self.q)
                t.x, t.y = float(tip[0]), float(tip[1])

    def tip(self, q):
        return np.asarray(self.kin.forward_kinematics(np.asarray(q, float)))[:3, 3]

    def ik(self, seed, p, pitch, roll):
        return self._ik.solve(np.asarray(seed, float), np.asarray(p, float),
                              pitch_deg=pitch, roll_deg=roll)

    # ---- camera ----
    def _T(self, q):
        return self.geom.T_base_cam(np.asarray(q, float))

    def project(self, p, q):
        return self.geom.project(np.asarray(p, float), self._T(q))

    def cast(self, uv, q):
        pt = self.geom.ray_to_plane(uv, self._T(q), 0.008)
        return None if pt is None else (float(pt[0]), float(pt[1]))

    def frame(self):
        w, h = self.size
        img = np.full((h, w, 3), 45, np.uint8)
        T = self._T(self.q)
        for t in self.tubes:
            cap, end = t.points()
            a, b = self.geom.project(cap, T), self.geom.project(end, T)
            if a is None or b is None:
                continue
            r = self._px_radius(cap, T)
            cv2.line(img, _i(a), _i(b), (215, 215, 215), max(2, int(1.6 * r)))
            cv2.circle(img, _i(a), max(2, int(r)), BGR[t.colour], -1)
        return img

    def _px_radius(self, p, T):
        depth = (np.linalg.inv(T) @ np.append(p, 1.0))[2]
        return max(self.geom.fx * 0.008 / max(depth, 1e-3), 2.0)

    def _closing_axis_deg(self):
        """The image angle of the ee x axis: the line the SO-101's jaws close along."""
        T, Tc = np.asarray(self.kin.forward_kinematics(self.q)), self._T(self.q)
        c = Tc[:3, 3] + 0.15 * Tc[:3, 2]                 # a point in front of the lens
        a = self.project(c + 0.02 * T[:3, 0], self.q)
        b = self.project(c - 0.02 * T[:3, 0], self.q)
        return math.degrees(math.atan2(a[1] - b[1], a[0] - b[0])) % 180.0

    # ---- gripper ----
    def grip(self, pct):
        self.opening = float(pct)

    def release(self):
        self.opening = 100.0
        for t in self.tubes:
            t.held = False

    def close(self, from_pct):
        """Holds a tube if it lies between the jaws (open ~4cm), low, square across them."""
        T = np.asarray(self.kin.forward_kinematics(self.q))
        tip, jaw_x = T[:3, 3], T[:3, 0]
        self.opening = 1.0
        for t in self.tubes:
            cap, end = t.points()
            ax = (end - cap) / np.linalg.norm(end - cap)
            s = float(np.clip(np.dot(tip[:2] - cap[:2], ax[:2]), 0.0, 0.09))
            miss = np.linalg.norm(tip[:2] - (cap[:2] + s * ax[:2]))
            across = abs(float(np.dot(ax, jaw_x))) < math.sin(math.radians(40))
            if miss < 0.018 and tip[2] < 0.03 and across and pitch_of(self, self.q) > 60:
                t.held, self.opening = True, 8.0
                return True
        return False

    def grip_pos(self):
        return self.opening

    # ---- operator ----
    def checkpoint(self):
        pass

    def log(self, msg):
        self.messages.append(str(msg))

    def phase(self, name, note=""):
        self.messages.append(f"[{name}] {note}")


def _i(p):
    return int(round(p[0])), int(round(p[1]))
