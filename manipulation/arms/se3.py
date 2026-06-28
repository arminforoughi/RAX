"""SE(3) helpers for visual servoing — condensed from lerobot ``cvs_engine``.

Pure geometry, no robot/camera state, so it is trivially unit-testable. The gaze
engine uses these to build a "look at the object" orientation and to step the EE
pose toward a target at a bounded Cartesian velocity.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


def approach_unit_vector(az_deg: float, el_deg: float) -> np.ndarray:
    """Unit vector FROM the object TO the EE, in base frame.

    ``el=+90`` -> (0, 0, +1) top-down; ``el=0, az=0`` -> (-1, 0, 0) side-on;
    ``az`` rotates around the object's vertical axis.
    """
    az, el = math.radians(az_deg), math.radians(el_deg)
    horiz = math.cos(el)
    return np.array([-horiz * math.cos(az), horiz * math.sin(az), math.sin(el)], dtype=np.float64)


def vantage_dir(up: np.ndarray, az_deg: float, el_deg: float) -> np.ndarray:
    """Unit vector FROM the object TO the EE for a chosen approach angle.

    Generalises lerobot's ``approach_unit_vector`` to an arbitrary world-up axis
    so it works in any base convention (SO-101 +Z up, the mock -Y up, ...):

        el = 90  -> straight up the ``up`` axis      (top-down approach)
        el = 0   -> in the horizontal plane          (side approach)
        az       -> rotates the side direction around ``up``

    The EE ends up on that side of the object, camera looking back at it.
    """
    up = np.asarray(up, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(up))
    up = up / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    # Two orthonormal axes spanning the horizontal plane (perpendicular to up).
    a = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    fwd = a - float(a @ up) * up
    fwd /= np.linalg.norm(fwd)
    right = np.cross(up, fwd)
    el, az = math.radians(el_deg), math.radians(az_deg)
    horiz = math.cos(el) * (math.cos(az) * fwd + math.sin(az) * right)
    return math.sin(el) * up + horiz


def look_at_R(R_base_ee_cur: np.ndarray, T_ee_cam: np.ndarray, optical_axis_target_base: np.ndarray) -> np.ndarray:
    """R_base_ee that aligns the camera optical axis (cam +Z) with the target dir.

    Wrist roll is left free (required on a 5-DoF arm). Returns a 3x3 rotation.
    """
    R_ee_cam = np.asarray(T_ee_cam, dtype=np.float64)[:3, :3]
    R_base_cam_cur = np.asarray(R_base_ee_cur, dtype=np.float64) @ R_ee_cam

    z = np.asarray(optical_axis_target_base, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(z))
    z = z / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])

    x_cur = R_base_cam_cur[:, 0]
    x_proj = x_cur - float(np.dot(x_cur, z)) * z
    m = float(np.linalg.norm(x_proj))
    if m < 1e-6:
        for cand in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
            x_proj = cand - float(np.dot(cand, z)) * z
            m = float(np.linalg.norm(x_proj))
            if m >= 1e-6:
                break
    x = x_proj / m
    y = np.cross(z, x)
    R_base_cam_target = np.column_stack([x, y, z])
    return R_base_cam_target @ R_ee_cam.T


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation matrix that maps unit-ish vector ``a`` onto ``b``."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0.0:
            return np.eye(3)
        # 180°: rotate about any axis orthogonal to a.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(math.pi * axis).as_matrix()
    return Rotation.from_rotvec(math.atan2(s, c) * (v / s)).as_matrix()


def look_at_ray_R(
    R_base_ee_cur: np.ndarray,
    T_ee_cam: np.ndarray,
    target_dir_base: np.ndarray,
    aim_ray_cam: np.ndarray,
) -> np.ndarray:
    """R_base_ee that aligns a *chosen camera ray* (not the optical axis) with the target.

    ``look_at_R`` points the camera optical axis (cam +Z) at the object, which puts
    it at the image centre. For grasping we want the object to land on the *aim ray*
    — the pixel where the fingertips sit (bottom-centre) — so the object ends up
    where the gripper actually closes. ``aim_ray_cam`` is that ray in camera frame
    (e.g. back-projected from the aim pixel). Wrist roll is left free (5-DoF arm).
    """
    R_ee_cam = np.asarray(T_ee_cam, dtype=np.float64)[:3, :3]
    # Orientation that puts the optical axis on the target.
    R_ee_optical = look_at_R(R_base_ee_cur, T_ee_cam, target_dir_base)
    R_base_cam_optical = R_ee_optical @ R_ee_cam
    # Rotate so the aim ray (rather than +Z) lands on the target:
    #   R_base_cam @ aim = R_base_cam_optical @ R_z2aim.T @ aim = R_base_cam_optical @ z = target.
    z = np.array([0.0, 0.0, 1.0])
    aim = np.asarray(aim_ray_cam, dtype=np.float64).reshape(3)
    aim = aim / max(float(np.linalg.norm(aim)), 1e-12)
    R_z2aim = _rotation_between(z, aim)
    R_base_cam = R_base_cam_optical @ R_z2aim.T
    return R_base_cam @ R_ee_cam.T


def interpolate_se3(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """LERP translation, SLERP rotation between two 4x4 poses."""
    a = float(np.clip(alpha, 0.0, 1.0))
    T0 = np.asarray(T0, dtype=np.float64)
    T1 = np.asarray(T1, dtype=np.float64)
    out = np.eye(4)
    out[:3, 3] = (1.0 - a) * T0[:3, 3] + a * T1[:3, 3]
    r0 = Rotation.from_matrix(T0[:3, :3])
    r1 = Rotation.from_matrix(T1[:3, :3])
    rel = (r1 * r0.inv()).as_rotvec()
    out[:3, :3] = (Rotation.from_rotvec(a * rel) * r0).as_matrix()
    return out


def rate_limited_step(
    T_start: np.ndarray,
    T_target: np.ndarray,
    *,
    dt: float,
    max_lin_vel_m_s: float,
    max_ang_vel_deg_s: float,
) -> np.ndarray:
    """Step from ``T_start`` toward ``T_target``, capped by per-tick lin/ang velocity."""
    T_start = np.asarray(T_start, dtype=np.float64)
    T_target = np.asarray(T_target, dtype=np.float64)
    d_t = float(np.linalg.norm(T_target[:3, 3] - T_start[:3, 3]))
    R_rel = T_target[:3, :3] @ T_start[:3, :3].T
    d_r = float(np.linalg.norm(Rotation.from_matrix(R_rel).as_rotvec()))
    a_t = 1.0 if d_t < 1e-9 else min(1.0, max_lin_vel_m_s * dt / d_t)
    a_r = 1.0 if d_r < 1e-6 else min(1.0, math.radians(max_ang_vel_deg_s) * dt / d_r)
    return interpolate_se3(T_start, T_target, min(a_t, a_r, 1.0))


def make_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 pose from a 3x3 rotation and a 3-vector translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T
