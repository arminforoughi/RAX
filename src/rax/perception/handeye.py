"""Fitting the wrist camera's pose on the gripper from the robot's own motion.

No chessboard and no tape measure. Two facts pin the transform down:

  (A) the fingertip is in the picture at a fixed, measured pixel (the camera is rigid to
      the gripper), which fixes where the camera points;
  (B) one still object seen from several poses: every sightline must pass through the
      same point on the table, which fixes where the camera sits.

Unknowns: the transform (3 + 3) and the object (3). Residuals: 2 per view + 2 for the
fingertip. With no transform to start from (a new arm), several seeds are tried and the
best kept.

Known limitation: with one point target the camera can slide a little along its viewing
direction and the fit barely notices, so judge a fit by its fingertip gap and use wide
pose spread.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np

MIN_VIEWS = 6
MAX_TIP_GAP_PX = 12.0
MAX_RMS_PX = 100.0          # colour-blob centroids have a noise floor of tens of px


@dataclass
class HandEyeFit:
    T_ee_cam: np.ndarray
    n_views: int
    converged: bool
    rms_px: float
    tip_gap_px: float
    reason: str = ""

    @property
    def tf(self) -> str:
        return tf_string(self.T_ee_cam)

    def save(self, path: str) -> dict:
        d = {"tf": self.tf, "views": self.n_views, "rms_px": round(self.rms_px, 2),
             "tip_px": round(self.tip_gap_px, 2), "source": "reprojection",
             "fitted": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        return d


def tf_string(T: np.ndarray, places: int = 4) -> str:
    """4x4 -> ``"x,y,z,rx,ry,rz"`` (metres, rotation vector in radians)."""
    from scipy.spatial.transform import Rotation
    T = np.asarray(T, float)
    rv = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return ",".join(f"{v:.{places}f}" for v in list(T[:3, 3]) + list(rv))


def _unpack(x):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, 3] = x[:3]
    T[:3, :3] = Rotation.from_rotvec(x[3:6]).as_matrix()
    return T, np.asarray(x[6:9])


def fit_reprojection(views, geometry, tip_uv, T_seed=None, table_z=(0.0, 0.05),
                     mount_m: float = 0.10) -> HandEyeFit:
    """Fit from ``views`` = [(T_base_ee 4x4, (u, v) of the still object), ...].

    ``T_seed`` is the current transform, or None for a new arm (then a spread of camera
    orientations is tried). ``mount_m`` bounds how far the camera can sit from the tip.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    views = [(np.asarray(T, float), np.asarray(uv, float)) for T, uv in views]
    n = len(views)
    if n < MIN_VIEWS:
        return HandEyeFit(np.eye(4) if T_seed is None else np.asarray(T_seed), n, False,
                          math.nan, math.nan, f"only {n} usable views (need {MIN_VIEWS})")
    w_tip = math.sqrt(n)

    def resid(x):
        tf, p = _unpack(x)
        r = []
        for T, uv in views:
            pu = geometry.project(p, T @ tf)
            r += [400.0, 400.0] if pu is None else [pu[0] - uv[0], pu[1] - uv[1]]
        T0 = views[0][0]
        pt = geometry.project(T0[:3, 3], T0 @ tf)
        r += ([400.0, 400.0] if pt is None else
              [w_tip * (pt[0] - tip_uv[0]), w_tip * (pt[1] - tip_uv[1])])
        return np.array(r)

    def scores(x):
        r = resid(x)
        rms = float(np.sqrt((r[:2 * n] ** 2).reshape(-1, 2).sum(1).mean()))
        return rms, float(np.hypot(*r[2 * n:]) / w_tip)

    # the object: roughly where the first view's centre ray meets the table
    p0 = views[0][0][:3, 3] * np.array([1.2, 1.2, 0.0]) + np.array([0, 0, 0.02])
    b = mount_m * 1.6
    lo = np.array([-b, -b, -b, -4, -4, -4, -0.6, -0.6, table_z[0]])
    hi = np.array([b, b, b, 4, 4, 4, 0.6, 0.6, table_z[1]])
    seeds = []
    if T_seed is not None:
        T_seed = np.asarray(T_seed, float)
        seeds.append((T_seed[:3, 3], Rotation.from_matrix(T_seed[:3, :3]).as_rotvec()))
    # a new arm: the camera looks along one of the gripper's axes, a little off it
    for axis in ((0, 0, 0), (math.pi, 0, 0), (math.pi / 2, 0, 0), (-math.pi / 2, 0, 0),
                 (0, math.pi / 2, 0), (0, -math.pi / 2, 0)):
        for t in ((0, -0.05, -0.05), (0, 0.05, -0.05), (-0.05, 0, -0.05), (0.05, 0, -0.05)):
            seeds.append((np.array(t, float), np.array(axis, float)))
    best = None
    for t, rv in seeds:
        x0 = np.clip(np.concatenate([t, rv, p0]), lo + 1e-6, hi - 1e-6)
        sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=2000)
        if best is None or sol.cost < best.cost:
            best = sol
    tf, _ = _unpack(best.x)
    rms, tip = scores(best.x)
    ok = tip <= MAX_TIP_GAP_PX and rms <= MAX_RMS_PX
    return HandEyeFit(tf, n, ok, rms, tip,
                      "" if ok else f"did not converge (rms {rms:.0f}px, tip {tip:.0f}px)")


def calibrate(arm, target, save_to: str | None = None, n_views: int = 14) -> HandEyeFit:
    """Move the arm around one still object in view, fit, and apply the result.

    Put any object ``target`` can detect in the wrist camera's view first. The poses
    turn the base (rotation), tilt the wrist (rotation about another axis) and move the
    shoulder and elbow together (translation): all three are needed or the fit is
    degenerate. ``arm`` needs ``kin``, ``geom`` and ``tip_uv`` (the fingertip's pixel)
    as well as the usual rax.pick.Arm methods.
    """
    q0 = arm.joints()
    s, e, w = arm.pitch_chain[:3]
    deltas = [{arm.pan: d} for d in (-9, -4.5, 0, 4.5, 9)]
    deltas += [{w: d} for d in (-9, -4, 4, 9)]
    deltas += [{s: a, e: b} for a, b in ((-6, 6), (6, -6), (-4, 10), (4, -10), (-8, 4))]
    views, uv_last = [], None
    arm.phase("CALIB", "hand-eye: looking at one still object from several poses")
    for dq in deltas:
        arm.checkpoint()
        q = q0.copy()
        for i, d in dq.items():
            q[i] += d
        arm.move(q, speed=0.8, settle=0.3)
        bgr = arm.frame()
        dets = target.detect(bgr) if bgr is not None else []
        if not dets:
            continue
        d = (max(dets, key=lambda d: d.area) if uv_last is None else
             min(dets, key=lambda d: (d.u - uv_last[0]) ** 2 + (d.v - uv_last[1]) ** 2))
        uv_last = (d.u, d.v)
        views.append((np.asarray(arm.kin.forward_kinematics(arm.joints()), float), uv_last))
        if len(views) >= n_views:
            break
    arm.move(q0, speed=0.8, settle=0.3)
    seed = arm.geom.pose.T_ee_cam if getattr(arm, "has_handeye", True) else None
    fit = fit_reprojection(views, arm.geom, arm.tip_uv, T_seed=seed)
    arm.log(f"        hand-eye: {fit.n_views} views, fingertip {fit.tip_gap_px:.1f}px, "
            f"object rms {fit.rms_px:.1f}px -> {'OK' if fit.converged else fit.reason}")
    if fit.converged:
        arm.geom.pose.T_ee_cam = fit.T_ee_cam
        arm.has_handeye = True
        if save_to:
            fit.save(save_to)
    return fit
