"""Fitting the gripper->camera transform from the robot's own motion.

No chessboard, no tape measure. Every new arm needs its own hand-eye fit, so this is
part of plugging one in rather than a one-off setup step.

WHY IT MATTERS. A camera bolted to the gripper that *thinks* it is pitched further down
than it really is back-projects every sightline too steeply, so each ray hits the table
too soon, so every object is reported nearer than it is. On this rig the shipped
transform was ~40 degrees out in camera pitch, and that single error produced most of
the symptoms chased for two days — including a standing complaint that objects "should
be further out", arrived at independently from the other end.

The error needs no fitting to *detect*, only to correct. The camera is rigid to the end
effector, so the fingertip projects to ONE fixed pixel in every pose. Measure that pixel
directly, ask the transform to predict it, and the gap between them is the error in
pixels. A broken transform put it 330 px away, in the wrong half of the frame — no
intrinsics can reconcile that.

Two objectives are offered, differing in what they ask of the samples:

:func:`fit_reprojection`
    Every sightline to one static target must pass through a single 3D point, and the
    fingertip must land on its measured pixel. Solves the transform AND the target
    position together. Nails pointing direction from the tip anchor and position from
    the parallax between poses.

    **Known limitation.** This objective has a flat direction. With one point target,
    the camera can slide along its viewing direction and the target follow it,
    reproducing every observed pixel — so the fit is only pinned by the parallax
    between poses, and a small pose spread leaves that nearly unobserved. Measured on
    noise-free synthetic data seeded with the EXACT truth, it settles ~14 mm and ~3 deg
    away while reporting 0.5 px reprojection RMS and 0.01 px fingertip error. The
    numbers look excellent because the parts they measure ARE excellent.

    Practical consequence: re-running this on an already-good robot can move the
    transform. Judge it by the fingertip gap, which is a hard geometric constraint
    rather than a fitted one, and use :func:`fit_consistency` to CHECK a calibration.
    Widening the pose spread is what actually shrinks the freedom.

:func:`fit_consistency`
    A stationary object must map to the SAME table position from every viewpoint. Does
    not need the target's position as an unknown, and each extra viewpoint adds
    constraints rather than parameters.

Both keep the fingertip term as an anchor. Without it, constraining only the tip pixel
gives 2 equations for 3 rotation angles — roll about the tip ray is unobservable — so a
fit holds near one pose and drifts everywhere else.

Only the maths lives here. Collecting the samples needs to move the arm, so the caller
owns that.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "HandEyeSample", "HandEyeFit", "fit_reprojection", "fit_consistency",
    "load_hand_eye", "save_hand_eye", "tf_string",
]

# Accept a fit when the fingertip anchor nails it and the target reprojection is merely
# sane. Do NOT gate tightly on the reprojection RMS: a colour-blob centroid has a
# ~50-80 px noise floor that more poses do not lower, so a 25 px bar rejected a GOOD fit
# (tip 0 px, rms 49 px) and kept the broken shipped transform.
MAX_TIP_GAP_PX = 12.0
MAX_RMS_PX = 100.0
#: How far the same object may appear to move between viewpoints, for fit_consistency.
MAX_SPREAD_M = 0.04

MIN_VIEWS_REPROJECTION = 6
MIN_VIEWS_CONSISTENCY = 5


@dataclass(frozen=True)
class HandEyeSample:
    """One view of a static target: where the end effector was, and which pixel the
    target appeared at."""

    T_base_ee: np.ndarray      # 4x4 FK pose at this sample
    uv: np.ndarray             # (u, v) of the target in that frame


@dataclass
class HandEyeFit:
    """The outcome of a fit. ``converged`` is the only thing a caller should trust
    before writing the transform anywhere."""

    T_ee_cam: np.ndarray
    n_views: int
    method: str
    converged: bool = False
    reason: str = ""
    rms_px: float = float("nan")
    tip_gap_px: float = float("nan")
    spread_m: float = float("nan")
    target_p: np.ndarray | None = None
    before: dict = field(default_factory=dict)

    @property
    def tf(self) -> str:
        return tf_string(self.T_ee_cam)

    def to_dict(self) -> dict:
        d = {"tf": self.tf, "views": self.n_views, "source": self.method,
             "fitted": time.strftime("%Y-%m-%d %H:%M:%S")}
        if not math.isnan(self.rms_px):
            d["rms_px"] = round(float(self.rms_px), 2)
        if not math.isnan(self.tip_gap_px):
            d["tip_px"] = round(float(self.tip_gap_px), 2)
        if not math.isnan(self.spread_m):
            d["spread_cm"] = round(float(self.spread_m) * 100, 2)
        return d


def tf_string(T: np.ndarray, places: int = 4) -> str:
    """4x4 -> the ``"x,y,z,rx,ry,rz"`` form profiles and the JSON file use."""
    from scipy.spatial.transform import Rotation

    T = np.asarray(T, dtype=np.float64)
    rv = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return ",".join(f"{v:.{places}f}" for v in list(T[:3, 3]) + list(rv))


def save_hand_eye(path: str, fit: HandEyeFit) -> dict:
    d = fit.to_dict()
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return d


def load_hand_eye(path: str) -> dict | None:
    """The stored transform, or None when there is no usable calibration."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        raise ValueError(f"bad {os.path.basename(path)}: {e}") from e


# --- shared pieces -------------------------------------------------------------------
def _unpack_t_first(x):
    """x = [tx, ty, tz, rx, ry, rz, ...] -> (4x4, rest)."""
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, 3] = x[:3]
    T[:3, :3] = Rotation.from_rotvec(x[3:6]).as_matrix()
    return T, np.asarray(x[6:])


def _unpack_r_first(x):
    """x = [rx, ry, rz, tx, ty, tz] -> 4x4."""
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:6]
    return T


def _tip_error_uv(geometry, T_base_ee, T_ee_cam, tip_uv, tip_offset_m=0.0):
    """(du, dv) between where the transform SAYS the fingertip is and where it was
    measured, or None if it projects behind the camera.

    Returned as components, not a magnitude: a least-squares residual needs the signed
    error in each axis. Collapsing it to a distance first discards the direction and
    changes what the optimiser converges to.
    """
    tip = T_base_ee[:3, 3]
    if tip_offset_m:
        tip = tip + T_base_ee[:3, :3] @ np.array([0.0, 0.0, float(tip_offset_m)])
    uv = geometry.project(tip, T_base_ee @ T_ee_cam)
    return None if uv is None else (uv[0] - tip_uv[0], uv[1] - tip_uv[1])


def _tip_gap(geometry, T_base_ee, T_ee_cam, tip_uv, tip_offset_m=0.0) -> float:
    """Pixel distance between the predicted and measured fingertip. The single most
    diagnostic number about a hand-eye transform."""
    d = _tip_error_uv(geometry, T_base_ee, T_ee_cam, tip_uv, tip_offset_m)
    return 999.0 if d is None else math.hypot(d[0], d[1])


# --- objective 1: all sightlines meet at one point -----------------------------------
def fit_reprojection(samples, geometry, *, tip_uv, T_seed, target_seed=None,
                     cam_tip_m: float = 0.10, max_tip_gap_px: float = MAX_TIP_GAP_PX,
                     max_rms_px: float = MAX_RMS_PX) -> HandEyeFit:
    """Fit the transform AND the target position from views of one static target.

    Unknowns: transform translation (3) + rotation vector (3) + the target (3, with z
    bounded to the table). Residuals: 2 per view + 2 for the fingertip anchor. Seeded
    from the current transform, so a good one stays put.

    The fingertip anchor is pose-independent — the camera is rigid to the end effector —
    so it is ONE constraint however many poses were taken. It is weighted like sqrt(N)
    samples so it is not drowned out by the noisier target pixels.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    samples = list(samples)
    n = len(samples)
    if n < MIN_VIEWS_REPROJECTION:
        return HandEyeFit(np.asarray(T_seed), n, "reprojection", False,
                          f"only {n} usable views (need {MIN_VIEWS_REPROJECTION})")

    T_ee = [np.asarray(s.T_base_ee, dtype=np.float64) for s in samples]
    uvs = [np.asarray(s.uv, dtype=np.float64) for s in samples]
    w_tip = math.sqrt(n)

    def resid(x):
        tf, p = _unpack_t_first(x)
        r = []
        for T, uv in zip(T_ee, uvs):
            pu = geometry.project(p, T @ tf)
            r += [400.0, 400.0] if pu is None else [pu[0] - uv[0], pu[1] - uv[1]]
        pt = geometry.project(T_ee[0][:3, 3], T_ee[0] @ tf)
        r += ([400.0, 400.0] if pt is None else
              [w_tip * (pt[0] - tip_uv[0]), w_tip * (pt[1] - tip_uv[1])])
        return r

    def rms(x):
        r = np.array(resid(x))[: 2 * n]
        return float(np.sqrt((r ** 2).reshape(-1, 2).sum(1).mean()))

    def tipgap(x):
        tf, _ = _unpack_t_first(x)
        return _tip_gap(geometry, T_ee[0], tf, tip_uv)

    # SEED THE TRANSLATION AT THE MEASURED MOUNT DISTANCE, not at the old transform's
    # value. Starting a nonlinear fit 2x off in translation invites a bad local minimum,
    # so keep the old direction (the mount geometry is roughly right), rescale it, and
    # bound |t| to what a camera bolted to this gripper can physically be.
    T_seed = np.asarray(T_seed, dtype=np.float64)
    t_old = np.asarray(T_seed[:3, 3], dtype=np.float64)
    t_seed = t_old / max(1e-6, float(np.linalg.norm(t_old))) * float(cam_tip_m)
    p_seed = np.array([0.18, 0.0, 0.02]) if target_seed is None else np.asarray(target_seed)
    x0 = np.concatenate([t_seed,
                         Rotation.from_matrix(T_seed[:3, :3]).as_rotvec(),
                         np.asarray(p_seed, dtype=np.float64)])
    lo = np.array([-0.16, -0.16, -0.16, -4.0, -4.0, -4.0, -0.45, -0.45, 0.005])
    hi = np.array([0.16, 0.16, 0.16, 4.0, 4.0, 4.0, 0.45, 0.45, 0.050])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    before = {"rms_px": rms(x0), "tip_gap_px": tipgap(x0)}

    sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac",
                        max_nfev=4000, ftol=1e-10, xtol=1e-10)
    tf, p = _unpack_t_first(sol.x)
    got_rms, got_tip = rms(sol.x), tipgap(sol.x)
    ok = got_tip <= max_tip_gap_px and got_rms <= max_rms_px
    return HandEyeFit(
        tf, n, "reprojection", ok,
        "" if ok else (f"fit did not converge (RMS {got_rms:.0f}px, tip {got_tip:.0f}px) "
                       f"— more pose spread needed"),
        rms_px=got_rms, tip_gap_px=got_tip, target_p=p, before=before)


# --- objective 2: the same object lands in the same place ----------------------------
def fit_consistency(samples, geometry, *, tip_uv, T_seed, z_plane: float = 0.0,
                    tip_offset_m: float = 0.0, max_spread_m: float = MAX_SPREAD_M
                    ) -> HandEyeFit:
    """Fit the transform by demanding a stationary object map to ONE table position.

    That is the honest objective, and it pins the mount orientation properly. Fitting
    the target's reprojection instead has a ~55-80 px noise floor, so the fit wanders
    and an acceptance gate throws away good solutions; here every extra viewpoint adds
    constraints rather than noise.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    samples = list(samples)
    n = len(samples)
    if n < MIN_VIEWS_CONSISTENCY:
        return HandEyeFit(np.asarray(T_seed), n, "multi-view consistency", False,
                          f"only {n} views (need {MIN_VIEWS_CONSISTENCY}) — "
                          f"keep the object visible while the arm pans")

    T_seed = np.asarray(T_seed, dtype=np.float64)
    pairs = [(np.asarray(s.T_base_ee, np.float64), np.asarray(s.uv, np.float64))
             for s in samples]
    t0 = T_seed[:3, 3].copy()
    g = geometry

    def table_pts(T_ee_cam):
        """Where each view says the object is, on the table plane. None if any ray
        points the wrong way — a fit built on one is meaningless."""
        pts = []
        for T_ee, uv in pairs:
            T = T_ee @ T_ee_cam
            o = T[:3, 3]
            d = T[:3, :3] @ np.array([(uv[0] - g.cx) / g.fx, (uv[1] - g.cy) / g.fy, 1.0])
            if d[2] >= -1e-3:
                return None
            t = (z_plane - o[2]) / d[2]
            if not (0.02 < t < 2.0):
                return None
            pts.append((o + t * d)[:2])
        return np.array(pts) if pts else None

    def resid(x):
        T = _unpack_r_first(x)
        pts = table_pts(T)
        if pts is None:
            return np.full(2 * n + 2, 10.0)
        spread = (pts - pts.mean(axis=0)).ravel() * 40.0        # metres -> weighted
        # The anchor keeps the solution from sliding into a mirrored or degenerate
        # pose; it is weighted low so it guides rather than dominates.
        d = _tip_error_uv(g, pairs[0][0], T, tip_uv, tip_offset_m)
        anchor = np.array([10.0, 10.0]) if d is None else np.array(d) * 0.02
        return np.concatenate([spread, anchor])

    def spread_of(x):
        pts = table_pts(_unpack_r_first(x))
        if pts is None:
            return 999.0
        return float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean())

    x0 = np.concatenate([Rotation.from_matrix(T_seed[:3, :3]).as_rotvec(), t0])
    lo = np.concatenate([x0[:3] - 1.2, t0 - 0.06])
    hi = np.concatenate([x0[:3] + 1.2, t0 + 0.06])
    before = spread_of(x0)
    sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac",
                        max_nfev=3000, ftol=1e-10, xtol=1e-10)
    after = spread_of(sol.x)
    T_new = _unpack_r_first(sol.x)
    ok = after <= before and after <= max_spread_m
    return HandEyeFit(
        T_new, n, "multi-view consistency", ok,
        "" if ok else f"did not converge (spread {after * 100:.1f}cm) — mount NOT changed",
        spread_m=after,
        tip_gap_px=_tip_gap(g, pairs[0][0], T_new, tip_uv, tip_offset_m),
        before={"spread_m": before})
