"""Fitting the mount against objects whose position a GRASP has proved.

Two fitters already existed and both failed on this rig, for the same underlying
reason: neither objective knows where the object actually is.

``fit_reprojection`` solves for the transform AND the target together — nine unknowns,
so translation trades against rotation with the target absorbing the difference. Seeded
with the exact true transform on noise-free data it still drifts while reporting a
near-perfect residual.

``fit_consistency`` demands only that the viewpoints AGREE. Agreement is not
correctness: on hardware it reached 0.3cm agreement across ten views and picking got
WORSE, because the ten views now agreed on the wrong place.

A grasp settles it. When the jaws close the object is between the fingertips, and
forward kinematics says where those are — the arm's own encoders, no camera involved.
Fixing the target collapses nine unknowns to six, and takes the flat direction with it.

    pytest tests/test_handeye_grasp_fit.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO = pathlib.Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from rax.perception.camera_geometry import (  # noqa: E402
    CameraGeometry, EyeInHand, intrinsics_from_dict, parse_tf,
)
from rax.perception.handeye import (  # noqa: E402
    GraspSample, fit_reprojection, fit_to_known_points,
)

T_TRUE = parse_tf("-0.0503,0.0906,-0.1730,-0.2921,1.0770,-2.1688")
HOME = np.array([-14.1, -99.1, 90.8, 33.2, -4.7])


def _kin():
    from rax.manipulation.arms.kinematics import make_kinematics
    from rax.robots.profiles import load_profile
    p = load_profile("so101")
    return make_kinematics(p.urdf_path, p.ee_frame, list(p.joint_names))


def _rig():
    kin = _kin()
    geom = CameraGeometry(
        intrinsics_from_dict({"fx": 517.0, "fy": 517.0, "cx": 329.5, "cy": 231.4}, 640, 480),
        EyeInHand(lambda q: kin.forward_kinematics(q), T_TRUE))
    return kin, geom


#: Objects at DIFFERENT places — that is what a run of grasps gives you, and what the
#: single-static-target fitters never had.
TARGETS = [np.array(p) for p in [
    (0.24, -0.05, 0.02), (0.30, 0.06, 0.02), (0.21, 0.10, 0.02),
    (0.34, -0.02, 0.02), (0.27, 0.13, 0.02), (0.32, -0.11, 0.02),
    (0.25, 0.02, 0.02), (0.29, -0.08, 0.02),
]]

POSES = [HOME + np.array([d, a, b, 0, 0]) for d, a, b in
         [(-9, 0, 0), (-4, 3, -3), (0, 0, 0), (5, -3, 3), (9, 0, 0),
          (-6, 6, -6), (6, -6, 6), (2, 4, -4)]]


def _grasp_samples(kin, geom, noise_px=0.0, seed=0):
    """One sample per grasp: the pixel it appeared at, and where FK proved it was."""
    rng = np.random.default_rng(seed)
    out = []
    for q, p in zip(POSES, TARGETS):
        T_ee = np.asarray(kin.forward_kinematics(q))
        uv = geom.project(p, T_ee @ T_TRUE)
        if uv is None:
            continue
        uv = np.array(uv, dtype=np.float64)
        if noise_px:
            uv = uv + rng.normal(0.0, noise_px, 2)
        out.append(GraspSample(T_ee, uv, p))
    return out


def _bad_seed(rot_err=(0.20, 0.25, -0.30), scale=1.6):
    T = np.eye(4)
    T[:3, 3] = T_TRUE[:3, 3] * scale
    rv = Rotation.from_matrix(T_TRUE[:3, :3]).as_rotvec() + np.array(rot_err)
    T[:3, :3] = Rotation.from_rotvec(rv).as_matrix()
    return T


def _error(T):
    dt = float(np.linalg.norm(T[:3, 3] - T_TRUE[:3, 3]))
    dr = float(np.degrees(np.linalg.norm(
        Rotation.from_matrix(T[:3, :3].T @ T_TRUE[:3, :3]).as_rotvec())))
    return dt, dr


def test_it_recovers_the_transform_from_a_badly_wrong_seed():
    kin, geom = _rig()
    fit = fit_to_known_points(_grasp_samples(kin, geom), geom, T_seed=_bad_seed())
    assert fit.converged, fit.reason
    dt, dr = _error(fit.T_ee_cam)
    assert dt < 0.005, f"translation off by {dt*1000:.1f}mm"
    assert dr < 1.0, f"rotation off by {dr:.2f}deg"


def test_it_has_no_flat_direction_where_reprojection_does():
    """Seeded with the EXACT truth, it must stay there. fit_reprojection drifts."""
    kin, geom = _rig()
    s = _grasp_samples(kin, geom)
    fit = fit_to_known_points(s, geom, T_seed=T_TRUE)
    dt, dr = _error(fit.T_ee_cam)
    assert dt < 0.001 and dr < 0.2, f"drifted {dt*1000:.1f}mm / {dr:.2f}deg from the truth"


def test_the_older_fitter_really_does_drift_from_the_same_truth():
    """The control: this is the defect the known-point fit exists to avoid."""
    kin, geom = _rig()
    from rax.perception.handeye import HandEyeSample
    static = TARGETS[0]
    views = []
    for q in POSES:
        T_ee = np.asarray(kin.forward_kinematics(q))
        uv = geom.project(static, T_ee @ T_TRUE)
        if uv is not None:
            views.append(HandEyeSample(T_ee, np.array(uv)))
    tip = geom.project(views[0].T_base_ee[:3, 3], views[0].T_base_ee @ T_TRUE)
    fit = fit_reprojection(views, geom, tip_uv=tip, T_seed=T_TRUE)
    dt, dr = _error(fit.T_ee_cam)
    assert dt > 0.002 or dr > 0.5, (
        f"expected the known flat direction; got {dt*1000:.1f}mm / {dr:.2f}deg")


def test_it_survives_realistic_pixel_noise():
    kin, geom = _rig()
    fit = fit_to_known_points(_grasp_samples(kin, geom, noise_px=2.0, seed=7),
                              geom, T_seed=_bad_seed())
    assert fit.converged, fit.reason
    dt, dr = _error(fit.T_ee_cam)
    assert dt < 0.02 and dr < 4.0, f"{dt*1000:.1f}mm / {dr:.2f}deg under 2px noise"


def test_too_few_grasps_is_refused_not_guessed():
    kin, geom = _rig()
    fit = fit_to_known_points(_grasp_samples(kin, geom)[:3], geom, T_seed=_bad_seed())
    assert not fit.converged
    assert "grasp samples" in fit.reason


def test_a_fit_that_cannot_reach_the_bar_is_refused():
    """Garbage pixels must not overwrite a working calibration."""
    kin, geom = _rig()
    s = _grasp_samples(kin, geom)
    rng = np.random.default_rng(3)
    junk = [GraspSample(x.T_base_ee, x.uv + rng.normal(0, 200, 2), x.p_base) for x in s]
    fit = fit_to_known_points(junk, geom, T_seed=_bad_seed())
    assert not fit.converged and "mount NOT changed" in fit.reason


def test_the_residual_is_reported_in_pixels_you_can_judge():
    kin, geom = _rig()
    fit = fit_to_known_points(_grasp_samples(kin, geom), geom, T_seed=_bad_seed())
    assert fit.rms_px < 1.0, fit.rms_px
    assert fit.before["rms_px"] > fit.rms_px
