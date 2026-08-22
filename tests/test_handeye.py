"""Tests for the hand-eye fitters.

The transform from the end effector to the camera is the one calibration everything
else inherits. Get it wrong and every sightline back-projects at the wrong angle, so
every object is localized in the wrong place — and the failure is silent, because the
detections still look fine.

These build a synthetic rig with a KNOWN transform, render what the camera would see,
perturb the seed the way a wrong shipped transform is wrong, and require the fitters to
recover it. They also require the fitters to REFUSE when the data cannot support a fit,
since accepting a bad one overwrites a working calibration.

    python tests/test_handeye.py
    pytest tests/test_handeye.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rax.perception.camera_geometry import (  # noqa: E402
    CameraGeometry,
    EyeInHand,
    intrinsics_from_dict,
    parse_tf,
)
from rax.perception.handeye import (  # noqa: E402
    HandEyeSample,
    fit_consistency,
    fit_reprojection,
    load_hand_eye,
    save_hand_eye,
    tf_string,
)

#: The transform the tests must recover. Realistic for a camera bolted behind the jaws.
T_TRUE = parse_tf("-0.0503,0.0906,-0.1730,-0.2921,1.0770,-2.1688")
TARGET = np.array([0.24, -0.05, 0.02])
HOME = np.array([-14.1, -99.1, 90.8, 33.2, -4.7])


def _kin():
    from lerobot.model.kinematics import RobotKinematics

    from rax.robots.profiles import load_profile

    p = load_profile("so101")
    return RobotKinematics(p.urdf_path, p.ee_frame, list(p.joint_names))


def _rig():
    kin = _kin()
    geom = CameraGeometry(
        intrinsics_from_dict({"fx": 517.0, "fy": 517.0, "cx": 329.5, "cy": 231.4}, 640, 480),
        EyeInHand(lambda q: kin.forward_kinematics(q), T_TRUE))
    return kin, geom


def _samples(kin, geom, poses=None):
    """Render the target's pixel from a spread of poses, using the TRUE transform."""
    if poses is None:
        poses = ([HOME + np.array([d, 0, 0, 0, 0]) for d in (-9, -4.5, 0, 4.5, 9)]
                 + [HOME + np.array([0, 0, 0, d, 0]) for d in (-9, -4, 4, 9)]
                 + [HOME + np.array([0, a, b, 0, 0])
                    for a, b in ((-6, 6), (6, -6), (-4, 10), (4, -10), (-8, 4))])
    out = []
    for q in poses:
        T_ee = np.asarray(kin.forward_kinematics(q))
        uv = geom.project(TARGET, T_ee @ T_TRUE)
        if uv is not None:
            out.append(HandEyeSample(T_ee, np.array(uv)))
    return out


def _tip_uv(geom, sample):
    T = sample.T_base_ee
    return geom.project(T[:3, 3], T @ T_TRUE)


def _bad_seed(rot_err=(0.2, 0.25, -0.3), scale=1.6):
    """A seed wrong the way the shipped transform was wrong: badly rotated, and with
    the camera placed much further from the fingertip than it really is."""
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


def test_reprojection_fit_recovers_the_transform():
    kin, geom = _rig()
    s = _samples(kin, geom)
    assert len(s) >= 10, "the synthetic sweep should keep the target in view"
    fit = fit_reprojection(s, geom, tip_uv=_tip_uv(geom, s[0]), T_seed=_bad_seed(),
                           cam_tip_m=float(np.linalg.norm(T_TRUE[:3, 3])))
    assert fit.converged, fit.reason
    # The fingertip anchor is a hard geometric constraint and must lock on.
    assert fit.tip_gap_px < 1.0, f"fingertip still {fit.tip_gap_px:.1f} px out"
    assert fit.rms_px < 5.0, f"reprojection RMS {fit.rms_px:.1f} px"
    assert fit.before["tip_gap_px"] > 50.0, "the seed should have been badly wrong"
    dt, dr = _error(fit.T_ee_cam)
    assert dt < 0.03 and dr < 6.0, f"recovered {dt * 1000:.0f} mm / {dr:.1f} deg off"


def test_consistency_fit_makes_the_views_agree():
    """Its objective is that one object maps to one place from every viewpoint, so
    that spread is what must collapse."""
    kin, geom = _rig()
    s = _samples(kin, geom)
    fit = fit_consistency(s, geom, tip_uv=_tip_uv(geom, s[0]), T_seed=_bad_seed(),
                          z_plane=0.02)
    assert fit.converged, fit.reason
    assert fit.spread_m < 0.01, f"views still disagree by {fit.spread_m * 100:.1f} cm"
    assert fit.spread_m < fit.before["spread_m"], "the fit made agreement worse"


def test_reprojection_objective_has_a_flat_direction():
    """Pin a real limitation: a GOOD transform does not stay put, and the fit reports
    excellent numbers while moving it.

    Seeded with the exact truth and given perfect noise-free pixels, the fit still
    settles ~14 mm and ~3 deg away — at 0.5 px reprojection RMS and 0.01 px fingertip
    error. That is not a solver bug (this was verified bit-identical to the original
    implementation); it is gauge freedom in the objective. With ONE point target, the
    camera can slide along the viewing direction and the target follow it, reproducing
    every observed pixel. The parallax between poses is what would pin it down, and a
    small pose spread leaves that direction nearly unobserved.

    The consequence for an operator: re-running the reprojection calibration on an
    already-good robot can move the transform while reporting a great fit. Prefer
    fit_consistency to CHECK a calibration, and judge fit_reprojection by the fingertip
    gap, which is a hard geometric constraint rather than a fitted one.
    """
    kin, geom = _rig()
    s = _samples(kin, geom)
    fit = fit_reprojection(s, geom, tip_uv=_tip_uv(geom, s[0]), T_seed=T_TRUE,
                           cam_tip_m=float(np.linalg.norm(T_TRUE[:3, 3])))
    assert fit.converged
    # The fingertip anchor still nails it, and the reprojection is excellent...
    assert fit.tip_gap_px < 1.0 and fit.rms_px < 5.0
    # ...yet the transform itself has moved. If this ever tightens below a millimetre,
    # the flat direction has been closed and this test should be simplified.
    dt, dr = _error(fit.T_ee_cam)
    assert dt > 0.001, "the flat direction seems gone — good, revisit this test"
    assert dt < 0.05 and dr < 10.0, f"drift grew to {dt * 1000:.0f} mm / {dr:.1f} deg"


def test_consistency_fit_is_stable_from_a_good_seed():
    """The consistency objective does not have that freedom in the same way: it is
    scored on agreement between views, which a wrong transform cannot fake."""
    kin, geom = _rig()
    s = _samples(kin, geom)
    fit = fit_consistency(s, geom, tip_uv=_tip_uv(geom, s[0]), T_seed=T_TRUE, z_plane=0.02)
    assert fit.converged
    assert fit.spread_m < 0.005, f"a true transform should agree, got {fit.spread_m*100:.2f} cm"


def test_too_few_views_is_refused_not_guessed():
    """Accepting a fit from three views would overwrite a working calibration with
    something the data cannot support."""
    kin, geom = _rig()
    s = _samples(kin, geom)
    tip = _tip_uv(geom, s[0])
    r = fit_reprojection(s[:3], geom, tip_uv=tip, T_seed=_bad_seed())
    assert not r.converged and "views" in r.reason
    # and it hands back the seed unchanged, so a caller that ignores `converged`
    # still does not corrupt the transform
    assert np.allclose(r.T_ee_cam, _bad_seed())
    c = fit_consistency(s[:2], geom, tip_uv=tip, T_seed=_bad_seed())
    assert not c.converged and "views" in c.reason


def test_fit_is_serializable_and_round_trips(tmp_path=None):
    kin, geom = _rig()
    s = _samples(kin, geom)
    fit = fit_reprojection(s, geom, tip_uv=_tip_uv(geom, s[0]), T_seed=_bad_seed(),
                           cam_tip_m=float(np.linalg.norm(T_TRUE[:3, 3])))
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "handeye_tf.json")
        written = save_hand_eye(path, fit)
        assert written["tf"] == fit.tf and written["views"] == fit.n_views
        loaded = load_hand_eye(path)
        assert loaded["tf"] == fit.tf
        # the string must reconstruct the same transform
        assert np.allclose(parse_tf(loaded["tf"]), fit.T_ee_cam, atol=1e-4)
        assert load_hand_eye(str(pathlib.Path(d) / "nope.json")) is None


def test_tf_string_round_trip():
    assert np.allclose(parse_tf(tf_string(T_TRUE)), T_TRUE, atol=1e-4)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
