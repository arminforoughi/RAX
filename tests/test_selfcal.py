"""Tests for solving the localization correction instead of dialling it in.

Two things must hold for this to be worth using. It has to recover a known error
accurately — better than a human turning a dial — and, just as importantly, it has to
REFUSE when its three-parameter model does not describe the rig. A fudge factor fitted
on top of a real geometric bug is how the bug survives, so a confident wrong answer here
is worse than no answer.

    python tests/test_selfcal.py
    pytest tests/test_selfcal.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manipulation.approach import ApproachConfig  # noqa: E402
from perception.selfcal import (  # noqa: E402
    LocalizationModel, LocalizationSample, apply_to_config, diagnose, fit_localization)


def _grid(radii=(0.16, 0.22, 0.28, 0.34, 0.40), bearings=(-40, -20, 0, 20, 40)):
    """Where the arm would present a held object: a spread of radii and bearings."""
    return [np.array([r * math.cos(math.radians(b)), r * math.sin(math.radians(b))])
            for r in radii for b in bearings]


def _samples(distort, noise_m=0.0, seed=0):
    """Ground truth from FK, observation from a distorted 'camera'."""
    rng = np.random.default_rng(seed)
    out = []
    for xy_true in _grid():
        obs = np.asarray(distort(xy_true), dtype=np.float64)
        if noise_m:
            obs = obs + rng.normal(0, noise_m, 2)
        r = float(np.hypot(*xy_true))
        out.append(LocalizationSample(xy_true, obs,
                                      off_axis_deg=math.degrees(math.atan2(r, 0.35))))
    return out


def _polar_distort(scale=1.0, offset=0.0, bearing_deg=0.0):
    """The inverse of the model: given truth, produce what a bad rig would report."""
    def f(xy_true):
        r = float(np.hypot(*xy_true))
        th = math.atan2(xy_true[1], xy_true[0]) - math.radians(bearing_deg)
        r_obs = (r - offset) / scale
        return np.array([r_obs * math.cos(th), r_obs * math.sin(th)])
    return f


def test_recovers_a_known_range_and_bearing_error():
    """The headline claim: solve the knobs, do not guess them."""
    truth = LocalizationModel(range_scale=0.92, push_out_m=0.035, bearing_offset_deg=4.5)
    fit = fit_localization(_samples(_polar_distort(0.92, 0.035, 4.5)))
    assert fit.trustworthy, fit.summary()
    assert abs(fit.model.range_scale - truth.range_scale) < 0.01
    assert abs(fit.model.push_out_m - truth.push_out_m) < 0.002
    assert abs(fit.model.bearing_offset_deg - truth.bearing_offset_deg) < 0.5
    assert fit.rms_after_m < 1e-6, "a noise-free fit should be essentially exact"
    assert fit.improvement > 100


def test_survives_realistic_detector_noise():
    """5 mm of localization jitter is normal; the fit must still beat a hand-tuned dial."""
    fit = fit_localization(_samples(_polar_distort(0.92, 0.035, 4.5), noise_m=0.005, seed=7))
    assert fit.trustworthy, fit.summary()
    assert abs(fit.model.range_scale - 0.92) < 0.03
    assert abs(fit.model.push_out_m - 0.035) < 0.01
    assert abs(fit.model.bearing_offset_deg - 4.5) < 2.0
    # It should still remove most of the error, not merely some.
    assert fit.improvement > 3.0, fit.summary()


def test_an_already_good_rig_is_left_alone():
    """Calibrating a correct rig must not invent a correction."""
    fit = fit_localization(_samples(lambda xy: xy))
    assert abs(fit.model.range_scale - 1.0) < 0.005
    assert abs(fit.model.push_out_m) < 0.002
    assert abs(fit.model.bearing_offset_deg) < 0.2


def test_refuses_when_the_model_does_not_describe_the_rig():
    """A hand-eye ROTATION error does not reduce to range and bearing. The fit must say
    so rather than hand back three confident numbers."""
    def bad_rotation(xy_true):
        # error that grows with bearing in a way no radial model can absorb
        r = float(np.hypot(*xy_true))
        th = math.atan2(xy_true[1], xy_true[0])
        r_obs = r * (1.0 + 0.35 * math.sin(3.0 * th))
        return np.array([r_obs * math.cos(th), r_obs * math.sin(th)])

    fit = fit_localization(_samples(bad_rotation))
    assert not fit.trustworthy
    assert any("NOT a simple" in w for w in fit.warnings), fit.warnings


def test_refuses_too_few_samples():
    fit = fit_localization(_samples(_polar_distort(0.9, 0.03))[:3])
    assert not fit.trustworthy and "samples" in fit.warnings[0]


def test_warns_when_radii_cannot_separate_scale_from_offset():
    """All samples at one radius: any scale trades against any offset. Say so."""
    ring = [np.array([0.30 * math.cos(math.radians(b)), 0.30 * math.sin(math.radians(b))])
            for b in range(-40, 41, 10)]
    d = _polar_distort(0.92, 0.035)
    samples = [LocalizationSample(xy, d(xy)) for xy in ring]
    fit = fit_localization(samples)
    assert any("cannot be separated" in w for w in fit.warnings), fit.warnings
    assert fit.model.range_scale == 1.0, "it must not pretend to have solved the scale"


def test_diagnoses_the_axial_depth_signature():
    """The real bug found earlier: radial error growing with off-axis angle. It must be
    NAMED, not absorbed into push-out."""
    samples = []
    for xy_true in _grid():
        r = float(np.hypot(*xy_true))
        off = math.degrees(math.atan2(r, 0.35))            # off-axis angle from 35cm up
        obs = xy_true * math.cos(math.radians(off))        # the cos(angle) inward pull
        samples.append(LocalizationSample(xy_true, obs, off_axis_deg=off))
    notes = diagnose(samples)
    assert any("off-axis" in n for n in notes), notes
    assert any("axial_depth" in n for n in notes), notes


def test_diagnoses_a_pure_scale_error():
    samples = _samples(_polar_distort(scale=0.85))
    notes = diagnose(samples)
    assert any("scales with radius" in n for n in notes), notes


def test_diagnose_stays_quiet_on_a_clean_rig():
    notes = diagnose(_samples(lambda xy: xy, noise_m=0.002, seed=3))
    assert any("no known systematic signature" in n for n in notes), notes


def test_applying_to_config_writes_the_wire_knobs():
    cfg = ApproachConfig()
    fit = fit_localization(_samples(_polar_distort(0.92, 0.035, 4.5)))
    assert apply_to_config(fit, cfg)
    assert abs(cfg.get_knob("range_scale") - 0.92) < 0.01
    assert abs(cfg.get_knob("push_out_cm") - 3.5) < 0.2
    assert abs(cfg.get_knob("bearing_deg") - 4.5) < 0.5


def test_untrustworthy_fits_are_not_installed():
    """The failure this whole module exists to prevent."""
    cfg = ApproachConfig()
    baseline = cfg.get_knob("range_scale")
    bad = fit_localization(_samples(_polar_distort(0.9, 0.03))[:3])
    assert not apply_to_config(bad, cfg)
    assert cfg.get_knob("range_scale") == baseline, "an untrusted fit was installed"
    assert apply_to_config(bad, cfg, force=True), "force must still be possible"


def test_model_apply_round_trips():
    m = LocalizationModel(0.92, 0.035, 4.5)
    d = _polar_distort(0.92, 0.035, 4.5)
    for xy in _grid():
        assert np.allclose(m.apply(d(xy)), xy, atol=1e-9)


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
