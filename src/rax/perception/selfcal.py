"""Solving the localization correction instead of dialling it in.

``push_out_cm``, ``range_scale``, ``bearing_deg`` and the approach trims were not
preferences. Each one is a human turning a dial until grasps stopped missing — a
compensation for a *systematic* error nobody had measured. That is why they had to be
retuned whenever the mount shifted, and why an autotuner was needed to walk them around
in the first place.

They can be computed instead, because **the arm can generate its own ground truth**.
When the gripper is holding an object, forward kinematics says exactly where that object
is, to the accuracy of the joint encoders. Present it to the camera at a spread of
radii and bearings, localize it the normal way, and the difference between "where the
camera says it is" and "where it demonstrably is" is the error — measured directly, at
every place you care about, with no ruler and no human judgement.

Fit a model to those differences and the knobs fall out with a residual attached, so you
know whether they are trustworthy rather than hoping.

THE MODEL. In polar coordinates about the base, a miscalibrated eye-in-hand rig gets
range and bearing wrong nearly independently:

    r_true     = range_scale * r_observed + push_out_m
    theta_true = theta_observed + bearing_offset_deg

That is exactly the three knobs, which is not a coincidence — they were invented one at
a time to patch these three effects. ``range_scale`` catches a proportional range error
(a wrong assumed object size, or an intrinsics scale error), ``push_out_m`` catches a
constant radial offset (the camera sitting behind the fingertips), and
``bearing_offset_deg`` catches a hand-eye yaw error, which rotates the whole map.

WHAT THIS DELIBERATELY WILL NOT DO. If the residual after fitting still has structure,
the model is wrong for this rig and the knobs are papering over something else — most
likely a hand-eye rotation error, which does not reduce to three numbers.
:func:`diagnose` says so instead of returning a confident fit, because a fudge factor
tuned on top of a real geometric bug is how the bug survives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "LocalizationSample", "LocalizationModel", "LocalizationFit", "Signature",
    "fit_localization", "diagnose", "apply_to_config",
]

#: Below this many samples the fit is not worth trusting: three unknowns need spread in
#: both radius and bearing, not just three points.
MIN_SAMPLES = 6

#: Residual above this (metres, RMS) means the three-parameter model does not describe
#: this rig, whatever the fitted numbers say.
RESIDUAL_STRUCTURED_M = 0.015


@dataclass(frozen=True)
class LocalizationSample:
    """One paired observation: where the object really was, and where localization put it.

    ``xy_true`` normally comes from forward kinematics while the gripper holds the
    object, which is the whole point — it is ground truth the robot produces itself.
    """

    xy_true: np.ndarray
    xy_observed: np.ndarray
    off_axis_deg: float = 0.0     # angle of the object from the camera's optical axis
    label: str = ""

    @property
    def r_true(self) -> float:
        return float(np.hypot(*self.xy_true))

    @property
    def r_observed(self) -> float:
        return float(np.hypot(*self.xy_observed))

    @property
    def bearing_true_deg(self) -> float:
        return math.degrees(math.atan2(self.xy_true[1], self.xy_true[0]))

    @property
    def bearing_observed_deg(self) -> float:
        return math.degrees(math.atan2(self.xy_observed[1], self.xy_observed[0]))


@dataclass
class LocalizationModel:
    """The polar correction applied to every localization."""

    range_scale: float = 1.0
    push_out_m: float = 0.0
    bearing_offset_deg: float = 0.0

    def apply(self, xy) -> np.ndarray:
        """Correct one observed (x, y)."""
        xy = np.asarray(xy, dtype=np.float64)
        r = float(np.hypot(xy[0], xy[1]))
        if r < 1e-9:
            return xy.copy()
        th = math.atan2(xy[1], xy[0]) + math.radians(self.bearing_offset_deg)
        r2 = self.range_scale * r + self.push_out_m
        return np.array([r2 * math.cos(th), r2 * math.sin(th)])

    def unapply(self, xy) -> np.ndarray:
        """The exact inverse of :meth:`apply` — recover the raw observation.

        Needed to fit from samples gathered DURING NORMAL OPERATION rather than during a
        dedicated calibration. Ordinary picks are localized through whatever corrections
        are dialled in at the time, so a sample recorded from one is already partly
        corrected. Fitting on that measures the error that is LEFT, and applying the
        result on top of the corrections already in place double-counts them: the fit
        must be absolute — "this rig's range reads 6% long" — not a delta on the current
        dial positions.

        Record the correction state alongside each observation, undo it with this, and
        every fit is against the raw rig no matter what was set when the sample was
        taken. That is what lets the calibration converge over many runs instead of
        oscillating as it chases its own last answer.
        """
        xy = np.asarray(xy, dtype=np.float64)
        r = float(np.hypot(xy[0], xy[1]))
        if r < 1e-9:
            return xy.copy()
        th = math.atan2(xy[1], xy[0]) - math.radians(self.bearing_offset_deg)
        scale = self.range_scale if abs(self.range_scale) > 1e-9 else 1.0
        r2 = max((r - self.push_out_m) / scale, 0.0)
        return np.array([r2 * math.cos(th), r2 * math.sin(th)])

    def describe(self) -> str:
        return (f"range x{self.range_scale:.4f} {self.push_out_m * 100:+.2f}cm, "
                f"bearing {self.bearing_offset_deg:+.2f}deg")


@dataclass
class LocalizationFit:
    """A fitted model plus everything needed to decide whether to believe it."""

    model: LocalizationModel
    n: int
    rms_before_m: float
    rms_after_m: float
    max_after_m: float
    residual_by_radius: list[tuple[float, float]]      # (r_true, residual) per sample
    trustworthy: bool
    warnings: list[str]

    @property
    def improvement(self) -> float:
        """Factor by which the error shrank. Below ~1.5 the fit is not earning its keep."""
        return (self.rms_before_m / self.rms_after_m) if self.rms_after_m > 1e-9 else float("inf")

    def summary(self) -> str:
        lines = [
            f"localization self-calibration over {self.n} samples",
            f"  model:  {self.model.describe()}",
            f"  error:  {self.rms_before_m * 1000:.1f}mm -> {self.rms_after_m * 1000:.1f}mm RMS "
            f"({self.improvement:.1f}x), worst {self.max_after_m * 1000:.1f}mm",
            f"  verdict: {'USE IT' if self.trustworthy else 'DO NOT APPLY'}",
        ]
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def _rms(errs) -> float:
    e = np.asarray(errs, dtype=np.float64)
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def _circular_mean_deg(angles_deg) -> float:
    a = np.radians(np.asarray(angles_deg, dtype=np.float64))
    return float(math.degrees(math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a)))))


def fit_localization(samples) -> LocalizationFit:
    """Solve range_scale, push_out and bearing_offset from paired observations.

    Bearing separates cleanly from range — a rotation about the base does not change
    any radius — so it is solved first as a circular mean of the per-sample bearing
    error, then the radial model falls out of a linear least squares. Solving them
    jointly buys nothing and makes the answer harder to sanity-check.
    """
    samples = list(samples)
    n = len(samples)
    warnings: list[str] = []
    if n < MIN_SAMPLES:
        return LocalizationFit(LocalizationModel(), n, float("nan"), float("nan"),
                               float("nan"), [], False,
                               [f"only {n} samples (need {MIN_SAMPLES})"])

    before = [float(np.linalg.norm(s.xy_true - s.xy_observed)) for s in samples]

    # --- bearing: a pure rotation about the base ---------------------------------
    d_theta = [((s.bearing_true_deg - s.bearing_observed_deg + 180.0) % 360.0) - 180.0
               for s in samples]
    bearing = _circular_mean_deg(d_theta)

    # --- range: r_true = scale * r_obs + offset ----------------------------------
    r_obs = np.array([s.r_observed for s in samples])
    r_true = np.array([s.r_true for s in samples])
    spread = float(r_obs.max() - r_obs.min())
    if spread < 0.05:
        # Without radial spread, scale and offset are not separable: any scale can be
        # traded against an offset to fit the same points. Solve offset only and say so.
        warnings.append(f"radii span only {spread * 100:.0f}cm — scale and offset cannot "
                        f"be separated; solved offset only. Sample from 15cm to 40cm.")
        scale, offset = 1.0, float(np.mean(r_true - r_obs))
    else:
        A = np.stack([r_obs, np.ones_like(r_obs)], axis=1)
        (scale, offset), *_ = np.linalg.lstsq(A, r_true, rcond=None)
        scale, offset = float(scale), float(offset)

    model = LocalizationModel(scale, offset, bearing)
    after = [float(np.linalg.norm(s.xy_true - model.apply(s.xy_observed))) for s in samples]

    rms_b, rms_a = _rms(before), _rms(after)
    max_a = float(np.max(after))
    by_radius = sorted((s.r_true, e) for s, e in zip(samples, after))

    # --- is the model actually right for this rig? -------------------------------
    if rms_a > RESIDUAL_STRUCTURED_M:
        warnings.append(
            f"{rms_a * 1000:.0f}mm residual after fitting — the error is NOT a simple "
            f"range/bearing offset. Check the hand-eye rotation before trusting these "
            f"numbers; a fudge tuned on top of a geometry bug hides the bug.")
    if rms_a > rms_b:
        warnings.append("the fit made things worse — data is probably inconsistent")
    if not (0.5 < scale < 2.0):
        warnings.append(f"range_scale {scale:.2f} is implausible; suspect the assumed "
                        f"object size or the intrinsics")
    if abs(bearing) > 30.0:
        warnings.append(f"bearing offset {bearing:.0f}deg is large — that is a hand-eye "
                        f"yaw error, better fixed in the transform than compensated here")

    bearings = [s.bearing_true_deg for s in samples]
    if max(bearings) - min(bearings) < 30.0:
        warnings.append(f"bearings span only {max(bearings) - min(bearings):.0f}deg — "
                        f"sample across the workspace or the rotation is poorly observed")

    # A KNOWN BUG SIGNATURE VETOES THE FIT, however well it fits.
    #
    # This is the module's whole reason for existing. Measured on the real rig, the
    # axial-depth projection error is absorbed ~80% by these three knobs: RMS drops
    # 40mm -> 8mm, a 5x improvement that any operator would accept. But the fitted
    # range_scale comes out 1.32, which is not a range scale — it is a lie about the
    # object's size, standing in for a projection that is simply wrong. Install it and
    # the knobs now depend on the camera height, the object, and where on the table it
    # sits, so they need retuning forever. That is exactly the trap these knobs were.
    found = _signatures(samples)
    blocking = [f for f in found if f.blocking]
    if blocking:
        warnings.append("a known geometric bug explains this error — fix that first, "
                        "then re-run; these numbers would only hide it")
    warnings.extend(f.note for f in found)

    trustworthy = (rms_a <= RESIDUAL_STRUCTURED_M and rms_a < rms_b
                   and 0.5 < scale < 2.0 and not blocking)
    return LocalizationFit(model, n, rms_b, rms_a, max_a, by_radius, trustworthy, warnings)


@dataclass(frozen=True)
class Signature:
    """A recognised error pattern.

    ``blocking`` separates "this model cannot express your error" from "this model
    exists precisely to correct that". A proportional range error IS what range_scale
    is for, so naming it is useful but it must not veto its own fix. An error that
    varies with off-axis angle cannot be written as any range/bearing correction, so
    fitting one only hides it — that blocks.
    """

    note: str
    blocking: bool


def _signatures(samples) -> list[Signature]:
    """The structured form of :func:`diagnose`."""
    samples = list(samples)
    found: list[Signature] = []
    if len(samples) < MIN_SAMPLES:
        return found

    dr = np.array([s.r_true - s.r_observed for s in samples])
    r = np.array([s.r_true for s in samples])
    off = np.array([abs(s.off_axis_deg) for s in samples])

    if np.ptp(off) > 5.0 and np.std(dr) > 1e-6:
        c = float(np.corrcoef(off, dr)[0, 1])
        if c > 0.6:
            found.append(Signature(
                f"radial error grows with off-axis angle (corr {c:+.2f}): the signature of "
                f"placing an axial DEPTH along the sightline. Fix the projection "
                f"(ApparentSizeLocalizer.axial_depth) rather than adding push-out.",
                blocking=True))
    if np.ptp(r) > 0.05 and np.std(dr) > 1e-6:
        c = float(np.corrcoef(r, dr)[0, 1])
        if abs(c) > 0.7:
            found.append(Signature(
                f"radial error scales with radius (corr {c:+.2f}): a proportional range "
                f"error — assumed object size or focal length. range_scale corrects this "
                f"exactly, but the underlying number is worth fixing at the source.",
                blocking=False))
    return found


def diagnose(samples) -> list[str]:
    """Look for known error SIGNATURES, so a systematic bug is named rather than absorbed.

    The point of a self-calibration is not only to produce numbers — it is to notice
    when the numbers should not exist. Two signatures are checked:

    * **Radial bias growing with off-axis angle.** Apparent-size ranging yields axial
      depth, but placing the point that far along the sightline puts an off-axis object
      too near by ``cos(angle)``. That looks like a push-out fudge, and compensating it
      with a constant is wrong at every radius except the one it was tuned at. Blocking.
    * **Radial error proportional to radius.** A pure scale error — an assumed object
      size or a focal length that is off. ``range_scale`` corrects this exactly, so it
      is reported but does not block.
    """
    samples = list(samples)
    if len(samples) < MIN_SAMPLES:
        return [f"only {len(samples)} samples — not enough to diagnose"]
    found = _signatures(samples)
    return [f.note for f in found] or ["no known systematic signature; the residual "
                                       "looks like noise"]


def apply_to_config(fit: LocalizationFit, cfg, *, force: bool = False) -> bool:
    """Write a trustworthy fit into an :class:`ApproachConfig`. Returns whether it did.

    Refuses an untrustworthy fit unless forced: silently installing numbers that do not
    describe the rig is exactly how the dial-turning started.
    """
    if not fit.trustworthy and not force:
        return False
    cfg.set_knob("range_scale", fit.model.range_scale)
    cfg.set_knob("push_out_cm", fit.model.push_out_m * 100.0)
    cfg.set_knob("bearing_deg", fit.model.bearing_offset_deg)
    return True
