"""Joint-space trajectory generation — the smooth-transit maths, without the I/O.

A transit move that steps a fixed number of degrees per tick is just a velocity cap:
it still commands an abrupt start and an abrupt stop, and on an arm with any inertia
that shows up as a visible jump at both ends (and as camera shake, which matters when
the camera is bolted to the hand).

A quintic profile fixes it by ramping velocity AND acceleration to zero at both ends:

    s(u) = 10u^3 - 15u^4 + 6u^5,  u = t/T

with peak velocity ``1.875/T`` and peak acceleration ``5.78/T^2`` in normalized units.
Solving both against the per-joint limits gives the shortest duration that violates
neither, for every joint at once.

Only the maths lives here — no robot, no clock, no sleeping — so it is testable and
shared. The caller owns the send loop and its real-time pacing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MotionLimits", "quintic_duration", "quintic_waypoints", "rate_limit"]

# Peak velocity and acceleration of the quintic in normalized units (T = 1).
_QUINTIC_PEAK_VEL = 1.875
_QUINTIC_PEAK_ACC = 5.78

# Never shorter than this, so a tiny correction still gets a few command ticks rather
# than becoming a single step change.
MIN_DURATION_S = 0.12

# Joints moving less than this are treated as stationary when sizing the move.
STATIONARY_DEG = 0.001


@dataclass(frozen=True)
class MotionLimits:
    """Per-joint velocity and acceleration ceilings for transit moves.

    Per-joint rather than global because the joints do not carry equal inertia: on a
    tabletop arm the base swings the whole robot and causes the visible jump, so it
    wants the gentlest limits, while the wrist can move several times faster.
    """

    vmax_dps: np.ndarray
    amax_dps2: np.ndarray
    dt_s: float = 0.02          # 50 Hz command rate

    @classmethod
    def from_profile(cls, profile) -> "MotionLimits":
        return cls(np.asarray(profile.goto_vmax_dps, dtype=np.float64),
                   np.asarray(profile.goto_amax_dps2, dtype=np.float64),
                   float(profile.goto_dt_s))

    def scaled(self, speed: float) -> "MotionLimits":
        s = max(float(speed), 1e-6)
        return MotionLimits(self.vmax_dps * s, self.amax_dps2 * s, self.dt_s)


def quintic_duration(q0, q1, limits: MotionLimits) -> float:
    """Shortest duration (s) for which no joint exceeds its velocity or acceleration
    limit. Both constraints are solved per joint and the worst one wins."""
    delta = np.asarray(q1, dtype=np.float64) - np.asarray(q0, dtype=np.float64)
    abs_d = np.abs(delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_v = np.where(abs_d > STATIONARY_DEG,
                       _QUINTIC_PEAK_VEL * abs_d / np.maximum(limits.vmax_dps, 1e-6), 0.0)
        t_a = np.where(abs_d > STATIONARY_DEG,
                       np.sqrt(_QUINTIC_PEAK_ACC * abs_d / np.maximum(limits.amax_dps2, 1e-6)),
                       0.0)
    return max(float(np.max(np.maximum(t_v, t_a))), MIN_DURATION_S)


def quintic_waypoints(q0, q1, limits: MotionLimits) -> tuple[list[np.ndarray], float]:
    """The joint vectors to command, one per tick, plus the total duration.

    The last tick is clamped to ``t = T``, where ``s(1) = 1`` exactly, so the move ends
    on the commanded target rather than wherever the tick grid happened to land. It
    arrives via ``q0 + (q1 - q0)``, so the final waypoint matches ``q1`` to within
    floating-point rounding (measured sub-ULP) rather than bitwise.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    delta = np.asarray(q1, dtype=np.float64) - q0
    T = quintic_duration(q0, q1, limits)
    n = int(np.ceil(T / limits.dt_s))
    out = []
    for k in range(n + 1):
        u = min(k * limits.dt_s, T) / T
        s = 10.0 * u ** 3 - 15.0 * u ** 4 + 6.0 * u ** 5
        out.append(q0 + delta * s)
    return out, T


def rate_limit(q_from, q_to, max_step_deg) -> np.ndarray:
    """Clamp a commanded jump so no joint moves more than ``max_step_deg`` at once.

    The last line of defence for anything that commands a pose directly rather than
    going through a profile — a solver that returns a far-away branch should crawl
    there, not snap.
    """
    q_from = np.asarray(q_from, dtype=np.float64)
    d = np.asarray(q_to, dtype=np.float64) - q_from
    return q_from + np.clip(d, -abs(float(max_step_deg)), abs(float(max_step_deg)))
