"""``Plane`` — the measured table surface, as ``z = a*x + b*y + c`` in base frame.

Treating the table as the robot's own z=0 plane is wrong in two ways that ADD UP as the
arm reaches out: the table is not exactly parallel to the base plane, and the arm SAGS
under its own weight by more at full extension than near the base. Either alone tilts
the effective floor; together they are why a grasp clears fine at r=18 cm and scrapes
at r=34 cm.

Those two do not have to be separated. Touch the table and write down the FK z where
contact happens: that number already contains the table height, the tilt AND the sag at
that reach, because it is measured in the same coordinates the arm is commanded in.
Probe several points, fit a plane, and everything that used to assume 0 reads off it.

This module owns the plane and the fit. The probing motion stays with the arm, since
that needs joint control and load sensing — :func:`fit_plane` takes the touch points
once they have been collected.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np

__all__ = ["Plane", "PlaneFit", "fit_plane", "DEFAULT_PLANE_C"]

# What someone measured by hand before the calibration existed ("table contact is
# z=-0.022, sag included"). a=b=0 means "flat and level" until a probe says otherwise.
DEFAULT_PLANE_C = -0.022

# A flat table should fit far better than this; above it, a probe point probably caught
# an object rather than the surface.
FIT_RMS_WARN_M = 0.004


@dataclass
class Plane:
    """A tilted plane in base frame. Mutable: a calibration re-fits it in place, so
    every holder of the reference sees the new surface."""

    a: float = 0.0
    b: float = 0.0
    c: float = DEFAULT_PLANE_C

    def z(self, x: float, y: float) -> float:
        """Table height in base z at (x, y)."""
        return float(self.a * float(x) + self.b * float(y) + self.c)

    __call__ = z

    @property
    def tilt_deg(self) -> float:
        return math.degrees(math.atan(math.hypot(self.a, self.b)))

    def as_list(self) -> list[float]:
        return [float(self.a), float(self.b), float(self.c)]

    def set(self, a: float, b: float, c: float) -> "Plane":
        self.a, self.b, self.c = float(a), float(b), float(c)
        return self

    def describe(self) -> str:
        return f"z = {self.a:+.4f}x {self.b:+.4f}y {self.c:+.4f}"

    # --- persistence ---------------------------------------------------------
    def load(self, path: str) -> dict | None:
        """Load a fitted plane in place. Returns the file's dict, or None if there
        is no usable calibration — the caller decides what to tell the operator."""
        try:
            with open(path) as f:
                d = json.load(f)
            self.set(float(d["a"]), float(d["b"]), float(d["c"]))
            return d
        except FileNotFoundError:
            return None
        except Exception as e:
            raise ValueError(f"bad {os.path.basename(path)}: {e}") from e

    def save(self, path: str, **extra) -> dict:
        d = {"a": float(self.a), "b": float(self.b), "c": float(self.c), **extra}
        with open(path, "w") as f:
            json.dump(d, f, indent=1)
        return d


@dataclass
class PlaneFit:
    """The outcome of a least-squares plane fit through the touch points."""

    plane: Plane
    rms_m: float
    points: list[tuple[float, float, float]]

    @property
    def tilt_deg(self) -> float:
        return self.plane.tilt_deg

    @property
    def suspicious(self) -> bool:
        return self.rms_m > FIT_RMS_WARN_M

    def to_dict(self) -> dict:
        return {
            "a": float(self.plane.a), "b": float(self.plane.b), "c": float(self.plane.c),
            "tilt_deg": round(self.tilt_deg, 3),
            "rms_mm": round(self.rms_m * 1000, 2),
            "points": [[round(v, 4) for v in p] for p in self.points],
            "fitted": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def fit_plane(points) -> PlaneFit:
    """Least-squares ``z = a*x + b*y + c`` through the measured contact points.

    Needs at least three: two points do not define a tilt, and accepting them would
    produce a confident plane through a line.
    """
    pts = [(float(x), float(y), float(z)) for x, y, z in points]
    if len(pts) < 3:
        raise ValueError(f"only {len(pts)} touch points — need 3 to fit a plane")
    A = np.array([[p[0], p[1], 1.0] for p in pts])
    zz = np.array([p[2] for p in pts])
    (a, b, c), *_ = np.linalg.lstsq(A, zz, rcond=None)
    resid = zz - A @ np.array([a, b, c])
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return PlaneFit(Plane(float(a), float(b), float(c)), rms, pts)
