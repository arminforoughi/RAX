"""``ObjectTrack`` — one tagged object and its live point cloud.

A track is the unit of memory in :class:`perception.depth_cloud.CloudTracker`. It
carries a *stable* integer tag (so an agent can say "grasp tag 3" or "place it on
tag 1"), the latest 2D detection, and the most recent point cloud — expressed in
the robot **base frame** so manipulation can consume it directly.

Clouds are summarised lazily into the handful of numbers the gaze engine needs:
the centroid (approach target) and the top-surface point (place target).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


@dataclass
class ObjectTrack:
    """A persistent, tagged object with its most recent base-frame point cloud."""

    tag: int
    label: str
    box: tuple[float, float, float, float]
    confidence: float = 0.0
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float64))
    last_seen_t: float = field(default_factory=time.time)
    last_cloud_t: float = 0.0
    n_cloud_updates: int = 0
    misses: int = 0  # consecutive detection ticks without a match
    range_cam_m: float = float("nan")  # median stereo depth in mask (camera frame)

    # cached centroid, invalidated whenever ``points`` is replaced
    _centroid: np.ndarray | None = field(default=None, repr=False)

    @property
    def center_px(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def has_cloud(self) -> bool:
        return self.points.shape[0] > 0

    @property
    def centroid(self) -> np.ndarray:
        """Base-frame centroid of the cloud (the approach/grasp target)."""
        if self._centroid is None:
            self._centroid = (
                self.points.mean(axis=0) if self.has_cloud else np.full(3, np.nan)
            )
        return self._centroid

    def top_point(self, up: np.ndarray, *, quantile: float = 0.85) -> np.ndarray:
        """Base-frame point on the object's top face along ``up`` (the place target).

        Horizontal position = cloud centroid; height = robust max along ``up`` —
        i.e. "the middle of the lid". ``up`` is the world-up direction in base frame.
        """
        if not self.has_cloud:
            return np.full(3, np.nan)
        u = _unit(up)
        proj = self.points @ u
        thresh = np.quantile(proj, quantile)
        top = self.points[proj >= thresh]
        top = top if top.shape[0] else self.points
        c = self.points.mean(axis=0)
        return c + u * float((top.mean(axis=0) - c) @ u)

    def extent_along(self, up: np.ndarray) -> float:
        """Cloud thickness (m) along ``up`` — used to clear the object when placing."""
        if not self.has_cloud:
            return 0.0
        proj = self.points @ _unit(up)
        return float(proj.max() - proj.min())

    def set_cloud(
        self,
        points_base: np.ndarray,
        *,
        t: float | None = None,
        range_cam_m: float | None = None,
    ) -> None:
        """Replace the cloud and invalidate cached summaries."""
        self.points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
        self.last_cloud_t = float(t if t is not None else time.time())
        self.n_cloud_updates += 1
        self._centroid = None
        if range_cam_m is not None and np.isfinite(range_cam_m):
            self.range_cam_m = float(range_cam_m)

    def observe(self, box: tuple[float, float, float, float], confidence: float, t: float) -> None:
        """Refresh the 2D detection (called when a detection associates to us)."""
        self.box = box
        self.confidence = float(confidence)
        self.last_seen_t = float(t)
        self.misses = 0
