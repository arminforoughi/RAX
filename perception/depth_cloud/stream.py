"""``PointCloudStream`` — fan out per-object cloud updates to subscribers.

:class:`perception.depth_cloud.CloudTracker` publishes a small batch of
:class:`CloudUpdate` each tick (only the objects whose clouds actually changed).
Subscribers are plain callbacks, so a viewer, a logger, or the exchange WebSocket
bridge can all listen without the tracker knowing about any of them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class CloudUpdate:
    """One object's freshly-updated, base-frame point cloud."""

    tag: int
    label: str
    points: np.ndarray  # Nx3, base frame
    centroid: np.ndarray  # 3, base frame
    is_focus: bool = False
    t: float = field(default_factory=time.time)

    def to_serializable(self, *, max_points: int = 512) -> dict:
        """Compact dict for JSON transport (e.g. over the exchange WebSocket)."""
        pts = self.points
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]
        return {
            "tag": self.tag,
            "label": self.label,
            "is_focus": self.is_focus,
            "centroid": [float(v) for v in self.centroid],
            "points": pts.astype(np.float32).round(4).tolist(),
            "t": self.t,
        }


Subscriber = Callable[[list[CloudUpdate]], None]


class PointCloudStream:
    """Tiny synchronous pub/sub for cloud updates."""

    def __init__(self) -> None:
        self._subs: list[Subscriber] = []

    def subscribe(self, cb: Subscriber) -> Callable[[], None]:
        """Register a callback; returns an unsubscribe function."""
        self._subs.append(cb)
        return lambda: self._subs.remove(cb) if cb in self._subs else None

    def publish(self, updates: list[CloudUpdate]) -> None:
        if not updates:
            return
        for cb in list(self._subs):
            cb(updates)
