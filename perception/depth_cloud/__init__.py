"""Depth cloud: fuse stereo depth with detections into per-object point clouds.

Each detected object becomes a tagged :class:`ObjectTrack` carrying its own
base-frame cloud. :class:`CloudTracker` keeps those clouds fresh on a budget
(focused object every tick, others round-robin) and streams the changes through
:class:`PointCloudStream` — giving manipulation a 3D target (centroid) and a
place target (top surface) without re-scanning the whole scene every frame.
"""

from __future__ import annotations

from perception.depth_cloud.cloud_tracker import CloudTracker
from perception.depth_cloud.object_cloud import ObjectTrack
from perception.depth_cloud.stream import CloudUpdate, PointCloudStream

__all__ = ["CloudTracker", "ObjectTrack", "CloudUpdate", "PointCloudStream"]
