"""Perception — connections to vision; turns sensors into world understanding.

Subpackages:
    vision/       Camera frame pipeline: capture -> detection -> labeled scene.
    depth_cloud/  Fuse depth + detections into a point cloud and recover the
                  3D position of each object (x, y, depth) for manipulation
                  and navigation.
"""
