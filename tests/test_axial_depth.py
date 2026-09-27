"""The pinhole relation gives DEPTH, not range along the sightline.

`fx * real_width / pixel_width` answers "how far in front of the image plane", not
"how far along the ray to that pixel". Walking it along the ray puts an off-axis object
too close by cos(off-axis angle) — always inward, and growing with the angle. The
approach deliberately keeps the object off-centre so it stays in frame, so this is live
on every pick, and a radial push-out fudge was compensating for it.

Seen on the rig with two cubes at once, converted to camera distance so they compare:

    green, near the optical axis   observed 20cm   mapped 20.2cm   ok
    red,   off in the corner       observed 26cm   mapped 14.8cm   11cm inward

Which is not merely inaccurate — it reported the two cubes in the WRONG ORDER, and
"which of these is nearer" is the question the pick actually asks.

    pytest tests/test_axial_depth.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from rax.perception.camera_geometry import CameraGeometry, FixedCamera, intrinsics_from_dict  # noqa: E402
from rax.perception.locate import ApparentSizeLocalizer  # noqa: E402
from rax.perception.object_priors import PRIORS  # noqa: E402

FX = FY = 517.0
CX, CY = 320.0, 240.0
CUBE_M = 0.0508


CAM_XYZ = (0.30, 0.0, 0.60)


def _geom():
    """A camera 60cm up, 30cm out, looking straight down — so 'off-axis' is
    unambiguous and an on-axis object still lands at a sane radius."""
    T = np.eye(4)
    T[:3, 3] = CAM_XYZ
    T[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return CameraGeometry(
        intrinsics_from_dict({"fx": FX, "fy": FY, "cx": CX, "cy": CY}, 640, 480),
        FixedCamera(T))


def _loc(axial):
    # A wide reach window on purpose: this exercises where the placement PUTS a point,
    # not whether the arm could get there.
    loc = ApparentSizeLocalizer(_geom(), PRIORS, reach_m=(0.0, 5.0))
    loc.axial_depth = axial
    return loc


def _box(u, v, w_px=100.0):
    return (u - w_px / 2, v - w_px / 2, u + w_px / 2, v + w_px / 2)


def _T():
    return _geom().pose.T_base_cam(None)


def _cam_dist(fix):
    """Distance from the CAMERA to where the fix was placed.

    The quantity the cosine relation is about, and the one the live diagnosis used:
    the map records base-frame positions, but the camera rides 22cm out on the arm, so
    a base radius and an observed range are simply different numbers.
    """
    assert fix.ok, f"fix failed: {fix.method}"
    p = np.array([fix.xy[0], fix.xy[1]])
    return float(np.hypot(*(p - np.array(CAM_XYZ[:2]))))


def _off_axis_cos(du, dv):
    return FX / float(np.sqrt(du * du + dv * dv + FX * FX))


@pytest.mark.parametrize("axial", [True, False])
def test_an_object_on_the_optical_axis_is_unaffected(axial):
    """cos(0) = 1. The correction must change nothing at the centre of the frame."""
    fix = _loc(axial).locate(_box(CX, CY), _T(), label="red cube")
    assert fix.ok
    assert _cam_dist(fix) < 0.005


def test_the_old_placement_pulls_an_off_axis_object_inward():
    off = _box(CX + 200.0, CY + 150.0)
    old = _cam_dist(_loc(False).locate(off, _T(), label="red cube"))
    new = _cam_dist(_loc(True).locate(off, _T(), label="red cube"))
    assert old < new, "the sightline placement must land SHORT of the axial one"


def test_the_error_grows_with_the_off_axis_angle():
    """Always inward, and worse the further off-centre — that is the cos signature."""
    gaps = []
    for du in (50.0, 150.0, 250.0):
        b = _box(CX + du, CY)
        old = _cam_dist(_loc(False).locate(b, _T(), label="red cube"))
        new = _cam_dist(_loc(True).locate(b, _T(), label="red cube"))
        gaps.append(new - old)
    assert all(g > 0 for g in gaps), gaps
    assert gaps[0] < gaps[1] < gaps[2], f"error should grow with angle: {gaps}"


def test_the_correction_matches_the_cosine_it_is_named_for():
    du, dv = 220.0, 160.0
    b = _box(CX + du, CY + dv)
    old = _cam_dist(_loc(False).locate(b, _T(), label="red cube"))
    new = _cam_dist(_loc(True).locate(b, _T(), label="red cube"))
    assert old == pytest.approx(new * _off_axis_cos(du, dv), rel=0.02), (old, new)


def test_two_objects_can_be_ordered_wrongly_by_the_old_placement():
    """The failure as the operator met it: a genuinely-further object mapped nearer.

    A big box near the axis (near) against a smaller box off in the corner (far) —
    which is exactly the green/red pair on the table.
    """
    near_box = _box(CX + 20.0, CY, w_px=150.0)              # nearer: bigger box
    far_box = _box(CX + 260.0, CY + 180.0, w_px=100.0)      # further: smaller box
    old_near = _cam_dist(_loc(False).locate(near_box, _T(), label="green cube"))
    old_far = _cam_dist(_loc(False).locate(far_box, _T(), label="red cube"))
    new_near = _cam_dist(_loc(True).locate(near_box, _T(), label="green cube"))
    new_far = _cam_dist(_loc(True).locate(far_box, _T(), label="red cube"))
    assert new_far > new_near, "with the fix, the further object maps further"
    assert (new_far - new_near) > (old_far - old_near), (
        "the old placement compresses the gap, which is how the order inverts")


def test_the_flag_defaults_off_in_the_library():
    """The library stays behaviour-preserving; the server opts in. Anything relying on
    the old placement keeps it until it chooses otherwise."""
    assert ApparentSizeLocalizer.axial_depth is False
