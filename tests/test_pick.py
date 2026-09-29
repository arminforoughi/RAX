"""rax.pick: the generic pick, place and scan, closed-loop on the simulated SO-101."""

import math
import os
import sys

import cv2
import numpy as np
import pytest

from rax.pick import ColourTarget, pick, place, scan
from rax.pick.arm import bearing_of, pan_for_bearing, with_pan
from rax.pick.pick import _roll_within_limits
from rax.pick.sim import SimArm, Tube
from rax.pick.targets import Detection, body_axis, edge_axis


@pytest.fixture(scope="module")
def arm():
    return SimArm()


def test_pan_for_bearing_faces_the_bearing(arm):
    for bear in (-20.0, 0.0, 26.0, 40.0):
        q = with_pan(arm, arm.home, pan_for_bearing(arm, arm.home, math.radians(bear)))
        assert abs(math.degrees(bearing_of(arm.tip(q))) - bear) < 1.0


def test_roll_takes_the_equivalent_turn_inside_the_limits(arm):
    arm.q = arm.home.copy()
    arm.q[arm.roll] = 150.0
    assert _roll_within_limits(arm, 60.0) == -120.0     # +60 would pass the 162 limit
    assert _roll_within_limits(arm, -30.0) == -30.0


def _stick(angle_deg):
    img = np.full((240, 320, 3), 40, np.uint8)
    a = math.radians(angle_deg)
    c = np.array([160.0, 120.0])
    d = 80 * np.array([math.cos(a), math.sin(a)])
    cv2.line(img, tuple(int(v) for v in c - d), tuple(int(v) for v in c + d), (220, 220, 220), 10)
    return img, c, d


@pytest.mark.parametrize("angle", [0.0, 35.0, 90.0, 140.0])
def test_edge_axis_reads_a_long_object(angle):
    img, c, d = _stick(angle)
    box = (min(c[0] - d[0], c[0] + d[0]) - 8, min(c[1] - d[1], c[1] + d[1]) - 8,
           max(c[0] - d[0], c[0] + d[0]) + 8, max(c[1] - d[1], c[1] + d[1]) + 8)
    got = edge_axis(img, Detection(c[0], c[1], box))
    assert got is not None and abs(((got - angle) + 90) % 180 - 90) < 6


def test_edge_axis_has_no_answer_for_a_round_object():
    img = np.full((200, 200, 3), 40, np.uint8)
    cv2.circle(img, (100, 100), 40, (220, 220, 220), -1)
    assert edge_axis(img, Detection(100, 100, (55, 55, 145, 145))) is None


def test_body_axis_runs_from_the_cap_along_the_tube():
    img = np.full((300, 300, 3), 40, np.uint8)
    cv2.line(img, (150, 150), (150 + 110, 150 + 64), (215, 215, 215), 14)   # ~30deg
    cv2.circle(img, (150, 150), 12, (146, 190, 26), -1)
    got = body_axis(img, Detection(150, 150, (138, 138, 162, 162)))
    assert got is not None and abs(got - 30.0) < 6


@pytest.mark.parametrize("r, bear, yaw", [(0.26, 10, 297), (0.21, 28, 223),
                                          (0.28, 2, 117), (0.20, 38, 289)])
def test_pick_a_tube_in_the_sim(r, bear, yaw):
    x, y = r * math.cos(math.radians(bear)), r * math.sin(math.radians(bear))
    arm = SimArm([Tube(x, y, yaw_deg=yaw)])
    res = pick(arm, ColourTarget(), near_xy=(x, y))
    assert arm.tubes[0].held, "\n".join(arm.messages)
    assert res.pitch > 85                                 # taken square to the table


def test_pick_raises_when_nothing_is_there():
    arm = SimArm([])
    with pytest.raises(RuntimeError, match="in view"):
        pick(arm, ColourTarget(), near_xy=(0.25, 0.05))


def test_scan_maps_each_tube_once():
    tubes = [Tube(0.24, 0.10, colour="green"), Tube(0.27, -0.02, 60, colour="red")]
    found = scan(SimArm(tubes), ColourTarget())
    assert sorted(f.label for f in found) == ["green", "red"]
    for f in found:
        t = next(t for t in tubes if t.colour == f.label)
        assert math.hypot(f.x - t.x, f.y - t.y) < 0.02


def test_place_releases_over_the_spot():
    arm = SimArm([Tube(0.24, 0.05, yaw_deg=100)])
    pick(arm, ColourTarget(), near_xy=(0.24, 0.05))
    x, y = place(arm, (0.22, -0.10), release_z=0.10, pitch=0.0, roll=90.0)
    assert not arm.tubes[0].held
    assert math.hypot(arm.tubes[0].x - 0.22, arm.tubes[0].y + 0.10) < 0.01


def test_tube_mode_rules():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples",
                                    "mission_server"))
    import tube_mode as tm
    hole = tm.top_px_to_xy(*tm.TOP_RACKS[0]["holes_px"][0])
    assert tm.in_rack_zone(hole) and not tm.in_rack_zone((0.25, 0.10))
    assert tm.xy_to_top_px(*tm.top_px_to_xy(600, 300)) == pytest.approx((600, 300), abs=1.0)
    assert tm.tag_of(RuntimeError("closed on nothing (jaws at 1.0)")) == "missed grasp"
    assert tm.tag_of(RuntimeError("no green in view")) == "not found"
