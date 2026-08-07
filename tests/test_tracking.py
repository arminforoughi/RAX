"""Tests for the between-detection trackers and the monocular measurer.

Both fill gaps that a detector alone leaves: the trackers follow an object between
detection cycles, and the measurer turns one frame into metric size and orientation.
Neither needs a robot, and AnchorTracker's base-frame anchor works off any camera
geometry — which is what these check.

    python tests/test_tracking.py
    pytest tests/test_tracking.py
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.detection.tracking import AnchorTracker, PixelTracker  # noqa: E402
from perception.camera_geometry import (  # noqa: E402
    CameraGeometry, FixedCamera, intrinsics_from_dict, parse_tf)
from perception.measure import ObjectMeasurer, classify_shape, silhouette_mask  # noqa: E402
from perception.object_priors import PRIORS  # noqa: E402

TABLE_BGR = (160, 140, 110)     # a wood-ish table, in RGB order as the code sees it


def _geom():
    """A camera 60 cm above the table looking straight down."""
    return CameraGeometry(
        intrinsics_from_dict({"fx": 517.0, "fy": 517.0, "cx": 320.0, "cy": 240.0}, 640, 480),
        FixedCamera(parse_tf("0,0,0.60,3.14159265,0,0")))


def _scene(box=(300, 220, 370, 290), colour=(220, 30, 30)):
    img = np.zeros((480, 640, 3), np.uint8)
    img[:, :] = TABLE_BGR
    cv2.rectangle(img, box[:2], box[2:], colour, -1)
    return img


def test_anchor_tracker_finds_a_colour_blob():
    t = AnchorTracker("red", _geom())
    tr = t.track(_scene())
    assert tr is not None, "a saturated red square on a wood table must be found"
    assert 330 < tr.uv[0] < 342 and 250 < tr.uv[1] < 262
    assert not tr.clipped


def test_anchor_round_trips_through_the_camera_geometry():
    """The anchor is stored in the BASE frame, so re-projecting it must land back on
    the pixel it came from. That round trip is what lets the tracker predict a window
    after a dropout from the arm's pose alone."""
    geom = _geom()
    t = AnchorTracker("red", geom)
    tr = t.track(_scene())
    T = geom.T_base_cam()
    t.update_anchor(tr.uv, 0.60, T, 0.0)
    uv = t.predict_uv(T)
    assert uv is not None
    assert abs(uv[0] - tr.uv[0]) < 0.01 and abs(uv[1] - tr.uv[1]) < 0.01


def test_anchor_prediction_needs_an_anchor():
    t = AnchorTracker("red", _geom())
    assert t.predict_uv(_geom().T_base_cam()) is None
    t.track(_scene())
    t.reset()
    assert t.p_anchor is None and t.last is None


def test_pixel_tracker_follows_a_moved_object():
    """Tag on one frame, then track on the next: the window must move with the object
    rather than staying where the detection put it."""
    pt = PixelTracker("thing")
    pt.tag(_scene(), (300, 220, 370, 290))
    assert pt.hist is not None, "tagging must build a histogram"
    moved = pt.track(_scene(box=(320, 240, 390, 310)))
    assert moved is not None, "the tracker lost an object that moved 20 px"
    assert moved.uv[0] > 345 and moved.uv[1] > 265


def test_pixel_tracker_reports_nothing_before_being_tagged():
    assert PixelTracker("thing").track(_scene()) is None


def test_pixel_tracker_gives_up_when_the_object_leaves():
    """A tracker that keeps reporting a window after the object is gone is worse than
    one that admits it lost: the map would keep confirming a ghost."""
    pt = PixelTracker("thing")
    pt.tag(_scene(), (300, 220, 370, 290))
    empty = np.zeros((480, 640, 3), np.uint8)
    empty[:, :] = TABLE_BGR
    assert pt.track(empty) is None


def test_silhouette_mask_separates_object_from_table():
    mask = silhouette_mask(_scene(), (300, 220, 370, 290))
    assert mask is not None
    inside = mask[225:285, 305:365]
    assert inside.mean() > 200, "the object's interior should be masked in"
    assert mask[50:100, 50:100].max() == 0, "far-away table must not be masked"


def test_silhouette_mask_declines_when_object_matches_the_table():
    """Returning a confident mask for an invisible object is what produces phantom
    footprints, so this must fail rather than guess."""
    flat = np.zeros((480, 640, 3), np.uint8)
    flat[:, :] = TABLE_BGR
    assert silhouette_mask(flat, (300, 220, 370, 290)) is None


def test_classify_shape_keeps_a_known_prior():
    # A cup stays a cylinder even if the measured footprint came out rectangular.
    assert classify_shape(0.08, 0.08, 0.10, "cylinder") == "cylinder"
    assert classify_shape(0.02, 0.14, 0.01, "cuboid") == "cuboid"    # clearly elongated
    assert classify_shape(0.05, 0.05, 0.05, "cuboid") == "cube"
    assert classify_shape(0.04, 0.04, 0.12, "cuboid") == "cylinder"  # tall, square-ish


def test_measurer_records_why_it_rejected_a_measurement():
    """'everything says (prior)' has to be diagnosable, not a mystery — the stats are
    surfaced in /status for exactly that reason."""
    geom = _geom()
    m = ObjectMeasurer(geom, PRIORS, reach_m=(0.05, 0.60), table_z=0.0)
    flat = np.zeros((480, 640, 3), np.uint8)
    flat[:, :] = TABLE_BGR
    assert m.measure(flat, (300, 220, 370, 290), geom.T_base_cam(), "cup") is None
    assert m.stats.get("no_silhouette", 0) == 1


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
