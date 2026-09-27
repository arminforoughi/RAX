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

from rax.models.detection.tracking import AnchorTracker, PixelTracker  # noqa: E402
from rax.perception.camera_geometry import (  # noqa: E402
    CameraGeometry,
    FixedCamera,
    intrinsics_from_dict,
    parse_tf,
)
from rax.perception.measure import ObjectMeasurer, classify_shape, silhouette_mask  # noqa: E402
from rax.perception.object_priors import PRIORS  # noqa: E402

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


# --- which edge is clipped, and what survives it -------------------------------------
# Track.clipped is one bool for four different situations, and callers that acted on it
# threw away boxes they could still have used. A real pick logged "half out of frame" on
# two of its three approach checks, refused both, and drove the whole approach on an
# initial estimate that was 5.2 cm out — the refusals WERE the correction it needed.

from rax.models.detection.tracking import clipped_edges, table_ray_is_usable  # noqa: E402

SHAPE = (480, 640)      # H, W — the OAK-D frame the server runs on


def test_a_box_inside_the_frame_touches_no_edge():
    assert clipped_edges((100, 100, 200, 200), SHAPE) == ()


def test_each_edge_is_named_on_its_own():
    assert clipped_edges((0, 100, 200, 200), SHAPE) == ("left",)
    assert clipped_edges((100, 0, 200, 200), SHAPE) == ("top",)
    assert clipped_edges((100, 100, 639, 200), SHAPE) == ("right",)
    assert clipped_edges((100, 100, 200, 479), SHAPE) == ("bottom",)


def test_a_corner_names_both_edges_in_reading_order():
    assert clipped_edges((0, 0, 200, 200), SHAPE) == ("left", "top")
    assert clipped_edges((100, 100, 639, 479), SHAPE) == ("right", "bottom")


def test_a_top_cut_box_can_still_be_ranged():
    """THE case this exists for. Closing on an object, the gripper looms into the top of
    the frame and cuts the box there. Apparent-size ranging is dead — the width is a
    lie — but where the object meets the table is still visible, and that needs neither
    its size nor its orientation."""
    assert table_ray_is_usable(clipped_edges((100, 0, 200, 300), SHAPE))


def test_a_bottom_cut_box_cannot():
    """The contact point is below the picture; the ray would hit the table short."""
    assert not table_ray_is_usable(clipped_edges((100, 100, 200, 479), SHAPE))


def test_a_side_cut_box_cannot():
    """The visible centroid is not the object's centre, so the bearing is biased inward
    by up to half the hidden width — and the approach takes a refine's bearing in full."""
    assert not table_ray_is_usable(clipped_edges((0, 100, 200, 300), SHAPE))
    assert not table_ray_is_usable(clipped_edges((100, 100, 639, 300), SHAPE))


def test_an_unclipped_box_is_usable():
    assert table_ray_is_usable(())


def test_the_margin_matches_what_track_itself_uses():
    """clipped_edges must agree with the bool Track computes, or the server would take
    the table-ray branch on a box the tracker never called clipped (or worse, not take
    it on one it did). Track uses: x1 <= 1 or y1 <= 1 or x2 >= W-2 or y2 >= H-2.
    """
    H, W = SHAPE
    for bbox in ((1, 100, 200, 200), (100, 1, 200, 200),
                 (100, 100, W - 2, 200), (100, 100, 200, H - 2),
                 (2, 2, W - 3, H - 3), (100, 100, 200, 200)):
        x1, y1, x2, y2 = bbox
        track_says = x1 <= 1 or y1 <= 1 or x2 >= W - 2 or y2 >= H - 2
        assert bool(clipped_edges(bbox, SHAPE)) == track_says, bbox
