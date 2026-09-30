"""Cap detection measured on THIS rig's camera, and why the ported one could not be.

The X250's `caps2.py` does the same job with the same shape, and it is not broken — it is
calibrated for a different bench. Pointed at the SO-101's OAK-D over a pale wooden
turntable it returned FOURTEEN caps in a frame containing two: the gripper's own body, the
turntable, a wooden block and the frame edges all passed its gates, and `restrict_to_mat`
changed nothing because there is no dark rubber mat here to restrict to.

The constants that matter turned out not to be the hue windows. Measured on a live frame:

    blue cap        H  96- 99   S 209-233   V 187-231
    green cap       H  83- 87   S 190-235   V 119-148
    turntable       H   0-178   S  12- 40   V  74-173
    gripper body    H   6-173   S   8-188   V  30-103

The caps are saturated plastic; everything that fooled the ported detector is washed out
or dark. So SATURATION is the discriminator and hue is the tie-break — the reverse of how
the ported version is gated, and the reason it let a whole bench through.

The second trap is subtler: green and blue are ten degrees of hue apart here, and caps2
splits them at H=85 with this green cap measuring a median of 86. The split sat INSIDE the
measurement. These tests pin it in the observed gap instead.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from rax.perception.tube_caps import (
    CAP_HSV,
    MAX_AREA,
    MIN_AREA,
    MIN_SAT,
    MIN_VAL,
    draw,
    find_caps,
    fold,
    tube_axis,
    twist_error,
)


def frame_with(*blobs, bg_hsv=(20, 25, 165), size=(480, 640)):
    """A pale low-saturation background with HSV blobs painted on it.

    The background is the turntable as measured: hue is meaningless at S=25, which is
    exactly why a hue-first detector reads wood as a cap.
    """
    img = np.zeros((size[0], size[1], 3), np.uint8)
    img[:, :] = bg_hsv
    for (cx, cy, r, hsv) in blobs:
        cv2.circle(img, (cx, cy), r, hsv, -1)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


#: The two caps as measured, as (H, S, V).
BLUE = (97, 224, 214)
GREEN = (86, 213, 132)


class TestItFindsRealCaps:
    def test_a_blue_cap_is_found(self):
        caps = find_caps(frame_with((320, 240, 22, BLUE)))
        assert len(caps) == 1
        assert caps[0].colour == "blue"
        assert caps[0].x == pytest.approx(320, abs=3)
        assert caps[0].y == pytest.approx(240, abs=3)

    def test_a_green_cap_is_found(self):
        caps = find_caps(frame_with((200, 150, 20, GREEN)))
        assert len(caps) == 1 and caps[0].colour == "green"

    def test_both_at_once_biggest_first(self):
        caps = find_caps(frame_with((320, 240, 24, BLUE), (150, 120, 16, GREEN)))
        assert [c.colour for c in caps] == ["blue", "green"]
        assert caps[0].area > caps[1].area

    def test_the_box_covers_the_cap(self):
        caps = find_caps(frame_with((300, 200, 20, BLUE)))
        x1, y1, x2, y2 = caps[0].bbox
        assert x1 < 300 < x2 and y1 < 200 < y2
        assert caps[0].w == pytest.approx(40, abs=6)


class TestTheThingsThatFooledThePortedDetector:
    """Each of these was a real false positive on a real frame."""

    def test_pale_wood_is_not_a_cap(self):
        # S ~ 25. A hue-first gate reads this as whatever hue the noise lands on.
        assert find_caps(frame_with((320, 240, 40, (100, 25, 165)))) == []

    def test_a_lit_wooden_block_is_not_a_cap(self):
        # H 103 — squarely inside blue's hue window — but only S ~ 112.
        assert find_caps(frame_with((320, 240, 30, (103, 112, 150)))) == []

    def test_the_dark_gripper_is_not_a_cap(self):
        # H 104, S 142: passes hue, and would pass a lenient saturation gate. It is
        # rejected on VALUE, because the gripper sits in its own shadow at V ~ 44.
        assert find_caps(frame_with((320, 240, 30, (104, 142, 44)))) == []

    def test_a_long_smear_is_not_a_cap(self):
        # The ported detector returned a 198x96 "gold" blob of bench. A cap is a blob.
        img = np.zeros((480, 640, 3), np.uint8)
        img[:, :] = (20, 25, 165)
        cv2.rectangle(img, (100, 300), (300, 318), BLUE, -1)      # 200x18
        assert find_caps(cv2.cvtColor(img, cv2.COLOR_HSV2BGR)) == []

    def test_gold_is_offered_now_that_the_bench_has_gold_caps(self):
        # It was NOT offered, deliberately: warm wood lands in gold's hue window and a
        # colour nobody has a cap for is all false positives and no true ones. The
        # bench has gold-capped tubes now, so the trade has changed -- but the reason
        # gold was risky has not, which is what the next two tests pin.
        assert "gold" in CAP_HSV

    def test_a_gold_cap_is_found(self):
        # Hue 25, and saturated the way an anodised cap is.
        caps = find_caps(frame_with((320, 240, 22, (25, 205, 150))), colours=("gold",))
        assert [c.colour for c in caps] == ["gold"]

    def test_warm_bench_wood_is_still_not_a_gold_cap(self):
        # THE WHOLE RISK OF ADDING GOLD, in one case. Bare wood sits right inside
        # gold's hue window -- measured 10-25 on this bench -- and is rejected only
        # because it is washed out: S 12-40 against a gate of 160. If this test ever
        # fails, the gate moved, and the table is about to be reported as a tube.
        assert find_caps(frame_with((320, 240, 40, (22, 35, 190)))) == []

    def test_gold_is_off_by_default_red_is_on(self):
        # 2026-09-27: gold saw wood, stains and a hand as caps; the tubes got red caps.
        assert find_caps(frame_with((320, 240, 22, (25, 205, 150)))) == []
        caps = find_caps(frame_with((320, 240, 22, (178, 220, 180))))
        assert [c.colour for c in caps] == ["red"]

    def test_red_is_found_on_both_sides_of_the_hue_seam(self):
        for h in (2, 176):
            caps = find_caps(frame_with((320, 240, 22, (h, 220, 180))))
            assert [c.colour for c in caps] == ["red"], h

    def test_dark_wood_is_not_red(self):
        # Measured: the dark wood the arm looks at, H 9-21, S median 64, p95 102.
        assert find_caps(frame_with((320, 240, 40, (12, 102, 150)))) == []
        assert find_caps(frame_with((320, 240, 40, (5, 95, 130)))) == []

    def test_the_measured_pale_green_cap_is_found(self):
        # Missed in plain view on 2026-09-28: S 147-175 against a gate of 160.
        caps = find_caps(frame_with((320, 240, 22, (85, 150, 160))))
        assert [c.colour for c in caps] == ["green"]

    def test_the_silver_rack_is_not_a_blue_cap(self):
        # 2026-09-28: the silver rack's shadowed metal, H 104-106 S 117-131 V 136-186,
        # was boxed "blue cap" and approached.
        assert find_caps(frame_with((320, 240, 40, (105, 122, 153)))) == []

    def test_the_overexposed_close_blue_cap_is_blue(self):
        caps = find_caps(frame_with((320, 240, 30, (97, 124, 250))))
        assert [c.colour for c in caps] == ["blue"]

    def test_the_blue_tinted_jaw_is_not_a_blue_cap(self):
        # Live frame, 2026-09-27: the left jaw read H 107 S 185 V 105 and was reported
        # as a blue cap sitting on the gripper, then cast into the map as a tube.
        assert find_caps(frame_with((320, 240, 30, (107, 185, 105)))) == []

    def test_the_measured_gold_cap_is_found(self):
        # Same frame: the gold cap read S 134-162 -- under the old single gate of 160,
        # so it came and went, and the pick failed with the cap on screen.
        caps = find_caps(frame_with((320, 240, 22, (18, 140, 140))), colours=("gold",))
        assert [c.colour for c in caps] == ["gold"]


class TestTheGreenBlueSplit:
    def test_the_measured_green_is_green_not_blue(self):
        # H=86 is where caps2 put its boundary; this green cap's median sits on it.
        caps = find_caps(frame_with((320, 240, 22, (86, 213, 132))))
        assert [c.colour for c in caps] == ["green"]

    def test_the_measured_blue_is_blue(self):
        caps = find_caps(frame_with((320, 240, 22, (97, 224, 214))))
        assert [c.colour for c in caps] == ["blue"]

    def test_the_split_lies_in_the_observed_gap(self):
        # green measured 83-87, blue 96-99. The boundary has to be strictly between.
        assert CAP_HSV["green"][1] > 87
        assert CAP_HSV["blue"][0] <= 96
        assert CAP_HSV["green"][1] < CAP_HSV["blue"][0]

    def test_one_unit_of_hue_noise_does_not_flip_a_colour(self):
        for h in (83, 84, 85, 86, 87, 88):
            caps = find_caps(frame_with((320, 240, 22, (h, 213, 132))))
            assert [c.colour for c in caps] == ["green"], f"H={h} misread"
        for h in (95, 96, 97, 98, 99, 100):
            caps = find_caps(frame_with((320, 240, 22, (h, 224, 214))))
            assert [c.colour for c in caps] == ["blue"], f"H={h} misread"


class TestTheFingertipExclusion:
    """The jaws wear tape and read as a cap. pick.py lost whole runs to exactly this."""

    def test_a_cap_on_the_fingertip_is_dropped(self):
        img = frame_with((440, 394, 22, BLUE))
        assert find_caps(img, exclude=[(440, 394)], exclude_r=70) == []

    def test_the_same_cap_elsewhere_is_kept(self):
        img = frame_with((150, 120, 22, BLUE))
        assert len(find_caps(img, exclude=[(440, 394)], exclude_r=70)) == 1

    def test_no_exclusion_configured_drops_nothing(self):
        img = frame_with((440, 394, 22, BLUE))
        assert len(find_caps(img)) == 1


class TestBoundsAndRobustness:
    def test_a_speck_is_too_small(self):
        assert find_caps(frame_with((320, 240, 4, BLUE))) == []

    def test_A_CAP_SEEN_CLOSE_UP_IS_STILL_A_CAP(self):
        """The regression that made the arm grab the tube's body instead of its cap.

        Apparent area is a function of RANGE, and the whole point of an approach is to
        reduce the range. MAX_AREA was 3200, measured when the cap read 40x48px from
        25cm — a measurement of one viewing distance being used as a gate. Mid-approach
        the cap measured 55x66 = 2679px and was still growing, so one more increment
        crossed the limit, the box vanished, the arm lost track and closed over the
        BODY. The operator saw it "grab the tail".
        """
        for r in (22, 40, 60, 90):
            caps = find_caps(frame_with((320, 240, r, BLUE)))
            assert len(caps) == 1, f"a cap of radius {r}px was dropped"
            assert caps[0].colour == "blue"

    def test_a_blob_filling_the_frame_is_still_refused(self):
        # The bound is not gone, only loosened: something saturated covering most of the
        # picture is not one cap.
        img = np.zeros((480, 640, 3), np.uint8)
        img[:, :] = BLUE
        assert find_caps(cv2.cvtColor(img, cv2.COLOR_HSV2BGR)) == []

    def test_the_area_bounds_bracket_a_real_cap(self):
        # The real caps measured 909 and 1734 px.
        assert MIN_AREA < 909 and 1734 < MAX_AREA

    def test_the_gates_sit_below_the_measured_caps(self):
        assert MIN_SAT < 190, "the greener cap measured S=190 at its lowest"
        assert MIN_VAL < 119, "the green cap measured V=119 at its darkest"

    def test_an_empty_frame_finds_nothing(self):
        assert find_caps(frame_with()) == []

    def test_none_and_empty_are_handled(self):
        assert find_caps(None) == []
        assert find_caps(np.zeros((0, 0, 3), np.uint8)) == []

    def test_draw_does_not_modify_the_input(self):
        img = frame_with((320, 240, 22, BLUE))
        before = img.copy()
        draw(img, find_caps(img), aim=(440, 394))
        assert np.array_equal(img, before)

    def test_draw_marks_every_cap(self):
        img = frame_with((320, 240, 22, BLUE), (150, 120, 18, GREEN))
        vis = draw(img, find_caps(img))
        assert vis.shape == img.shape
        assert not np.array_equal(vis, img)


# ---------------------------------------------------------------------------------
class TestTubeAxis:
    """Which way the tube lies. The ported orient.tube_axis returned None for BOTH
    tubes at every window size on this bench, for two separate reasons.

    THE TUBE IS NOT ALWAYS THE BRIGHT THING. The port keeps pixels ABOVE Otsu's level,
    correct on a near-black rubber mat and wrong on a pale wooden turntable, where the
    background is brighter than the tube.

    AND "LARGEST" SELECTS THE BENCH. Measured in a 220x220 window around the blue cap:
    wood 35950 px at 220x220, elongation 1.00; tube 2459 px at 42x85, elongation 2.02.
    Largest picks the wood, which is then thrown out for not being elongated, and the
    function returns None with the tube sitting there unexamined.

    The obvious repair — reject anything touching the window edge — is also wrong, and
    that is the case worth keeping a test for: the blue tube is longer than the window
    and legitimately runs off one side.
    """

    def _scene(self, bg_v=165, tube_v=120, angle=0.0, length=96, width=26,
               centre=(160, 160), size=320):
        """A tube laid on a background, either of which may be the brighter."""
        img = np.zeros((size, size, 3), np.uint8)
        img[:, :] = (20, 25, bg_v)
        box = cv2.boxPoints(((centre[0], centre[1]), (length, width), angle))
        cv2.fillPoly(img, [np.int32(box)], (15, 20, tube_v))
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def test_it_finds_a_tube_DARKER_than_the_background(self):
        # The pale-turntable case, which the port cannot see at all.
        ax = tube_axis(self._scene(bg_v=180, tube_v=110, angle=30.0), (160, 160))
        assert ax is not None
        assert abs(fold(ax.angle_deg - 30.0)) < 8

    def test_it_still_finds_a_tube_BRIGHTER_than_the_background(self):
        # The black-mat case the port was written for must keep working.
        ax = tube_axis(self._scene(bg_v=40, tube_v=200, angle=-25.0), (160, 160))
        assert ax is not None
        assert abs(fold(ax.angle_deg - (-25.0))) < 8

    def test_a_tube_running_OFF_the_window_is_still_found(self):
        # The blue tube's case. Rejecting any edge contact loses it.
        img = self._scene(angle=90.0, length=300, width=26, centre=(160, 160))
        ax = tube_axis(img, (160, 160), r=70)
        assert ax is not None, "a tube longer than the window must still be measured"

    def test_a_featureless_background_yields_no_axis(self):
        img = np.zeros((320, 320, 3), np.uint8)
        img[:, :] = (20, 25, 165)
        assert tube_axis(cv2.cvtColor(img, cv2.COLOR_HSV2BGR), (160, 160)) is None

    def test_a_round_blob_is_not_an_axis(self):
        img = np.zeros((320, 320, 3), np.uint8)
        img[:, :] = (20, 25, 165)
        cv2.circle(img, (160, 160), 34, (15, 20, 110), -1)
        assert tube_axis(cv2.cvtColor(img, cv2.COLOR_HSV2BGR), (160, 160)) is None

    def test_angles_are_folded_so_a_tube_has_no_head_or_tail(self):
        a = tube_axis(self._scene(angle=10.0), (160, 160))
        b = tube_axis(self._scene(angle=190.0), (160, 160))
        assert a is not None and b is not None
        assert abs(fold(a.angle_deg - b.angle_deg)) < 8

    def test_elongation_reports_confidence(self):
        slim = tube_axis(self._scene(length=120, width=18), (160, 160))
        stub = tube_axis(self._scene(length=44, width=30), (160, 160))
        assert slim is not None and slim.is_confident
        assert stub is None or not stub.is_confident

    def test_none_and_empty_are_handled(self):
        assert tube_axis(None, (10, 10)) is None
        assert tube_axis(np.zeros((0, 0, 3), np.uint8), (10, 10)) is None


class TestTwistError:
    """A parallel gripper has to close ACROSS the tube."""

    def _tube(self, angle):
        img = np.zeros((320, 320, 3), np.uint8)
        img[:, :] = (20, 25, 175)
        box = cv2.boxPoints(((160, 160), (110, 24), angle))
        cv2.fillPoly(img, [np.int32(box)], (15, 20, 105))
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def test_a_tube_square_to_the_jaws_needs_no_twist(self):
        # jaws horizontal (0 deg) want the tube vertical (90 deg)
        err, _ax = twist_error(self._tube(90.0), (160, 160), 0.0)
        assert abs(err) < 8

    def test_a_tube_along_the_jaws_needs_a_quarter_turn(self):
        err, _ax = twist_error(self._tube(0.0), (160, 160), 0.0)
        assert abs(abs(err) - 90.0) < 10 or abs(err) > 80

    def test_the_error_is_folded_not_wrapped(self):
        # 1 and 179 differ by 2 degrees. A controller fed 178 drives the wrong way
        # through its whole range.
        err, _ax = twist_error(self._tube(89.0), (160, 160), 0.0)
        assert -90.0 <= err < 90.0
        assert abs(err) < 15

    def test_the_jaw_axis_is_a_parameter_not_a_config_file(self):
        # It differs per arm and per mount; this module must not know which arm it is.
        import inspect
        assert "jaw_axis_deg" in inspect.signature(twist_error).parameters

    def test_no_tube_means_no_error(self):
        img = np.zeros((320, 320, 3), np.uint8)
        img[:, :] = (20, 25, 165)
        assert twist_error(cv2.cvtColor(img, cv2.COLOR_HSV2BGR), (160, 160), 0.0) is None
