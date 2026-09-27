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
    CAP_HSV, MAX_AREA, MIN_AREA, MIN_SAT, MIN_VAL, Cap, draw, find_caps)


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

    def test_gold_is_not_offered_at_all(self):
        # This bench has no gold caps, and warm wood lands in gold's hue window. A
        # colour nobody has a cap for is all false positives and no true ones.
        assert "gold" not in CAP_HSV


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

    def test_a_huge_blob_is_too_big(self):
        assert find_caps(frame_with((320, 240, 90, BLUE))) == []

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
