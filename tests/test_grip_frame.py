"""The grip's image frame: the symmetry fold, the roll that squares the jaws, and the cost of not.

Three things are pinned here, and all three are places this rig has already been wrong.

THE SIGN AND THE FOLD. Every geometric quantity on this arm has had a sign error at some
point -- the lateral aim, the grasp offset, the hand-eye heading. Orientation is worse
than most because the errors are modular: "5 degrees off" and "355 degrees off" are the
same physical situation, and only one of them makes a controller slam the wrist into its
limit. So the fold is tested directly, at the wrap points, not just in the middle.

THE SYMMETRY ESCAPE. A square looks the same every quarter turn, so several rolls are
the same grasp. The predecessor of this code clamped to the wrist limit instead of
trying them, which saturates for most yaws and throws the orientation away while still
reporting success. ``test_uses_symmetry_when_the_direct_roll_is_unreachable`` is the
regression test for exactly that.

THE REASON ANY OF IT IS WORTH BUILDING. ``ungraspable_fraction`` is the claim that a
fixed wrist loses a large share of picks for geometric reasons alone. If that is not
true, the alignment is not worth its complexity, so the number is checked against the
closed form at the two yaws where it is known exactly.
"""

from __future__ import annotations

import math

import pytest

from rax.manipulation.approach.grip_frame import (
    GripGrid, JawFrame, fold_angle, required_opening_m, roll_to_align,
    ungraspable_fraction, yaw_error_deg)


# The SO-101's measured frame: the fixed finger where /caltip marked it, the moving one
# where open-vs-closed differencing found it, and the roll gain from rolling the wrist
# 12 degrees twice and watching a static object turn (+0.96, +1.01).
MEASURED = JawFrame.from_tips(fixed_uv=(440.0, 394.0), moving_uv=(210.0, 476.0),
                              roll_gain=1.0)
WRIST_LIMITS = (-157.2, 162.8)


class TestFold:
    @pytest.mark.parametrize("raw, folded", [
        (0.0, 0.0), (10.0, 10.0), (-10.0, -10.0),
        (44.0, 44.0), (46.0, -44.0),        # just past the fold, comes back negative
        (90.0, 0.0), (91.0, 1.0), (-90.0, 0.0),
        (135.0, -45.0), (180.0, 0.0), (355.0, -5.0),
    ])
    def test_square_period(self, raw, folded):
        assert fold_angle(raw, 90.0) == pytest.approx(folded, abs=1e-9)

    def test_never_leaves_the_half_open_interval(self):
        for i in range(-2000, 2000):
            assert -45.0 <= fold_angle(i * 0.37, 90.0) < 45.0

    def test_a_line_has_no_head_or_tail(self):
        # Period 180: a line at 170 degrees is 10 degrees from horizontal, not 170.
        assert fold_angle(170.0, 180.0) == pytest.approx(-10.0)


class TestJawFrame:
    def test_axis_is_the_line_between_the_fingertips(self):
        jaw = JawFrame.from_tips((100.0, 100.0), (0.0, 100.0))
        assert jaw.axis_deg == pytest.approx(0.0)
        jaw = JawFrame.from_tips((100.0, 0.0), (100.0, 100.0))
        assert jaw.axis_deg == pytest.approx(90.0)

    def test_centre_and_opening(self):
        jaw = JawFrame.from_tips((100.0, 100.0), (0.0, 100.0))
        assert jaw.centre_uv == (50.0, 100.0)
        assert jaw.opening_px == pytest.approx(100.0)
        assert jaw.opening_m(px_per_m=1000.0) == pytest.approx(0.1)

    def test_the_measured_frame_points_where_the_photograph_does(self):
        # Fixed finger up and to the right of the moving one: the span runs up-right,
        # which is a shallow negative image angle, i.e. just under 180 after the mod.
        assert MEASURED.axis_deg == pytest.approx(160.4, abs=0.5)
        assert MEASURED.opening_px == pytest.approx(244.0, abs=2.0)


class TestYawError:
    def test_square_on_is_zero_error(self):
        assert yaw_error_deg(MEASURED.axis_deg, MEASURED) == pytest.approx(0.0)

    def test_a_quarter_turn_is_the_same_grasp(self):
        # The whole reason the condition carries no +90: a square's edges lie at both
        # t and t+90, so either can face the jaws.
        for k in (-2, -1, 0, 1, 2):
            assert yaw_error_deg(MEASURED.axis_deg + 90.0 * k,
                                 MEASURED) == pytest.approx(0.0, abs=1e-9)

    def test_worst_case_is_forty_five_degrees(self):
        e = yaw_error_deg(MEASURED.axis_deg + 45.0, MEASURED)
        assert abs(e) == pytest.approx(45.0)

    def test_a_general_object_only_has_the_half_turn(self):
        # A pen is not square: t and t+90 are NOT the same grasp, t and t+180 are. The
        # quarter turn lands exactly on the fold, where +90 and -90 are the same
        # orientation -- so the magnitude is the claim, not its sign.
        assert abs(yaw_error_deg(MEASURED.axis_deg + 90.0, MEASURED,
                                 symmetry_deg=180.0)) == pytest.approx(90.0)
        assert yaw_error_deg(MEASURED.axis_deg + 180.0, MEASURED,
                             symmetry_deg=180.0) == pytest.approx(0.0)


class TestRollToAlign:
    def test_it_nulls_the_error(self):
        for offset in (-40.0, -20.0, -7.0, 7.0, 20.0, 40.0):
            obj = MEASURED.axis_deg + offset
            roll, err = roll_to_align(obj, MEASURED, current_roll_deg=104.0,
                                      limits=WRIST_LIMITS, deadband_deg=0.0)
            assert err == pytest.approx(offset, abs=1e-6)
            # Rolling by d turns the scene by +gain*d, so the object's angle AFTER the
            # move is obj + gain*(roll - current). That must land on the jaw axis.
            after = obj + MEASURED.roll_gain * (roll - 104.0)
            assert yaw_error_deg(after, MEASURED) == pytest.approx(0.0, abs=1e-6)

    def test_the_sign_is_the_measured_one(self):
        # Gain is POSITIVE on this rig, so an object rotated +10 past the jaws needs the
        # wrist to roll BACK by 10. A sign slip here turns a 10-degree error into 20.
        roll, _e = roll_to_align(MEASURED.axis_deg + 10.0, MEASURED,
                                 current_roll_deg=104.0, limits=WRIST_LIMITS,
                                 deadband_deg=0.0)
        assert roll == pytest.approx(94.0)

    def test_deadband_leaves_the_wrist_alone(self):
        roll, err = roll_to_align(MEASURED.axis_deg + 2.0, MEASURED,
                                  current_roll_deg=104.0, limits=WRIST_LIMITS,
                                  deadband_deg=3.0)
        assert roll == 104.0
        assert err == pytest.approx(2.0)

    def test_uses_symmetry_when_the_direct_roll_is_unreachable(self):
        # Near the upper limit with a big positive correction wanted: the direct answer
        # is off the end of the wrist's travel, but a quarter turn down is the same
        # grasp and is reachable. The predecessor clamped here and lost the alignment.
        obj = MEASURED.axis_deg - 40.0
        roll, err = roll_to_align(obj, MEASURED, current_roll_deg=160.0,
                                  limits=WRIST_LIMITS, deadband_deg=0.0)
        assert err == pytest.approx(-40.0)
        assert WRIST_LIMITS[0] <= roll <= WRIST_LIMITS[1]
        after = obj + MEASURED.roll_gain * (roll - 160.0)
        assert yaw_error_deg(after, MEASURED) == pytest.approx(0.0, abs=1e-6)

    def test_every_yaw_is_reachable_on_this_wrist(self):
        # The claim the feature rests on: with 320 degrees of travel and a 90-degree
        # symmetry, no presentation is out of reach. If a rig ever fails this, wrist
        # alignment cannot fix all of its picks and the test says so.
        for i in range(180):
            obj = i * 1.0
            roll, _e = roll_to_align(obj, MEASURED, current_roll_deg=104.0,
                                     limits=WRIST_LIMITS, deadband_deg=0.0)
            after = obj + MEASURED.roll_gain * (roll - 104.0)
            assert yaw_error_deg(after, MEASURED) == pytest.approx(0.0, abs=1e-6)

    def test_it_prefers_the_smallest_move(self):
        # Several rolls are the same grasp; taking a needless quarter turn beside a cube
        # is slow and a chance to knock it over.
        roll, _e = roll_to_align(MEASURED.axis_deg + 10.0, MEASURED,
                                 current_roll_deg=104.0, limits=WRIST_LIMITS,
                                 deadband_deg=0.0)
        assert abs(roll - 104.0) <= 45.0

    def test_a_camera_before_the_roll_joint_cannot_align(self):
        # roll_gain 0 means rolling does not turn the picture, so there is nothing to
        # solve. It must not divide by zero and must not pretend it moved.
        blind = JawFrame.from_tips((440.0, 394.0), (210.0, 476.0), roll_gain=0.0)
        roll, err = roll_to_align(blind.axis_deg + 30.0, blind, current_roll_deg=104.0,
                                  limits=WRIST_LIMITS, deadband_deg=0.0)
        assert roll == 104.0
        assert err == pytest.approx(30.0)


class TestWhyItMatters:
    def test_the_diagonal_is_the_worst_case(self):
        a = 0.0508
        assert required_opening_m(a, 0.0) == pytest.approx(a)
        assert required_opening_m(a, 90.0) == pytest.approx(a)
        assert required_opening_m(a, 45.0) == pytest.approx(a * math.sqrt(2.0))

    def test_a_fixed_wrist_loses_most_yaws_on_a_narrow_gripper(self):
        a = 0.0508
        assert ungraspable_fraction(a, 0.060) == pytest.approx(0.74, abs=0.02)
        assert ungraspable_fraction(a, 0.065) == pytest.approx(0.57, abs=0.02)
        assert ungraspable_fraction(a, 0.070) == pytest.approx(0.30, abs=0.02)

    def test_a_gripper_wider_than_the_diagonal_never_fails_on_yaw(self):
        a = 0.0508
        assert ungraspable_fraction(a, a * math.sqrt(2.0) + 1e-4) == 0.0

    def test_alignment_is_what_makes_a_narrow_gripper_work(self):
        # ungraspable_fraction is about a FIXED wrist meeting a random yaw, so a gripper
        # barely wider than the edge fails almost always -- it only ever gets the few
        # presentations that happen to arrive square.
        a = 0.0508
        assert ungraspable_fraction(a, a + 1e-4) > 0.99
        # Aligning removes the yaw from the problem entirely: the jaws always see the
        # edge, never the diagonal. That is the whole argument for the feature.
        assert required_opening_m(a, 0.0) < a + 1e-4


class TestGripGrid:
    GRID = GripGrid(width=640, height=480, cols=6, rows=5)

    def test_cells_tile_the_frame(self):
        assert self.GRID.cell_of((0, 0)) == (0, 0)
        assert self.GRID.cell_of((639, 479)) == (5, 4)
        assert self.GRID.cell_of((320, 240)) == (3, 2)

    def test_out_of_frame_pixels_clamp(self):
        assert self.GRID.cell_of((-50, -50)) == (0, 0)
        assert self.GRID.cell_of((9999, 9999)) == (5, 4)

    def test_the_grip_spans_several_cells(self):
        cells = self.GRID.grip_cells(MEASURED)
        # 244px of opening across a 106px cell: the grip is a band, not a point, and
        # treating only its midpoint as the target would reject good grasps.
        assert len(cells) >= 3
        assert self.GRID.cell_of(MEASURED.fixed_uv) in cells
        assert self.GRID.cell_of(MEASURED.moving_uv) in cells
        assert self.GRID.cell_of(MEASURED.centre_uv) in cells

    def test_in_grip_rejects_the_far_side_of_the_picture(self):
        assert self.GRID.in_grip(MEASURED.centre_uv, MEASURED)
        assert not self.GRID.in_grip((600.0, 20.0), MEASURED)
