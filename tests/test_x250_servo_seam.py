"""The X250 adapter satisfies the servo's seam, and satisfies the SAFETY half of it.

THE POINT OF THIS FILE is that one approach can drive two different arms. The SO-101 is
a 5-DoF arm with a URDF, forward kinematics and a calibrated hand-eye transform; the
X250 is a 6-joint arm with none of those in this repo, commanded in normalised units it
does not relate to any metric frame. If both satisfy ``ServoArm`` then
``visual_servo.step`` drives both, and everything arm-specific stays in the adapter.

The safety property is the one worth a test rather than a reading. ``sense()`` returning
None is what makes the servo hold instead of moving blind, so it has to return None for
"the detector found nothing" AND for every degenerate case around it -- a missing
camera, an empty frame, a detection that is really the gripper's own tape. A `sense()`
that hallucinated a Sighting would defeat the guarantee the servo tests pin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from rax.manipulation.approach.servo_arm import Axis, ServoArm, Sighting

PICK = Path(__file__).resolve().parents[1] / "examples" / "vla" / "pick"
sys.path.insert(0, str(PICK))

cv2 = pytest.importorskip("cv2")
x250_servo = pytest.importorskip("x250_servo")

X250ServoArm = x250_servo.X250ServoArm
TUBE_AXES = x250_servo.TUBE_AXES

MOTORS = ("base", "shoulder_2", "elbow", "wrist", "tool", "gripper")


class FakeRobot:
    """An X250 that reports joints and hands back whatever frame it was given."""

    def __init__(self, frame=None, pose=None):
        self.pose = dict(pose or {m: 0.0 for m in MOTORS})
        self.frame = frame
        self.sent = []

    def get_observation(self):
        o = {f"{m}.pos": v for m, v in self.pose.items()}
        if self.frame is not None:
            o["wrist"] = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        return o

    def send_action(self, action):
        self.sent.append(dict(action))
        # move straight there, so apply() is testing its own logic and not a servo lag
        for k, v in action.items():
            if k.endswith(".pos"):
                self.pose[k[:-4]] = float(v)
        return action


def mat_with_cap(colour_bgr=(60, 180, 60), centre=(300, 200), r=22):
    """A dark mat filling the frame with one bright cap blob on it."""
    img = np.full((480, 640, 3), 18, np.uint8)          # the mat
    cv2.circle(img, centre, r, colour_bgr, -1)
    return img


class TestSeam:
    def test_the_adapter_is_a_servo_arm(self):
        arm = X250ServoArm(FakeRobot(), "green")
        assert isinstance(arm, ServoArm)

    def test_it_exposes_only_the_approach_axes(self):
        arm = X250ServoArm(FakeRobot(), "green")
        assert [a.name for a in arm.axes] == ["base", "shoulder_2", "elbow"]
        assert arm.actuators().shape == (3,)

    def test_axis_steps_stay_inside_the_drivers_relative_limit(self):
        # The driver clips every command to max_relative_target=8.0. A servo allowed a
        # bigger step would have its command silently clipped, then learn the wrong
        # thing from the move it thought it made.
        for a in TUBE_AXES:
            assert a.max_step <= 8.0, a.name

    def test_actuators_reads_the_live_pose(self):
        bot = FakeRobot(pose={**{m: 0.0 for m in MOTORS},
                              "base": -3.0, "shoulder_2": -55.0, "elbow": -58.0})
        arm = X250ServoArm(bot, "green")
        assert arm.actuators() == pytest.approx([-3.0, -55.0, -58.0])


class TestApplyHoldsTheOtherJoints:
    def test_wrist_and_tool_are_carried_through_untouched(self):
        # The demonstrated wrist and tool angles are what put the gripper at the right
        # attitude; the servo does not drive them and must not disturb them.
        bot = FakeRobot(pose={**{m: 0.0 for m in MOTORS}, "wrist": -6.0, "tool": 41.5,
                              "gripper": 54.0})
        arm = X250ServoArm(bot, "green")
        assert arm.apply(np.array([-2.0, -50.0, -55.0])) is True
        last = bot.sent[-1]
        assert last["wrist.pos"] == pytest.approx(-6.0)
        assert last["tool.pos"] == pytest.approx(41.5)
        assert last["gripper.pos"] == pytest.approx(54.0)

    def test_commands_are_clamped_to_the_envelope(self):
        env = {"base": (-40.6, 11.7), "shoulder_2": (-100.0, 37.0), "elbow": (-100.0, 41.4)}
        bot = FakeRobot()
        arm = X250ServoArm(bot, "green", envelope=env)
        arm.apply(np.array([999.0, -999.0, 999.0]))
        last = bot.sent[-1]
        assert last["base.pos"] == pytest.approx(11.7)
        assert last["shoulder_2.pos"] == pytest.approx(-100.0)
        assert last["elbow.pos"] == pytest.approx(41.4)


class TestSenseIsTheSafetyBoundary:
    def test_a_cap_is_seen_as_pixels(self):
        bot = FakeRobot(frame=mat_with_cap(centre=(300, 200)))
        arm = X250ServoArm(bot, "green")
        seen = arm.sense()
        assert isinstance(seen, Sighting)
        assert seen.cx == pytest.approx(300, abs=12)
        assert seen.cy == pytest.approx(200, abs=12)
        assert seen.height > 8

    def test_no_cap_of_that_colour_is_None(self):
        bot = FakeRobot(frame=mat_with_cap(colour_bgr=(60, 180, 60)))
        assert X250ServoArm(bot, "blue").sense() is None

    def test_an_empty_frame_is_None(self):
        bot = FakeRobot(frame=np.full((480, 640, 3), 18, np.uint8))
        assert X250ServoArm(bot, "green").sense() is None

    def test_no_camera_at_all_is_None(self):
        # A robot constructed with cameras={} has no "wrist" key. This must read as
        # "nothing seen" -- which makes the servo hold -- and never raise into the loop.
        assert X250ServoArm(FakeRobot(frame=None), "green").sense() is None

    def test_a_cap_on_the_gripper_is_excluded(self):
        # The jaws wear blue tape and detect as a blue cap. pick.py lost whole runs to
        # exactly this, so the adapter has to drop anything sitting on a fingertip.
        bot = FakeRobot(frame=mat_with_cap(colour_bgr=(200, 120, 40), centre=(280, 400)))
        arm = X250ServoArm(bot, "blue", exclude=[(274, 405), (393, 403)], exclude_r=70)
        assert arm.sense() is None

    def test_the_same_cap_away_from_the_fingers_is_kept(self):
        bot = FakeRobot(frame=mat_with_cap(colour_bgr=(200, 120, 40), centre=(120, 150)))
        arm = X250ServoArm(bot, "blue", exclude=[(274, 405), (393, 403)], exclude_r=70)
        assert arm.sense() is not None


class TestTheSharedServoDrivesIt:
    def test_a_blind_tick_never_produces_motion(self):
        """The property the whole seam exists to preserve, checked through the adapter."""
        from rax.manipulation.approach.jacobian import ImageJacobian
        from rax.manipulation.approach.visual_servo import (
            HOLD, ABORT, ServoConfig, begin, step)

        jac = ImageJacobian(J=np.eye(3)[:, :3], axes=TUBE_AXES,
                            probe_response_px=np.array([50.0, 50.0, 50.0]))
        state = begin(jac)
        cfg = ServoConfig()
        bot = FakeRobot(frame=None)
        arm = X250ServoArm(bot, "green")
        before = dict(bot.pose)
        for _ in range(3):
            state, cmd = step(state, arm.sense(), cfg)
            assert cmd.kind in (HOLD, ABORT)
            assert getattr(cmd, "dq", None) is None or not np.any(cmd.dq)
        assert bot.pose == before, "a blind servo must not have moved the arm"
