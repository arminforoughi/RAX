"""The tube rig: one interface over two arms and a simulator, and it must be able to FAIL.

WHAT IS WORTH TESTING HERE, given the hardware is absent. Not "does the arm pick up a
tube" — nothing here can answer that. What can be answered, and matters:

  * the seam is satisfied by every backend, so adding an arm is describing one
  * the simulator is capable of MISSING. A sim that always succeeds is worse than no sim,
    because a UI developed against it never renders its own failure path and a shared
    controller never has its retry loop exercised.
  * the two arms' grip sensors are genuinely different mechanisms behind one verdict
  * a place is verified by the TUBE BEING IN THE RACK, not by the gripper — the bug the
    first end-to-end run walked into. See tube_server.run_pick's `verify`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SERVER = Path(__file__).resolve().parents[1] / "examples" / "tube_server"
sys.path.insert(0, str(SERVER))

rig_mod = pytest.importorskip("rig")
SimRig, Rack, Tube = rig_mod.SimRig, rig_mod.Rack, rig_mod.Tube


@pytest.fixture
def sim():
    return SimRig("x250")


# ---------------------------------------------------------------------------------
class TestTheSeamIsSatisfied:
    def test_the_sim_reports_a_resolved_profile_and_real_joint_angles(self, sim):
        assert sim.profile.name == "x250"
        q = sim.joints_deg()
        assert q.shape == (5,)
        lo, hi = sim.profile.limits()
        assert np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6)

    def test_it_declares_itself_simulated(self, sim):
        # The UI shows this. A demo indistinguishable from a real run is worse than none.
        assert sim.simulated is True

    def test_the_kinematics_are_not_faked(self, sim):
        # Same URDF, same FK as the hardware path — that is what makes the 3D view a view
        # of the real description rather than a cartoon.
        tip = sim.tip_xyz()
        assert tip.shape == (3,)
        assert np.hypot(tip[0], tip[1]) < sim.profile.reach_max_m

    def test_racks_and_tubes_are_reported(self, sim):
        assert {r.name for r in sim.racks()} == {"black", "grey"}
        assert len(sim.tubes()) == 3

    def test_every_backend_class_offers_the_same_members(self):
        want = {"joints_deg", "frame", "gripper", "grip_sensor", "tubes", "racks"}
        for cls in (rig_mod.SimRig, rig_mod.X250Rig, rig_mod.So101Rig):
            missing = want - set(dir(cls))
            assert not missing, f"{cls.__name__} is missing {missing}"


class TestRackGeometry:
    def test_holes_are_placed_in_the_racks_own_frame(self):
        r = Rack("t", 1.0, 2.0, 0.0, ((0.01, 0.0), (-0.01, 0.0)))
        assert r.hole_xy(0) == pytest.approx((1.01, 2.0))
        assert r.hole_xy(1) == pytest.approx((0.99, 2.0))

    def test_yaw_rotates_the_hole_grid(self):
        r = Rack("t", 0.0, 0.0, 90.0, ((0.01, 0.0),))
        x, y = r.hole_xy(0)
        assert (x, y) == pytest.approx((0.0, 0.01), abs=1e-9)

    def test_a_six_hole_rack_is_a_three_by_two_grid(self, sim):
        black = next(r for r in sim.racks() if r.name == "black")
        assert len(black.holes) == 6
        xs = sorted({round(h[0], 4) for h in black.holes})
        assert len(xs) == 3, "three columns"


# ---------------------------------------------------------------------------------
class TestTheSimCanMiss:
    """The property that makes the simulator worth having. See the module docstring."""

    def test_grasping_from_the_home_pose_FAILS(self, sim):
        # The hand starts raised and retracted; the tubes are out on the bench. Closing
        # from there must come up empty, or nothing downstream ever sees a failure.
        assert sim.grasp(1) is False
        assert sim.tube(1).held is False

    def test_a_failed_close_reads_as_empty_on_the_grip_sensor(self, sim):
        sim.grasp(1)
        r = sim.grip_sensor().verdict(sim.gripper())
        assert r.held is False

    def test_grasping_with_the_tip_on_the_tube_SUCCEEDS(self, sim):
        t = sim.tube(1)
        _put_tip_near(sim, t.x, t.y)
        assert sim.grasp(1) is True
        assert sim.tube(1).held is True

    def test_a_successful_close_reads_as_held(self, sim):
        t = sim.tube(1)
        _put_tip_near(sim, t.x, t.y)
        sim.grasp(1)
        assert sim.grip_sensor().verdict(sim.gripper()).held is True

    def test_grasping_a_tube_that_is_not_there_fails_rather_than_raising(self, sim):
        assert sim.grasp(99) is False


class TestPlacing:
    def test_a_release_puts_the_tube_in_the_hole(self, sim):
        t = sim.tube(1)
        _put_tip_near(sim, t.x, t.y)
        sim.grasp(1)
        assert sim.release_into("black", 2) is True
        t = sim.tube(1)
        assert t.rack == "black" and t.hole == 2 and not t.held
        rack = next(r for r in sim.racks() if r.name == "black")
        assert (t.x, t.y) == pytest.approx(rack.hole_xy(2))

    def test_releasing_nothing_fails(self, sim):
        assert sim.release_into("black", 0) is False

    def test_an_occupied_hole_is_no_longer_free(self, sim):
        before = set(sim.free_holes("black"))
        t = sim.tube(1)
        _put_tip_near(sim, t.x, t.y)
        sim.grasp(1)
        sim.release_into("black", 3)
        assert set(sim.free_holes("black")) == before - {3}

    def test_a_bad_hole_index_is_refused(self, sim):
        t = sim.tube(1)
        _put_tip_near(sim, t.x, t.y)
        sim.grasp(1)
        assert sim.release_into("black", 99) is False
        assert sim.release_into("nosuchrack", 0) is False

    def test_AFTER_A_PLACE_THE_GRIPPER_READS_OPEN_NOT_HOLDING(self, sim):
        """The exact reading that made the first end-to-end run report DONE wrongly.

        The place-open position is above the holding threshold, so the naive position
        verdict said "holding" about jaws that had just released a tube. The guard turns
        that into "you are reading an open gripper", which is the honest answer.
        """
        t = sim.tube(1)
        _put_tip_near(sim, t.x, t.y)
        sim.grasp(1)
        sim.release_into("black", 0)
        r = sim.grip_sensor().verdict(sim.gripper())
        assert r.held is False, "a released gripper must not read as holding"
        assert "open" in r.detail


class TestAHeldTubeTravelsWithTheHand:
    def test_the_tube_follows_the_tip(self, sim):
        t = sim.tube(1)
        _put_tip_near(sim, t.x, t.y)
        sim.grasp(1)
        _put_tip_near(sim, 0.20, -0.10)
        tip = sim.tip_xyz()
        held = sim.tube(1)
        assert (held.x, held.y) == pytest.approx((tip[0], tip[1]), abs=1e-6)

    def test_an_unheld_tube_does_not(self, sim):
        before = (sim.tube(2).x, sim.tube(2).y)
        _put_tip_near(sim, 0.20, -0.10)
        assert (sim.tube(2).x, sim.tube(2).y) == pytest.approx(before)


# ---------------------------------------------------------------------------------
class TestTheTwoArmsUseDifferentSensorsForOneVerdict:
    def test_the_x250_reads_a_position(self):
        from rax.manipulation.grip import PositionThreshold
        r = rig_mod.X250Rig.__new__(rig_mod.X250Rig)
        from rax.robots.profiles import load_profile
        r.profile = load_profile("x250")
        s = r.grip_sensor()
        assert isinstance(s, PositionThreshold)
        # The measured pair, not rounded.
        assert (s.holding, s.empty) == (31.9, 30.2)
        assert s.open_above is not None, "the position sensor needs the open guard"

    def test_the_so101_reads_a_current_rise(self):
        from rax.manipulation.grip import CurrentRise
        r = rig_mod.So101Rig(joints=lambda: np.zeros(5), frame=lambda: None,
                             gripper=lambda: 0.0, idle_current=102.0)
        s = r.grip_sensor()
        assert isinstance(s, CurrentRise)
        assert s.idle == 102.0
        assert s.delta == r.profile.gripper.contact_current_delta


# ---------------------------------------------------------------------------------
class TestTubeOrientationHasThreeAnswers:
    """Which way a tube is lying, including "we cannot tell" — which is a real answer.

    The obvious test is the measured height: a 100 mm tube standing reads ~100 mm tall,
    lying it reads ~16 mm. That was tried on the rig and it QUIETLY LIED. The first tube
    the SO-101 mapped measured w/d/h = 9/13/41 mm, which the height test called "standing"
    while the wrist camera plainly showed it lying on the turntable.

    9/13/41 is not a tube in any pose — standing is ~16x16x100, lying ~100x16x16 — so the
    silhouette solve had measured a fragment, most likely just the cap. A two-way test fed
    a number of that shape does not fail honestly, it answers confidently. So the test has
    to positively match one hypothesis or return None.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def tm(cls):
        import importlib.util
        from pathlib import Path
        pytest.importorskip("flask")
        path = (Path(__file__).resolve().parents[1] / "examples" / "mission_server"
                / "tube_mode.py")
        spec = importlib.util.spec_from_file_location("tube_mode_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_THE_MEASUREMENT_THAT_FOOLED_THE_HEIGHT_TEST_IS_UNKNOWN(self, tm):
        # The actual numbers off the rig. This is the regression.
        assert tm._orientation(0.009, 0.013, 0.041) is None

    def test_a_tube_that_really_is_upright(self, tm):
        assert tm._orientation(0.016, 0.016, 0.100) is True

    def test_a_tube_that_really_is_lying(self, tm):
        assert tm._orientation(0.100, 0.016, 0.016) is False
        assert tm._orientation(0.016, 0.100, 0.016) is False, "w/d order must not matter"

    def test_lying_survives_a_height_that_reads_low(self, tm):
        # This rig's height measurement is known to read low; the test must tolerate that
        # without becoming so loose it accepts a fragment.
        assert tm._orientation(0.092, 0.015, 0.011) is False

    def test_nothing_measured_is_unknown_not_a_guess(self, tm):
        assert tm._orientation(0.0, 0.0, 0.0) is None

    def test_an_object_that_is_not_tube_shaped_is_unknown(self, tm):
        assert tm._orientation(0.051, 0.051, 0.051) is None      # a cube
        assert tm._orientation(0.33, 0.24, 0.02) is None         # the "laptop" it saw

    def test_an_unclassifiable_orientation_also_clears_yaw_known(self, tm):
        # A yaw measured off a fragment is not a yaw worth drawing, however sure the
        # silhouette solve was about the angle of the fragment.
        src = (pathlib_Path(tm.__file__)).read_text(encoding="utf-8")
        assert "standing is not None" in src, (
            "yaw_known must be gated on the orientation being classifiable")


from pathlib import Path as pathlib_Path  # noqa: E402  (used by the test above)


# ---------------------------------------------------------------------------------
class TestIkReSeeding:
    """The X250 has an elbow-flip dead band too, and walking into it is silent.

    Found by the simulator: stepping a straight Cartesian line from the look pose to a
    tube at (0.24, +0.07), the solver tracked twelve waypoints at 0.00 cm and then missed
    the last by 2.06 cm -- while that same final point solves EXACTLY when seeded from the
    look pose. Nothing was out of reach; the chain had walked into a branch it could not
    finish from. The SO-101 profile carries `ik_seeds` for exactly this; the X250 has none
    characterised, so the fallback seeds are its own named poses.
    """

    @pytest.fixture
    def server(self, sim):
        ts = pytest.importorskip("tube_server")
        ts.RIG[0] = sim
        ts.KIN[0] = None
        return ts

    def test_the_bad_waypoint_reproduces_without_re_seeding(self, server, sim):
        # Pin the failure itself, so the fix cannot be mistaken for decoration.
        import numpy as np
        k = server.kin()
        q = sim.joints_deg()
        start = np.asarray(k.forward_kinematics(q))[:3, 3]
        target = np.array([0.24, 0.07, 0.10])
        T = np.eye(4)
        worst = 0.0
        for i in range(1, 15):
            want = start + (target - start) * (i / 14)
            T[:3, 3] = want
            q = np.asarray(k.inverse_kinematics(q, T, orientation_weight=0.0), np.float64)
            reached = np.asarray(k.forward_kinematics(q))[:3, 3]
            worst = max(worst, float(np.linalg.norm(reached - want)))
        assert worst > 0.01, ("the naive walk used to end up >1cm off; if this no longer "
                              "reproduces, the re-seeding fallback may be untested")

    def test_re_seeding_solves_the_same_walk(self, server, sim):
        import numpy as np
        k = server.kin()
        q = sim.joints_deg()
        start = np.asarray(k.forward_kinematics(q))[:3, 3]
        target = np.array([0.24, 0.07, 0.10])
        worst = 0.0
        for i in range(1, 15):
            want = start + (target - start) * (i / 14)
            q, err = server._ik(k, want, q)
            worst = max(worst, err)
        assert worst <= 0.003, f"re-seeded walk still misses by {worst * 100:.2f}cm"

    def test_a_genuinely_unreachable_point_is_still_reported_as_unreachable(
            self, server, sim):
        # Re-seeding must not turn "cannot reach" into a quiet teleport — that would
        # defeat the one thing the simulator is for.
        import numpy as np
        _q, err = server._ik(server.kin(), np.array([1.5, 0.0, 0.5]), sim.joints_deg())
        assert err > 0.1


# ---------------------------------------------------------------------------------
def _put_tip_near(sim, x, y, z=0.09):
    """Drive the sim's joints so the fingertip lands at (x, y, z), via real IK."""
    from rax.manipulation.arms.urdf_kinematics import UrdfKinematics
    p = sim.profile
    k = UrdfKinematics(p.urdf_path, ee_frame=p.ee_frame, joint_names=list(p.joint_names))
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    q = np.asarray(k.inverse_kinematics(sim.joints_deg(), T, orientation_weight=0.0),
                   np.float64)
    sim.set_joints(q)
    sim.tip_xyz()
    return q
