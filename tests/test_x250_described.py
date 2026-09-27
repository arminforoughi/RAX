"""The X250 is DESCRIBED well enough to draw, and honest about what it does not know.

Two things are being pinned here, and they are different in kind.

THE FIRST IS CORRECTNESS OF THE THINGS THAT ARE KNOWN. The joint structure and the joint
travel come off the real arm (a broadcast ping, and the lerobot calibration file), so the
chain, the axes, the travel conversion and the mesh winding are all checkable and are
checked. Winding especially: the 3D viewer backface-culls from each triangle's own
normal, so an inward-wound primitive does not fail loudly -- it draws the inside of the
arm over the outside and reads as "the robot looks like shattered glass". A signed-volume
test catches that; an eyeball does not, reliably.

THE SECOND IS HONESTY ABOUT THE THINGS THAT ARE NOT KNOWN. Every link LENGTH in the X250
URDF is nominal, and every `zero_norm` offset is an assumption, because the arm is not on
the bus and nobody has put a ruler on it. The danger with a file like that is not that it
is wrong -- it is that six months from now it stops looking provisional and someone reads
a reach off it. So there are tests that the disclaimers are still present, and a test
that the profile's reach bounds are not mistaken for the SO-101's measured ones. A test
guarding a comment looks odd until you have watched a "nominal" constant get quietly
promoted to a measurement.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from rax.robots.arms.x250.normalise import (
    DEG_PER_TICK, FALLBACK_TICKS, X250Units, load_units)
from rax.robots.profiles import load_profile
from rax.robots.urdf_visuals import (
    box_mesh, cylinder_mesh, link_visuals, signed_volume, sphere_mesh)

URDF = (pathlib.Path(__file__).resolve().parents[1] / "src" / "rax" / "robots" /
        "arms" / "x250" / "X250" / "x250.urdf")


# ---------------------------------------------------------------------------------
class TestPrimitivesAreWoundOutward:
    """Signed volume > 0 iff every normal points out. See the module docstring."""

    def test_a_box_encloses_exactly_its_own_volume(self):
        V, F = box_mesh(0.2, 0.3, 0.4)
        assert signed_volume(V, F) == pytest.approx(0.2 * 0.3 * 0.4, rel=1e-12)

    def test_a_cylinder_encloses_its_volume_up_to_faceting(self):
        V, F = cylinder_mesh(0.1, 0.5)
        exact = math.pi * 0.1 ** 2 * 0.5
        vol = signed_volume(V, F)
        # A 16-gon inscribes ~97.4% of the circle's area; it must be under, never over.
        assert 0.95 * exact < vol < exact

    def test_a_sphere_encloses_its_volume_up_to_faceting(self):
        V, F = sphere_mesh(0.1, 32)
        exact = 4.0 / 3.0 * math.pi * 0.1 ** 3
        assert 0.95 * exact < signed_volume(V, F) < exact

    def test_flipping_one_face_is_detected(self):
        # The guard is only worth having if it actually fails on bad winding.
        V, F = box_mesh(0.2, 0.2, 0.2)
        good = signed_volume(V, F)
        F2 = F.copy()
        F2[0] = F2[0][::-1]
        assert signed_volume(V, F2) < good

    def test_every_x250_link_is_a_positively_wound_solid(self):
        vis = link_visuals(URDF)
        assert vis, "the X250 URDF produced no visual geometry at all"
        for name, (V, F) in vis.items():
            assert signed_volume(V, F) > 0.0, f"{name} is wound inside-out"

    def test_decimation_keeps_the_winding(self):
        """The dedupe that scrambles winding is the classic way to break this.

        Sorting a face's three indices canonicalises it for dedupe and destroys its
        orientation, so half the normals end up inward and the backface cull paints the
        inside of the arm over the outside. That was once blamed on the triangle budget.
        """
        from rax.robots.urdf_visuals import decimate
        V, F = sphere_mesh(0.1, 32)
        Vd, Fd = decimate(V, F, 0.02)
        assert len(Fd) < len(F), "decimation should actually remove triangles"
        assert signed_volume(Vd, Fd) > 0, "decimation flipped some normals"

    def test_decimation_shrinks_a_dense_mesh_a_lot(self):
        from rax.robots.urdf_visuals import decimate
        V, F = sphere_mesh(0.1, 64)
        _Vd, Fd = decimate(V, F, 0.03)
        assert len(Fd) < len(F) / 4

    def test_an_over_coarse_voxel_collapses_rather_than_corrupting(self):
        """Everything landing in ONE cell must give an empty face list, not garbage.

        The box has to be moved OFF a cell boundary to test this. Straddling zero its
        vertices floor to -1 and 0 on every axis, and eight cells survive however coarse
        the grid; the same happens at +100 with a voxel of 10, because 100 is a multiple
        of it. A voxel far larger than the offset puts every corner in cell 0.
        """
        from rax.robots.urdf_visuals import decimate
        V, F = box_mesh(0.01, 0.01, 0.01)
        Vd, Fd = decimate(V + 100.0, F, 1e6)
        assert len(Vd) == 1, "all eight corners should have snapped together"
        assert len(Fd) == 0, "every triangle is degenerate and must be dropped"
        assert Fd.shape[1:] == (3,), "an empty result still has to be a face array"

    def test_decimation_never_emits_an_out_of_range_index(self):
        from rax.robots.urdf_visuals import decimate
        for voxel in (0.005, 0.02, 0.05, 0.2):
            V, F = sphere_mesh(0.1, 32)
            Vd, Fd = decimate(V, F, voxel)
            if len(Fd):
                assert Fd.min() >= 0 and Fd.max() < len(Vd)

    def test_the_so101_meshes_are_decimated_to_something_drawable(self):
        # The SO-101's raw STLs are ~399k triangles against a viewer that redraws at 5 Hz
        # next to a camera stream. If this regresses, the 3D view becomes unusable on the
        # arm that HAS real meshes — the one case the primitive path cannot cover for.
        from rax.robots.urdf_visuals import link_visuals as lv
        p = load_profile("so101")
        try:
            vis = lv(p.urdf_path, mesh_dir=p.mesh_path)
        except Exception:
            pytest.skip("SO-101 meshes unavailable here")
        tris = sum(len(F) for _V, F in vis.values())
        if tris == 0:
            pytest.skip("no mesh loader installed")
        assert tris < 40_000, f"{tris} triangles — decimation is not being applied"
        for name, (V, F) in vis.items():
            if len(F):
                assert signed_volume(V, F) > 0, f"{name} is wound inside-out"

    def test_the_whole_arm_is_cheap_to_draw(self):
        # The browser redraws this at ~5 Hz alongside two camera streams. The decimated
        # SO-101 STLs cost ~6k triangles; primitives should cost far less, and if a
        # future edit makes a link 10x denser this is where it shows up.
        tris = sum(len(F) for _V, F in link_visuals(URDF).values())
        assert tris < 1500, f"{tris} triangles is more than a primitive arm should need"


# ---------------------------------------------------------------------------------
class TestTheUnitsConversionIsTheArmsOwn:
    def test_scale_comes_from_the_tick_range_and_nothing_else(self):
        u = X250Units({"elbow": (1243, 2469)})
        assert u.deg_per_norm("elbow") == pytest.approx((2469 - 1243) / 200.0 * DEG_PER_TICK)

    def test_the_gripper_is_one_sided(self):
        # 0..100 rather than -100..100, so the same tick span is twice the scale.
        u = X250Units({"gripper": (1002, 2681), "elbow": (1002, 2681)})
        assert u.deg_per_norm("gripper") == pytest.approx(2 * u.deg_per_norm("elbow"))

    def test_the_conversion_round_trips(self):
        u = load_units()
        for j in ("base", "shoulder_2", "elbow", "wrist", "tool"):
            for n in (-37.5, 0.0, 12.25):
                assert u.to_norm(j, u.to_deg(j, n)) == pytest.approx(n, abs=1e-9)

    def test_full_normalised_travel_is_the_calibrated_travel(self):
        u = X250Units({"shoulder_2": (825, 3006)})
        span_deg = u.to_deg("shoulder_2", 100.0) - u.to_deg("shoulder_2", -100.0)
        assert span_deg == pytest.approx((3006 - 825) * DEG_PER_TICK)

    def test_an_unranged_joint_is_reported_as_such(self):
        # base and wrist read 0..4095 -- the Dynamixel power-on default, i.e. nobody
        # ranged them. A caller has to be able to tell that from a measurement.
        u = load_units()
        assert u.unranged("base") is True
        assert u.unranged("wrist") is True
        assert u.unranged("shoulder_2") is False
        assert u.unranged("elbow") is False

    def test_a_missing_calibration_falls_back_rather_than_raising(self, tmp_path):
        # The 3D view must still draw on a machine that has never seen this arm.
        u = load_units(tmp_path / "nope.json")
        assert u.ticks == {k: (float(a), float(b)) for k, (a, b) in FALLBACK_TICKS.items()}

    def test_a_pose_missing_a_joint_still_converts_rather_than_raising(self):
        # The viewer must keep drawing when an observation arrives short a key. A missing
        # joint is treated as normalised 0, which lands at that joint's URDF angle for
        # normalised zero -- NOT necessarily 0 deg, because ZERO_NORM is offset.
        u = load_units()
        got = u.pose_to_deg({"base": 10.0}, ["base", "elbow"])
        assert len(got) == 2
        assert all(math.isfinite(v) for v in got)
        assert got[1] == pytest.approx(u.to_deg("elbow", 0.0))


# ---------------------------------------------------------------------------------
class TestTheProfileAndItsKinematics:
    def test_it_loads_and_resolves(self):
        p = load_profile("x250")
        assert p.name == "x250"
        assert p.joint_names == ("base", "shoulder_2", "elbow", "wrist", "tool")

    def test_the_topology_is_pan_plus_a_parallel_pitch_chain_plus_roll(self):
        # This is what makes ik="pitch_hold" legitimate, and it is a claim about the
        # hardware: shoulder_2, elbow and wrist must share one axis in the URDF.
        p = load_profile("x250")
        assert p.ik == "pitch_hold"
        assert p.pan_joint == 0 and p.pitch_chain == (1, 2, 3) and p.roll_joint == 4

        import xml.etree.ElementTree as ET
        axes = {}
        for j in ET.parse(URDF).getroot().findall("joint"):
            a = j.find("axis")
            if a is not None:
                axes[j.get("name")] = tuple(float(v) for v in a.get("xyz").split())
        assert axes["shoulder_2"] == axes["elbow"] == axes["wrist"], (
            "the pitch chain is only parallel if these three share an axis")
        assert axes["base"] != axes["shoulder_2"]

    def test_the_demonstrated_envelope_fits_inside_the_physical_travel(self):
        # If it did not, either the envelope or the URDF limits would be wrong, and the
        # arm would be clamped to a box it cannot reach.
        from rax.robots.profiles.urdf_limits import read_urdf_limits
        p = load_profile("x250")
        urdf = read_urdf_limits(p.urdf_path)
        lo, hi = p.limits()
        for i, n in enumerate(p.joint_names):
            ul, uh = urdf[n]
            assert ul - 1e-6 <= lo[i] <= hi[i] <= uh + 1e-6, f"{n} envelope escapes travel"

    def test_THE_URDF_LIMITS_ARE_THE_CALIBRATED_TRAVEL_ABOUT_THE_FITTED_ZERO(self):
        """The coupling this got wrong once, now pinned.

        A URDF <limit> is an angle about THAT URDF's zero. ZERO_NORM offsets that zero away
        from the centre of the calibrated tick range, so the travel window is asymmetric.
        Written symmetric (the obvious way) it put the demonstrated shoulder_2 envelope
        9.6 deg outside the travel the arm supposedly had -- i.e. the description
        contradicted itself, and the profile's own consistency check caught it.

        So if ZERO_NORM is ever refitted, this test fails until the URDF is regenerated,
        which is exactly the reminder that was missing.
        """
        from rax.robots.profiles.urdf_limits import read_urdf_limits
        p = load_profile("x250")
        urdf = read_urdf_limits(p.urdf_path)
        u = load_units()
        for j in p.joint_names:
            lo, hi = urdf[j]
            # Rounded OUTWARD, so the file's window contains the computed one. Never
            # inward: that would clamp away travel the arm really has.
            assert lo <= u.to_deg(j, -100.0) + 1e-6, f"{j} lower limit rounded inward"
            assert hi >= u.to_deg(j, +100.0) - 1e-6, f"{j} upper limit rounded inward"
            # ...and not by more than rounding, or it is not the travel any more.
            assert lo > u.to_deg(j, -100.0) - 0.01, f"{j} lower limit is not the travel"
            assert hi < u.to_deg(j, +100.0) + 0.01, f"{j} upper limit is not the travel"

    def test_the_offsets_are_reproduced_by_the_fitter_that_produced_them(self):
        # ZERO_NORM is a committed result of fit_zero_norm against the demonstrated grasp
        # pose. If the URDF's link lengths change, the fit moves and the table is stale --
        # this is what says so.
        from rax.robots.arms.x250.normalise import ZERO_NORM, fit_zero_norm
        got = fit_zero_norm(URDF)
        for k, v in got.items():
            assert v == pytest.approx(ZERO_NORM[k], abs=0.02), (
                f"{k}: the committed offset {ZERO_NORM[k]} no longer matches the fit "
                f"{v} -- re-run fit_zero_norm and update ZERO_NORM")

    def test_the_fit_puts_the_demonstrated_grasp_on_the_bench_pointing_down(self):
        # The physical claim the offsets are fitted to, checked through the real FK.
        from rax.manipulation.arms.urdf_kinematics import UrdfKinematics
        from rax.robots.arms.x250.normalise import GRASP_FACT, GRASP_NORM
        p = load_profile("x250")
        u = load_units()
        k = UrdfKinematics(p.urdf_path, ee_frame=p.ee_frame,
                           joint_names=list(p.joint_names))
        q = np.array(u.pose_to_deg(GRASP_NORM, p.joint_names))
        tip = np.asarray(k.forward_kinematics(q))[:3, 3]
        want_r, want_z, want_pitch = GRASP_FACT
        assert tip[0] == pytest.approx(want_r, abs=0.005)
        assert tip[2] == pytest.approx(want_z, abs=0.005)
        assert -(q[1] + q[2] + q[3]) == pytest.approx(want_pitch, abs=0.5)

    def test_the_look_pose_is_raised_and_retracted_which_was_NOT_fitted(self):
        """The independent check that the fit is not nonsense.

        Only the grasp pose was fitted. pick.py's README describes `look` in its own words
        as "the retracted `look` pose (raised, not reached out)", and the fitted offsets
        have to reproduce that for free or they are just three numbers that satisfy three
        equations.
        """
        from rax.manipulation.arms.urdf_kinematics import UrdfKinematics
        from rax.robots.profiles.x250 import GRASP_NORM, LOOK_FAR_NORM, LOOK_NORM
        p = load_profile("x250")
        u = load_units()
        k = UrdfKinematics(p.urdf_path, ee_frame=p.ee_frame,
                           joint_names=list(p.joint_names))

        def tip(pose):
            return np.asarray(k.forward_kinematics(
                np.array(u.pose_to_deg(pose, p.joint_names))))[:3, 3]

        look, far, grasp = tip(LOOK_NORM), tip(LOOK_FAR_NORM), tip(GRASP_NORM)
        assert look[2] > 0.35, "'raised' — the look pose should be well up"
        assert np.hypot(look[0], look[1]) < 0.12, "'not reached out'"
        # look -> look_far -> grasp should march outward and downward, which is what those
        # three names mean. None of look or look_far took part in the fit.
        rs = [np.hypot(t[0], t[1]) for t in (look, far, grasp)]
        assert rs[0] < rs[1] < rs[2], f"reach should increase along look/far/grasp: {rs}"
        assert grasp[2] < far[2] and grasp[2] < look[2]

    def test_forward_kinematics_reaches_forward_and_pans_with_the_base(self):
        from rax.manipulation.arms.urdf_kinematics import UrdfKinematics
        p = load_profile("x250")
        k = UrdfKinematics(p.urdf_path, ee_frame=p.ee_frame,
                           joint_names=list(p.joint_names))
        tip0 = np.asarray(k.forward_kinematics(np.zeros(5)))[:3, 3]
        assert tip0[0] > 0.2, "at the zero pose the hand should be out in front"
        assert abs(tip0[1]) < 1e-9, "and on the centre line"

        # 45 deg of base yaw must swing it, and preserve radius.
        q = np.array([45.0, 0, 0, 0, 0])
        tip45 = np.asarray(k.forward_kinematics(q))[:3, 3]
        assert np.hypot(*tip45[:2]) == pytest.approx(np.hypot(*tip0[:2]), abs=1e-9)
        assert tip45[1] > 0.1, "positive base should swing to +Y"

    def test_the_chain_covers_every_link_the_viewer_wants_to_pose(self):
        # /geom streams a 4x4 per chain link and the page looks each up by name. A link
        # with geometry but no transform silently vanishes from the render.
        from rax.manipulation.arms.urdf_kinematics import UrdfKinematics
        p = load_profile("x250")
        k = UrdfKinematics(p.urdf_path, ee_frame=p.ee_frame,
                           joint_names=list(p.joint_names))
        posed = {n for n, _T in k.get_link_transforms_chain(np.zeros(5))}
        drawn = set(link_visuals(p.urdf_path))
        missing = drawn - posed
        # The fingers are deliberately off-chain (one actuator drives both), exactly as
        # the SO-101's moving jaw is; the viewer composes them from gripper_link.
        assert missing == {"left_finger_link", "right_finger_link"}, (
            f"unexpected links absent from the FK chain: {missing}")


# ---------------------------------------------------------------------------------
class TestItStaysHonestAboutWhatIsNotMeasured:
    """Guards on the disclaimers. See the module docstring for why these exist."""

    def test_the_urdf_says_its_link_lengths_are_nominal(self):
        text = URDF.read_text().lower()
        assert "nominal" in text
        assert "kinematic sketch" in text

    def test_the_units_module_says_the_offsets_are_assumed(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / "src" / "rax" / "robots" /
               "arms" / "x250" / "normalise.py").read_text().lower()
        assert "assum" in src, "zero_norm being an assumption must stay documented"

    def test_the_reach_bounds_are_not_the_so101s_measured_ones(self):
        # The SO-101's reach_grasp_max_m came out of a 2 cm IK sweep. Copying it across
        # would hand the X250 a measured-looking number nobody measured.
        x, so = load_profile("x250"), load_profile("so101")
        assert x.reach_grasp_max_m != so.reach_grasp_max_m
        assert x.reach_max_m != so.reach_max_m

    def test_the_x250_claims_no_hand_eye_calibration(self):
        # The servo seam exists precisely so this arm needs none; an invented extrinsic
        # would be a number that looks calibrated and is not.
        p = load_profile("x250")
        assert p.camera.calibration_file is None
        assert p.camera.extrinsics == "0,0,0,0,0,0"
        assert p.camera.use_depth is False
