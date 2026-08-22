"""Tests for the final centring servo, against a simulated arm.

This is control logic, so golden values cannot prove it — what matters is whether the
loop converges, and whether it does something sane when it cannot. The simulated arm
below has a deliberately awkward property: its pixel gains are INVERTED relative to the
naive expectation, which is exactly what a hand-eye transform carrying roll or yaw error
does on a real rig. A servo with hand-modelled signs drives the wrong way and stalls; one
that measures its gains by probing does not care.

    python tests/test_visual_center.py
    pytest tests/test_visual_center.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rax.manipulation.approach import ApproachConfig  # noqa: E402
from rax.manipulation.approach.visual_center import CenteringOps, center_on_object  # noqa: E402

AIM = (440.0, 394.0)


class _Track:
    def __init__(self, uv):
        self.uv = uv


class FakeArm:
    """A tip that moves, and an object whose pixel is a linear function of the pose.

    Signs are inverted on purpose (see the module docstring). ``lost_after`` makes the
    object disappear mid-loop, and ``blocked`` makes every reach unreachable.
    """

    def __init__(self, *, du_dpan=-6.0, dv_dr=-900.0, pan=0.0, radius=0.30,
                 err_u=120.0, err_v=80.0, lost_after=None, blocked=False,
                 range_m=0.30):
        self.du_dpan, self.dv_dr = du_dpan, dv_dr
        self.pan, self.radius = pan, radius
        self.err_u, self.err_v = err_u, err_v      # current pixel error from AIM
        self.pan0, self.radius0 = pan, radius
        self.lost_after, self.blocked = lost_after, blocked
        self._range = range_m
        self.log: list[str] = []
        self.tracks = 0
        self.moves = 0

    # --- the CenteringOps surface ------------------------------------------
    def joints(self):
        return np.array([self.pan, 0.0, 0.0, 0.0, 0.0])

    def tip(self, q=None):
        return np.array([self.radius, 0.0, 0.05])

    def track(self, tries=3):
        self.tracks += 1
        if self.lost_after is not None and self.tracks > self.lost_after:
            return None
        u = AIM[0] + self.err_u + (self.pan - self.pan0) * self.du_dpan
        v = AIM[1] + self.err_v + (self.radius - self.radius0) * self.dv_dr
        return _Track((u, v))

    def range_m(self, track):
        return self._range

    def move_pan(self, delta_deg, *, settle, step):
        self.moves += 1
        self.pan += float(delta_deg)

    def move_tip(self, p_base, *, settle, step):
        if self.blocked:
            return False
        self.moves += 1
        self.radius = float(np.hypot(p_base[0], p_base[1]))
        return True

    def say(self, msg):
        self.log.append(msg)

    def checkpoint(self):
        pass

    # --- for assertions ----------------------------------------------------
    def pixel_error(self):
        t = self.track()
        self.tracks -= 1        # do not let the assertion perturb lost_after
        return abs(t.uv[0] - AIM[0]), abs(t.uv[1] - AIM[1])


def test_ops_protocol_is_satisfied():
    assert isinstance(FakeArm(), CenteringOps)


def test_servo_converges_despite_inverted_gains():
    """The whole point of probing: signs that would break a hand-modelled loop."""
    arm = FakeArm()
    before = arm.pixel_error()
    res = center_on_object(arm, AIM, ApproachConfig())
    after = arm.pixel_error()
    assert res.centered, f"did not converge: {res.reason}; log={arm.log}"
    assert after[0] < before[0] and after[1] < before[1]
    cfg = ApproachConfig()
    assert after[0] < cfg.align_tol_px and after[1] < cfg.align_tol_px * 1.3


def test_servo_measures_its_gains_rather_than_assuming_them():
    """Flip the sign of both gains and it must still converge, unchanged."""
    for du, dv in ((-6.0, -900.0), (6.0, 900.0), (-6.0, 900.0), (6.0, -900.0)):
        arm = FakeArm(du_dpan=du, dv_dr=dv)
        res = center_on_object(arm, AIM, ApproachConfig())
        assert res.centered, f"gains ({du}, {dv}) failed: {res.reason}"


def test_already_centred_costs_no_moves():
    arm = FakeArm(err_u=2.0, err_v=2.0)
    res = center_on_object(arm, AIM, ApproachConfig())
    assert res.centered and res.iterations == 1
    # Only the two probe moves plus their returns; no corrective motion.
    assert arm.moves <= 4


def test_a_useless_pan_gain_is_refused_not_divided_by():
    """If rotating the base barely moves the object, the gain is noise — dividing by
    it produces a wild step. Report and stop instead."""
    arm = FakeArm(du_dpan=0.2)
    res = center_on_object(arm, AIM, ApproachConfig())
    assert not res.centered and res.reason == "pan_probe_failed"
    assert any("barely moves" in m for m in arm.log)
    assert res.xy is not None, "it should still report where the tip ended up"


def test_object_not_in_view_is_reported():
    arm = FakeArm(lost_after=0)
    res = center_on_object(arm, AIM, ApproachConfig())
    assert not res.centered and res.reason == "not_in_view" and res.xy is None


def test_losing_the_object_mid_loop_stops_cleanly():
    """The object usually disappears because it is under the jaws — that is not a
    failure to retry, it is a reason to stop and let the descent happen."""
    # Three tracks are consumed before the loop starts: the initial fix and one per
    # probe. Losing it on the fourth puts the disappearance inside the servo loop.
    arm = FakeArm(lost_after=4)
    res = center_on_object(arm, AIM, ApproachConfig())
    assert not res.centered and res.reason == "lost", f"{res.reason}; log={arm.log}"
    assert any("stopping" in m for m in arm.log)


def test_unreachable_radial_moves_do_not_spin_forever():
    arm = FakeArm(blocked=True, err_u=0.0, err_v=200.0)
    res = center_on_object(arm, AIM, ApproachConfig())
    assert not res.centered
    assert res.iterations <= ApproachConfig().align_iters


def test_close_range_moves_more_gently():
    """Speed scales with range: the base carries the arm's inertia, so a near object
    must be approached with smaller steps."""
    far = FakeArm(range_m=0.40)
    near = FakeArm(range_m=0.05)
    center_on_object(far, AIM, ApproachConfig())
    center_on_object(near, AIM, ApproachConfig())
    # With a tighter per-iteration cap, the close arm cannot swing as far in one step.
    assert abs(near.pan - near.pan0) <= 4.5 * ApproachConfig().align_iters + 1e-9


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
