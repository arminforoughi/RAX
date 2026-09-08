"""The staged approach's commit rule: the last stage only spends the last of the
margin once a close-up sighting has confirmed the target.

Pinned because the failure it prevents is invisible from inside the loop. A pick where
every re-measure silently failed still drove all of its stages, the last one at full
distance, and finished past the object with it out of frame — the arm looked like it
was closing in on something it had checked, and had checked nothing.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rax.manipulation.approach import (  # noqa: E402
    ApproachConfig,
    approach_target,
    cap_reach,
    stage_step,
)

CFG = ApproachConfig()


def _march(target, *, confirmed, tip=(0.20, 0.0), steps=None):
    """Walk the staging to its end, returning the tip's path. `confirmed` says whether
    a sighting agreed with the target before the last stage."""
    steps = CFG.steps if steps is None else steps
    tip = np.asarray(tip, np.float64)
    path = [tip]
    for i in range(steps):
        waypoint, remaining = stage_step(tip, target, stage=i, total=steps,
                                         first_frac=CFG.first_step_frac,
                                         max_first_m=CFG.max_first_step_m,
                                         commit=confirmed)
        if waypoint is None or remaining < CFG.arrived_m:
            break
        tip = waypoint
        path.append(tip)
    return path


def test_confirmed_approach_arrives_at_the_hover():
    target = np.array([0.40, 0.10])
    end = _march(target, confirmed=True)[-1]
    assert np.linalg.norm(end - target) < CFG.arrived_m


def test_unconfirmed_approach_always_stops_short():
    """Never seen up close: the arm may close in, but not onto the estimate."""
    target = np.array([0.40, 0.10])
    path = _march(target, confirmed=False)
    left = float(np.linalg.norm(path[-1] - target))
    assert left > 0.0, "an unconfirmed approach must not land exactly on the estimate"
    # It still makes progress — stopping short is not the same as refusing to move.
    assert np.linalg.norm(path[-1] - path[0]) > np.linalg.norm(target - path[0]) * 0.5


def test_holding_back_errs_on_the_near_side():
    """Short of the object, never past it — the direction of the error is the point."""
    tip = np.array([0.20, 0.0])
    target = np.array([0.40, 0.0])
    end = _march(target, confirmed=False, tip=tip)[-1]
    assert end[0] < target[0], "stopped past the target with nothing confirming it"


def test_commit_only_changes_the_last_stage():
    tip, target = np.array([0.20, 0.0]), np.array([0.40, 0.10])
    for stage in range(CFG.steps - 1):
        a, _ = stage_step(tip, target, stage=stage, total=CFG.steps, commit=True)
        b, _ = stage_step(tip, target, stage=stage, total=CFG.steps, commit=False)
        assert np.allclose(a, b)


def test_the_hover_stays_short_of_the_object_and_to_its_right():
    """The offset the stages aim at: radially short, and off to one side so the object
    stays in frame instead of vanishing under the gripper."""
    obj = np.array([0.35, 0.20])
    hover = approach_target(obj, back_m=CFG.back_m, right_trim_m=CFG.right_trim_m)
    assert np.hypot(*hover) < np.hypot(*obj)
    # Right of the object as seen from the base means a negative 2D cross product
    # (spelled out — np.cross on 2-vectors is deprecated).
    off = hover - obj
    assert float(obj[0] * off[1] - obj[1] * off[0]) < 0.0


def test_refine_gains_are_damped_but_effective():
    """A gain of 0 ignores every look; one above 1 overshoots every correction. And the
    first look counts for more than the ones after it — that is the whole ordering."""
    for gain in (CFG.first_refine_gain, CFG.refine_gain):
        assert 0.0 < gain <= 1.0
    assert CFG.first_refine_gain > CFG.refine_gain


def _refine(cube_xy, read_xy, *, r_cap, first):
    """One stage of the loop's refine arithmetic: cap the reach, then damp the move."""
    read_xy = cap_reach(read_xy, r_cap)
    gain = CFG.first_refine_gain if first else CFG.refine_gain
    return cube_xy + (read_xy - cube_xy) * gain


# The reads a real pick actually took, recovered from its log by undoing the 0.7 damping
# it applied. The cube was at (36.1, 24.2) — r = 43.5cm — and every one of these reads
# put it further out than that.
MEASURED_READS = [np.array(p) for p in ((0.349, 0.309), (0.349, 0.281), (0.378, 0.283))]
MEASURED_START = np.array([0.295, 0.285])     # what the map said: r = 41.0cm
MEASURED_TRUTH = np.array([0.361, 0.242])     # where the cube turned out to be


def test_the_measured_run_no_longer_walks_past_the_object():
    """Replay of the pick that prompted this: r went 41.0 -> 44.9 -> 46.5 while the cube
    sat at 43.5, and the arm ended up past it."""
    cube = MEASURED_START.copy()
    r_cap = float(np.hypot(*cube)) + CFG.max_refine_out_m
    radii = []
    for i, read in enumerate(MEASURED_READS):
        cube = _refine(cube, read, r_cap=r_cap, first=(i == 0))
        radii.append(float(np.hypot(*cube)))
    assert max(radii) <= r_cap + 1e-9, f"still creeping outward: {radii}"
    # and it should be a better answer than the map it started from, not just a safer one
    before = float(np.linalg.norm(MEASURED_START - MEASURED_TRUTH))
    after = float(np.linalg.norm(cube - MEASURED_TRUTH))
    assert after < before, f"refining left it worse: {before*100:.1f} -> {after*100:.1f}cm"


def test_no_sequence_of_long_reads_can_walk_the_target_out():
    """The cap is on the total, not per stage — the failure was the accumulation."""
    cube = np.array([0.30, 0.20])
    r_cap = float(np.hypot(*cube)) + CFG.max_refine_out_m
    for i in range(12):
        runaway = cube * 1.5           # every read says "further out than you think"
        cube = _refine(cube, runaway, r_cap=r_cap, first=(i == 0))
        assert float(np.hypot(*cube)) <= r_cap + 1e-9


def test_capping_the_reach_keeps_the_bearing():
    read = np.array([0.30, 0.40])      # r = 50cm
    held = cap_reach(read, 0.40)
    assert abs(float(np.hypot(*held)) - 0.40) < 1e-9
    assert abs(float(np.arctan2(held[1], held[0]))
               - float(np.arctan2(read[1], read[0]))) < 1e-9


def test_reads_that_are_near_enough_pass_through_untouched():
    read = np.array([0.30, 0.10])
    assert np.allclose(cap_reach(read, 0.40), read)


def test_pulling_the_target_inward_is_never_capped():
    """Stopping short is recoverable — the object stays in view and the jaws are open.
    Driving past it is not. Only the outward half is held."""
    cube = np.array([0.40, 0.0])
    r_cap = float(np.hypot(*cube)) + CFG.max_refine_out_m
    near = np.array([0.30, 0.0])
    assert np.allclose(cap_reach(near, r_cap), near)


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


# --- trim decay across the stages ----------------------------------------------------
# The trim parks the hover to the object's side so it does not vanish under the gripper
# in transit. Held fixed, it is also a lateral offset the centring servo must undo at the
# end — ~250 px at grasp range, handed to a loop capped at 4.5 deg of pan per iteration.
# The approach converging perfectly still left that error standing by construction.

from rax.manipulation.approach import stage_trim  # noqa: E402


def test_trim_is_full_on_the_first_stage():
    """The operator's number is what the first hop uses — the decay only takes away
    later, where the offset has stopped buying visibility."""
    assert stage_trim(0.05, stage=0, total=3) == 0.05


def test_trim_decays_to_the_configured_fraction_by_the_last_stage():
    got = stage_trim(0.05, stage=2, total=3, final_frac=0.3)
    assert abs(got - 0.015) < 1e-12


def test_trim_decays_monotonically():
    vals = [stage_trim(0.05, stage=i, total=5) for i in range(5)]
    assert all(a >= b for a, b in zip(vals, vals[1:])), vals


def test_a_single_stage_approach_keeps_the_full_trim():
    """With no later hop to decay toward, the visibility offset is all there is."""
    assert stage_trim(0.05, stage=0, total=1) == 0.05


def test_final_frac_of_one_restores_the_old_fixed_behaviour():
    assert all(stage_trim(0.05, stage=i, total=3, final_frac=1.0) == 0.05
               for i in range(3))


def test_the_decayed_trim_shrinks_what_the_servo_must_undo():
    """The point of the whole change, stated as the number that matters.

    The centring servo inherits the trim as pixel error. Whatever the camera's scale,
    the last stage must hand it strictly less than the first.
    """
    first = stage_trim(CFG.right_trim_m, stage=0, total=CFG.steps,
                       final_frac=CFG.trim_final_frac)
    last = stage_trim(CFG.right_trim_m, stage=CFG.steps - 1, total=CFG.steps,
                      final_frac=CFG.trim_final_frac)
    assert last < first
    # And it must not go to zero: the object still has to stay out from under the jaws.
    assert last > 0.0


def test_stages_still_reach_the_hover_when_the_trim_moves_under_them():
    """The hover target shifts every stage as the trim decays, so the staging has to
    converge on a MOVING point. It does, because each stage recomputes from the current
    tip — pinned so a future change to stage_step cannot quietly break it."""
    cube = np.array([0.40, 0.10])
    tip = np.array([0.20, 0.0])
    for i in range(CFG.steps):
        trim = stage_trim(CFG.right_trim_m, stage=i, total=CFG.steps,
                          final_frac=CFG.trim_final_frac)
        target = approach_target(cube, back_m=CFG.back_m, right_trim_m=trim)
        waypoint, remaining = stage_step(tip, target, stage=i, total=CFG.steps,
                                         first_frac=CFG.first_step_frac,
                                         max_first_m=CFG.max_first_step_m, commit=True)
        if waypoint is None or remaining < CFG.arrived_m:
            break
        tip = waypoint
    final_target = approach_target(cube, back_m=CFG.back_m,
                                   right_trim_m=stage_trim(
                                       CFG.right_trim_m, stage=CFG.steps - 1,
                                       total=CFG.steps,
                                       final_frac=CFG.trim_final_frac))
    assert np.linalg.norm(tip - final_target) < CFG.arrived_m * 2


# --- the lateral offset must be held, not re-applied at the hover -------------------
# Reported from the rig: "it still goes from middle, then goes to right, which makes it
# push the object away most of the time". The trim decayed to ~30% of itself, walking
# the gripper onto the object's centre line, and the centring servo then pushed it back
# out sideways at the hover — with the jaws already beside the object, which is the
# worst moment for a lateral move. The approach and the servo must agree on ONE offset.

from rax.manipulation.approach.derive import grasp_bias_m  # noqa: E402
from rax.manipulation.approach.geometry import stage_trim  # noqa: E402

REF_CUBE_M = 0.0508


def _trims(total=3, right_trim_m=0.05, final_m=None):
    return [stage_trim(right_trim_m, stage=i, total=total, final_m=final_m)
            for i in range(total)]


def test_the_trim_never_falls_below_the_grasp_offset():
    bias = grasp_bias_m(REF_CUBE_M)
    for total in (2, 3, 4, 6):
        for t in _trims(total, final_m=bias):
            assert t >= bias - 1e-9, f"trim {t:.4f} dipped under the grasp bias {bias:.4f}"


def test_the_last_stage_lands_exactly_on_the_grasp_offset():
    """So the centring servo has no sideways correction left to make."""
    bias = grasp_bias_m(REF_CUBE_M)
    assert abs(_trims(3, final_m=bias)[-1] - bias) < 1e-9


def test_the_first_stage_still_gets_the_full_visibility_trim():
    """The wide offset early is what keeps the object in frame; only the floor moved."""
    assert abs(_trims(3, final_m=grasp_bias_m(REF_CUBE_M))[0] - 0.05) < 1e-9


def test_the_trim_decreases_monotonically_toward_the_grasp():
    ts = _trims(4, final_m=grasp_bias_m(REF_CUBE_M))
    for a, b in zip(ts, ts[1:]):
        assert b <= a + 1e-9, f"trim rose mid-approach: {ts}"


def test_the_old_fraction_decay_dipped_below_the_grasp_offset():
    """Pins the defect: without a floor the last stage sat well inside the bias, which
    is the sideways move the servo then had to make next to the object."""
    bias = grasp_bias_m(REF_CUBE_M)
    old_last = stage_trim(0.05, stage=2, total=3, final_frac=0.3)
    assert old_last < bias, f"old last-stage trim {old_last:.4f} vs bias {bias:.4f}"


def test_a_single_stage_approach_is_the_grasp_and_takes_the_bias():
    bias = grasp_bias_m(REF_CUBE_M)
    assert abs(stage_trim(0.05, stage=0, total=1, final_m=bias) - bias) < 1e-9
