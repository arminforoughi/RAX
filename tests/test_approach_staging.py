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
