"""Tests for deriving an arm's workspace instead of hand-measuring it.

The thing under test replaces hand-typed constants — a grasp-pitch table, reach limits,
and IK seeds someone added one at a time while watching an arm fail. So the tests check
the two properties that make it a replacement rather than a rewrite: it recovers the
physics (shallower angles reach further, and the numbers match what was measured by
hand), and it derives seeds from where the solver actually fails rather than from a list.

    python tests/test_workspace.py
    pytest tests/test_workspace.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manipulation.arms.ik_strategy import make_ik  # noqa: E402
from manipulation.arms.workspace import (  # noqa: E402
    ReachResult, WorkspaceMap, analyze_workspace, default_grid)
from robots.profiles import load_profile  # noqa: E402

#: What a human measured on this arm and typed into a comment, pitch -> reach in cm.
HAND_MEASURED = {90: 30.8, 70: 36.4, 60: 39.8, 50: 42.7,
                 40: 45.2, 30: 46.2, 20: 47.0, 10: 47.5, 0: 47.8}

_CACHE: dict[str, WorkspaceMap] = {}


def _so101():
    from lerobot.model.kinematics import RobotKinematics

    p = load_profile("so101")
    kin = RobotKinematics(p.urdf_path, p.ee_frame, list(p.joint_names))
    return p, make_ik(kin, p)


def _coarse_map() -> WorkspaceMap:
    """A deliberately small probe so the suite stays quick; the full sweep is a
    one-off per arm, not something a test should redo."""
    if "coarse" not in _CACHE:
        p, ik = _so101()
        grid = {"radii": (0.12, 0.20, 0.28, 0.36, 0.44),
                "heights": (0.02,),
                "pitches": (90.0, 70.0, 50.0, 30.0, 10.0)}
        _CACHE["coarse"] = analyze_workspace(ik, p, grid=grid)
    return _CACHE["coarse"]


def test_shallower_angles_reach_further():
    """The core physical fact the hand-measured table encodes. If this inverts, the
    probe is measuring something other than reach."""
    ws = _coarse_map()
    env = ws.reach_envelope(0.02)
    steep, shallow = env[90.0], env[10.0]
    assert shallow > steep, f"10deg reached {shallow}, 90deg reached {steep}"
    # and it should be monotonic-ish across the range, not noise
    ordered = [env[p] for p in (90.0, 70.0, 50.0, 30.0, 10.0)]
    assert ordered == sorted(ordered), f"reach not monotonic in pitch: {ordered}"


def test_derived_envelope_matches_what_was_measured_by_hand():
    """The claim that makes this a replacement: it rediscovers the numbers."""
    ws = _coarse_map()
    step = 0.08          # this coarse grid's radius step
    for pitch in (90.0, 70.0, 50.0, 30.0, 10.0):
        got = ws.max_reach(pitch, 0.02)
        want = HAND_MEASURED[int(pitch)] / 100.0
        # the probe reports the furthest GRID cell that solved, so it under-reports by
        # up to one step and can never over-report by more than rounding
        assert want - step - 1e-9 <= got <= want + step, (
            f"pitch {pitch:.0f}: derived {got*100:.1f}cm vs hand {want*100:.1f}cm")


def test_working_radii_replace_hand_set_limits():
    ws = _coarse_map()
    lo, hi = ws.working_radii(0.02)
    assert 0.05 < lo < 0.25 and 0.30 < hi < 0.60, (lo, hi)
    assert lo < hi


def test_pitch_candidates_are_feasible_and_steepest_first():
    """Replaces a hardcoded ordered list: the order is a preference (steep grips a
    table object best), the membership is a fact about this arm."""
    ws = _coarse_map()
    near = ws.pitch_candidates(np.array([0.20, 0.0, 0.02]))
    assert near, "nothing reachable at r=20cm, which cannot be right"
    assert list(near) == sorted(near, reverse=True), "not steepest-first"
    far = ws.pitch_candidates(np.array([0.44, 0.0, 0.02]))
    # a far target must not be offered a steep angle the arm cannot hold there
    assert 90.0 not in far, f"90deg offered at r=44cm: {far}"
    assert max(near) >= max(far), "near targets should allow steeper grasps than far ones"


def test_unreachable_pitch_reports_zero_not_a_guess():
    ws = _coarse_map()
    assert ws.max_reach(999.0, 0.02) == 0.0


def test_seeds_are_valid_for_the_profile():
    """Derived seeds must be usable as profile ik_seeds — right length, and only the
    joints the solver drives are pinned."""
    p = load_profile("so101")
    ws = _coarse_map()
    for seed in ws.seeds():
        assert len(seed) == p.n_joints
        pinned = {i for i, v in enumerate(seed) if v is not None}
        assert pinned <= set(p.positioning_joints), (
            f"seed pins {pinned}, which is not a subset of the driven joints")
        lo, hi = p.limits()
        for i in pinned:
            assert lo[i] <= seed[i] <= hi[i], f"seed joint {i}={seed[i]} outside limits"


def test_seed_selection_is_a_cover_not_an_accumulation():
    """Every chosen seed must rescue something no earlier one did — otherwise the list
    grows the same way the hand-written one did."""
    ws = _coarse_map()
    if not ws.chosen_seeds:
        return          # this arm's bare solver needed no help on this grid
    rescued = {}
    for x in ws.results:
        if x.reached and x.seed_index >= 0:
            rescued.setdefault(x.seed_index, set()).add((x.r_m, x.z_m, x.pitch_deg))
    covered = set()
    for i in ws.chosen_seeds:
        gain = rescued.get(i, set()) - covered
        assert gain, f"seed {i} was chosen but rescues nothing new"
        covered |= gain


def test_dead_bands_are_where_a_seed_was_needed():
    ws = _coarse_map()
    for band in ws.dead_bands():
        assert band["r_min_m"] <= band["r_max_m"]
        assert band["cells"] >= 2
        assert any(x.seed_index >= 0 and x.reached
                   and abs(x.pitch_deg - band["pitch_deg"]) < 1e-9 for x in ws.results)


def test_round_trips_through_json():
    """It is cached per arm, so a reload must reproduce the same answers."""
    ws = _coarse_map()
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "ws.json")
        ws.save(path)
        back = WorkspaceMap.load(path)
    assert back.profile_name == ws.profile_name
    assert len(back.results) == len(ws.results)
    assert back.reach_envelope(0.02) == ws.reach_envelope(0.02)
    assert back.seeds() == ws.seeds()


def test_default_grid_covers_past_the_nominal_reach():
    """The envelope has to be discovered, so the grid must extend beyond the profile's
    declared limit rather than stopping at it and confirming what it was told."""
    p = load_profile("so101")
    g = default_grid(p)
    assert max(g["radii"]) > p.reach_max_m
    assert min(g["radii"]) <= p.reach_min_m + 0.01
    assert 0.0 in g["pitches"] and 90.0 in g["pitches"]


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
