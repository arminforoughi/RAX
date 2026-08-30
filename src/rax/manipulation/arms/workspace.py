"""Deriving an arm's reachable workspace instead of hand-measuring it.

A pile of constants in this stack are really one fact — *what this arm can reach, at
what tool angle* — discovered by a human driving the robot and typing the answers in:

* an ordered list of grasp pitches to try, with a comment recording "90deg -> 30.8cm,
  70 -> 36.4, 60 -> 39.8, ... 0 -> 47.8" measured by hand;
* a minimum and maximum working radius;
* a handful of IK seed poses, added one at a time to escape a dead band somebody found
  by watching the arm fail;
* scattered notes like "measured achievable pitch at z=2cm: r=10cm -> 85..95 only".

Every one of those is a *derivable property of the URDF and the solver*, and none of
them transfer to a different arm. So derive them: probe the (radius, height, pitch)
space with the arm's own IK, and keep what solved.

That turns arm-specific folklore into a computation which any robot can run on itself:

    ws = analyze_workspace(ik, profile)      # once, cached to JSON
    ws.max_reach(pitch_deg=70)               # -> the real number, for THIS arm
    ws.pitch_candidates(p_target)            # -> feasible pitches, steepest first
    ws.seeds()                               # -> seeds chosen by set cover, not by hand
    ws.dead_bands()                          # -> where a single seed is not enough

The seed derivation is the part that most repays being automatic. A dead band is a
region where the solver's natural branch cannot reach a pose that IS reachable, so
escaping it needs a seed from a different branch. Rather than waiting for someone to
notice, this samples a pool of seeds across the joint space, records which targets each
one rescues, and greedily picks the smallest set that covers the workspace.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ["WorkspaceMap", "ReachResult", "analyze_workspace", "default_grid"]

#: A solve counts as reached when the tip lands this close. Matches the tolerance the
#: approach itself accepts; a pose worse than this is not a pose the arm can hold.
REACH_TOL_M = 0.004
#: Shallow / far-reach poses are legitimately a little looser (see plan_pitch).
REACH_TOL_SHALLOW_M = 0.025
SHALLOW_PITCH_DEG = 15.0


def default_grid(profile):
    """A sampling grid that covers a tabletop arm's useful space.

    Radii span from inside the base column out past the nominal reach, so the envelope
    is found rather than assumed; heights cover the table surface up to a transit
    height; pitches cover straight-down to horizontal.
    """
    r_max = max(0.60, float(profile.reach_max_m) + 0.10)
    return {
        "radii": tuple(np.round(np.arange(0.08, r_max + 1e-9, 0.02), 3)),
        "heights": (0.02, 0.06, 0.12),
        "pitches": tuple(float(p) for p in range(0, 91, 5)),
    }


@dataclass(frozen=True)
class ReachResult:
    """One probe: could the arm put its tip here at this tool angle?"""

    r_m: float
    z_m: float
    pitch_deg: float
    residual_m: float
    reached: bool
    seed_index: int = -1        # which seed rescued it; -1 = the default seed sufficed


@dataclass
class WorkspaceMap:
    """What an arm can actually reach, computed from its own kinematics."""

    profile_name: str
    results: list[ReachResult] = field(default_factory=list)
    seed_poses: list[list[float]] = field(default_factory=list)
    chosen_seeds: list[int] = field(default_factory=list)
    grid: dict = field(default_factory=dict)
    computed: str = ""

    # --- queries ------------------------------------------------------------
    def max_reach(self, pitch_deg: float, z_m: float | None = None) -> float:
        """Furthest radius reachable at this tool angle. 0.0 if none."""
        rs = [x.r_m for x in self.results
              if x.reached and abs(x.pitch_deg - pitch_deg) < 1e-6
              and (z_m is None or abs(x.z_m - z_m) < 1e-6)]
        return float(max(rs)) if rs else 0.0

    def reach_envelope(self, z_m: float | None = None) -> dict[float, float]:
        """pitch -> max radius. This is the table a human used to measure by hand."""
        return {p: self.max_reach(p, z_m) for p in sorted(self.pitches)}

    @property
    def pitches(self) -> set[float]:
        return {x.pitch_deg for x in self.results}

    def reachable(self, r_m: float, z_m: float, pitch_deg: float) -> bool:
        best = None
        for x in self.results:
            d = (abs(x.r_m - r_m), abs(x.z_m - z_m), abs(x.pitch_deg - pitch_deg))
            score = d[0] + d[1] + d[2] / 90.0
            if best is None or score < best[0]:
                best = (score, x)
        return bool(best and best[1].reached)

    def pitch_candidates(self, p_target, *, steepest_first: bool = True
                         ) -> tuple[float, ...]:
        """Tool angles that actually work at this target, best first.

        Replaces a hardcoded ordered list. Steep is preferred because a parallel jaw
        grips a table object best from above, and because on many arms a shallow angle
        at close radius demands the wrist sit inside the robot's own base column.
        """
        p = np.asarray(p_target, dtype=np.float64)
        r, z = float(np.hypot(p[0], p[1])), float(p[2])
        ok = []
        for pitch in sorted(self.pitches, reverse=steepest_first):
            if self.reachable(r, z, pitch):
                ok.append(pitch)
        return tuple(ok)

    def working_radii(self, z_m: float | None = None) -> tuple[float, float]:
        """(min, max) radius reachable at ANY tool angle — replaces hand-set limits."""
        rs = [x.r_m for x in self.results
              if x.reached and (z_m is None or abs(x.z_m - z_m) < 1e-6)]
        return (float(min(rs)), float(max(rs))) if rs else (0.0, 0.0)

    def seeds(self) -> tuple[tuple[float | None, ...], ...]:
        """The IK seeds this arm actually needs, in profile form.

        Only the joints the solver drives are pinned; the rest carry over from the
        caller, matching how ``ik_seeds`` is consumed.
        """
        return tuple(tuple(v) for v in (self.seed_poses[i] for i in self.chosen_seeds))

    def dead_bands(self, min_cells: int = 2) -> list[dict]:
        """Regions that ONLY a non-default seed could reach.

        These are the failures that get discovered by watching an arm mysteriously
        stall. Naming them automatically means a new arm does not have to earn that
        knowledge the same way.
        """
        rescued = [x for x in self.results if x.reached and x.seed_index >= 0]
        if len(rescued) < min_cells:
            return []
        bands = []
        for pitch in sorted({x.pitch_deg for x in rescued}):
            rs = sorted(x.r_m for x in rescued if x.pitch_deg == pitch)
            if len(rs) >= min_cells:
                bands.append({"pitch_deg": pitch, "r_min_m": rs[0], "r_max_m": rs[-1],
                              "cells": len(rs)})
        return bands

    def summary(self) -> str:
        lo, hi = self.working_radii()
        env = self.reach_envelope()
        line = "  ".join(f"{int(p)}deg->{r*100:.1f}cm"
                         for p, r in sorted(env.items(), reverse=True) if r > 0)
        n_ok = sum(1 for x in self.results if x.reached)
        rescued = sum(1 for x in self.results if x.reached and x.seed_index >= 0)
        radii = self.grid.get("radii") or [0, 0]
        step = (radii[1] - radii[0]) if len(radii) > 1 else 0.0
        out = [f"workspace of '{self.profile_name}' — {n_ok}/{len(self.results)} poses reachable",
               f"  radii    : {lo*100:.0f}..{hi*100:.0f} cm  (grid step {step*100:.0f}cm, so "
               f"each envelope figure is a lower bound to within one step)",
               f"  envelope : {line}",
               f"  seeds    : {len(self.chosen_seeds)} chosen from {len(self.seed_poses)} "
               f"sampled; they rescue {rescued} poses the bare solver could not reach"]
        for b in self.dead_bands():
            out.append(f"  DEAD BAND: pitch {b['pitch_deg']:.0f}deg needs a re-seed at "
                       f"r={b['r_min_m']*100:.0f}-{b['r_max_m']*100:.0f}cm ({b['cells']} cells)")
        return "\n".join(out)

    # --- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "profile": self.profile_name, "computed": self.computed, "grid": self.grid,
            "seed_poses": self.seed_poses, "chosen_seeds": self.chosen_seeds,
            "results": [[x.r_m, x.z_m, x.pitch_deg, round(x.residual_m, 6),
                         x.reached, x.seed_index] for x in self.results],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> WorkspaceMap:
        with open(path) as f:
            d = json.load(f)
        return cls(d["profile"],
                   [ReachResult(*r) for r in d["results"]],
                   d["seed_poses"], d["chosen_seeds"], d["grid"], d.get("computed", ""))


def _tol(pitch_deg: float) -> float:
    return REACH_TOL_SHALLOW_M if pitch_deg <= SHALLOW_PITCH_DEG else REACH_TOL_M


def _seed_pool(profile, n_per_joint: int = 3) -> list[list[float]]:
    """Candidate seeds spread across the joints the solver drives.

    Sampled from the arm's OWN limits rather than chosen by hand, so a different arm
    gets a pool that suits it. The joints the solver does not drive are left as None
    and carry over from the caller.
    """
    lo, hi = profile.limits()
    driven = [j for j in profile.positioning_joints if j != profile.pan_joint]
    if not driven:                       # pose-IK arms: one neutral seed is enough
        return [[None] * profile.n_joints]
    axes = []
    for j in driven:
        # inset from the hard stops: a seed sitting exactly on a limit cannot move
        span = hi[j] - lo[j]
        axes.append(np.linspace(lo[j] + 0.15 * span, hi[j] - 0.15 * span, n_per_joint))
    pool = []
    for combo in itertools.product(*axes):
        seed: list[float | None] = [None] * profile.n_joints
        for j, v in zip(driven, combo):
            seed[j] = float(round(v, 1))
        pool.append(seed)
    return pool


def analyze_workspace(ik, profile, *, grid=None, seed_pool=None, base_seed=None,
                      max_seeds: int = 6, progress=None) -> WorkspaceMap:
    """Probe the arm's reachable space and derive its pitch table, limits and seeds.

    Runs the arm's own IK over a (radius, height, pitch) grid. Each cell is first tried
    from the natural seed; only cells that fail are retried against the sampled pool,
    which is what makes the cost bearable and also identifies exactly which cells need
    rescuing — those are the dead bands.

    Seeds are then chosen by greedy set cover over the rescued cells, so the profile
    ends up with the fewest seeds that actually buy coverage rather than an accumulated
    pile of one-offs.
    """
    grid = dict(grid or default_grid(profile))
    pool = list(seed_pool if seed_pool is not None else _seed_pool(profile))
    base = (np.array(profile.home_deg, dtype=np.float64) if base_seed is None
            else np.asarray(base_seed, dtype=np.float64))

    def seeded(seed):
        q = base.copy()
        for i, v in enumerate(seed):
            if v is not None:
                q[i] = float(v)
        return q

    roll = float(base[profile.roll_joint]) if profile.roll_joint is not None else None

    def solve(q_seed, p, pitch):
        """One BARE solve — the solver's own re-seeding is disabled.

        This matters more than it looks. A strategy that already retries from a list of
        built-in seeds will rescue itself, so probing through it measures the arm *with*
        its seeds applied and concludes, wrongly, that no seeds are needed. Measuring
        the bare solver is the only way to see where its natural branch actually fails,
        which is exactly what a dead band is.
        """
        try:
            return ik.solve(q_seed, p, pitch_deg=pitch, roll_deg=roll, _retry=False)
        except TypeError:
            return ik.solve(q_seed, p, pitch_deg=pitch, roll_deg=roll)

    cells = list(itertools.product(grid["radii"], grid["heights"], grid["pitches"]))
    results: list[ReachResult] = []
    rescued_by: dict[int, set[int]] = {i: set() for i in range(len(pool))}

    for idx, (r, z, pitch) in enumerate(cells):
        if progress and idx % 200 == 0:
            progress(idx, len(cells))
        p = np.array([r, 0.0, z], dtype=np.float64)     # bearing is free: probe on +x
        tol = _tol(pitch)
        _q, e = solve(base, p, pitch)
        if e <= tol:
            results.append(ReachResult(r, z, pitch, float(e), True, -1))
            continue
        # the natural seed failed — does any sampled seed rescue it? Try them ALL, so
        # the set cover can choose between overlapping rescuers rather than being
        # handed whichever happened to be first in the pool.
        best_e, best_i = e, -1
        cell = len(results)
        for i, seed in enumerate(pool):
            _q2, e2 = solve(seeded(seed), p, pitch)
            if e2 < best_e:
                best_e, best_i = e2, i
            if e2 <= tol:
                rescued_by[i].add(cell)
        results.append(ReachResult(r, z, pitch, float(best_e), best_e <= tol, best_i))

    # --- greedy set cover: fewest seeds that rescue the most cells ---------
    chosen: list[int] = []
    remaining = {c for s in rescued_by.values() for c in s}
    while remaining and len(chosen) < max_seeds:
        i, gain = max(((i, len(s & remaining)) for i, s in rescued_by.items()),
                      key=lambda t: t[1])
        if gain == 0:
            break
        chosen.append(i)
        remaining -= rescued_by[i]

    return WorkspaceMap(profile.name, results, [list(s) for s in pool], chosen, grid,
                        time.strftime("%Y-%m-%d %H:%M:%S"))
