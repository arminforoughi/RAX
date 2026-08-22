# How `mission_server.py` got modularized

> **This is a historical record, not the current architecture.** It describes the
> session that first broke the monolith apart, and its paths predate the move to
> `src/rax/`. For how the seams work today, read
> [architecture.md](architecture.md); for how to run the stack on your own robot,
> read [porting.md](porting.md).

A record of one working session that turned `stack_mission2.py` — a 5,885-line,
single-file pick/place server welded to one SO-101 arm — into `mission_server.py`, the
same server running on ~5,200 lines of reusable packages. The original file was never
edited; it still runs unchanged.

## Why

The ask was narrow at first: *detect, locate, and map objects; approach and grasp them
— using other robot factors* (a different arm, a different camera). Everything in
`stack_mission2.py` that answered that question was written as literal SO-101 facts —
joint index `0` is the pan, `q[1]+q[2]+q[3]` is the gripper pitch, the camera transform
is a module-level constant, the serial port is `"COM4"` in three places. None of that
transfers to a second robot.

The scope grew once actually reading the constants: numbers like `PICK_GRASP_Z = 1.5cm`
or `TARGET_RIGHT_TRIM_M = 5cm` were not arbitrary — they were a human's compensation for
an *unmeasured* error, discovered by trial and dialled until grasps stopped missing.
"Generic" turned out to mean two different things: an algorithm that runs on a different
skeleton, and an algorithm that measures its own errors instead of having them dialled
in by hand.

## What moved, in the order it happened

| Phase | Package | What it owns |
|---|---|---|
| 0 | `tests/test_extraction_parity.py` | golden values for every pure function, pinned **before** anything moved |
| 1 | `robots/profiles/` | the arm as data — URDF, joint limits (read from the URDF, not transcribed), which index is pan/pitch-chain/roll, camera mount, gripper thresholds |
| 2 | `perception/camera_geometry.py`, `table_plane.py`, `object_priors.py` | pixels ↔ base frame for a wrist **or** fixed camera; the measured table; the 97-entry class-size table |
| 3 | `manipulation/arms/ik_strategy.py`, `motion.py` | `PitchHoldIK` (parallel-pitch arms) and `PoseIK` (generic 6-DOF) behind one protocol; quintic trajectories |
| 4 | `perception/locate.py`, `measure.py`; `models/detection/tracking.py`; `mobility/slam/object_map.py` | three range strategies; monocular footprint/height/yaw; the between-detection trackers; the bird's-eye map and its merge rules |
| 5 | `manipulation/approach/` (`config.py`, `geometry.py`, `visual_center.py`) | every tunable behind a knob-name API; staging geometry; the centring servo (gains measured by probing, not modelled) |
| 6 | `robots/profiles/mock.py`, `tests/test_second_arm.py` | a second, opposite-choice profile (pose IK, fixed camera, no URDF) proving the seams actually work, headless |
| — | `perception/handeye.py` | the two calibration fitters (a Phase-2 item originally missed, added later) |
| — | `models/detection/detector_service.py` | the throttled detection loop, query handling, NMS, per-label trackers |
| — | `common/guest_sessions.py` | the public-access queue/turn/cooldown policy (unrelated to the robot algorithm; extracted separately because it had never had tests) |

Deliberately **not** extracted: `run_mission` / `place_at` (~440 lines). Measured
first — 23 distinct module-level dependencies, roughly a third of the statements are
`say`/`set_phase` narration. What's left after everything else moved out is orchestration
over already-packaged pieces, not an algorithm; a 23-member interface to pull it behind
would be a facade, not an abstraction.

## How each move was checked

- **Golden-value parity.** `tests/test_extraction_parity.py` freezes ~450 cases before
  every extraction and re-checks after. A negative-control run (perturbing one constant
  by 0.1px) confirmed the harness actually fails when something is wrong, not just when
  it's asked to.
- **Bit-identical, not just "looks right."** The generalized IK matched the original
  over 120 cases; the derived seed set matched hand-written seeds over 700 poses with
  zero regressions (36/60 solvable either way, nothing reachable by one set and not the
  other); the extracted trajectory generator matched to within one ULP.
- **Second-arm proof.** `test_second_arm.py` runs detect → locate → map → approach on
  `robots/profiles/mock.py` — pose IK, fixed camera, no hardware — closing the loop
  through real geometry rather than mocked numbers.
- **On real hardware.** Running the server surfaced three regressions golden values
  couldn't: a missing `now = time.time()` after refactoring `world2d_snapshot`, a
  `None`-vs-NaN crash in the same function, and a viewer mesh path broken by moving the
  URDF source. All three were only caught by hitting the actual endpoints.

## Turning dialled constants into derived ones

Five algorithms replace hand-tuned numbers with values computed from the robot's own
kinematics or its own measurements:

- **`manipulation/arms/workspace.py`** — probes `(radius, height, pitch)` with the arm's
  *bare* solver (no built-in retry, or the strategy rescues itself and the probe
  concludes nothing is needed) and derives the reach envelope and IK seeds. On the
  SO-101 this reproduced the hand-measured reach table to within one grid step, and
  replaced 5 hand-accumulated seeds with 1 derived seed at identical coverage.
- **`perception/selfcal.py`** — the arm holds an object, FK gives ground truth,
  observed-vs-true position across a pose sweep fits `range_scale` / `push_out_m` /
  `bearing_offset_deg`. It **refuses to apply** a fit when a known error signature (like
  the axial-depth bug below) would explain the residual better than a linear
  correction — fitting anyway would hide the real bug under a well-tuned patch.
- **`manipulation/approach/derive.py`** — grasp height, centring pixel-tolerance, and
  approach trim, each computed from the object's measured size/height and the camera's
  field of view instead of one constant tuned on a 5cm cube. The grasp-height fraction
  is *anchored* to the original tuned pair (1.5cm / 5.08cm cube) so the cube case is
  reproduced exactly and every other object inherits the same rule rather than the same
  number.
- **`ObjectMap.suggested_merge_radius`** — the map's own position residuals over time
  give a live estimate of localization noise; the merge radius can track it (3σ) instead
  of sitting at a fixed `0.14`.

None of these were switched on by guessing they'd help — each was checked against the
constant it replaces first, on the case that constant was tuned for.

## Defects the process found, not just moved

- **Axial-depth localization bias.** Apparent-size ranging computes an axial depth but
  the code placed it along the sightline; off-axis objects land too close by
  `cos(angle)` (~90mm at 45cm on a synthetic rig). Diagnosed, pinned by
  `ApparentSizeLocalizer.axial_depth`, left off pending a hardware re-check since it
  changes where the arm drives.
- **The hand-eye reprojection fit has a flat direction.** `/calib`, seeded with the
  *exact* true transform on noise-free synthetic data, still drifts ~14mm / ~3° while
  reporting a perfect 0.5px residual — the objective can trade translation against
  rotation and not know it moved. `/calibmount`'s consistency objective doesn't have
  this failure mode.
- **The footprint measurement was rejecting effectively every attempt.** Two rounds of
  wrong guesses (an assumed pose, then an assumed row-in-frame) before the fix was to
  make the code report *which* geometric gate rejected and the actual incidence number,
  rather than diagnosing blind. Confirmed on the live server: the object's contact
  pixels sit at frame rows ~150-175, where the sightline barely grazes the table
  (incidence –0.3 to +0.17 against a 0.22 gate) — a real geometric limit of the current
  camera angle, not a solver bug. Unresolved; requires either measuring from a
  steeper-angle pose or accepting priors at the current one.
- **The camera wedges if the process dies uncleanly.** An OAK-D whose owner exits without
  disconnecting stays booted with no owner (`X_LINK_BOOTED` / `X_LINK_ERROR`) and refuses
  to re-enumerate without a physical unplug — this caused a 41-hour outage in production
  before this session started. `POST /shutdown` now quiesces the frame-reading threads,
  stops the detector, and disconnects the robot before the camera, in that order — the
  ordering mattered: releasing the camera first left the robot reporting "not connected"
  and skipping its own bus teardown. On Windows, `Stop-Process` (a hard terminate) still
  cannot be intercepted by anything — `/shutdown` is the only stop path proven not to
  wedge the device.

## Net effect

```
stack_mission2.py    5,885 lines   — original, byte-for-byte untouched, still runs
mission_server.py    4,864 lines   — same server, same routes, same /status shape
extracted packages   5,178 lines   — 16 files, usable by any robot that fits the seams
```

`mission_server.py` imports 15 package modules `stack_mission2.py` never does. The
routes, the knob names, and the `/status` JSON shape are unchanged — verified against
golden values, not just "should be the same."

## What's still open

- **No pick has been run on hardware.** Detection, mapping, calibration loading, and the
  viewer are all confirmed live; the grasp path (IK → staged approach → visual centring
  → descent → close) is confirmed by tests only.
- The two diagnosed geometry defects (axial-depth bias, hand-eye flat direction) are
  live but their fixes are switched off pending a hardware check, since both change
  where the arm physically drives.
- The footprint measurement's rejection is diagnosed with real numbers but not yet
  resolved — no policy decision has been made about measuring from a different pose.
- `guest sessions`, the detector service, and a few smaller pieces were extracted for
  testability rather than because the stated arm-generality goal required them.
