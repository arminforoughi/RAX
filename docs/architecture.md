# How the modularization works

The stack is **four seams and the algorithms between them**. This document is the
current state — what the seams are, why each one is where it is, and what is still
wrong. For the step-by-step "how do I run this on my robot", see
[porting.md](porting.md). For the history of how the monolith was broken up, see
[mission_server_modularization.md](mission_server_modularization.md).

---

## The one rule

**The dependency arrow points one way.** `robots/` knows about `manipulation/` and
`perception/`. They never know about it.

That is what "modular" means here concretely: you can read, test and change every
algorithm in the stack without knowing which robot it runs on. Verified mechanically —
`grep -rn "from rax.robots" src/rax/{perception,manipulation,models,mobility}` is the
check, and it should return nothing.

---

## The four seams

```
                       rax.grasp                     the one entry point
                           │
        reads profile.camera.mount, picks the pipeline
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
   eye_in_hand                             fixed
   visual servo                    locate → fuse → stage
   (GazeEngine)                (localize + ObjectMap + approach)
        │                                     │
        └──────────────────┬──────────────────┘
                           │
                         Rig                        joins arm to sensor
                       ╱      ╲
            ArmInterface      CameraInterface
            get_state()       frame() ─→ Frame
            send_joint_targets()        ├ rgb      (always)
            set_gripper()               ├ right    (stereo only)
            read_gripper_current()      └ depth_m  (rgbd only)
                 │                            │
            ArmProfile                  DepthSource picks itself
        (URDF, joint topology,     stereo → matcher
         limits, gripper,          rgbd   → read off the sensor
         camera mount)             mono   → None, and that is fine
```

### 1. `ArmProfile` — the arm as data

`src/rax/robots/profiles/`. A robot described declaratively: its URDF, which joint index
is the pan, which joints form the pitch chain, its gripper thresholds, where its camera
is mounted.

The algorithms then read *which joint does what* instead of assuming. `ik_strategy.py`
asks the profile for `pan_joint` and `pitch_chain`; it does not hardcode that `q[0]` is
the base. Joint limits are read from the URDF rather than transcribed, because a
transcribed table is correct for exactly one robot.

Adding an arm is writing one of these. It is data, not code.

### 2. `ArmInterface` — four methods

`src/rax/manipulation/arms/arm_interface.py`. Report joints and gripper, accept joint
and gripper commands, optionally report gripper current for contact sensing. That is the
entire contract a new arm has to satisfy.

An arm that *owns* its camera (a lerobot follower with an integrated OAK-D) can instead
implement `get_observation()` and hand over everything at once. `So101Arm` is that case.

### 3. `CameraInterface` — one frame, whatever the sensor

`src/rax/perception/camera_interface.py`. A camera hands over a `Frame`: always an RGB
image and its intrinsics, plus whatever extras that rig happens to have.

This is the seam that makes the "your camera probably works" claim true, and the
mechanism matters: **nothing branches on sensor type.** A mono camera does not take a
special code path. It produces a `Frame` with no `right` and no `depth_m`,
`make_depth_source` hands back a `NoDepthSource`, and every caller that already coped
with a failed stereo match copes identically.

| Rig | `Frame` carries | Depth from |
|---|---|---|
| stereo (OAK-D, ZED) | `rgb` + `right` | a matcher runs over the pair |
| RGB-D (RealSense, Femto) | `rgb` + `depth_m` | read off the device |
| mono (any webcam) | `rgb` | nothing — and two range strategies never needed it |

Mono is a supported configuration, not a degraded one. `PlaneRayLocalizer` intersects
the sightline with the measured table and solves range and size together;
`ApparentSizeLocalizer` divides a known class size by the apparent one. What depth
actually buys you is objects that are *not* on a known surface.

### 4. `Rig` — the join

`src/rax/manipulation/arms/rig.py`. One arm plus one camera equals a working pick rig.
The arm does not import the camera and the camera does not import the arm; `Rig` owns
the single fact relating them, which is the **mount**:

- `eye_in_hand` — camera on the wrist, so `T_base_cam = FK(q) @ T_ee_cam`, recomputed
  from the joints every observation.
- `fixed` — camera on a mast, tripod, torso or ceiling. `T_base_cam` is constant and the
  joints do not enter into it.

Either way the rest of the stack sees the same `Observation`, which is why the gaze
engine, the cloud tracker and the object map needed no changes to support a head camera.

---

## Mount × sensor

The two are independent choices, so there are six combinations. All six work, and all
six are covered by tests that run with no hardware:

|                 | stereo | RGB-D | mono |
|---|---|---|---|
| **eye_in_hand** | cloud depth | sensor depth | apparent-size range |
| **fixed**       | cloud depth | sensor depth | table-plane range |

The mount also decides which *pipeline* runs, and `rax.grasp` picks it from the profile
rather than from a flag, because the two are genuinely different problems:

**Eye-in-hand.** The object's bearing is the thing being controlled — every joint move
changes the view — so the arm visually servos: keep the object on the aim pixel, drive
the range down, close.

**Fixed.** The view never changes, so there is nothing to servo. The object is localized
into base frame, fused over several frames, and the arm staged toward the result
open-loop. Deliberately open-loop: a head camera watching an arm reach sees the arm
occlude the object at exactly the moment a correction would matter, so a servo loop
there chases its own gripper.

---

## What is derived rather than dialled

A seam lets a different robot *run* the code. It does not make the code *work* on that
robot — that is what these do, and it is the half most ports actually get stuck on.

| Instead of guessing | Run | It derives |
|---|---|---|
| a kinematics library | nothing — `make_kinematics` reads your URDF in numpy | FK by chain composition, IK by damped least squares |
| reach envelope, IK seeds | `manipulation/arms/workspace.py` | probes your own solver |
| grasp height, centring tolerance, approach trim | `manipulation/approach/derive.py` | from measured object size and camera FOV |
| range scale, push-out, bearing offset | `perception/selfcal.py` | arm holds an object, FK is ground truth |
| hand-eye transform | `perception/handeye.py` | from arm motion, by reprojection and consistency |

`selfcal` **refuses to apply** a fit when a known error signature explains the residual
better than a linear correction — fitting anyway would hide a real bug under a
well-tuned patch.

---

## Package map

| Package | Owns | Imports |
|---|---|---|
| `rax.robots` | profiles, drivers, vendor CLIs | manipulation, perception |
| `rax.manipulation` | arm seam, IK, trajectories, gaze engine, `Rig`, approach | perception, models |
| `rax.perception` | camera geometry, camera seam + adapters, three range strategies, hand-eye, self-cal | manipulation, models |
| `rax.models` | detectors (YOLO-World, blob) and stereo backends (RAFT, FoundationStereo, SGBM) | perception |
| `rax.mobility` | the bird's-eye object map and its merge rules | perception |

---

## Known gaps

Stated here rather than discovered later.

**1. `manipulation/arms/run_gaze.py:42` imports `So101Arm`.** A generic package naming a
specific robot — the same violation that was fixed by moving the SO-101 CLI to
`robots/arms/lerobot_so101/`, missed in this one file. `run_gaze.py` is the CLI that
`rax.grasp` superseded; nothing imports it. It should move to `examples/` or lose its
`--backend so101` branch.

**2. `perception` and `manipulation` import each other.** Look at the table above: the
arrow runs both ways.

- `manipulation/arms/arm_interface.py` → `perception.camera_interface.Frame`
- `perception/depth_cloud/cloud_tracker.py` → `manipulation.arms.arm_interface.Observation`

The root cause is that `Observation` is *the join of both seams* — arm state, camera
frame, and the pose relating them — so it belongs to neither package. Right now the two
cannot be separated or reasoned about independently, which is the property the seams
were supposed to buy. The fix is small: make `CloudTracker` take `(frame, T_base_cam)`
instead of an `Observation` — those are the only two things it uses — which removes the
back-edge entirely.

**3. Four lazy `lerobot` imports remain**, all on optional paths: the OAK-D opener
(`perception/cameras/stereo.py`), the Rerun mesh viewer (inside a `try`), and the
FoundationStereo backend (SGBM is the always-available fallback). None block a plain
arm + URDF + webcam. The one that *did* block it — kinematics — is gone.

**4. `examples/` is not held to any of this.** Several demos still point at a local
`lerobot` checkout by absolute path. They are demos; the library above them is not.

---

## How the claims are checked

```bash
pytest -q     # 180 tests, no robot, no camera, no model weights
```

- `test_head_camera.py` — one scene and one arm built as three rigs (mono, RGB-D,
  stereo); all three must locate the object within 2 cm of each other and of the truth.
  Welding the stack back to stereo fails the mono cases.
- `test_wrist_mono.py` — the sixth combination, and a regression pin for the silent
  failure it used to have: the servo ran on a hardcoded 0.25 m placeholder while
  reporting healthy progress.
- `test_second_arm.py` — detect → locate → map → approach on a profile taking the
  *opposite* branch of every seam: pose IK, fixed camera, no URDF.
- `test_urdf_kinematics.py` — the FK/IK backend that replaced the lerobot import,
  including that a prismatic joint is metres and a revolute one is degrees.
- `test_grasp_cli.py` — the published entry point end to end on every advertised rig,
  checking the reported position against ground truth, because a pipeline can "succeed"
  while reaching to the wrong place.
- `test_extraction_parity.py` — ~450 golden values against the original monolith, so the
  extraction is provably faithful.
