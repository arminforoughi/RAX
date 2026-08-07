# Adding an arm

The pick stack — detect, locate, map, approach, grasp — is arm-agnostic. Pointing it at
a new robot means writing one profile, not editing the algorithms.

Everything a profile has to answer falls into four questions:

1. **What is the robot?** URDF, end-effector frame, joint names, transport.
2. **Which joint does what?** Which is the base yaw, which form a parallel pitch chain,
   which is the wrist roll. This is what replaces `q[0]`, `q[1]+q[2]+q[3]`, `q[4]`
   appearing as literals inside the solver.
3. **Where is the camera, and what does it see with?** Wrist-mounted or world-fixed,
   its extrinsics, its resolution.
4. **How does the gripper behave?** Open/closed percentages, and what current rise
   counts as touching something.

## Write the profile

Copy `robots/profiles/so101.py` and change what differs. The two worked examples sit at
opposite ends of every choice: `so101.py` is a parallel-pitch arm with a wrist camera,
`mock.py` is a generic 6-DOF arm with a fixed overhead camera and no URDF at all.

```python
from robots.profiles import ArmProfile, CameraProfile, GripperProfile

PROFILE = ArmProfile(
    name="my_arm",
    urdf="path/to/my_arm.urdf",     # repo-relative or absolute
    ee_frame="tool0",
    joint_names=("j1", "j2", "j3", "j4", "j5", "j6"),
    port="/dev/ttyUSB0",
    ik="pose",                      # or "pitch_hold" — see below
    home_deg=(...), view_deg=(...),
    gripper=GripperProfile(...),
    camera=CameraProfile(mount="eye_in_hand", extrinsics="x,y,z,rx,ry,rz"),
)
```

Register it in `robots/profiles/__init__.py`'s `_PROFILES`, then select it at runtime:

```
RAX_ARM=my_arm python stack_mission2.py
```

Joint limits are read from the URDF automatically. Declare `limits_deg` explicitly only
if the URDF lacks them or lies — an arm whose limits are wrong will park against a stop
while the solver reports success.

## Choosing the IK strategy

**`ik="pitch_hold"`** — for arms whose pitch joints share a parallel axis, so the tool's
world pitch is exactly their sum. The last joint of the chain is then slaved
algebraically rather than solved, which is why the held pitch never drifts. Declare
`pan_joint`, a `pitch_chain` of two or more joints, and `roll_joint`.

```python
ik="pitch_hold", pan_joint=0, pitch_chain=(1, 2, 3), roll_joint=4,
ik_seeds=((None, -95.0, 90.0, 30.0, None), ...)   # None = keep the caller's value
```

`ik_seeds` matters if the arm has dead bands — configurations where the solver's natural
branch cannot reach a pose it should. On the SO-101 there is a genuine elbow-flip band at
r=10–15 cm with a shallow pitch where no single seed converges. If reaches fail
intermittently at particular radii, add seeds that approach from a different branch.

**`ik="pose"`** — for arms with a real 6-DOF wrist. The requested pitch and roll become a
target orientation handed to the `Kinematics` backend. If your tool frame does not point
along +Z, subclass `PoseIK` and override `target_rotation`.

## Choosing the camera mount

**`mount="eye_in_hand"`** — the camera rides the gripper, `T_base_cam = FK(q) @ T_ee_cam`.
The `extrinsics` string is the hand-eye transform. Set `gripper.hand_uv` to the
fingertip's measured pixel: the hand-eye independently *predicts* that same pixel, so the
gap between the two is a live read-out of calibration error, drawn on the video overlay.

**`mount="fixed"`** — the camera is bolted to the world and `extrinsics` is `T_base_cam`
directly. Localization and the map work unchanged. `tip_pixel` returns `None`, because
"where the fingertip appears" is only a constant pixel when the camera moves with the hand.

## Verify before touching hardware

```
python tests/test_arm_profile.py       # profile loads, limits match the URDF, topology is coherent
python tests/test_second_arm.py        # detect -> locate -> map -> approach, headless
python tests/test_extraction_parity.py # the shared maths is unchanged
```

`validate()` runs at load and rejects an incoherent profile — a pan joint that is also in
the pitch chain, a pose with the wrong number of joints, a `pitch_hold` arm with a
one-joint chain. A profile that misdescribes the arm is worse than none, because the IK
then drives the wrong joint confidently.

Then, on the robot, in this order — each step is cheap and rules out a whole class of
problem before the next:

1. **Jog every axis.** Confirms joint order, direction, and limits.
2. **`/caltip`**, if eye-in-hand. Measures the fingertip pixel; a large gap from the
   predicted one means the hand-eye is wrong and everything downstream inherits it.
3. **`/calib`** to fit the hand-eye, then re-check the overlay gap.
4. **`/floorcal`** to measure the table plane. Absorbs both table tilt and arm sag, which
   is why a grasp can clear near the base and scrape at full reach.
5. **Scan → 2D map.** Confirms detection and localization before anything moves at speed.
6. **One pick**, with a hand near the stop button.

## Calibrating the hand-eye

Two fitters live in `perception/handeye.py`, with different objectives:

* **`fit_reprojection`** (`/calib`) — every sightline to one static target must meet at
  a point, and the fingertip must land on its measured pixel. Solves the transform and
  the target position together.
* **`fit_consistency`** (`/calibmount`) — a stationary object must map to the same table
  position from every viewpoint.

Judge a reprojection fit by the **fingertip gap**, not the reprojection RMS. The gap is
a hard geometric constraint — the camera is rigid to the end effector, so FK's fingertip
*must* land on the measured pixel — while a colour-blob centroid has a 50–80 px noise
floor that more poses do not lower. Gating tightly on RMS rejects good fits and keeps
broken transforms.

**`fit_reprojection` has a flat direction, and it does not announce itself.** With one
point target the camera can slide along its viewing direction and the target follow it,
reproducing every observed pixel; only parallax between poses pins it down. Measured on
noise-free data seeded with the exact truth, it settles ~14 mm and ~3° away while
reporting 0.5 px RMS and 0.01 px fingertip error — the numbers look excellent because
what they measure is excellent.

So: re-running `/calib` on an already-good robot can move the transform. Use
`/calibmount` to *check* a calibration, and widen the pose spread if you need the
reprojection fit to be better determined.

## Solving the localization knobs instead of dialling them

`push_out_cm`, `range_scale` and `bearing_deg` are not preferences. Each is a
compensation for a systematic error, historically found by turning a dial until grasps
stopped missing — which is why they needed retuning whenever anything moved.

They can be solved, because **the arm can generate its own ground truth**. While the
gripper holds an object, forward kinematics says exactly where that object is. Show it to
the camera at a spread of radii and bearings, localize it normally, and the difference is
the error — measured, everywhere you care about, with no ruler.

**The procedure** (`perception/selfcal.py`):

1. Grasp any object whose label the detector reports reliably.
2. For each of ~6 radii (15–40 cm) × ~5 bearings (−40°…+40°), move the held object there
   and record `LocalizationSample(xy_true=FK_tip_xy, xy_observed=localizer_result,
   off_axis_deg=...)`.
3. `fit = fit_localization(samples)` → `print(fit.summary())`.
4. `apply_to_config(fit, CFG)` — which **refuses** unless the fit is trustworthy.

Radial and angular spread both matter. All samples at one radius makes scale and offset
inseparable (any scale trades against any offset); a narrow bearing span leaves the
rotation poorly observed. The fit says so rather than returning confident numbers.

**It vetoes itself when a known bug explains the error.** Measured on a synthetic
overhead rig, the axial-depth projection error produces 40 mm RMS, and the three knobs
absorb 80% of it — a 5× improvement any operator would accept, with a fitted
`range_scale` of 1.32 that is not a range scale at all but a lie about the object's size.
Install it and the knobs now depend on camera height, object, and table position, so they
need retuning forever. That is the trap. So the fit reports `DO NOT APPLY` and names the
bug. With the projection corrected, the same rig measures **0.0 mm before any
correction** — the knobs have nothing left to do.

On real hardware they will not go to exactly zero: hand-eye residual, encoder error and
detector noise remain. But they should end up small and *stable*, and if a knob is still
carrying centimetres, that is a bug worth finding rather than a number worth tuning.

## Known issue to check on a new rig

`ApparentSizeLocalizer` places objects using the size-derived distance as a *range along
the sightline*, but the pinhole relation actually yields *axial depth*. The two agree only
on the optical axis; off-axis, objects land too close by `cos(off-axis angle)` — always
inward. Measured with a camera 60 cm up: 10 mm at r=20 cm, 48 mm at 35 cm, 90 mm at 45 cm.

This is inherited behaviour, and the trims on the SO-101 were tuned with it present.
Setting `ApparentSizeLocalizer.axial_depth = True` corrects it, but that changes where the
arm drives, so re-check the approach trims afterwards. `tests/test_second_arm.py` pins
both behaviours.

## Where things live

| Concern | Module |
|---|---|
| Arm description, URDF limits | `robots/profiles/` |
| FK/IK backends | `manipulation/arms/kinematics.py` |
| Pitch-hold and pose IK | `manipulation/arms/ik_strategy.py` |
| Trajectory profiles | `manipulation/arms/motion.py` |
| Mission phase / log / stop flag | `common/mission_state.py` |
| Pixels ↔ base frame | `perception/camera_geometry.py` |
| Detection → position | `perception/locate.py` |
| Size, height and yaw from one frame | `perception/measure.py` |
| Hand-eye calibration | `perception/handeye.py` |
| Table plane | `perception/table_plane.py` |
| Class size priors | `perception/object_priors.py` |
| Following between detections | `models/detection/tracking.py` |
| Object map | `mobility/slam/object_map.py` |
| Approach tunables, staging, centring | `manipulation/approach/` |
| HTTP server, UI, guest sessions | `stack_mission2.py` |

Each package's `__init__.py` lists its public surface, so `from perception import
CameraGeometry, ObjectMeasurer` and similar work directly.

### What deliberately stays in the server

`run_mission` and `place_at` are not extracted, and that is a decision rather than an
omission. Measured, `run_mission` touches 23 distinct module-level names, and roughly a
third of its statements are `say`/`set_phase` calls. What remains in it after the
extractions is narration, phase transitions, retry policy, and abort handling wrapped
around pieces that already live in packages. Hoisting that out would need an interface
of 23 members — a facade over the whole robot, not an abstraction — and would make the
sequence harder to read, not easier.

The reusable parts of the approach *are* extracted: the staging geometry, the tunables,
and the centring servo. A new arm reuses those and writes its own mission sequence, which
is the part that legitimately differs between robots.
