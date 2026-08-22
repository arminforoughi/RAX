# Porting RAX to your robot

The whole contract is three things:

1. **A profile** — your arm described as data (one Python file, no code).
2. **A camera** — usually none of your work; pick a `kind` and go.
3. **A driver** — four methods, and only if your arm isn't already supported.

Nothing in `rax.perception` or `rax.manipulation` knows what robot it is running on.
If you find yourself editing either to support your hardware, that is a bug in the
seams — please open an issue. [architecture.md](architecture.md) explains why the seams
are where they are, and where they are still imperfect.

---

## 0. Check whether you need to do anything

```bash
python -m rax.grasp --list
```

```
  head_mono    6 joints, ik=pose, camera=mono/fixed
  mock         6 joints, ik=pose, camera=stereo/fixed
  so101        5 joints, ik=pitch_hold, camera=stereo/eye_in_hand
  wrist_mono   6 joints, ik=pose, camera=mono/eye_in_hand
```

Run the whole pipeline with no hardware first, so you know the software works before
you start debugging your robot:

```bash
python -m rax.grasp --arm head_mono --query "red object"
```

---

## 1. The camera

Pick the row that matches what you own. This is usually the entire camera step.

| You have | `kind` | Depth comes from | Extra install |
|---|---|---|---|
| Any webcam, phone cam, RTSP stream | `mono` | the table plane + class sizes | none |
| RealSense, Femto, Kinect, Astra | `rgbd` | the sensor | `pip install rax[realsense]` |
| OAK-D, ZED, two synced cameras | `stereo` | a matcher over the pair | `pip install rax[oakd]` |

All three work on either mount. A wrist-mounted mono camera ranges by apparent size
(`z = fx * W / w_px`); a fixed mono camera intersects the sightline with the table.
Both are real measurements, and `tests/test_wrist_mono.py` and
`tests/test_head_camera.py` hold them to it.

**A mono camera is a supported rig, not a degraded one.** Two of the three range
strategies never needed depth: `PlaneRayLocalizer` intersects the sightline with the
table and solves range and size together, and `ApparentSizeLocalizer` divides a known
class size by the apparent one. What depth actually buys you is objects that are *not*
on a known surface. For a robot picking things off a desk, a $20 webcam is enough.

### Mount

```
mount="eye_in_hand"   camera on the gripper.  T_base_cam = FK(q) @ T_ee_cam
mount="fixed"         camera anywhere else.   T_base_cam is constant
```

A head mast, a tripod, a torso, a ceiling — all `fixed`. The mount decides which
pipeline runs (`rax/grasp.py` picks; you don't pass a flag), because a wrist camera can
visually servo and a fixed one cannot.

### Intrinsics

Best to worst:

1. Calibrate with `cv2.calibrateCamera` on a chessboard and put `(fx, fy, cx, cy)` in
   the profile.
2. Give a horizontal FOV; `MonoCamera` computes `fx = (w/2) / tan(hfov/2)`.
3. Accept the 60-degree default.

Option 3 is genuinely allowed. A wrong focal length scales every distance by a
constant, and `rax.perception.selfcal` measures and removes exactly that from the
robot's own motion — so an uncalibrated camera converges to a calibrated one instead of
being a dead end.

### Extrinsics

For `fixed`, `extrinsics` is `T_base_cam` as `"x,y,z,rx,ry,rz"` (metres, rotation
vector in radians, OpenCV optical convention: +Z along the view, +Y image-down). Measure
it with a tape if you like — then run the hand-eye fitter in `rax.perception.handeye`,
which refits it from the arm's own motion and writes the correction to a JSON file the
profile points at.

---

## 2. The profile

Copy `src/rax/robots/profiles/head_mono.py` — it is the shortest complete one — and
change what differs. Then register it in `_PROFILES` at the bottom of
`src/rax/robots/profiles/__init__.py`.

The fields that actually matter:

```python
PROFILE = ArmProfile(
    name="my_arm",
    urdf="urdf/my_arm.urdf",     # "" if you use CartesianKinematics
    ee_frame="tool0",
    joint_names=("j1", "j2", "j3", "j4", "j5", "j6"),

    ik="pose",                   # or "pitch_hold" — see below
    home_deg=(...), view_deg=(...),
    table_z_m=0.0,               # the surface objects rest on, in base frame
    reach_min_m=0.05, reach_max_m=0.60,

    gripper=GripperProfile(open_pct=95.0, closed_pct=2.0,
                           contact_current_delta=8.0),
    camera=CameraProfile(kind="mono", mount="fixed", extrinsics="...")
)
```

Joint limits are **read from your URDF** — do not transcribe them. Declare
`limits_deg` explicitly only if you have no URDF, or if the URDF lies.

### Which `ik`?

**`ik="pitch_hold"`** if your arm's pitch joints share a parallel axis, so the
gripper's world pitch is exactly their sum (SO-101, most hobby 5-DOF arms). Declare
`pan_joint` and a `pitch_chain` of 2+ joints; the last one is slaved algebraically,
which is what makes the pitch hold *exactly* rather than drift.

**`ik="pose"`** for a real 6-DOF wrist, solved as a full pose.

### What you do *not* have to tune

This is the part that saves the most time, and the reason this repo exists rather than
being a gist. These are computed, not dialled:

| Instead of guessing | Run | It derives |
|---|---|---|
| a kinematics library | nothing — `make_kinematics` reads your URDF in numpy | FK by chain composition, IK by damped least squares |
| reach envelope, IK seeds | `rax.manipulation.arms.workspace.analyze_workspace` | probes your solver, writes `<arm>_workspace.json` |
| grasp height, centring tolerance, approach trim | `rax.manipulation.approach.derive` | from the object's measured size and your camera's FOV |
| range scale, push-out, bearing offset | `rax.perception.selfcal.fit_localization` | arm holds an object, FK is ground truth, fits the error |
| hand-eye transform | `rax.perception.handeye.fit_reprojection` | from arm motion |

`selfcal` will **refuse** to apply a fit when a known error signature explains the
residual better than a linear correction — fitting anyway would hide a real bug under a
well-tuned patch.

---

## 3. The driver (only if your arm is new)

Four methods. Full contract in `src/rax/manipulation/arms/arm_interface.py`.

```python
class MyArm:
    joint_names = ["j1", "j2", "j3", "j4", "j5", "j6"]

    def get_state(self) -> ArmState:
        return ArmState(joints_deg=np.array([...]), gripper_pct=42.0)

    def send_joint_targets(self, q_deg: np.ndarray) -> None: ...
    def set_gripper(self, pct: float) -> None: ...          # 0 closed, 100 open
    def read_gripper_current(self) -> float | None: ...     # None if unsupported
```

`read_gripper_current` returning `None` is fine — you lose contact sensing, and the
grasp closes open-loop.

Then join it to a camera:

```python
from rax.manipulation.arms.rig import Rig, geometry_for
from rax.perception.cameras import make_camera
from rax.robots.profiles import load_profile

profile = load_profile("my_arm")
rig = Rig(MyArm(), make_camera(profile), geometry_for(profile, kin))
rig.get_observation()   # frame + joints + T_base_cam, ready for the rest of the stack
```

If your arm *does* own its camera (a lerobot follower with an integrated OAK-D, say),
implement `get_observation()` returning an `Observation` directly and skip the `Rig` —
`So101Arm` is the worked example.

---

## 4. Check your port before trusting it

```bash
python -m rax.grasp --arm my_arm --query "cup" --no-move
```

`--no-move` runs detection and localization and reports where it thinks the object is,
without commanding a single joint. Put a tape measure on the table. If the reported
position is off by a constant scale, run `selfcal`; if it is off by a rotation, your
extrinsics are wrong and `handeye` will fix them; if it is off by a sign, check that
your base frame is Z-up and your camera follows the OpenCV optical convention.

Then add a test. `tests/test_head_camera.py` is the template: it builds a rig from a
profile, renders a synthetic scene through it, and asserts the arm converges on an
object it was never told the position of — no hardware. A port with a test like that
keeps working; a port without one breaks the next time somebody refactors a seam.
