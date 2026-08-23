# RAX — approach and grasp, with an arm and a camera you already own

Point a camera at a table, say what you want, and a robot arm finds it and picks it up.

```bash
pip install -e .
python -m rax.grasp --arm head_mono --query "red cube"
```

Most pick-and-place code is welded to the robot it was written on: joint indices are
literals, the camera transform is a module global, and the constants that make grasps
land were dialled in by hand against one object on one table. Porting it means rewriting
it.

RAX is the same stack with the robot pulled out into data and the sensor pulled out into
an interface — and with the hand-tuned constants replaced by things the robot measures
about itself. Two of those are the pitch:

**Your camera probably works.** Stereo, RGB-D, or a plain webcam; on the wrist or on a
head mast. Not three code paths — one, with the depth strategy chosen from what the
sensor can actually deliver.

**You should not have to tune it.** Grasp height comes from the object's measured size
and your camera's FOV. The reach envelope and IK seeds are probed from your own solver.
The localization error is fitted from the arm's own motion, with forward kinematics as
ground truth.

---

## Does it run on my robot?

| You have | Status |
|---|---|
| An arm with a URDF and a way to command joint angles | Write a profile — one file, no code |
| A webcam | `kind="mono"` — supported, not a downgrade |
| A RealSense / Femto / Kinect | `kind="rgbd"` |
| An OAK-D / ZED / stereo pair | `kind="stereo"` |
| Camera on the gripper | `mount="eye_in_hand"` — visual servo |
| Camera on a mast, tripod, torso | `mount="fixed"` — locate, fuse, stage in |
| No hardware at all | `--arm mock` runs the whole pipeline synthetically |

Mount and sensor are independent choices, and all six combinations are covered by tests
that run with no hardware:

|                 | stereo | RGB-D | mono |
|---|---|---|---|
| **eye_in_hand** | cloud depth | sensor depth | apparent-size range |
| **fixed**       | cloud depth | sensor depth | table-plane range |

Read [docs/porting.md](docs/porting.md) — the actual contract, and it is short.
[docs/architecture.md](docs/architecture.md) explains how the seams work and lists
what is still wrong with them. Once your robot is described,
[docs/calibration.md](docs/calibration.md) covers the three things that must be
*measured* on your hardware rather than declared.

---

## Try it with no hardware

```bash
git clone https://github.com/arminforoughi/RAX && cd RAX
pip install -e ".[dev]"

python -m rax.grasp --list                        # what profiles are installed
python -m rax.grasp --arm head_mono --query "red object"
pytest -q                                         # 180 tests, no robot needed
```

`head_mono` is a webcam on a mast watching a 6-DOF arm — the cheapest rig the stack
supports, and the one that proves the claims. The run surveys the scene, fuses several
views into a position, stages the arm in, descends, and closes on the motor current:

```
[rax] rig: head_mono + mono camera (fixed), detector=color_blob, depth=none
[rax] surveying for 'red object'...
[rax] 'red object' at (+0.259, -0.071) m from 6 views
[rax]   stage 1/3: 12.9 cm to go (ik 0.0 mm)
[rax] descending to z=1.5 cm and closing
[rax] grasp: contact=True (delta_current=220)
[rax] holding. Done.
```

On real hardware:

```bash
# SO-101 with an OAK-D on the wrist
python -m rax.grasp --arm so101 --port /dev/ttyACM0 --query "red cube"

# your arm, a webcam on a mast
python -m rax.grasp --arm my_arm --camera-source 0 --query "cup" --no-move
```

`--no-move` runs perception only and reports where it thinks the object is. Run that
first, with a tape measure.

---

## How it fits together

```
                    rax.grasp                      one entry point
                        |
        +---------------+---------------+
        |                               |
   eye_in_hand                       fixed
   visual servo                  locate -> fuse -> stage
   (GazeEngine)                  (localize + ObjectMap + approach)
        |                               |
        +---------------+---------------+
                        |
                      Rig                          joins the two seams
                   /       \
          ArmInterface   CameraInterface
        (4 methods)      (frame + intrinsics)
               |                |
          ArmProfile       stereo / rgbd / mono
        (your arm as data)  (DepthSource picks itself)
```

| Package | Owns |
|---|---|
| `rax.robots.profiles` | the arm as data — URDF, joint topology, limits, gripper, camera mount |
| `rax.manipulation.arms` | the arm seam, IK strategies, trajectories, the gaze engine, `Rig` |
| `rax.perception` | camera geometry, the camera seam and adapters, three range strategies, hand-eye, self-calibration |
| `rax.models` | swappable detectors (YOLO-World, blob) and stereo backends (RAFT, FoundationStereo, SGBM) |
| `rax.mobility.slam` | the bird's-eye object map and its merge rules |
| `rax.manipulation.approach` | staging geometry, visual centring, and the derived tunables |

Nothing in `perception` or `manipulation` imports a robot. The one rule and the seams
it buys are written up in [docs/architecture.md](docs/architecture.md), including the
two places the rule is still bent.

---

## What is actually derived rather than dialled

The part that is hard to get from a tutorial, and the reason a port converges instead of
needing a week of tuning:

- **`manipulation/arms/workspace.py`** probes `(radius, height, pitch)` with the arm's
  *bare* solver and derives the reach envelope and IK seeds. On the SO-101 it reproduced
  a hand-measured reach table to within one grid step and replaced five
  hand-accumulated seeds with one derived seed at identical coverage.
- **`perception/selfcal.py`** has the arm hold an object; FK gives ground truth, and
  observed-vs-true across a pose sweep fits range scale, push-out and bearing offset. It
  **refuses to apply** a fit when a known error signature would explain the residual
  better — fitting anyway would hide a real bug under a well-tuned patch.
- **`manipulation/approach/derive.py`** computes grasp height, centring tolerance and
  approach trim from the object's measured size and the camera's FOV, rather than from
  one constant tuned on a 5 cm cube.
- **`perception/handeye.py`** refits the camera transform from arm motion, by
  reprojection and by consistency.

---

## Tests

```bash
pytest -q     # 180 tests, no robot, no camera, no model weights
```

The suite is the evidence for the claims above, not a formality:

- `test_head_camera.py` builds one scene and one arm as **three** rigs — mono, RGB-D and
  stereo — and asserts all three locate the object within 2 cm of each other and of the
  truth. If a change welds the stack back to stereo, the mono cases fail.
- `test_wrist_mono.py` covers the sixth combination, a webcam on the hand with no depth
  at all, and pins the specific silent failure it used to have: the servo ran on a
  hardcoded 0.25 m placeholder while reporting healthy progress.
- `test_urdf_kinematics.py` pins the FK/IK backend that replaced the one imported from
  lerobot, including that a prismatic joint is metres and a revolute one is degrees.
- `test_second_arm.py` runs detect → locate → map → approach on a profile that takes the
  *opposite* branch of every seam: pose IK, fixed camera, no URDF.
- `test_grasp_cli.py` runs the published entry point end to end on every advertised rig
  and checks the reported position against ground truth — because a pipeline can
  "succeed" while reaching to the wrong place.
- `test_extraction_parity.py` pins ~450 golden values taken from the original 5,885-line
  monolith before it was broken up, so the extraction is provably faithful. The monolith
  itself is not in the tree — it lives in git history, and the goldens are what survive it.

---

## Examples

The library is the product; these are things built on it, and they are demos rather than
supported surface:

| Path | What |
|---|---|
| `examples/mission_server/` | full pick/place web server with a 3D viewer, guest queue and calibration routes (SO-101 specific) |
| `examples/livekit_gaze/` | LiveKit + Gemini voice agent — talk to the robot, it picks things up |
| `examples/exchange/` | WebSocket remote-control hub |
| `examples/humanoid_k1/` | Booster K1 humanoid control |
| `examples/fpv/`, `examples/teleop/`, `examples/self_learn/` | first-person approach, SLAM, keyboard teleop, self-learning experiments |

Several of these still point at a local `lerobot` checkout by absolute path. They are
demos; the library above them does not.

---

## License

MIT — see [LICENSE](LICENSE).
