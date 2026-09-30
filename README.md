# RAX

**Pick up any object a camera can see, with no training.**

RAX drives a low-cost arm with a wrist camera to an object, lines the object up
between the jaws, squares the grip to it, and grasps. It needs the arm's URDF, a wrist
camera and a detector: a colour, or a text prompt through an open-vocabulary model.
There are no demonstrations, no learned policy and no hand-tuned signs.

On an SO-101 sorting lab tubes into racks, it picks about 80% of the tubes it maps and
stands about half of them upright in a 1 mm-clearance hole. Every run is logged as a
self-labelled episode.

Project page: https://arminforoughi.github.io/RAX/

## How it works

The pick is visual servoing on the arm's own kinematic model. It repeats one short
loop:

1. **See.** The detector gives the object's pixel. The error is how far that pixel is
   from where the fingertips will land, projected into the same image.
2. **Measure.** A small test turn of the base shows how many pixels the object moves
   per degree, so the correction is `error / gain`. No sign or scale is assumed; on the
   SO-101 the base turns the opposite way to intuition, and a hard-coded sign drove it
   steadily away from the object.
3. **Solve.** Every move is inverse kinematics on the URDF with the hand's angle held,
   started from several poses, keeping the solution nearest the current one.

In order: LOOK, PROBE, AIM, APPROACH, HOVER, STAND, TRIM, TWIST, DESCEND, GRASP, LIFT.
The image only ever turns the base. The reach comes from casting the pixel onto the
table, and then from a measured pixels-per-metre once the hand is over the object. The
last centimetres are a straight vertical drop, so the fingers never sweep the object
away. The grip is judged by where the jaws stop: shut means empty, stopped short means
holding, and too wide means it took two.

## Use it

```python
from rax.pick import pick, place, scan, ColourTarget, PromptTarget, TOP, SIDE

target = PromptTarget(prompt="cup", grasp_z=0.04)      # or ColourTarget(colours=("red",))
for obj in scan(arm, target):                          # map the table once
    pick(arm, target, near_xy=(obj.x, obj.y), grasp=TOP)   # TOP: 90deg down, SIDE: level
    place(arm, (0.20, -0.15), release_z=0.08, pitch=0, roll=90)
```

`arm` is anything that implements [`rax.pick.arm.Arm`](src/rax/pick/arm.py), about
fifteen methods: joints, FK, IK, a camera frame, pixel-to-table and back, the gripper.
[`rax.robots.so101.So101`](src/rax/robots/so101.py) is the real SO-101 with a wrist
OAK-D.

### No robot? Run the simulator

```
pip install -e .
python examples/pick_demo.py
```

[`rax.pick.sim.SimArm`](src/rax/pick/sim.py) uses the SO-101's real URDF, IK and camera
model, with a drawn image and a simple gripper, so the whole loop runs closed on a
laptop. It tests the logic, not the camera.

### The tube-sorting rig

```
pip install -e ".[robot]"
python examples/tube_sorting/server.py --port COM4      # UI on http://127.0.0.1:8486
```

It maps the mat with one wrist-camera sweep, picks each tube, stands it up and drops it
into its colour's rack. An optional overhead camera lines the tube up over the hole and
checks that it went in. See [`examples/tube_sorting`](examples/tube_sorting).

## Layout

```
src/rax/pick/         the method: pick, place, scan, targets, simulator, episodes
src/rax/kinematics/   URDF forward kinematics, pitch-holding IK, smooth moves
src/rax/perception/   camera model (pixels <-> base frame), tube-cap detector
src/rax/robots/       SO-101 driver, profile and URDF
examples/             pick_demo.py (simulator), tube_sorting/ (the real rig)
tests/                pytest; no hardware needed
```

## License

See [LICENSE](LICENSE).
