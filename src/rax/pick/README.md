# rax.pick

Pick up any object a wrist camera can detect, with no training and no hand-tuned
signs. It needs the arm's URDF, a wrist camera, and a detector for the object.

```python
from rax.pick import pick, place, scan, PromptTarget, TOP, SIDE

cup = PromptTarget(prompt="cup", grasp_z=0.04)        # open-vocabulary, no training
for obj in scan(arm, cup):                            # map the table once
    pick(arm, cup, near_xy=(obj.x, obj.y), grasp=SIDE)
    place(arm, (0.20, -0.15), release_z=0.06, pitch=0, roll=0)
```

## How it works

The pick is visual servoing on the arm's own model:

1. **See.** The detector gives a pixel. The error is how far that pixel is from where
   the fingertips will land, projected into the same image.
2. **Measure.** A small test turn of the base shows how many pixels the object moves
   per degree. Dividing the error by that gives the correction, so no sign or scale is
   ever assumed.
3. **Solve.** Every move is inverse kinematics on the URDF with the hand's angle held,
   started from several poses, keeping the solution nearest the current one.

The stages are LOOK, PROBE, AIM, APPROACH, HOVER, STAND, TRIM, TWIST, DESCEND, GRASP
and LIFT (see `pick.py`). The image only ever turns the base. The reach comes from
casting the pixel onto the table, and then from a measured pixels-per-metre once the
hand is over the object.

**Grasp angle.** `TOP` comes straight down (pitch 90) and rolls the wrist square to the
object's long axis. `SIDE` keeps the hand level (pitch 0) and reaches in horizontally.

**Grip check.** The grip is judged by where the jaws stop: closed means empty, stopped
short means holding, and too wide means it took two.

## Plugging in

- **A robot:** implement `rax.pick.arm.Arm`, about fifteen methods covering joints, FK,
  IK, camera frame, cast and project, gripper, and log.
  `examples/mission_server/so101_arm.py` does it for the SO-101.
- **An object:** subclass `Target` and write `detect()`. `ColourTarget` finds HSV blobs
  (test-tube caps). `PromptTarget` uses YOLO-World with a text prompt.

## Try it without a robot

```
python examples/pick_demo.py
```

`rax.pick.sim.SimArm` is the SO-101's real kinematics and camera model with a drawn
image and a simple gripper, so the full loop runs closed on a laptop. The sim tests the
logic, not the camera: a real run is the only real test.
