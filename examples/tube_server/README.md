# Tube server — one pick-and-place, any arm, with a 3D view

Picks a lab tube up and puts it in a rack. Runs on the X250, on the SO-101, or on a
simulator that needs no hardware at all — and the pick logic is the same code in all
three cases.

```bash
# no hardware needed: real kinematics, simulated arm and tubes
python tube_server.py --arm sim --profile x250
python tube_server.py --arm sim --profile so101     # the same page, the other arm

# the real X250
python tube_server.py --arm x250 --serial COM5 --wrist-cam 0
```

Then open <http://127.0.0.1:8486/>.

| Flag | Default | Meaning |
|---|---|---|
| `--arm` | `sim` | `sim`, `x250` or `so101` |
| `--profile` | `x250` | which arm the simulator should *be* (sim only) |
| `--port` | `8486` | |
| `--serial` | `$RAX_X250_PORT` or `COM5` | X250 serial port |
| `--wrist-cam` | `0` | OpenCV index for the wrist camera |
| `--episodes` | `./episodes.jsonl` | where runs are recorded |

## What the UI shows

The 3D scene is the page, because the question an operator actually has — *where does it
think the tube is, and is the hand going there* — is a spatial one. The arm is posed from
its own URDF, tubes draw as capped cylinders at their mapped position, racks as hole grids
with occupied holes filled in. Down the side: which tube to pick, which rack to drop it in,
the live gripper verdict, and the log.

Every tube carries its **source** — `seen`, `mapped` or `assumed` — because a map that does
not distinguish a fresh fix from a stale one invites trusting the stale one.

When the rig is simulated the page says so in a banner it cannot dismiss. A demo you cannot
tell apart from a real run is worse than no demo.

## What is shared, and what is not

This is not a third implementation of a pick. It is the X250's `pick.py` (878 lines) and the
SO-101's `mission_server.py` (9,682 lines) with the parts they had in common taken out:

| Concern | Lives in | Was duplicated? |
|---|---|---|
| the approach decision table | `rax.manipulation.approach.visual_servo` | no — already shared |
| what each joint does to the picture | `rax.manipulation.approach.jacobian` | no — already shared |
| the control seam | `rax.manipulation.approach.servo_arm` | no — already shared |
| **did the jaws get it** | `rax.manipulation.grip` | **yes → extracted** |
| **try again** | `rax.manipulation.attempt` | **yes → extracted** |
| **what happened** | `rax.manipulation.episodes` | **yes → extracted** |
| **where to put it** | `rax.perception.rack_holes` | **yes → extracted** |
| **what the arm IS** | `rax.robots.profiles.{x250,so101}` | **yes → added for X250** |
| **link geometry for 3D** | `rax.robots.urdf_visuals` | **yes → generalised** |
| detect / goto / clamp | the backend, `rig.py` | no — genuinely per-arm |

So the arm-specific surface left is a `TubeRig` (about 60 lines per arm) and nothing else.
**Adding a third arm means describing it, not reimplementing a pick.**

### The two seams stack

```
TubeRig        what a SERVER needs: description, frames, gripper, racks, map
  └─ .servo_arm()  →  ServoArm      what the APPROACH needs: four members
```

`ServoArm` stays deliberately tiny — no kinematics, no intrinsics, no hand-eye — because
that is what lets it drive an arm with no URDF at all. Everything a *server* additionally
needs went into `TubeRig` rather than being pushed down into the servo, which would have
made the servo unportable.

### One verdict, two sensors

The arms do not agree on how to tell whether they are holding something:

- **SO-101** — the gripper's **current** rises while closing. Resistance means contact.
- **X250** — the gripper's settled **position**. Measured over 113 demonstrations,
  holding reads above 31.9 and closing on air settles at 30.2, *with no overlap*.

Different signals, same blind spot: neither can tell you **what** the jaws met. So both get
the same asymmetric camera override — a confident camera "empty" clears a carry, a camera
"holding" never authorises one. See `rax/manipulation/grip.py`, which is mostly an
explanation of why that asymmetry is the design.

## The X250's 3D model

**There is no vendor X250 URDF on this rig, in this repo, or in site-packages.** So
`src/rax/robots/arms/x250/X250/x250.urdf` was written, and it is explicit about which of its
numbers are which:

- **Measured** — the joint structure (a broadcast ping found ids 2–7) and the joint travel
  (the arm's own lerobot calibration, at 360°/4096 ticks).
- **Fitted** — the zero offsets, solved so the demonstrated grasp pose puts the fingertip on
  the bench, 27 cm out, hand pointing down. Re-runnable via
  `rax.robots.arms.x250.normalise.fit_zero_norm`.
- **Nominal** — every link length. Nobody has put a ruler on this arm.

**So: the rendered arm moves correctly and is drawn approximately.** Watch an approach with
it; do not read a reach off it. To fix that, measure the distances between joint axes and
replace the `<origin xyz>` values — then re-run `fit_zero_norm`, because the offsets absorb
whatever the lengths get wrong and the two are coupled. A test fails if you forget.

Dropping in the genuine Interbotix URDF is a one-line change to `profiles/x250.py`; the
viewer needs no changes either way, because `urdf_visuals` tessellates primitives *and*
loads STL meshes.

## Status

- **Simulator** — complete. Picks, places, retries, records episodes, and can miss.
- **X250 hardware** — the approach path is wired to the shared visual servo through
  `X250ServoArm`, but the **gripper close is not implemented** and it has not been run:
  COM5 does not currently enumerate. `hardware_pick` raises `NotImplementedError` at that
  point rather than pretending.
- **SO-101** — `So101Rig` takes callables rather than a robot, because the OAK-D and the bus
  are owned by the mission server's threads and two processes opening that camera is how it
  dies. It has to be constructed *inside* that process.

## Episodes

Every run appends one JSON line: what was attempted, every individual attempt, and how long
it took. `GET /episodes` returns the tally. This is what makes "did that change help" a
countable question instead of an impression — the same mechanism that killed the
wrist-yaw-alignment experiment at 0/6 against 6/6.
