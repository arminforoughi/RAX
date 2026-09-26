# X250 tube pick (and place)

A scripted, vision-guided pick of a lab tube by cap colour on the X250 arm, with an
optional place into the matching rack. It is the non-learned baseline that runs next to
the SmolVLA / ACT policies: it uses the same robot driver and the same wrist camera, and
every pose it moves through comes from the 113 teleoperated demonstrations.

```
pick.sh      wrapper: sets the ffmpeg library path and runs pick.py inside the lerobot_trossen uv env
pick.py      the controller
config/      snapshot of the calibration files pick.py loads (see "Calibration files")
```

## Running it

```bash
./pick.sh --colour blue --wrist-cam 0 --top-cam 1 --ui           # pick and hold for 5s
./pick.sh --colour gold --wrist-cam 0 --top-cam 1 --ui --place   # pick, then place in the rack
./pick.sh --colour green --dry-run                               # print poses and envelope, no hardware
```

| Flag | Default | Meaning |
|---|---|---|
| `--colour` | required | `gold`, `blue` or `green` cap |
| `--port` | `/dev/tty.usbserial-FTA9DQBQ` | arm serial port |
| `--wrist-cam`, `--top-cam` | `-1` (auto) | OpenCV indices. Auto-detect moves the arm and picks the camera whose image changes; it can leave a device in a mode the robot then refuses, so **pass both explicitly for a real run** |
| `--aim-steps` | 14 | max base-rotation steps in AIM |
| `--aim-tol` | 150 | px of horizontal error accepted before approaching |
| `--approach-steps` | 15 | increments in the descent |
| `--attempts` | 2 | full LOOK→GRASP retries |
| `--ui` | off | live window with wrist + overhead feeds and the controller's view |
| `--place` | off | after a confirmed grasp, carry to the rack and release into a free hole |
| `--twist` | off | rotate the tool so the jaws close across the tube axis |
| `--dry-run` | off | print LOOK/GRASP poses and the joint envelope, then exit |

`Ctrl-C` stops cleanly; torque is released on exit either way.

## What it does

Nothing extends until there is a target. The previous version lunged to an extended
standoff pose before locating anything, and that lunge is what hit the bench.

1. **LOOK**: move to the retracted `look` pose (raised, not reached out) with the jaws open.
2. **AIM**: rotate the **base only** until the cap is in view and within `--aim-tol` px of
   the grasp point horizontally. If the cap is not visible it sweeps, reversing every 4 steps.
3. **Probe base direction**: nudge the base (+7, −10, +14) until the cap moves at least 18px,
   which gives the sign of the correction. Without this, a wrong sign stays hidden under
   detector noise.
4. **APPROACH**: interpolate shoulder and elbow from the current pose toward the
   demonstrated `grasp` pose in `--approach-steps` increments. Each step first *looks*,
   then does one combined move (reach plus a base correction). The image does not decide
   the vertical: the wrist camera tilts as the arm reaches, so image-space and physical
   alignment disagree. The descent follows the demonstrations and vision corrects only the
   horizontal.
   - Close when the cap is within `GRASP_RADIUS` (70px) of the grasp point **and** the
     descent is ≥85% complete.
   - Losing the cap when close (<170px) and ≥70% down counts as "passed under the jaws":
     finish the descent and close. Losing it far away holds position; after 4 misses the
     attempt stops.
   - With `--twist`, the tool gain is measured at step 3 and the tube axis is aligned from
     step 4 on.
5. **GRASP**: close, wait until the gripper reading **settles**, then compare it with the
   thresholds: holding reads > 31.9, closed on nothing reads 30.2. There was no overlap
   across the 113 demonstrations, so a failed grasp is detected rather than assumed.
   After a grasp the arm lifts in five gentle 4-unit hops, checking the grip after each one.
   It saves `held_<colour>.png`.
6. **PLACE** (`--place`): gold goes to the grey rack, blue and green to the black rack
   (routing taken from the demonstrations). Fold to the LOOK shoulder/elbow, rotate the base,
   then unfold over the rack, checking the grip at each stage. Descend a third of the way,
   pick a free hole from the wrist view, and follow the demonstrated `release` profile while
   correcting the base toward the hole. Open, verify the gripper is > 45, back off and save
   `place_target_<rack>.png` and `placed_<rack>.png`.

### Safety bounds

- Every commanded pose is clamped to `safe_envelope.json`, the joint box the demonstrations
  visited.
- `goto()` ramps to the target and then re-sends it until every joint is within 3 units. It
  reports failure if the arm stalls (an obstruction or a limit) instead of grinding.
- The robot is created with `max_relative_target=8.0`.
- Only the wrist camera is attached to the robot. The overhead camera is opened separately
  on a best-effort basis for the UI, because a stale overhead frame used to abort runs.

> **Note:** the shipped `safe_envelope.json` allows base −40.6 … +11.7. That is wider than
> the −15.4 … +12.7 the `pick.py` docstring quotes, presumably to admit the rack carry pose
> (base ≈ −28.8). Tighten it if you only run picks.

## Calibration files

`pick.py` reads these from `~/smolvla_runs/` (`RUNS` in the script). It also writes its
photos there. `config/` holds the versions this was last run with:

| File | Contents |
|---|---|
| `poses.json` | `look`, `look_far` and `grasp` joint poses (grasp = demonstrated median) |
| `safe_envelope.json` | per-joint `[min, max]` from the demonstrations |
| `place_poses.json` | per rack (`grey`, `black`): `carry` and `release` poses |
| `gripper_geometry.json` | fixed grasp point, finger positions and exclude radius in the 640×480 wrist image |

These are specific to one rig's camera mount and bench layout. Re-measure them if either
changes.

## Dependencies outside this repo

The script is not yet wired into `rax`. It runs against the original lab checkout:

- `~/lerobot_trossen`: the uv environment `pick.sh` runs in (`uv run --no-sync`); it provides
  `lerobot.cameras.opencv`.
- `~/Documents/lab-robot`, added to `sys.path` by the script:
  - `x250_driver`: `X250Follower`, `X250FollowerConfig`
  - `perception/caps2.py` (`find_caps`), `orient.py` (`tube_axis`, `twist_error`),
    `holes.py` (`find_holes`, `pick_free_hole`, `draw`), `jaws.py` (`grasp_point`,
    `find_jaws`), `cameras.py` (`find_by_motion`, `force_mode`), `ui.py` (`show`, `close`)
- ffmpeg 7 from Homebrew (`/opt/homebrew/opt/ffmpeg@7/lib`) for the video decoding in lerobot.

Porting it onto `rax.robots` / `rax.perception` (for example `models/detection/lab_tubes.py`,
`rack_holes.py`, `manipulation/visual_servo.py`) is the natural next step.
