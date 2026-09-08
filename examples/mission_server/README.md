# mission_server — first-person-view manipulation demo

See an object, locate it, drive at it, grasp it, put it somewhere. One Flask port
(`:8484`) carries the admin UI, a public guest UI, and the MJPEG first-person view.

**If you want the pipeline rather than the demo, read [`src/rax/grasp.py`](../../src/rax/grasp.py)** —
about 400 lines, no Flask, no threads, and it runs with no hardware at all:

```bash
python -m rax.grasp --arm mock --query "red cube"
```

`mission_server.py` is that pipeline plus everything an unattended public demo needs
(guest queueing, a tunnel, a 3D viewer, teleop, live tuning), so most of its bulk is
the demo, not the robot.

## Running it

```powershell
./run_server.ps1              # detached; also -Status, -Stop, -Force, -Port COMn
```

`run_server.ps1` preflights the two host-side faults that have twice masqueraded as
"camera not found" on this rig. **Read [TROUBLESHOOTING.md](TROUBLESHOOTING.md) before
debugging any camera problem** — neither fault travels with the hardware, so "it works
on my Mac" does not rule them out.

### Environment

| Variable | Default | What it does |
|---|---|---|
| `RAX_ARM` | `so101` | Which arm profile (see `src/rax/robots/profiles/`) |
| `RAX_ARM_PORT` | the profile's | The arm's serial port |
| `RAX_PORT` | `8484` | This server's HTTP port |
| `RAX_LEROBOT_SRC` | *(unset)* | A lerobot **source checkout**, if it is not pip-installed |
| `RAX_YOLO_WEIGHTS` | `yolov8s-worldv2.pt` | YOLO-World weights |
| `GOOGLE_API_KEY` | *(unset)* | Optional; enables the advisory vision checks |
| `RAX_CAMSURV_URL` / `RAX_CAMSURV_PASSWORD` | *(unset)* | Optional room camera served at `/stream2` |

lerobot is a **hardware** dependency, imported only at connect time
(`connect_hardware()`). The module imports, and the tests run, on a machine without it.

## The loop

| Stage | What happens | Where |
|---|---|---|
| **Detect** | YOLO-World (open vocabulary) in a parallel thread (~2.5 s) so it never sits in the control path. The two cube colours also get strict-HSV trackers, tighter during close approach. A label naming a colour is gated on containing it. | `models/detection/`, `color_filters.py` |
| **Locate** | Monocular. The table *is* the base plane, so an object's pixel ray meets it at a known height; apparent size against a class prior is the fallback. Neither needs depth — which is why the OAK-D runs colour-only. | `perception/locate.py` |
| **Map** | Detections fold into a 2D bird's-eye map, merged by **position** (one object fires under several labels) and expiring after `MAP_TTL_S`. | `mobility/slam/object_map.py` |
| **Approach** | Three stages — the first covers ~90%, the rest correct — re-measuring between them, so error is corrected while there is still room to correct it. IK holds the tool pitch throughout. | `manipulation/approach/` |
| **Centre** | The last correction is by eye: servo the object onto the fingertip pixel. The servo **measures its own gains by probing** rather than modelling them. | `approach/visual_center.py` |
| **Grasp** | Descend, close on servo current, and measure fingertip height at the instant of contact — that one number is the whole carry's grip→bottom distance. | `run_mission` |
| **Place** | Release height needs only the destination's height plus that measured offset. | `place_at` |

## Tasks: "red on green"

A phrase naming a destination is an *instruction*, not a vocabulary. Typed into
**either** the task box or the detection-query box, it runs as a task:

- `/start` checks `reads_as_a_task()` before falling back to a plain pick.
- `/setquery` expands it to the vocabulary the task needs, so both the object **and its
  destination** can reach the map.
- `run_task` scans if anything it names is not on the map yet, instead of refusing.

This used to be a trap. The phrase in the query box read as two class names, `/start`
picked whichever came first, and the arm grasped the red cube and folded home reporting
success — nothing had failed, the place half was simply never asked for.
`tests/test_task_routing.py` pins that exact run.

## Routes

Everything not in the guest allowlist is admin-only, enforced by a Host-header gate
(`_gate_public`) so the public tunnel exposes only a small surface.

**Mission** — `/start` `/stop` `/task` `/pickplace` `/place` `/home` `/reset`
`/relax` `/wake` `/setrelaxidle`
**Perception** — `/setquery` `/setconf` `/preset` `/scan2d` `/map2d` `/goto2d`
`/clearmap2d` `/locate` `/survey` `/relocate` `/gemini`
**Tuning** — `/setknob` (any knob by name; `GET` lists them), `/setcubesize`,
`/pushout`
**Calibration** — `/calib` `/calibmount` `/caltip` `/floorcal` `/floor` `/probe3d`
`/debugdepth` `/selfcal/*`
**View** — `/` `/status` `/stream` `/stream2` `/geom` `/urdf` `/ui/<asset>`
**Teleop** — `/jogpress` `/jogrelease` `/jogvec`
**Guest** — `/guest` `/guest/*` `/guests` `/guestkick` `/guestlink` `/feedback`
`/feedbacklist`
**Lifecycle** — `/shutdown`

`POST /shutdown` is the only stop path proven not to wedge the OAK-D: it quiesces the
frame threads, stops the detector, and disconnects the robot **before** the camera. On
Windows a hard `Stop-Process` cannot be intercepted by anything.

### Live tuning

`/setknob?name=<knob>&value=<v>` tunes anything in `KNOBS`; `GET /setknob` lists what
there is. Six per-knob routes (`/setaimdu`, `/settrim`, `/setback`, `/setsteps`,
`/setrangescale`, `/setbearing`) were folded into it — each re-implemented the same
clamp-and-log against one hardcoded name, so a new knob was untunable until someone
wrote it a seventh route.

`/setcubesize` is deliberately separate: it sets a perception prior
(`PRIORS.fallback_edge_m`), not an approach knob, so it has no entry in `KNOBS`.

## Known limits

- **The hand-eye TF is rotationally wrong.** Table rays come out too shallow, so
  absolute ranges are compressed and mapped positions are approximate. Most of the
  localization care in this file works around it: locating from *one* fixed pose turns
  a varying error into a constant one, and the visual centring absorbs the rest.
- **`/calib`'s reprojection fit has a flat direction** — seeded with the exact true
  transform on noise-free data it still drifts ~14 mm / ~3° while reporting a perfect
  0.5 px residual. `/calibmount`'s consistency objective does not have this failure mode.
- **Footprint measurement is geometrically limited at the current camera angle.** The
  contact pixels sit where the sightline barely grazes the table; it needs a
  steeper-angle pose, or to accept priors at this one.
- The stacking path (`run_task` → pick → place) is confirmed by tests; the **place**
  half has not been re-run on hardware since the task-routing fix.

## Files

| | |
|---|---|
| `mission_server.py` | the server |
| `gemini_vision.py` | advisory vision checks — miss diagnosis, grasp verification |
| `guest_sessions.py` | the public queue/turn/cooldown policy |
| `supervise.py` | restarts this and camserver if either dies |
| `ui/` | admin, guest, and 3D-viewer front ends |
| `run_server.ps1`, `fix_oakd.ps1` | launch with preflight; the host-side OAK-D repair |
| `TROUBLESHOOTING.md` | the OAK-D dual-identity trap — read before debugging the camera |
| `modularization.md` | historical: how the monolith was broken apart |
