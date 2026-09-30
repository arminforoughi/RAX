"""Put the held object down at a spot, and find objects on the table in the first place."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .arm import Arm, bearing_of, move_to, pan_for_bearing, solve, with_pan
from .targets import Target


def place(arm: Arm, xy, release_z: float, pitch: float, roll: float,
          hover_z: float | None = None, carry_z: float = 0.25, align=None,
          holding=None, align_steps: int = 5, align_gain: float = 0.7,
          align_max_m: float = 0.04) -> tuple[float, float]:
    """Carry the held object high, come over ``xy``, line up, lower, release.

    ``release_z`` and ``hover_z`` are fingertip heights. ``align()`` (optional) returns
    the remaining (dx, dy) error in metres from any other sensor, e.g. an overhead
    camera, or None if it cannot see; the hover is nudged by it until it is small.
    ``holding()`` (optional) says whether the object is still in the jaws. Returns the
    (x, y) the object was released at.
    """
    hover_z = release_z + 0.05 if hover_z is None else hover_z
    pitch = _usable_pitch(arm, xy, (hover_z, release_z), pitch, roll)

    # UP, TURN, OVER, DOWN: the object passes above everything on the table, as high
    # as the arm can hold this hand angle here (straight down, it cannot go very high)
    tip = arm.tip(arm.joints())
    lift = _highest(arm, tip[:2], max(carry_z, hover_z), hover_z, pitch, roll)
    if lift is not None:
        arm.move(lift, speed=1.5, settle=0.1)
    q = arm.joints()
    arm.move(with_pan(arm, q, pan_for_bearing(arm, q, bearing_of(xy))), speed=1.5, settle=0.1)
    high = _highest(arm, xy, max(carry_z, hover_z), hover_z, pitch, roll)
    if high is not None:
        arm.move(high, speed=1.5, settle=0.1)
    over = move_to(arm, (xy[0], xy[1], hover_z), pitch, roll, "the hover", speed=1.2)

    aim = np.array(xy[:2], float)
    if align is not None:
        arm.phase("ALIGN", "lining up over the spot")
        moved = 0.0
        for _ in range(align_steps):
            arm.checkpoint()
            e = align()
            if e is None:                          # lost it: trust the spot, not a nudge
                if moved:
                    aim = np.array(xy[:2], float)
                    arm.move(over, speed=0.8, settle=0.3)
                break
            if math.hypot(*e) < 0.004:
                break
            step = align_gain * np.asarray(e, float)
            moved += float(np.linalg.norm(step))
            if moved > align_max_m:
                arm.log("        align: that much correction is a mis-detection — stopping")
                break
            aim = aim + step
            move_to(arm, (aim[0], aim[1], hover_z), pitch, roll, "the adjusted hover",
                    speed=0.8, settle=0.3)

    if holding is not None and not holding():
        raise RuntimeError("dropped it while carrying")
    arm.phase("RELEASE", "down and open")
    move_to(arm, (aim[0], aim[1], release_z), pitch, roll, "the release height",
            speed=0.8, settle=0.3)
    arm.release()
    time.sleep(0.5)
    up = _highest(arm, aim, hover_z + 0.03, release_z + 0.01, pitch, roll)  # back up,
    if up is not None:                                   # as far as this angle allows
        arm.move(up, speed=1.4, settle=0.1)
    return float(aim[0]), float(aim[1])


def _usable_pitch(arm: Arm, xy, heights, pitch: float, roll: float) -> float:
    """The hand angle nearest ``pitch`` (within 40deg) that reaches ``xy`` at every one of
    ``heights``. Far out, the arm cannot point straight down; it tilts, like the pick."""
    for d in (0, -5, 5, -10, 10, -15, 15, -20, 20, -25, 25, -30, 30, -35, 35, -40, 40):
        p = pitch + d
        if all(solve(arm, (xy[0], xy[1], z), p, roll, tol=0.01) is not None for z in heights):
            if d:
                arm.log(f"        placing at {p:.0f}deg: {pitch:.0f} does not reach there")
            return float(p)
    raise RuntimeError(f"cannot reach ({xy[0]*100:+.1f},{xy[1]*100:+.1f})cm at any "
                       f"angle near {pitch:.0f}deg")


def _highest(arm: Arm, xy, z_top: float, z_min: float, pitch: float, roll: float):
    """Joints for the highest fingertip height in [z_min, z_top] above ``xy`` that holds
    ``pitch``, or None."""
    z = z_top
    while z >= z_min - 1e-6:
        q = solve(arm, (xy[0], xy[1], z), pitch, roll, tol=0.01)
        if q is not None:
            return q
        z -= 0.02
    return None


@dataclass
class Found:
    """An object on the map: a running mean of every sighting."""

    label: str
    x: float
    y: float
    n: int = 1


def scan(arm: Arm, target: Target, bearings_deg=(40.0, 20.0, 0.0), tilts_deg=(0.0, 25.0),
         merge_m: float = 0.045, max_v: float | None = None, avoid=None,
         reach=(0.08, 0.55), frames: int = 4) -> list[Found]:
    """Sweep the wrist camera across the table once and map every object it sees.

    At each bearing the wrist also tilts down, to see near the base. Sightings of the
    same label within ``merge_m`` are one object. ``max_v`` drops detections below that
    image row (the gripper's own strip).
    """
    found: list[Found] = []
    q = np.array(arm.home, float)
    arm.move(q, speed=1.4, settle=0.2)
    wrist = arm.pitch_chain[-1]
    w0 = float(q[wrist])
    for n, bear in enumerate(bearings_deg):
        arm.checkpoint()
        q[arm.pan] = pan_for_bearing(arm, q, math.radians(bear))
        for tilt in (tilts_deg if n % 2 == 0 else tuple(reversed(tilts_deg))):
            q[wrist] = float(np.clip(w0 + tilt, arm.lo[wrist], arm.hi[wrist]))
            arm.move(q, speed=1.5, settle=0.25)
            for _ in range(frames):
                bgr = arm.frame()
                if bgr is None:
                    continue
                qn = arm.joints()
                for d in target.detect(bgr):
                    if max_v is not None and d.v > max_v:
                        continue
                    xy = arm.cast((d.u, d.v), qn)
                    if xy is None or not reach[0] <= math.hypot(*xy) <= reach[1]:
                        continue
                    if avoid is not None and avoid(xy):
                        continue
                    _merge(found, d.label, xy, merge_m)
    arm.move(np.array(arm.home, float), speed=1.5, settle=0.2)
    return found


def _merge(found: list[Found], label: str, xy, merge_m: float) -> None:
    for f in found:
        if f.label == label and math.hypot(f.x - xy[0], f.y - xy[1]) <= merge_m:
            f.n += 1
            f.x += (xy[0] - f.x) / f.n
            f.y += (xy[1] - f.y) / f.n
            return
    found.append(Found(label, float(xy[0]), float(xy[1])))
