"""Pick any object the wrist camera can detect: no training, no hand-tuned signs.

The loop is visual servoing on the arm's own model. The camera judges one thing,
the pixel error between the object and the grip; test moves measure what a joint does
to that error; inverse kinematics on the URDF turns each correction into joint angles.

    LOOK      raise the arm, open the jaws, face the object's bearing (from the map)
    PROBE     turn the base a few degrees and count how far the object moves: px/deg
    AIM       turn the base until the object is near the jaws horizontally
    APPROACH  reach toward it in small bites, re-reading the image before each one
    HOVER     straight down to a hover, still able to see it
    STAND     tilt the hand to the grasp angle (90 = straight down, 0 = level)
    TRIM      reach by eye until the object sits at the fingers' landing point
    TWIST     roll the wrist square to the object's long axis, and check it turned
    DESCEND   straight down at a fixed angle, so the fingers never sweep it away
    GRASP     close, and judge the grip by where the jaws stopped
    LIFT      raise it clear

The image moves the BASE only, never the reach: correcting the tip in x and y from a
pixel error drives the arm out past the object. The reach comes from casting the pixel
onto the table and, once the hand is over the object, from a measured px-per-metre.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .arm import (
    Arm,
    bearing_of,
    fold,
    pan_for_bearing,
    pitch_of,
    solve,
    steepest,
    straight_down,
    with_pan,
)
from .targets import Detection, Target


@dataclass
class Grasp:
    """How the hand meets the object."""

    pitch: float = 90.0     # 90: straight down onto it; 0: level, from the side
    twist: bool = True      # roll the wrist square to the object's long axis


TOP = Grasp(90.0, True)
SIDE = Grasp(0.0, False)


@dataclass
class PickConfig:
    """Every number the pick uses. Heights are fingertip heights above the table."""

    approach_z: float = 0.08     # reach out at this height (lower slides the object)
    hover_z: float = 0.06        # come down to here before standing the hand up
    square_z: float = 0.06       # square the hand to the grasp angle here...
    square_low_z: float = 0.035  # ...or lower, where the arm reaches 90 further out
    lift_m: float = 0.09

    probe_deg: float = 8.0       # base test move that measures px/deg
    default_gain: float = 0.0    # px/deg if the probe loses the object (0: don't steer)
    aim_tol_px: float = 150.0
    aim_offset_px: float = 0.0   # >0 puts the object this far LEFT of the grip centre
    centre_tol_px: float = 28.0  # across the reach
    along_tol_px: float = 45.0   # along the reach
    target_v: float | None = None  # image row to reach the object to; None: the landing point

    approach_steps: int = 12
    approach_lead_m: float = 0.02   # stop this short of the cast
    approach_tol_m: float = 0.015
    bite_m: float = 0.025           # largest reach step
    probe_bite_m: float = 0.015     # first trim step, to measure px/m
    trim_budget_m: float = 0.07
    trim_steps: int = 14
    max_joint_jump: float = 45.0    # a bigger jump is an elbow flip, not a reach

    twist_tol: float = 15.0      # degrees out of square worth fixing
    twist_fraction: float = 1.0  # roll this much of the error
    twist_max: float = 180.0     # largest single roll
    twist_ambiguous: float | None = None  # past this, always roll negative (rig quirk)

    # the grip verdict, from where the jaws stop (percent open); None: the arm's own
    # ``grip_levels`` (air, blocked, jammed, two)
    grip_air: float | None = None       # closed on nothing
    grip_blocked: float | None = None   # stopped above this: something is between the jaws
    grip_jammed: float | None = None    # stopped above this: a false contact, never closed
    grip_two: float | None = None       # wider than one object allows: took two

    @classmethod
    def for_arm(cls, arm, **overrides) -> PickConfig:
        """The defaults, then what this arm was tuned to (``arm.pick_tuning``), then
        ``overrides``: every app on the same arm picks the same way."""
        return cls(**{**dict(getattr(arm, "pick_tuning", None) or {}), **overrides})


@dataclass
class PickResult:
    grip_pct: float
    off_px: float | None
    pitch: float


class Eye:
    """Keeps sight of ONE object across frames: same label, nearest where it last was."""

    def __init__(self, arm: Arm, target: Target, avoid=None):
        self.arm, self.target, self.avoid = arm, target, avoid
        self.label: str | None = None
        self.uv: tuple[float, float] | None = None
        self.bgr = None
        self.hits = 0                 # detections since the last reset

    def all(self) -> list[Detection]:
        self.bgr = self.arm.frame()
        if self.bgr is None:
            return []
        dets = self.target.detect(self.bgr)
        if self.avoid is not None:
            q = self.arm.joints()
            dets = [d for d in dets if not self._avoided(d, q)]
        return dets

    def _avoided(self, d, q):
        xy = self.arm.cast((d.u, d.v), q)
        return xy is not None and self.avoid(xy)

    def see(self, tries: int = 4) -> Detection | None:
        for _ in range(tries):
            dets = [d for d in self.all() if self.label is None or d.label == self.label]
            if dets:
                if self.uv is None:
                    d = max(dets, key=lambda d: d.area)
                else:
                    d = min(dets, key=lambda d: (d.u - self.uv[0]) ** 2 + (d.v - self.uv[1]) ** 2)
                self.uv, self.label = (d.u, d.v), d.label
                self.hits += 1
                return d
            time.sleep(0.08)
        return None


def _pan_step(ex: float, gain: float, lo: float, hi: float) -> float:
    """Degrees of base that remove ``ex`` pixels: ex/gain, its size clipped to [lo, hi]."""
    if not gain:
        return 0.0
    step = ex / gain
    return math.copysign(float(np.clip(abs(step), lo, hi)), step)


def pick(arm: Arm, target: Target, near_xy=None, label: str | None = None,
         grasp: Grasp = TOP, cfg: PickConfig | None = None, avoid=None) -> PickResult:
    """Pick one object. Returns how it went, or raises RuntimeError saying why not.

    ``near_xy``: where the map says the object is (base frame, metres); the arm faces it
    first and takes the detection nearest it. ``label``: only detections with this label.
    ``avoid(xy)``: detections casting to where this is True are ignored (e.g. racks).
    """
    cfg = cfg or PickConfig.for_arm(arm)
    eye = Eye(arm, target, avoid)
    eye.label = label
    roll = float(arm.joints()[arm.roll])

    # ---- LOOK --------------------------------------------------------------------
    arm.phase("LOOK", f"looking for the {target.name}")
    arm.move(arm.home, speed=1.4, settle=0.2)
    arm.grip(target.open_pct)
    det = _find(arm, eye, near_xy)
    if det is None:
        raise RuntimeError(f"no {label or target.name} in view")
    xy = arm.cast((det.u, det.v), arm.joints())       # None without a hand-eye transform
    arm.log(f"        {det.label or target.name} at ({det.u:.0f},{det.v:.0f})px"
            + ("" if xy is None else f" -> ({xy[0]*100:+.1f},{xy[1]*100:+.1f})cm"))
    if near_xy is None and xy is not None:           # face it before anything else
        det = _find(arm, eye, xy) or det
        xy = arm.cast((det.u, det.v), arm.joints()) or xy

    # ---- PROBE, AIM ----------------------------------------------------------------
    arm.phase("PROBE", "measuring how far the base moves the object")
    gain = _probe(arm, eye, cfg)
    arm.phase("AIM", "turning the base to bring it across")
    xy = _aim(arm, eye, gain, cfg) or xy

    # ---- APPROACH at the look angle, then HOVER ------------------------------------
    arm.phase("APPROACH", "reaching over it, keeping it in view")
    pitch_see = pitch_of(arm, arm.joints())
    eye.hits = 0
    r_goal, off = _approach(arm, eye, gain, xy, pitch_see, roll, cfg)
    if eye.hits == 0:
        raise RuntimeError(f"lost the {label or target.name} before reaching it "
                           f"— not going down blind")
    if r_goal is None:
        tip = arm.tip(arm.joints())
        r_goal = math.hypot(tip[0], tip[1])

    if grasp.pitch < 45.0:
        return _side_grasp(arm, eye, target, gain, r_goal, roll, grasp, cfg, off)

    arm.phase("HOVER", f"down to {cfg.hover_z*100:.0f}cm")
    straight_down(arm, cfg.hover_z, pitch_see, roll)

    # ---- STAND: the steepest angle that still reaches the object -------------------
    arm.phase("STAND", f"tilting the hand toward {grasp.pitch:.0f}deg over it")
    pitch = _stand(arm, r_goal, grasp.pitch, roll)

    # ---- TRIM by eye -----------------------------------------------------------------
    arm.phase("TRIM", "reaching by eye until it sits at the fingers' landing point")
    off = _trim(arm, eye, target, gain, r_goal, pitch, roll, cfg) or off

    # ---- TWIST, then straight DOWN ------------------------------------------------
    spot = arm.tip(arm.joints())
    if grasp.twist:
        arm.phase("TWIST", "squaring the jaws to the object")
        roll = _twist(arm, eye, target, roll, cfg)
        q = solve(arm, spot, pitch_of(arm, arm.joints()), roll)   # tips back over the spot
        if q is not None:
            arm.move(q, speed=0.9, settle=0.2)

    arm.phase("DESCEND", "straight down onto it")
    pitch = pitch_of(arm, arm.joints())
    for z in (cfg.square_z, cfg.square_low_z):
        if pitch >= grasp.pitch - 2.0:
            break
        straight_down(arm, z, pitch, roll)
        q, p = steepest(arm, (spot[0], spot[1], z), grasp.pitch, pitch, roll)
        if q is not None and p > pitch:
            arm.move(q, speed=0.85, settle=0.25)
            arm.log(f"        hand squared to {p:.0f}deg at {z*100:.1f}cm")
        pitch = pitch_of(arm, arm.joints())
    straight_down(arm, target.grasp_z, pitch, roll)

    return _grasp_and_lift(arm, eye, target, roll, cfg, off)


# ---- stages ------------------------------------------------------------------------

def _find(arm: Arm, eye: Eye, near_xy, tilts=(0.0, 25.0)):
    """Face the mapped bearing and take the detection whose cast lands nearest it.

    If nothing is in view the wrist tilts down, to see objects close to the base.
    """
    q = arm.joints()
    if near_xy is not None:
        q = with_pan(arm, q, pan_for_bearing(arm, q, bearing_of(near_xy)))
    wrist, w0 = arm.pitch_chain[-1], float(q[arm.pitch_chain[-1]])
    for tilt in tilts:
        q[wrist] = float(np.clip(w0 + tilt, arm.lo[wrist], arm.hi[wrist]))
        arm.move(q, speed=1.4, settle=0.3)
        if near_xy is None:
            d = eye.see()
            if d is not None:
                return d
            continue
        best = None
        for d in eye.all():
            if eye.label is not None and d.label != eye.label:
                continue
            xy = arm.cast((d.u, d.v), q)
            dist = 9.9 if xy is None else math.hypot(xy[0] - near_xy[0], xy[1] - near_xy[1])
            if best is None or dist < best[0]:
                best = (dist, d)
        if best is not None:
            eye.uv, eye.label = (best[1].u, best[1].v), best[1].label
            return best[1]
    return None


def _probe(arm: Arm, eye: Eye, cfg: PickConfig) -> float:
    """px of horizontal image motion per degree of base, measured with one test move."""
    before = eye.see()
    q0 = arm.joints()
    arm.move(with_pan(arm, q0, q0[arm.pan] + cfg.probe_deg), speed=1.3, settle=0.2)
    after = eye.see()
    arm.move(q0, speed=1.3, settle=0.2)
    if before is not None and after is not None and abs(after.u - before.u) >= 18.0:
        gain = (after.u - before.u) / cfg.probe_deg
        arm.log(f"        base gain {gain:+.2f}px/deg")
        return gain
    arm.log(f"        probe lost it — using {cfg.default_gain:+.1f}px/deg")
    return cfg.default_gain


def _aim(arm: Arm, eye: Eye, gain: float, cfg: PickConfig):
    """Base only, until the object is within aim_tol_px of the jaws. Returns a fresh cast."""
    for _ in range(6):
        arm.checkpoint()
        d = eye.see()
        if d is None:
            break
        ex = arm.jaw_uv[0] - d.u
        if abs(ex) < cfg.aim_tol_px:
            break
        q = arm.joints()
        arm.move(with_pan(arm, q, q[arm.pan] + _pan_step(ex, gain, 1.0, 5.0)),
                 speed=1.3, settle=0.18)
    d = eye.see()
    return None if d is None else arm.cast((d.u, d.v), arm.joints())


def _approach(arm: Arm, eye: Eye, gain, xy, pitch, roll, cfg: PickConfig):
    """Reach over the object at the look angle. Returns (radius to grasp at, px off)."""
    tip = arm.tip(arm.joints())
    if tip[2] < cfg.approach_z:                      # up first: never skim the table
        q = solve(arm, (tip[0], tip[1], cfg.approach_z), pitch, roll)
        if q is not None:
            arm.move(q, speed=1.0, settle=0.2)
    r_goal, off = (None if xy is None else math.hypot(*xy)), None
    for k in range(cfg.approach_steps):
        arm.checkpoint()
        d = eye.see()
        if d is None:
            arm.log(f"        approach {k+1}: lost it — holding here")
            break
        q = arm.joints()
        ex = (arm.jaw_uv[0] - d.u) - cfg.aim_offset_px
        off = abs(ex)
        cast = arm.cast((d.u, d.v), q)
        if cast is not None:                         # nearer looks are better: average in
            r_goal = math.hypot(*cast) if r_goal is None else 0.5 * (r_goal + math.hypot(*cast))
        tip = arm.tip(q)
        r_now = math.hypot(tip[0], tip[1])
        # no cast (no hand-eye): the approach only turns; the trim reaches by eye
        gap = 0.0 if r_goal is None else r_goal - cfg.approach_lead_m - r_now
        arm.log(f"        approach {k+1}: dx {ex:+.0f}px off, {gap*100:+.1f}cm to go")
        if abs(gap) <= cfg.approach_tol_m and abs(ex) <= cfg.centre_tol_px:
            break
        q_next = q.copy()
        bite = float(np.clip(gap, -cfg.bite_m, cfg.bite_m))
        if abs(bite) > 0.002:
            b = bearing_of(tip)
            q_s = solve(arm, ((r_now + bite) * math.cos(b), (r_now + bite) * math.sin(b),
                              max(float(tip[2]), cfg.approach_z)), pitch, roll, seed=q,
                        tol=0.03, max_jump=cfg.max_joint_jump)
            if q_s is not None:
                q_next = q_s
        if abs(ex) > cfg.centre_tol_px:
            q_next[arm.pan] = q[arm.pan] + _pan_step(ex, gain, 0.4, 2.2)
        if np.allclose(q_next, q, atol=1e-3):
            break
        arm.move(q_next, speed=1.1, settle=0.14)
    return r_goal, off


def _stand(arm: Arm, r_goal: float, pitch_want: float, roll: float) -> float:
    """Tilt to the steepest angle (down to 35 short of ``pitch_want``) that reaches r_goal."""
    q0 = arm.joints()
    tip = arm.tip(q0)
    b = bearing_of(tip)
    for pitch in np.arange(pitch_want, pitch_want - 40.0, -5.0):
        for r in (r_goal, r_goal - 0.01, r_goal - 0.02):
            q = solve(arm, (r * math.cos(b), r * math.sin(b), float(tip[2])), float(pitch),
                      roll, seed=q0, tol=0.02, max_jump=70.0)
            if q is not None:
                arm.move(q, speed=1.1, settle=0.28)
                arm.log(f"        {pitch:.0f}deg reaches {r*100:.1f}cm")
                return float(pitch)
    arm.log(f"        no angle reaches {r_goal*100:.1f}cm — staying at "
            f"{pitch_of(arm, q0):.0f}deg and trimming by eye")
    return pitch_of(arm, q0)


def _grasp_uv(arm: Arm, target: Target):
    """Where the fingertips will appear once down at the object: the tip's (x, y) at
    grasp height, projected into the image. NOT the jaw pixel: a cap lined up with the
    jaws is lined up with the sightline through them, which lands beyond the fingers."""
    q = arm.joints()
    t = arm.tip(q)
    return arm.project((t[0], t[1], target.grasp_z), q) or arm.jaw_uv


def _trim(arm: Arm, eye: Eye, target: Target, gain, r_goal, pitch, roll, cfg: PickConfig):
    """Put the object on the fingers' landing point: sideways with the base, then along
    the reach. One or the other per step, so each measures only its own effect.

    The reach is solved from a MEASURED px per metre, re-estimated every bite, so it
    has no calibration in it. Returns the remaining horizontal error in px.
    """
    tip = arm.tip(arm.joints())
    r, z = math.hypot(tip[0], tip[1]), float(tip[2])
    gone, px_per_m, prev, off = 0.0, None, None, None
    for k in range(cfg.trim_steps):
        arm.checkpoint()
        d = eye.see()
        if d is None:
            arm.log(f"        trim {k+1}: lost it — stopping here")
            break
        gu = _grasp_uv(arm, target)
        ex = (gu[0] - cfg.aim_offset_px) - d.u
        dy = (gu[1] if cfg.target_v is None else cfg.target_v) - d.v
        off = abs(ex)
        if abs(ex) <= cfg.centre_tol_px and abs(dy) <= cfg.along_tol_px:
            arm.log(f"        trim {k+1}: on target ({ex:+.0f}, {dy:+.0f}px)")
            break
        q = arm.joints()
        if abs(ex) > cfg.centre_tol_px:                   # sideways: turn the base
            arm.log(f"        trim {k+1}: dx {ex:+.0f}px — turning the base")
            arm.move(with_pan(arm, q, q[arm.pan] + _pan_step(ex, gain, 0.3, 1.8)),
                     speed=1.0, settle=0.12)
            prev = None
            continue
        if prev is not None:                              # what the last bite did
            g = (prev[0] - dy) / prev[1]
            if abs(g) < 200.0:
                arm.log(f"        trim {k+1}: reaching does not move it — stopping")
                break
            px_per_m = g if px_per_m is None else 0.5 * (px_per_m + g)
        if px_per_m is None:                              # first bite: toward the cast
            bite = math.copysign(cfg.probe_bite_m, r_goal - r)
        else:
            bite = float(np.clip(dy / px_per_m, -cfg.bite_m, cfg.bite_m))
        bite = math.copysign(min(abs(bite), cfg.trim_budget_m - gone), bite)
        if abs(bite) <= 0.001:
            arm.log(f"        trim {k+1}: reach budget used ({dy:+.0f}px along)")
            break
        b = bearing_of(arm.tip(q))
        q_r = solve(arm, ((r + bite) * math.cos(b), (r + bite) * math.sin(b), z), pitch,
                    roll, seed=q, tol=0.03, max_jump=cfg.max_joint_jump)
        if q_r is None:
            arm.log(f"        trim {k+1}: {(r+bite)*100:.1f}cm is out of reach — stopping")
            break
        arm.log(f"        trim {k+1}: dy {dy:+.0f}px — reaching {bite*100:+.1f}cm")
        arm.move(q_r, speed=1.0, settle=0.12)
        r, gone, prev = r + bite, gone + abs(bite), (dy, bite)
    return off


def _read_axis(arm: Arm, eye: Eye, target: Target, max_spread=18.0):
    """The object's long axis in the image: median of three reads, or None if they disagree."""
    vals = []
    for _ in range(3):
        d = eye.see()
        if d is not None and eye.bgr is not None:
            a = target.axis(eye.bgr, d)
            if a is not None:
                vals.append(a)
    if len(vals) < 2:
        return None
    vals = [vals[0] + fold(v - vals[0]) for v in vals]
    if max(vals) - min(vals) > max_spread:
        return None
    return float(np.median(vals)) % 180.0


def _twist(arm: Arm, eye: Eye, target: Target, roll: float, cfg: PickConfig,
           tries: int = 2) -> float:
    """Roll the wrist square to the object's axis. A roll is kept only if the object
    reads squarer afterwards; otherwise it is undone (it was a misread)."""
    square = arm.jaw_axis_deg + 90.0
    a = _read_axis(arm, eye, target)
    if a is None:
        arm.log("        twist: no clear axis — leaving the wrist")
        return roll
    for _ in range(tries):
        err = fold(a - square)
        arm.log(f"        twist: axis {a:.0f}deg, {err:+.0f}deg off square")
        if abs(err) <= cfg.twist_tol:
            break
        d_roll = _roll_within_limits(arm, -cfg.twist_fraction * err / (arm.roll_gain or 1.0))
        if cfg.twist_ambiguous is not None and abs(err) > cfg.twist_ambiguous:
            d_roll = -abs(d_roll)
        d_roll = float(np.clip(d_roll, -cfg.twist_max, cfg.twist_max))
        q0 = arm.joints()
        q = q0.copy()
        q[arm.roll] = float(np.clip(q[arm.roll] + d_roll, arm.lo[arm.roll], arm.hi[arm.roll]))
        arm.move(q, speed=1.2, settle=0.25)
        a2 = _read_axis(arm, eye, target)
        if a2 is None or abs(fold(a2 - square)) >= abs(err):
            arm.log(f"        twist: {d_roll:+.0f}deg did not square it — rolling back")
            arm.move(q0, speed=1.2, settle=0.25)
            break
        roll, a = float(q[arm.roll]), a2
        arm.log(f"        rolled {d_roll:+.0f}deg -> {abs(fold(a - square)):.0f}deg off square")
    return roll


def _roll_within_limits(arm: Arm, d: float) -> float:
    """``d`` or ``d -+ 180``, whichever is shortest and stays inside the roll joint's
    limits. A jaw line is the same every 180 degrees, so both square the grip."""
    now = float(arm.joints()[arm.roll])
    ok = [c for c in (d, d - 180.0, d + 180.0)
          if arm.lo[arm.roll] <= now + c <= arm.hi[arm.roll]]
    return min(ok, key=abs) if ok else d


def _side_grasp(arm, eye, target, gain, r_goal, roll, grasp, cfg, off):
    """Level hand: stand off, drop to grasp height, then reach in, steering by the base."""
    arm.phase("STAND", f"hand at {grasp.pitch:.0f}deg, standing off")
    tip = arm.tip(arm.joints())
    b = bearing_of(tip)
    r0 = max(r_goal - 0.07, math.hypot(tip[0], tip[1]) - 0.02)
    q = solve(arm, (r0 * math.cos(b), r0 * math.sin(b), cfg.approach_z), grasp.pitch, roll)
    if q is None:
        raise RuntimeError(f"cannot reach the stand-off at {grasp.pitch:.0f}deg")
    arm.move(q, speed=1.0, settle=0.25)
    straight_down(arm, target.grasp_z, grasp.pitch, roll)
    arm.phase("REACH", "reaching in level")
    for _ in range(cfg.approach_steps):
        arm.checkpoint()
        q = arm.joints()
        tip = arm.tip(q)
        r_now = math.hypot(tip[0], tip[1])
        gap = r_goal - r_now
        if gap <= 0.005:
            break
        bite = min(gap, cfg.bite_m)
        b = bearing_of(tip)
        q_s = solve(arm, ((r_now + bite) * math.cos(b), (r_now + bite) * math.sin(b),
                          float(tip[2])), grasp.pitch, roll, seed=q, tol=0.02,
                    max_jump=cfg.max_joint_jump)
        if q_s is None:
            break
        d = eye.see(tries=2)
        if d is not None:
            ex = (arm.jaw_uv[0] - d.u) - cfg.aim_offset_px
            off = abs(ex)
            if abs(ex) > cfg.centre_tol_px:
                q_s[arm.pan] = q[arm.pan] + _pan_step(ex, gain, 0.3, 1.8)
        arm.move(q_s, speed=0.9, settle=0.12)
    return _grasp_and_lift(arm, eye, target, roll, cfg, off)


def _grasp_and_lift(arm, eye, target, roll, cfg: PickConfig, off) -> PickResult:
    """Close, judge the grip by where the jaws stopped, lift, and check for two."""
    arm.phase("GRASP", "closing")
    arm.grip(target.open_pct)
    contact = arm.close(from_pct=target.open_pct)
    pos = arm.grip_pos()
    lv = dict(getattr(arm, "grip_levels", None) or
              {"air": 1.2, "blocked": 3.5, "jammed": 36.0, "two": 16.0})
    air, blocked, jammed, two = (v if v is not None else lv[k] for k, v in (
        ("air", cfg.grip_air), ("blocked", cfg.grip_blocked),
        ("jammed", cfg.grip_jammed), ("two", cfg.grip_two)))
    held = blocked < pos < jammed or (contact and air + 1.0 < pos < jammed)
    arm.log(f"        jaws stopped at {pos:.1f} (air {air:.1f}, held above "
            f"{blocked:.1f}) -> {'HOLDING' if held else 'EMPTY'}")
    if not held and pos >= jammed:
        raise RuntimeError(f"the jaws never closed (stopped at {pos:.1f})")
    if not held:
        raise RuntimeError(f"closed on nothing (jaws at {pos:.1f})")

    arm.phase("LIFT", "lifting clear")
    q = arm.joints()
    tip, pitch = arm.tip(q), pitch_of(arm, q)
    up = solve(arm, (tip[0], tip[1], tip[2] + cfg.lift_m), pitch, roll, tol=0.03)
    if up is not None:
        arm.move(up, speed=0.8, settle=0.2)
    if pos >= two:
        arm.log(f"        jaws at {pos:.1f}: two objects — putting them back")
        down = solve(arm, (tip[0], tip[1], tip[2] + 0.01), pitch, roll, tol=0.03)
        if down is not None:
            arm.move(down, speed=0.8, settle=0.2)
        arm.release()
        time.sleep(0.4)
        if up is not None:
            arm.move(up, speed=1.0, settle=0.2)
        raise RuntimeError(f"grabbed two (jaws at {pos:.1f})")
    return PickResult(grip_pct=pos, off_px=off, pitch=pitch)
