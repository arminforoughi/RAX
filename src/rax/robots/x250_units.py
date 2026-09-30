"""Normalised units <-> degrees for the X250. The ONE place this conversion lives.

WHY THE CONVERSION EXISTS AT ALL. The X250's entire control path speaks lerobot's
NORMALISED units: `poses.json`, `safe_envelope.json`, `place_poses.json`, the driver's
`send_action`, and `x250_servo.TUBE_AXES` are all in -100..100 (0..100 for the gripper).
Nothing in that path needs degrees, and it works. Kinematics, on the other hand, speaks
degrees, because a URDF does.

So this module is a BRIDGE FOR THE VIEW, not a change of units for the robot. The 3D
page needs joint angles to pose links; the arm keeps being commanded in the units its
calibration is expressed in. Converting the control path instead would mean recomputing
113 demonstrations' worth of poses through a calibration-dependent transform, to gain
nothing -- the same class of unforced error as "tidying" a measured constant.

The SO-101 already has a display-only correction of exactly this kind
(`WRIST_RENDER_OFFSET` / `GripperProfile.render_offset_deg`), for the same reason: the
servo's zero and the URDF's zero are different conventions, and the fix belongs in the
renderer.

THE TWO HALVES OF THE CONVERSION ARE NOT EQUALLY KNOWN, and conflating them would be
the mistake here:

  SCALE (`deg_per_norm`) IS MEASURED. Normalised -100..100 spans the joint's calibrated
  tick range, so degrees per unit follows from the calibration file and 360deg/4096
  ticks and nothing else. Computed from this arm's own calibration:

      base        4095 ticks   1.79956 deg/unit    (full turn -- see below)
      shoulder_2  2181 ticks   0.95845 deg/unit
      elbow       1226 ticks   0.53877 deg/unit
      wrist       4095 ticks   1.79956 deg/unit    (full turn -- see below)
      tool        2461 ticks   1.08149 deg/unit
      gripper     1679 ticks   1.47568 deg/unit    (0..100, one-sided)

  `base` and `wrist` read range_min=0, range_max=4095. That is the Dynamixel power-on
  default, not a measurement -- those two joints were never ranged, so their scale is
  "as if a full turn" and is the least trustworthy number here. It is also the least
  harmful: it only stretches how far those two appear to rotate.

  OFFSET (`zero_norm`) IS NOT MEASURED, and cannot be from here. It is the normalised
  reading at which a joint's URDF angle is zero, and pinning it needs the physical arm
  in a known pose -- the arm is on COM5, which does not currently enumerate. Every
  entry in ZERO_NORM below is therefore a STATED ASSUMPTION, and the module says so
  rather than burying it: the assumption is that normalised 0 (the centre of each
  joint's calibrated travel) is the URDF zero pose.

WHAT FOLLOWS FROM THAT SPLIT, and it is the useful part: RELATIVE motion in the 3D view
is as correct as the scale, i.e. correct. Absolute limb attitude is only as correct as
the offsets. So the view faithfully shows the arm folding, swinging and reaching -- an
approach looks like the approach it is -- while the whole arm may sit at a constant
attitude error. Watch a pick with it; do not read a reach off it.

TO PIN THE OFFSETS, once the arm is on the bus: fold it to a pose you can describe
physically (upper arm vertical, forearm horizontal is the easiest to eyeball on this
arm), read `get_observation()`, and write those readings in as ZERO_NORM. That is the
same procedure that fixed the SO-101's wrist_flex zero after its motor was swapped --
fit the zero to a physical reference, never to whatever the joint happens to be
resting at. See [[rax-servo-replacement-calibration]].
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["DEG_PER_TICK", "ZERO_NORM", "X250Units", "load_units"]

#: A Dynamixel turn is 4096 ticks.
DEG_PER_TICK = 360.0 / 4096.0

#: Where lerobot keeps this arm's calibration. Same path the driver reads.
CALIB = (Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" /
         "robots" / "x250_follower" / "x250_follower.json")

#: Joints whose normalised range is 0..100 rather than -100..100.
RANGE_0_100 = frozenset({"gripper"})

#: The normalised reading at which each joint's URDF angle is zero.
#:
#: FITTED TO THE DEMONSTRATED GRASP POSE, not assumed and not measured on the arm. The
#: distinction matters, so here is exactly what was done and what it is worth.
#:
#: All six started at zero -- "the centre of each joint's travel is the URDF zero" --
#: and that was visibly wrong: it rendered the demonstrated `look` pose with the hand
#: 16 cm BEHIND the base and 64 cm up, which is not a pose anybody looked at a bench
#: from. Rather than leave a view nobody could read, the three pitch offsets were fitted
#: against a physical fact the demonstrations really do establish:
#:
#:     at the `grasp` pose, the fingertip is on the bench in front of the arm, at a
#:     tube's cap height, with the hand pointing straight down.
#:
#: That is three constraints (reach 0.27 m, height 0.085 m, hand pitch -90deg) for three
#: unknowns, and `fit_zero_norm` below solves it exactly. The other two are pinned by
#: the same kind of statement: `base` is zero where the `look` pose aims, because that
#: is the neutral survey heading, and `tool` is zero where the `grasp` pose holds it,
#: because that is where the jaws are square to a tube.
#:
#: WHAT IT IS WORTH. The view is now self-consistent AT THE GRASP, and degrades smoothly
#: away from it: since the fit absorbs whatever the nominal link lengths get wrong, error
#: reappears as you move away from the pose it was fitted at. The independent check that
#: it is not nonsense is the `look` pose, which was NOT fitted and which comes out raised
#: ~0.50 m and barely reaching out (r = 0.06 m) -- exactly what pick.py's own README
#: calls it: "the retracted `look` pose (raised, not reached out)".
#:
#: This is the same procedure that fixed the SO-101's wrist_flex zero after its motor was
#: swapped: fit the zero to a physical reference you can state, never to wherever the
#: joint happens to be resting. See [[rax-servo-replacement-calibration]].
#:
#: RE-RUN `fit_zero_norm()` after correcting any link length in the URDF -- the offsets
#: and the lengths are coupled, so a better ruler makes these stale.
ZERO_NORM = {
    "base": -2.6,          # the `look` pose's heading = straight ahead
    "shoulder_2": 10.05,   # fitted
    "elbow": 0.46,         # fitted
    "wrist": -47.74,       # fitted
    "tool": 38.923,        # the `grasp` pose's tool angle = jaws square to the tube
    "gripper": 0.0,        # off the FK chain; the viewer uses the opening fraction
}

#: Fallback tick ranges, used only when the calibration file is absent (so the 3D view
#: still draws something on a machine that has never had this arm plugged in). These
#: are this rig's measured ranges, copied here as a default rather than a guess about
#: some other arm -- a different X250 will have different ones and should be read from
#: its own calibration.
FALLBACK_TICKS = {
    "base": (0, 4095),
    "shoulder_2": (825, 3006),
    "elbow": (1243, 2469),
    "wrist": (0, 4095),
    "tool": (748, 3209),
    "gripper": (1002, 2681),
}

#: Joints read straight off the default 0..4095 -- i.e. never ranged. Flagged so a
#: caller can warn instead of quietly trusting them.
UNRANGED_SPAN = 4095


class X250Units:
    """Converts this arm's normalised joint units to degrees, and back.

    Construct via :func:`load_units` so the calibration is read once.
    """

    def __init__(self, ticks: dict[str, tuple[float, float]],
                 zero_norm: dict[str, float] | None = None):
        self.ticks = {k: (float(a), float(b)) for k, (a, b) in ticks.items()}
        self.zero_norm = dict(ZERO_NORM if zero_norm is None else zero_norm)

    # ---- the measured half ------------------------------------------------------
    def deg_per_norm(self, joint: str) -> float:
        """Degrees of joint rotation per normalised unit. Derived from calibration."""
        lo, hi = self.ticks[joint]
        span = 100.0 if joint in RANGE_0_100 else 200.0
        return (hi - lo) / span * DEG_PER_TICK

    def unranged(self, joint: str) -> bool:
        """True when this joint's 'calibration' is the Dynamixel default, i.e. absent.

        `base` and `wrist` are both like this on this arm. Their scale is a placeholder
        for a full turn, so a caller that cares about absolute angle should say so
        rather than trusting the number.
        """
        lo, hi = self.ticks[joint]
        return lo == 0.0 and hi == float(UNRANGED_SPAN)

    # ---- the conversion ---------------------------------------------------------
    def to_deg(self, joint: str, norm: float) -> float:
        """Normalised units -> degrees about the URDF axis."""
        return (float(norm) - self.zero_norm.get(joint, 0.0)) * self.deg_per_norm(joint)

    def to_norm(self, joint: str, deg: float) -> float:
        """Degrees about the URDF axis -> normalised units."""
        dpn = self.deg_per_norm(joint)
        if abs(dpn) < 1e-12:
            return self.zero_norm.get(joint, 0.0)
        return float(deg) / dpn + self.zero_norm.get(joint, 0.0)

    def pose_to_deg(self, pose: dict[str, float], joints) -> list[float]:
        """A normalised pose dict -> a degree vector in ``joints`` order.

        Missing joints read as zero rather than raising: the viewer must keep drawing
        when an observation arrives short a key, and a link at its zero angle is a
        visibly wrong pose rather than a blank page.
        """
        return [self.to_deg(j, pose.get(j, 0.0)) for j in joints]

    def describe(self) -> str:
        rows = []
        for j in self.ticks:
            lo, hi = self.ticks[j]
            flag = "  UNRANGED (Dynamixel default)" if self.unranged(j) else ""
            rows.append(f"{j:<11} {int(hi - lo):5d} ticks  {self.deg_per_norm(j):8.5f} "
                        f"deg/unit  zero at {self.zero_norm.get(j, 0.0):+6.1f}{flag}")
        return "\n".join(rows)


#: The pose the offsets are fitted at, and the physical claim made about it. Separated
#: out so the claim is inspectable and so a rig with a different bench can restate it.
GRASP_NORM = {"base": -4.371, "shoulder_2": 29.390, "elbow": 30.832,
              "wrist": -17.118, "tool": 38.923}
#: reach (m), fingertip height (m), hand pitch (deg, negative = pointing down)
GRASP_FACT = (0.27, 0.085, -90.0)


def fit_zero_norm(urdf_path: str | Path, *, grasp=None, fact=None,
                  joints=("base", "shoulder_2", "elbow", "wrist", "tool")) -> dict:
    """Solve for the pitch-joint zero offsets that put the grasp pose where it belongs.

    Returns a full ZERO_NORM dict. See :data:`ZERO_NORM` for why this is the honest way
    to pin the offsets without the arm on the bus, and for what the result is worth.

    RE-RUN THIS after changing a link length in the URDF: the offsets absorb the link
    lengths' error, so the two are coupled and a corrected ruler makes the old fit stale.
    """
    import numpy as np
    from scipy.optimize import least_squares

    from rax.kinematics.urdf import UrdfKinematics

    grasp = dict(grasp or GRASP_NORM)
    want_r, want_z, want_pitch = fact or GRASP_FACT
    units = load_units()
    kin = UrdfKinematics(str(urdf_path), ee_frame="gripper_frame_link",
                         joint_names=list(joints))

    def degrees(z_pitch):
        z = {"base": ZERO_NORM["base"], "shoulder_2": z_pitch[0], "elbow": z_pitch[1],
             "wrist": z_pitch[2], "tool": grasp["tool"]}
        return np.array([(grasp[j] - z[j]) * units.deg_per_norm(j) for j in joints])

    def resid(z_pitch):
        d = degrees(z_pitch)
        p = np.asarray(kin.forward_kinematics(d))[:3, 3]
        # Scaled so a centimetre of reach and a degree of attitude trade sensibly; the
        # system is exactly determined, so the scaling only shapes the path to the root.
        return [(p[0] - want_r) * 10.0, (p[2] - want_z) * 10.0,
                (-(d[1] + d[2] + d[3]) - want_pitch) / 50.0]

    sol = least_squares(resid, [0.0, 0.0, 0.0])
    return {"base": ZERO_NORM["base"],
            "shoulder_2": round(float(sol.x[0]), 2),
            "elbow": round(float(sol.x[1]), 2),
            "wrist": round(float(sol.x[2]), 2),
            "tool": grasp["tool"], "gripper": 0.0}


def load_units(path: Path | str | None = None,
               zero_norm: dict[str, float] | None = None) -> X250Units:
    """Read the arm's calibration and return the converter.

    Falls back to :data:`FALLBACK_TICKS` if the file is missing, and says so in the log
    rather than silently — a view drawn from fallback numbers on somebody else's X250
    would be wrong in a way nobody could see.
    """
    p = Path(path) if path is not None else CALIB
    try:
        raw = json.loads(p.read_text())
        ticks = {k: (v["range_min"], v["range_max"]) for k, v in raw.items()}
    except Exception as e:
        logger.warning("x250: no calibration at %s (%s: %s); the 3D view is using this "
                       "rig's recorded tick ranges, which are wrong for any other X250",
                       p, type(e).__name__, e)
        ticks = dict(FALLBACK_TICKS)
    return X250Units(ticks, zero_norm)
