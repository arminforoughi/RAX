"""The gripper's own frame, measured in the picture: where the grip is, and which way it closes.

WHY THIS EXISTS. Every approach on this rig so far has steered the arm to a PLACE --
back-project a pixel, range it by apparent size, move the fingertip to a base-frame
coordinate. That works only when the hand-eye transform, the range model and the table
plane are all right at once, and the history of this rig is that they take turns being
wrong: 183px of hand-eye error, a size constant 34% off, a plane solve still 2.2x out.

There is a second way to say the same thing that needs none of them. The gripper is
bolted to the camera, so **the grip is at a fixed place in the picture**. "Is the cube
inside the jaws" is then a question about two things in one image -- no transform
between them, nothing to calibrate, nothing that can drift. Put the target in the grip's
cell and close. That is the whole idea, and it is the operator's, from the x250 branch.

WHAT IS MEASURED, AND HOW. Nothing here is derived from a URDF. On the SO-101:

  * ONE JAW MOVES; the other is fixed and sits at HAND_UV. Established by opening the
    jaws, photographing, closing them, photographing, and subtracting: exactly one lobe
    of the picture changes. The fixed finger is the one the operator already marked by
    hand with /caltip, which is why HAND_UV is the right anchor for the grip.
  * THE CAMERA IS AFTER wrist_roll IN THE CHAIN, so rolling the wrist spins the scene
    under a jaw line that does not move. Measured by rolling a known amount and watching
    a static object turn: d(image angle)/d(wrist_roll) = +0.96 and +1.01 over two
    12-degree steps. Gain +1.00, and the SIGN -- the part that cannot be reasoned out --
    is positive.

That second fact is what makes orientation cheap. Aligning the jaws to an object's face
is a rotation about the camera axis, so it is one joint, measured in one image, with no
geometry in between: ``roll_to_align`` below is four lines and a symmetry fold.

WHY ORIENTATION MATTERS MORE THAN IT LOOKS. A parallel gripper closing on a square
presents a width of ``a*(|cos t| + |sin t|)`` across the closing direction -- 5.1cm
square-on for this cube, but **7.2cm at 45 degrees**. With the wrist held at a constant
roll, as the live pick did, the yaw the cube happens to be lying at decides whether the
grasp is geometrically possible at all:

    jaw opening 6.0cm -> 74% of yaws cannot be closed on
    jaw opening 6.5cm -> 57%
    jaw opening 7.0cm -> 30%
    jaw opening 7.5cm ->  0%

Those failures are not aiming errors and no amount of positional accuracy removes them:
the jaws meet two corners and shove the cube, which is exactly the "it pushes it away"
the operator has been reporting. ``ungraspable_fraction`` is that table, so the claim
stays checkable rather than becoming folklore.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "JawFrame",
    "GripGrid",
    "fold_angle",
    "yaw_error_deg",
    "roll_to_align",
    "required_opening_m",
    "ungraspable_fraction",
]


def fold_angle(deg: float, period: float = 90.0) -> float:
    """Fold an angle into ``[-period/2, +period/2)``.

    Orientation arithmetic here is always modular and the modulus is never 360: a line
    has no head or tail (period 180), and a square looks the same every quarter turn
    (period 90). Folding through this one function is what keeps "5 degrees off" from
    being reported as "355 degrees off", which is the form that makes a controller slam
    the wrist to its limit.
    """
    p = float(period)
    return ((float(deg) + p / 2.0) % p) - p / 2.0


@dataclass(frozen=True)
class JawFrame:
    """Where the grip is in the image and which way it closes. All of it measured.

    ``fixed_uv``   the non-moving fingertip -- HAND_UV, marked by hand with /caltip.
    ``moving_uv``  the moving fingertip at the opening the grasp actually uses.
    ``axis_deg``   the CLOSING direction in the image, mod 180. The object needs a face
                   perpendicular to this, so it is the reference every yaw is measured
                   against.
    ``roll_gain``  d(image angle) / d(wrist_roll degree). Measured +1.00 on the SO-101
                   because the camera rides on the rolling link; a rig whose camera is
                   mounted BEFORE the roll joint measures 0 here, and then no amount of
                   rolling will align anything -- which this makes visible rather than
                   mysterious.
    """

    fixed_uv: tuple[float, float]
    moving_uv: tuple[float, float]
    axis_deg: float
    roll_gain: float = 1.0

    @classmethod
    def from_tips(cls, fixed_uv, moving_uv, roll_gain: float = 1.0) -> "JawFrame":
        """Build from the two fingertips, taking the closing direction as the line between.

        The jaws have to close AROUND the object, so what matters is the direction the
        gap spans -- not the arc the moving jaw's centroid travels, which on a pivoting
        jaw like this one differs from it by about 18 degrees.
        """
        fu, fv = float(fixed_uv[0]), float(fixed_uv[1])
        mu, mv = float(moving_uv[0]), float(moving_uv[1])
        axis = math.degrees(math.atan2(fv - mv, fu - mu)) % 180.0
        return cls((fu, fv), (mu, mv), axis, float(roll_gain))

    @property
    def centre_uv(self) -> tuple[float, float]:
        """The middle of the grip -- the pixel a target should be driven onto."""
        return ((self.fixed_uv[0] + self.moving_uv[0]) / 2.0,
                (self.fixed_uv[1] + self.moving_uv[1]) / 2.0)

    @property
    def opening_px(self) -> float:
        """How far apart the fingertips are, in pixels, right now."""
        return math.hypot(self.fixed_uv[0] - self.moving_uv[0],
                          self.fixed_uv[1] - self.moving_uv[1])

    def opening_m(self, px_per_m: float) -> float:
        """The opening in metres, given the scale at the object's range.

        Worth computing and logging: it is the number that decides which yaws are
        graspable at all, and until it was measured the failure it causes was being
        read as an aiming problem.
        """
        return self.opening_px / float(px_per_m) if px_per_m else float("nan")


def yaw_error_deg(object_angle_deg: float, jaw: JawFrame,
                  symmetry_deg: float = 90.0) -> float:
    """How far the object's face is from square to the jaws, folded by its symmetry.

    ``object_angle_deg`` is the object's silhouette angle in the image -- a minAreaRect
    angle, in the same picture as the jaws, which is the point: two measurements in one
    frame need no transform between them.

    The condition looks like it should involve a right angle and does not. A square's
    edges lie at both ``t`` and ``t+90``, and the jaws want an edge perpendicular to
    their closing direction, i.e. at ``axis+90``. Requiring ``t == axis+90 (mod 90)`` is
    the same as requiring ``t == axis (mod 90)``, so the +90 cancels. Spelled out
    because the version of this that carried a hand-written +90 around was marked
    UNVERIFIED for good reason.
    """
    return fold_angle(float(object_angle_deg) - jaw.axis_deg, symmetry_deg)


def roll_to_align(object_angle_deg: float, jaw: JawFrame, current_roll_deg: float,
                  *, limits: tuple[float, float], symmetry_deg: float = 90.0,
                  deadband_deg: float = 3.0) -> tuple[float, float]:
    """Wrist roll that squares the jaws to the object. Returns ``(roll_deg, error_deg)``.

    The scene turns with the wrist at ``roll_gain`` degrees per degree, and the jaw line
    does not turn at all, so nulling the error is a single division. What takes the rest
    of the function is the wrist's limited travel: an object's symmetry means several
    rolls are the SAME grasp, so when the direct answer is out of reach a symmetric one
    usually is not. Clamping instead -- which is what happens if nobody thinks about it
    -- saturates at the limit for most yaws and quietly throws the orientation away,
    while still reporting success.
    """
    err = yaw_error_deg(object_angle_deg, jaw, symmetry_deg)
    if abs(err) <= float(deadband_deg) or not jaw.roll_gain:
        return float(current_roll_deg), err
    want = float(current_roll_deg) - err / float(jaw.roll_gain)
    lo, hi = (float(limits[0]), float(limits[1]))
    cands = [want + k * float(symmetry_deg) for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4)]
    reachable = [w for w in cands if lo <= w <= hi]
    if not reachable:
        return float(min(max(want, lo), hi)), err
    # Of the equivalent grasps, take the smallest wrist move: the jaws are already near
    # the object when this runs, and a needless 90-degree slew is both slow and a chance
    # to knock it over.
    return float(min(reachable, key=lambda w: abs(w - float(current_roll_deg)))), err


def required_opening_m(edge_m: float, yaw_deg: float) -> float:
    """Width a square of side ``edge_m`` presents across the jaws at this yaw.

    ``a*(|cos t| + |sin t|)``: the square's extent along the closing direction. Equal to
    the edge when square-on, and sqrt(2) times it -- the diagonal -- at 45 degrees.
    """
    t = math.radians(float(yaw_deg))
    return float(edge_m) * (abs(math.cos(t)) + abs(math.sin(t)))


def ungraspable_fraction(edge_m: float, opening_m: float) -> float:
    """Fraction of yaws a FIXED wrist cannot close on. The cost of not aligning.

    Integrated over the quarter turn the square is symmetric under, so it reads as the
    share of randomly-oriented presentations that fail for geometric reasons alone --
    the number that says whether wrist alignment is worth building.
    """
    n = 900
    bad = sum(1 for i in range(n)
              if required_opening_m(edge_m, 90.0 * i / n) > float(opening_m))
    return bad / float(n)


@dataclass(frozen=True)
class GripGrid:
    """The view cut into cells, with the grip's own cell marked.

    A grid is a coarse, honest way to say where something is relative to the jaws, and
    coarse is a virtue here: the question a grasp actually asks is "is the cube in the
    grip or not", and answering it in cells cannot pretend to a precision the detector
    does not have. It is also the one representation of this that a person can check at
    a glance on the FPV, which matters -- the misses that took longest to find on this
    rig were the ones where the arm's account of itself and the picture disagreed and
    nothing drew both.

    The cells are for display and for the go/no-go gate. STEERING uses the continuous
    pixel error, because quantising the error to a cell width would stall the servo as
    soon as the target shared a cell with the grip.
    """

    width: int
    height: int
    cols: int = 6
    rows: int = 5

    def cell_of(self, uv) -> tuple[int, int]:
        """Which (col, row) a pixel falls in, clamped to the grid."""
        c = int(float(uv[0]) * self.cols / self.width)
        r = int(float(uv[1]) * self.rows / self.height)
        return (max(0, min(self.cols - 1, c)), max(0, min(self.rows - 1, r)))

    def cell_box(self, col: int, row: int) -> tuple[int, int, int, int]:
        """Pixel bounds ``(x1, y1, x2, y2)`` of one cell."""
        return (int(col * self.width / self.cols), int(row * self.height / self.rows),
                int((col + 1) * self.width / self.cols),
                int((row + 1) * self.height / self.rows))

    def grip_cells(self, jaw: JawFrame) -> set[tuple[int, int]]:
        """Every cell the grip spans -- both fingertips and the gap between them.

        The span, not just the midpoint: the jaws are a couple of hundred pixels apart
        at this range, which is several cells, and an object anywhere along that line is
        between the fingers. Treating only the centre cell as the target would reject
        grasps that are already good.
        """
        (fu, fv), (mu, mv) = jaw.fixed_uv, jaw.moving_uv
        out = set()
        for i in range(21):
            t = i / 20.0
            out.add(self.cell_of((fu + (mu - fu) * t, fv + (mv - fv) * t)))
        return out

    def in_grip(self, uv, jaw: JawFrame) -> bool:
        """Is this pixel in one of the grip's cells?"""
        return self.cell_of(uv) in self.grip_cells(jaw)
