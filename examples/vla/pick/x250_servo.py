"""The X250 as a ``ServoArm``, so it can use the same visual servo the SO-101 uses.

WHAT THIS BUYS. `pick.py`'s approach is hand-written for this arm: it interpolates the
demonstrated descent, probes the base direction with a +7/-10/+14 nudge to recover the
sign of the correction, and corrects horizontally with a gain worked out on the spot.
`rax.manipulation.approach.visual_servo` already does all of that for the SO-101,
measures the gains instead of assuming them, and its safety property -- a tick with no
detection can never produce motion -- is pinned by tests. None of it is SO-101 specific.

The seam it asks for is four members, and deliberately excludes everything that needs a
calibration: no kinematics, no intrinsics, no hand-eye, no table plane. That matters
here more than it does on the SO-101, because the X250 has no URDF in this repo at all
and therefore cannot do forward kinematics even in principle. An arm that can report its
actuators, move them, and find the target in a picture is enough.

WHICH AXES, AND WHY NOT ALL OF THEM. Only the three that change where the tube appears:
the base swings the view sideways, and shoulder and elbow together carry the camera in
and down. `wrist` and `tool` aim the camera without moving it usefully, and the
demonstrations already fix them; `gripper` is not an approach axis at all. Handing the
servo axes that do not help is not free -- each one is a Jacobian column it has to
measure by moving the arm, and a near-singular column makes the solve worse, not better.

UNITS ARE NORMALISED THROUGHOUT, the same -100..100 the poses and the envelope use, so
`probe`, `max_step` and the limits all read in the units an operator already knows from
safe_envelope.json.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from rax.manipulation.approach.servo_arm import Axis, Sighting

logger = logging.getLogger(__name__)

__all__ = ["X250ServoArm", "TUBE_AXES"]


#: The approach axes, in the order the Jacobian's columns use.
#:
#: `probe` has to move the picture well clear of detector noise while the local linear
#: model still holds. pick.py measured the equivalent for its own controller and found
#: the base needed a nudge of 7 to 14 units before the cap moved a reliable 18px, so
#: these start there. `max_step` is bounded by the driver's own max_relative_target of
#: 8.0 -- asking for more would be silently clipped, which would make the servo's model
#: of what it just did wrong, and a servo that mispredicts its own move learns nonsense.
TUBE_AXES = (
    Axis(name="base", index=0, probe=7.0, max_step=5.0, lo=-40.6, hi=11.7),
    Axis(name="shoulder_2", index=1, probe=5.0, max_step=5.0, lo=-100.0, hi=37.0),
    Axis(name="elbow", index=2, probe=5.0, max_step=5.0, lo=-100.0, hi=41.4),
)


class X250ServoArm:
    """Adapts `X250Follower` plus a cap detector to the servo's four-member seam.

    Holds the joints the servo does not drive at whatever the caller left them: the
    actuator vector it exposes is ONLY the approach axes, and everything else is carried
    through unchanged from the pose at construction. That keeps the demonstrated wrist
    and tool angles, which are what make the gripper arrive at the right attitude.
    """

    def __init__(self, robot, colour: str, *, axes=TUBE_AXES, envelope=None,
                 settle_s: float = 0.35, exclude=(), exclude_r: float = 0.0,
                 frame_wh=(640, 480)):
        self.robot = robot
        self.colour = colour
        self.axes = tuple(axes)
        self.envelope = envelope or {}
        self.settle_s = float(settle_s)
        self.exclude = list(exclude)
        self.exclude_r = float(exclude_r)
        self.frame_w, self.frame_h = frame_wh
        self._held = {m: v for m, v in self._pose().items()
                      if m not in {a.name for a in self.axes}}

    # ---- helpers ---------------------------------------------------------------
    def _pose(self) -> dict:
        o = self.robot.get_observation()
        return {k[:-4]: float(v) for k, v in o.items() if k.endswith(".pos")}

    def _clamp(self, pose: dict) -> dict:
        out = {}
        for m, v in pose.items():
            lo, hi = self.envelope.get(m, (-1e9, 1e9))
            out[m] = float(min(max(v, lo), hi))
        return out

    # ---- the ServoArm seam -----------------------------------------------------
    def actuators(self) -> np.ndarray:
        p = self._pose()
        return np.array([p[a.name] for a in self.axes], dtype=np.float64)

    def apply(self, q) -> bool:
        """Command the approach axes and BLOCK until the arm has settled.

        Blocking is not a nicety. Measured on the SO-101, a non-blocking version let the
        loop read the next frame ~50ms later, compute the same correction from the same
        unchanged picture, and overwrite a command the arm had not begun to execute --
        60 ticks in 3 seconds all issuing the same correction against a frozen error.
        The X250 is slower than that, not faster, so it waits for arrival and gives up
        rather than pretending.
        """
        want = dict(self._held)
        want.update({a.name: float(v) for a, v in zip(self.axes, np.asarray(q).ravel())})
        want = self._clamp(want)
        target = {f"{m}.pos": v for m, v in want.items()}

        deadline = time.time() + 4.0
        while time.time() < deadline:
            self.robot.send_action(target)
            time.sleep(self.settle_s)
            now = self._pose()
            worst = max(abs(want[a.name] - now[a.name]) for a in self.axes)
            if worst <= 1.5:
                return True
            # The driver clips every command to max_relative_target, so a large move
            # takes several sends; only a move that stops PROGRESSING is a failure.
            still = max(abs(want[a.name] - now[a.name]) for a in self.axes)
            if abs(still - worst) < 1e-9 and time.time() > deadline - 1.2:
                break
        now = self._pose()
        worst = max(abs(want[a.name] - now[a.name]) for a in self.axes)
        if worst > 6.0:
            logger.warning("apply: arm stalled %.1f units short on the worst axis", worst)
            return False
        return True

    def sense(self) -> Sighting | None:
        """The chosen cap in the current wrist frame, as pixels. No range, no 3D."""
        import cv2

        from caps2 import find_caps

        o = self.robot.get_observation()
        if "wrist" not in o:
            return None
        frame = cv2.cvtColor(o["wrist"], cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]
        hits = [c for c in find_caps(frame, restrict_to_mat=True) if c.colour == self.colour]
        # The jaws wear blue tape and read as a blue cap; pick.py lost whole runs to
        # this. Anything within exclude_r of a fingertip is the gripper, not a tube.
        if self.exclude and self.exclude_r > 0:
            hits = [c for c in hits
                    if all((c.x - fx) ** 2 + (c.y - fy) ** 2 > self.exclude_r ** 2
                           for fx, fy in self.exclude)]
        if not hits:
            return None
        c = max(hits, key=lambda h: h.area)
        # find_caps reports a centroid and an area, not a box, so the box is
        # reconstructed from the blob's own extent -- the servo reads HEIGHT as its
        # size feature, and a cap's height is what shrinks with range.
        half_w = max(c.w, 4.0) / 2.0
        half_h = max(c.h, 4.0) / 2.0
        bbox = (c.x - half_w, c.y - half_h, c.x + half_w, c.y + half_h)
        return Sighting(bbox=bbox, frame_w=w, frame_h=h,
                        clipped_bottom=bool(c.y + half_h >= h - 2))
