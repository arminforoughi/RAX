"""Did the jaws actually get it? One answer, from two different sensors.

THE PROBLEM THIS SOLVES TWICE OVER. Both arms on this rig decide "am I holding
something" from the gripper alone, and both were doing it in their own file with their
own constants. They do not use the same signal:

    SO-101   the gripper's CURRENT rises while closing. Fingers on an object means
             resistance, so dI over idle crossing `contact_current_delta` is contact.
    X250     the gripper's settled POSITION. The jaws are back-drivable enough that
             where they come to rest says whether anything stopped them: measured over
             113 demonstrations, holding reads above 31.9 and closing on air settles at
             30.2, WITH NO OVERLAP between the two populations.

Different sensors, but the same shape of answer and — this is the part worth sharing —
the same blind spot. Neither can tell you WHAT the jaws met. Resistance reads identically
for the object, for a fingertip fouled on its corner, and for the table edge; a position
short of empty reads identically for a tube and for a jaw jammed on the rack. So both
arms need the same asymmetric override, and both had to learn it the same expensive way.

THE OVERRIDE IS ONE-WAY, AND THAT IS THE LOAD-BEARING DESIGN DECISION HERE.

A camera looking at the jaws CAN tell what is between them. When the mechanical signal
says HELD and the camera says EMPTY, the camera wins: the carry flag clears and the arm
refuses to place. Before that, the arm would traverse to the destination and solemnly
open its empty jaws — observed repeatedly on this rig, and it is worse than a plain miss
because the task then reports success.

When the mechanical signal says EMPTY and the camera says HOLDING, the camera does NOT
win. It stays advisory, and is logged as "the contact threshold may be too high". Letting
it win would have a remote vision model authorise a carry on nothing but its own say-so,
and the cost of the two errors is not symmetric: a discarded good pick costs one retry,
while a carry authorised on a hallucination drives the arm through a place with nothing
in it.

`GRASP_EMPTY_TRUST` exists for the same reason — only a CONFIDENT empty clears a carry.
An unsure model is not evidence.

THE SETTLE IS NOT OPTIONAL, on either arm. A gripper read mid-motion reads wherever it
happens to be passing, which is above the empty threshold on the way down — i.e. it reads
as a successful grasp precisely while the grasp is still happening. `settled` waits for
the reading to stop changing before believing it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal, Protocol

__all__ = ["GripReading", "GripSensor", "PositionThreshold", "CurrentRise",
           "CameraVerdict", "Decision", "settled", "reconcile",
           "GRASP_EMPTY_TRUST"]

#: How sure a camera has to be that the jaws are empty before its word overrides a
#: mechanical "held" and clears the carry. Below this the disagreement is only logged.
GRASP_EMPTY_TRUST = 0.70


@dataclass(frozen=True)
class GripReading:
    """What the gripper's own sensor says, and how clearly."""

    held: bool
    value: float
    kind: Literal["position", "current_rise"]
    #: Signed distance past the deciding threshold, in the sensor's own units.
    #: Positive means "on the held side". NEAR ZERO IS THE INTERESTING CASE: pick.py
    #: logged a real pick that held a tube by its edge at 31.98 against a 31.9 line,
    #: which is a hold nobody should have much confidence in.
    margin: float
    detail: str = ""

    @property
    def marginal(self) -> bool:
        """True when the reading is close enough to the threshold to deserve a second
        opinion from a camera before anything is carried anywhere."""
        return abs(self.margin) < self.marginal_band

    #: What counts as "close to the line", per sensor kind. The position sensor's two
    #: populations are 1.7 units apart, so a third of that is a sensible caution band;
    #: the current sensor's contact delta is 8 counts over idle.
    @property
    def marginal_band(self) -> float:
        return 0.6 if self.kind == "position" else 2.5


class GripSensor(Protocol):
    """Turns one raw gripper reading into a verdict."""

    def verdict(self, value: float) -> GripReading: ...


@dataclass(frozen=True)
class PositionThreshold:
    """The X250's test: where the jaws came to REST.

    `holding` and `empty` are two measured populations, not a threshold and a tolerance.
    Anything at or below `empty` closed on air; anything above `holding` has something
    between the jaws. The gap between them is the whole confidence budget, so neither
    number may be rounded — see the profile's comment.

    THIS VERDICT IS ONLY MEANINGFUL AFTER THE GRIPPER HAS BEEN COMMANDED SHUT, and that
    is not a footnote — it is a trap this sensor walked straight into. "Above `holding`"
    means "something stopped the jaws closing" ONLY if they were trying to close. An OPEN
    gripper also reads above `holding`, and by a mile: on the X250, shut is 30.2 and the
    place-open position is 50.0. Read at the wrong moment, the sensor cheerfully reports a
    firm grasp of the empty air it has just released a tube into — observed, in the first
    end-to-end place this server ran, which reported DONE for entirely the wrong reason.

    `open_above` closes the trap. A reading at or past it is not scored as a grasp at all;
    it is reported as "you are reading an open gripper", which is a question this sensor
    cannot answer rather than an answer it gets wrong. It sits well clear of a real hold: a
    16 mm tube stalls the jaws in the low 30s (pick.py logged 31.98), while every OPEN
    position on this arm is 39 or above.
    """

    holding: float
    empty: float
    #: Reading at or above which the jaws are plainly open, so no grasp verdict is given.
    #: None disables the guard — only correct if the caller can guarantee it never asks
    #: except straight after a close.
    open_above: float | None = None

    def verdict(self, value: float) -> GripReading:
        v = float(value)
        if self.open_above is not None and v >= self.open_above:
            return GripReading(
                held=False, value=v, kind="position",
                # Margin is measured from the OPEN boundary here, not the holding one:
                # "how far open" is the only thing this reading actually tells us.
                margin=self.open_above - v,
                detail=f"gripper reads {v:.2f}, at or past the open position "
                       f"({self.open_above:.1f}) — the jaws are open, so this is not a "
                       f"grasp verdict. Ask again after commanding them shut.")
        return GripReading(
            held=v > self.holding, value=v, kind="position",
            margin=v - self.holding,
            detail=f"gripper settled at {v:.2f} (holding > {self.holding:.1f}, "
                   f"air reads {self.empty:.1f})")


@dataclass(frozen=True)
class CurrentRise:
    """The SO-101's test: how much the gripper's CURRENT rose over its idle draw.

    `idle` has to be measured on the spot each close — it drifts with temperature and
    with whatever the arm is already holding — which is why it is a constructor argument
    rather than a constant.
    """

    idle: float
    delta: float

    def verdict(self, value: float) -> GripReading:
        rise = abs(float(value) - self.idle)
        return GripReading(
            held=rise >= self.delta, value=float(value), kind="current_rise",
            margin=rise - self.delta,
            detail=f"gripper current {value:.1f} vs idle {self.idle:.1f} "
                   f"(rise {rise:.1f}, contact at {self.delta:.1f})")


@dataclass(frozen=True)
class CameraVerdict:
    """A vision model's read of the jaws. ``ok`` False means it could not look."""

    ok: bool
    answer: Literal["holding", "empty", "unsure"] = "unsure"
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class Decision:
    """The reconciled answer, plus why — the 'why' is what gets logged and argued with."""

    held: bool
    #: True when the camera overturned the mechanical verdict.
    overridden: bool
    #: True when the two sources disagreed but the disagreement was left advisory.
    advisory_disagreement: bool
    detail: str


def settled(read: Callable[[], float], *, tol: float = 0.05, timeout: float = 1.5,
            dt: float = 0.05) -> float:
    """Read the gripper only once it has STOPPED MOVING, and return that reading.

    Mid-motion the jaws are passing through every value between open and shut, so a
    single read taken too early reports a grasp that has not happened yet. pick.py's
    version of this is the reason its 30.2 "closed on air" figure is reproducible at
    all: the value is only reached once motion has finished.

    Returns the last reading when the timeout expires — a gripper still creeping after
    `timeout` is a real condition (something is slowly yielding), and the caller's
    threshold test is still the right thing to apply to it.
    """
    last = float(read())
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        time.sleep(dt)
        v = float(read())
        if abs(v - last) <= tol:
            return v
        last = v
    return last


def reconcile(mech: GripReading, camera: CameraVerdict | None = None, *,
              trust_empty_at: float = GRASP_EMPTY_TRUST) -> Decision:
    """Combine the gripper's verdict with a camera's, one way only.

    See the module docstring for why the override is asymmetric. In short: a confident
    camera "empty" beats a mechanical "held", and nothing beats a mechanical "empty".
    """
    if camera is None or not camera.ok:
        why = "no camera check" if camera is None else f"camera unavailable ({camera.reason})"
        return Decision(held=mech.held, overridden=False, advisory_disagreement=False,
                        detail=f"{mech.detail}; {why}")

    if camera.answer == "unsure":
        return Decision(held=mech.held, overridden=False, advisory_disagreement=False,
                        detail=f"{mech.detail}; camera could not tell ({camera.reason})")

    camera_says_held = camera.answer == "holding"
    if camera_says_held == mech.held:
        return Decision(held=mech.held, overridden=False, advisory_disagreement=False,
                        detail=f"{mech.detail}; camera agrees ({camera.answer}, "
                               f"{camera.confidence:.0%})")

    # They disagree. Only one direction acts.
    if mech.held and not camera_says_held and camera.confidence >= trust_empty_at:
        return Decision(
            held=False, overridden=True, advisory_disagreement=False,
            detail=f"{mech.detail}; CAMERA OVERRULES: jaws empty at "
                   f"{camera.confidence:.0%} — {camera.reason}. A place from here would "
                   f"put nothing down.")

    if mech.held and not camera_says_held:
        return Decision(
            held=True, overridden=False, advisory_disagreement=True,
            detail=f"{mech.detail}; camera says empty but only at "
                   f"{camera.confidence:.0%} (needs {trust_empty_at:.0%}) — not acting")

    # Mechanically empty, camera says holding. Advisory, deliberately: see the docstring.
    return Decision(
        held=False, overridden=False, advisory_disagreement=True,
        detail=f"{mech.detail}; camera says HOLDING at {camera.confidence:.0%} — left "
               f"advisory, but the contact threshold may be set too high")
