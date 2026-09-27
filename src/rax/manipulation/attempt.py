"""Try, check, try again — the retry loop both arms need and both had their own copy of.

WHY A MISS IS NOT A REASON TO STOP. By the time a pick fails, the arm has already proved
it can SEE the object: it found it, aimed at it, and closed on it. The commonest failure
on both arms is a couple of centimetres at the very end, which a fresh approach from a
known pose usually fixes. Measured on the SO-101: the green cube went 6/6 with retries
enabled where individual approaches were closer to 4/6.

WHAT IT MUST NOT DO is carry on as though it succeeded. That is a worse outcome than a
clean failure, because the task above it then goes and "places" nothing.

THE SUBTLE PART, AND IT COST A LIVE RUN. The grasp check can be SLOWER than the action it
is checking. On the SO-101 the camera's read of the jaws runs off-thread and takes 1.5-5
seconds, so it routinely lands after the mechanical verdict has already been returned.
Observed live: contact detected, the retry loop returns success, the stack starts carrying,
and only THEN does "the jaws are empty" arrive — by which point the retry that exists for
exactly this case has been left behind, and the task goes on to open empty jaws over the
destination.

So `verify` here is allowed to be slow and is allowed to CHANGE ITS MIND. The loop calls
it, and then keeps re-asking (`settle_s`) while it still says held, so a late overrule
lands inside the attempt that can act on it. The cost is a few seconds, and only on an
attempt that was about to be thrown away.

WHAT IS DELIBERATELY NOT HERE: anything about busy latches, arm claims or stop buttons.
Those are properties of the server that owns the arm, not of retrying, and the version
this was lifted from had them tangled in — including a subtle re-arming of a "running"
flag that only makes sense when the pick is one step of a longer task. The rig passes
`between` to do its own re-homing and re-arming between tries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

__all__ = ["Outcome", "with_retries"]


@dataclass(frozen=True)
class Outcome:
    ok: bool
    detail: str
    tries: int
    #: True when a verify overturned an action that had reported success.
    overturned: bool = False


def with_retries(
    action: Callable[[int], str],
    verify: Callable[[], tuple[bool, str]],
    *,
    tries: int = 3,
    between: Callable[[int], None] | None = None,
    settle_s: float = 0.0,
    poll_s: float = 0.25,
    record: Callable[[int, bool, str], None] | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> Outcome:
    """Run ``action`` until ``verify`` confirms it, up to ``tries`` times.

    action(n)      -> a detail string. Raising is a failed attempt, not a crash: the
                      exception's message becomes the detail and the loop continues.
    verify()       -> (held, detail). Called after every attempt. May be slow.
    between(n)     -> called BEFORE attempt n for n > 1 (re-home, re-acquire, re-arm).
    settle_s       -> keep re-asking `verify` for this long while it still says held, so
                      a late overrule lands inside the attempt that can retry. See the
                      module docstring; this is the bug that motivated the parameter.
    record(n,ok,d) -> per-attempt hook, e.g. EpisodeLog.attempt.
    checkpoint()   -> raises to abort the whole loop (a Stop button). Called each round.
    """
    n_tries = max(1, int(tries))
    last = ""
    overturned = False

    for n in range(1, n_tries + 1):
        if checkpoint is not None:
            checkpoint()
        if n > 1 and between is not None:
            between(n)

        failed = False
        try:
            last = str(action(n))
        except Exception as e:
            failed = True
            last = f"{type(e).__name__}: {e}"

        held, vdetail = verify()
        # DO NOT LET A VERIFY DETAIL BURY WHY THE ACTION FAILED. When the action raised,
        # its message is the only thing that explains the attempt, and overwriting it
        # with "gripper current 0.0 vs idle 0.8" hides a TypeError behind a reading that
        # looks like an ordinary miss. That cost a debugging round: the real fault was a
        # crash mid-descent and the log said the jaws were empty, which was true and
        # entirely beside the point.
        if vdetail:
            last = f"{last}; {vdetail}" if (failed and last) else vdetail

        # Let a slow verdict land before believing a success.
        if held and settle_s > 0:
            deadline = time.time() + float(settle_s)
            while time.time() < deadline:
                time.sleep(poll_s)
                held, vdetail = verify()
                if not held:
                    overturned = True
                    last = vdetail or "the grasp check overturned it: the jaws are empty"
                    break

        if record is not None:
            record(n, held, last)
        if held:
            return Outcome(ok=True, detail=last, tries=n, overturned=False)

    return Outcome(ok=False, tries=n_tries, overturned=overturned,
                   detail=last or f"{n_tries} attempts, never confirmed")
