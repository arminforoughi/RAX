"""``MissionState`` — the shared "what is the robot doing right now" record.

Every layer of the pick stack reports through the same three things: a phase string,
a rolling log, and a stop flag that long-running motions poll. In the monolith those
were a module-level dict, a ``deque`` and an ``Event``, written from about twenty
places and read by the HTTP layer. Extracting the algorithms means they need something
to report *to* that is not a module global, so it becomes an object that gets passed in.

The dict is kept as a plain dict rather than becoming typed fields on purpose: the
admin and guest UIs read these keys straight out of ``/status`` as JSON, and several
keys (``obj3d``, ``carry``, ``jog_xyz``) are set opportunistically by whichever
subsystem happens to be running. :meth:`snapshot` therefore returns exactly what the
UI has always seen.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

__all__ = ["Abort", "MissionState", "DEFAULT_STATE"]

# The keys the UI expects to exist on a cold server, with their cold values.
DEFAULT_STATE: dict[str, Any] = {
    "phase": "IDLE",
    "detail": "",
    "joints": [],
    "gripper": None,
    "p_red": None,
    "p_green": None,
    "t0": None,
    "running": False,
    "loop_hz": 0.0,
    "dist_mm": None,   # live camera->target range during approach/place
}

LOG_MAXLEN = 140


class Abort(Exception):
    """Raised at a checkpoint when the user has asked the current mission to stop.

    Missions are cooperative: they call :meth:`MissionState.checkpoint` between
    motions rather than being killed, so the arm always stops at a pose it chose.
    """


class MissionState:
    """Phase, log, and stop flag for one robot, safe to share across threads."""

    def __init__(
        self,
        *,
        log_maxlen: int = LOG_MAXLEN,
        defaults: dict[str, Any] | None = None,
        echo: Callable[[str], None] | None = None,
    ):
        self.lock = threading.RLock()
        self.log: deque[str] = deque(maxlen=log_maxlen)
        self.stop_flag = threading.Event()
        self._state: dict[str, Any] = dict(DEFAULT_STATE if defaults is None else defaults)
        # Where say() mirrors its lines. Default is the console, as before; tests pass
        # a collector, and a headless run can pass a no-op.
        self._echo = echo if echo is not None else (lambda line: print(line, flush=True))

    # --- reporting --------------------------------------------------------------
    def say(self, msg: str) -> None:
        """Append one timestamped line to the rolling log and echo it."""
        self.log.appendleft(f"{time.strftime('%H:%M:%S')}  {msg}")
        self._echo(msg)

    def set_phase(self, phase: str, detail: str = "") -> None:
        """Move to a new phase and log it. Clears the live range read-out, which
        belongs to the phase that just ended."""
        with self.lock:
            self._state["phase"] = phase
            self._state["detail"] = detail
            self._state["dist_mm"] = None
        self.say(f"[{phase}] {detail}" if detail else f"[{phase}]")

    # --- cooperative stop -------------------------------------------------------
    def checkpoint(self) -> None:
        """Raise :class:`Abort` if the user pressed stop. Call between motions."""
        with self.lock:
            running = self._state.get("running", False)
        if self.stop_flag.is_set() and running:
            raise Abort("stopped by user")

    def begin_run(self) -> None:
        """Mark a mission as started and clear any stale stop request."""
        self.stop_flag.clear()
        with self.lock:
            self._state["running"] = True
            self._state["t0"] = time.time()

    def end_run(self) -> None:
        with self.lock:
            self._state["running"] = False

    @property
    def running(self) -> bool:
        with self.lock:
            return bool(self._state.get("running", False))

    def request_stop(self) -> None:
        self.stop_flag.set()

    # --- state access -----------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        with self.lock:
            return self._state.get(key, default)

    def update(self, **kw: Any) -> None:
        with self.lock:
            self._state.update(kw)

    def __getitem__(self, key: str) -> Any:
        with self.lock:
            return self._state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self.lock:
            self._state[key] = value

    def __contains__(self, key: str) -> bool:
        with self.lock:
            return key in self._state

    def snapshot(self) -> dict[str, Any]:
        """A shallow copy of the state dict — exactly what ``/status`` serializes."""
        with self.lock:
            return dict(self._state)

    def log_lines(self) -> list[str]:
        return list(self.log)
