"""``GuestSessions`` — timed, queued access to a single shared robot.

One robot, many strangers with the link. The rules that make that work are small but
each one exists because of a way people actually behave:

* **Turns expire.** Nobody hands the arm back voluntarily.
* **Expiry is lazy.** Checked when someone asks, not by a background timer — so the
  queue can never drift out of sync with a thread that died.
* **Waiters who stop polling drop out.** Otherwise the queue fills with closed tabs and
  the next real visitor waits behind ghosts.
* **A cooldown after your turn**, so one enthusiast cannot hold the robot all evening.
* **Cooldown yields to newcomers, but does not block.** Someone still cooling goes
  behind anyone who has not played yet — and still gets a turn if nobody else wants it,
  rather than being locked out.

The whole system is driven by visitors polling :meth:`tick`. There is no background
thread, so there is nothing to get out of sync.

No Flask and no robot in here — the caller supplies the visitor id, and hands in a
callback for whatever must happen when a turn ends (parking the arm, opening the jaws).
That makes the policy testable in isolation, which it was not when it lived among the
route handlers.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass

__all__ = ["GuestSessions", "GuestConfig", "LobbyStatus"]


@dataclass(frozen=True)
class GuestConfig:
    minutes: float = 5.0
    #: After your turn ends, how long before you may claim the robot again.
    cooldown_min: float = 10.0
    #: A waiter who stops polling for this long has closed the tab.
    queue_ttl_s: float = 25.0


@dataclass
class LobbyStatus:
    """What one visitor is told when they poll."""

    playing: bool
    position: int          # 0 when playing or not queued; else 1-based place
    waiting: int
    left: float            # seconds remaining in the CURRENT turn
    eta: float             # seconds until this visitor's turn starts
    cooldown: float
    minutes: float
    token: str | None      # only ever the caller's own token

    def to_dict(self) -> dict:
        return {"playing": self.playing, "position": self.position,
                "waiting": self.waiting, "left": round(self.left),
                "eta": round(self.eta), "cooldown": round(self.cooldown),
                "minutes": self.minutes, "token": self.token}


class GuestSessions:
    """Queue, turn timer and cooldown for one shared robot."""

    def __init__(self, config: GuestConfig | None = None, *, on_turn_end=None,
                 on_turn_start=None, log=None, now=time.time):
        self.cfg = config or GuestConfig()
        self.lock = threading.RLock()
        self.holder: dict = {"token": None, "expires": 0.0, "started": 0.0}
        self.visitors: dict[str, dict] = {}   # vid -> {ip, last_end, turns, seen, token}
        self.queue: list[str] = []
        # Injected: ending a turn has to park the robot, which this module knows
        # nothing about. `now` is injectable so tests can run a whole evening in a
        # millisecond instead of sleeping through it.
        self._on_turn_end = on_turn_end or (lambda: None)
        self._on_turn_start = on_turn_start or (lambda vid, waiting: None)
        self._log = log or (lambda event, vid, **kw: None)
        self._now = now

    # --- visitors ----------------------------------------------------------
    def see(self, vid: str, ip: str = "", **extra) -> bool:
        """Record that a visitor is present. Returns True if this is a new one."""
        now = self._now()
        with self.lock:
            v = self.visitors.get(vid)
            if v is None:
                self.visitors[vid] = {"ip": ip, "last_end": 0.0, "turns": 0,
                                      "seen": now, "token": None}
                self._log("scanned", vid, **extra)
                return True
            v["seen"] = now
            if ip:
                v["ip"] = ip
            return False

    def cooldown_left(self, vid: str, now: float | None = None) -> float:
        """Seconds of cooldown remaining, 0 if none. A visitor who has never played
        has no cooldown, however recently they arrived."""
        now = self._now() if now is None else now
        v = self.visitors.get(vid)
        if not v or not v["turns"]:
            return 0.0
        return max(0.0, v["last_end"] + self.cfg.cooldown_min * 60.0 - now)

    # --- internals (caller holds the lock) ---------------------------------
    def _prune(self, now: float) -> None:
        self.queue = [v for v in self.queue
                      if v in self.visitors
                      and now - self.visitors[v]["seen"] < self.cfg.queue_ttl_s]

    def _end_turn(self, now: float) -> None:
        tok = self.holder["token"]
        if tok:
            for vid, v in self.visitors.items():
                if v.get("token") == tok:
                    v["last_end"], v["turns"] = now, v["turns"] + 1
                    self._log("turn_end", vid, turns=v["turns"],
                              held_s=round(now - self.holder["started"], 1))
                    break
            self._on_turn_end()
        self.holder.update(token=None, expires=0.0, started=0.0)

    def _promote(self, now: float) -> str | None:
        """Hand the robot to the front of the queue."""
        self._prune(now)
        while self.queue:
            vid = self.queue[0]
            # someone still cooling yields to anyone who has not played yet — but if
            # nobody else is waiting they still get their turn rather than starving
            if (self.cooldown_left(vid, now) > 0
                    and any(self.cooldown_left(o, now) <= 0 for o in self.queue[1:])):
                self.queue.append(self.queue.pop(0))
                continue
            self.queue.pop(0)
            tok = secrets.token_urlsafe(16)
            self.holder.update(token=tok, expires=now + self.cfg.minutes * 60.0,
                               started=now)
            self.visitors[vid]["token"] = tok
            self._log("turn_start", vid, waiting=len(self.queue),
                      turns=self.visitors[vid].get("turns", 0))
            self._on_turn_start(vid, len(self.queue))
            return vid
        return None

    # --- the one call the lobby makes --------------------------------------
    def tick(self, vid: str) -> LobbyStatus:
        """Register or refresh this visitor's place, and report where they stand.

        Expires the running turn, promotes whoever is next, and sweeps the queue. Doing
        all of that here is what removes the need for a background thread.
        """
        now = self._now()
        with self.lock:
            # Ticking IS the poll, so it refreshes this visitor first. Without that a
            # visitor who polls exactly at the TTL boundary prunes themselves out of
            # the queue inside their own call — they are demonstrably still there.
            if vid in self.visitors:
                self.visitors[vid]["seen"] = now
            if self.holder["token"] and now >= self.holder["expires"]:
                self._end_turn(now)
            self._prune(now)
            mine = self.visitors.get(vid, {}).get("token")
            playing = bool(self.holder["token"]) and mine == self.holder["token"]

            if not playing:
                if vid not in self.queue:
                    self._enqueue(vid, now)
                if not self.holder["token"]:
                    playing = (self._promote(now) == vid)

            pos = 0 if playing else (self.queue.index(vid) + 1 if vid in self.queue else 0)
            left = max(0.0, self.holder["expires"] - now) if self.holder["token"] else 0.0
            ahead = max(0, pos - 1)
            return LobbyStatus(
                playing=playing, position=pos, waiting=len(self.queue),
                left=left, eta=(left + ahead * self.cfg.minutes * 60.0) if pos else 0.0,
                cooldown=self.cooldown_left(vid, now), minutes=self.cfg.minutes,
                token=self.holder["token"] if playing else None)

    def _enqueue(self, vid: str, now: float) -> None:
        """Join the queue — behind everyone still on their first turn if cooling."""
        if self.cooldown_left(vid, now) > 0:
            self.queue.append(vid)
            return
        at = len(self.queue)
        for i, other in enumerate(self.queue):
            if self.cooldown_left(other, now) > 0:
                at = i
                break
        self.queue.insert(at, vid)

    # --- read-only state ---------------------------------------------------
    def state(self) -> tuple[bool, float, str | None]:
        """``(active, seconds_left, token)``. Expiry is lazy — checked on read."""
        with self.lock:
            now = self._now()
            if self.holder["token"] and now >= self.holder["expires"]:
                self._end_turn(now)
            left = max(0.0, self.holder["expires"] - now) if self.holder["token"] else 0.0
            return bool(self.holder["token"]), left, self.holder["token"]

    def holds_turn(self, vid: str | None) -> bool:
        """Is this visitor the one whose turn it currently is?"""
        active, left, tok = self.state()
        if not (active and left > 0) or not vid:
            return False
        return self.visitors.get(vid, {}).get("token") == tok

    def end_current(self) -> bool:
        """Cut the active turn short (an operator kicking someone off)."""
        with self.lock:
            if not self.holder["token"]:
                return False
            self._end_turn(self._now())
            return True

    def roster(self) -> list[dict]:
        with self.lock:
            now = self._now()
            return [{"vid": vid, "ip": v["ip"], "turns": v["turns"],
                     "seen_s": round(now - v["seen"], 1),
                     "cooldown": round(self.cooldown_left(vid, now)),
                     "queued": vid in self.queue,
                     "playing": bool(self.holder["token"]) and v.get("token") == self.holder["token"]}
                    for vid, v in self.visitors.items()]
