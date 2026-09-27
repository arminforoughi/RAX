"""What was attempted, and how it went. One JSON line per run, with the frames.

WHY THIS IS WORTH A MODULE. Every real fault on this rig was found by comparing what the
log CLAIMED against what a camera SHOWED: "green cube picked" while the overhead showed it
still on the table, a grasp "held" that the vision model called empty, a survey that
confidently placed a cube 25 cm from where it was. A run nobody recorded cannot be argued
with afterwards, and tuning against unrecorded runs is how an afternoon disappears into
the wrong number.

It also makes "did that change help" a COUNTABLE question. The grip metric was validated
across 13 trials this way (<=67px picked, >=96px missed); the yaw-alignment experiment was
killed by 0/6 against 6/6. Neither conclusion was available from watching.

WHY IT TAKES CALLBACKS INSTEAD OF DOING IT ITSELF. The version this was lifted from
reached straight into one server's globals — `observe()` for the wrist frame, a hardcoded
surveillance-camera URL and password for the room view, `_tip()` for the fingertip, `say()`
for the log. None of that is about recording episodes; it is about which rig is recording.
So the rig supplies them:

    views   () -> {name: path}   save whatever cameras exist, return where they went
    probe   () -> dict           extra per-attempt facts (fingertip position, joints)
    note    (str) -> None        where a human-readable line should go

All four default to doing nothing, so recording NEVER fails a run. That is deliberate and
it is the one place "swallow the exception" is right here: an episode is diagnostic, and a
pick that worked must not be reported as failed because a disk was full.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

__all__ = ["Episode", "EpisodeLog"]


@dataclass
class Episode:
    """One attempt-sequence in progress. Plain data; the log writes it out at the end."""

    id: int
    kind: str
    label: str
    t0: float
    attempts: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    frames_before: dict = field(default_factory=dict)
    frames_after: dict = field(default_factory=dict)
    ok: bool | None = None
    detail: str = ""

    def as_record(self) -> dict:
        r = {"id": self.id, "kind": self.kind, "label": self.label,
             **self.extra,
             "attempts": self.attempts,
             "frames_before": self.frames_before,
             "frames_after": self.frames_after,
             "tries": len(self.attempts),
             "seconds": round(time.time() - self.t0, 1)}
        if self.ok is not None:
            r["ok"] = bool(self.ok)
            r["detail"] = self.detail
        return r


class EpisodeLog:
    """Append-only episode recorder. Construct one per server, reuse for every run."""

    def __init__(self, path: str | Path, *,
                 views: Callable[[str, int], dict] | None = None,
                 probe: Callable[[], dict] | None = None,
                 note: Callable[[str], None] | None = None):
        self.path = Path(path)
        self._views = views
        self._probe = probe
        self._note = note or (lambda _m: None)

    # ---- the three calls a controller makes -------------------------------------
    def start(self, kind: str, label: str, **extra: Any) -> Episode:
        ep = Episode(id=int(time.time()), kind=kind, label=label, t0=time.time(),
                     extra=dict(extra))
        ep.frames_before = self._snap("before", ep.id)
        return ep

    def attempt(self, ep: Episode, n: int, ok: bool, detail: str, **extra: Any) -> None:
        """Record ONE try.

        Attempts are kept individually rather than collapsed to the last one, because the
        interesting question is almost always how many it took — a pick that works on the
        third try every time is a different problem from one that works on the first.
        """
        rec = {"n": int(n), "ok": bool(ok), "detail": str(detail)[:300],
               "t": round(time.time() - ep.t0, 1), **extra}
        if self._probe is not None:
            try:
                rec.update(self._probe())
            except Exception:
                pass
        ep.attempts.append(rec)
        self._note(f"[episode {ep.id} try {n}] {'OK' if ok else 'FAILED'}: "
                   f"{str(detail)[:90]}")

    def end(self, ep: Episode, ok: bool, detail: str) -> Episode:
        ep.frames_after = self._snap("after", ep.id)
        ep.ok, ep.detail = bool(ok), str(detail)[:300]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ep.as_record()) + "\n")
        except Exception as e:
            self._note(f"(could not write the episode: {type(e).__name__}: {e})")
        tries = len(ep.attempts)
        self._note(f"[episode {ep.id}] {'SUCCESS' if ok else 'FAILED'} after {tries} "
                   f"tr{'y' if tries == 1 else 'ies'}, "
                   f"{time.time() - ep.t0:.0f}s")
        return ep

    # ---- reading them back ------------------------------------------------------
    def records(self, kind: str | None = None, label: str | None = None) -> list[dict]:
        """Every recorded episode, oldest first. Bad lines are skipped, not fatal.

        A half-written last line is normal — the file is appended to by a process that
        can be stopped at any moment — so a reader that raises on it would make the log
        unreadable exactly when something has gone wrong.
        """
        out = []
        try:
            text = self.path.read_text(encoding="utf-8")
        except Exception:
            return out
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if kind is not None and r.get("kind") != kind:
                continue
            if label is not None and r.get("label") != label:
                continue
            out.append(r)
        return out

    def tally(self, kind: str | None = None) -> dict:
        """Successes, attempts and mean tries — the numbers a change gets judged on."""
        recs = [r for r in self.records(kind) if "ok" in r]
        if not recs:
            return {"runs": 0, "ok": 0, "rate": 0.0, "mean_tries": 0.0}
        ok = sum(1 for r in recs if r["ok"])
        return {"runs": len(recs), "ok": ok, "rate": ok / len(recs),
                "mean_tries": sum(r.get("tries", 1) for r in recs) / len(recs)}

    # ---- internals --------------------------------------------------------------
    def _snap(self, when: str, ep_id: int) -> dict:
        if self._views is None:
            return {}
        try:
            return dict(self._views(when, ep_id) or {})
        except Exception:
            # Diagnostic only — never fail a run because a camera would not answer.
            return {}
