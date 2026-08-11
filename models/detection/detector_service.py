"""``DetectorService`` — an open-vocabulary detector that keeps up with a control loop.

An open-vocab detector is slow. Running one inline turns a 20 Hz control loop into a
0.4 Hz one, so it runs on its own thread and the control path reads whatever it last
produced. That alone is not enough, because a cached box does not move: between cycles
the position is frozen, then jumps when the next detection lands. On an approaching
wrist camera that reads as "the box cannot keep up".

So this owns three things and the relationship between them:

* **the detector**, ticked on a background thread at a fixed period;
* **the query**, which can only be mutated on the thread that runs the model — calling
  ``set_query`` from an HTTP handler crashes it, so changes are queued and applied at
  the top of the next cycle;
* **the trackers**, one per label, re-tagged from each fresh detection and then stepped
  frame-to-frame in between, so a caller asking "where is it now" gets a box that
  followed the object rather than one that is up to a full cycle stale.

Nothing here knows about a robot. It needs a detector, a frame source, and somewhere to
report — which is what lets the same service back a wrist camera, a fixed camera, or a
test harness feeding it synthetic frames.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from models.detection.tracking import PixelTracker, Track

__all__ = ["DetectorService", "DetectorConfig", "box_iou", "dedupe_boxes"]


@dataclass
class DetectorConfig:
    """Timing and filtering for the detection loop."""

    period_s: float = 2.5
    #: Boxes of the SAME label overlapping by more than this are one object. Several
    #: prompts routinely fire on the same thing.
    nms_iou: float = 0.55
    #: A cached detection older than this is not worth reporting.
    max_age_s: float = 4.0
    #: Minimum fraction of a box that must actually BE the colour its label names.
    #: Only labels containing a colour word are gated, so "pen" is never asked to be
    #: red while "red cube" still cannot latch onto the green one.
    colour_min_frac: float = 0.15
    #: Labels with their own dedicated trackers, which must not be given a generic one.
    reserved_labels: tuple[str, ...] = ()


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 1e-9 else 0.0


def dedupe_boxes(instances, iou: float) -> list:
    """Greedy NMS within one label, highest confidence first."""
    kept = []
    for e in instances:
        if all(box_iou(e["xyxy"], k["xyxy"]) < iou for k in kept):
            kept.append(e)
    return kept


def track_from_box(rgb, xyxy, t) -> Track:
    """A Track from a raw box, flagged if it touches the frame edge.

    Clipped boxes matter downstream: an object cut off by the frame has no reliable
    width, so anything ranging by apparent size has to know not to trust it.
    """
    x1, y1, x2, y2 = xyxy
    clipped = x1 <= 1 or y1 <= 1 or x2 >= rgb.shape[1] - 2 or y2 >= rgb.shape[0] - 2
    return Track(((x1 + x2) / 2.0, (y1 + y2) / 2.0), tuple(xyxy),
                 int((x2 - x1) * (y2 - y1)), clipped, t)


@dataclass
class _Latest:
    t: float = 0.0
    dets: dict = field(default_factory=dict)


class DetectorService:
    """Runs a detector off the control loop and keeps per-label trackers current.

    ``frame_source`` returns the most recent RGB frame (or None). ``colour_ok`` is an
    optional gate ``(rgb, label, xyxy) -> bool``; ``log`` receives one-line notices.
    """

    def __init__(self, detector, frame_source, *, config: DetectorConfig | None = None,
                 colour_ok=None, log=None, on_query_change=None):
        self.detector = detector
        self.frame_source = frame_source
        self.cfg = config or DetectorConfig()
        self._colour_ok = colour_ok
        self._log = log or (lambda msg: None)
        self._on_query_change = on_query_change
        self.lock = threading.RLock()
        self.latest = _Latest()
        self.trackers: dict[str, PixelTracker] = {}
        self.query = ""
        self._pending: list[str | None] = [None]
        self._thread: threading.Thread | None = None
        #: Set to end run_forever. The loop must stop touching the frame source before
        #: the camera behind it is disconnected, or the device is torn down underneath
        #: an in-flight read.
        self.stopping = threading.Event()

    # --- query ------------------------------------------------------------------
    def labels(self) -> list[str]:
        """The active query parsed into labels, order preserved."""
        return [p.strip() for p in (self.query or "").lower().split(",") if p.strip()]

    def request_query(self, q: str) -> None:
        """Queue a query change. Applied at the top of the next cycle.

        NOT applied here. The model may only be mutated on the thread that runs it —
        calling into it from an HTTP handler crashes the process.
        """
        self._pending[0] = q

    def apply_pending_query(self) -> bool:
        pending = self._pending[0]
        if pending is None or self.detector is None:
            return False
        self._pending[0] = None
        try:
            self.detector.set_query(pending)
            self.query = pending
            if self._on_query_change:
                self._on_query_change(pending)
            self._log(f"detection query set: {pending}")
            return True
        except Exception as e:
            self._log(f"query change failed: {e}")
            return False

    def wait_for_query(self, q: str, timeout: float = 6.0) -> bool:
        """Request a query and wait until a detection cycle has run under it.

        Callers that immediately look for the new labels need this: asking for a class
        and searching for it in the same breath searches under the OLD vocabulary.
        """
        self.request_query(q)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._pending[0] is None and self.query == q:
                t0 = self.latest.t
                while time.time() < deadline and self.latest.t == t0:
                    time.sleep(0.05)
                return True
            time.sleep(0.05)
        return False

    # --- the loop ---------------------------------------------------------------
    def start(self, daemon: bool = True) -> threading.Thread:
        self._thread = threading.Thread(target=self.run_forever, daemon=daemon)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self.stopping.set()

    def run_forever(self) -> None:
        while not self.stopping.is_set():
            time.sleep(self.cfg.period_s)
            if self.stopping.is_set():
                break
            try:
                self.tick()
            except Exception as e:
                self._log(f"detector tick failed: {type(e).__name__}: {e}")

    def tick(self) -> dict:
        """One detection cycle. Returns the per-label instances it produced."""
        self.apply_pending_query()
        rgb = self.frame_source()
        if rgb is None or self.detector is None:
            return {}
        try:
            dets = self.detector.predict_rgb(np.ascontiguousarray(rgb))
        except Exception:
            return {}

        labels = self.labels()
        by_label: dict[str, list] = {}
        for d in dets:
            if not (0 <= d.class_id < len(labels)):
                continue
            lbl = labels[d.class_id]
            if self._colour_ok and not self._colour_ok(rgb, lbl, d.xyxy):
                continue
            by_label.setdefault(lbl, []).append(
                {"xyxy": tuple(float(v) for v in d.xyxy), "conf": float(d.confidence)})
        for lbl, inst in by_label.items():
            inst.sort(key=lambda e: -e["conf"])
            by_label[lbl] = dedupe_boxes(inst, self.cfg.nms_iou)

        with self.lock:
            self.latest = _Latest(time.time(), by_label)
            self._retag(rgb, by_label, labels)
        return by_label

    def _retag(self, rgb, by_label, labels) -> None:
        """Re-anchor each label's tracker on this cycle's best box, and retire the
        trackers for labels nobody is asking for any more — a stale one keeps
        confidently reporting an object that is no longer being looked for."""
        for lbl, inst in by_label.items():
            if lbl in self.cfg.reserved_labels or not inst:
                continue
            self.trackers.setdefault(lbl, PixelTracker(lbl)).tag(rgb, inst[0]["xyxy"])
        for lbl in [l for l in self.trackers if l not in labels]:
            del self.trackers[lbl]

    # --- queries against the latest cycle ---------------------------------------
    def instances(self, label: str) -> tuple[list, float]:
        with self.lock:
            return list(self.latest.dets.get(str(label).strip().lower(), ())), self.latest.t

    def find_all(self, rgb, label: str, max_age: float | None = None) -> list[Track]:
        """EVERY current instance of a label. A table can hold three cups and a map has
        to carry all three, so this cannot collapse to one box per label."""
        inst, t = self.instances(label)
        age = self.cfg.max_age_s if max_age is None else max_age
        if not inst or time.time() - t > age:
            return []
        return [track_from_box(rgb, e["xyxy"], t) for e in inst]

    def find(self, rgb, label: str) -> Track | None:
        """The single best instance, tracked to THIS frame.

        Steps the label's tracker rather than returning the cached detection, so the
        box follows the object between cycles instead of sitting still and then
        jumping. Bootstraps a tracker from the cached box when there is not one yet,
        so the next call is already smooth rather than waiting a full cycle.
        """
        label = str(label).strip().lower()
        pt = self.trackers.get(label)
        if pt is not None:
            tr = pt.track(rgb)
            if tr is not None:
                return tr
        tracks = self.find_all(rgb, label)
        if not tracks:
            return None
        self.trackers.setdefault(label, PixelTracker(label)).tag(rgb, tracks[0].bbox_xyxy)
        return tracks[0]

    def visible(self, label: str, max_age: float | None = None) -> bool:
        inst, t = self.instances(label)
        age = self.cfg.max_age_s if max_age is None else max_age
        return bool(inst) and (time.time() - t) <= age
