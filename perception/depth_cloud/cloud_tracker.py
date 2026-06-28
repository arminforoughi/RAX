"""``CloudTracker`` — detect, tag, and efficiently stream per-object point clouds.

This is the core of the new direction. Each tick it:

  1. (throttled) runs the open-vocab detector to discover/refresh every matching
     object, and associates detections to persistent, tagged tracks;
  2. updates point clouds on a **budget**: the *focused* object every tick (with
     a precise SAM2 mask), the surrounding objects round-robin one per tick (with
     a cheap box mask) — so cost is bounded no matter how cluttered the scene;
  3. publishes only the clouds that changed to a :class:`PointCloudStream`.

The focused object's cloud centroid is what the gaze engine approaches/grasps;
another track's top surface is the place-on target. Everything is in base frame.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from manipulation.arms.arm_interface import Observation
from models.depth.stereo import StereoDepth, clamp_roi
from models.detection import Detection, MaskTracker, PromptDetector
from perception.depth_cloud.backproject import (
    backproject_masked,
    box_iou,
    box_mask,
    box_to_roi,
)
from perception.depth_cloud.object_cloud import ObjectTrack
from perception.depth_cloud.stream import CloudUpdate, PointCloudStream

logger = logging.getLogger(__name__)


class CloudTracker:
    def __init__(
        self,
        detector: PromptDetector,
        mask_tracker: MaskTracker,
        stereo: StereoDepth,
        query: str,
        *,
        extra_labels: tuple[str, ...] = (),
        detect_every: int = 5,
        roi_pad: int = 14,
        max_points: int = 3000,
        max_misses: int = 30,
        assoc_iou: float = 0.2,
        stream: PointCloudStream | None = None,
    ):
        self.detector = detector
        self.mask_tracker = mask_tracker
        self.stereo = stereo
        # First query is the focused object; the rest (e.g. a place-on target) are
        # tracked too so the whole working set of objects gets streamed clouds.
        self.focus_label = query
        self.queries = [query, *(q for q in extra_labels if q and q != query)]
        self.detect_every = max(1, int(detect_every))
        self.roi_pad = int(roi_pad)
        self.max_points = int(max_points)
        self.max_misses = int(max_misses)
        self.assoc_iou = float(assoc_iou)
        self.stream = stream or PointCloudStream()

        self.tracks: dict[int, ObjectTrack] = {}
        self._next_tag = 1
        self.focus_tag: int | None = None
        self._need_focus_init = False
        self._tick = 0
        self._rr = 0  # round-robin cursor over non-focused tracks
        self._focus_locked = False

    # -- public API ---------------------------------------------------------

    def lock_focus(self) -> None:
        """After the gaze engine locks an object, stop spawning rival tracks."""
        if self.focus_tag is None:
            return
        self._focus_locked = True
        for tag in list(self.tracks):
            if tag != self.focus_tag:
                self._drop(tag)
        logger.info("[cloud] focus locked on tag=%d (no new tracks)", self.focus_tag)

    def focus(self, tag: int | None) -> None:
        """Set the followed object. Re-seeds the SAM2 mask on the next update."""
        if tag is not None and tag not in self.tracks:
            raise KeyError(f"no track with tag {tag}")
        self.focus_tag = tag
        self._need_focus_init = tag is not None

    def focus_track(self) -> ObjectTrack | None:
        return self.tracks.get(self.focus_tag) if self.focus_tag is not None else None

    def get(self, tag: int) -> ObjectTrack | None:
        return self.tracks.get(tag)

    def list_tracks(self) -> list[ObjectTrack]:
        return list(self.tracks.values())

    def update(self, obs: Observation) -> list[CloudUpdate]:
        """Advance one tick; returns (and publishes) the changed clouds."""
        self._tick += 1
        t = obs.t

        if self._tick % self.detect_every == 1 or self.detect_every == 1:
            self._detect_and_associate(obs.left, t)
            self._auto_focus_if_needed()

        updates = self._update_clouds(obs)
        self.stream.publish(updates)
        return updates

    # -- detection / association -------------------------------------------

    def _detect_and_associate(self, rgb, t: float) -> None:
        dets: list[Detection] = []
        for q in self.queries:
            dets.extend(self.detector.detect(rgb, q))
        matched: set[int] = set()
        used: set[int] = set()  # consumed detection indices

        # 1) The focus track gets FIRST claim. Prefer IoU with the current box so a
        #    drifting blob (gripper edge, table) does not steal the lock.
        ft = self.tracks.get(self.focus_tag) if self.focus_tag is not None else None
        if ft is not None:
            best_i, best_score = None, -1.0
            fcx, fcy = ft.center_px
            fdiag = float(np.hypot(*(np.subtract(ft.box[2:], ft.box[:2]))))
            for i, det in enumerate(dets):
                if det.label != ft.label:
                    continue
                dcx, dcy = det.center
                d = float(np.hypot(dcx - fcx, dcy - fcy))
                ddiag = float(np.hypot(*(np.subtract(det.box[2:], det.box[:2]))))
                if d > max(160.0, 2.5 * max(fdiag, ddiag)):
                    continue
                score = box_iou(det.box, ft.box) * float(det.confidence) * math.sqrt(max(1.0, det.area))
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_i is not None:
                ft.observe(dets[best_i].box, dets[best_i].confidence, t)
                matched.add(ft.tag)
                used.add(best_i)

        if self._focus_locked:
            return

        # 2) Remaining detections associate to remaining tracks (or spawn new ones).
        for i, det in enumerate(dets):
            if i in used:
                continue
            if self._should_ignore_detection(det):
                continue
            tag = self._best_match(det, matched)
            if tag is None and self.focus_tag is not None:
                tag = self._merge_to_focus(det)
            if tag is None:
                tag = self._spawn(det, t)
            else:
                self.tracks[tag].observe(det.box, det.confidence, t)
            matched.add(tag)

        # 3) Age out unmatched tracks — but NEVER the focus track (it stays locked
        #    even through brief detection dropouts).
        for tag, tr in list(self.tracks.items()):
            if tag in matched or tag == self.focus_tag:
                continue
            tr.misses += 1
            if tr.misses > self.max_misses:
                self._drop(tag)

    def _merge_to_focus(self, det: Detection) -> int | None:
        """If a detection is near the locked focus, absorb it instead of spawning."""
        if self.focus_tag is None or self.focus_tag not in self.tracks:
            return None
        ft = self.tracks[self.focus_tag]
        if det.label != ft.label:
            return None
        fcx, fcy = ft.center_px
        dcx, dcy = det.center
        diag = float(np.hypot(*(np.subtract(det.box[2:], det.box[:2]))))
        if float(np.hypot(dcx - fcx, dcy - fcy)) < max(140.0, 3.0 * diag):
            return self.focus_tag
        return None

    def _should_ignore_detection(self, det: Detection) -> bool:
        """Drop low-confidence speckles far from the focused object."""
        if self.focus_tag is None or self.focus_tag not in self.tracks:
            return False
        ft = self.tracks[self.focus_tag]
        fcx, fcy = ft.center_px
        dcx, dcy = det.center
        fd = float(np.hypot(fcx - dcx, fcy - dcy))
        return det.confidence < 0.35 and fd > 200.0

    def _best_match(self, det: Detection, used: set[int]) -> int | None:
        """Associate to a same-label track by IoU, falling back to box-centre distance.

        Between throttled detections the box can grow/shift as the camera moves, so
        IoU alone is brittle; a centre-distance gate (within half the detection's
        diagonal) keeps the tag stable through that drift.
        """
        dcx, dcy = det.center
        diag = float(np.hypot(*(np.subtract(det.box[2:], det.box[:2]))))
        best_tag, best_iou = None, self.assoc_iou
        for tag, tr in self.tracks.items():
            if tag in used or tr.label != det.label:
                continue
            iou = box_iou(det.box, tr.box)
            if iou >= best_iou:
                best_tag, best_iou = tag, iou
        if best_tag is not None:
            return best_tag
        # Fallback: nearest same-label track centre. Generous gate (>1 diagonal)
        # so a fast-moving camera keeps the object as ONE track instead of spawning
        # duplicates every detection round.
        best_tag, best_d = None, 1.5 * diag
        for tag, tr in self.tracks.items():
            if tag in used or tr.label != det.label:
                continue
            tcx, tcy = tr.center_px
            d = float(np.hypot(dcx - tcx, dcy - tcy))
            if d <= best_d:
                best_tag, best_d = tag, d
        return best_tag

    def _spawn(self, det: Detection, t: float) -> int:
        tag = self._next_tag
        self._next_tag += 1
        self.tracks[tag] = ObjectTrack(
            tag=tag, label=det.label, box=det.box, confidence=det.confidence, last_seen_t=t
        )
        logger.info("[cloud] new track tag=%d label=%r conf=%.2f", tag, det.label, det.confidence)
        return tag

    def _drop(self, tag: int) -> None:
        self.tracks.pop(tag, None)
        if self.focus_tag == tag:
            self.focus_tag = None
        logger.info("[cloud] dropped track tag=%d", tag)

    def _auto_focus_if_needed(self) -> None:
        if self.focus_tag in self.tracks:
            return
        # Default focus: highest-confidence track of the primary (focus) label.
        cands = [tr for tr in self.tracks.values() if tr.label == self.focus_label]
        if not cands:
            return
        tag = max(cands, key=lambda tr: tr.confidence).tag
        self.focus(tag)
        logger.info("[cloud] auto-focus tag=%d", tag)

    # -- cloud updates (the budget) ----------------------------------------

    def _scheduled_tags(self) -> list[int]:
        """Focus every tick + one round-robin non-focus track per tick."""
        if self._focus_locked and self.focus_tag in self.tracks:
            return [self.focus_tag]
        scheduled: list[int] = []
        if self.focus_tag in self.tracks:
            scheduled.append(self.focus_tag)
        others = [tg for tg in self.tracks if tg != self.focus_tag]
        if others:
            self._rr %= len(others)
            scheduled.append(others[self._rr])
            self._rr += 1
        return scheduled

    def _update_clouds(self, obs: Observation) -> list[CloudUpdate]:
        h, w = obs.left.shape[:2]
        updates: list[CloudUpdate] = []
        for tag in self._scheduled_tags():
            tr = self.tracks[tag]
            roi = clamp_roi(box_to_roi(tr.box), w, h, pad=self.roi_pad)
            is_focus = tag == self.focus_tag

            if is_focus:
                if self._need_focus_init:
                    mask = self.mask_tracker.init(obs.left, tr.box)
                    self._need_focus_init = False
                else:
                    mask = self.mask_tracker.track(obs.left, box_hint=tr.box)
            else:
                mask = box_mask((h, w), tr.box)

            depth = self.stereo.depth_meters(obs.left, obs.right, intr=obs.intrinsics, roi=roi)
            pts = backproject_masked(
                depth, mask, obs.intrinsics, obs.T_base_cam, roi=roi, max_points=self.max_points
            )
            if pts.shape[0] == 0:
                continue
            valid = np.isfinite(depth) & mask
            range_cam = float(np.median(depth[valid])) if np.any(valid) else float("nan")
            tr.set_cloud(pts, t=obs.t, range_cam_m=range_cam)
            updates.append(
                CloudUpdate(tag, tr.label, pts, tr.centroid, is_focus=is_focus, t=obs.t)
            )
        return updates
