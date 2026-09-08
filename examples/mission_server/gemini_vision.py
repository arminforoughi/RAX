"""Asking a vision model the questions geometry cannot answer.

WHAT THIS IS FOR, and what it deliberately is not. A VLM behind a network call answers
in roughly a second. A visual servo closing a 250 px error runs at the camera's frame
rate. So nothing here steers the arm — the servo in ``manipulation/approach`` does that,
and it does it from pixels it measures itself. What a model is good at is the questions
the pipeline currently answers with a hardcoded number or cannot answer at all:

``explain_miss``
    The centring step reports "object not in view" and dumps the frame to disk hoping a
    human looks at it later. Two real dumps: in one the cube is present but cut off by
    the BOTTOM of the frame and badly motion-blurred; in the other the camera is aimed
    at a bag of clutter with no cube anywhere. Those need opposite responses — reframe
    versus re-survey — and the log said "object not in view" for both.

``verify_grasp``
    Whether the jaws actually hold the object is currently inferred from a gripper
    current delta, which cannot tell "holding the cube" from "holding the table edge"
    or from a finger fouled on the object. One frame after the lift settles it.

``same_object``
    A close-up re-measure that disagrees with the target by more than
    ``max_refine_jump_m`` is discarded as "a different object". That is a real question
    about identity being decided by a distance threshold — 12 cm, tuned once. The model
    looks at the two crops and answers the question that was actually being asked.

EVERY CALL FAILS SOFT. No API key, no SDK, no network, a timeout, a malformed answer —
all return a verdict with ``ok=False`` and the server proceeds exactly as it did before.
A vision model is an improvement to a working pick, never a dependency of one. It is
also OFF the critical path by default: see ``ADVISORY_ONLY``.
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
import time
from dataclasses import dataclass, field

__all__ = [
    "Verdict", "GeminiVision", "MISS_REASONS", "USEFUL_WITHIN_S",
    "parse_verdict", "load_api_key", "available",
]

#: Default model. Flash-class on purpose: this sits near a control loop, so latency is
#: the property that matters and the questions are easy ones. Override with RAX_GEMINI_MODEL.
#:
#: Chosen by running both candidates over two real center_miss dumps from this rig.
#: 3.1-flash-lite answered both correctly (clipped_bottom / not_in_frame, confidence 1.0,
#: median 2.3 s and 3.0 s); 3.5-flash-lite got the empty frame right but read-timed-out
#: repeatedly on the harder one. Re-run scripts/gemini_bench.py before changing this.
DEFAULT_MODEL = os.environ.get("RAX_GEMINI_MODEL", "gemini-3.1-flash-lite")

#: Hard ceiling on any single call, seconds. Past this the answer is worthless anyway —
#: the arm has moved on — so the call is abandoned rather than waited on.
#:
#: The API refuses a deadline under 10 s outright ("Manually set deadline 6s is too
#: short"), so this cannot be tightened to control-loop timescales; it is a ceiling on a
#: hung call, not a latency budget. What actually keeps this off the critical path is
#: running it on another thread (``ask_async``) and treating a slow answer as no answer.
MIN_TIMEOUT_S = 10.0
DEFAULT_TIMEOUT_S = max(float(os.environ.get("RAX_GEMINI_TIMEOUT_S", "12.0")), MIN_TIMEOUT_S)

#: How long an answer may take and still be worth acting on. Past this the arm has moved
#: and the frame the model judged no longer describes where it is, so the verdict is
#: logged and ignored. Separate from the transport deadline above on purpose.
USEFUL_WITHIN_S = float(os.environ.get("RAX_GEMINI_USEFUL_S", "3.0"))

#: When true, nothing this module returns is allowed to change what the arm does; the
#: verdicts are logged only. Start here. Turning it off is a deliberate decision to let
#: a remote model gate a physical motion, and it should be made on evidence from the
#: logs it produces while it is on.
ADVISORY_ONLY = os.environ.get("RAX_GEMINI_ADVISORY", "1") != "0"

#: The answers ``explain_miss`` may give, each mapped to what the robot should do about
#: it. A free-text answer would be unusable: the caller has to branch on this, so it is
#: a closed set and the model is constrained to it by the response schema.
MISS_REASONS = {
    "visible":            "the object is fully visible and unobstructed — this is a "
                          "DETECTOR failure, not a framing one",
    "clipped_bottom":     "cut off by the bottom of the frame — the camera is aimed "
                          "over it; it cannot be ranged from its table contact",
    "clipped_top":        "cut off by the top of the frame — still rangeable from where "
                          "its bottom edge meets the table",
    "clipped_side":       "cut off at the left or right — its centre, and any bearing "
                          "taken from it, is biased inward",
    "occluded_by_gripper": "the gripper is in front of it — back off before looking again",
    "not_in_frame":       "no such object in the picture at all — the camera is aimed "
                          "somewhere else; re-survey rather than reframe",
    "too_blurred":        "the frame is too blurred or dark to tell — settle longer "
                          "before grabbing a frame",
}


@dataclass
class Verdict:
    """One answer, plus enough to decide whether to act on it.

    ``ok`` is the honest signal: False means the model was not consulted or did not
    answer usefully, and the caller must fall back to whatever it did before. It is
    separate from ``answer`` on purpose — "I could not tell" and "I did not run" are
    different facts and collapsing them is how a silent outage becomes a silent
    behaviour change.
    """

    ok: bool
    answer: str = ""
    reason: str = ""
    confidence: float = 0.0
    latency_s: float = 0.0
    raw: str = field(default="", repr=False)

    def describe(self) -> str:
        if not self.ok:
            return f"gemini: unavailable ({self.reason})"
        return (f"gemini: {self.answer} ({self.confidence:.0%}) — {self.reason} "
                f"[{self.latency_s:.1f}s]")


def load_api_key(repo_root=None) -> str:
    """GOOGLE_API_KEY from the environment, else from the repo's .env.local.

    The server does not use python-dotenv, so a key sitting in .env.local — where this
    repo's key actually lives — is invisible to it. Read the file directly rather than
    making the operator export it by hand, and never log the value.
    """
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if key:
        return key
    root = pathlib.Path(repo_root) if repo_root else pathlib.Path(__file__).resolve().parents[2]
    for name in (".env.local", ".env"):
        p = root / name
        try:
            for line in p.read_text().splitlines():
                line = line.strip()
                if line.startswith("GOOGLE_API_KEY") and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


def available() -> bool:
    """Is a call even possible? Cheap, no network."""
    if not load_api_key():
        return False
    try:
        import google.genai  # noqa: F401
    except Exception:
        return False
    return True


def parse_verdict(text: str, allowed, started: float = 0.0) -> Verdict:
    """Turn the model's JSON into a Verdict, refusing anything off the closed set.

    Pure, so the branch the server acts on is testable without a network. A model that
    answers outside ``allowed`` is treated as not having answered: a caller branching on
    an unexpected token would fall through every case silently, which is worse than
    knowing the call failed.
    """
    dt = (time.time() - started) if started else 0.0
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        return Verdict(False, reason="model did not return JSON", latency_s=dt, raw=str(text)[:400])
    if not isinstance(d, dict):
        return Verdict(False, reason="model returned JSON that is not an object",
                       latency_s=dt, raw=str(text)[:400])
    ans = str(d.get("answer", "")).strip()
    if allowed is not None and ans not in allowed:
        return Verdict(False, reason=f"model answered {ans!r}, which is not one of "
                                     f"{sorted(allowed)}", latency_s=dt, raw=str(text)[:400])
    try:
        conf = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return Verdict(True, answer=ans, reason=str(d.get("reason", "")).strip()[:300],
                   confidence=min(max(conf, 0.0), 1.0), latency_s=dt, raw=str(text)[:400])


#: Grounding coordinates come back on this scale, not in pixels: it is the convention
#: these models are trained to emit, and asking for raw pixels instead measurably
#: degrades the boxes. The caller converts using the frame it actually sent.
BOX_SCALE = 1000.0

#: Reject a box thinner than this fraction of the frame in either axis. A degenerate
#: sliver is the shape a hallucinated box takes when there is nothing to point at, and
#: it converts to a centre pixel that looks perfectly reasonable.
MIN_BOX_FRAC = 0.01

#: ...and one larger than this. A box covering most of the frame is the model saying
#: "somewhere in here", which carries no more information than not answering.
MAX_BOX_FRAC = 0.90


@dataclass
class Fix:
    """Where the model says the object is, in the pixel frame that was sent to it.

    Deliberately NOT a Verdict. A verdict is a classification the caller branches on; a
    fix is a measurement the caller steers by, and the two want different guards — a
    fix has to survive being wrong by a plausible-looking amount, which a closed answer
    set cannot express.

    NOTE what is absent: any distance. The model is never asked how far away anything
    is. Range is where it is least reliable and where the rig's own geometry is already
    correct given a right pixel — so it supplies the pixel and nothing else.
    """

    ok: bool
    uv: tuple = (0.0, 0.0)
    bbox_xyxy: tuple = (0.0, 0.0, 0.0, 0.0)
    clipped: bool = False
    confidence: float = 0.0
    reason: str = ""
    latency_s: float = 0.0
    raw: str = field(default="", repr=False)

    def describe(self) -> str:
        if not self.ok:
            return f"vlm-locate: no fix ({self.reason})"
        return (f"vlm-locate: ({self.uv[0]:.0f},{self.uv[1]:.0f})px "
                f"conf {self.confidence:.0%}{' CLIPPED' if self.clipped else ''} "
                f"[{self.latency_s:.1f}s]")


def parse_fix(text: str, width: int, height: int, started: float = 0.0,
              min_confidence: float = 0.4) -> Fix:
    """Model JSON -> a pixel Fix, refusing anything that cannot be steered by.

    Pure, so every rejection path is testable without a network — which matters more
    here than for a verdict, because a bad fix does not fail loudly. It moves the arm
    somewhere wrong and the run looks normal until the jaws close on air.
    """
    dt = (time.time() - started) if started else 0.0
    def no(reason):
        return Fix(False, reason=reason, latency_s=dt, raw=str(text)[:400])

    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        return no("model did not return JSON")
    if not isinstance(d, dict):
        return no("model returned JSON that is not an object")
    if not d.get("found"):
        return no(str(d.get("reason", "")).strip()[:200] or "model reports no such object")

    box = d.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return no(f"box is not four numbers: {box!r}")
    try:
        ymin, xmin, ymax, xmax = (float(v) for v in box)
    except (TypeError, ValueError):
        return no(f"box is not numeric: {box!r}")
    if not all(-1.0 <= v <= BOX_SCALE + 1.0 for v in (ymin, xmin, ymax, xmax)):
        return no(f"box outside 0-{BOX_SCALE:.0f}: {box!r}")
    if xmax <= xmin or ymax <= ymin:
        return no(f"box is inverted or empty: {box!r}")

    try:
        conf = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(max(conf, 0.0), 1.0)
    if conf < min_confidence:
        return no(f"confidence {conf:.0%} below {min_confidence:.0%}")

    w, h = float(width), float(height)
    x1, x2 = xmin / BOX_SCALE * w, xmax / BOX_SCALE * w
    y1, y2 = ymin / BOX_SCALE * h, ymax / BOX_SCALE * h
    fw, fh = (x2 - x1) / max(w, 1.0), (y2 - y1) / max(h, 1.0)
    if fw < MIN_BOX_FRAC or fh < MIN_BOX_FRAC:
        return no(f"box is a sliver ({fw:.1%}x{fh:.1%} of frame)")
    if fw > MAX_BOX_FRAC and fh > MAX_BOX_FRAC:
        return no(f"box covers the frame ({fw:.0%}x{fh:.0%}) — no information")

    # Touching an edge means the centre is biased inward, and any bearing taken from it
    # with it. The caller may still servo on it; it must not RANGE from it.
    edge = 1.5
    clipped = (x1 <= edge or y1 <= edge or x2 >= w - edge or y2 >= h - edge)
    return Fix(True, uv=(0.5 * (x1 + x2), 0.5 * (y1 + y2)),
               bbox_xyxy=(x1, y1, x2, y2), clipped=clipped, confidence=conf,
               reason=str(d.get("reason", "")).strip()[:200], latency_s=dt,
               raw=str(text)[:400])

_LOCATE_PROMPT = """You are looking at a frame from a camera mounted on a robot \
gripper. Find the {label} and give its bounding box.

Report the box as [ymin, xmin, ymax, xmax] on a 0-1000 grid over this image. Box the \
{label} ITSELF, tightly — not the shadow it casts, not the gripper finger in front of \
it, and not a group of nearby objects together. If part of it is cut off by an edge of \
the frame, box the part you can see and say so in the reason.

If there is no {label} in this picture, set found=false and say what you see instead. \
Do not guess a location to be helpful: a confident box around the wrong thing sends the \
robot to the wrong place, which is worse than saying you cannot see it. Set confidence \
to how sure you are that the box is on the {label}."""


_MISS_PROMPT = """You are looking at a frame from a camera mounted on a robot gripper \
that is trying to pick up a {label}. The robot's detector just failed to find the \
{label} in this frame and the grasp was abandoned.

Say why, choosing exactly one answer:
{options}

Judge only what is in the picture. "visible" means the {label} is fully in frame, \
unobstructed and sharp enough to detect — if you pick that, the detector is at fault. \
If part of the {label} touches an edge of the frame, say which edge it is cut off by. \
Give a one-sentence reason describing where in the frame the object actually is."""

_GRASP_PROMPT = """You are looking at a frame from a camera on a robot gripper, taken \
just after it closed its jaws and lifted, trying to hold a {label}.

Is the {label} actually held between the jaws? Answer "holding" or "empty", or \
"unsure" if the jaws or the object are not clearly visible. Give a one-sentence reason."""

_SAME_PROMPT = """These are two crops from a robot's camera, taken seconds apart while \
its arm moved closer to a {label} it is trying to pick up.

Are they the same physical object? Answer "same" or "different", or "unsure" if either \
crop is too unclear to tell. The viewpoint and scale change between the two, so judge by \
what the object IS, not by where it appears. Give a one-sentence reason."""


class GeminiVision:
    """A small, fail-soft client for the three questions above.

    Thread-safe enough for this server's use: the SDK client is built once under a lock
    and the calls themselves are stateless.
    """

    def __init__(self, *, model: str = DEFAULT_MODEL, timeout_s: float = DEFAULT_TIMEOUT_S,
                 api_key: str = "", log=None):
        self.model = model
        self.timeout_s = max(float(timeout_s), MIN_TIMEOUT_S)
        # .strip() matters: a key of whitespace is falsy to a human and truthy to
        # Python, and an unstripped one sailed past the no-key guard and made a
        # live request that came back 403. An unusable key must fail here, before
        # the network, or 'not configured' looks identical to 'misconfigured'.
        self._key = (api_key or load_api_key() or "").strip()
        self._log = log or (lambda _m: None)
        self._client = None
        self._lock = threading.Lock()
        self._dead = ""              # non-empty once we know calls cannot work
        self.calls = 0
        self.failures = 0

    # --- plumbing ---------------------------------------------------------------
    def _get_client(self):
        if self._dead:
            return None
        with self._lock:
            if self._client is not None:
                return self._client
            if not self._key:
                self._dead = "no GOOGLE_API_KEY in the environment or .env.local"
                return None
            try:
                from google import genai
                from google.genai import types
                self._client = genai.Client(
                    api_key=self._key,
                    http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
                )
            except Exception as e:
                self._dead = f"{type(e).__name__}: {e}"
                return None
            return self._client

    def _ask(self, prompt: str, images, allowed) -> Verdict:
        """One constrained call. Never raises."""
        client = self._get_client()
        if client is None:
            return Verdict(False, reason=self._dead or "client unavailable")
        started = time.time()
        self.calls += 1
        try:
            from google.genai import types
            parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in images]
            resp = client.models.generate_content(
                model=self.model,
                contents=[*parts, prompt],
                config=types.GenerateContentConfig(
                    # A closed answer set plus a sentence of justification. The schema is
                    # what keeps this parseable; without it the model writes prose and
                    # the caller has to guess which branch that was.
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "answer": {"type": "STRING", "enum": sorted(allowed)},
                            "reason": {"type": "STRING"},
                            "confidence": {"type": "NUMBER"},
                        },
                        "required": ["answer", "reason", "confidence"],
                    },
                    temperature=0.0,
                    # NO thinking_config here. Setting thinking_budget=0 to cut latency
                    # is rejected outright by the flash-lite models this runs on —
                    # "400 INVALID_ARGUMENT" with no field named — and the same request
                    # succeeds the moment it is dropped. Isolated by bisecting the
                    # config: plain, +json, +schema and +temperature all pass.
                ),
            )
            v = parse_verdict(resp.text, allowed, started)
        except Exception as e:
            v = Verdict(False, reason=f"{type(e).__name__}: {e}",
                        latency_s=time.time() - started)
        if not v.ok:
            self.failures += 1
        return v

    @staticmethod
    def _jpeg(rgb, quality: int = 80) -> bytes:
        """Encode a frame. **RGB in** — this server's frames are RGB throughout.

        The channel swap is not cosmetic here. ``cv2.imencode`` writes its input as if
        it were BGR, so handing it an RGB array silently exchanges red and blue in the
        picture the model sees, leaving green untouched. This shipped that way, and the
        model duly reported "the gripper is grasping a BLUE object" three separate times
        while the jaws were on the red cube — including once as a confident 100% "the
        jaws are empty" that cleared a perfectly good carry, because it was asked about
        a red cube and had been shown a blue one.

        Nothing raised: swapped colours are a plausible picture, just not this one. Every
        other consumer of these frames already converts (``publish`` does
        ``rgb[:, :, ::-1]``); this one did not.
        """
        import cv2
        import numpy as np

        arr = np.asarray(rgb)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = np.ascontiguousarray(arr[:, :, ::-1])   # RGB -> BGR for the encoder
        ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()

    # --- the three questions ----------------------------------------------------
    def explain_miss(self, rgb, label: str) -> Verdict:
        """Why did the detector not find ``label`` in this frame?"""
        opts = "\n".join(f'- "{k}": {v}' for k, v in MISS_REASONS.items())
        return self._ask(_MISS_PROMPT.format(label=label, options=opts),
                         [self._jpeg(rgb)], set(MISS_REASONS))

    def locate_object(self, rgb, label: str, min_confidence: float = 0.4) -> Fix:
        """Where is the ``label`` in THIS frame, in pixels? No distance is asked for.

        This is the one call that steers rather than comments. It exists because the
        strict-HSV trackers and the open-vocabulary detector both fail in the same
        place: close in, at an odd angle, where the object's colour shifts out of band
        (a green cube reads teal from a steep view and is not found at all) or its box
        clips the frame edge. The centring servo then reports "object not in view",
        bails to the uncorrected staged estimate, and the jaws close on air.

        So it is asked only when the classical detectors have already returned nothing.
        Its alternative is not a working pick — it is a known miss.
        """
        import cv2  # noqa: F401  (only to read the frame's shape consistently)
        h, w = rgb.shape[:2]
        client = self._get_client()
        if client is None:
            return Fix(False, reason=self._dead or "client unavailable")
        started = time.time()
        self.calls += 1
        try:
            from google.genai import types
            resp = client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_bytes(data=self._jpeg(rgb), mime_type="image/jpeg"),
                          _LOCATE_PROMPT.format(label=label)],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "found": {"type": "BOOLEAN"},
                            # [ymin, xmin, ymax, xmax] on a 0-1000 grid: the grounding
                            # convention these models are trained to emit. Asking for
                            # raw pixels instead measurably degrades the boxes.
                            "box": {"type": "ARRAY", "items": {"type": "NUMBER"},
                                    "minItems": 4, "maxItems": 4},
                            "confidence": {"type": "NUMBER"},
                            "reason": {"type": "STRING"},
                        },
                        "required": ["found", "box", "confidence", "reason"],
                    },
                    temperature=0.0,
                ),
            )
            f = parse_fix(resp.text, w, h, started, min_confidence=min_confidence)
        except Exception as e:
            f = Fix(False, reason=f"{type(e).__name__}: {e}",
                    latency_s=time.time() - started)
        if not f.ok:
            self.failures += 1
        return f

    def verify_grasp(self, rgb, label: str) -> Verdict:
        """Is the object actually in the jaws right now?"""
        return self._ask(_GRASP_PROMPT.format(label=label), [self._jpeg(rgb)],
                         {"holding", "empty", "unsure"})

    def same_object(self, crop_a, crop_b, label: str) -> Verdict:
        """Are these two crops the same physical object?"""
        return self._ask(_SAME_PROMPT.format(label=label),
                         [self._jpeg(crop_a), self._jpeg(crop_b)],
                         {"same", "different", "unsure"})

    def ask_async(self, fn_name: str, *args, on_done=None) -> threading.Thread:
        """Run one of the above off the pick's thread.

        The grasp check happens after the lift, where a second of latency costs nothing
        but blocking the mission thread on a network call still would.
        """
        def _run():
            v = getattr(self, fn_name)(*args)
            self._log(v.describe())
            if on_done is not None:
                try:
                    on_done(v)
                except Exception:
                    pass
        t = threading.Thread(target=_run, name=f"gemini-{fn_name}", daemon=True)
        t.start()
        return t
