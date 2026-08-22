#!/usr/bin/env python3
"""
Remote Server — runs on YOUR machine (laptop/desktop with GPU).
Receives camera + depth + audio from the robot over WebSocket.
Runs YOLO, face recognition, Gemini Live API, tracking/follow logic.
Sends control commands back to the robot.

Usage:
    export GEMINI_API_KEY="your-key"
    python3 server.py
    python3 server.py --voice Charon --no-faces --port 8080
"""

import os
import sys
import asyncio
import threading
import base64
import time
import argparse
import json
import re
import math
import struct
import zlib
from collections import deque
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import cv2
# Make shared common/audio importable (pyaudio_compat) after the dir reorg
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common', 'audio'))
# Prefer compat on macOS to avoid "Could not import the PyAudio C module" and dylib issues
if sys.platform == 'darwin':
    import pyaudio_compat as pyaudio
else:
    try:
        import pyaudio
    except ImportError:
        import pyaudio_compat as pyaudio

from ultralytics import YOLO
try:
    import face_recognition
except (ImportError, OSError):
    face_recognition = None  # e.g. dlib wrong arch on Apple Silicon

from google import genai
from google.genai import types

# Audio config
SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHUNK = 1024

# Binary message type prefixes (robot -> server)
MSG_VIDEO = 0x01
MSG_DEPTH = 0x02
MSG_AUDIO_IN = 0x03
# server -> robot
MSG_AUDIO_OUT = 0x10

# Detection colors (BGR)
_COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255),
    (0, 255, 255), (128, 255, 0), (255, 0, 128), (128, 0, 255),
]

_NAME_PATTERNS = [
    re.compile(r"\bmy name is (\w+)", re.IGNORECASE),
    re.compile(r"\bi'm (\w+)", re.IGNORECASE),
    re.compile(r"\bi am (\w+)", re.IGNORECASE),
    re.compile(r"\bcall me (\w+)", re.IGNORECASE),
]

_CMD_PATTERNS = [
    (re.compile(r"\b(?:i'll |i will |let me |okay,? |ok,? )?(?:follow|following)\b(?:\s+(?:you|him|her|them|that person|(\w+)))?", re.IGNORECASE), "follow"),
    (re.compile(r"\b(?:i'll |i will |let me )?stop(?:ping)?\b(?:\s+(?:follow|track|mov))?", re.IGNORECASE), "stop"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:go(?:ing)?|walk(?:ing)?|head(?:ing)?|mov(?:e|ing))\s+(?:to(?:ward)?|over to)\s+(?:(\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|ft|feet)\s+(?:from|away from|near|of)\s+)?(?:the\s+|that\s+)?(\w+)", re.IGNORECASE), "go_to"),
    (re.compile(r"\b(?:i'll |i will |let me |here'?s? |okay,? |ok,? )?(?:do |doing |start )?(?:a |the )?(?:dance|dancing)\b(?:\s+(?:the\s+)?(\w+))?", re.IGNORECASE), "dance"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:wave|waving)\b", re.IGNORECASE), "wave"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:handshake|shake hands?|shaking hands?)\b", re.IGNORECASE), "handshake"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+(?:to\s+(?:the\s+)?)?left\b", re.IGNORECASE), "look_left"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+(?:to\s+(?:the\s+)?)?right\b", re.IGNORECASE), "look_right"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+up\b", re.IGNORECASE), "look_up"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+down\b", re.IGNORECASE), "look_down"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+(?:center|straight|forward|ahead)\b", re.IGNORECASE), "look_center"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:look(?:ing)? at|track(?:ing)?|watch(?:ing)?)\b(?:\s+(?:the\s+)?(\w+))?", re.IGNORECASE), "track"),
    (re.compile(r"\b(?:i'll |let me )?turn(?:ing)?\s+(?:to\s+(?:the\s+)?)?left\b", re.IGNORECASE), "turn_left"),
    (re.compile(r"\b(?:i'll |let me )?turn(?:ing)?\s+(?:to\s+(?:the\s+)?)?right\b", re.IGNORECASE), "turn_right"),
    (re.compile(r"\b(?:i'll |let me )?turn(?:ing)?\s+around\b", re.IGNORECASE), "turn_around"),
    (re.compile(r"\b(?:i'll |let me )?(?:walk(?:ing)?|mov(?:e|ing))\s+forward\b", re.IGNORECASE), "forward"),
    (re.compile(r"\b(?:i'll |let me )?(?:walk(?:ing)?|mov(?:e|ing))\s+backward\b", re.IGNORECASE), "backward"),
    (re.compile(r"\b(?:com(?:e|ing)\s+closer|approach(?:ing)?)\b", re.IGNORECASE), "approach"),
    (re.compile(r"\b(?:back(?:ing)?\s+up|step(?:ping)?\s+back|mov(?:e|ing)\s+back)\b", re.IGNORECASE), "back_up"),
    (re.compile(r"\b(?:i'll |let me )?(?:straf(?:e|ing)|sidestep(?:ping)?|mov(?:e|ing)\s+sideways)\s+(?:to\s+(?:the\s+)?)?left\b", re.IGNORECASE), "strafe_left"),
    (re.compile(r"\b(?:i'll |let me )?(?:straf(?:e|ing)|sidestep(?:ping)?|mov(?:e|ing)\s+sideways)\s+(?:to\s+(?:the\s+)?)?right\b", re.IGNORECASE), "strafe_right"),
    (re.compile(r"\b(?:i'll |let me |okay,? |here'?s? (?:a )?)?dab(?:bing)?\b", re.IGNORECASE), "dab"),
    (re.compile(r"\b(?:i'll |let me |okay,? |here'?s? (?:a )?)?flex(?:ing)?\b", re.IGNORECASE), "flex"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:do )?(?:the )?new\s*year(?:'?s?)?\s*dance\b", re.IGNORECASE), "dance_newyear"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:do )?(?:the )?nezha\s*dance\b", re.IGNORECASE), "dance_nezha"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:do )?(?:the )?future\s*dance\b", re.IGNORECASE), "dance_future"),
    (re.compile(r"\b(?:kick(?:ing)?|boxing\s*kick)\b", re.IGNORECASE), "dance_kick"),
    (re.compile(r"\b(?:moonwalk(?:ing)?)\b", re.IGNORECASE), "dance_moonwalk"),
    (re.compile(r"\b(?:michael\s*jackson)\b", re.IGNORECASE), "dance_michael jackson"),
    (re.compile(r"\b(?:roundhouse(?:\s*kick)?)\b", re.IGNORECASE), "dance_roundhouse"),
    (re.compile(r"\b(?:salsa|arabic\s*dance)\b", re.IGNORECASE), "dance_salsa"),
    (re.compile(r"\b(?:ultraman)\b", re.IGNORECASE), "dance_ultraman"),
    (re.compile(r"\b(?:respect)\b", re.IGNORECASE), "dance_respect"),
    (re.compile(r"\b(?:celebrat(?:e|ing|ion)|cheer(?:ing)?)\b", re.IGNORECASE), "dance_celebrate"),
    (re.compile(r"\b(?:lucky\s*cat)\b", re.IGNORECASE), "dance_luckycat"),
    (re.compile(r"\b(?:macarena)\b", re.IGNORECASE), "dance_macarena"),
    (re.compile(r"\b(?:twist(?:ing)?)\b", re.IGNORECASE), "dance_twist"),
    (re.compile(r"\b(?:take a |do a )?bow(?:ing)?\b", re.IGNORECASE), "dance_bow"),
    (re.compile(r"\b(?:chicken\s*dance|do(?:ing)?\s+(?:the\s+)?chicken)\b", re.IGNORECASE), "dance_chicken"),
    (re.compile(r"\b(?:disco)\b", re.IGNORECASE), "dance_disco"),
    (re.compile(r"\b(?:karate|kung\s*fu)\b", re.IGNORECASE), "dance_karate"),
    (re.compile(r"\b(?:nod(?:ding)?)\b", re.IGNORECASE), "nod"),
    (re.compile(r"\b(?:shak(?:e|ing)\s+(?:my\s+)?head)\b", re.IGNORECASE), "head_shake"),
]


def _color_for_class(cls_id):
    return _COLORS[cls_id % len(_COLORS)]


# ── Face Cache ───────────────────────────────────────────────────────────────

FACE_CACHE_DIR = os.path.expanduser('~/.face_cache')
FACE_CACHE_FILE = os.path.join(FACE_CACHE_DIR, 'known_faces.json')


class FaceCache:
    def __init__(self, tolerance=0.6):
        self.tolerance = tolerance
        self.entries = []
        self._lock = threading.Lock()
        os.makedirs(FACE_CACHE_DIR, exist_ok=True)
        self._load()

    def _load(self):
        if not os.path.exists(FACE_CACHE_FILE):
            return
        try:
            with open(FACE_CACHE_FILE) as f:
                data = json.load(f)
            for entry in data:
                self.entries.append({
                    'name': entry['name'],
                    'encoding': np.array(entry['encoding'], dtype=np.float64),
                    'saved_at': entry.get('saved_at', ''),
                })
            print(f"Face cache: loaded {len(self.entries)} known face(s)")
        except Exception as e:
            print(f"Warning: failed to load face cache: {e}")

    def _persist(self):
        try:
            data = [
                {'name': e['name'], 'encoding': e['encoding'].tolist(), 'saved_at': e['saved_at']}
                for e in self.entries
            ]
            with open(FACE_CACHE_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: failed to save face cache: {e}")

    def recognize(self, encoding):
        if face_recognition is None:
            return None
        with self._lock:
            if not self.entries:
                return None
            known = [e['encoding'] for e in self.entries]
            distances = face_recognition.face_distance(known, encoding)
            best_idx = int(np.argmin(distances))
            if distances[best_idx] <= self.tolerance:
                return self.entries[best_idx]['name']
            return None

    def save_face(self, name, encoding):
        with self._lock:
            for e in self.entries:
                if e['name'].lower() == name.lower():
                    e['encoding'] = encoding
                    e['saved_at'] = datetime.now().isoformat()
                    self._persist()
                    return
            self.entries.append({
                'name': name, 'encoding': encoding,
                'saved_at': datetime.now().isoformat(),
            })
            self._persist()
        print(f"Face cache: saved '{name}'")

    def delete_face(self, name):
        with self._lock:
            self.entries = [e for e in self.entries if e['name'].lower() != name.lower()]
            self._persist()

    def list_known(self):
        with self._lock:
            return [{'name': e['name'], 'saved_at': e['saved_at']} for e in self.entries]


# ── Frame Processor (replaces CameraDetectionNode — no ROS2 needed) ─────────


class FrameProcessor:
    """Receives frames from robot WebSocket, runs YOLO + face recognition."""

    def __init__(self, model_path='yolov8n.pt', confidence=0.5,
                 face_cache=None, enable_faces=True):
        print(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.enable_faces = enable_faces
        self.face_cache = face_cache

        self._unknown_faces = {}
        self._next_unknown_id = 1
        self._last_face_time = 0.0
        self._face_interval = 0.5
        self._cached_face_results = []

        self._lock = threading.Lock()
        self.latest_frame = None       # annotated frame (for web UI)
        self.latest_detections = []
        self._raw_frame = None         # original decoded frame
        self._depth_map = None         # uint16 depth
        self._frame_shape = None       # (h, w)
        self._fps = 0.0
        self._fps_counter = 0
        self._fps_time = time.time()

        self._detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._pending_frame = None
        self._detect_event = threading.Event()
        self._detect_thread.start()

    def on_video_frame(self, jpeg_bytes):
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            self._raw_frame = frame
            self._frame_shape = frame.shape[:2]
            self._pending_frame = frame
            self._detect_event.set()

    def on_depth_frame(self, data):
        try:
            w, h = struct.unpack('<HH', data[:4])
            raw = zlib.decompress(data[4:])
            depth = np.frombuffer(raw, dtype=np.uint16).reshape((h, w))
            self._depth_map = depth
        except Exception as e:
            print(f"Depth decode error: {e}")

    def _detect_loop(self):
        while True:
            self._detect_event.wait()
            self._detect_event.clear()
            frame = self._pending_frame
            if frame is None:
                continue
            try:
                self._run_detection(frame)
            except Exception as e:
                print(f"Detection error: {e}")

    def _get_depth_at(self, x, y, window=5):
        depth_map = self._depth_map
        if depth_map is None:
            return None
        dh, dw = depth_map.shape
        fh, fw = self._frame_shape or (dh * 2, dw * 2)
        sx = x * dw / fw
        sy = y * dh / fh
        ix, iy = int(sx), int(sy)
        half = window // 2
        y1, y2 = max(0, iy - half), min(dh, iy + half + 1)
        x1, x2 = max(0, ix - half), min(dw, ix + half + 1)
        patch = depth_map[y1:y2, x1:x2].astype(np.float32)
        valid = patch[(patch > 0) & (patch < 65535)]
        if len(valid) == 0:
            return None
        return float(np.median(valid)) / 1000.0

    def _run_detection(self, frame):
        results = self.model(frame, conf=self.confidence, verbose=False)
        annotated = frame.copy()
        detections = []
        depth_available = self._depth_map is not None
        has_persons = False

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                if cls_name == 'person':
                    has_persons = True
                cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
                distance_m = self._get_depth_at(cx, cy) if depth_available else None
                detections.append({
                    'class': cls_name, 'confidence': round(float(conf), 2),
                    'distance_m': round(float(distance_m), 2) if distance_m else None,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': [int(cx), int(cy)],
                    'name': None, 'unknown_id': None,
                })

        now = time.time()
        if self.enable_faces and has_persons and now - self._last_face_time >= self._face_interval:
            self._last_face_time = now
            self._cached_face_results = self._run_face_recognition(frame)

        for fr in self._cached_face_results:
            matched_det = self._match_face_to_person(fr, detections)
            if matched_det:
                matched_det['name'] = fr['name']
                matched_det['unknown_id'] = fr.get('unknown_id')
            top, right, bottom, left = fr['face_loc']
            is_known = fr['unknown_id'] is None
            face_color = (0, 255, 255) if is_known else (0, 165, 255)
            cv2.rectangle(annotated, (left, top), (right, bottom), face_color, 2)
            name = fr['name']
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (left, bottom), (left + tw + 4, bottom + th + 8), face_color, -1)
            cv2.putText(annotated, name, (left + 2, bottom + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class']
            conf = det['confidence']
            distance_m = det['distance_m']
            cls_id = list(self.model.names.values()).index(cls_name) if cls_name in self.model.names.values() else 0
            color = _color_for_class(cls_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.0%}"
            if det['name'] and cls_name == 'person':
                label = f"{det['name']} {conf:.0%}"
            if distance_m is not None:
                label += f" {distance_m:.1f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        self._fps_counter += 1
        if now - self._fps_time >= 1.0:
            self._fps = self._fps_counter / (now - self._fps_time)
            self._fps_counter = 0
            self._fps_time = now

        faces_str = f"Faces: {len(self._cached_face_results)}" if self.enable_faces else "Faces: off"
        depth_str = "Depth: ON" if depth_available else "Depth: waiting..."
        status = f"FPS: {self._fps:.0f} | Objects: {len(detections)} | {faces_str} | {depth_str}"
        cv2.putText(annotated, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        with self._lock:
            self.latest_frame = annotated
            self.latest_detections = detections

    def _run_face_recognition(self, frame):
        if face_recognition is None:
            return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb, model='cnn')
        if not face_locs:
            return []
        face_encs = face_recognition.face_encodings(rgb, face_locs, model='small')
        results = []
        for loc, enc in zip(face_locs, face_encs):
            top, right, bottom, left = loc
            name = self.face_cache.recognize(enc) if self.face_cache else None
            unknown_id = None
            if name is None:
                unknown_id = self._get_or_assign_unknown_id(enc)
                name = f"Unknown #{unknown_id}"
            results.append({
                'name': name, 'unknown_id': unknown_id,
                'face_loc': (top, right, bottom, left), 'encoding': enc,
            })
        return results

    def _get_or_assign_unknown_id(self, encoding):
        if face_recognition is None:
            return 0
        best_dist, best_id = 999.0, None
        for uid, enc in self._unknown_faces.items():
            dist = float(face_recognition.face_distance([enc], encoding)[0])
            if dist < best_dist:
                best_dist, best_id = dist, uid
        if best_id is not None and best_dist < 0.5:
            self._unknown_faces[best_id] = encoding
            return best_id
        uid = self._next_unknown_id
        self._next_unknown_id += 1
        self._unknown_faces[uid] = encoding
        return uid

    def _match_face_to_person(self, face_result, detections):
        top, right, bottom, left = face_result['face_loc']
        face_cx = (left + right) // 2
        face_cy = (top + bottom) // 2
        for det in detections:
            if det['class'] != 'person':
                continue
            bx1, by1, bx2, by2 = det['bbox']
            if bx1 <= face_cx <= bx2 and by1 <= face_cy <= by2:
                return det
        return None

    def save_unknown_face(self, unknown_id, name):
        enc = self._unknown_faces.get(unknown_id)
        if enc is None:
            return False
        self.face_cache.save_face(name, enc)
        del self._unknown_faces[unknown_id]
        self._cached_face_results = []
        self._last_face_time = 0
        return True

    def try_learn_name_from_transcript(self, text):
        if not self.enable_faces or not self._unknown_faces:
            return
        for pattern in _NAME_PATTERNS:
            match = pattern.search(text)
            if match:
                name = match.group(1).capitalize()
                latest_uid = max(self._unknown_faces.keys())
                if self.save_unknown_face(latest_uid, name):
                    print(f"Auto-learned face: '{name}' from speech")
                    add_transcript("System", f"Learned face: {name}")
                return

    def get_frame_b64jpeg(self, max_dim=640, quality=60):
        with self._lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode('utf-8')

    def get_detection_summary(self):
        with self._lock:
            dets = list(self.latest_detections)
        if not dets:
            return ""
        lines = []
        for d in dets:
            dist = f"{d['distance_m']:.1f}m away" if d['distance_m'] else "unknown dist"
            name_str = f" ({d['name']})" if d.get('name') else ""
            lines.append(f"- {d['class']}{name_str} ({d['confidence']:.0%}, {dist})")
        return "Detected objects:\n" + "\n".join(lines)


# ── Remote Robot Controller ──────────────────────────────────────────────────


class RobotController:
    """Controls the remote robot by sending commands over WebSocket.
    Runs tracking/follow loops locally (has access to detections)."""

    def __init__(self):
        self._ws = None
        self._loop = None
        self.lock = threading.Lock()
        self.head_pitch = 0.0
        self.head_yaw = 0.0

        self.tracking_active = False
        self.tracking_target = None
        self.tracking_thread = None

        self.follow_active = False
        self.follow_target = None
        self.follow_thread = None
        self.follow_target_distance = 1.0

        self.move_active = False
        self.move_thread = None

        self.frame_processor = None

    def set_connection(self, ws, loop):
        self._ws = ws
        self._loop = loop

    def set_frame_processor(self, fp: FrameProcessor):
        self.frame_processor = fp

    def _send(self, cmd_dict):
        ws, loop = self._ws, self._loop
        if ws and loop:
            asyncio.run_coroutine_threadsafe(ws.send(json.dumps(cmd_dict)), loop)

    # ── Head control ─────────────────────────────────────────────────────

    def rotate_head(self, pitch, yaw):
        pitch = max(-0.5, min(1.0, pitch))
        yaw = max(-0.785, min(0.785, yaw))
        self.head_pitch, self.head_yaw = pitch, yaw
        self._send({'cmd': 'rotate_head', 'pitch': pitch, 'yaw': yaw})

    def nod(self):
        self._send({'cmd': 'nod'})

    def head_shake(self):
        self._send({'cmd': 'head_shake'})

    # ── Movement ─────────────────────────────────────────────────────────

    def _move(self, x, y, yaw):
        self._send({'cmd': 'move', 'x': x, 'y': y, 'yaw': yaw})

    def move_timed(self, x, y, yaw, duration):
        self.stop_movement()
        def _run():
            self.move_active = True
            start = time.time()
            while self.move_active and (time.time() - start) < duration:
                self._move(x, y, yaw)
                time.sleep(0.05)
            self._move(0, 0, 0)
            self.move_active = False
        self.move_thread = threading.Thread(target=_run, daemon=True)
        self.move_thread.start()

    def stop_movement(self):
        self.move_active = False
        if self.move_thread and self.move_thread.is_alive():
            self.move_thread.join(timeout=1.0)
        self._move(0, 0, 0)

    def turn_around(self):
        self.move_timed(0, 0, 0.5, 3.0)

    def approach(self):
        self.move_timed(0.4, 0, 0, 2.0)

    def back_up(self):
        self.move_timed(-0.2, 0, 0, 1.5)

    def turn_left(self):
        self.move_timed(0, 0, 0.5, 1.5)

    def turn_right(self):
        self.move_timed(0, 0, -0.5, 1.5)

    def forward(self):
        self.move_timed(0.5, 0, 0, 2.0)

    def backward(self):
        self.move_timed(-0.3, 0, 0, 2.0)

    def strafe_left(self):
        self.move_timed(0, 0.3, 0, 1.5)

    def strafe_right(self):
        self.move_timed(0, -0.3, 0, 1.5)

    # ── Dances / gestures (delegated to robot client) ────────────────────

    def do_dance(self, dance_name=None):
        self._send({'cmd': 'dance', 'name': dance_name or 'robot'})

    def do_wave(self):
        self._send({'cmd': 'wave'})

    def do_handshake(self):
        self._send({'cmd': 'handshake'})

    def do_dab(self):
        self._send({'cmd': 'dab'})

    def do_flex(self):
        self._send({'cmd': 'flex'})

    # ── Tracking ─────────────────────────────────────────────────────────

    def start_tracking(self, target=None):
        self.stop_tracking()
        self.tracking_active = True
        self.tracking_target = target
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        print(f"[Robot] Head tracking started: {target or 'closest person'}")

    def stop_tracking(self):
        self.tracking_active = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
            self.tracking_thread = None
        self._move(0, 0, 0)

    def _tracking_loop(self):
        YAW_BODY_TURN_THRESHOLD = 0.45
        BODY_TURN_SPEED = 0.35

        while self.tracking_active:
            if not self.frame_processor or self.frame_processor._raw_frame is None:
                time.sleep(0.1)
                continue

            det = self._find_target_detection()
            if det is None:
                if not self.follow_active:
                    self._move(0, 0, 0)
                time.sleep(0.1)
                continue

            shape = self.frame_processor._frame_shape
            if shape is None:
                time.sleep(0.1)
                continue

            h, w = shape
            cx, cy = det['center']
            err_x = (cx - w / 2) / (w / 2)
            err_y = (cy - h / 2) / (h / 2)

            kp_yaw, kp_pitch = 0.15, 0.1
            new_yaw = self.head_yaw - err_x * kp_yaw
            new_pitch = self.head_pitch + err_y * kp_pitch

            if abs(err_x) > 0.08 or abs(err_y) > 0.08:
                self.rotate_head(new_pitch, new_yaw)

            if not self.follow_active:
                if abs(self.head_yaw) > YAW_BODY_TURN_THRESHOLD:
                    body_rot = BODY_TURN_SPEED if self.head_yaw > 0 else -BODY_TURN_SPEED
                    self._move(0, 0, body_rot)
                else:
                    self._move(0, 0, 0)

            time.sleep(0.1)

    def _find_target_detection(self):
        if not self.frame_processor:
            return None
        with self.frame_processor._lock:
            dets = list(self.frame_processor.latest_detections)
        if not dets:
            return None

        target = self.tracking_target
        if target is None or target.lower() in ('person', 'people', 'someone', 'anyone'):
            persons = [d for d in dets if d['class'] == 'person']
            if not persons:
                return None
            with_dist = [p for p in persons if p.get('distance_m')]
            if with_dist:
                return min(with_dist, key=lambda p: p['distance_m'])
            return max(persons, key=lambda p: (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]))

        named = [d for d in dets if d.get('name') and target.lower() in d['name'].lower()]
        if named:
            return named[0]

        classed = [d for d in dets if d['class'].lower() == target.lower()]
        if classed:
            with_dist = [c for c in classed if c.get('distance_m')]
            if with_dist:
                return min(with_dist, key=lambda c: c['distance_m'])
            return max(classed, key=lambda c: (c['bbox'][2] - c['bbox'][0]) * (c['bbox'][3] - c['bbox'][1]))
        return None

    # ── Follow ───────────────────────────────────────────────────────────

    def start_follow(self, target=None):
        self.stop_follow()
        self.follow_active = True
        self.follow_target = target
        self.start_tracking(target or 'person')
        self.follow_thread = threading.Thread(target=self._follow_loop, daemon=True)
        self.follow_thread.start()
        print(f"[Robot] Following started: {target or 'closest person'}")

    def stop_follow(self):
        self.follow_active = False
        self.stop_tracking()
        if self.follow_thread:
            self.follow_thread.join(timeout=1.0)
            self.follow_thread = None
        self._move(0, 0, 0)

    def _follow_loop(self):
        OBSTACLE_DIST = 0.9
        OBSTACLE_EMERGENCY = 0.5
        STRAFE_SPEED = 0.25
        MAX_FWD = 0.5

        while self.follow_active:
            fp = self.frame_processor
            if not fp or fp._raw_frame is None:
                time.sleep(0.1)
                continue

            det = self._find_target_detection()
            if det is None:
                self._move(0, 0, 0)
                time.sleep(0.2)
                continue

            shape = fp._frame_shape
            if shape is None:
                time.sleep(0.1)
                continue

            h, w = shape
            cx = det['center'][0]
            distance = det.get('distance_m')
            target_id = id(det)

            err_x = (cx - w / 2) / (w / 2)
            rot_speed = -err_x * 0.8 if abs(err_x) > 0.10 else 0.0

            fwd_speed = 0.0
            if distance is not None:
                dist_error = distance - self.follow_target_distance
                if dist_error > 0.3:
                    fwd_speed = min(MAX_FWD, dist_error * 0.4)
                elif dist_error < -0.3:
                    fwd_speed = max(-0.2, dist_error * 0.2)
            else:
                bbox_w = det['bbox'][2] - det['bbox'][0]
                bbox_ratio = bbox_w / w
                if bbox_ratio < 0.15:
                    fwd_speed = 0.3
                elif bbox_ratio > 0.4:
                    fwd_speed = -0.1

            if abs(err_x) > 0.35:
                fwd_speed *= 0.2

            strafe_speed = 0.0
            obstacle_ahead = False

            depth_map = fp._depth_map
            if depth_map is not None and fwd_speed > 0:
                dh, dw = depth_map.shape
                strip_y1, strip_y2 = int(dh * 0.35), int(dh * 0.75)
                strip = depth_map[strip_y1:strip_y2, :]
                n_zones = 5
                zone_w = dw // n_zones
                zone_depths = []
                for i in range(n_zones):
                    col1 = i * zone_w
                    col2 = col1 + zone_w if i < n_zones - 1 else dw
                    zone = strip[:, col1:col2]
                    valid = zone[(zone > 0) & (zone < 65535)].astype(np.float32)
                    d = float(np.median(valid)) / 1000.0 if len(valid) > 20 else 10.0
                    zone_depths.append(d)

                center_d = zone_depths[2]
                near_left_d, near_right_d = zone_depths[1], zone_depths[3]
                far_left_d, far_right_d = zone_depths[0], zone_depths[4]
                target_d = distance if distance else 5.0

                if center_d < OBSTACLE_DIST and center_d < target_d - 0.3:
                    obstacle_ahead = True
                    left_clear = (far_left_d + near_left_d) / 2
                    right_clear = (far_right_d + near_right_d) / 2

                    if center_d < OBSTACLE_EMERGENCY:
                        fwd_speed = min(fwd_speed, 0.05)
                        if left_clear > right_clear:
                            strafe_speed, rot_speed = STRAFE_SPEED, max(rot_speed, 0.3)
                        else:
                            strafe_speed, rot_speed = -STRAFE_SPEED, min(rot_speed, -0.3)
                    else:
                        fwd_speed = min(fwd_speed, 0.15)
                        if left_clear > right_clear:
                            rot_speed, strafe_speed = max(rot_speed, 0.25), 0.15
                        else:
                            rot_speed, strafe_speed = min(rot_speed, -0.25), -0.15

                elif near_left_d < OBSTACLE_EMERGENCY:
                    strafe_speed = -0.15
                elif near_right_d < OBSTACLE_EMERGENCY:
                    strafe_speed = 0.15

            if fwd_speed > 0 and not obstacle_ahead:
                with fp._lock:
                    all_dets = list(fp.latest_detections)
                for obj in all_dets:
                    if id(obj) == target_id:
                        continue
                    obj_dist = obj.get('distance_m')
                    if obj_dist is None or obj_dist > OBSTACLE_DIST:
                        continue
                    obj_cx = obj['center'][0]
                    obj_err = (obj_cx - w / 2) / (w / 2)
                    if abs(obj_err) < 0.4:
                        fwd_speed = min(fwd_speed, 0.1)
                        if obj_err <= 0:
                            strafe_speed = max(strafe_speed, -0.15)
                        else:
                            strafe_speed = min(strafe_speed, 0.15)
                        break

            self._move(fwd_speed, strafe_speed, rot_speed)
            time.sleep(0.05)

    # ── Go to object ─────────────────────────────────────────────────────

    def go_to_object(self, target, stop_distance=0.5):
        self.stop_follow()
        self.follow_active = True
        self.follow_target = target
        self.start_tracking(target)
        self.follow_thread = threading.Thread(
            target=self._go_to_loop, args=(target, stop_distance), daemon=True
        )
        self.follow_thread.start()
        print(f"[Robot] Going to: {target} (stop {stop_distance:.1f}m away)")

    def _go_to_loop(self, target, stop_distance=0.5):
        TIMEOUT = 30.0
        OBSTACLE_DIST = 0.8
        MIN_FWD, MAX_FWD = 0.15, 0.5
        STEER_SPEED = 0.35
        LOST_PATIENCE = 3.0

        start = time.time()
        lost_since = None

        while self.follow_active and (time.time() - start) < TIMEOUT:
            fp = self.frame_processor
            if not fp or fp._raw_frame is None:
                time.sleep(0.1)
                continue

            det = self._find_target_detection()
            if det is None:
                if lost_since is None:
                    lost_since = time.time()
                self._move(0, 0, 0)
                if time.time() - lost_since > LOST_PATIENCE:
                    print(f"[Robot] Lost target {target}, giving up")
                    break
                time.sleep(0.2)
                continue
            lost_since = None

            shape = fp._frame_shape
            if shape is None:
                time.sleep(0.1)
                continue

            h, w = shape
            cx = det['center'][0]
            distance = det.get('distance_m')

            if distance is not None and distance <= stop_distance:
                print(f"[Robot] Arrived at {target} ({distance:.1f}m)")
                break

            bbox_w = det['bbox'][2] - det['bbox'][0]
            bbox_ratio = bbox_w / w
            bbox_threshold = max(0.30, 0.55 - stop_distance * 0.15)
            if distance is None and bbox_ratio >= bbox_threshold:
                print(f"[Robot] Arrived at {target} (bbox {bbox_ratio:.2f})")
                break

            err_x = (cx - w / 2) / (w / 2)
            rot_speed = -err_x * 0.5 if abs(err_x) > 0.10 else 0.0

            if distance is not None:
                remaining = distance - stop_distance
                if remaining <= 0:
                    break
                fwd_speed = min(MAX_FWD, max(MIN_FWD, remaining * 0.35))
            else:
                fwd_speed = 0.3

            depth_map = fp._depth_map
            if depth_map is not None:
                dh, dw = depth_map.shape
                strip_y1, strip_y2 = int(dh * 0.4), int(dh * 0.7)
                strip = depth_map[strip_y1:strip_y2, :]
                third = dw // 3

                def _zone_depth(s):
                    valid = s[(s > 0) & (s < 65535)].astype(np.float32)
                    return float(np.median(valid)) / 1000.0 if len(valid) > 20 else 10.0

                left_d = _zone_depth(strip[:, :third])
                center_d = _zone_depth(strip[:, third:2*third])
                right_d = _zone_depth(strip[:, 2*third:])

                target_dist = distance if distance else 5.0
                if center_d < OBSTACLE_DIST and center_d < target_dist - 0.5:
                    rot_speed = STEER_SPEED if left_d > right_d else -STEER_SPEED
                    fwd_speed = min(fwd_speed, 0.2)

            self._move(fwd_speed, 0.0, rot_speed)
            time.sleep(0.05)

        self._move(0, 0, 0)
        self.follow_active = False
        print(f"[Robot] Go-to complete: {target}")

    # ── Stop all ─────────────────────────────────────────────────────────

    def stop_all(self):
        self.stop_follow()
        self.stop_tracking()
        self.stop_movement()
        self.rotate_head(0.0, 0.0)
        print("[Robot] All stopped")

    def shutdown(self):
        self.stop_all()


# ── Command Dispatcher ───────────────────────────────────────────────────────


class CommandDispatcher:
    def __init__(self, robot: RobotController):
        self.robot = robot
        self._last_cmd_time = 0
        self._cmd_cooldown = 2.0

    def check_transcript(self, text):
        now = time.time()
        if now - self._last_cmd_time < self._cmd_cooldown:
            return None
        text_lower = text.lower().strip()
        for pattern, cmd_name in _CMD_PATTERNS:
            match = pattern.search(text_lower)
            if match:
                self._last_cmd_time = now
                self._execute(cmd_name, match)
                return cmd_name
        return None

    def _execute(self, cmd, match):
        if cmd == "go_to":
            groups = match.groups()
            dist_str, go_target = groups[0], groups[1]
            stop_dist = float(dist_str) if dist_str else 0.5
            target = go_target or "person"
            print(f"[CMD] go_to {target} ({stop_dist}m)")
            add_transcript("Action", f"go_to {target} ({stop_dist}m)")
            self.robot.go_to_object(target, stop_distance=stop_dist)
            return

        target = None
        for g in match.groups():
            if g:
                target = g.strip()
                break

        print(f"[CMD] {cmd}" + (f" ({target})" if target else ""))
        add_transcript("Action", f"{cmd}" + (f" ({target})" if target else ""))

        actions = {
            "follow": lambda: self.robot.start_follow(target),
            "stop": self.robot.stop_all,
            "dance": lambda: self.robot.do_dance(target),
            "wave": self.robot.do_wave,
            "handshake": self.robot.do_handshake,
            "dab": self.robot.do_dab,
            "flex": self.robot.do_flex,
            "track": lambda: self.robot.start_tracking(target),
            "look_left": lambda: self.robot.rotate_head(0.0, 0.5),
            "look_right": lambda: self.robot.rotate_head(0.0, -0.5),
            "look_up": lambda: self.robot.rotate_head(-0.3, 0.0),
            "look_down": lambda: self.robot.rotate_head(0.5, 0.0),
            "look_center": lambda: self.robot.rotate_head(0.0, 0.0),
            "turn_left": self.robot.turn_left,
            "turn_right": self.robot.turn_right,
            "turn_around": self.robot.turn_around,
            "forward": self.robot.forward,
            "backward": self.robot.backward,
            "strafe_left": self.robot.strafe_left,
            "strafe_right": self.robot.strafe_right,
            "approach": self.robot.approach,
            "back_up": self.robot.back_up,
            "nod": self.robot.nod,
            "head_shake": self.robot.head_shake,
        }

        if cmd in actions:
            actions[cmd]()
        elif cmd.startswith("dance_"):
            self.robot.do_dance(cmd.replace("dance_", ""))


# ── Transcript ───────────────────────────────────────────────────────────────

transcript = deque(maxlen=200)
transcript_lock = threading.Lock()
_frame_processor_ref = None
_cmd_dispatcher_ref = None
_session_ref = None
_event_loop_ref = None


def add_transcript(role, text):
    with transcript_lock:
        transcript.append({"role": role, "text": text, "ts": time.time()})
    if role == "You" and _frame_processor_ref:
        _frame_processor_ref.try_learn_name_from_transcript(text)


def get_transcript():
    with transcript_lock:
        return list(transcript)


def send_text_to_gemini(text):
    session, loop = _session_ref, _event_loop_ref
    if not session or not loop:
        print("[Chat] No active Gemini session")
        return False

    async def _send():
        try:
            await session.send_client_content(
                turns=[types.Content(role="user", parts=[types.Part(text=text)])],
                turn_complete=True,
            )
        except Exception as e:
            print(f"[Chat] Send error: {e}")

    asyncio.run_coroutine_threadsafe(_send(), loop)
    return True


# ── Web UI ───────────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>Gemini Robot Control (Remote)</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#111; color:#eee; font-family:system-ui,sans-serif; display:flex; height:100vh; }
  #left { flex:1; display:flex; align-items:center; justify-content:center; background:#000; min-width:0; }
  #left img { max-width:100%; max-height:100%; object-fit:contain; }
  #right { width:420px; display:flex; flex-direction:column; border-left:1px solid #333; }
  #header { padding:12px 16px; border-bottom:1px solid #333; font-size:14px; color:#888; }
  #header span { color:#4CAF50; font-weight:bold; }
  #controls { padding:8px 16px; border-bottom:1px solid #333; }
  #controls h3 { font-size:12px; color:#888; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px; }
  .btn-row { display:flex; gap:6px; margin-bottom:6px; flex-wrap:wrap; }
  .ctrl-btn { padding:6px 12px; border:none; border-radius:6px; cursor:pointer; font-size:12px; font-weight:bold; transition:transform 0.1s; }
  .ctrl-btn:hover { transform:scale(1.05); }
  .ctrl-btn:active { transform:scale(0.95); }
  .btn-follow { background:#4CAF50; color:#000; }
  .btn-dance { background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; }
  .btn-action { background:#2196F3; color:#fff; }
  .btn-head { background:#43e97b; color:#000; }
  .btn-stop { background:#f44336; color:#fff; }
  .btn-mode { background:#FF9800; color:#000; }
  #robot-status { padding:6px 16px; border-bottom:1px solid #333; font-size:12px; }
  .connected { color:#4CAF50; } .disconnected { color:#f44336; }
  #detections { padding:8px 16px; border-bottom:1px solid #333; max-height:150px; overflow-y:auto; }
  #detections h3 { font-size:12px; color:#888; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px; }
  .det { padding:4px 8px; margin:3px 0; background:#1a2a1a; border-radius:4px; font-size:13px; display:flex; justify-content:space-between; }
  .det .cls { color:#4CAF50; } .det .name { color:#00BCD4; font-weight:bold; }
  .det .unknown { color:#FF9800; } .det .dist { color:#2196F3; font-weight:bold; }
  .det .conf { color:#666; font-size:11px; }
  .det .name-input { width:80px; padding:2px 4px; background:#222; border:1px solid #555; color:#eee; border-radius:3px; font-size:12px; }
  .det .save-btn { padding:2px 8px; background:#4CAF50; color:#000; border:none; border-radius:3px; font-size:11px; cursor:pointer; margin-left:4px; }
  #known-faces { padding:8px 16px; border-bottom:1px solid #333; max-height:100px; overflow-y:auto; }
  #known-faces h3 { font-size:12px; color:#888; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px; }
  .known { padding:3px 8px; margin:2px 0; background:#1a2a2a; border-radius:4px; font-size:13px; display:flex; justify-content:space-between; }
  .known .kname { color:#00BCD4; }
  .known .del-btn { padding:1px 6px; background:#c62828; color:#fff; border:none; border-radius:3px; font-size:11px; cursor:pointer; }
  #chat { flex:1; overflow-y:auto; padding:12px 16px; display:flex; flex-direction:column; gap:8px; }
  .msg { padding:8px 12px; border-radius:8px; max-width:95%; font-size:14px; line-height:1.4; word-wrap:break-word; }
  .msg.you { background:#1a3a5c; align-self:flex-end; }
  .msg.robot { background:#2d2d2d; align-self:flex-start; }
  .msg.system { background:#2a2a1a; align-self:center; font-size:12px; color:#aaa; }
  .msg.action { background:#1a2a1a; align-self:center; font-size:12px; color:#4CAF50; border:1px solid #4CAF50; }
  .msg .role { font-size:11px; color:#888; margin-bottom:2px; }
  #chat-input { display:flex; padding:8px 12px; border-top:1px solid #333; gap:6px; }
  #msg-input { flex:1; padding:8px 12px; background:#1a1a1a; border:1px solid #444; color:#eee; border-radius:8px; font-size:14px; outline:none; }
  #msg-input:focus { border-color:#4CAF50; }
  #send-btn { padding:8px 16px; background:#4CAF50; color:#000; border:none; border-radius:8px; font-weight:bold; cursor:pointer; font-size:14px; }
  #send-btn:hover { background:#66BB6A; }
  #status { padding:8px 16px; border-top:1px solid #333; font-size:12px; color:#666; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#4CAF50; margin-right:6px; }
</style>
</head>
<body>
  <div id="left"><img id="feed" src="/frame" alt="Camera"></div>
  <div id="right">
    <div id="header"><span>Gemini Robot Control</span> &mdash; Remote Mode</div>
    <div id="robot-status"><span id="conn-dot" class="disconnected">&#9679;</span> <span id="conn-text">Robot: connecting...</span></div>
    <div id="controls">
      <h3>Robot Controls</h3>
      <div class="btn-row">
        <button class="ctrl-btn btn-follow" onclick="cmd('follow')">Follow Me</button>
        <button class="ctrl-btn btn-follow" onclick="cmd('track')">Track Person</button>
        <button class="ctrl-btn btn-follow" onclick="goToPrompt()">Go To...</button>
        <button class="ctrl-btn btn-stop" onclick="cmd('stop')">STOP ALL</button>
      </div>
      <div class="btn-row">
        <button class="ctrl-btn btn-dance" onclick="cmd('dance')">Dance</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_newyear')">New Year</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_nezha')">Nezha</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_future')">Future</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_moonwalk')">Moonwalk</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_michael jackson')">MJ Dance</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_kick')">Kick</button>
      </div>
      <div class="btn-row">
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_roundhouse')">Roundhouse</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_salsa')">Salsa</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_ultraman')">Ultraman</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_respect')">Respect</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_celebrate')">Celebrate</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_luckycat')">Lucky Cat</button>
      </div>
      <div class="btn-row">
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_macarena')">Macarena</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_twist')">Twist</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_disco')">Disco</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_chicken')">Chicken</button>
        <button class="ctrl-btn btn-dance" onclick="cmd('dance_bow')">Bow</button>
      </div>
      <div class="btn-row">
        <button class="ctrl-btn btn-action" onclick="cmd('wave')">Wave</button>
        <button class="ctrl-btn btn-action" onclick="cmd('handshake')">Handshake</button>
        <button class="ctrl-btn btn-action" onclick="cmd('dab')">Dab</button>
        <button class="ctrl-btn btn-action" onclick="cmd('flex')">Flex</button>
        <button class="ctrl-btn btn-action" onclick="cmd('nod')">Nod</button>
        <button class="ctrl-btn btn-action" onclick="cmd('head_shake')">Shake Head</button>
      </div>
      <div class="btn-row">
        <button class="ctrl-btn btn-head" onclick="cmd('look_up')">Look Up</button>
        <button class="ctrl-btn btn-head" onclick="cmd('look_down')">Look Down</button>
        <button class="ctrl-btn btn-head" onclick="cmd('look_left')">Look Left</button>
        <button class="ctrl-btn btn-head" onclick="cmd('look_right')">Look Right</button>
        <button class="ctrl-btn btn-head" onclick="cmd('look_center')">Center</button>
      </div>
      <div class="btn-row">
        <button class="ctrl-btn btn-mode" onclick="cmd('forward')">Forward<br><small>W</small></button>
        <button class="ctrl-btn btn-mode" onclick="cmd('backward')">Backward<br><small>S</small></button>
        <button class="ctrl-btn btn-mode" onclick="cmd('strafe_left')">Strafe L<br><small>A</small></button>
        <button class="ctrl-btn btn-mode" onclick="cmd('strafe_right')">Strafe R<br><small>D</small></button>
        <button class="ctrl-btn btn-mode" onclick="cmd('turn_left')">Turn L<br><small>Q</small></button>
        <button class="ctrl-btn btn-mode" onclick="cmd('turn_right')">Turn R<br><small>E</small></button>
        <button class="ctrl-btn btn-mode" onclick="cmd('turn_around')">Turn 180</button>
      </div>
    </div>
    <div id="detections"><h3>Detections</h3><div id="det-list">Waiting for robot...</div></div>
    <div id="known-faces"><h3>Known Faces</h3><div id="kf-list">None yet</div></div>
    <div id="chat"></div>
    <div id="chat-input">
      <input type="text" id="msg-input" placeholder="Type a message or question..." autocomplete="off">
      <button id="send-btn" onclick="sendChat()">Send</button>
    </div>
    <div id="status"><span class="dot"></span>Listening... speak commands or use buttons above</div>
  </div>
<script>
  const img = document.getElementById('feed');
  function refreshFrame() {
    const next = new Image();
    next.onload = () => { img.src = next.src; setTimeout(refreshFrame, 100); };
    next.onerror = () => { setTimeout(refreshFrame, 500); };
    next.src = '/frame?t=' + Date.now();
  }
  refreshFrame();

  async function cmd(action, target) {
    try {
      const payload = {action};
      if (target) payload.target = target;
      const r = await fetch('/cmd', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
      console.log('cmd:', await r.json());
    } catch(e) { console.error(e); }
  }

  function goToPrompt() {
    const target = prompt('Go to what? (e.g. person, chair, bottle, backpack, a name)');
    if (!target || !target.trim()) return;
    const distStr = prompt('Stop distance in meters? (default 0.5)', '0.5');
    const dist = parseFloat(distStr) || 0.5;
    fetch('/cmd', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'go_to', target:target.trim(), distance:dist})
    }).then(r=>r.json()).then(d=>console.log('go_to:',d)).catch(console.error);
  }

  async function sendChat() {
    const input = document.getElementById('msg-input');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    try { await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})}); } catch(e) { console.error(e); }
  }

  document.getElementById('msg-input').addEventListener('keydown', function(e) { if (e.key==='Enter') { e.preventDefault(); sendChat(); } });

  const wasdMap = {w:'forward', s:'backward', a:'strafe_left', d:'strafe_right', q:'turn_left', e:'turn_right'};
  document.addEventListener('keydown', function(e) {
    if (document.activeElement.tagName==='INPUT'||document.activeElement.tagName==='TEXTAREA') return;
    const action = wasdMap[e.key.toLowerCase()];
    if (action) { e.preventDefault(); cmd(action); }
  });

  const chat = document.getElementById('chat');
  let lastLen = 0;
  async function pollTranscript() {
    try {
      const r = await fetch('/transcript');
      const msgs = await r.json();
      if (msgs.length !== lastLen) {
        lastLen = msgs.length;
        chat.innerHTML = msgs.map(m => {
          const cls = m.role==='You'?'you':m.role==='Action'?'action':m.role==='System'?'system':'robot';
          return '<div class="msg '+cls+'"><div class="role">'+m.role+'</div>'+m.text+'</div>';
        }).join('');
        chat.scrollTop = chat.scrollHeight;
      }
    } catch(e) {}
    setTimeout(pollTranscript, 500);
  }
  pollTranscript();

  const detList = document.getElementById('det-list');
  async function pollDetections() {
    try {
      const r = await fetch('/detections');
      const dets = await r.json();
      if (dets.length === 0) {
        detList.innerHTML = '<div style="color:#666;font-size:12px;">No objects detected</div>';
      } else {
        detList.innerHTML = dets.map(d => {
          const dist = d.distance_m !== null ? d.distance_m.toFixed(1)+'m' : '?';
          let nameHtml = '';
          if (d.class==='person' && d.name) {
            if (d.unknown_id !== null && d.unknown_id !== undefined) {
              nameHtml = '<span class="unknown">'+d.name+'</span> '
                +'<input class="name-input" placeholder="Name..." id="ni_'+d.unknown_id+'">'
                +'<button class="save-btn" onclick="saveFace('+d.unknown_id+')">Save</button>';
            } else {
              nameHtml = '<span class="name">'+d.name+'</span> ';
            }
          }
          return '<div class="det"><span>'+nameHtml+'<span class="cls">'+d.class+'</span> <span class="conf">'+(d.confidence*100).toFixed(0)+'%</span></span><span class="dist">'+dist+'</span></div>';
        }).join('');
      }
    } catch(e) {}
    setTimeout(pollDetections, 300);
  }
  pollDetections();

  async function saveFace(unknownId) {
    const input = document.getElementById('ni_'+unknownId);
    if (!input||!input.value.trim()) return;
    try { await fetch('/save_face', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({unknown_id:unknownId, name:input.value.trim()})}); } catch(e) { console.error(e); }
  }

  const kfList = document.getElementById('kf-list');
  async function pollKnownFaces() {
    try {
      const r = await fetch('/known_faces');
      const faces = await r.json();
      if (faces.length===0) {
        kfList.innerHTML = '<div style="color:#666;font-size:12px;">No saved faces</div>';
      } else {
        kfList.innerHTML = faces.map(f =>
          '<div class="known"><span class="kname">'+f.name+'</span><button class="del-btn" onclick="deleteFace(\\''+f.name+'\\')">x</button></div>'
        ).join('');
      }
    } catch(e) {}
    setTimeout(pollKnownFaces, 2000);
  }
  pollKnownFaces();

  async function deleteFace(name) {
    if (!confirm('Delete face: '+name+'?')) return;
    try { await fetch('/delete_face', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name})}); } catch(e) { console.error(e); }
  }

  async function pollRobotStatus() {
    try {
      const r = await fetch('/robot_status');
      const s = await r.json();
      document.getElementById('conn-dot').className = s.connected ? 'connected' : 'disconnected';
      document.getElementById('conn-text').textContent = s.connected ? 'Robot: connected' : 'Robot: disconnected';
    } catch(e) {}
    setTimeout(pollRobotStatus, 2000);
  }
  pollRobotStatus();
</script>
</body>
</html>"""


class WebHandler(BaseHTTPRequestHandler):
    frame_processor = None
    robot_controller = None

    def log_message(self, format, *args):
        pass

    def _read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        return self.rfile.read(length) if length else b''

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path.startswith('/frame'):
            fp = self.frame_processor
            if fp and fp.latest_frame is not None:
                with fp._lock:
                    frame = fp.latest_frame.copy()
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(buf.tobytes())
            else:
                self.send_error(503, 'No frame')
        elif self.path == '/transcript':
            self._json_response(get_transcript())
        elif self.path == '/detections':
            fp = self.frame_processor
            dets = []
            if fp:
                with fp._lock:
                    dets = list(fp.latest_detections)
            self._json_response(dets)
        elif self.path == '/known_faces':
            fp = self.frame_processor
            faces = fp.face_cache.list_known() if (fp and fp.face_cache) else []
            self._json_response(faces)
        elif self.path == '/robot_status':
            robot = self.robot_controller
            connected = robot is not None and robot._ws is not None
            self._json_response({'connected': connected})
        else:
            self.send_error(404)

    def do_POST(self):
        fp = self.frame_processor
        robot = self.robot_controller
        try:
            body = json.loads(self._read_body())
        except Exception:
            self.send_error(400, 'Invalid JSON')
            return

        if self.path == '/chat':
            text = body.get('text', '').strip()
            if not text:
                self._json_response({'error': 'need text'}, 400)
                return
            add_transcript("You", text)
            ok = send_text_to_gemini(text)
            self._json_response({'ok': ok, 'text': text})
        elif self.path == '/cmd':
            action = body.get('action', '')
            result = self._handle_cmd(action, robot, body)
            self._json_response(result)
        elif self.path == '/save_face':
            if not fp:
                self._json_response({'error': 'not ready'}, 503)
                return
            uid = body.get('unknown_id')
            name = body.get('name', '').strip()
            if not name or uid is None:
                self._json_response({'error': 'need unknown_id and name'}, 400)
                return
            ok = fp.save_unknown_face(int(uid), name)
            self._json_response({'ok': ok, 'name': name})
        elif self.path == '/delete_face':
            if not fp or not fp.face_cache:
                self._json_response({'error': 'not ready'}, 503)
                return
            name = body.get('name', '').strip()
            if not name:
                self._json_response({'error': 'need name'}, 400)
                return
            fp.face_cache.delete_face(name)
            self._json_response({'ok': True})
        else:
            self.send_error(404)

    def _handle_cmd(self, action, robot, body=None):
        if not robot:
            return {'status': 'error', 'message': 'Robot not connected'}
        add_transcript("Action", action)
        target = body.get('target') if isinstance(body, dict) else None
        distance = body.get('distance') if isinstance(body, dict) else None

        if action == 'follow':
            robot.start_follow(target)
        elif action == 'track':
            robot.start_tracking(target)
        elif action == 'stop':
            robot.stop_all()
        elif action.startswith('go_to'):
            obj = target or action.replace('go_to_', '').replace('go_to', 'person')
            stop_dist = float(distance) if distance else 0.5
            robot.go_to_object(obj or 'person', stop_distance=stop_dist)
            return {'status': 'ok', 'action': 'go_to', 'target': obj, 'distance': stop_dist}
        elif action == 'dance':
            robot.do_dance()
        elif action.startswith('dance_'):
            robot.do_dance(action.replace('dance_', ''))
        elif action == 'wave':
            robot.do_wave()
        elif action == 'handshake':
            robot.do_handshake()
        elif action == 'dab':
            robot.do_dab()
        elif action == 'flex':
            robot.do_flex()
        elif action == 'nod':
            robot.nod()
        elif action == 'head_shake':
            robot.head_shake()
        elif action == 'look_up':
            robot.rotate_head(-0.3, 0.0)
        elif action == 'look_down':
            robot.rotate_head(0.5, 0.0)
        elif action == 'look_left':
            robot.rotate_head(0.0, 0.5)
        elif action == 'look_right':
            robot.rotate_head(0.0, -0.5)
        elif action == 'look_center':
            robot.rotate_head(0.0, 0.0)
        elif action == 'forward':
            robot.forward()
        elif action == 'backward':
            robot.backward()
        elif action == 'strafe_left':
            robot.strafe_left()
        elif action == 'strafe_right':
            robot.strafe_right()
        elif action == 'turn_left':
            robot.turn_left()
        elif action == 'turn_right':
            robot.turn_right()
        elif action == 'turn_around':
            robot.turn_around()
        else:
            return {'status': 'error', 'message': f'Unknown: {action}'}
        return {'status': 'ok', 'action': action}


def start_web_server(frame_processor, robot_controller, host, port):
    WebHandler.frame_processor = frame_processor
    WebHandler.robot_controller = robot_controller
    httpd = HTTPServer((host, port), WebHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


# ── Gemini Session ───────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are a Booster K1 humanoid robot with stereo vision cameras, face recognition, \
and full body control. Video frames are streamed to you with real-time object detection overlays — \
each detected object has a bounding box, class label, confidence score, and distance in meters. \
People you recognize are labeled with their name; unknown people are labeled 'Unknown #N'.

You can PHYSICALLY ACT by saying certain trigger phrases in your responses. When you decide to act, \
naturally include one of these phrases — your body will respond automatically:

FOLLOWING / TRACKING:
- "I'll follow you" or "Following you now" — walks toward and follows the person
- "I'll track that" or "Tracking the [object]" — moves head AND body to keep an object centered
- "Stopping now" or "I'll stop" — stops all movement and tracking

GO TO OBJECTS:
- "Going to the [object]" or "Walking to the [object]" — walks toward a detected object
- "Going to 0.5 meters from the [object]" — walks to a specific distance from the object
- "Moving over to the [person name]" — walks toward a specific named person

LOOKING / HEAD CONTROL:
- "Looking left/right/up/down" — head movement
- "Looking forward" or "Looking straight" — centers head

MOVEMENT:
- "Walking forward/backward" — walks briefly
- "Strafing left/right" — sidesteps
- "Turning left/right" — rotates body
- "Turning around" — turns 180 degrees
- "Coming closer" or "Backing up"

DANCES & GESTURES:
- "Let me dance" — does a robot dance
- "Moonwalk!" / "Michael Jackson dance!" / "Kick!" / "Roundhouse!" / "Salsa!" etc.
- "I'll wave" / "Let me shake hands" / "Dabbing!" / "Flexing!"
- "Nodding" / "Shaking my head"

IMPORTANT RULES:
- When someone says "follow me", respond with "I'll follow you!"
- When someone says "go to the chair", respond with "Going to the chair!"
- Keep responses short and conversational.
- Only trigger actions when explicitly asked or socially appropriate.
"""


async def gemini_send_video(session, frame_processor, interval):
    try:
        while True:
            b64 = frame_processor.get_frame_b64jpeg()
            if b64:
                await session.send_realtime_input(
                    video=types.Blob(data=b64, mime_type="image/jpeg")
                )
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def gemini_send_audio(session, audio_queue):
    """Forward audio from robot mic (via audio_queue) to Gemini."""
    try:
        while True:
            data = await audio_queue.get()
            await session.send_realtime_input(
                audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}")
            )
    except asyncio.CancelledError:
        pass


async def gemini_send_local_audio(session, pya, mic_device=None, mic_gain=1.0):
    """Capture audio from local mic and send to Gemini."""
    kwargs = dict(
        format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=SEND_SAMPLE_RATE,
        input=True, frames_per_buffer=AUDIO_CHUNK,
    )
    if mic_device is not None:
        kwargs['input_device_index'] = mic_device
    stream = pya.open(**kwargs)
    apply_gain = mic_gain > 1.01
    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await loop.run_in_executor(
                None, lambda: stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            )
            if apply_gain:
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                samples *= mic_gain
                np.clip(samples, -32768, 32767, out=samples)
                data = samples.astype(np.int16).tobytes()
            await session.send_realtime_input(
                audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}")
            )
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()


async def gemini_receive(session, pya, cmd_dispatcher, robot_ws_ref):
    """Receive Gemini responses: play audio locally + send to robot, parse commands."""
    stream = pya.open(
        format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=RECV_SAMPLE_RATE,
        output=True, frames_per_buffer=AUDIO_CHUNK,
    )
    loop = asyncio.get_event_loop()
    try:
        while True:
            async for msg in session.receive():
                if msg.data:
                    await loop.run_in_executor(None, stream.write, msg.data)
                    ws = robot_ws_ref.get('ws')
                    if ws:
                        try:
                            await ws.send(bytes([MSG_AUDIO_OUT]) + msg.data)
                        except Exception:
                            pass

                sc = msg.server_content
                if sc:
                    if sc.input_transcription and sc.input_transcription.text:
                        txt = sc.input_transcription.text
                        print(f"  You: {txt}")
                        add_transcript("You", txt)
                    if sc.output_transcription and sc.output_transcription.text:
                        txt = sc.output_transcription.text
                        print(f"Robot: {txt}")
                        add_transcript("Robot", txt)
                        cmd_dispatcher.check_transcript(txt)
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()


# ── WebSocket Server (robot connection) ──────────────────────────────────────


async def handle_robot_ws(websocket, frame_processor: FrameProcessor,
                          robot: RobotController, audio_queue: asyncio.Queue,
                          robot_ws_ref: dict):
    """Handle a single robot WebSocket connection."""
    print(f"[WS] Robot connected from {websocket.remote_address}")
    robot.set_connection(websocket, asyncio.get_event_loop())
    robot_ws_ref['ws'] = websocket

    try:
        async for message in websocket:
            if isinstance(message, bytes) and len(message) > 1:
                msg_type = message[0]
                payload = message[1:]
                if msg_type == MSG_VIDEO:
                    frame_processor.on_video_frame(payload)
                elif msg_type == MSG_DEPTH:
                    frame_processor.on_depth_frame(payload)
                elif msg_type == MSG_AUDIO_IN:
                    await audio_queue.put(payload)
            # text messages from robot (status, etc.) — currently unused
    except Exception as e:
        print(f"[WS] Robot disconnected: {e}")
    finally:
        robot.set_connection(None, None)
        robot_ws_ref['ws'] = None
        print("[WS] Robot connection closed")


# ── Main ─────────────────────────────────────────────────────────────────────


async def run_server(args):
    global _frame_processor_ref, _cmd_dispatcher_ref, _session_ref, _event_loop_ref

    api_key = args.api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("Error: provide --api-key or set GEMINI_API_KEY env variable")
        sys.exit(1)

    os.environ.pop('GOOGLE_API_KEY', None)
    os.environ.pop('GEMINI_API_KEY', None)

    face_cache = None
    if not args.no_faces and face_recognition is not None:
        face_cache = FaceCache(tolerance=args.face_tolerance)

    enable_faces = (not args.no_faces) and (face_recognition is not None)
    if not enable_faces and face_recognition is None:
        print("Note: face_recognition/dlib not available; running without face recognition.")
    frame_processor = FrameProcessor(
        model_path=args.model, confidence=args.confidence,
        face_cache=face_cache, enable_faces=enable_faces,
    )
    _frame_processor_ref = frame_processor

    robot = RobotController()
    robot.set_frame_processor(frame_processor)
    robot.follow_target_distance = args.follow_distance

    cmd_dispatcher = CommandDispatcher(robot)
    _cmd_dispatcher_ref = cmd_dispatcher

    httpd = start_web_server(frame_processor, robot, '0.0.0.0', args.port)
    print(f"Web UI: http://0.0.0.0:{args.port}")

    audio_queue = asyncio.Queue(maxsize=100)
    robot_ws_ref = {'ws': None}

    # WebSocket server for robot connection
    import websockets
    print(f"websockets version: {websockets.__version__}")

    async def _ws_handler(websocket, path=None):
        await handle_robot_ws(websocket, frame_processor, robot, audio_queue, robot_ws_ref)

    ws_server = await websockets.serve(
        _ws_handler,
        '0.0.0.0', args.ws_port,
        max_size=10 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    )
    print(f"Robot WebSocket: ws://0.0.0.0:{args.ws_port}")
    print(f"  Tell robot_client.py to connect to: ws://<THIS_IP>:{args.ws_port}")

    # Gemini session
    client = genai.Client(api_key=api_key)
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=args.voice)
            ),
        ),
        system_instruction=SYSTEM_INSTRUCTION,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    pya = pyaudio.PyAudio()
    _event_loop_ref = asyncio.get_event_loop()

    print("Connecting to Gemini Live...")
    try:
        async with client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            config=config,
        ) as session:
            _session_ref = session
            print("Connected to Gemini!")
            print("Waiting for robot to connect...")
            print("Press Ctrl+C to stop.\n")

            tasks = [
                asyncio.create_task(gemini_send_video(session, frame_processor, args.frame_interval)),
                asyncio.create_task(gemini_receive(session, pya, cmd_dispatcher, robot_ws_ref)),
            ]

            if args.audio_source == 'local':
                tasks.append(asyncio.create_task(
                    gemini_send_local_audio(session, pya, args.mic_device, args.mic_gain)
                ))
            else:
                tasks.append(asyncio.create_task(
                    gemini_send_audio(session, audio_queue)
                ))

            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                pass
            finally:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            _session_ref = None
    finally:
        pya.terminate()
        ws_server.close()
        robot.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='Remote Server — YOLO + Face + Gemini + Robot Control'
    )
    parser.add_argument('--api-key', type=str, default=None,
                        help='Gemini API key (or set GEMINI_API_KEY)')
    parser.add_argument('--voice', type=str, default='Puck',
                        choices=['Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'])
    parser.add_argument('--frame-interval', type=float, default=1.0,
                        help='Seconds between frames sent to Gemini')
    parser.add_argument('--port', type=int, default=8080, help='Web UI port')
    parser.add_argument('--ws-port', type=int, default=9090,
                        help='WebSocket port for robot connection')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model')
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--face-tolerance', type=float, default=0.6)
    parser.add_argument('--no-faces', action='store_true')
    parser.add_argument('--follow-distance', type=float, default=1.0)
    parser.add_argument('--audio-source', choices=['local', 'robot'], default='robot',
                        help="'local' = use this machine's mic; 'robot' = stream from robot mic")
    parser.add_argument('--mic-gain', type=float, default=3.0,
                        help='Mic gain (only for --audio-source local)')
    parser.add_argument('--mic-device', type=int, default=None,
                        help='PyAudio mic device (only for --audio-source local)')
    args = parser.parse_args()

    print("=" * 60)
    print("Remote Robot Server")
    print("  YOLO + Face Recognition + Gemini Live + Robot Control")
    print(f"  Audio: {args.audio_source} mic")
    print("=" * 60)

    try:
        asyncio.run(run_server(args))
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
