#!/usr/bin/env python3
"""
Gemini Live Camera - Talk to your robot about what it sees.
Uses onboard stereo cameras (via ROS2) for neural depth + YOLOv8 for object detection
+ face_recognition for remembering people by name.

Usage:
    export GEMINI_API_KEY="your-key"
    python3 gemini_live_camera.py
    python3 gemini_live_camera.py --voice Charon --frame-interval 2.0
    python3 gemini_live_camera.py --confidence 0.4 --face-tolerance 0.5
    python3 gemini_live_camera.py --no-faces  # disable face recognition
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
from collections import deque
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import cv2
import pyaudio

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from ultralytics import YOLO
import face_recognition

from google import genai
from google.genai import types

# Audio config
SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024

# Detection colors (BGR)
_COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255),
    (0, 255, 255), (128, 255, 0), (255, 0, 128), (128, 0, 255),
]

# Name extraction patterns from speech
_NAME_PATTERNS = [
    re.compile(r"\bmy name is (\w+)", re.IGNORECASE),
    re.compile(r"\bi'm (\w+)", re.IGNORECASE),
    re.compile(r"\bi am (\w+)", re.IGNORECASE),
    re.compile(r"\bcall me (\w+)", re.IGNORECASE),
]


def _color_for_class(cls_id):
    return _COLORS[cls_id % len(_COLORS)]


# ── Face Cache ───────────────────────────────────────────────────────────────


FACE_CACHE_DIR = os.path.expanduser('~/.face_cache')
FACE_CACHE_FILE = os.path.join(FACE_CACHE_DIR, 'known_faces.json')


class FaceCache:
    """Persistent cache of known face encodings + names on disk."""

    def __init__(self, tolerance=0.6):
        self.tolerance = tolerance
        self.entries = []  # [{"name": str, "encoding": np.array, "saved_at": str}]
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
                {
                    'name': e['name'],
                    'encoding': e['encoding'].tolist(),
                    'saved_at': e['saved_at'],
                }
                for e in self.entries
            ]
            with open(FACE_CACHE_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: failed to save face cache: {e}")

    def recognize(self, encoding):
        """Return best matching name, or None if no match within tolerance."""
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
        """Save a new face or update an existing name's encoding."""
        with self._lock:
            # Check if name already exists — update encoding
            for e in self.entries:
                if e['name'].lower() == name.lower():
                    e['encoding'] = encoding
                    e['saved_at'] = datetime.now().isoformat()
                    self._persist()
                    return
            self.entries.append({
                'name': name,
                'encoding': encoding,
                'saved_at': datetime.now().isoformat(),
            })
            self._persist()
        print(f"Face cache: saved '{name}'")

    def delete_face(self, name):
        """Remove a face by name."""
        with self._lock:
            self.entries = [e for e in self.entries if e['name'].lower() != name.lower()]
            self._persist()
        print(f"Face cache: deleted '{name}'")

    def list_known(self):
        """Return list of known face info (without encodings)."""
        with self._lock:
            return [{'name': e['name'], 'saved_at': e['saved_at']} for e in self.entries]


# ── Camera + Detection Node ─────────────────────────────────────────────────


class CameraDetectionNode(Node):
    """ROS2 node: left camera + stereo depth + YOLO detection + face recognition."""

    def __init__(self, model_path='yolov8n.pt', confidence=0.5,
                 face_cache=None, enable_faces=True):
        super().__init__('gemini_live_camera')
        self.bridge = CvBridge()

        # Load YOLO model
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.get_logger().info(
            f'YOLO ready — {len(self.model.names)} classes, conf>={confidence}'
        )

        # Face recognition
        self.enable_faces = enable_faces
        self.face_cache = face_cache
        self._unknown_faces = {}        # temp_id → encoding
        self._next_unknown_id = 1
        self._last_face_time = 0.0
        self._face_interval = 0.5       # run face recognition every 0.5s
        self._cached_face_results = []  # [{name, face_loc, unknown_id}]

        if enable_faces:
            self.get_logger().info('Face recognition enabled (CNN + CUDA)')

        # Shared state
        self._lock = threading.Lock()
        self.latest_frame = None
        self.latest_detections = []
        self._depth_map = None
        self._raw_frame = None
        self._fps = 0.0
        self._fps_counter = 0
        self._fps_time = time.time()

        # Subscribe to left camera image
        self.create_subscription(
            Image, '/image_left_raw', self._on_image, 10
        )
        self.get_logger().info('Subscribed to /image_left_raw')

        # Fallback: compressed stream
        self.create_subscription(
            CompressedImage, '/booster_video_stream', self._on_compressed, 10
        )
        self.get_logger().info('Subscribed to /booster_video_stream (fallback)')

        # Subscribe to stereo neural depth
        self.create_subscription(
            Image, '/StereoNetNode/stereonet_depth', self._on_depth, 10
        )
        self.get_logger().info('Subscribed to /StereoNetNode/stereonet_depth')

        # Run detection at ~10 Hz
        self._pending_frame = None
        self.create_timer(0.1, self._detect_tick)

    def _convert_image(self, msg):
        try:
            if msg.encoding == 'nv12':
                h, w = msg.height, msg.width
                yuv = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (int(h * 1.5), w)
                )
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            else:
                return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image convert error: {e}')
            return None

    def _on_image(self, msg):
        frame = self._convert_image(msg)
        if frame is not None:
            self._pending_frame = frame
            self._raw_frame = frame

    def _on_compressed(self, msg):
        if self._raw_frame is not None:
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self._pending_frame = frame
                self._raw_frame = frame
        except Exception as e:
            self.get_logger().error(f'Compressed image error: {e}')

    def _on_depth(self, msg):
        try:
            if msg.encoding == 'mono16':
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                    (msg.height, msg.width)
                )
                self._depth_map = depth
            else:
                self._depth_map = self.bridge.imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(f'Depth error: {e}')

    def _get_depth_at(self, x, y, window=5):
        depth_map = self._depth_map
        if depth_map is None:
            return None
        h, w = depth_map.shape
        half = window // 2
        y1, y2 = max(0, y - half), min(h, y + half + 1)
        x1, x2 = max(0, x - half), min(w, x + half + 1)
        patch = depth_map[y1:y2, x1:x2].astype(np.float32)
        valid = patch[(patch > 0) & (patch < 65535)]
        if len(valid) == 0:
            return None
        return float(np.median(valid)) / 1000.0

    def _get_or_assign_unknown_id(self, encoding):
        """Find existing unknown temp ID for this face, or assign a new one."""
        best_dist = 999.0
        best_id = None
        for uid, enc in self._unknown_faces.items():
            dist = float(face_recognition.face_distance([enc], encoding)[0])
            if dist < best_dist:
                best_dist = dist
                best_id = uid
        # If close enough to an existing unknown, reuse that ID
        if best_id is not None and best_dist < 0.5:
            self._unknown_faces[best_id] = encoding  # update encoding
            return best_id
        # New unknown
        uid = self._next_unknown_id
        self._next_unknown_id += 1
        self._unknown_faces[uid] = encoding
        return uid

    def _run_face_recognition(self, frame):
        """Run face detection + encoding + matching. Returns list of face results."""
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
                'name': name,
                'unknown_id': unknown_id,
                'face_loc': (top, right, bottom, left),
                'encoding': enc,
            })
        return results

    def _match_face_to_person(self, face_result, detections):
        """Find which person detection a face belongs to."""
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

    def _detect_tick(self):
        frame = self._pending_frame
        if frame is None:
            return

        try:
            # YOLO detection
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
                        'class': cls_name,
                        'confidence': round(float(conf), 2),
                        'distance_m': round(float(distance_m), 2) if distance_m else None,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'center': [int(cx), int(cy)],
                        'name': None,
                        'unknown_id': None,
                    })

            # Face recognition (rate-limited, only when persons present)
            now = time.time()
            if (self.enable_faces and has_persons
                    and now - self._last_face_time >= self._face_interval):
                self._last_face_time = now
                self._cached_face_results = self._run_face_recognition(frame)

            # Apply cached face results to detections
            for fr in self._cached_face_results:
                matched_det = self._match_face_to_person(fr, detections)
                if matched_det:
                    matched_det['name'] = fr['name']
                    matched_det['unknown_id'] = fr.get('unknown_id')

                # Draw face rectangle + name on annotated frame
                top, right, bottom, left = fr['face_loc']
                is_known = fr['unknown_id'] is None
                face_color = (0, 255, 255) if is_known else (0, 165, 255)  # cyan=known, orange=unknown

                cv2.rectangle(annotated, (left, top), (right, bottom), face_color, 2)

                name = fr['name']
                (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated, (left, bottom), (left + tw + 4, bottom + th + 8), face_color, -1)
                cv2.putText(
                    annotated, name, (left + 2, bottom + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
                )

            # Draw YOLO boxes
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
                cv2.putText(
                    annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
                )

            # FPS counter
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

        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')

    def save_unknown_face(self, unknown_id, name):
        """Save an unknown face by its temp ID."""
        enc = self._unknown_faces.get(unknown_id)
        if enc is None:
            return False
        self.face_cache.save_face(name, enc)
        del self._unknown_faces[unknown_id]
        # Clear cached results so next tick picks up the new name
        self._cached_face_results = []
        self._last_face_time = 0
        return True

    def try_learn_name_from_transcript(self, text):
        """Check if user said their name, auto-save if one unknown face is active."""
        if not self.enable_faces or not self._unknown_faces:
            return
        for pattern in _NAME_PATTERNS:
            match = pattern.search(text)
            if match:
                name = match.group(1).capitalize()
                # Save for the most recent unknown face
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


# ── Shared transcript ───────────────────────────────────────────────────────

transcript = deque(maxlen=200)
transcript_lock = threading.Lock()
_camera_node_ref = None  # set in main() for transcript hook


def add_transcript(role, text):
    with transcript_lock:
        transcript.append({"role": role, "text": text, "ts": time.time()})
    # Try to learn names from user speech
    if role == "You" and _camera_node_ref:
        _camera_node_ref.try_learn_name_from_transcript(text)


def get_transcript():
    with transcript_lock:
        return list(transcript)


# ── Web server ──────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>Gemini Live Camera + Detection</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#111; color:#eee; font-family:system-ui,sans-serif; display:flex; height:100vh; }
  #left { flex:1; display:flex; align-items:center; justify-content:center; background:#000; min-width:0; }
  #left img { max-width:100%; max-height:100%; object-fit:contain; }
  #right { width:400px; display:flex; flex-direction:column; border-left:1px solid #333; }
  #header { padding:12px 16px; border-bottom:1px solid #333; font-size:14px; color:#888; }
  #header span { color:#4CAF50; font-weight:bold; }

  #detections { padding:8px 16px; border-bottom:1px solid #333; max-height:180px; overflow-y:auto; }
  #detections h3 { font-size:12px; color:#888; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px; }
  .det { padding:4px 8px; margin:3px 0; background:#1a2a1a; border-radius:4px; font-size:13px; display:flex; justify-content:space-between; align-items:center; }
  .det .cls { color:#4CAF50; }
  .det .name { color:#00BCD4; font-weight:bold; }
  .det .unknown { color:#FF9800; }
  .det .dist { color:#2196F3; font-weight:bold; }
  .det .conf { color:#666; font-size:11px; }
  .det .name-input { width:80px; padding:2px 4px; background:#222; border:1px solid #555; color:#eee; border-radius:3px; font-size:12px; }
  .det .save-btn { padding:2px 8px; background:#4CAF50; color:#000; border:none; border-radius:3px; font-size:11px; cursor:pointer; margin-left:4px; }

  #known-faces { padding:8px 16px; border-bottom:1px solid #333; max-height:120px; overflow-y:auto; }
  #known-faces h3 { font-size:12px; color:#888; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px; }
  .known { padding:3px 8px; margin:2px 0; background:#1a2a2a; border-radius:4px; font-size:13px; display:flex; justify-content:space-between; align-items:center; }
  .known .kname { color:#00BCD4; }
  .known .del-btn { padding:1px 6px; background:#c62828; color:#fff; border:none; border-radius:3px; font-size:11px; cursor:pointer; }

  #chat { flex:1; overflow-y:auto; padding:12px 16px; display:flex; flex-direction:column; gap:8px; }
  .msg { padding:8px 12px; border-radius:8px; max-width:95%; font-size:14px; line-height:1.4; word-wrap:break-word; }
  .msg.you { background:#1a3a5c; align-self:flex-end; }
  .msg.robot { background:#2d2d2d; align-self:flex-start; }
  .msg.system { background:#2a2a1a; align-self:center; font-size:12px; color:#aaa; }
  .msg .role { font-size:11px; color:#888; margin-bottom:2px; }
  #status { padding:8px 16px; border-top:1px solid #333; font-size:12px; color:#666; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#4CAF50; margin-right:6px; }
</style>
</head>
<body>
  <div id="left"><img id="feed" src="/frame" alt="Camera"></div>
  <div id="right">
    <div id="header"><span>Gemini Live</span> &mdash; Detection + Depth + Faces</div>
    <div id="detections"><h3>Detections</h3><div id="det-list">Waiting...</div></div>
    <div id="known-faces"><h3>Known Faces</h3><div id="kf-list">None yet</div></div>
    <div id="chat"></div>
    <div id="status"><span class="dot"></span>Listening...</div>
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

  const chat = document.getElementById('chat');
  let lastLen = 0;
  async function pollTranscript() {
    try {
      const r = await fetch('/transcript');
      const msgs = await r.json();
      if (msgs.length !== lastLen) {
        lastLen = msgs.length;
        chat.innerHTML = msgs.map(m => {
          const cls = m.role === 'You' ? 'you' : m.role === 'System' ? 'system' : 'robot';
          return `<div class="msg ${cls}"><div class="role">${m.role}</div>${m.text}</div>`;
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
          const dist = d.distance_m !== null ? d.distance_m.toFixed(1) + 'm' : '?';
          let nameHtml = '';
          if (d.class === 'person' && d.name) {
            if (d.unknown_id !== null && d.unknown_id !== undefined) {
              nameHtml = `<span class="unknown">${d.name}</span> `
                + `<input class="name-input" placeholder="Name..." id="ni_${d.unknown_id}">`
                + `<button class="save-btn" onclick="saveFace(${d.unknown_id})">Save</button>`;
            } else {
              nameHtml = `<span class="name">${d.name}</span> `;
            }
          }
          return `<div class="det"><span>${nameHtml}<span class="cls">${d.class}</span> <span class="conf">${(d.confidence*100).toFixed(0)}%</span></span><span class="dist">${dist}</span></div>`;
        }).join('');
      }
    } catch(e) {}
    setTimeout(pollDetections, 300);
  }
  pollDetections();

  async function saveFace(unknownId) {
    const input = document.getElementById('ni_' + unknownId);
    if (!input || !input.value.trim()) return;
    const name = input.value.trim();
    try {
      await fetch('/save_face', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({unknown_id: unknownId, name: name})
      });
    } catch(e) { console.error(e); }
  }

  const kfList = document.getElementById('kf-list');
  async function pollKnownFaces() {
    try {
      const r = await fetch('/known_faces');
      const faces = await r.json();
      if (faces.length === 0) {
        kfList.innerHTML = '<div style="color:#666;font-size:12px;">No saved faces</div>';
      } else {
        kfList.innerHTML = faces.map(f =>
          `<div class="known"><span class="kname">${f.name}</span><button class="del-btn" onclick="deleteFace('${f.name}')">x</button></div>`
        ).join('');
      }
    } catch(e) {}
    setTimeout(pollKnownFaces, 2000);
  }
  pollKnownFaces();

  async function deleteFace(name) {
    if (!confirm('Delete face: ' + name + '?')) return;
    try {
      await fetch('/delete_face', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name: name})
      });
    } catch(e) { console.error(e); }
  }
</script>
</body>
</html>"""


class WebHandler(BaseHTTPRequestHandler):
    camera_node = None

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
            node = self.camera_node
            if node and node.latest_frame is not None:
                with node._lock:
                    frame = node.latest_frame.copy()
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
            node = self.camera_node
            dets = []
            if node:
                with node._lock:
                    dets = list(node.latest_detections)
            self._json_response(dets)

        elif self.path == '/known_faces':
            node = self.camera_node
            faces = node.face_cache.list_known() if (node and node.face_cache) else []
            self._json_response(faces)

        else:
            self.send_error(404)

    def do_POST(self):
        node = self.camera_node
        if not node:
            self.send_error(503, 'Not ready')
            return

        try:
            body = json.loads(self._read_body())
        except Exception:
            self.send_error(400, 'Invalid JSON')
            return

        if self.path == '/save_face':
            uid = body.get('unknown_id')
            name = body.get('name', '').strip()
            if not name or uid is None:
                self._json_response({'error': 'need unknown_id and name'}, 400)
                return
            ok = node.save_unknown_face(int(uid), name)
            self._json_response({'ok': ok, 'name': name})

        elif self.path == '/delete_face':
            name = body.get('name', '').strip()
            if not name:
                self._json_response({'error': 'need name'}, 400)
                return
            node.face_cache.delete_face(name)
            self._json_response({'ok': True})

        else:
            self.send_error(404)


def start_web_server(camera_node, host, port):
    WebHandler.camera_node = camera_node
    httpd = HTTPServer((host, port), WebHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


# ── Async tasks ─────────────────────────────────────────────────────────────


async def send_audio(session, pya):
    stream = pya.open(
        format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
        input=True, frames_per_buffer=CHUNK_SIZE,
    )
    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await loop.run_in_executor(
                None, lambda: stream.read(CHUNK_SIZE, exception_on_overflow=False),
            )
            await session.send_realtime_input(
                audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}")
            )
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()


async def send_video(session, camera_node, interval):
    try:
        while True:
            b64 = camera_node.get_frame_b64jpeg()
            if b64:
                await session.send_realtime_input(
                    video=types.Blob(data=b64, mime_type="image/jpeg")
                )
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def receive_responses(session, pya):
    stream = pya.open(
        format=FORMAT, channels=CHANNELS, rate=RECV_SAMPLE_RATE,
        output=True, frames_per_buffer=CHUNK_SIZE,
    )
    loop = asyncio.get_event_loop()
    try:
        while True:
            async for msg in session.receive():
                if msg.data:
                    await loop.run_in_executor(None, stream.write, msg.data)

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
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()


# ── Main session ────────────────────────────────────────────────────────────


async def run_session(api_key, camera_node, voice, frame_interval):
    client = genai.Client(api_key=api_key)

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            ),
        ),
        system_instruction=(
            "You are a Booster K1 humanoid robot with stereo vision cameras and face recognition. "
            "Video frames are streamed to you continuously with real-time object detection overlays — "
            "each detected object has a bounding box, class label, confidence score, and calibrated "
            "distance in meters. People you recognize are labeled with their name; unknown people "
            "are labeled 'Unknown #N'. When you see an unknown person, politely ask their name "
            "in a friendly way (e.g. 'Hey, I don't think we've met! What's your name?'). "
            "When you see a recognized person, greet them by name warmly. "
            "You remember people between sessions. "
            "Be conversational and friendly. Keep answers short unless asked for detail."
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    print("Connecting to Gemini Live...")
    pya = pyaudio.PyAudio()

    try:
        async with client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            config=config,
        ) as session:
            print("Connected! Start talking — ask the robot what it sees.")
            print("Press Ctrl+C to stop.\n")

            tasks = [
                asyncio.create_task(send_audio(session, pya)),
                asyncio.create_task(send_video(session, camera_node, frame_interval)),
                asyncio.create_task(receive_responses(session, pya)),
            ]

            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                pass
            finally:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        pya.terminate()


# ── Entrypoint ──────────────────────────────────────────────────────────────


def main():
    global _camera_node_ref

    parser = argparse.ArgumentParser(
        description='Gemini Live + Stereo Depth + YOLOv8 + Face Recognition'
    )
    parser.add_argument(
        '--api-key', type=str, default=None,
        help='Gemini API key (or set GEMINI_API_KEY env var)',
    )
    parser.add_argument(
        '--voice', type=str, default='Puck',
        choices=['Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'],
        help='Gemini voice (default: Puck)',
    )
    parser.add_argument(
        '--frame-interval', type=float, default=1.0,
        help='Seconds between camera frame sends to Gemini (default: 1.0)',
    )
    parser.add_argument(
        '--port', type=int, default=8080,
        help='Web UI port (default: 8080)',
    )
    parser.add_argument(
        '--model', type=str, default='yolov8n.pt',
        help='YOLO model path (default: yolov8n.pt)',
    )
    parser.add_argument(
        '--confidence', type=float, default=0.5,
        help='Detection confidence threshold (default: 0.5)',
    )
    parser.add_argument(
        '--face-tolerance', type=float, default=0.6,
        help='Face matching tolerance — lower is stricter (default: 0.6)',
    )
    parser.add_argument(
        '--no-faces', action='store_true',
        help='Disable face recognition',
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("Error: provide --api-key or set GEMINI_API_KEY env variable")
        sys.exit(1)

    os.environ.pop('GOOGLE_API_KEY', None)
    os.environ.pop('GEMINI_API_KEY', None)

    print("=" * 60)
    print("Gemini Live + Depth + YOLOv8 + Face Recognition")
    print("=" * 60)

    # Face cache
    face_cache = None
    if not args.no_faces:
        face_cache = FaceCache(tolerance=args.face_tolerance)
        print(f"Face recognition: ON (tolerance={args.face_tolerance})")
    else:
        print("Face recognition: OFF")

    # Start ROS2 + detection node
    rclpy.init()
    camera_node = CameraDetectionNode(
        model_path=args.model,
        confidence=args.confidence,
        face_cache=face_cache,
        enable_faces=not args.no_faces,
    )
    _camera_node_ref = camera_node

    ros_thread = threading.Thread(
        target=rclpy.spin, args=(camera_node,), daemon=True
    )
    ros_thread.start()

    # Start web server
    httpd = start_web_server(camera_node, '0.0.0.0', args.port)
    print(f"Web UI: http://0.0.0.0:{args.port}")

    # Wait for first frame
    print("Waiting for camera frame...")
    deadline = time.time() + 10
    while camera_node.latest_frame is None:
        if time.time() > deadline:
            print("Warning: no frame after 10s — continuing without video")
            break
        time.sleep(0.1)
    else:
        has_depth = camera_node._depth_map is not None
        print(f"Camera ready! Depth: {'active' if has_depth else 'waiting...'}")

    try:
        asyncio.run(run_session(api_key, camera_node, args.voice, args.frame_interval))
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
