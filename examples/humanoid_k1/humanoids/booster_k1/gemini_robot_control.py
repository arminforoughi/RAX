#!/usr/bin/env python3
"""
Gemini Live Camera + Robot Control
Talk to your robot and it can follow people, track objects with its head,
dance, wave, and move around — all driven by voice + vision.

Usage:
    export GEMINI_API_KEY="your-key"
    python3 gemini_robot_control.py eth0
    python3 gemini_robot_control.py eth0 --voice Charon --no-faces
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

# Robot SDK
from booster_robotics_sdk_python import (
    B1LocoClient, ChannelFactory, RobotMode, B1HandIndex, B1HandAction,
    Position, Orientation, Posture, DexterousFingerParameter,
)

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

# Name extraction patterns
_NAME_PATTERNS = [
    re.compile(r"\bmy name is (\w+)", re.IGNORECASE),
    re.compile(r"\bi'm (\w+)", re.IGNORECASE),
    re.compile(r"\bi am (\w+)", re.IGNORECASE),
    re.compile(r"\bcall me (\w+)", re.IGNORECASE),
]

# Command patterns parsed from Gemini output transcription
_CMD_PATTERNS = [
    # Follow
    (re.compile(r"\b(?:i'll |i will |let me |okay,? |ok,? )?(?:follow|following)\b(?:\s+(?:you|him|her|them|that person|(\w+)))?", re.IGNORECASE), "follow"),
    # Stop following / stop tracking / stop
    (re.compile(r"\b(?:i'll |i will |let me )?stop(?:ping)?\b(?:\s+(?:follow|track|mov))?", re.IGNORECASE), "stop"),
    # Go to / walk to object with optional distance (must be before generic approach)
    # Captures: group(1)=distance (e.g. "0.5"), group(2)=target object
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:go(?:ing)?|walk(?:ing)?|head(?:ing)?|mov(?:e|ing))\s+(?:to(?:ward)?|over to)\s+(?:(\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|ft|feet)\s+(?:from|away from|near|of)\s+)?(?:the\s+|that\s+)?(\w+)", re.IGNORECASE), "go_to"),
    # Dance
    (re.compile(r"\b(?:i'll |i will |let me |here'?s? |okay,? |ok,? )?(?:do |doing |start )?(?:a |the )?(?:dance|dancing)\b(?:\s+(?:the\s+)?(\w+))?", re.IGNORECASE), "dance"),
    # Wave
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:wave|waving)\b", re.IGNORECASE), "wave"),
    # Handshake
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:handshake|shake hands?|shaking hands?)\b", re.IGNORECASE), "handshake"),
    # Look directions (must be before generic "look at" track)
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+(?:to\s+(?:the\s+)?)?left\b", re.IGNORECASE), "look_left"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+(?:to\s+(?:the\s+)?)?right\b", re.IGNORECASE), "look_right"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+up\b", re.IGNORECASE), "look_up"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+down\b", re.IGNORECASE), "look_down"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?look(?:ing)?\s+(?:center|straight|forward|ahead)\b", re.IGNORECASE), "look_center"),
    # Look at / track
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:look(?:ing)? at|track(?:ing)?|watch(?:ing)?)\b(?:\s+(?:the\s+)?(\w+))?", re.IGNORECASE), "track"),
    # Turn left / right / around
    (re.compile(r"\b(?:i'll |let me )?turn(?:ing)?\s+(?:to\s+(?:the\s+)?)?left\b", re.IGNORECASE), "turn_left"),
    (re.compile(r"\b(?:i'll |let me )?turn(?:ing)?\s+(?:to\s+(?:the\s+)?)?right\b", re.IGNORECASE), "turn_right"),
    (re.compile(r"\b(?:i'll |let me )?turn(?:ing)?\s+around\b", re.IGNORECASE), "turn_around"),
    # Walk forward / backward
    (re.compile(r"\b(?:i'll |let me )?(?:walk(?:ing)?|mov(?:e|ing))\s+forward\b", re.IGNORECASE), "forward"),
    (re.compile(r"\b(?:i'll |let me )?(?:walk(?:ing)?|mov(?:e|ing))\s+backward\b", re.IGNORECASE), "backward"),
    # Come closer / back up
    (re.compile(r"\b(?:com(?:e|ing)\s+closer|approach(?:ing)?)\b", re.IGNORECASE), "approach"),
    (re.compile(r"\b(?:back(?:ing)?\s+up|step(?:ping)?\s+back|mov(?:e|ing)\s+back)\b", re.IGNORECASE), "back_up"),
    # Strafe
    (re.compile(r"\b(?:i'll |let me )?(?:straf(?:e|ing)|sidestep(?:ping)?|mov(?:e|ing)\s+sideways)\s+(?:to\s+(?:the\s+)?)?left\b", re.IGNORECASE), "strafe_left"),
    (re.compile(r"\b(?:i'll |let me )?(?:straf(?:e|ing)|sidestep(?:ping)?|mov(?:e|ing)\s+sideways)\s+(?:to\s+(?:the\s+)?)?right\b", re.IGNORECASE), "strafe_right"),
    # Dab / Flex
    (re.compile(r"\b(?:i'll |let me |okay,? |here'?s? (?:a )?)?dab(?:bing)?\b", re.IGNORECASE), "dab"),
    (re.compile(r"\b(?:i'll |let me |okay,? |here'?s? (?:a )?)?flex(?:ing)?\b", re.IGNORECASE), "flex"),
    # Specific dances
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:do )?(?:the )?new\s*year(?:'?s?)?\s*dance\b", re.IGNORECASE), "dance_newyear"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:do )?(?:the )?nezha\s*dance\b", re.IGNORECASE), "dance_nezha"),
    (re.compile(r"\b(?:i'll |let me |okay,? )?(?:do )?(?:the )?future\s*dance\b", re.IGNORECASE), "dance_future"),
    # Named dances (SDK whole-body + upper-body + custom)
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
    # Nod / shake head
    (re.compile(r"\b(?:nod(?:ding)?)\b", re.IGNORECASE), "nod"),
    (re.compile(r"\b(?:shak(?:e|ing)\s+(?:my\s+)?head)\b", re.IGNORECASE), "head_shake"),
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
                'name': name,
                'encoding': encoding,
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


# ── Robot Controller ─────────────────────────────────────────────────────────


class RobotController:
    """High-level robot control: movement, head tracking, following, dances."""

    def __init__(self, client: B1LocoClient):
        self.client = client
        self.lock = threading.Lock()

        # Head state
        self.head_pitch = 0.0  # current pitch (positive=down)
        self.head_yaw = 0.0    # current yaw (positive=left)

        # Tracking state
        self.tracking_active = False
        self.tracking_target = None   # class name or person name to track
        self.tracking_thread = None

        # Follow state
        self.follow_active = False
        self.follow_target = None     # person name or "person" for any
        self.follow_thread = None
        self.follow_target_distance = 1.0  # desired distance in meters

        # Movement state
        self.move_active = False
        self.move_thread = None

        # Reference to camera node (set externally)
        self.camera_node = None

        # Arm position tracking for custom animations
        self.right_arm_pos = [0.35, -0.25, 0.1]
        self.left_arm_pos = [0.35, 0.25, 0.1]

    def set_camera_node(self, node):
        self.camera_node = node

    # ── Head control ─────────────────────────────────────────────────────

    def rotate_head(self, pitch, yaw):
        """Set head position (pitch: + down/- up, yaw: + left/- right)."""
        pitch = max(-0.5, min(1.0, pitch))
        yaw = max(-0.785, min(0.785, yaw))
        self.head_pitch = pitch
        self.head_yaw = yaw
        with self.lock:
            self.client.RotateHead(pitch, yaw)

    def nod(self):
        """Nod head yes."""
        def _nod():
            for _ in range(3):
                self.rotate_head(0.3, self.head_yaw)
                time.sleep(0.25)
                self.rotate_head(-0.1, self.head_yaw)
                time.sleep(0.25)
            self.rotate_head(0.0, 0.0)
        threading.Thread(target=_nod, daemon=True).start()

    def head_shake(self):
        """Shake head no."""
        def _shake():
            for _ in range(3):
                self.rotate_head(self.head_pitch, 0.4)
                time.sleep(0.2)
                self.rotate_head(self.head_pitch, -0.4)
                time.sleep(0.2)
            self.rotate_head(0.0, 0.0)
        threading.Thread(target=_shake, daemon=True).start()

    # ── Head tracking ────────────────────────────────────────────────────

    def start_tracking(self, target=None):
        """Start tracking a target with head. target=None tracks closest person."""
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
        # Stop any body rotation from tracking
        with self.lock:
            self.client.Move(0, 0, 0)

    def _tracking_loop(self):
        """Continuously adjust head to keep target centered in frame.
        When head yaw nears its limit, rotate the body to recenter."""
        # Head yaw limit is ~0.785 rad (45 deg). Start body turn at 60% of limit.
        YAW_BODY_TURN_THRESHOLD = 0.45
        BODY_TURN_SPEED = 0.35  # rad/s rotation when recentering

        while self.tracking_active:
            if not self.camera_node:
                time.sleep(0.1)
                continue

            det = self._find_target_detection()
            if det is None:
                # Lost target — stop body rotation if any, keep looking
                if not self.follow_active:
                    with self.lock:
                        self.client.Move(0, 0, 0)
                time.sleep(0.1)
                continue

            frame = self.camera_node._raw_frame
            if frame is None:
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            cx, cy = det['center']

            # Compute error: how far target is from frame center (normalized -1 to 1)
            err_x = (cx - w / 2) / (w / 2)  # positive = target is right of center
            err_y = (cy - h / 2) / (h / 2)  # positive = target is below center

            # Proportional control gains
            kp_yaw = 0.15
            kp_pitch = 0.1

            # Adjust head (yaw: negative = look right, pitch: positive = look down)
            new_yaw = self.head_yaw - err_x * kp_yaw
            new_pitch = self.head_pitch + err_y * kp_pitch

            # Deadzone to avoid jitter
            if abs(err_x) > 0.08 or abs(err_y) > 0.08:
                self.rotate_head(new_pitch, new_yaw)

            # Body rotation: when head yaw approaches limits, rotate body to
            # bring head back toward center. Skip if follow_loop handles movement.
            if not self.follow_active:
                if abs(self.head_yaw) > YAW_BODY_TURN_THRESHOLD:
                    # Turn body in the direction the head is looking
                    # head_yaw positive = looking left, so turn body left (positive yaw in SDK)
                    body_rot = BODY_TURN_SPEED if self.head_yaw > 0 else -BODY_TURN_SPEED
                    with self.lock:
                        self.client.Move(0, 0, body_rot)
                else:
                    with self.lock:
                        self.client.Move(0, 0, 0)

            time.sleep(0.1)  # 10 Hz tracking

    def _find_target_detection(self):
        """Find the detection matching our tracking target."""
        if not self.camera_node:
            return None

        with self.camera_node._lock:
            dets = list(self.camera_node.latest_detections)

        if not dets:
            return None

        target = self.tracking_target

        if target is None or target.lower() in ('person', 'people', 'someone', 'anyone'):
            # Track closest person
            persons = [d for d in dets if d['class'] == 'person']
            if not persons:
                return None
            # Prefer person with distance, pick closest
            with_dist = [p for p in persons if p.get('distance_m')]
            if with_dist:
                return min(with_dist, key=lambda p: p['distance_m'])
            # Fallback: largest bounding box (likely closest)
            return max(persons, key=lambda p: (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]))

        # Track by person name
        named = [d for d in dets if d.get('name') and target.lower() in d['name'].lower()]
        if named:
            return named[0]

        # Track by object class
        classed = [d for d in dets if d['class'].lower() == target.lower()]
        if classed:
            # Pick closest or largest
            with_dist = [c for c in classed if c.get('distance_m')]
            if with_dist:
                return min(with_dist, key=lambda c: c['distance_m'])
            return max(classed, key=lambda c: (c['bbox'][2] - c['bbox'][0]) * (c['bbox'][3] - c['bbox'][1]))

        return None

    # ── Follow person ────────────────────────────────────────────────────

    def start_follow(self, target=None):
        """Start following a person. Combines head tracking + walking."""
        self.stop_follow()
        self.follow_active = True
        self.follow_target = target
        # Start head tracking too
        self.start_tracking(target or "person")
        self.follow_thread = threading.Thread(target=self._follow_loop, daemon=True)
        self.follow_thread.start()
        print(f"[Robot] Following started: {target or 'closest person'}")

    def stop_follow(self):
        self.follow_active = False
        self.stop_tracking()
        if self.follow_thread:
            self.follow_thread.join(timeout=1.0)
            self.follow_thread = None
        # Stop movement
        with self.lock:
            self.client.Move(0, 0, 0)

    def _follow_loop(self):
        """Walk towards followed person with obstacle avoidance.

        Uses both depth map (5-zone scan of path ahead) and YOLO detections
        to build an obstacle map each tick, then steers/strafes around obstacles
        while heading toward the target.
        """
        OBSTACLE_DIST = 0.9       # meters — anything closer is an obstacle
        OBSTACLE_EMERGENCY = 0.5  # meters — very close, must strafe away
        STRAFE_SPEED = 0.25
        MAX_FWD = 0.5

        while self.follow_active:
            if not self.camera_node:
                time.sleep(0.1)
                continue

            det = self._find_target_detection()
            if det is None:
                with self.lock:
                    self.client.Move(0, 0, 0)
                time.sleep(0.2)
                continue

            frame = self.camera_node._raw_frame
            if frame is None:
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            cx = det['center'][0]
            distance = det.get('distance_m')
            target_id = id(det)

            # ── Horizontal error / rotation toward person ──
            err_x = (cx - w / 2) / (w / 2)  # -1 (left) to +1 (right)
            rot_speed = 0.0
            if abs(err_x) > 0.10:
                rot_speed = -err_x * 0.8

            # ── Forward speed based on distance ──
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

            # When person is far off-center, prioritize turning
            if abs(err_x) > 0.35:
                fwd_speed *= 0.2

            # ── Obstacle avoidance ──
            strafe_speed = 0.0
            obstacle_ahead = False

            # --- Method 1: Depth map scan (5 zones) ---
            depth_map = self.camera_node._depth_map
            if depth_map is not None and fwd_speed > 0:
                dh, dw = depth_map.shape
                strip_y1 = int(dh * 0.35)
                strip_y2 = int(dh * 0.75)
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

                # Center zone (index 2) and near-center zones (1, 3)
                center_d = zone_depths[2]
                near_left_d = zone_depths[1]
                near_right_d = zone_depths[3]
                far_left_d = zone_depths[0]
                far_right_d = zone_depths[4]

                target_d = distance if distance else 5.0

                if center_d < OBSTACLE_DIST and center_d < target_d - 0.3:
                    obstacle_ahead = True
                    # Determine which side is clearer
                    left_clear = (far_left_d + near_left_d) / 2
                    right_clear = (far_right_d + near_right_d) / 2

                    if center_d < OBSTACLE_EMERGENCY:
                        # Very close — strafe hard, minimal forward
                        fwd_speed = min(fwd_speed, 0.05)
                        if left_clear > right_clear:
                            strafe_speed = STRAFE_SPEED
                            rot_speed = max(rot_speed, 0.3)
                        else:
                            strafe_speed = -STRAFE_SPEED
                            rot_speed = min(rot_speed, -0.3)
                        print(f"[Follow] CLOSE obstacle {center_d:.1f}m — strafe {'left' if strafe_speed > 0 else 'right'}")
                    else:
                        # Moderate obstacle — slow down, steer away
                        fwd_speed = min(fwd_speed, 0.15)
                        if left_clear > right_clear:
                            rot_speed = max(rot_speed, 0.25)
                            strafe_speed = 0.15
                        else:
                            rot_speed = min(rot_speed, -0.25)
                            strafe_speed = -0.15
                        print(f"[Follow] Obstacle {center_d:.1f}m — steering {'left' if rot_speed > 0 else 'right'}")

                # Also check side zones even if center is clear
                # (robot could be drifting into something on the side)
                elif near_left_d < OBSTACLE_EMERGENCY:
                    strafe_speed = -0.15  # push away from left obstacle
                elif near_right_d < OBSTACLE_EMERGENCY:
                    strafe_speed = 0.15   # push away from right obstacle

            # --- Method 2: YOLO detection obstacle map ---
            if fwd_speed > 0 and not obstacle_ahead:
                with self.camera_node._lock:
                    all_dets = list(self.camera_node.latest_detections)
                # Check non-target objects that are close and in our path
                for obj in all_dets:
                    if id(obj) == target_id:
                        continue
                    obj_dist = obj.get('distance_m')
                    if obj_dist is None or obj_dist > OBSTACLE_DIST:
                        continue
                    # Object is close — check if it's in our forward path
                    obj_cx = obj['center'][0]
                    obj_err = (obj_cx - w / 2) / (w / 2)
                    if abs(obj_err) < 0.4:
                        # Object in our path
                        fwd_speed = min(fwd_speed, 0.1)
                        if obj_err <= 0:
                            strafe_speed = max(strafe_speed, -0.15)  # go right
                        else:
                            strafe_speed = min(strafe_speed, 0.15)   # go left
                        print(f"[Follow] {obj['class']} {obj_dist:.1f}m in path — dodge {'right' if obj_err <= 0 else 'left'}")
                        break

            with self.lock:
                self.client.Move(fwd_speed, strafe_speed, rot_speed)

            time.sleep(0.05)  # 20 Hz control

    # ── Timed movement ───────────────────────────────────────────────────

    def move_timed(self, x, y, yaw, duration):
        """Move for a duration then stop."""
        self.stop_movement()

        def _move():
            self.move_active = True
            start = time.time()
            while self.move_active and (time.time() - start) < duration:
                with self.lock:
                    self.client.Move(x, y, yaw)
                time.sleep(0.05)
            with self.lock:
                self.client.Move(0, 0, 0)
            self.move_active = False

        self.move_thread = threading.Thread(target=_move, daemon=True)
        self.move_thread.start()

    def stop_movement(self):
        self.move_active = False
        if self.move_thread and self.move_thread.is_alive():
            self.move_thread.join(timeout=1.0)
        with self.lock:
            self.client.Move(0, 0, 0)

    def turn_around(self):
        """Turn 180 degrees."""
        self.move_timed(0, 0, 0.5, 3.0)

    def approach(self):
        """Walk forward briefly."""
        self.move_timed(0.4, 0, 0, 2.0)

    def back_up(self):
        """Back up briefly."""
        self.move_timed(-0.2, 0, 0, 1.5)

    def turn_left(self):
        """Turn left briefly."""
        self.move_timed(0, 0, 0.5, 1.5)

    def turn_right(self):
        """Turn right briefly."""
        self.move_timed(0, 0, -0.5, 1.5)

    def forward(self):
        """Walk forward briefly."""
        self.move_timed(0.5, 0, 0, 2.0)

    def backward(self):
        """Walk backward briefly."""
        self.move_timed(-0.3, 0, 0, 2.0)

    def strafe_left(self):
        """Strafe left briefly."""
        self.move_timed(0, 0.3, 0, 1.5)

    def strafe_right(self):
        """Strafe right briefly."""
        self.move_timed(0, -0.3, 0, 1.5)

    # ── Dab & Flex ────────────────────────────────────────────────────────

    def do_dab(self):
        """Dab animation using arms and head."""
        def _dab():
            DELAY = 0.25
            self._arm_to_side("right")
            self._arm_to_side("left")
            time.sleep(0.6)
            # Right arm sweeps across face
            for _ in range(5):
                self._arm_move_inc("back", "right")
                time.sleep(DELAY)
            for _ in range(5):
                self._arm_move_inc("in", "right")
                time.sleep(DELAY)
            for _ in range(6):
                self._arm_move_inc("up", "right")
                time.sleep(DELAY)
            # Left arm extends up and out
            for _ in range(7):
                self._arm_move_inc("up", "left")
                time.sleep(DELAY)
            for _ in range(4):
                self._arm_move_inc("out", "left")
                time.sleep(DELAY)
            # Head into elbow
            self.rotate_head(0.5, 0.5)
            time.sleep(2.5)
            # Reset
            self.rotate_head(0.0, 0.0)
            time.sleep(0.3)
            self._arm_to_side("right")
            self._arm_to_side("left")
        threading.Thread(target=_dab, daemon=True).start()

    def do_flex(self):
        """Flex/victory pose animation."""
        def _flex():
            DELAY = 0.25
            self._arm_to_side("right")
            self._arm_to_side("left")
            time.sleep(0.6)
            # Both arms UP
            for _ in range(7):
                self._arm_move_inc("up", "right")
                self._arm_move_inc("up", "left")
                time.sleep(DELAY)
            # Arms OUT for T-pose
            for _ in range(3):
                self._arm_move_inc("out", "right")
                self._arm_move_inc("out", "left")
                time.sleep(DELAY)
            # Head up proudly
            self.rotate_head(-0.3, 0.0)
            time.sleep(2.0)
            # Reset
            self.rotate_head(0.0, 0.0)
            time.sleep(0.3)
            self._arm_to_side("right")
            self._arm_to_side("left")
        threading.Thread(target=_flex, daemon=True).start()

    # ── Named Dances & Moves ─────────────────────────────────────────────

    def _dance_kick(self):
        """Forward lunge step + arm pump."""
        DELAY = 0.2
        self._arm_to_side("right")
        self._arm_to_side("left")
        time.sleep(0.5)
        # Pump right arm up
        for _ in range(6):
            self._arm_move_inc("up", "right")
            time.sleep(DELAY)
        # Step forward (lunge)
        with self.lock:
            self.client.Move(0.4, 0, 0)
        time.sleep(0.8)
        with self.lock:
            self.client.Move(0, 0, 0)
        time.sleep(0.5)
        # Reset
        self._arm_to_side("right")
        self._arm_to_side("left")

    def _dance_moonwalk(self):
        """Backward walk + forward arm slide (Michael Jackson)."""
        DELAY = 0.2
        self._arm_to_side("right")
        self._arm_to_side("left")
        time.sleep(0.5)
        # Right arm slides forward
        for _ in range(5):
            self._arm_move_inc("forward", "right")
            self._arm_move_inc("up", "right")
            time.sleep(DELAY)
        # Walk backward (moonwalk)
        for _ in range(3):
            with self.lock:
                self.client.Move(-0.3, 0, 0)
            self._arm_move_inc("forward", "left")
            self._arm_move_inc("up", "left")
            time.sleep(0.5)
            self._arm_move_inc("back", "left")
            self._arm_move_inc("down", "left")
            time.sleep(0.3)
        with self.lock:
            self.client.Move(0, 0, 0)
        time.sleep(0.3)
        self._arm_to_side("right")
        self._arm_to_side("left")

    def _dance_macarena(self):
        """Arms out / flip / hips / turn sequence."""
        DELAY = 0.25
        self._arm_to_side("right")
        self._arm_to_side("left")
        time.sleep(0.5)
        # Arms out in front
        for _ in range(5):
            self._arm_move_inc("forward", "right")
            self._arm_move_inc("forward", "left")
            time.sleep(DELAY)
        time.sleep(0.3)
        # Arms up
        for _ in range(5):
            self._arm_move_inc("up", "right")
            self._arm_move_inc("up", "left")
            time.sleep(DELAY)
        time.sleep(0.3)
        # Arms to hips (in + down)
        for _ in range(4):
            self._arm_move_inc("in", "right")
            self._arm_move_inc("in", "left")
            self._arm_move_inc("down", "right")
            self._arm_move_inc("down", "left")
            time.sleep(DELAY)
        time.sleep(0.3)
        # Turn (quarter turn)
        with self.lock:
            self.client.Move(0, 0, 0.5)
        time.sleep(1.0)
        with self.lock:
            self.client.Move(0, 0, 0)
        time.sleep(0.3)
        self._arm_to_side("right")
        self._arm_to_side("left")

    def _dance_twist(self):
        """Body rotate left-right with arms."""
        DELAY = 0.2
        self._arm_to_side("right")
        self._arm_to_side("left")
        time.sleep(0.5)
        # Arms up
        for _ in range(5):
            self._arm_move_inc("up", "right")
            self._arm_move_inc("up", "left")
            time.sleep(DELAY)
        # Twist left-right
        for _ in range(3):
            with self.lock:
                self.client.Move(0, 0, 0.4)
            self.rotate_head(0.0, 0.4)
            time.sleep(0.6)
            with self.lock:
                self.client.Move(0, 0, -0.4)
            self.rotate_head(0.0, -0.4)
            time.sleep(0.6)
        with self.lock:
            self.client.Move(0, 0, 0)
        self.rotate_head(0.0, 0.0)
        time.sleep(0.3)
        self._arm_to_side("right")
        self._arm_to_side("left")

    def _dance_bow(self):
        """Head down deeply, pause, head up."""
        self.rotate_head(0.8, 0.0)
        time.sleep(2.0)
        self.rotate_head(0.0, 0.0)

    def _dance_chicken(self):
        """Arms flap in/out rapidly + head bobs."""
        DELAY = 0.15
        self._arm_to_side("right")
        self._arm_to_side("left")
        time.sleep(0.5)
        for _ in range(5):
            # Flap out
            for _ in range(3):
                self._arm_move_inc("out", "right")
                self._arm_move_inc("out", "left")
                time.sleep(DELAY)
            self.rotate_head(0.3, 0.0)
            # Flap in
            for _ in range(3):
                self._arm_move_inc("in", "right")
                self._arm_move_inc("in", "left")
                time.sleep(DELAY)
            self.rotate_head(-0.1, 0.0)
        self.rotate_head(0.0, 0.0)
        time.sleep(0.3)
        self._arm_to_side("right")
        self._arm_to_side("left")

    def _dance_disco(self):
        """Point up-right, up-left alternating + stepping."""
        DELAY = 0.2
        self._arm_to_side("right")
        self._arm_to_side("left")
        time.sleep(0.5)
        for _ in range(3):
            # Right arm up and out (point right)
            for _ in range(6):
                self._arm_move_inc("up", "right")
                self._arm_move_inc("out", "right")
                time.sleep(DELAY)
            self.rotate_head(-0.2, -0.3)
            with self.lock:
                self.client.Move(0, -0.2, 0)
            time.sleep(0.5)
            self._arm_to_side("right")
            # Left arm up and out (point left)
            for _ in range(6):
                self._arm_move_inc("up", "left")
                self._arm_move_inc("out", "left")
                time.sleep(DELAY)
            self.rotate_head(-0.2, 0.3)
            with self.lock:
                self.client.Move(0, 0.2, 0)
            time.sleep(0.5)
            self._arm_to_side("left")
        with self.lock:
            self.client.Move(0, 0, 0)
        self.rotate_head(0.0, 0.0)

    def _dance_salsa(self):
        """Side-to-side strafe with opposite arm raises."""
        DELAY = 0.2
        self._arm_to_side("right")
        self._arm_to_side("left")
        time.sleep(0.5)
        for _ in range(3):
            # Strafe right, raise left arm
            for _ in range(5):
                self._arm_move_inc("up", "left")
                time.sleep(DELAY)
            with self.lock:
                self.client.Move(0, -0.3, 0)
            time.sleep(0.7)
            self._arm_to_side("left")
            # Strafe left, raise right arm
            for _ in range(5):
                self._arm_move_inc("up", "right")
                time.sleep(DELAY)
            with self.lock:
                self.client.Move(0, 0.3, 0)
            time.sleep(0.7)
            self._arm_to_side("right")
        with self.lock:
            self.client.Move(0, 0, 0)

    def _dance_celebrate(self):
        """Both arms up + spin + head up."""
        DELAY = 0.2
        self._arm_to_side("right")
        self._arm_to_side("left")
        time.sleep(0.5)
        # Both arms up
        for _ in range(7):
            self._arm_move_inc("up", "right")
            self._arm_move_inc("up", "left")
            time.sleep(DELAY)
        self.rotate_head(-0.3, 0.0)
        time.sleep(0.5)
        # Spin
        with self.lock:
            self.client.Move(0, 0, 0.5)
        time.sleep(3.0)
        with self.lock:
            self.client.Move(0, 0, 0)
        time.sleep(0.3)
        self.rotate_head(0.0, 0.0)
        self._arm_to_side("right")
        self._arm_to_side("left")

    def _dance_karate(self):
        """Arms forward thrust + step forward."""
        DELAY = 0.2
        self._arm_to_side("right")
        self._arm_to_side("left")
        time.sleep(0.5)
        # Right arm thrust forward
        for _ in range(5):
            self._arm_move_inc("forward", "right")
            self._arm_move_inc("up", "right")
            time.sleep(DELAY)
        with self.lock:
            self.client.Move(0.3, 0, 0)
        time.sleep(0.6)
        with self.lock:
            self.client.Move(0, 0, 0)
        time.sleep(0.3)
        self._arm_to_side("right")
        # Left arm thrust forward
        for _ in range(5):
            self._arm_move_inc("forward", "left")
            self._arm_move_inc("up", "left")
            time.sleep(DELAY)
        with self.lock:
            self.client.Move(0.3, 0, 0)
        time.sleep(0.6)
        with self.lock:
            self.client.Move(0, 0, 0)
        time.sleep(0.3)
        self._arm_to_side("left")

    # ── Go to object ─────────────────────────────────────────────────────

    def go_to_object(self, target, stop_distance=0.5):
        """Walk toward a detected object with obstacle avoidance.

        Args:
            target: Object class name or person name to walk toward
            stop_distance: How far (meters) from the object to stop
        """
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
        """Walk toward target with depth-based obstacle avoidance."""
        TIMEOUT = 30.0
        OBSTACLE_DIST = 0.8   # meters — closer than this is an obstacle
        MIN_FWD = 0.15
        MAX_FWD = 0.5
        STEER_SPEED = 0.35
        LOST_PATIENCE = 3.0   # give up after losing target for this long

        start = time.time()
        lost_since = None

        while self.follow_active and (time.time() - start) < TIMEOUT:
            if not self.camera_node:
                time.sleep(0.1)
                continue

            det = self._find_target_detection()

            # Handle lost target
            if det is None:
                if lost_since is None:
                    lost_since = time.time()
                with self.lock:
                    self.client.Move(0, 0, 0)
                if time.time() - lost_since > LOST_PATIENCE:
                    print(f"[Robot] Lost target {target}, giving up")
                    break
                time.sleep(0.2)
                continue
            lost_since = None

            frame = self.camera_node._raw_frame
            if frame is None:
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            cx = det['center'][0]
            distance = det.get('distance_m')

            # ── Arrival check (depth) ──
            if distance is not None and distance <= stop_distance:
                print(f"[Robot] Arrived at {target} ({distance:.1f}m, stop={stop_distance:.1f}m)")
                break

            # ── Arrival check (bbox fallback when no depth) ──
            bbox_w = det['bbox'][2] - det['bbox'][0]
            bbox_ratio = bbox_w / w
            bbox_threshold = max(0.30, 0.55 - stop_distance * 0.15)
            if distance is None and bbox_ratio >= bbox_threshold:
                print(f"[Robot] Arrived at {target} (bbox {bbox_ratio:.2f})")
                break

            # ── Steering toward target ──
            err_x = (cx - w / 2) / (w / 2)
            rot_speed = -err_x * 0.5 if abs(err_x) > 0.10 else 0.0

            # ── Forward speed (proportional to remaining distance) ──
            if distance is not None:
                remaining = distance - stop_distance
                if remaining <= 0:
                    break
                fwd_speed = min(MAX_FWD, max(MIN_FWD, remaining * 0.35))
            else:
                fwd_speed = 0.3

            # ── Obstacle avoidance using depth map ──
            depth_map = self.camera_node._depth_map
            if depth_map is not None:
                dh, dw = depth_map.shape
                # Scan horizontal strip in middle-lower portion (path ahead)
                strip_y1 = int(dh * 0.4)
                strip_y2 = int(dh * 0.7)
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
                    # Obstacle blocking path — steer toward clearer side
                    if left_d > right_d:
                        rot_speed = STEER_SPEED   # turn left
                    else:
                        rot_speed = -STEER_SPEED  # turn right
                    fwd_speed = min(fwd_speed, 0.2)
                    print(f"[Robot] Obstacle {center_d:.1f}m, steering {'left' if rot_speed > 0 else 'right'}")

            with self.lock:
                self.client.Move(fwd_speed, 0.0, rot_speed)
            time.sleep(0.05)

        # Done — stop walking, keep tracking
        with self.lock:
            self.client.Move(0, 0, 0)
        self.follow_active = False
        print(f"[Robot] Go-to complete: {target}")

    # ── Dances ───────────────────────────────────────────────────────────

    def do_dance(self, dance_name=None):
        """Execute a dance. Runs in background thread.

        Prefers SDK dances (smooth, full-body) over custom choreography.
        SDK upper-body dances use kDance (API 2016).
        SDK whole-body dances use kWholeBodyDance (API 2029).
        """
        def _dance():
            name = (dance_name or "robot").lower()
            try:
                from booster_robotics_sdk_python import B1LocoApiId

                # SDK upper-body dances (DanceId enum)
                sdk_dance_map = {
                    "newyear": 0, "new year": 0,
                    "nezha": 1,
                    "future": 2,
                    "dab": 3,
                    "ultraman": 4,
                    "respect": 5,
                    "cheer": 6, "cheering": 6, "celebrate": 6,
                    "luckycat": 7, "lucky cat": 7,
                }

                # SDK whole-body dances (WholeBodyDanceId enum)
                sdk_wholebody_map = {
                    "arabic": 0, "salsa": 0,
                    "michael jackson": 1, "michael": 1, "mj": 1,
                    "michael2": 2, "mj2": 2,
                    "michael3": 3, "mj3": 3,
                    "moonwalk": 4,
                    "kick": 5, "boxing": 5,
                    "roundhouse": 6, "karate": 6, "roundhouse kick": 6,
                }

                if name in sdk_wholebody_map:
                    print(f"[Robot] SDK whole-body dance: {name}")
                    self.client.SendApiRequest(
                        B1LocoApiId(2029),  # kWholeBodyDance
                        json.dumps({"dance_id": sdk_wholebody_map[name]})
                    )
                    return

                if name in sdk_dance_map:
                    print(f"[Robot] SDK dance: {name}")
                    self.client.SendApiRequest(
                        B1LocoApiId.kDance,
                        json.dumps({"dance_id": sdk_dance_map[name]})
                    )
                    return
            except ImportError:
                pass

            # Custom choreography fallback for dances without SDK equivalent
            custom_dances = {
                "macarena": self._dance_macarena,
                "twist": self._dance_twist,
                "bow": self._dance_bow,
                "chicken": self._dance_chicken,
                "disco": self._dance_disco,
            }
            if name in custom_dances:
                print(f"[Robot] Custom dance: {name}")
                custom_dances[name]()
                return

            # Default robot dance
            print(f"[Robot] Custom dance")
            self._robot_dance()

        threading.Thread(target=_dance, daemon=True).start()

    def _robot_dance(self):
        """A simple custom dance using head and arms."""
        DELAY = 0.2
        # Head bobs + arm waves
        for _ in range(2):
            self.rotate_head(0.0, -0.5)
            self._arm_to_side("right")
            time.sleep(0.5)
            for _ in range(5):
                self._arm_move_inc("up", "right")
                time.sleep(DELAY)
            self.rotate_head(0.0, 0.5)
            self._arm_to_side("left")
            time.sleep(0.5)
            for _ in range(5):
                self._arm_move_inc("up", "left")
                time.sleep(DELAY)
        # Both arms up
        for _ in range(4):
            self._arm_move_inc("up", "right")
            self._arm_move_inc("up", "left")
            time.sleep(DELAY)
        self.rotate_head(-0.2, 0.0)
        time.sleep(1.5)
        # Reset
        self.rotate_head(0.0, 0.0)
        self._arm_to_side("right")
        self._arm_to_side("left")

    def do_wave(self):
        """Wave hand."""
        def _wave():
            with self.lock:
                self.client.WaveHand(B1HandAction.kHandOpen)
        threading.Thread(target=_wave, daemon=True).start()

    def do_handshake(self):
        """Offer handshake."""
        def _hs():
            with self.lock:
                self.client.Handshake(B1HandAction.kHandOpen)
        threading.Thread(target=_hs, daemon=True).start()

    # ── Arm helpers ──────────────────────────────────────────────────────

    def _arm_to_side(self, hand):
        is_left = hand == "left"
        y_sign = 1 if is_left else -1
        hand_idx = B1HandIndex.kLeftHand if is_left else B1HandIndex.kRightHand
        posture = Posture()
        posture.position = Position(0.35, y_sign * 0.25, 0.1)
        posture.orientation = Orientation(-y_sign * 1.57, -1.57, 0.0)
        with self.lock:
            self.client.MoveHandEndEffectorV2(posture, 800, hand_idx)
        if is_left:
            self.left_arm_pos = [0.35, 0.25, 0.1]
        else:
            self.right_arm_pos = [0.35, -0.25, 0.1]

    def _arm_move_inc(self, direction, hand):
        STEP = 0.03
        is_left = hand == "left"
        pos = self.left_arm_pos if is_left else self.right_arm_pos
        hand_idx = B1HandIndex.kLeftHand if is_left else B1HandIndex.kRightHand
        y_sign = 1 if is_left else -1

        if direction == "up":
            pos[2] = min(pos[2] + STEP, 0.35)
        elif direction == "down":
            pos[2] = max(pos[2] - STEP, -0.10)
        elif direction == "forward":
            pos[0] = min(pos[0] + STEP, 0.40)
        elif direction == "back":
            pos[0] = max(pos[0] - STEP, 0.20)
        elif direction == "out":
            pos[1] = pos[1] + STEP * y_sign
        elif direction == "in":
            pos[1] = pos[1] - STEP * y_sign

        posture = Posture()
        posture.position = Position(pos[0], pos[1], pos[2])
        posture.orientation = Orientation(-y_sign * 1.57, -1.57, 0.0)
        with self.lock:
            self.client.MoveHandEndEffectorV2(posture, 300, hand_idx)

    # ── Stop all ─────────────────────────────────────────────────────────

    def stop_all(self):
        """Stop everything — following, tracking, movement."""
        self.stop_follow()
        self.stop_tracking()
        self.stop_movement()
        self.rotate_head(0.0, 0.0)
        print("[Robot] All stopped")

    def shutdown(self):
        self.stop_all()


# ── Command Dispatcher ───────────────────────────────────────────────────────


class CommandDispatcher:
    """Parses Gemini output transcription and dispatches robot commands."""

    def __init__(self, robot: RobotController):
        self.robot = robot
        self._last_cmd_time = 0
        self._cmd_cooldown = 2.0  # seconds between commands

    def check_transcript(self, text):
        """Check text for command triggers. Returns command name or None."""
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
        """Execute a parsed command."""
        # For go_to, regex groups are: (distance, target) — handle specially
        if cmd == "go_to":
            groups = match.groups()
            dist_str = groups[0]  # e.g. "0.5" or None
            go_target = groups[1]  # e.g. "chair"
            stop_dist = float(dist_str) if dist_str else 0.5
            target = go_target or "person"
            print(f"[CMD] Executing: go_to {target} (stop_distance={stop_dist}m)")
            add_transcript("Action", f"go_to {target} ({stop_dist}m)")
            self.robot.go_to_object(target, stop_distance=stop_dist)
            return

        # Extract optional target from regex groups
        target = None
        for g in match.groups():
            if g:
                target = g.strip()
                break

        print(f"[CMD] Executing: {cmd} (target={target})")
        add_transcript("Action", f"{cmd}" + (f" ({target})" if target else ""))

        if cmd == "follow":
            self.robot.start_follow(target)
        elif cmd == "stop":
            self.robot.stop_all()
        elif cmd == "dance":
            self.robot.do_dance(target)
        elif cmd.startswith("dance_"):
            self.robot.do_dance(cmd.replace("dance_", ""))
        elif cmd == "wave":
            self.robot.do_wave()
        elif cmd == "handshake":
            self.robot.do_handshake()
        elif cmd == "dab":
            self.robot.do_dab()
        elif cmd == "flex":
            self.robot.do_flex()
        elif cmd == "track":
            self.robot.start_tracking(target)
        elif cmd == "look_left":
            self.robot.rotate_head(0.0, 0.5)
        elif cmd == "look_right":
            self.robot.rotate_head(0.0, -0.5)
        elif cmd == "look_up":
            self.robot.rotate_head(-0.3, 0.0)
        elif cmd == "look_down":
            self.robot.rotate_head(0.5, 0.0)
        elif cmd == "look_center":
            self.robot.rotate_head(0.0, 0.0)
        elif cmd == "turn_left":
            self.robot.turn_left()
        elif cmd == "turn_right":
            self.robot.turn_right()
        elif cmd == "turn_around":
            self.robot.turn_around()
        elif cmd == "forward":
            self.robot.forward()
        elif cmd == "backward":
            self.robot.backward()
        elif cmd == "strafe_left":
            self.robot.strafe_left()
        elif cmd == "strafe_right":
            self.robot.strafe_right()
        elif cmd == "approach":
            self.robot.approach()
        elif cmd == "back_up":
            self.robot.back_up()
        elif cmd == "nod":
            self.robot.nod()
        elif cmd == "head_shake":
            self.robot.head_shake()


# ── Camera + Detection Node ─────────────────────────────────────────────────


class CameraDetectionNode(Node):
    """ROS2 node: camera + stereo depth + YOLO + face recognition."""

    def __init__(self, model_path='yolov8n.pt', confidence=0.5,
                 face_cache=None, enable_faces=True):
        super().__init__('gemini_robot_control')
        self.bridge = CvBridge()

        self.get_logger().info(f'Loading YOLO model: {model_path}')
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
        self.latest_frame = None
        self.latest_detections = []
        self._depth_map = None
        self._raw_frame = None
        self._fps = 0.0
        self._fps_counter = 0
        self._fps_time = time.time()

        self.create_subscription(Image, '/image_left_raw', self._on_image, 10)
        self.create_subscription(CompressedImage, '/booster_video_stream', self._on_compressed, 10)
        self.create_subscription(Image, '/StereoNetNode/stereonet_depth', self._on_depth, 10)

        self._pending_frame = None
        self.create_timer(0.1, self._detect_tick)

    def _convert_image(self, msg):
        try:
            if msg.encoding == 'nv12':
                h, w = msg.height, msg.width
                yuv = np.frombuffer(msg.data, dtype=np.uint8).reshape((int(h * 1.5), w))
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
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
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
        best_dist = 999.0
        best_id = None
        for uid, enc in self._unknown_faces.items():
            dist = float(face_recognition.face_distance([enc], encoding)[0])
            if dist < best_dist:
                best_dist = dist
                best_id = uid
        if best_id is not None and best_dist < 0.5:
            self._unknown_faces[best_id] = encoding
            return best_id
        uid = self._next_unknown_id
        self._next_unknown_id += 1
        self._unknown_faces[uid] = encoding
        return uid

    def _run_face_recognition(self, frame):
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

    def _detect_tick(self):
        frame = self._pending_frame
        if frame is None:
            return
        try:
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

        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')

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


# ── Shared transcript ────────────────────────────────────────────────────────

transcript = deque(maxlen=200)
transcript_lock = threading.Lock()
_camera_node_ref = None
_cmd_dispatcher_ref = None
_session_ref = None       # Gemini Live session (set when connected)
_event_loop_ref = None    # asyncio event loop (for cross-thread sends)


def add_transcript(role, text):
    with transcript_lock:
        transcript.append({"role": role, "text": text, "ts": time.time()})
    if role == "You" and _camera_node_ref:
        _camera_node_ref.try_learn_name_from_transcript(text)


def get_transcript():
    with transcript_lock:
        return list(transcript)


def send_text_to_gemini(text):
    """Send a text message to the Gemini Live session from any thread."""
    session = _session_ref
    loop = _event_loop_ref
    if not session or not loop:
        print("[Chat] No active Gemini session")
        return False

    async def _send():
        try:
            await session.send_client_content(
                turns=[types.Content(
                    role="user",
                    parts=[types.Part(text=text)],
                )],
                turn_complete=True,
            )
        except Exception as e:
            print(f"[Chat] Send error: {e}")

    asyncio.run_coroutine_threadsafe(_send(), loop)
    return True


# ── Web server ───────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>Gemini Robot Control</title>
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
    <div id="header"><span>Gemini Robot Control</span> &mdash; Vision + Voice + Movement</div>

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

    <div id="detections"><h3>Detections</h3><div id="det-list">Waiting...</div></div>
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
      const payload = {action: action};
      if (target) payload.target = target;
      const r = await fetch('/cmd', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await r.json();
      console.log('cmd result:', data);
    } catch(e) { console.error(e); }
  }

  function goToPrompt() {
    const target = prompt('Go to what? (e.g. person, chair, bottle, backpack, a name)');
    if (!target || !target.trim()) return;
    const distStr = prompt('Stop distance in meters? (default 0.5)', '0.5');
    const dist = parseFloat(distStr) || 0.5;
    fetch('/cmd', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: 'go_to', target: target.trim(), distance: dist})
    }).then(r => r.json()).then(d => console.log('go_to:', d)).catch(console.error);
  }

  async function sendChat() {
    const input = document.getElementById('msg-input');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    try {
      await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text})
      });
    } catch(e) { console.error(e); }
  }

  document.getElementById('msg-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') { e.preventDefault(); sendChat(); }
  });

  // WASD keyboard controls for movement
  const wasdMap = {w:'forward', s:'backward', a:'strafe_left', d:'strafe_right', q:'turn_left', e:'turn_right'};
  document.addEventListener('keydown', function(e) {
    if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA') return;
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
          const cls = m.role === 'You' ? 'you' : m.role === 'Action' ? 'action' : m.role === 'System' ? 'system' : 'robot';
          return '<div class="msg ' + cls + '"><div class="role">' + m.role + '</div>' + m.text + '</div>';
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
              nameHtml = '<span class="unknown">' + d.name + '</span> '
                + '<input class="name-input" placeholder="Name..." id="ni_' + d.unknown_id + '">'
                + '<button class="save-btn" onclick="saveFace(' + d.unknown_id + ')">Save</button>';
            } else {
              nameHtml = '<span class="name">' + d.name + '</span> ';
            }
          }
          return '<div class="det"><span>' + nameHtml + '<span class="cls">' + d.class + '</span> <span class="conf">' + (d.confidence*100).toFixed(0) + '%</span></span><span class="dist">' + dist + '</span></div>';
        }).join('');
      }
    } catch(e) {}
    setTimeout(pollDetections, 300);
  }
  pollDetections();

  async function saveFace(unknownId) {
    const input = document.getElementById('ni_' + unknownId);
    if (!input || !input.value.trim()) return;
    try {
      await fetch('/save_face', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({unknown_id: unknownId, name: input.value.trim()})
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
          '<div class="known"><span class="kname">' + f.name + '</span><button class="del-btn" onclick="deleteFace(\\''+f.name+'\\')">x</button></div>'
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
            if not node:
                self._json_response({'error': 'not ready'}, 503)
                return
            uid = body.get('unknown_id')
            name = body.get('name', '').strip()
            if not name or uid is None:
                self._json_response({'error': 'need unknown_id and name'}, 400)
                return
            ok = node.save_unknown_face(int(uid), name)
            self._json_response({'ok': ok, 'name': name})

        elif self.path == '/delete_face':
            if not node or not node.face_cache:
                self._json_response({'error': 'not ready'}, 503)
                return
            name = body.get('name', '').strip()
            if not name:
                self._json_response({'error': 'need name'}, 400)
                return
            node.face_cache.delete_face(name)
            self._json_response({'ok': True})
        else:
            self.send_error(404)

    def _handle_cmd(self, action, robot, body=None):
        if not robot:
            return {'status': 'error', 'message': 'Robot not connected'}

        add_transcript("Action", action)

        # Extract optional target and distance from request body
        target = body.get('target') if isinstance(body, dict) else None
        distance = body.get('distance') if isinstance(body, dict) else None

        if action == 'follow':
            robot.start_follow(target)
            return {'status': 'ok', 'action': 'follow'}
        elif action == 'track':
            robot.start_tracking(target)
            return {'status': 'ok', 'action': 'track'}
        elif action == 'stop':
            robot.stop_all()
            return {'status': 'ok', 'action': 'stop'}
        elif action.startswith('go_to'):
            obj = target or action.replace('go_to_', '').replace('go_to', 'person')
            stop_dist = float(distance) if distance else 0.5
            robot.go_to_object(obj or 'person', stop_distance=stop_dist)
            return {'status': 'ok', 'action': 'go_to', 'target': obj, 'distance': stop_dist}
        elif action == 'dance':
            robot.do_dance()
            return {'status': 'ok', 'action': 'dance'}
        elif action.startswith('dance_'):
            robot.do_dance(action.replace('dance_', ''))
            return {'status': 'ok', 'action': action}
        elif action == 'wave':
            robot.do_wave()
            return {'status': 'ok', 'action': 'wave'}
        elif action == 'handshake':
            robot.do_handshake()
            return {'status': 'ok', 'action': 'handshake'}
        elif action == 'dab':
            robot.do_dab()
            return {'status': 'ok', 'action': 'dab'}
        elif action == 'flex':
            robot.do_flex()
            return {'status': 'ok', 'action': 'flex'}
        elif action == 'nod':
            robot.nod()
            return {'status': 'ok', 'action': 'nod'}
        elif action == 'head_shake':
            robot.head_shake()
            return {'status': 'ok', 'action': 'head_shake'}
        elif action == 'look_up':
            robot.rotate_head(-0.3, 0.0)
            return {'status': 'ok', 'action': 'look_up'}
        elif action == 'look_down':
            robot.rotate_head(0.5, 0.0)
            return {'status': 'ok', 'action': 'look_down'}
        elif action == 'look_left':
            robot.rotate_head(0.0, 0.5)
            return {'status': 'ok', 'action': 'look_left'}
        elif action == 'look_right':
            robot.rotate_head(0.0, -0.5)
            return {'status': 'ok', 'action': 'look_right'}
        elif action == 'look_center':
            robot.rotate_head(0.0, 0.0)
            return {'status': 'ok', 'action': 'look_center'}
        elif action == 'forward':
            robot.forward()
            return {'status': 'ok', 'action': 'forward'}
        elif action == 'backward':
            robot.backward()
            return {'status': 'ok', 'action': 'backward'}
        elif action == 'strafe_left':
            robot.strafe_left()
            return {'status': 'ok', 'action': 'strafe_left'}
        elif action == 'strafe_right':
            robot.strafe_right()
            return {'status': 'ok', 'action': 'strafe_right'}
        elif action == 'turn_left':
            robot.turn_left()
            return {'status': 'ok', 'action': 'turn_left'}
        elif action == 'turn_right':
            robot.turn_right()
            return {'status': 'ok', 'action': 'turn_right'}
        elif action == 'turn_around':
            robot.turn_around()
            return {'status': 'ok', 'action': 'turn_around'}
        else:
            return {'status': 'error', 'message': f'Unknown action: {action}'}


def start_web_server(camera_node, robot_controller, host, port):
    WebHandler.camera_node = camera_node
    WebHandler.robot_controller = robot_controller
    httpd = HTTPServer((host, port), WebHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


# ── Async tasks ──────────────────────────────────────────────────────────────


def _amplify_audio(data, gain):
    """Apply software gain to 16-bit PCM audio. Clips to int16 range."""
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    samples *= gain
    np.clip(samples, -32768, 32767, out=samples)
    return samples.astype(np.int16).tobytes()


async def send_audio(session, pya, mic_device=None, mic_gain=1.0):
    kwargs = dict(
        format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
        input=True, frames_per_buffer=CHUNK_SIZE,
    )
    if mic_device is not None:
        kwargs['input_device_index'] = mic_device
    stream = pya.open(**kwargs)
    loop = asyncio.get_event_loop()
    apply_gain = mic_gain > 1.01  # skip processing if gain is ~1.0
    try:
        while True:
            data = await loop.run_in_executor(
                None, lambda: stream.read(CHUNK_SIZE, exception_on_overflow=False),
            )
            if apply_gain:
                data = _amplify_audio(data, mic_gain)
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


async def receive_responses(session, pya, cmd_dispatcher):
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
                        # Check for action commands in robot speech
                        cmd_dispatcher.check_transcript(txt)
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()


# ── Main session ─────────────────────────────────────────────────────────────


SYSTEM_INSTRUCTION = """You are a Booster K1 humanoid robot with stereo vision cameras, face recognition, \
and full body control. Video frames are streamed to you with real-time object detection overlays — \
each detected object has a bounding box, class label, confidence score, and distance in meters. \
People you recognize are labeled with their name; unknown people are labeled 'Unknown #N'.

You can PHYSICALLY ACT by saying certain trigger phrases in your responses. When you decide to act, \
naturally include one of these phrases — your body will respond automatically:

FOLLOWING / TRACKING:
- "I'll follow you" or "Following you now" — walks toward and follows the person, turning body to keep them in view
- "I'll track that" or "Tracking the [object]" — moves head AND body to keep an object centered
- "Stopping now" or "I'll stop" — stops all movement and tracking

GO TO OBJECTS (with distance awareness and obstacle avoidance):
- "Going to the [object]" or "Walking to the [object]" — walks toward a detected object, stops 0.5m away by default
- "Going to 0.5 meters from the [object]" — walks to a specific distance from the object (e.g. 0.5m, 1m, 2m)
- "Moving over to the [person name]" — walks toward a specific named person
- You can specify any distance: "Walking to 1 meter from the chair", "Going to 0.3 meters from the bottle"
- The robot automatically detects and avoids obstacles in its path while navigating
- Works with any detected object: chair, bottle, backpack, laptop, cup, person, etc.

LOOKING / HEAD CONTROL:
- "Looking left" or "I'll look left" — turns head left
- "Looking right" or "I'll look right" — turns head right
- "Looking up" — tilts head up
- "Looking down" — tilts head down
- "Looking forward" or "Looking straight" — centers head

MOVEMENT:
- "Walking forward" — walks forward briefly
- "Walking backward" or "Moving backward" — walks backward
- "Strafing left" or "Moving sideways left" — sidesteps left
- "Strafing right" or "Moving sideways right" — sidesteps right
- "Turning left" — rotates body left
- "Turning right" — rotates body right
- "Turning around" — turns 180 degrees
- "Coming closer" or "Approaching" — walks forward briefly
- "Backing up" or "Stepping back" — backs up

DANCES & GESTURES (you have many built-in dances!):
- "Let me dance" or "Here's a dance" — does a robot dance
- "New year dance!" or "Nezha dance!" or "Future dance!" — SDK choreographed dances
- "Moonwalk!" — real moonwalk (whole body)
- "Michael Jackson dance!" — Michael Jackson dance (whole body)
- "Kick!" or "Boxing kick!" — boxing-style kick (whole body)
- "Roundhouse kick!" or "Karate!" — roundhouse kick (whole body)
- "Salsa!" or "Arabic dance!" — Arabic/salsa dance (whole body)
- "Ultraman!" — Ultraman gesture pose
- "Respect!" — respect gesture
- "Celebrate!" or "Cheering!" — cheering celebration
- "Lucky cat!" — lucky cat gesture
- "Macarena!" — does the macarena
- "Twist!" or "Twisting!" — does the twist dance
- "Disco!" — disco pointing moves
- "Chicken dance!" — chicken dance (arm flaps + head bobs)
- "Bow!" or "Take a bow!" — bows deeply
- "I'll wave" or "Waving!" — waves hand
- "Let me shake hands" — offers a handshake
- "Dabbing!" or "Let me dab" — does a dab pose
- "Flexing!" or "Let me flex" — does a flex/victory pose

HEAD EXPRESSIONS:
- "Nodding" — nods yes
- "Shaking my head" — shakes head no

IMPORTANT RULES:
- When someone says "follow me", respond with "I'll follow you!" to trigger following.
- When someone says "go to the chair" or "walk to that person", respond with "Going to the chair!" etc.
- When someone says "go 0.5 meters from the chair", respond with "Going to 0.5 meters from the chair!" to specify distance.
- When someone says "look left" or "look right", respond with "Looking left!" etc.
- When someone says "turn left/right", respond with "Turning left!" etc.
- When someone asks you to dance, say "Let me dance!" or "Here's a dance for you!"
- When someone asks you to dab, say "Dabbing!" — when they ask you to flex, say "Flexing!"
- When you see someone wave, wave back by saying "Waving!"
- When you see an unknown person, ask their name in a friendly way.
- When you see a recognized person, greet them by name warmly.
- Keep responses short and conversational.
- You can combine speech with actions naturally, e.g. "Sure, I'll follow you! Let's go!"
- Only trigger actions when explicitly asked or when socially appropriate (like waving back).
- When tracking, your body automatically turns to follow the target — you don't just move your head.
- You understand distances — when asked to go to an object, you can see how far it is in the detection overlay.
"""


async def run_session(api_key, camera_node, cmd_dispatcher, voice, frame_interval,
                      mic_device=None, mic_gain=1.0):
    global _session_ref, _event_loop_ref

    client = genai.Client(api_key=api_key)

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            ),
        ),
        system_instruction=SYSTEM_INSTRUCTION,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    print("Connecting to Gemini Live...")
    pya = pyaudio.PyAudio()
    _event_loop_ref = asyncio.get_event_loop()

    try:
        async with client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            config=config,
        ) as session:
            _session_ref = session
            print("Connected! Talk to the robot — ask it to follow you, dance, wave, etc.")
            print("You can also type in the web UI chat box.")
            print("Press Ctrl+C to stop.\n")

            tasks = [
                asyncio.create_task(send_audio(session, pya, mic_device, mic_gain)),
                asyncio.create_task(send_video(session, camera_node, frame_interval)),
                asyncio.create_task(receive_responses(session, pya, cmd_dispatcher)),
            ]

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


# ── Entrypoint ───────────────────────────────────────────────────────────────


def main():
    global _camera_node_ref, _cmd_dispatcher_ref

    parser = argparse.ArgumentParser(
        description='Gemini Live + Robot Control (follow, track, dance, wave)'
    )
    parser.add_argument(
        'interface', type=str,
        help='Network interface for robot SDK (e.g. eth0)',
    )
    parser.add_argument(
        '--api-key', type=str, default=None,
        help='Gemini API key (or set GEMINI_API_KEY env var)',
    )
    parser.add_argument(
        '--voice', type=str, default='Puck',
        choices=['Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'],
    )
    parser.add_argument('--frame-interval', type=float, default=1.0)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--face-tolerance', type=float, default=0.6)
    parser.add_argument('--no-faces', action='store_true')
    parser.add_argument(
        '--follow-distance', type=float, default=1.0,
        help='Target follow distance in meters (default: 1.0)',
    )
    parser.add_argument(
        '--mic-gain', type=float, default=3.0,
        help='Software mic gain multiplier (default: 3.0, try 2-6)',
    )
    parser.add_argument(
        '--mic-device', type=int, default=None,
        help='PyAudio input device index (default: auto-detect iFlytek, or system default)',
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("Error: provide --api-key or set GEMINI_API_KEY env variable")
        sys.exit(1)

    os.environ.pop('GOOGLE_API_KEY', None)
    os.environ.pop('GEMINI_API_KEY', None)

    print("=" * 60)
    print("Gemini Robot Control")
    print("  Vision + Voice + Follow + Dance + Head Tracking")
    print("=" * 60)

    # Initialize robot SDK
    print(f"Connecting to robot via {args.interface}...")
    ChannelFactory.Instance().Init(0, args.interface)
    loco_client = B1LocoClient()
    loco_client.Init()
    time.sleep(1.0)

    # Switch to walking mode
    print("Switching to walking mode...")
    loco_client.ChangeMode(RobotMode.kPrepare)
    time.sleep(2.0)
    loco_client.ChangeMode(RobotMode.kWalking)
    time.sleep(1.0)
    print("Robot ready in walking mode")

    # Robot controller
    robot = RobotController(loco_client)
    robot.follow_target_distance = args.follow_distance

    # Face cache
    face_cache = None
    if not args.no_faces:
        face_cache = FaceCache(tolerance=args.face_tolerance)

    # Start ROS2 + detection node
    rclpy.init()
    camera_node = CameraDetectionNode(
        model_path=args.model, confidence=args.confidence,
        face_cache=face_cache, enable_faces=not args.no_faces,
    )
    _camera_node_ref = camera_node
    robot.set_camera_node(camera_node)

    ros_thread = threading.Thread(target=rclpy.spin, args=(camera_node,), daemon=True)
    ros_thread.start()

    # Command dispatcher
    cmd_dispatcher = CommandDispatcher(robot)
    _cmd_dispatcher_ref = cmd_dispatcher

    # Start web server
    httpd = start_web_server(camera_node, robot, '0.0.0.0', args.port)
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

    # Auto-detect mic device if not specified
    mic_device = args.mic_device
    if mic_device is None:
        pya_tmp = pyaudio.PyAudio()
        for i in range(pya_tmp.get_device_count()):
            info = pya_tmp.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0 and 'xfm' in info['name'].lower():
                mic_device = i
                print(f"Auto-detected mic: [{i}] {info['name']}")
                break
        pya_tmp.terminate()
    if mic_device is not None:
        print(f"Mic device: {mic_device}, gain: {args.mic_gain}x")
    else:
        print(f"Mic device: system default, gain: {args.mic_gain}x")

    try:
        asyncio.run(run_session(
            api_key, camera_node, cmd_dispatcher, args.voice, args.frame_interval,
            mic_device=mic_device, mic_gain=args.mic_gain,
        ))
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        robot.shutdown()
        camera_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
