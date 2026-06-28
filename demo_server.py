"""Minimal demo server — serves demo.html and mints LiveKit tokens.

Usage:
    python demo_server.py          # http://localhost:8888
    python demo_server.py --port 9000

No dependencies beyond what livekit-agents already installs.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

load_dotenv(".env.local")

LIVEKIT_URL        = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY    = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]
ROOM_NAME          = os.environ.get("LIVEKIT_ROOM", "rax-demo")
HTML_FILE          = Path(__file__).parent / "demo.html"


def _make_token(identity: str) -> str:
    from livekit.api import AccessToken, VideoGrants
    return (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(
            room_join=True,
            room=ROOM_NAME,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        ))
        .to_jwt()
    )


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request noise

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/token":
            identity = f"user-{uuid.uuid4().hex[:6]}"
            token = _make_token(identity)
            body = json.dumps({
                "token": token,
                "url": LIVEKIT_URL,
                "room": ROOM_NAME,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path in ("/", "/demo.html"):
            html = HTML_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        else:
            self.send_response(404)
            self.end_headers()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8888)
    args = p.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), _Handler)
    print(f"\n  RAX Demo UI → http://localhost:{args.port}")
    print(f"  Room        → {ROOM_NAME}")
    print(f"  LiveKit     → {LIVEKIT_URL}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
