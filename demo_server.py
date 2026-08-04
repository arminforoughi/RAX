"""Minimal demo server — serves demo.html, mints LiveKit tokens, dispatches agent.

Usage:
    python demo_server.py          # http://localhost:8888
    python demo_server.py --port 9000

No dependencies beyond what livekit-agents already installs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv(".env.local")

LIVEKIT_URL        = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY    = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]
ROOM_NAME          = os.environ.get("LIVEKIT_ROOM", "rax-demo")
AGENT_NAME         = "rax-gaze-agent"
HTML_FILE          = Path(__file__).parent / "demo.html"

# background asyncio loop for agent dispatch calls
_loop: asyncio.AbstractEventLoop | None = None


def _start_async_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    _loop.run_forever()


threading.Thread(target=_start_async_loop, daemon=True).start()


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


async def _dispatch_agent_async():
    """Tell LiveKit to dispatch the rax-gaze-agent to our room."""
    from livekit.api import LiveKitAPI, CreateAgentDispatchRequest
    try:
        async with LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET) as lk:
            resp = await lk.agent_dispatch.create_dispatch(
                CreateAgentDispatchRequest(agent_name=AGENT_NAME, room=ROOM_NAME)
            )
            print(f"  [dispatch] agent dispatched: {resp}")
    except Exception as e:
        print(f"  [dispatch] warning: {e}")


def _dispatch_agent():
    if _loop:
        asyncio.run_coroutine_threadsafe(_dispatch_agent_async(), _loop)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/token":
            identity = f"user-{uuid.uuid4().hex[:6]}"
            token = _make_token(identity)
            # Dispatch the agent to the room so it's ready when user joins
            _dispatch_agent()
            body = json.dumps({
                "token": token,
                "url": LIVEKIT_URL,
                "room": ROOM_NAME,
            }).encode()
            self._json(body)

        elif parsed.path == "/status":
            body = json.dumps({
                "room": ROOM_NAME,
                "agent": AGENT_NAME,
                "livekit_url": LIVEKIT_URL,
            }).encode()
            self._json(body)

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

    def _json(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8888)
    args = p.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), _Handler)
    print(f"\n  RAX Demo UI  →  http://localhost:{args.port}")
    print(f"  Room         →  {ROOM_NAME}")
    print(f"  Agent        →  {AGENT_NAME}")
    print(f"  LiveKit      →  {LIVEKIT_URL}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
