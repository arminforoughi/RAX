"""Self-learning robot mission control server.

This server sits in front of the existing RAX stack_mission2.py robot UI and adds:

- a high-level /agent/run endpoint that runs the self-learning agent loop
- proxy routes so the existing camera / 3D UI still works
- endpoints for the Actian memory and Pioneer learner
- an SSE-style /agent/events feed for the new mission-control UI

Usage:
    1. Start stack_mission2.py (it owns the robot on :8484).
    2. python self_learn_server.py    # http://localhost:8686
    3. Open http://localhost:8686 in a browser.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_from_directory

ROBOT_BASE_URL = os.environ.get("ROBOT_BASE_URL", "http://localhost:8484")
SELF_PORT = int(os.environ.get("SELF_PORT", "8686"))
STATIC_DIR = Path(__file__).parent

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Shared event log (used by agent + UI)
# ---------------------------------------------------------------------------
_events: list[dict] = []
_events_lock = threading.Lock()


def emit_event(role: str, text: str, **kwargs) -> None:
    ev = {"id": str(uuid.uuid4())[:8], "t": time.time(), "role": role, "text": text}
    if kwargs:
        ev.update(kwargs)
    with _events_lock:
        _events.append(ev)
        if len(_events) > 1000:
            _events.pop(0)
    print(f"[{role}] {text}")


def get_events(since: float = 0.0) -> list[dict]:
    with _events_lock:
        return [e for e in _events if e["t"] >= since]


# ---------------------------------------------------------------------------
# Lazy imports so the server starts even if optional deps are missing
# ---------------------------------------------------------------------------
_memory = None
_learner = None
_agent_class = None


def get_memory():
    global _memory
    if _memory is None:
        from memory import VectorMemory
        _memory = VectorMemory(emit=emit_event)
    return _memory


def get_learner():
    global _learner
    if _learner is None:
        from learner import PioneerLearner
        _learner = PioneerLearner(emit=emit_event)
    return _learner


def get_agent_class():
    global _agent_class
    if _agent_class is None:
        from agent import SelfLearningAgent
        _agent_class = SelfLearningAgent
    return _agent_class


# ---------------------------------------------------------------------------
# Robot proxy helpers
# ---------------------------------------------------------------------------
def robot_request(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{ROBOT_BASE_URL}{path}"
    timeout = kwargs.pop("timeout", 30)
    stream = kwargs.pop("stream", False)
    return requests.request(method, url, timeout=timeout, stream=stream, **kwargs)


def proxy_response(r: requests.Response, stream: bool = False) -> Response:
    content_type = r.headers.get("Content-Type", "application/octet-stream")
    if stream:
        def gen():
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        return Response(gen(), status=r.status_code, content_type=content_type)
    return Response(r.content, status=r.status_code, content_type=content_type)


# ---------------------------------------------------------------------------
# New UI
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "self_learn_ui.html")


@app.route("/how_it_works.html")
def how_it_works():
    return send_from_directory(STATIC_DIR, "how_it_works.html")


@app.route("/slide.html")
def slide():
    return send_from_directory(STATIC_DIR, "slide.html")


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------
@app.route("/agent/events")
def agent_events():
    since = float(request.args.get("since", "0"))
    return jsonify(get_events(since))


@app.route("/agent/status")
def agent_status():
    try:
        r = robot_request("GET", "/status")
        robot_status = r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        robot_status = {"error": str(e)}
    return jsonify(
        {
            "robot_connected": "error" not in robot_status,
            "robot_status": robot_status,
            "event_count": len(_events),
        }
    )


@app.route("/agent/run", methods=["POST"])
def agent_run():
    body = request.get_json(silent=True) or {}
    task = (body.get("task") or "pick the red cube").strip()
    max_attempts = int(body.get("max_attempts", "3"))

    def loop():
        try:
            agent_cls = get_agent_class()
            agent = agent_cls(
                robot_base=ROBOT_BASE_URL,
                memory=get_memory(),
                learner=get_learner(),
                emit=emit_event,
            )
            agent.run(task, max_attempts=max_attempts)
        except Exception as e:
            emit_event("System", f"Agent crashed: {e}")

    threading.Thread(target=loop, daemon=True).start()
    return jsonify(ok=True, task=task, max_attempts=max_attempts)


@app.route("/agent/memory/recall", methods=["POST"])
def memory_recall():
    body = request.get_json(silent=True) or {}
    task = body.get("task", "")
    k = int(body.get("k", "3"))
    try:
        hits = get_memory().recall(task, k=k)
        return jsonify(ok=True, hits=hits)
    except Exception as e:
        return jsonify(ok=False, error=str(e))


@app.route("/agent/memory/remember", methods=["POST"])
def memory_remember():
    body = request.get_json(silent=True) or {}
    try:
        get_memory().remember(body)
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, error=str(e))


@app.route("/agent/learner/predict", methods=["POST"])
def learner_predict():
    body = request.get_json(silent=True) or {}
    task = body.get("task", "")
    context = body.get("context", "")
    try:
        params = get_learner().predict_params(task, context)
        return jsonify(ok=True, params=params)
    except Exception as e:
        return jsonify(ok=False, error=str(e))


@app.route("/agent/learner/train", methods=["POST"])
def learner_train():
    try:
        result = get_learner().start_fine_tune()
        return jsonify(ok=True, result=result)
    except Exception as e:
        return jsonify(ok=False, error=str(e))


# ---------------------------------------------------------------------------
# Proxy the existing stack_mission2 UI under /stackui
# ---------------------------------------------------------------------------
@app.route("/robot_view")
def robot_view():
    return send_from_directory(STATIC_DIR, "robot_view.html")


@app.route("/stackui", methods=["GET"])
@app.route("/stackui/", methods=["GET"])
def stackui_root():
    r = robot_request("GET", "/")
    return proxy_response(r)


@app.route("/stackui/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def stackui_proxy(path: str):
    return robot_proxy(path)


# ---------------------------------------------------------------------------
# Catch-all proxy to the existing stack_mission2 server
# ---------------------------------------------------------------------------
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def robot_proxy(path: str):
    method = request.method
    data = request.get_data() if method != "GET" else None
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    stream = method == "GET" and (path.startswith("stream") or path == "stream")
    try:
        r = robot_request(
            method,
            f"/{path}",
            params=request.args,
            data=data,
            headers=headers,
            stream=stream,
            timeout=5 if stream else 30,
        )
        return proxy_response(r, stream=stream)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


def main():
    emit_event("System", f"Self-learn server starting. Robot base: {ROBOT_BASE_URL}")
    emit_event("System", f"Open http://localhost:{SELF_PORT}")
    app.run(host="0.0.0.0", port=SELF_PORT, threaded=True, debug=False)


if __name__ == "__main__":
    main()
