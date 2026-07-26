"""Self-learning pick-and-place agent.

The agent decomposes a natural-language task, recalls similar attempts from
Actian memory, asks Pioneer for parameters, executes via the robot server,
observe the outcome, stores the episode, and retries until success.

Band AI integration is optional: if BAND_API_KEY and BAND_AGENT_ID are set,
the agent posts its reasoning to a Band chat room.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Callable

import requests

DEFAULT_ROBOT_BASE = os.environ.get("ROBOT_BASE_URL", "http://localhost:8484")
MAX_POLL_SECONDS = 90


class BandClient:
    """Minimal Band AI remote-agent client. Falls back to local logging."""

    def __init__(self, emit: Callable):
        self.emit = emit
        self.api_key = os.environ.get("BAND_API_KEY")
        self.agent_id = os.environ.get("BAND_AGENT_ID")
        self.room_id = os.environ.get("BAND_ROOM_ID")
        self.base = os.environ.get("BAND_API_URL", "https://api.band.ai")

    def enabled(self) -> bool:
        return bool(self.api_key and self.agent_id)

    def send(self, text: str, message_type: str = "text"):
        self.emit("Band", f"[{message_type}] {text}")
        if not self.enabled():
            return
        try:
            url = f"{self.base}/v1/agents/{self.agent_id}/messages"
            payload = {"content": text, "type": message_type}
            if self.room_id:
                payload["room_id"] = self.room_id
            requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=10,
            )
        except Exception as e:
            self.emit("Band", f"Send failed: {e}")


class SelfLearningAgent:
    def __init__(self, robot_base: str | None = None, memory=None, learner=None, emit=None):
        self.robot_base = (robot_base or DEFAULT_ROBOT_BASE).rstrip("/")
        self.memory = memory
        self.learner = learner
        self.emit = emit or (lambda role, text, **kw: print(f"[{role}] {text}"))
        self.band = BandClient(self.emit)
        self._run_id = str(uuid.uuid4())[:8]

    # ------------------------------------------------------------------
    # Robot helpers
    # ------------------------------------------------------------------
    def _post(self, path: str, **kwargs) -> dict:
        r = requests.post(f"{self.robot_base}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)
        try:
            return r.json()
        except Exception:
            return {"ok": r.ok, "text": r.text}

    def _get(self, path: str) -> dict:
        r = requests.get(f"{self.robot_base}{path}", timeout=10)
        return r.json()

    def _say(self, role: str, text: str, **extra):
        self.emit(role, text, **extra)
        self.band.send(f"**{role}**: {text}")

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def perceive(self) -> dict:
        status = self._get("/status")
        map2d = self._get("/map2d")
        return {"status": status, "map": map2d}

    def recall(self, task_text: str, k: int = 3) -> list[dict]:
        if self.memory is None:
            return []
        return self.memory.recall(task_text, k=k)

    def remember(self, task_text: str, params: dict, outcome: dict):
        if self.memory is None:
            return
        self.memory.remember(
            {
                "id": f"{self._run_id}-{uuid.uuid4().hex[:6]}",
                "task_text": task_text,
                "params": params,
                "outcome": outcome,
                "t": time.time(),
            }
        )

    def predict_params(self, task: str, perceive: dict, memories: list[dict]) -> dict:
        if self.learner is None:
            return {}
        context = self.learner.build_context(task, perceive, memories)
        return self.learner.predict_params(task, context)

    def set_params(self, params: dict):
        mapping = {
            "aim_du": "/setaimdu?px={}",
            "right_trim_cm": "/settrim?cm={}",
            "back_cm": "/setback?cm={}",
            "approach_steps": "/setsteps?n={}",
            "cube_edge_cm": "/setcubesize?cm={}",
            "range_scale": "/setrangescale?scale={}",
            "bearing_deg": "/setbearing?deg={}",
        }
        for key, fmt in mapping.items():
            if key in params:
                path = fmt.format(params[key])
                self._post(path)
                time.sleep(0.05)

    def execute_pick(self, color: str):
        query = f"{color} cube"
        self._post(f"/setquery?q={query}")
        time.sleep(0.1)
        return self._post("/start")

    def execute_place(self, destination_tag: int | None = None):
        """MVP place: hover over a mapped object tag and open gripper."""
        if destination_tag is not None:
            self._post(f"/goto2d?tag={destination_tag}")
            # wait for goto to finish
            self._wait_idle()
        self._post("/jogpress?dir=open")
        time.sleep(0.5)
        self._post("/jogrelease?dir=open")
        return {"ok": True}

    def _wait_idle(self, timeout: int = MAX_POLL_SECONDS):
        start = time.time()
        while time.time() - start < timeout:
            s = self._get("/status")
            if not s.get("running"):
                return s
            time.sleep(0.5)
        return self._get("/status")

    def observe_outcome(self) -> dict:
        s = self._wait_idle()
        phase = s.get("phase", "")
        detail = s.get("detail", "")
        detail_lower = detail.lower()
        success = (
            phase == "DONE"
            and "missed" not in detail_lower
            and "picked" in detail_lower
        )
        gripper = s.get("gripper")
        # Heuristic: if gripper closed (<15%) and we say picked, treat as success.
        if success and gripper is not None and float(gripper) > 20.0:
            success = False
        return {
            "success": success,
            "phase": phase,
            "detail": detail,
            "gripper": gripper,
            "status": s,
        }

    # ------------------------------------------------------------------
    # Task parsing
    # ------------------------------------------------------------------
    @staticmethod
    def parse_task(text: str) -> dict:
        text_lower = text.lower()
        color = None
        for c in ("red", "green", "blue", "yellow", "orange", "purple", "white", "black"):
            if c in text_lower:
                color = c
                break
        action = "pick"
        if "place" in text_lower or "drop" in text_lower:
            action = "place"
        dest_color = None
        if action == "place":
            # crude: color after "on" is destination
            m = re.search(r"on\s+(?:the\s+)?(\w+)\s+cube", text_lower)
            if m:
                dest_color = m.group(1)
        return {"action": action, "color": color or "red", "destination_color": dest_color}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, task: str, max_attempts: int = 3):
        self._say("Planner", f"Run {self._run_id}: {task}")
        parsed = self.parse_task(task)
        self._say("Planner", f"Parsed: {parsed}")

        if parsed["action"] == "place":
            # Find destination tag from 2D map
            perceive = self.perceive()
            tag = None
            dest_color = parsed.get("destination_color")
            for o in perceive.get("map", {}).get("objs", []):
                if dest_color and dest_color in o.get("label", "").lower():
                    tag = o.get("tag")
                    break
            self._say("Executor", f"Placing on tag={tag}")
            self.execute_place(tag)
            self._say("Planner", "Place sequence complete.")
            return

        # Pick loop with self-improvement
        for attempt in range(1, max_attempts + 1):
            self._say("Planner", f"Attempt {attempt}/{max_attempts}")

            perceive = self.perceive()
            memories = self.recall(task)
            self._say("Memory", f"Recalled {len(memories)} similar episode(s)")

            params = self.predict_params(task, perceive, memories)
            self._say("Learner", f"Predicted params: {json.dumps(params)}")

            self.set_params(params)
            self._say("Executor", f"Executing pick of {parsed['color']} cube")
            self.execute_pick(parsed["color"])

            outcome = self.observe_outcome()
            self._say(
                "Critic",
                f"Outcome: success={outcome['success']} | phase={outcome['phase']} | {outcome['detail']}",
                outcome=outcome,
            )

            # Learn
            context = self.learner.build_context(task, perceive, memories) if self.learner else ""
            if self.learner:
                self.learner.record_example(task, context, params, outcome)
            self.remember(task, params, outcome)

            if outcome["success"]:
                self._say("Planner", "Task succeeded.")
                return

            if attempt == max_attempts:
                self._say("Planner", "Max attempts reached. Triggering fine-tune...")
                if self.learner:
                    self.learner.start_fine_tune()
                return

            time.sleep(0.5)


if __name__ == "__main__":
    # Stand-alone test without memory/learner
    agent = SelfLearningAgent()
    agent.run("pick the red cube", max_attempts=2)
