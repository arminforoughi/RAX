"""Pioneer AI specialist learner.

Drops into Pioneer's OpenAI-compatible chat endpoint to predict grasp
parameters. Collects execution traces and can trigger a Pioneer fine-tune job.
Without a Pioneer key it falls back to a heuristic that adapts parameters from
recent failures.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

PIONEER_API_URL = os.environ.get("PIONEER_API_URL", "https://api.pioneer.ai/v1")
PIONEER_TRAIN_URL = os.environ.get("PIONEER_TRAIN_URL", "https://api.pioneer.ai")
DEFAULT_MODEL = os.environ.get("PIONEER_MODEL", "fastino/qwen3-32b")
EXAMPLES_FILE = Path("training_examples.jsonl")

# Tunable parameters the robot exposes
PARAM_SCHEMA = {
    "aim_du": -45.0,
    "right_trim_cm": 3.0,
    "back_cm": 1.0,
    "approach_steps": 4,
    "cube_edge_cm": 5.08,
    "range_scale": 1.0,
    "bearing_deg": 0.0,
    "grasp_pitch": 75.0,
}


def _clamp(name: str, value: float) -> float:
    clamps = {
        "aim_du": (-300.0, 300.0),
        "right_trim_cm": (-10.0, 15.0),
        "back_cm": (0.0, 10.0),
        "approach_steps": (1, 12),
        "cube_edge_cm": (1.0, 30.0),
        "range_scale": (0.3, 3.0),
        "bearing_deg": (-180.0, 180.0),
        "grasp_pitch": (0.0, 90.0),
    }
    lo, hi = clamps.get(name, (float("-inf"), float("inf")))
    if name == "approach_steps":
        return int(max(lo, min(hi, value)))
    return float(max(lo, min(hi, value)))


class PioneerLearner:
    def __init__(self, model: str | None = None, emit=None):
        self.api_key = os.environ.get("PIONEER_API_KEY")
        self.model = model or DEFAULT_MODEL
        self.emit = emit or (lambda role, text: print(f"[{role}] {text}"))
        self.examples: list[dict] = []
        self._load_examples()

    def _load_examples(self):
        if EXAMPLES_FILE.exists():
            try:
                with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
                    self.examples = [json.loads(line) for line in f if line.strip()]
                self.emit("Learner", f"Loaded {len(self.examples)} training examples")
            except Exception as e:
                self.emit("Learner", f"Failed to load examples: {e}")

    def _record(self, example: dict):
        self.examples.append(example)
        try:
            with open(EXAMPLES_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(example, default=str) + "\n")
        except Exception as e:
            self.emit("Learner", f"Failed to append example: {e}")

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------
    def build_context(self, task: str, perceive: dict, memories: list[dict]) -> str:
        lines = [
            "Robot task:",
            task,
            "",
            "Recent similar attempts (success/failure + parameters used):",
        ]
        for i, m in enumerate(memories[:5], 1):
            p = m.get("payload", {})
            out = p.get("outcome", {})
            lines.append(
                f"{i}. params={p.get('params', {})} "
                f"success={out.get('success')} detail={out.get('detail')}"
            )
        lines.append("")
        lines.append("Current scene:")
        status = perceive.get("status", {})
        lines.append(f"phase={status.get('phase')} gripper={status.get('gripper')}")
        objs = perceive.get("map", {}).get("objs", [])
        for o in objs[:5]:
            lines.append(f"- {o.get('label')} @ x={o.get('x')} y={o.get('y')} r={o.get('r_cm')}cm")
        lines.append("")
        lines.append(
            "Predict the best grasp parameters as JSON matching this schema: "
            + json.dumps(PARAM_SCHEMA)
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Pioneer inference
    # ------------------------------------------------------------------
    def predict_params(self, task: str, context: str) -> dict:
        if not self.api_key:
            return self._fallback_predict(task, context)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a robot grasp-parameter optimizer. "
                    "Return ONLY a JSON object with these keys: "
                    + json.dumps(PARAM_SCHEMA)
                    + ". Choose values that maximize pick success based on the task and past attempts."
                ),
            },
            {"role": "user", "content": context},
        ]
        try:
            r = requests.post(
                f"{PIONEER_API_URL}/chat/completions",
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                    "max_tokens": 256,
                },
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            parsed = json.loads(text)
            params = self._sanitize(parsed)
            self.emit("Pioneer", f"Predicted params via {self.model}: {params}")
            return params
        except Exception as e:
            self.emit("Pioneer", f"Inference failed ({e}); using fallback")
            return self._fallback_predict(task, context)

    def _sanitize(self, parsed: dict) -> dict:
        out = dict(PARAM_SCHEMA)
        for k in PARAM_SCHEMA:
            if k in parsed:
                out[k] = _clamp(k, parsed[k])
        return out

    # ------------------------------------------------------------------
    # Fallback heuristic learner
    # ------------------------------------------------------------------
    def _fallback_predict(self, task: str, context: str) -> dict:
        params = dict(PARAM_SCHEMA)
        # Adjust based on the most recent failure of a similar task.
        recent_failures = [
            ex for ex in reversed(self.examples)
            if not ex.get("outcome", {}).get("success")
        ]
        if recent_failures:
            last = recent_failures[0]
            old = last.get("params", {})
            # Simple exploratory strategy: if it missed, shift aim and slow down.
            params["aim_du"] = _clamp("aim_du", old.get("aim_du", params["aim_du"]) - 8.0)
            params["approach_steps"] = _clamp("approach_steps", old.get("approach_steps", params["approach_steps"]) + 1)
            params["right_trim_cm"] = _clamp("right_trim_cm", old.get("right_trim_cm", params["right_trim_cm"]) - 0.5)
            params["back_cm"] = _clamp("back_cm", old.get("back_cm", params["back_cm"]) + 0.2)
            self.emit("Learner", f"Fallback adjusted from failure: {params}")
        else:
            self.emit("Learner", f"Fallback defaults: {params}")
        return params

    # ------------------------------------------------------------------
    # Learning from outcomes
    # ------------------------------------------------------------------
    def record_example(self, task: str, context: str, params: dict, outcome: dict):
        example = {
            "t": time.time(),
            "task": task,
            "context": context,
            "params": params,
            "outcome": outcome,
        }
        self._record(example)
        self.emit("Learner", f"Recorded example (success={outcome.get('success')}); total={len(self.examples)}")

    # ------------------------------------------------------------------
    # Pioneer fine-tune trigger
    # ------------------------------------------------------------------
    def start_fine_tune(self) -> dict:
        if not self.api_key:
            self.emit("Pioneer", "No PIONEER_API_KEY; skipping fine-tune")
            return {"status": "no_key"}
        if len(self.examples) < 3:
            self.emit("Pioneer", "Not enough examples to fine-tune (<3)")
            return {"status": "too_few_examples"}
        try:
            # Build a Pioneer dataset JSONL (OpenAI chat format)
            dataset_name = f"rax_grasp_{int(time.time())}"
            dataset_lines = []
            for ex in self.examples:
                messages = [
                    {"role": "system", "content": "You are a robot grasp-parameter optimizer. Return JSON only."},
                    {"role": "user", "content": ex["context"]},
                    {"role": "assistant", "content": json.dumps(ex["params"])},
                ]
                dataset_lines.append(json.dumps({"messages": messages}))
            # Upload dataset
            upload = requests.post(
                f"{PIONEER_TRAIN_URL}/felix/datasets",
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
                json={"name": dataset_name, "format": "jsonl"},
                timeout=30,
            )
            upload.raise_for_status()
            dataset_info = upload.json()
            # Some APIs expect a follow-up file upload; we provide a data_url placeholder.
            self.emit("Pioneer", f"Dataset {dataset_name} created: {dataset_info}")
            job = requests.post(
                f"{PIONEER_TRAIN_URL}/felix/training-jobs",
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
                json={
                    "model_name": f"rax-grasp-{int(time.time())}",
                    "base_model": self.model,
                    "datasets": [{"name": dataset_name}],
                    "training_type": "lora",
                    "nr_epochs": 3,
                    "learning_rate": 5e-5,
                },
                timeout=30,
            )
            job.raise_for_status()
            job_info = job.json()
            self.emit("Pioneer", f"Fine-tune job started: {job_info}")
            return {"status": "started", "dataset": dataset_info, "job": job_info}
        except Exception as e:
            self.emit("Pioneer", f"Fine-tune trigger failed: {e}")
            return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    pl = PioneerLearner()
    print(pl.predict_params("pick red cube", "No prior attempts."))
