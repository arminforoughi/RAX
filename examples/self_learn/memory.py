"""Vector episodic memory for the self-learning robot.

Primary backend: Actian VectorAI DB (works edge/offline).
Fallback backend: local JSONL file if Actian is not running.

Each memory point stores:
- a 384-d embedding of the task text (+ optional object image features)
- payload: task, params, outcome, timestamp
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

VEC_SIZE = 384
COLLECTION = os.environ.get("ACTIAN_COLLECTION", "rax_memory")


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)) or 1.0
    return v / n


def _hash_embedding(text: str, size: int = VEC_SIZE) -> np.ndarray:
    """Deterministic bag-of-words embedding used when sentence-transformers
    is not installed. Good enough for local fallback demos."""
    vec = np.zeros(size, dtype=np.float32)
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % size] += 1.0
    return _normalize(vec)


class VectorMemory:
    """Actian-first vector memory with a transparent JSONL fallback."""

    def __init__(self, host: str | None = None, emit=None):
        self.host = host or os.environ.get("ACTIAN_HOST", "localhost:6574")
        self.emit = emit or (lambda role, text: print(f"[{role}] {text}"))
        self._client = None
        self._local_path = Path("memory_cache.jsonl")
        self._local: list[dict] = []
        self._encoder = None
        self._init()

    def _init(self):
        try:
            from actian_vectorai import Distance, VectorAIClient, VectorParams
            self._client = VectorAIClient(self.host)
            info = self._client.health_check()
            self.emit("Actian", f"Connected to {info.get('title')} v{info.get('version')}")
            try:
                self._client.collections.create(
                    COLLECTION,
                    vectors_config=VectorParams(size=VEC_SIZE, distance=Distance.Cosine),
                )
                self.emit("Actian", f"Created collection {COLLECTION}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    self.emit("Actian", f"Collection create note: {e}")
        except Exception as e:
            self.emit("Actian", f"Using local JSONL fallback ({e})")
            self._load_local()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def embed(self, text: str, image_features: list[float] | None = None) -> np.ndarray:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.emit("Memory", "Loaded sentence-transformers embedding model")
            except Exception as e:
                self.emit("Memory", f"sentence-transformers unavailable, using hash fallback ({e})")
                self._encoder = "hash"
        if self._encoder == "hash":
            vec = _hash_embedding(text, VEC_SIZE)
        else:
            vec = self._encoder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            vec = np.asarray(vec, dtype=np.float32)
            if len(vec) != VEC_SIZE:
                vec = np.resize(vec, VEC_SIZE)
                vec = _normalize(vec)
        if image_features:
            img = np.asarray(image_features, dtype=np.float32)
            if len(img) != VEC_SIZE:
                img = np.resize(img, VEC_SIZE)
            vec = _normalize(vec + 0.3 * img)
        return vec

    # ------------------------------------------------------------------
    # Local fallback helpers
    # ------------------------------------------------------------------
    def _load_local(self):
        if not self._local_path.exists():
            return
        try:
            with open(self._local_path, "r", encoding="utf-8") as f:
                self._local = [json.loads(line) for line in f if line.strip()]
            self.emit("Memory", f"Loaded {len(self._local)} local memories")
        except Exception as e:
            self.emit("Memory", f"Failed to load local cache: {e}")

    def _save_local(self):
        try:
            with open(self._local_path, "w", encoding="utf-8") as f:
                for item in self._local:
                    f.write(json.dumps(item, default=str) + "\n")
        except Exception as e:
            self.emit("Memory", f"Failed to save local cache: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def remember(self, episode: dict) -> str:
        """Store an episode. Returns the memory id."""
        episode_id = episode.get("id") or str(uuid.uuid4())
        text = episode.get("task_text", "") or episode.get("task", "")
        image_features = episode.get("image_features")
        vector = self.embed(text, image_features).tolist()
        payload = {
            "task": text,
            "params": episode.get("params", {}),
            "outcome": episode.get("outcome", {}),
            "t": episode.get("t", time.time()),
        }
        if self._client:
            try:
                from actian_vectorai import PointStruct
                self._client.points.upsert(
                    COLLECTION,
                    [PointStruct(id=episode_id, vector=vector, payload=payload)],
                )
            except Exception as e:
                self.emit("Actian", f"Upsert failed, falling back to local: {e}")
                self._client = None
                self._load_local()
        if not self._client:
            self._local.append({"id": episode_id, "vector": vector, "payload": payload})
            self._save_local()
        self.emit("Memory", f"Remembered episode {episode_id[:8]} ({'success' if payload['outcome'].get('success') else 'failure'})")
        return episode_id

    def recall(self, task_text: str, k: int = 3) -> list[dict]:
        """Return top-k similar episodes with cosine similarity."""
        vector = self.embed(task_text)
        if self._client:
            try:
                hits = self._client.points.search(
                    COLLECTION, vector=vector.tolist(), limit=max(k, 1)
                )
                return [
                    {"score": h.score, "payload": h.payload, "id": h.id}
                    for h in hits
                ]
            except Exception as e:
                self.emit("Actian", f"Search failed, using local: {e}")
                self._client = None
                self._load_local()
        # local cosine search
        if not self._local:
            return []
        v = np.asarray(vector, dtype=np.float32)
        scored = []
        for item in self._local:
            iv = np.asarray(item["vector"], dtype=np.float32)
            sim = float(np.dot(v, iv))
            scored.append((sim, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"score": round(sim, 4), "payload": item["payload"], "id": item["id"]}
            for sim, item in scored[:k]
        ]

    def all_memories(self) -> list[dict]:
        if self._client:
            try:
                # Actian client scroll API may vary; this is best-effort
                return []
            except Exception:
                pass
        return self._local


if __name__ == "__main__":
    mem = VectorMemory()
    print(mem.remember({"task_text": "pick red cube", "params": {"aim_du": -45}, "outcome": {"success": False}}))
    print(mem.recall("pick red cube"))
