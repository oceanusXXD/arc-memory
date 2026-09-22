"""Frozen retrieval vectors shared by retrieval and the selector."""
from __future__ import annotations

from typing import Any

import numpy as np


class FrozenEmbedder:
    def __init__(self, config: dict[str, Any]):
        from sentence_transformers import SentenceTransformer

        spec = config.get("retrieval") or {}
        name = str(spec.get("embedding_model") or "Qwen/Qwen3-Embedding-0.6B")
        revision = spec.get("embedding_revision")
        self.device = str(spec.get("embedding_device") or "cpu")
        self.model = SentenceTransformer(
            name,
            device=self.device,
            **({"revision": revision} if revision else {}),
        )
        self.model.eval()
        self.query_prompt = str(spec.get("query_prompt_name") or "query")
        first = self.model[0]
        model_config = getattr(getattr(first, "auto_model", None), "config", None)
        self.metadata = {
            "model": name,
            "revision": str(getattr(model_config, "_commit_hash", None) or revision or "main"),
            "dimension": int(self.model.get_sentence_embedding_dimension()),
            "query_prompt_name": self.query_prompt,
            "normalized": True,
            "device": self.device,
        }

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.metadata["dimension"]), dtype="float32")
        kwargs = {"prompt_name": self.query_prompt} if query else {}
        vectors = np.asarray(self.model.encode(texts, show_progress_bar=False, **kwargs), dtype="float32")
        if vectors.shape != (len(texts), self.metadata["dimension"]) or not np.isfinite(vectors).all():
            raise ValueError("invalid embedding shape or non-finite values")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("zero embedding vector")
        return vectors / norms

    def attach(self, question: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not sources:
            return []
        vectors = self.encode([str(source.get("text") or "") for source in sources])
        query = self.encode([question], query=True)[0].tolist()
        rows = []
        for source, vector in zip(sources, vectors):
            row = dict(source)
            row.pop("query_vector", None)
            row.pop("query_vector_question", None)
            row.update(embedding=vector.tolist(), embedding_metadata=dict(self.metadata))
            rows.append(row)
        rows[0].update(query_vector=query, query_vector_question=question)
        return rows
