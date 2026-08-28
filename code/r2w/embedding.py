"""使用 FutureMem 同款阿里 embeddings 接口，或固定的本地检索编码器。"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

from .config import R2WConfig
from .runtime_usage import UsageLedger


class EmbeddingProtocolError(RuntimeError):
    """远端 embedding 响应违反索引或向量协议。"""


class EmbeddingBackend:
    def __init__(self, cfg: R2WConfig):
        if cfg.embedding_api_batch_size < 1 or cfg.embedding_api_timeout <= 0:
            raise ValueError("embedding API batch size 与 timeout 必须为正数。")
        self.cfg = cfg
        self.model_name = cfg.embedding_model
        self.base_url = cfg.embedding_base_url.rstrip("/")
        self.api_key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get(
            "LLM_API_KEY", ""
        )
        self.model = None
        self.usage = UsageLedger()

    def encode(self, texts, normalize_embeddings: bool = True, batch_size: int = 128):
        texts = list(texts)
        if not texts:
            raise ValueError("embedding 输入不能为空。")
        if not normalize_embeddings:
            raise ValueError("R2W FINAL 固定使用 L2 归一化 embedding。")
        if self.base_url:
            if not self.api_key:
                raise RuntimeError(
                    "远端 embedding 需要 EMBEDDING_API_KEY 或 LLM_API_KEY。"
                )
            size = min(batch_size, self.cfg.embedding_api_batch_size)
            vectors = np.concatenate(
                [
                    self._api_encode(texts[start : start + size])
                    for start in range(0, len(texts), size)
                ],
                axis=0,
            )
        else:
            if self.model is None:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(
                    self.model_name, device=self.cfg.embedding_device
                )
            vectors = np.asarray(
                self.model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
        if (
            vectors.shape != (len(texts), self.cfg.embedding_dim)
            or not np.isfinite(vectors).all()
        ):
            raise RuntimeError(
                f"embedding 返回形状 {vectors.shape}，要求 ({len(texts)}, {self.cfg.embedding_dim})。"
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("embedding API 返回零向量。")
        return (vectors / norms).astype(np.float32)

    def _api_encode(self, texts: list[str]) -> np.ndarray:
        endpoint = "/embeddings" if self.base_url.endswith("/v1") else "/v1/embeddings"
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(
                {"model": self.model_name, "input": texts, "encoding_format": "float"}
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.cfg.embedding_api_timeout
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8", errors="replace"))
                error = payload.get("error", payload)
                message = (
                    error.get("message", error) if isinstance(error, dict) else error
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                message = exc.reason
            raise RuntimeError(
                f"embedding API 请求失败: HTTP {exc.code}: {message}"
            ) from exc
        self.usage.add("embedding", result.get("usage"))
        rows = result.get("data") or []
        indexed = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("index"), int):
                raise EmbeddingProtocolError("embedding API 返回了非法 data 项。")
            if row["index"] in indexed:
                raise EmbeddingProtocolError("embedding API 返回了重复 index。")
            indexed[row["index"]] = row.get("embedding")
        expected = set(range(len(texts)))
        if set(indexed) != expected:
            raise EmbeddingProtocolError(
                "embedding API 返回的 index 集合与输入不一致。"
            )
        return np.asarray(
            [indexed[index] for index in range(len(texts))], dtype=np.float32
        )
