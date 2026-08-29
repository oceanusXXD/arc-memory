"""使用 FutureMem 同款阿里 embeddings 接口，或固定的本地检索编码器。"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

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
        self._text_cache: dict[str, np.ndarray] = {}
        configured_cache = os.environ.get("R2W_EMBEDDING_CACHE", "").strip()
        self.cache_path = Path(configured_cache) if configured_cache else None
        self.cache_hits = 0
        self.cache_misses = 0
        if self.cache_path is not None and self.cache_path.exists():
            self._load_text_cache(self.cache_path)

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_text_cache(self, path: Path) -> None:
        """加载 R2W 自己的文本哈希缓存，拒绝 id-keyed 的外部缓存。"""
        try:
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"schema", "keys", "vectors"}:
                    raise ValueError("字段必须是 schema/keys/vectors")
                schema = str(np.asarray(data["schema"]).item())
                keys = [str(value) for value in data["keys"]]
                vectors = np.asarray(data["vectors"], dtype=np.float32)
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"R2W embedding 缓存无法读取: {path}") from exc
        if schema != "r2w-text-sha256-v1":
            raise ValueError(
                "R2W embedding 缓存版本不匹配；FutureMem 的 episode::memory_id 缓存"
                "不能在未验证表示文本相同的情况下直接复用。"
            )
        if (
            vectors.ndim != 2
            or vectors.shape != (len(keys), self.cfg.embedding_dim)
            or len(keys) != len(set(keys))
            or any(len(key) != 64 for key in keys)
            or not np.isfinite(vectors).all()
            or np.any(np.linalg.norm(vectors, axis=1) == 0)
        ):
            raise ValueError("R2W embedding 缓存维度、键或向量无效。")
        self._text_cache = {
            key: vector / np.linalg.norm(vector)
            for key, vector in zip(keys, vectors, strict=True)
        }

    def _save_text_cache(self) -> None:
        if self.cache_path is None or not self._text_cache:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._text_cache)
        vectors = np.asarray([self._text_cache[key] for key in keys], dtype=np.float32)
        temporary = self.cache_path.with_name(
            f"{self.cache_path.stem}.tmp{self.cache_path.suffix}"
        )
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray("r2w-text-sha256-v1"),
                keys=np.asarray(keys),
                vectors=vectors,
            )
        temporary.replace(self.cache_path)

    def cache_statistics(self) -> dict:
        return {
            "schema": "r2w-text-sha256-v1",
            "path": str(self.cache_path) if self.cache_path is not None else None,
            "entries": len(self._text_cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
        }

    def encode(self, texts, normalize_embeddings: bool = True, batch_size: int = 128):
        texts = list(texts)
        if not texts:
            raise ValueError("embedding 输入不能为空。")
        if not normalize_embeddings:
            raise ValueError("R2W FINAL 固定使用 L2 归一化 embedding。")
        digests = [self._digest(text) for text in texts]
        missing = {
            digest: text
            for text, digest in zip(texts, digests, strict=True)
            if digest not in self._text_cache
        }
        self.cache_hits += len(texts) - sum(
            digest not in self._text_cache for digest in digests
        )
        self.cache_misses += sum(digest not in self._text_cache for digest in digests)
        if not missing:
            return np.asarray(
                [self._text_cache[digest] for digest in digests], dtype=np.float32
            )
        missing_digests = list(missing)
        missing_texts = list(missing.values())
        if self.base_url:
            if not self.api_key:
                raise RuntimeError(
                    "远端 embedding 需要 EMBEDDING_API_KEY 或 LLM_API_KEY。"
                )
            size = min(batch_size, self.cfg.embedding_api_batch_size)
            vectors = np.concatenate(
                [
                    self._api_encode(missing_texts[start : start + size])
                    for start in range(0, len(missing_texts), size)
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
                    missing_texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
        if (
            vectors.shape != (len(missing_texts), self.cfg.embedding_dim)
            or not np.isfinite(vectors).all()
        ):
            raise RuntimeError(
                f"embedding 返回形状 {vectors.shape}，要求 ({len(texts)}, {self.cfg.embedding_dim})。"
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("embedding API 返回零向量。")
        vectors = (vectors / norms).astype(np.float32)
        for digest, vector in zip(missing_digests, vectors, strict=True):
            self._text_cache[digest] = vector
        self._save_text_cache()
        return np.asarray(
            [self._text_cache[digest] for digest in digests], dtype=np.float32
        )

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
