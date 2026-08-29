from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable, Protocol
import numpy as np
from .config import R2WConfig
from .actions import Representation
from .text import BM25

class Embedder(Protocol):
    def encode(self, texts, normalize_embeddings=True, batch_size=128): ...

@dataclass(frozen=True)
class IndexUnit:
    parent_id: Hashable
    text: str
    body: str

@dataclass(frozen=True)
class RetrievedParent:
    parent_id: Hashable
    sparse_rank: int | None
    dense_rank: int | None
    rrf: float
    body: str

class ParentRRFIndex:
    """Per-channel key->parent max, then parent ranks, then parent-level RRF."""
    def __init__(self, representations: list[Representation], embedder: Embedder, cfg: R2WConfig):
        self.cfg, self.embedder = cfg, embedder
        self.representations = {r.parent_id: r for r in representations}
        self.units = [IndexUnit(r.parent_id, key, r.body) for r in representations for key in r.keys]
        self._refresh()

    def _refresh(self):
        texts = [u.text for u in self.units]
        self.bm25 = BM25(texts) if texts else None
        self.emb = np.asarray(self.embedder.encode(texts, normalize_embeddings=True, batch_size=128), dtype=np.float32) if texts else np.empty((0,0), dtype=np.float32)

    def remove_parent(self, parent_id):
        old = self.representations.pop(parent_id, None)
        if old is not None:
            self.units = [u for u in self.units if u.parent_id != parent_id]; self._refresh()
        return old

    def add(self, representation: Representation):
        if representation.parent_id in self.representations:
            raise ValueError("parent 已存在。")
        self.representations[representation.parent_id] = representation
        self.units += [IndexUnit(representation.parent_id, k, representation.body) for k in representation.keys]
        self._refresh()

    def _parent_channel_ranks(self, scores: np.ndarray) -> dict[Hashable,int]:
        parent_max: dict[Hashable,float] = {}
        for i, s in enumerate(scores.tolist()):
            p = self.units[i].parent_id
            parent_max[p] = max(parent_max.get(p, float("-inf")), float(s))
        ordered = sorted(parent_max, key=lambda p: (-parent_max[p], str(p)))
        return {p:i+1 for i,p in enumerate(ordered)}

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedParent]:
        k = k or self.cfg.retrieval_k
        if not query.strip() or k < 1: raise ValueError("query/k 非法。")
        if not self.units: return []
        sparse = np.asarray(self.bm25.scores(query), dtype=np.float32)
        q = np.asarray(self.embedder.encode([query], normalize_embeddings=True)[0], dtype=np.float32)
        dense = self.emb @ q
        sr = self._parent_channel_ranks(sparse); dr = self._parent_channel_ranks(dense)
        parents = set(sr)|set(dr)
        fused = {p:(0 if p not in sr else 1/(self.cfg.rrf_k+sr[p])) + (0 if p not in dr else 1/(self.cfg.rrf_k+dr[p])) for p in parents}
        ordered = sorted(parents, key=lambda p: (-fused[p], str(p)))[:k]
        return [RetrievedParent(p, sr.get(p), dr.get(p), float(fused[p]), self.representations[p].body) for p in ordered]
