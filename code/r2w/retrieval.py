"""部署、测量和探针共用的可变 BM25+dense RRF 索引。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np

from .config import R2WConfig
from .representations import Repr
from .text import BM25


@dataclass(frozen=True)
class IndexUnit:
    turn_index: Hashable
    text: str
    session: int = 0
    unit_id: str = ""
    kind: str = "payload"


@dataclass(frozen=True)
class ScoredUnit:
    unit_index: int
    bm25: float
    dense: float
    rrf: float


@dataclass(frozen=True)
class RetrievedTurn:
    """按一个 turn 的最佳检索单元排序，reader 只消费 payload。"""

    turn_index: Hashable
    matched_unit_index: int
    matched_unit_rank: int
    unit_indices: tuple[int, ...]
    payload_unit_index: int | None = None


def representation_index_units(
    turn_index: Hashable, representation: Repr, session: int = 0
) -> list[IndexUnit]:
    prefix = str(turn_index)
    return [
        IndexUnit(turn_index, representation.payload, session, f"{prefix}:payload", "payload"),
        *[
            IndexUnit(turn_index, key, session, f"{prefix}:key:{index}", "key")
            for index, key in enumerate(representation.keys)
        ],
    ]


class UnifiedIndex:
    """统一索引。

    原有构造、`scored/retrieve/retrieve_turns/probe` 接口保持不变；新增
    `add_units/remove_turn/swap_turn/restore_turn` 供 L2/L3 与在线 writer 使用。
    稠密向量按文本缓存，单条交换时只重算 BM25 统计和未见过的文本向量。
    """

    def __init__(self, units: list[IndexUnit], embedder, cfg: R2WConfig):
        if any(not unit.text.strip() for unit in units):
            raise ValueError("统一索引单元文本不可为空。")
        self.units = list(units)
        self._embedder = embedder
        self._cfg = cfg
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._bm25: BM25 | None = None
        self._embeddings = np.empty((0, cfg.embedding_dim), dtype=np.float32)
        self._refresh()

    def _refresh(self) -> None:
        texts = [unit.text for unit in self.units]
        self._bm25 = (
            BM25(texts, k1=self._cfg.bm25_k1, b=self._cfg.bm25_b) if texts else None
        )
        missing = list(dict.fromkeys(text for text in texts if text not in self._embedding_cache))
        if missing:
            encoded = np.asarray(
                self._embedder.encode(
                    missing, normalize_embeddings=True, batch_size=128
                ),
                dtype=np.float32,
            )
            if encoded.shape != (len(missing), self._cfg.embedding_dim):
                raise RuntimeError("索引 embedding 维度与配置不一致。")
            self._embedding_cache.update(zip(missing, encoded, strict=True))
        self._embeddings = (
            np.vstack([self._embedding_cache[text] for text in texts]).astype(np.float32)
            if texts
            else np.empty((0, self._cfg.embedding_dim), dtype=np.float32)
        )

    def add_units(self, units: Iterable[IndexUnit]) -> None:
        values = list(units)
        if any(not value.text.strip() for value in values):
            raise ValueError("索引单元文本不可为空。")
        existing_ids = {unit.unit_id for unit in self.units if unit.unit_id}
        incoming_ids = [unit.unit_id for unit in values if unit.unit_id]
        if len(incoming_ids) != len(set(incoming_ids)) or existing_ids & set(incoming_ids):
            raise ValueError("索引 unit_id 必须唯一。")
        self.units.extend(values)
        self._refresh()

    def add_representation(
        self, turn_index: Hashable, representation: Repr, session: int = 0
    ) -> None:
        if any(unit.turn_index == turn_index for unit in self.units):
            raise ValueError(f"turn {turn_index!r} 已在索引中。")
        self.add_units(representation_index_units(turn_index, representation, session))

    def remove_turn(self, turn_index: Hashable) -> list[IndexUnit]:
        removed = [unit for unit in self.units if unit.turn_index == turn_index]
        if removed:
            self.units = [unit for unit in self.units if unit.turn_index != turn_index]
            self._refresh()
        return removed

    def swap_turn(
        self, turn_index: Hashable, new_units: Iterable[IndexUnit]
    ) -> list[IndexUnit]:
        values = list(new_units)
        if any(unit.turn_index != turn_index for unit in values):
            raise ValueError("swap_turn 的所有新单元必须属于目标 turn。")
        positions = [
            index for index, unit in enumerate(self.units) if unit.turn_index == turn_index
        ]
        old = [self.units[index] for index in positions]
        insert_at = min(positions, default=len(self.units))
        retained = [unit for unit in self.units if unit.turn_index != turn_index]
        self.units = retained[:insert_at] + values + retained[insert_at:]
        self._refresh()
        return old

    def restore_turn(self, turn_index: Hashable, old_units: Iterable[IndexUnit]) -> None:
        values = list(old_units)
        self.swap_turn(turn_index, values)

    def raw_scores(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """返回未融合 sparse/dense 分数，供真实检索和探针共用。"""
        if not query.strip():
            raise ValueError("检索 query 不能为空。")
        if not self.units:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        assert self._bm25 is not None
        lexical = np.asarray(self._bm25.scores(query), dtype=np.float32)
        query_embedding = np.asarray(
            self._embedder.encode([query], normalize_embeddings=True)[0],
            dtype=np.float32,
        )
        return lexical, self._embeddings @ query_embedding

    def fuse(self, lexical: np.ndarray, dense: np.ndarray) -> np.ndarray:
        """以历史 RRF 规则融合；拆分接口保证探针与检索完全同式。"""
        if lexical.shape != dense.shape:
            raise ValueError("sparse/dense 分数形状必须一致。")
        if not len(lexical):
            return np.empty(0, dtype=np.float32)
        lexical_order = np.argsort(-lexical, kind="stable")
        dense_order = np.argsort(-dense, kind="stable")
        lexical_rank = np.empty(len(lexical), dtype=np.int32)
        dense_rank = np.empty(len(dense), dtype=np.int32)
        lexical_rank[lexical_order] = np.arange(1, len(lexical) + 1)
        dense_rank[dense_order] = np.arange(1, len(dense) + 1)
        return (
            1.0 / (self._cfg.rrf_k + lexical_rank)
            + 1.0 / (self._cfg.rrf_k + dense_rank)
        ).astype(np.float32)

    def scored(self, query: str) -> list[ScoredUnit]:
        lexical, dense = self.raw_scores(query)
        rrf = self.fuse(lexical, dense)
        order = np.argsort(-rrf, kind="stable")
        return [
            ScoredUnit(int(index), float(lexical[index]), float(dense[index]), float(rrf[index]))
            for index in order
        ]

    def retrieve(self, query: str, k: int) -> list[int]:
        if k < 1:
            raise ValueError("k 必须大于 0。")
        return [entry.unit_index for entry in self.scored(query)[: min(k, len(self.units))]]

    def retrieve_turns(self, query: str, k: int) -> list[RetrievedTurn]:
        if k < 1:
            raise ValueError("k 必须大于 0。")
        grouped: dict[Hashable, list[int]] = {}
        for unit_index, unit in enumerate(self.units):
            grouped.setdefault(unit.turn_index, []).append(unit_index)
        result: list[RetrievedTurn] = []
        seen: set[Hashable] = set()
        for unit_rank, entry in enumerate(self.scored(query), start=1):
            turn_index = self.units[entry.unit_index].turn_index
            if turn_index in seen:
                continue
            seen.add(turn_index)
            indices = tuple(grouped[turn_index])
            payload_index = next(
                (index for index in indices if self.units[index].kind == "payload"), None
            )
            result.append(
                RetrievedTurn(
                    turn_index=turn_index,
                    matched_unit_index=entry.unit_index,
                    matched_unit_rank=unit_rank,
                    unit_indices=indices,
                    payload_unit_index=payload_index,
                )
            )
            if len(result) == k:
                break
        return result

    def reader_payloads(self, results: Iterable[RetrievedTurn]) -> list[str]:
        payloads = []
        for result in results:
            if result.payload_unit_index is None:
                raise RuntimeError(f"turn {result.turn_index!r} 缺少 payload 单元。")
            payloads.append(self.units[result.payload_unit_index].text)
        return payloads

    def probe(self, query: str, k: int) -> list[ScoredUnit]:
        if k < 1:
            raise ValueError("k 必须大于 0。")
        return self.scored(query)[: min(k, len(self.units))]
