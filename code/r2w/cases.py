"""R2W-GBM 的冻结案例库。

案例只提供局部特征/审计，不再参与线上效应的加权修正。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import R2WConfig


@dataclass
class Case:
    key: list[float]
    effects: list[float]
    text: str
    count: int = 1


class CaseMemory:
    def __init__(self, cfg: R2WConfig, cases: list[Case] | None = None):
        self.cfg = cfg
        self.cases = cases or []

    def add(self, key: np.ndarray, effects: np.ndarray, text: str) -> None:
        vector = np.asarray(key, dtype=np.float32)
        values = np.asarray(effects, dtype=np.float32)
        if values.shape != (10,):
            raise ValueError("案例效应必须覆盖十个存储臂。")
        if len(self.cases) >= self.cfg.case_max_size:
            self.cases.pop(0)
        self.cases.append(Case(vector.astype(float).tolist(), values.astype(float).tolist(), str(text)))

    def nearest(self, key: np.ndarray) -> tuple[float, np.ndarray] | None:
        if not self.cases:
            return None
        query = np.asarray(key, dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        values = np.asarray([case.key for case in self.cases], dtype=np.float32)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        index = int(np.argmax(values @ query))
        return float(values[index] @ query), np.asarray(self.cases[index].effects, dtype=np.float32)

    def local_audit(self, key: np.ndarray) -> dict | None:
        """返回冻结案例库的局部审计特征，不参与线上效应混合或动作修正。"""
        if not self.cases:
            return None
        query = np.asarray(key, dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        vectors = np.asarray([case.key for case in self.cases], dtype=np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        similarities = vectors @ query
        selected = np.argsort(-similarities, kind="stable")[: self.cfg.case_k]
        selected = [index for index in selected if similarities[index] >= self.cfg.case_similarity_floor]
        if not selected:
            return {
                "top_similarity": float(similarities.max()),
                "support": 0,
                "mean_effects": None,
            }
        weights = np.maximum(similarities[selected], 1e-6)
        effects = np.asarray([self.cases[index].effects for index in selected], dtype=np.float32)
        return {
            "top_similarity": float(similarities[selected[0]]),
            "support": len(selected),
            "mean_effects": (weights @ effects / weights.sum()).astype(float).tolist(),
        }

    def to_list(self) -> list[dict]:
        return [case.__dict__ for case in self.cases]

    @classmethod
    def from_list(cls, cfg: R2WConfig, values: list[dict]) -> "CaseMemory":
        return cls(cfg, [Case(**value) for value in values])
