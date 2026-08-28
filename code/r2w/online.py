"""R2W-GBM 在线写入存储：热索引、冷归档和影子索引。"""

from __future__ import annotations

import hashlib

from .config import R2WConfig
from .representations import Repr, build_representation
from .retrieval import UnifiedIndex


class InMemoryColdArchive:
    def __init__(self):
        self._values: dict[str, dict] = {}

    def put(self, turn: dict) -> None:
        self._values[str(turn["dia_id"])] = dict(turn)

    def get(self, turn_id: str) -> dict:
        return dict(self._values[turn_id])

    def __contains__(self, turn_id: str) -> bool:
        return turn_id in self._values


class ShadowAwareMemoryWriter:
    """`none` 永远入冷归档；影子与 raw shadow 均不改变正式热索引。"""

    def __init__(self, cfg: R2WConfig, embedder):
        self.cfg = cfg
        self.embedder = embedder
        self.archive = InMemoryColdArchive()
        self.hot = UnifiedIndex([], embedder, cfg)
        self.none_shadow = UnifiedIndex([], embedder, cfg)
        self.raw_shadow = UnifiedIndex([], embedder, cfg)

    def _shadow(self, turn: dict) -> bool:
        payload = f"{self.cfg.random_seed}:{turn['dia_id']}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
        return value < self.cfg.none_shadow_probability

    def write_decision(self, action: str, representation: Repr | None, turn: dict, result: dict) -> None:
        if action == "none":
            self.archive.put(turn)
            if self._shadow(turn):
                raw = build_representation(turn, "raw", None, self.cfg)
                self.none_shadow.add_representation(turn["dia_id"], raw, int(turn["session"]))
            return
        if representation is None:
            raise ValueError("非 none 动作必须提供表示。")
        self.hot.add_representation(turn["dia_id"], representation, int(turn["session"]))
        if action != "raw":
            raw = build_representation(turn, "raw", None, self.cfg)
            self.raw_shadow.add_representation(turn["dia_id"], raw, int(turn["session"]))

    def write(self, action: str, units: list[str], turn: dict) -> None:
        raise RuntimeError("R2W-GBM writer 必须接收完整 Repr，不能只接收字符串单元。")
