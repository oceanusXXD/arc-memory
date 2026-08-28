"""R2W-GBM 测量审计工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .representations import STRUCT_ACTIONS


@dataclass(frozen=True)
class HTAuditResult:
    """保留历史导出名称；内容改为真实 L2 p-vs-none 分量审计。"""

    mean_delta_e: dict[str, float]
    mean_signed_psi: dict[str, float]
    measured_cells: int
    config_hash: str

    def to_dict(self, include_assignments: bool = False) -> dict:
        return {
            "config_hash": self.config_hash,
            "measurement": "raw-background p-vs-none endpoint QA",
            "mean_delta_e": self.mean_delta_e,
            "mean_signed_psi": self.mean_signed_psi,
            "measured_cells": self.measured_cells,
        }


def run_ht_audit(rows, embedder=None, cfg=None, reader=None) -> HTAuditResult:
    """汇总已测 L2 标签；不再把旧 tau_swap 当作主效应。"""
    if cfg is None:
        raise ValueError("R2W-GBM audit 需要配置。")
    effects, psi, masks = [], [], []
    for row in rows:
        swap = row[2] if isinstance(row, tuple) else row
        effects.append(np.asarray(swap.delta_e, dtype=np.float32))
        psi.append(np.asarray(swap.psi, dtype=np.float32))
        masks.append(np.asarray(swap.measured, dtype=bool))
    values, externality, measured = np.vstack(effects), np.vstack(psi), np.vstack(masks)
    return HTAuditResult(
        {action: float(values[:, index][measured[:, index]].mean()) if measured[:, index].any() else float("nan") for index, action in enumerate(STRUCT_ACTIONS)},
        {action: float(externality[:, index][measured[:, index]].mean()) if measured[:, index].any() else float("nan") for index, action in enumerate(STRUCT_ACTIONS)},
        int(measured.sum()),
        cfg.hash(),
    )


def save_ht_audit(result: HTAuditResult, output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "measurement_audit.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
