"""R2W-GBM 的离线 L1/L2 与一次策略背景审计入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .constructor import build_constructor
from .embedding import EmbeddingBackend
from .llm import LLMClient
from .pipeline_train import load_training_data
from .query_generator import QueryGenerator
from .teacher import LLMReaderAdapter, build_effect_targets, run_swap_measure, run_uniform_measure
from .config import default_config, load_config


def audit_pipeline(cfg, conversations: list[dict], output_dir: str | Path) -> dict:
    embedder, qg, llm = EmbeddingBackend(cfg), QueryGenerator(cfg), LLMClient(cfg)
    constructor, reader = build_constructor(cfg, llm), LLMReaderAdapter(llm)
    effects, psi, measured = [], [], []
    for conversation in conversations:
        uniform = run_uniform_measure(conversation, embedder, constructor, qg, cfg, reader)
        swap = run_swap_measure(conversation, embedder, cfg, uniform, reader, np.ones((len(conversation["turns"]), 10), dtype=bool))
        targets = build_effect_targets(conversation, uniform, cfg, swap)
        effects.append(targets.effects)
        psi.append(targets.psi)
        measured.append(targets.measured)
    payload = {
        "config_hash": cfg.hash(),
        "measurement": "raw-background p-vs-none endpoint QA",
        "mean_delta_e": np.vstack(effects).mean(axis=0).tolist(),
        "mean_signed_psi": np.vstack(psi).mean(axis=0).tolist(),
        "measured_cells": int(np.vstack(measured).sum()),
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "measurement_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 R2W-GBM 测量审计")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", help="完整 R2WConfig JSON。")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else default_config()
    print(json.dumps(audit_pipeline(cfg, load_training_data(args.data), args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
