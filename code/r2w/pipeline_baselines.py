"""LoCoMo 上的 R2W 对照基线运行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import (
    evaluate_fixed_baseline,
    evaluate_full_context_baseline,
    evaluate_none_baseline,
)
from .config import default_config, load_config
from .constructor import build_constructor
from .embedding import EmbeddingBackend
from .llm import LLMClient
from .pipeline_train import load_training_data
from .query_generator import QueryGenerator
from .representations import STRUCT_ACTIONS
from .runtime_usage import provider_usage_payload
from .teacher import LLMReaderAdapter


def evaluate_baselines(
    cfg,
    conversations: list[dict],
    actions: list[str],
    *,
    embedder=None,
    constructor=None,
    qg=None,
    reader=None,
    llm=None,
    include_none: bool = True,
    include_full_context: bool = True,
) -> dict:
    """输出与 R2W 共用 reader/retriever 的固定写入基线。"""
    invalid = sorted(set(actions) - set(STRUCT_ACTIONS))
    if invalid:
        raise ValueError(f"未知 baseline action: {invalid}")
    if not actions:
        raise ValueError("至少指定一个固定表示动作。")
    embedder = embedder or EmbeddingBackend(cfg)
    llm = llm or LLMClient(cfg)
    reader = reader or LLMReaderAdapter(llm)
    needs_constructor = any(action.endswith(("+kv", "+event", "+graph")) for action in actions)
    needs_qg = any(action.endswith("+hq") for action in actions)
    constructor = constructor or (build_constructor(cfg, llm) if needs_constructor else None)
    qg = qg or (QueryGenerator(cfg) if needs_qg else None)
    results = []
    if include_none:
        results.append(evaluate_none_baseline(conversations, reader).to_dict())
    if include_full_context:
        results.append(evaluate_full_context_baseline(conversations, cfg, reader).to_dict())
    for action in actions:
        results.append(
            evaluate_fixed_baseline(
                conversations,
                action,
                embedder,
                cfg,
                reader,
                constructor=constructor,
                qg=qg,
            ).to_dict()
        )
    return {
        "config_hash": cfg.hash(),
        "metric": "token_f1",
        "cost_unit": "word_count",
        "results": results,
        "provider_usage": provider_usage_payload(llm=llm, embedding=embedder),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 R2W 的固定表示与 Naive RAG 基线")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", help="完整 R2WConfig JSON。")
    parser.add_argument(
        "--actions",
        default="raw,sum",
        help="逗号分隔的固定表示；raw 即 always-raw Naive RAG。",
    )
    parser.add_argument("--without-none", action="store_true")
    parser.add_argument("--without-full-context", action="store_true")
    arguments = parser.parse_args()
    actions = [value.strip() for value in arguments.actions.split(",") if value.strip()]
    cfg = load_config(arguments.config) if arguments.config else default_config()
    result = evaluate_baselines(
        cfg,
        load_training_data(arguments.data),
        actions,
        include_none=not arguments.without_none,
        include_full_context=not arguments.without_full_context,
    )
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
