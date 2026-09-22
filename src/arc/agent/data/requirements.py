"""Build compiler input rows from the frozen retrieval cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from arc.agent.config import cache_path, load_config, write_manifest
from arc.agent.data.locomo import load_blocks, load_qas
from arc.agent.data.retrieval import blocks_by_source, load_retrieval, select_sources
from arc.agent.io import write_jsonl
from arc.algorithm.memory import configured_input_cost


def _iter_retrieval(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    path = cache_path(config, "retrieval", "locomo_topk.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run python -m arc.agent.data.retrieval index first")
    yield from load_retrieval(config).iter_rows()


def _validate_retrieval_row(row: dict[str, Any], path: Path) -> None:
    required = {"sample_id", "qa_id", "source_ids", "scores", "backend", "query_vector", "source_vectors", "embedding_metadata"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"{path}: row {row.get('qa_id')} missing required fields: {sorted(missing)}")
    if row["backend"] != "hybrid_rrf":
        raise ValueError(f"{path}: row {row.get('qa_id')} backend must be hybrid_rrf, got {row['backend']!r}")
    query_vector = np.asarray(row.get("query_vector"), dtype="float32")
    if query_vector.ndim != 1 or not np.isfinite(query_vector).all():
        raise ValueError(f"{path}: row {row.get('qa_id')} has an invalid query_vector")


def _compiler_row(config: dict[str, Any], retrieval: dict[str, Any], lookup: dict[str, dict],
                  qa_lookup: dict[tuple[str, str], dict[str, Any]] | None = None) -> dict[str, Any]:
    question = str(retrieval.get("question") or "")
    sources, _ = select_sources(
        retrieval,
        lookup,
        int(config["retrieval"].get("source_cap_tokens", 8192)),
        int(config["retrieval"].get("max_blocks", 64)),
        cost_fn=lambda candidate: configured_input_cost(config, question, candidate),
        input_limit=int(config["budgets"].get("builder_input_limit", 15872)),
    )
    if not sources:
        raise ValueError(f"no compiler sources selected for {retrieval.get('sample_id')} / {retrieval.get('qa_id')}")
    qa = (qa_lookup or {}).get((str(retrieval.get("sample_id")), str(retrieval.get("qa_id"))), {})
    future_query = {"question": question, "answer": qa.get("answer"), "category": qa.get("category")}
    return {
        "sample_id": str(retrieval.get("sample_id")),
        "qa_id": str(retrieval.get("qa_id")),
        "question": question,
        "sources": sources,
        # The singleton is the smallest explicit Q_i for this benchmark's
        # per-QA update unit.  Larger history units may replace it with the
        # full future-query sample before compilation.
        "future_queries": [future_query],
    }


def _compiler_rows(config: dict[str, Any], lookup: dict[str, dict], path: Path) -> Iterator[dict[str, Any]]:
    qa_lookup = {(str(row.get("sample_id")), str(row.get("qa_id"))): row for row in load_qas(config)}
    for retrieval in _iter_retrieval(config):
        _validate_retrieval_row(retrieval, path)
        yield _compiler_row(config, retrieval, lookup, qa_lookup)


def build_requirements(config: dict[str, Any], output: str | Path | None = None) -> dict[str, Any]:
    blocks = load_blocks(config)
    lookup = blocks_by_source(blocks)
    cache_path_value = cache_path(config, "retrieval", "locomo_topk.jsonl")
    target = Path(output) if output else Path(config["paths"]["processed_dir"]) / "requirements.jsonl"
    count = write_jsonl(target, _compiler_rows(config, lookup, cache_path_value))
    write_manifest(config, "requirements", {"input": str(cache_path_value), "output": str(target), "items": count})
    return {"output": str(target), "items": count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compiler input rows from the frozen retrieval cache.")
    parser.add_argument("--config", default="configs/locomo.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(build_requirements(load_config(args.config), args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
