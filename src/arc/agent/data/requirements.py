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


def _history_candidates(config: dict[str, Any], sample_id: str,
                        history_by_sample: dict[str, list[dict[str, Any]]],
                        vectors_by_sample: dict[str, dict[str, list[float]]],
                        metadata_by_sample: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build E_i from observable history and frozen source vectors only.

    The compiler is allowed to receive a finite candidate set, but its
    deployment contract requires that set to be independent of the future
    question.  We therefore order a conversation by its observable session
    and turn fields, attach the already-frozen source vectors, and apply only
    the source/token caps from the configuration.
    """
    source_cap = int(config["retrieval"].get("source_cap_tokens", 8192))
    max_blocks = int(config["retrieval"].get("max_blocks", 64))
    input_limit = int(config["budgets"].get("builder_input_limit", 15872))
    vectors = vectors_by_sample.get(str(sample_id), {})
    metadata = metadata_by_sample.get(str(sample_id))
    if not metadata:
        raise ValueError(f"missing frozen embedding metadata for {sample_id}")
    ordered = sorted(
        history_by_sample.get(str(sample_id), []),
        key=lambda item: (
            int(item.get("session_index", 0)),
            int(item.get("turn_index", 0)),
            int(item.get("chunk_index", 0)),
            str(item.get("source_id")),
        ),
    )
    selected: list[dict[str, Any]] = []
    token_total = 0
    for block in ordered:
        source_id = str(block.get("source_id"))
        vector = vectors.get(source_id)
        if vector is None:
            # A source without a frozen vector cannot enter the selector
            # feature schema.  It remains outside this finite candidate
            # domain and is visible to the coverage audit.
            continue
        block_tokens = int(block.get("token_count") or 0)
        if len(selected) >= max_blocks or token_total + block_tokens > source_cap:
            continue
        item = dict(block)
        item["embedding"] = vector
        item["embedding_metadata"] = dict(metadata)
        if configured_input_cost(config, None, [*selected, item]) > input_limit:
            continue
        selected.append(item)
        token_total += block_tokens
    return selected


def _frozen_vectors(config: dict[str, Any]) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, Any]]]:
    vectors_by_sample: dict[str, dict[str, list[float]]] = {}
    metadata_by_sample: dict[str, dict[str, Any]] = {}
    for row in _iter_retrieval(config):
        sample_id = str(row.get("sample_id"))
        vectors_by_sample.setdefault(sample_id, {}).update(
            {str(source_id): vector for source_id, vector in (row.get("source_vectors") or {}).items()}
        )
        if row.get("embedding_metadata") and sample_id not in metadata_by_sample:
            metadata_by_sample[sample_id] = dict(row["embedding_metadata"])
    return vectors_by_sample, metadata_by_sample


def _compiler_row(config: dict[str, Any], retrieval: dict[str, Any], lookup: dict[str, dict],
                  qa_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
                  history_by_sample: dict[str, list[dict[str, Any]]] | None = None,
                  vectors_by_sample: dict[str, dict[str, list[float]]] | None = None,
                  metadata_by_sample: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    question = str(retrieval.get("question") or "")
    if history_by_sample is not None and vectors_by_sample is not None and metadata_by_sample is not None:
        sources = _history_candidates(config, str(retrieval.get("sample_id")), history_by_sample,
                                      vectors_by_sample, metadata_by_sample)
    else:
        # Compatibility path for callers that already materialized E_i.  New
        # pipeline entry points pass the history/vector maps above.
        sources, _ = select_sources(
            retrieval,
            lookup,
            int(config["retrieval"].get("source_cap_tokens", 8192)),
            int(config["retrieval"].get("max_blocks", 64)),
            cost_fn=lambda candidate: configured_input_cost(config, None, candidate),
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
        "candidate_sources": sources,
        "sources": sources,
        # The singleton is the smallest explicit Q_i for this benchmark's
        # per-QA update unit.  Larger history units may replace it with the
        # full future-query sample before compilation.
        "future_queries": [future_query],
    }


def _compiler_rows(config: dict[str, Any], lookup: dict[str, dict], path: Path) -> Iterator[dict[str, Any]]:
    qa_lookup = {(str(row.get("sample_id")), str(row.get("qa_id"))): row for row in load_qas(config)}
    blocks = load_blocks(config)
    history_by_sample: dict[str, list[dict[str, Any]]] = {}
    for block in blocks:
        history_by_sample.setdefault(str(block.get("sample_id")), []).append(block)
    vectors_by_sample, metadata_by_sample = _frozen_vectors(config)
    for retrieval in _iter_retrieval(config):
        _validate_retrieval_row(retrieval, path)
        yield _compiler_row(config, retrieval, lookup, qa_lookup,
                            history_by_sample, vectors_by_sample, metadata_by_sample)


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
