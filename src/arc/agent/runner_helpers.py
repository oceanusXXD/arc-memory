"""Small shared helpers for running real stages outside the batch evaluator.

The batch evaluator owns audit files and manifests.  These helpers exist so a
single real question can be driven through the same code path during
verification without duplicating the retrieval or scoring logic.
"""
from __future__ import annotations

from typing import Any

from .data.locomo import load_blocks
from .data.retrieval import blocks_by_source, load_retrieval, select_sources
from .data.score import score_prediction
from .memory import render_memory_text
from .runtime import question_answer_task, run_agent
from ..algorithm.memory import configured_input_cost


def build_evidence_for(config: dict[str, Any], qa: dict[str, Any], blocks: list[dict] | None = None) -> list[dict]:
    """Return the retrieved evidence E for one question using the frozen cache.

    Falls back to the question's own labelled support blocks when the retrieval
    cache has not been built yet, so verification can still exercise the
    builder and agent stages with genuine corpus text.
    """
    blocks = blocks if blocks is not None else load_blocks(config)
    lookup = blocks_by_source(blocks)
    try:
        retrieval = load_retrieval(config)
    except (FileNotFoundError, ValueError):
        retrieval = {}
    row = retrieval.get((str(qa["sample_id"]), str(qa["qa_id"])))
    if row is not None:
        evidence, _ = select_sources(
            row,
            lookup,
            int(config["retrieval"].get("source_cap_tokens", 8192)),
            int(config["retrieval"].get("max_blocks", 64)),
            cost_fn=lambda candidate: configured_input_cost(config, qa["question"], candidate),
            input_limit=int(config["budgets"].get("builder_input_limit", 15872)),
        )
        if evidence:
            return evidence
    # No cache: use the labelled evidence turns directly.  These are real
    # corpus rows, not synthetic filler.
    wanted = {str(value) for value in qa.get("evidence") or []}
    chosen = [block for block in blocks
              if str(block.get("sample_id")) == str(qa["sample_id"])
              and str(block.get("dia_id")) in wanted]
    return chosen[:16]


def answer_one(config: dict[str, Any], qa: dict[str, Any], evidence: list[dict]) -> dict[str, Any]:
    """Answer one real question from injected memory and score it."""
    memory_text = render_memory_text(evidence)
    try:
        result = run_agent(question_answer_task(qa["question"]), memory_text, config=config)
    except Exception as exc:  # noqa: BLE001 - verification records the failure
        return {"status": "agent_error", "error": str(exc)[:500],
                "agent_usage": dict(getattr(exc, "usage", {}) or {})}
    usage = result.usage
    return {
        "status": "ok",
        "prediction": result.outcome,
        "reference": qa.get("answer", ""),
        "score": score_prediction(result.outcome, qa.get("answer", ""), int(qa.get("category", 3))),
        "agent_usage": usage,
    }
