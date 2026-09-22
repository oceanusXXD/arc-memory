from __future__ import annotations

from collections import defaultdict
from typing import Any

from arc.agent.text import exact_match_score, f1_score, multi_answer_f1


def score_prediction(prediction: str, answer: str, category: int) -> float:
    output = str(prediction or "")
    gold = str(answer or "")
    if category == 1:
        return float(multi_answer_f1(output, gold))
    if category in {2, 4}:
        return float(f1_score(output, gold))
    if category == 3:
        return float(f1_score(output, gold.split(";")[0].strip()))
    if category == 5:
        lowered = output.lower()
        return 1.0 if "no information available" in lowered or "not mentioned" in lowered else 0.0
    raise ValueError(f"unknown LoCoMo category: {category}")


def exact_match_prediction(prediction: str, answer: str, category: int) -> float:
    if category == 3:
        answer = str(answer).split(";")[0].strip()
    if category == 5:
        text = str(prediction).lower()
        return float("no information available" in text or "not mentioned" in text)
    return exact_match_score(prediction, answer)


def retrieval_recall(evidence: list[str], retrieved_dia_ids: list[str]) -> float:
    if not evidence:
        return 1.0
    retrieved = set(retrieved_dia_ids)
    return sum(1 for ev in evidence if ev in retrieved) / len(evidence)


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(row)
    result: dict[str, Any] = {}
    for arm, items in sorted(by_arm.items()):
        cat_items = [r for r in items if int(r["category"]) in {1, 2, 3, 4}]
        adv_items = [r for r in items if int(r["category"]) == 5]
        group_scores: dict[str, list[float]] = defaultdict(list)
        cat_scores: dict[str, list[float]] = defaultdict(list)
        for row in items:
            score = float(row.get("score", 0.0))
            if int(row["category"]) != 5:
                group_scores[str(row["sample_id"])].append(score)
            cat_scores[str(row["category"])].append(score)
        result[arm] = {
            "items": len(items),
            "items_1_4": len(cat_items),
            "items_5": len(adv_items),
            "qa_weighted_1_4": sum(float(r.get("score", 0.0)) for r in cat_items) / len(cat_items) if cat_items else None,
            "category_5_refusal": sum(float(r.get("score", 0.0)) for r in adv_items) / len(adv_items) if adv_items else None,
            "group_macro": sum(sum(v) / len(v) for v in group_scores.values()) / len(group_scores) if group_scores else 0.0,
            "by_category": {cat: sum(vals) / len(vals) for cat, vals in sorted(cat_scores.items())},
            "exact_match": sum(float(r.get("exact_match", 0.0)) for r in cat_items) / len(cat_items) if cat_items else None,
            "ok_rate": sum(str(r.get("status")) == "ok" for r in items) / len(items) if items else 0.0,
            "construction_usage_complete_items": sum(bool(r.get("construction_usage_complete")) for r in items),
            "agent_usage_complete_items": sum(bool(r.get("agent_usage_complete")) for r in items),
            "fallback_items": sum(bool((r.get("selection") or {}).get("fallback")) for r in items),
        }
        for field in ("construction_prompt_tokens", "construction_completion_tokens", "construction_tokens",
                      "construction_calls", "agent_prompt_tokens", "agent_completion_tokens", "agent_total_tokens",
                      "memory_read_tokens", "memory_read_tokens_estimate", "selector_latency_ms", "retrieval_query_tokens", "retrieval_latency_ms", "total_latency_ms"):
            values = [r[field] for r in items if r.get(field) is not None]
            result[arm]["avg_" + field] = sum(values) / len(values) if values else None
            result[arm][field + "_known_items"] = len(values)
            result[arm]["total_" + field] = sum(values) if len(values) == len(items) else None
        times = [(r.get("agent_usage") or {}).get("latency_ms") for r in items]
        times = [v for v in times if v is not None]
        result[arm]["avg_agent_elapsed_seconds"] = sum(times) / len(times) / 1000 if times else None
    return result
