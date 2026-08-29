"""用真实 query 和人工逐臂 value 验证 pooled GBM 的 action 选择能力。

这是一个纯离线诊断：不调用 embedding、LLM、QG、reader 或外部 API。人工清单只负责
给真实 LoCoMo query 指定一个 value profile；每个 profile 必须显式包含十个 action 的
oracle value。GBM 仍通过生产代码的 ``train_gbm_ensemble`` 训练，并通过
``DecisionPolicy.decide`` 的 value argmax 选择动作。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .calibration import CostStatistics, DecisionPolicy
from .config import R2WConfig, default_config, load_config
from .model import train_gbm_ensemble
from .pipeline_train import load_training_data
from .representations import ACTION_FACTORS, STRUCT_ACTIONS
from .text import date_count, lexical_overlap, sentences, tokenize, word_count


QUERY_STARTERS = ("when", "who", "where", "why", "how", "which", "what")
FEATURE_NAMES = (
    "memory_words",
    "log_memory_words",
    "memory_sentences",
    "memory_unique_ratio",
    "memory_digit_tokens",
    "memory_date_mentions",
    "evidence_turn_count",
    "log_evidence_turn_count",
    "mean_evidence_position",
    "evidence_position_span",
    "mean_evidence_turn_words",
    "max_evidence_turn_words",
    "query_words",
    "log_query_words",
    "query_unique_ratio",
    "query_digit_tokens",
    "query_date_mentions",
    *(f"starts_{starter}" for starter in QUERY_STARTERS),
    "starts_auxiliary",
    "mentions_both_or_common",
    "contains_inference_cue",
    "contains_count_cue",
    "contains_list_cue",
    "contains_time_cue",
    "query_memory_jaccard",
    "query_token_coverage",
)


def _load_manifest(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("query oracle 清单根节点必须是 object。")
    if manifest.get("schema_version") != "manual-query-oracle-v1":
        raise ValueError("query oracle schema_version 必须是 manual-query-oracle-v1。")
    profiles = manifest.get("value_profiles")
    examples = manifest.get("examples")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("query oracle 必须包含非空 value_profiles。")
    if not isinstance(examples, list) or not examples:
        raise ValueError("query oracle 必须包含非空 examples。")
    action_set = set(STRUCT_ACTIONS)
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(profile, dict):
            raise ValueError("value profile 必须是命名 object。")
        values = profile.get("values")
        if not isinstance(values, dict) or set(values) != action_set:
            raise ValueError(f"profile {name!r} 必须给齐十个 action value。")
        numeric = np.asarray([values[action] for action in STRUCT_ACTIONS], dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"profile {name!r} 的 action value 必须是有限数。")
        if int(np.sum(numeric == numeric.max())) != 1:
            raise ValueError(f"profile {name!r} 必须有唯一最佳 action。")
    seen = set()
    for row in examples:
        if not isinstance(row, dict):
            raise ValueError("每条 query oracle example 必须是 object。")
        required = {"conversation_id", "question", "split", "profile"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"query oracle example 缺少 {sorted(missing)}。")
        if row["split"] not in {"train", "test"}:
            raise ValueError("query oracle split 只能为 train 或 test。")
        if row["profile"] not in profiles:
            raise ValueError(f"未知 value profile: {row['profile']!r}。")
        key = (row["conversation_id"], row["question"])
        if key in seen:
            raise ValueError(f"query oracle example 重复: {key!r}。")
        seen.add(key)
    return manifest


def _select_examples(conversations: list[dict], manifest: dict) -> list[dict]:
    conversations_by_id = {
        conversation["conversation_id"]: conversation for conversation in conversations
    }
    selected = []
    for label in manifest["examples"]:
        conversation = conversations_by_id.get(label["conversation_id"])
        if conversation is None:
            raise ValueError(f"找不到 conversation {label['conversation_id']!r}。")
        matches = [
            query for query in conversation["qa"] if query["question"] == label["question"]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"conversation {label['conversation_id']!r} 中 query 必须恰好匹配一次: "
                f"{label['question']!r}。"
            )
        query = matches[0]
        if not query["evidence"]:
            raise ValueError("query oracle example 必须至少有一条真实 evidence turn。")
        evidence_indexes = [conversation["dia_to_index"][value] for value in query["evidence"]]
        evidence_turns = [conversation["turns"][index] for index in evidence_indexes]
        profile = manifest["value_profiles"][label["profile"]]
        values = np.asarray(
            [profile["values"][action] for action in STRUCT_ACTIONS], dtype=np.float32
        )
        selected.append(
            {
                "label": label,
                "query": query,
                "conversation": conversation,
                "evidence_indexes": evidence_indexes,
                "evidence_turns": evidence_turns,
                "memory_text": "\n".join(turn["text"] for turn in evidence_turns),
                "oracle_values": values,
                "oracle_index": int(np.argmax(values)),
            }
        )
    return selected


def _contains_any(text: str, phrases: tuple[str, ...]) -> float:
    return float(any(phrase in text for phrase in phrases))


def _base_features(item: dict) -> np.ndarray:
    """仅用真实 query、evidence memory 和位置生成确定性数值特征。"""
    memory = item["memory_text"]
    question = item["query"]["question"]
    lowered = question.casefold().strip()
    memory_tokens = tokenize(memory)
    query_tokens = tokenize(question)
    memory_count = max(len(memory_tokens), 1)
    query_count = max(len(query_tokens), 1)
    conversation_size = max(len(item["conversation"]["turns"]) - 1, 1)
    positions = np.asarray(item["evidence_indexes"], dtype=np.float32) / conversation_size
    evidence_lengths = [word_count(turn["text"]) for turn in item["evidence_turns"]]
    query_set = set(query_tokens)
    memory_set = set(memory_tokens)
    values = [
        float(memory_count),
        float(np.log1p(memory_count)),
        float(len(sentences(memory))),
        float(len(memory_set) / memory_count),
        float(sum(any(character.isdigit() for character in token) for token in memory_tokens)),
        float(date_count(memory)),
        float(len(item["evidence_turns"])),
        float(np.log1p(len(item["evidence_turns"]))),
        float(positions.mean()),
        float(positions.max() - positions.min()),
        float(np.mean(evidence_lengths)),
        float(max(evidence_lengths)),
        float(query_count),
        float(np.log1p(query_count)),
        float(len(query_set) / query_count),
        float(sum(any(character.isdigit() for character in token) for token in query_tokens)),
        float(date_count(question)),
        *(float(lowered.startswith(starter + " ")) for starter in QUERY_STARTERS),
        float(lowered.startswith(("is ", "does ", "do ", "did ", "would ", "could "))),
        _contains_any(lowered, (" both ", "common", "share", "together", "relationship")),
        _contains_any(lowered, ("likely", "might", "would", "suspected", "prefer", "based on")),
        _contains_any(lowered, ("how many", "how long", "number of", "years passed")),
        _contains_any(
            lowered,
            (
                "what are",
                "which places",
                "which bands",
                "what kinds",
                "what kind of activities",
                "items",
                "goals",
                "hobbies",
                "dreams",
                "types of",
            ),
        ),
        _contains_any(
            lowered,
            ("when ", "which year", "which month", "what year", "what month", " on ", " in 20"),
        ),
        float(lexical_overlap(question, memory)),
        float(len(query_set & memory_set) / max(len(query_set), 1)),
    ]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(FEATURE_NAMES),):
        raise AssertionError("query oracle 特征名和数值没有对齐。")
    return result


def _draft_statistics(item: dict, cfg: R2WConfig) -> np.ndarray:
    """构造无需真实 key/summary 内容的可审计草稿维度代理。"""
    source_words = max(word_count(item["memory_text"]), 1)
    query_words = max(word_count(item["query"]["question"]), 1)
    evidence_count = len(item["evidence_turns"])
    temporal_mentions = max(
        date_count(item["memory_text"] + " " + item["query"]["question"]), 1
    )
    rows = np.zeros((len(STRUCT_ACTIONS), 5), dtype=np.float32)
    for index, action in enumerate(STRUCT_ACTIONS):
        compression, key_axis = ACTION_FACTORS[action]
        payload_words = (
            source_words
            if compression == 0
            else max(1, math.ceil(source_words * cfg.summary_ratio))
        )
        if key_axis == 0:
            key_words, key_count = 0, 0
        elif key_axis == 1:
            key_count = min(4, max(1, evidence_count))
            key_words = min(cfg.key_budget_words, 6 + query_words + 2 * key_count)
        elif key_axis == 2:
            key_count = min(cfg.event_max_items, temporal_mentions)
            key_words = min(cfg.key_budget_words, 8 + 4 * key_count)
        elif key_axis == 3:
            key_count = min(cfg.graph_max_items, max(1, evidence_count))
            key_words = min(cfg.key_budget_words, 10 + 6 * key_count)
        else:
            key_count = 1
            key_words = min(cfg.hq_total_words, query_words)
        rows[index] = (
            float(payload_words),
            float(key_words),
            float(key_count),
            float(payload_words / source_words),
            float(compression),
        )
    return rows


def _costs_from_drafts(drafts: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (drafts[:, 0] + drafts[:, 1], drafts[:, 1], drafts[:, 0])
    ).astype(np.float32)


def _arrays(items: list[dict], cfg: R2WConfig) -> tuple[np.ndarray, ...]:
    base = np.vstack([_base_features(item) for item in items]).astype(np.float32)
    drafts = np.stack([_draft_statistics(item, cfg) for item in items]).astype(np.float32)
    effects = np.stack([item["oracle_values"] for item in items]).astype(np.float32)
    hits = (effects >= 0.5).astype(np.float32)
    groups = np.asarray(
        [item["label"]["conversation_id"] for item in items], dtype=object
    )
    return base, drafts, effects, hits, groups


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def run_query_oracle_validation(
    cfg: R2WConfig, data_path: str | Path, manifest_path: str | Path
) -> dict:
    manifest = _load_manifest(manifest_path)
    items = _select_examples(load_training_data(data_path), manifest)
    train_items = [item for item in items if item["label"]["split"] == "train"]
    test_items = [item for item in items if item["label"]["split"] == "test"]
    train_groups = {item["label"]["conversation_id"] for item in train_items}
    test_groups = {item["label"]["conversation_id"] for item in test_items}
    if len(train_groups) < cfg.gbm_folds:
        raise ValueError("query oracle 训练部分至少需要五段独立 conversation。")
    if not test_items or train_groups & test_groups:
        raise ValueError("query oracle test 必须非空且与 train conversation 完全隔离。")

    train_base, train_drafts, train_effects, train_hits, groups = _arrays(
        train_items, cfg
    )
    model = train_gbm_ensemble(
        cfg,
        train_base,
        train_effects.max(axis=1),
        train_base,
        train_drafts,
        train_effects,
        train_hits,
        np.ones_like(train_effects, dtype=bool),
        np.full_like(train_effects, 0.05, dtype=np.float32),
        groups,
    )
    test_base, test_drafts, _, _, _ = _arrays(test_items, cfg)
    admission, admission_sigma = model.predict_admission(test_base)
    predicted_effects, predicted_hits, predicted_sigma = model.predict_arms(
        test_base, test_drafts
    )
    policy = DecisionPolicy(
        lambda_w=0.0,
        lambda_r=0.0,
        lambda_s=0.0,
        kappa_e=0.0,
        kappa_f=0.0,
        fidelity_ratio=cfg.fidelity_ratio,
        cost_statistics=CostStatistics(0.0, 0.0, 0.0),
        config_hash=cfg.hash(),
    )

    rows = []
    for index, item in enumerate(test_items):
        costs = _costs_from_drafts(test_drafts[index])
        decision = policy.decide(
            admission[index],
            admission_sigma[index],
            predicted_effects[index],
            predicted_hits[index],
            predicted_sigma[index],
            costs,
        )
        if decision.action_index == "none":
            predicted_index = None
            predicted_action = "none"
        else:
            predicted_index = int(decision.action_index)
            predicted_action = STRUCT_ACTIONS[predicted_index]
        oracle_index = item["oracle_index"]
        order = np.argsort(-predicted_effects[index], kind="stable")
        oracle_rank = int(np.flatnonzero(order == oracle_index)[0]) + 1
        chosen_oracle_value = (
            0.0
            if predicted_index is None
            else float(item["oracle_values"][predicted_index])
        )
        rows.append(
            {
                "conversation_id": item["label"]["conversation_id"],
                "question": item["query"]["question"],
                "evidence": item["query"]["evidence"],
                "profile": item["label"]["profile"],
                "oracle_action": STRUCT_ACTIONS[oracle_index],
                "predicted_action": predicted_action,
                "match": predicted_action == STRUCT_ACTIONS[oracle_index],
                "oracle_rank_by_predicted_value": oracle_rank,
                "top3_contains_oracle": oracle_rank <= 3,
                "oracle_regret": float(item["oracle_values"].max() - chosen_oracle_value),
                "value_pearson": _pearson(
                    item["oracle_values"], predicted_effects[index]
                ),
                "oracle_values": {
                    action: float(item["oracle_values"][arm])
                    for arm, action in enumerate(STRUCT_ACTIONS)
                },
                "predicted_values": {
                    action: float(predicted_effects[index, arm])
                    for arm, action in enumerate(STRUCT_ACTIONS)
                },
            }
        )

    matched = sum(row["match"] for row in rows)
    confusion = Counter(
        f"{row['oracle_action']} -> {row['predicted_action']}" for row in rows
    )
    train_action_counts = Counter(
        STRUCT_ACTIONS[item["oracle_index"]] for item in train_items
    )
    test_action_counts = Counter(
        STRUCT_ACTIONS[item["oracle_index"]] for item in test_items
    )
    majority_action = train_action_counts.most_common(1)[0][0]
    majority_matches = test_action_counts[majority_action]
    absolute_errors = np.abs(
        predicted_effects
        - np.stack([item["oracle_values"] for item in test_items]).astype(np.float32)
    )
    return {
        "kind": "manual_query_value_gbm_diagnostic",
        "warning": (
            "人工 value profile 只验证 GBM 的逐臂 value 学习和 argmax 链路；"
            "它不是 L2 causal-effect 或 LoCoMo benchmark 结果。"
        ),
        "offline_guarantees": [
            "no_api",
            "no_embedding",
            "no_llm",
            "no_query_generator",
            "no_reader",
        ],
        "feature_sources": [
            "real_query_lexical_statistics",
            "real_evidence_memory_statistics",
            "query_memory_interaction",
            "evidence_position",
            "deterministic_draft_dimensions",
            "production_arm_features",
        ],
        "excluded_feature_sources": [
            "answer",
            "locomo_category",
            "oracle_profile_name",
            "oracle_action",
            "oracle_values",
        ],
        "selection_rule": "DecisionPolicy with zero cost penalties; argmax(predicted action value)",
        "actions": list(STRUCT_ACTIONS),
        "base_feature_names": list(FEATURE_NAMES),
        "train_conversations": len(train_groups),
        "train_examples": len(train_items),
        "test_conversations": len(test_groups),
        "test_examples": len(test_items),
        "train_oracle_action_distribution": dict(sorted(train_action_counts.items())),
        "test_oracle_action_distribution": dict(sorted(test_action_counts.items())),
        "uniform_random_expected_accuracy": 1.0 / len(STRUCT_ACTIONS),
        "train_majority_action": majority_action,
        "train_majority_action_accuracy": majority_matches / len(test_items),
        "matched_oracle_actions": matched,
        "oracle_action_accuracy": matched / len(rows),
        "top3_oracle_accuracy": sum(row["top3_contains_oracle"] for row in rows)
        / len(rows),
        "mean_oracle_rank": float(
            np.mean([row["oracle_rank_by_predicted_value"] for row in rows])
        ),
        "mean_oracle_regret": float(np.mean([row["oracle_regret"] for row in rows])),
        "mean_value_mae": float(absolute_errors.mean()),
        "mean_value_pearson": float(np.mean([row["value_pearson"] for row in rows])),
        "confusion": dict(sorted(confusion.items())),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用真实 query 和人工十臂 value 离线验证 GBM action 选择"
    )
    parser.add_argument("--data", required=True, help="LoCoMo 原始 JSON。")
    parser.add_argument("--manifest", required=True, help="人工 query/value 清单。")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", help="可选 R2W v3 配置。")
    arguments = parser.parse_args()
    cfg = load_config(arguments.config) if arguments.config else default_config()
    result = run_query_oracle_validation(cfg, arguments.data, arguments.manifest)
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
