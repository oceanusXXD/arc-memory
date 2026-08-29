"""以人工 oracle 标签检查 R2W 的十臂 GBM 是否能在未见对话上选对动作。

这个工具是诊断，不生成训练 artifact，也不替代 L2 ``p vs none`` 的真实效应标签。
标签必须由人工写入 JSON；特征只读取 LongMemEval session 内容、日期和位置，绝不读取
问题、答案、evidence 或 question_type。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibration import CostStatistics, DecisionPolicy
from .config import R2WConfig, default_config, load_config
from .costs import representation_costs
from .model import train_gbm_ensemble
from .pipeline_train import _longmemeval_datetime
from .representations import STRUCT_ACTIONS, build_representation
from .text import date_count, normalize_relative_time, sentences, tokenize, word_count


def _load_labels(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, list) or not labels:
        raise ValueError("oracle 标签文件必须是非空 JSON 数组。")
    seen = set()
    for row in labels:
        if not isinstance(row, dict):
            raise ValueError("每条 oracle 标签必须是 object。")
        required = {"question_id", "session_id", "action", "split"}
        if required - set(row):
            raise ValueError(f"oracle 标签缺少 {sorted(required - set(row))}。")
        if row["action"] not in {"raw", "sum"}:
            raise ValueError("当前人工 oracle 诊断只接受实际构造的 raw 或 sum。")
        if row["split"] not in {"train", "test"}:
            raise ValueError("oracle 标签 split 只能为 train 或 test。")
        key = (row["question_id"], row["session_id"])
        if key in seen:
            raise ValueError(f"oracle 标签重复: {key}。")
        seen.add(key)
    return labels


def _session_turns(data_path: str | Path, labels: list[dict]) -> list[dict]:
    """按标签取 session；本函数有意不访问任何 QA 字段。"""
    with Path(data_path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("LongMemEval 数据根节点必须是数组。")
    records_by_id = {
        row.get("question_id"): row for row in records if isinstance(row, dict)
    }
    selected = []
    for label in labels:
        record = records_by_id.get(label["question_id"])
        if record is None:
            raise ValueError(f"找不到 LongMemEval record {label['question_id']!r}。")
        sessions = record.get("haystack_sessions")
        session_ids = record.get("haystack_session_ids")
        dates = record.get("haystack_dates")
        if not isinstance(sessions, list) or not isinstance(session_ids, list) or not isinstance(dates, list):
            raise ValueError("LongMemEval record 缺少 haystack session 字段。")
        try:
            position = session_ids.index(label["session_id"])
        except ValueError as exc:
            raise ValueError(
                f"record {label['question_id']!r} 不含 session {label['session_id']!r}。"
            ) from exc
        if position >= len(sessions) or position >= len(dates):
            raise ValueError("LongMemEval session/id/date 长度不一致。")
        session = sessions[position]
        if not isinstance(session, list) or not session:
            raise ValueError("被标记的 LongMemEval session 为空。")
        lines = []
        for dialog_turn in session:
            if not isinstance(dialog_turn, dict):
                raise ValueError("LongMemEval session turn 必须是 object。")
            role, content = dialog_turn.get("role"), dialog_turn.get("content")
            if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
                raise ValueError("LongMemEval session turn 缺少 role/content。")
            lines.append(f"{role}: {content}")
        timestamp = _longmemeval_datetime(dates[position], "haystack_dates")
        source_text = "\n".join(lines)
        selected.append(
            {
                "label": label,
                "turn": {
                    "dia_id": str(label["session_id"]),
                    "speaker": "session",
                    "text": normalize_relative_time(source_text, timestamp),
                    "timestamp": timestamp,
                    "session": position + 1,
                },
                "session_position": position,
                "session_count": len(sessions),
            }
        )
    return selected


def _base_features(item: dict) -> np.ndarray:
    text = item["turn"]["text"]
    tokens = tokenize(text)
    count = max(len(tokens), 1)
    unique_ratio = len(set(tokens)) / count
    return np.asarray(
        [
            float(count),
            float(np.log1p(count)),
            float(len(sentences(text))),
            float(unique_ratio),
            float(sum(any(character.isdigit() for character in token) for token in tokens)),
            float(date_count(text)),
            float(item["session_position"]) / max(item["session_count"] - 1, 1),
        ],
        dtype=np.float32,
    )


def _drafts_and_costs(item: dict, cfg: R2WConfig) -> tuple[np.ndarray, np.ndarray]:
    """raw/sum 是真实构造；其余臂只作为带负标签的未选候选占位。"""
    turn = item["turn"]
    source_words = max(word_count(turn["text"]), 1)
    raw = build_representation(turn, "raw", None, cfg)
    summary = build_representation(turn, "sum", None, cfg)
    raw_cost = np.asarray(representation_costs(raw), dtype=np.float32)
    summary_cost = np.asarray(representation_costs(summary), dtype=np.float32)
    drafts = np.zeros((len(STRUCT_ACTIONS), 5), dtype=np.float32)
    costs = np.zeros((len(STRUCT_ACTIONS), 3), dtype=np.float32)
    for index, action in enumerate(STRUCT_ACTIONS):
        representation = raw if index < 5 else summary
        payload_words = word_count(representation.payload)
        # key arms 的真实内容未被人工标注为可选，因此只给成本上界，且它们
        # 永远不会得到正 oracle 效应标签。
        key_words = 0 if action in {"raw", "sum"} else cfg.key_budget_words
        drafts[index] = (
            float(payload_words),
            float(key_words),
            float(key_words > 0),
            float(payload_words / source_words),
            float(index >= 5),
        )
        base_cost = raw_cost if index < 5 else summary_cost
        costs[index] = base_cost + np.asarray((key_words, key_words, 0.0), dtype=np.float32)
    return drafts, costs


def run_oracle_validation(
    cfg: R2WConfig, data_path: str | Path, labels_path: str | Path
) -> dict:
    labels = _load_labels(labels_path)
    items = _session_turns(data_path, labels)
    train_items = [item for item in items if item["label"]["split"] == "train"]
    test_items = [item for item in items if item["label"]["split"] == "test"]
    train_groups = {item["label"]["question_id"] for item in train_items}
    test_groups = {item["label"]["question_id"] for item in test_items}
    if len(train_groups) < cfg.gbm_folds:
        raise ValueError("人工 oracle 训练部分至少需要五段独立对话。")
    if not test_items or train_groups & test_groups:
        raise ValueError("人工 oracle test 必须非空且与 train conversation 完全隔离。")

    def arrays(values: list[dict]):
        base = np.vstack([_base_features(item) for item in values]).astype(np.float32)
        drafts_and_costs = [_drafts_and_costs(item, cfg) for item in values]
        drafts = np.stack([value[0] for value in drafts_and_costs]).astype(np.float32)
        costs = np.stack([value[1] for value in drafts_and_costs]).astype(np.float32)
        effects = np.full((len(values), len(STRUCT_ACTIONS)), -1.0, dtype=np.float32)
        hits = np.zeros_like(effects)
        for row, item in enumerate(values):
            action = STRUCT_ACTIONS.index(item["label"]["action"])
            effects[row, action] = 1.0
            hits[row, action] = 1.0
        groups = np.asarray([item["label"]["question_id"] for item in values], dtype=object)
        return base, drafts, costs, effects, hits, groups

    train_base, train_drafts, train_costs, train_effects, train_hits, groups = arrays(train_items)
    model = train_gbm_ensemble(
        cfg,
        train_base,
        train_effects.max(axis=1),
        train_base,
        train_drafts,
        train_effects,
        train_hits,
        np.ones_like(train_effects, dtype=bool),
        np.full_like(train_effects, 0.1, dtype=np.float32),
        groups,
    )
    test_base, test_drafts, test_costs, _, _, _ = arrays(test_items)
    admission, admission_sigma = model.predict_admission(test_base)
    effects, hits, sigma = model.predict_arms(test_base, test_drafts)
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
        decision = policy.decide(
            admission[index],
            admission_sigma[index],
            effects[index],
            hits[index],
            sigma[index],
            test_costs[index],
        )
        prediction = (
            "none"
            if decision.action_index == "none"
            else STRUCT_ACTIONS[int(decision.action_index)]
        )
        oracle = item["label"]["action"]
        rows.append(
            {
                "question_id": item["label"]["question_id"],
                "session_id": item["label"]["session_id"],
                "oracle_action": oracle,
                "predicted_action": prediction,
                "match": prediction == oracle,
                "word_count": word_count(item["turn"]["text"]),
                "predicted_oracle_effect": float(
                    effects[index, STRUCT_ACTIONS.index(oracle)]
                ),
                "predicted_selected_effect": float(
                    effects[index, int(decision.action_index)]
                )
                if decision.action_index != "none"
                else None,
            }
        )
    matches = sum(row["match"] for row in rows)
    return {
        "kind": "manual_oracle_gbm_diagnostic",
        "warning": "Manual raw/sum oracle labels are not L2 causal effects or a benchmark result.",
        "config_hash": cfg.hash(),
        "actions_with_human_positive_labels": ["raw", "sum"],
        "feature_sources": ["session_text", "session_date", "session_position"],
        "forbidden_feature_sources": ["question", "answer", "evidence", "question_type"],
        "train_conversations": len(train_groups),
        "train_examples": len(train_items),
        "test_conversations": len(test_groups),
        "test_examples": len(test_items),
        "matched_oracle_actions": matches,
        "oracle_action_accuracy": matches / len(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 GBM 对人工 LongMemEval oracle 的动作选择")
    parser.add_argument("--data", required=True, help="LongMemEval 原始 JSON。")
    parser.add_argument("--labels", required=True, help="人工 oracle 标签 JSON。")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", help="可选 R2W v3 配置。")
    arguments = parser.parse_args()
    cfg = load_config(arguments.config) if arguments.config else default_config()
    result = run_oracle_validation(cfg, arguments.data, arguments.labels)
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
