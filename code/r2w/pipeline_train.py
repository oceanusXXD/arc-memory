"""R2W-GBM 离线训练：LoCoMo → 反事实测量 → GBM → 验证配置。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from .calibration import CostStatistics, select_policy, save_policy
from .cases import CaseMemory
from .config import R2WConfig, default_config, load_config, save_config
from .constructor import build_constructor
from .embedding import EmbeddingBackend
from .features import FeatureBuilder
from .llm import LLMClient
from .model import stage2_matrix, train_gbm_ensemble
from .query_generator import QueryGenerator
from .representations import STRUCT_ACTIONS
from .runtime_usage import provider_usage_payload
from .teacher import LLMReaderAdapter, build_effect_targets, run_policy_measure, run_swap_measure, run_uniform_measure
from .text import normalize_relative_time, word_count


def _normalise_conversation(raw: dict) -> dict:
    turns = [dict(turn) for turn in raw.get("turns", [])]
    if not turns:
        raise ValueError("每段训练对话至少要有一个 turn。")
    for turn in turns:
        required = {"dia_id", "speaker", "text", "timestamp"}
        if required - set(turn) or any(not isinstance(turn[name], str) or not turn[name].strip() for name in required):
            raise ValueError("turn 必须包含非空 dia_id、speaker、text 与 timestamp。")
        if not isinstance(turn.get("session"), int) or turn["session"] < 1:
            raise ValueError("turn.session 必须是从 1 开始的整数。")
        turn["source_text"] = turn.get("source_text", turn["text"])
        turn["text"] = normalize_relative_time(turn["source_text"], turn["timestamp"])
    dia_to_index = {turn["dia_id"]: index for index, turn in enumerate(turns)}
    if len(dia_to_index) != len(turns):
        raise ValueError("同一段对话中的 dia_id 必须唯一。")
    qa = []
    for query in raw.get("qa", []):
        if not isinstance(query.get("question"), str) or not query["question"].strip() or not isinstance(query.get("evidence"), list):
            raise ValueError("qa 必须包含 question 与 evidence 数组。")
        evidence = list(dict.fromkeys(value for value in query["evidence"] if value in dia_to_index))
        if query["evidence"] and not evidence:
            raise ValueError("一条 QA 的 evidence 全部无法映射到 turn。")
        qa.append({**query, "evidence": evidence, "answer": query.get("answer")})
    return {
        "turns": turns,
        "qa": qa,
        "speaker_a": str(raw.get("speaker_a", "")),
        "speaker_b": str(raw.get("speaker_b", "")),
        "dia_to_index": dia_to_index,
        "conversation_id": str(raw.get("conversation_id", "")),
    }


def _locomo_conversation(raw: dict) -> dict:
    conversation = raw.get("conversation", {})
    turns = []
    sessions = []
    for key, values in conversation.items():
        suffix = key.removeprefix("session_")
        if key.startswith("session_") and suffix.isdigit() and isinstance(values, list):
            sessions.append((int(suffix), key, values))
    for session, key, values in sorted(sessions):
        timestamp = conversation.get(f"{key}_date_time")
        if not isinstance(timestamp, str) or not timestamp.strip():
            raise ValueError(f"LoCoMo {key} 缺少 date_time。")
        turns.extend({**turn, "timestamp": timestamp, "session": session} for turn in values)
    qa = []
    for query in raw.get("qa", []):
        evidence = []
        for value in query.get("evidence", []):
            for part in value.replace(";", " ").split() if isinstance(value, str) else [value]:
                prefix, separator, suffix = part.partition(":") if isinstance(part, str) else ("", "", "")
                evidence.append(f"{prefix}:{int(suffix)}" if separator and suffix.isdigit() else part)
        qa.append({**query, "evidence": evidence})
    return _normalise_conversation({
        "turns": turns,
        "qa": qa,
        "speaker_a": conversation.get("speaker_a", ""),
        "speaker_b": conversation.get("speaker_b", ""),
        "conversation_id": raw.get("sample_id", ""),
    })


def _longmemeval_datetime(value: object, field: str) -> str:
    """将 LongMemEval 的唯一时间格式转为 R2W 的绝对日期锚点。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LongMemEval {field} 必须是非空时间字符串。")
    try:
        return datetime.strptime(value, "%Y/%m/%d (%a) %H:%M").strftime(
            "%Y-%m-%d %H:%M"
        )
    except ValueError as exc:
        raise ValueError(
            f"LongMemEval {field} 不是当前数据集使用的 YYYY/MM/DD (Day) HH:MM 格式。"
        ) from exc


def _longmemeval_conversation(raw: dict) -> dict:
    """将一个 LongMemEval QA record 转为一个 R2W conversation。

    LongMemEval 的可检索原子是 session，而不是 LoCoMo 的单条 ``dia_id``。
    因此每个 session 成为一个 R2W turn，内容拼接规则与 FutureMem 的
    ``LongMemEvalBenchmark.convert`` 完全一致；``answer_session_ids`` 直接
    映射为 evidence。问题类型只作为 QA 审计字段保留，不进入写时特征。
    """
    question_id = raw.get("question_id")
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError("LongMemEval question_id 必须是非空字符串。")
    sessions = raw.get("haystack_sessions")
    session_ids = raw.get("haystack_session_ids")
    dates = raw.get("haystack_dates")
    if not isinstance(sessions, list) or not isinstance(session_ids, list) or not isinstance(dates, list):
        raise ValueError("LongMemEval 必须包含 haystack_sessions/session_ids/dates 数组。")
    if not sessions or len(sessions) != len(session_ids) or len(sessions) != len(dates):
        raise ValueError("LongMemEval session、session_id 与 date 数量必须相同且非空。")
    turns = []
    for index, (session, session_id, date) in enumerate(
        zip(sessions, session_ids, dates, strict=True), start=1
    ):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("LongMemEval haystack_session_ids 必须是非空字符串。")
        if not isinstance(session, list) or not session:
            raise ValueError("LongMemEval 的每个 haystack session 必须是非空 turn 数组。")
        lines = []
        for dialog_turn in session:
            if not isinstance(dialog_turn, dict):
                raise ValueError("LongMemEval session turn 必须是 object。")
            role, content = dialog_turn.get("role"), dialog_turn.get("content")
            if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
                raise ValueError("LongMemEval session turn 必须包含非空 role 与 content。")
            lines.append(f"{role}: {content}")
        turns.append(
            {
                "dia_id": session_id,
                # 原始 schema 没有 session 级单一说话人；显式标识这个事实，
                # 而不从 QA 或内容猜测作者。
                "speaker": "session",
                "text": "\n".join(lines),
                "timestamp": _longmemeval_datetime(date, "haystack_dates"),
                "session": index,
            }
        )
    evidence = raw.get("answer_session_ids")
    if not isinstance(evidence, list):
        raise ValueError("LongMemEval answer_session_ids 必须是数组。")
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("LongMemEval question 必须是非空字符串。")
    return _normalise_conversation(
        {
            "turns": turns,
            "qa": [
                {
                    "id": question_id,
                    "question": question,
                    "answer": raw.get("answer"),
                    "evidence": evidence,
                    "timestamp": _longmemeval_datetime(
                        raw.get("question_date"), "question_date"
                    ),
                    # 仅供离线报告/官方评测对齐；FeatureBuilder 不读取它。
                    "category": raw.get("question_type"),
                }
            ],
            "speaker_a": "session_a",
            "speaker_b": "session_b",
            "conversation_id": question_id,
        }
    )


def load_training_data(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list):
        raise TypeError("训练文件根节点必须是对话数组。")
    conversations = []
    for value in values:
        if not isinstance(value, dict):
            raise TypeError("训练文件中的每一项必须是 object。")
        if "conversation" in value:
            conversations.append(_locomo_conversation(value))
        elif "haystack_sessions" in value:
            conversations.append(_longmemeval_conversation(value))
        else:
            conversations.append(_normalise_conversation(value))
    return conversations


def _split_conversations(conversations: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    if len(conversations) < 9:
        raise ValueError("R2W-GBM 需要至少九段对话：五折训练、验证和测试。")
    indexes = list(range(len(conversations)))
    random.Random(seed).shuffle(indexes)
    train_count = max(5, int(len(indexes) * 0.6))
    validation_count = max(1, int(len(indexes) * 0.2))
    if train_count + validation_count >= len(indexes):
        validation_count = len(indexes) - train_count - 1
    return (
        [conversations[index] for index in indexes[:train_count]],
        [conversations[index] for index in indexes[train_count : train_count + validation_count]],
        [conversations[index] for index in indexes[train_count + validation_count :]],
    )


def _draft_statistics(conversation: dict, uniform) -> np.ndarray:
    rows = np.zeros((len(conversation["turns"]), len(STRUCT_ACTIONS), 5), dtype=np.float32)
    for turn_index, turn in enumerate(conversation["turns"]):
        source_words = max(word_count(turn["text"]), 1)
        for action_index, action in enumerate(STRUCT_ACTIONS):
            representation = uniform.representations[action][turn_index]
            payload_words = word_count(representation.payload)
            key_words = sum(word_count(value) for value in representation.keys)
            rows[turn_index, action_index] = (
                float(payload_words),
                float(key_words),
                float(len(representation.keys)),
                float(payload_words / source_words),
                float(action.startswith("sum")),
            )
    return rows


def _row(conversation: dict, features: FeatureBuilder, embedder, constructor, qg, cfg: R2WConfig, reader) -> dict:
    feature_values = features.build(conversation, qg)
    uniform = run_uniform_measure(conversation, embedder, constructor, qg, cfg, reader)
    # 用户已指定实际构建全部九个非 raw 草稿，因此 L1 不再剪掉 L2 标签。
    swap = run_swap_measure(
        conversation,
        embedder,
        cfg,
        uniform,
        reader,
        candidate_mask=np.ones((len(conversation["turns"]), len(STRUCT_ACTIONS)), dtype=bool),
    )
    targets = build_effect_targets(conversation, uniform, cfg, swap)
    stage1 = np.concatenate((feature_values["memory"], feature_values["hot"]), axis=1)
    stage2_base = np.concatenate(
        (feature_values["memory"], feature_values["query"], feature_values["interaction"], feature_values["hot"]),
        axis=1,
    )
    return {
        "conversation": conversation,
        "stage1": stage1,
        "stage2": stage2_base,
        "draft": _draft_statistics(conversation, uniform),
        "targets": targets,
        "uniform": uniform,
    }


def _stack(rows: list[dict], field: str) -> np.ndarray:
    return np.vstack([row[field] for row in rows]).astype(np.float32)


def _target_stack(rows: list[dict], field: str) -> np.ndarray:
    return np.vstack([getattr(row["targets"], field) for row in rows])


def _groups(rows: list[dict]) -> np.ndarray:
    return np.concatenate([np.full(len(row["stage1"]), row["conversation"]["conversation_id"], dtype=object) for row in rows])


def train_pipeline(cfg: R2WConfig, conversations: list[dict], output_dir: str | Path, *, embedder=None, constructor=None, qg=None, reader=None, ner=None, llm: LLMClient | None = None) -> dict:
    train_conversations, validation_conversations, test_conversations = _split_conversations(conversations, cfg.random_seed)
    embedder = embedder or EmbeddingBackend(cfg)
    qg = qg or QueryGenerator(cfg)
    llm = llm or LLMClient(cfg)
    constructor = constructor or build_constructor(cfg, llm)
    reader = reader or LLMReaderAdapter(llm)
    feature_builder = FeatureBuilder(cfg, embedder, ner=ner).fit(train_conversations, qg)
    train_rows = [_row(value, feature_builder, embedder, constructor, qg, cfg, reader) for value in train_conversations]
    validation_rows = [_row(value, feature_builder, embedder, constructor, qg, cfg, reader) for value in validation_conversations]
    test_rows = [_row(value, feature_builder, embedder, constructor, qg, cfg, reader) for value in test_conversations]

    train_effects = _target_stack(train_rows, "effects").astype(np.float32)
    train_measured = _target_stack(train_rows, "measured").astype(bool)
    # 第一关的收益包络只允许看到实际已测的 arm；没有标签的 arm 不能以
    # 填充值参与 max。
    admission_target = np.where(train_measured, train_effects, -np.inf).max(axis=1)
    if not np.isfinite(admission_target).all():
        raise RuntimeError("存在未测量全部动作的 memory，无法训练第一关收益包络。")
    train_model = train_gbm_ensemble(
        cfg,
        _stack(train_rows, "stage1"),
        admission_target,
        _stack(train_rows, "stage2"),
        _stack(train_rows, "draft"),
        train_effects,
        _target_stack(train_rows, "hit_rates"),
        train_measured,
        _target_stack(train_rows, "standard_errors"),
        _groups(train_rows),
    )
    validation_stage1 = _stack(validation_rows, "stage1")
    validation_stage2 = _stack(validation_rows, "stage2")
    validation_draft = _stack(validation_rows, "draft")
    admission, admission_sigma = train_model.predict_admission(validation_stage1)
    effect, hit, effect_sigma = train_model.predict_arms(validation_stage2, validation_draft)
    costs_train = _target_stack(train_rows, "costs")
    policy = select_policy(
        cfg,
        CostStatistics.fit(costs_train, _target_stack(train_rows, "hit_rates")),
        admission,
        admission_sigma,
        effect,
        hit,
        effect_sigma,
        _target_stack(validation_rows, "costs"),
        _target_stack(validation_rows, "effects"),
        _target_stack(validation_rows, "hit_rates"),
        _target_stack(validation_rows, "measured").astype(bool),
    )
    cases = CaseMemory(cfg)
    for row in train_rows:
        for index, turn in enumerate(row["conversation"]["turns"]):
            cases.add(row["stage1"][index], row["targets"].effects[index], turn["text"])

    # L3 只作一次策略背景诊断，不进入训练或 policy 选择。
    l3 = []
    offset = 0
    for row in validation_rows:
        count = len(row["stage1"])
        actions = []
        for index in range(count):
            decision = policy.decide(admission[offset + index], admission_sigma[offset + index], effect[offset + index], hit[offset + index], effect_sigma[offset + index], row["targets"].costs[index])
            actions.append("none" if decision.action_index == "none" else STRUCT_ACTIONS[int(decision.action_index)])
        l3.append(run_policy_measure(row["conversation"], embedder, cfg, row["uniform"], actions, reader))
        offset += count

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, ensure_ascii=False, indent=2)
    joblib.dump(train_model, destination / "model.joblib")
    feature_builder.save(destination / "features.joblib")
    with (destination / "case_memory.json").open("w", encoding="utf-8") as handle:
        json.dump({"config_hash": cfg.hash(), "cases": cases.to_list()}, handle, ensure_ascii=False)
    with (destination / "policy.json").open("w", encoding="utf-8") as handle:
        json.dump(save_policy(policy), handle, ensure_ascii=False, indent=2)
    with (destination / "provider_usage.json").open("w", encoding="utf-8") as handle:
        json.dump(provider_usage_payload(llm=llm, embedding=embedder), handle, ensure_ascii=False, indent=2)
    np.savez_compressed(
        destination / "measurement_data.npz",
        config_hash=np.asarray(cfg.hash()),
        validation_effects=_target_stack(validation_rows, "effects"),
        validation_hits=_target_stack(validation_rows, "hit_rates"),
        validation_costs=_target_stack(validation_rows, "costs"),
        validation_measured=_target_stack(validation_rows, "measured"),
        validation_stage1=validation_stage1,
        validation_stage2=validation_stage2,
        validation_draft=validation_draft,
        validation_delta_f=_target_stack(validation_rows, "form_effects"),
        validation_psi=_target_stack(validation_rows, "psi"),
        l3_effects=np.vstack([item.effects for item in l3]),
        l3_measured=np.vstack([item.measured for item in l3]),
        l3_actions=np.concatenate(
            [np.asarray(item.policy_actions, dtype="U16") for item in l3]
        ),
    )
    splits = {
        "train": [value["conversation_id"] for value in train_conversations],
        "validation": [value["conversation_id"] for value in validation_conversations],
        "test": [value["conversation_id"] for value in test_conversations],
    }
    with (destination / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "artifact_version": cfg.artifact_version,
            "config_hash": cfg.hash(),
            "model": "model.joblib",
            "cost_unit": "word_count",
            "provider_usage": "provider_usage.json",
            "measurement": "raw-background p-vs-none endpoint QA",
            "l3": "single policy-background diagnostic",
            "splits": splits,
            "split_sha256": hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest(),
        }, handle, ensure_ascii=False, indent=2)
    return {
        "config_hash": cfg.hash(),
        "train_conversations": len(train_conversations),
        "validation_conversations": len(validation_conversations),
        "held_out_conversations": len(test_conversations),
        "train_turns": int(sum(len(row["stage1"]) for row in train_rows)),
        "output_dir": str(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 R2W-GBM")
    parser.add_argument("--data")
    parser.add_argument("--out")
    parser.add_argument("--config", help="完整 R2WConfig JSON；建议真实运行时显式提供。")
    parser.add_argument("--write-default-config", help="写出默认完整配置后退出。")
    parser.add_argument("--lambda-w", type=float)
    parser.add_argument("--lambda-r", type=float)
    parser.add_argument("--lambda-s", type=float)
    arguments = parser.parse_args()
    if arguments.write_default_config:
        save_config(default_config(), arguments.write_default_config)
        return
    if not arguments.data or not arguments.out:
        parser.error("训练必须提供 --data 和 --out。")
    cfg = load_config(arguments.config) if arguments.config else default_config()
    overrides = {
        name: value
        for name, value in {
            "lambda_w": arguments.lambda_w,
            "lambda_r": arguments.lambda_r,
            "lambda_s": arguments.lambda_s,
        }.items()
        if value is not None
    }
    if overrides:
        cfg = R2WConfig(**(asdict(cfg) | overrides))
    print(json.dumps(train_pipeline(cfg, load_training_data(arguments.data), arguments.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
