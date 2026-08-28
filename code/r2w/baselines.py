"""R2W 的可复现实验基线。

所有检索基线共用 ``UnifiedIndex``、embedding、top-k 和 reader；唯一变化是
每个 turn 的固定表示。这样 ``raw`` 就是与 R2W 同条件的 Naive RAG，而不是
反事实测量里移除目标 memory 的 ``none``。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import R2WConfig
from .costs import COST_UNIT, representation_costs
from .representations import STRUCT_ACTIONS, attach_hypothetical_queries, build_representation
from .retrieval import UnifiedIndex, representation_index_units
from .teacher import ReaderAdapter, token_f1
from .text import word_count


@dataclass(frozen=True)
class BaselineResult:
    name: str
    fixed_action: str | None
    conversations: int
    qa_count: int
    mean_token_f1: float
    mean_evidence_recall_at_k: float
    mean_retrieved_turns: float
    mean_reader_context_words: float
    total_write_words: float
    total_index_words: float
    cost_unit: str = COST_UNIT

    def to_dict(self) -> dict:
        return asdict(self)


def _mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if len(array) else float("nan")


def _fixed_representations(conversation: dict, action: str, constructor, qg, cfg: R2WConfig):
    if action not in STRUCT_ACTIONS:
        raise ValueError(f"未知固定表示基线 {action!r}。")
    if action.endswith("+hq"):
        if qg is None:
            raise ValueError("HQ 基线需要 QueryGenerator。")
        attach_hypothetical_queries(conversation["turns"], qg)
    if action.endswith(("+kv", "+event", "+graph")) and constructor is None:
        raise ValueError(f"{action} 基线需要结构化构建器。")
    return [build_representation(turn, action, constructor, cfg) for turn in conversation["turns"]]


def evaluate_fixed_baseline(
    conversations: list[dict],
    action: str,
    embedder,
    cfg: R2WConfig,
    reader: ReaderAdapter,
    *,
    constructor=None,
    qg=None,
) -> BaselineResult:
    """评估一个固定写入表示；``raw`` 是正式 Naive RAG 基线。"""
    f1, recall, retrieved_counts, reader_words = [], [], [], []
    write_words = index_words = 0.0
    qa_count = 0
    for conversation in conversations:
        representations = _fixed_representations(conversation, action, constructor, qg, cfg)
        costs = np.vstack([representation_costs(value) for value in representations])
        write_words += float(costs[:, 0].sum())
        index_words += float(costs[:, 1].sum())
        units = [
            unit
            for turn_index, (turn, representation) in enumerate(
                zip(conversation["turns"], representations, strict=True)
            )
            for unit in representation_index_units(turn_index, representation, int(turn["session"]))
        ]
        index = UnifiedIndex(units, embedder, cfg)
        for query in conversation["qa"]:
            results = index.retrieve_turns(query["question"], cfg.retrieval_k)
            payloads = index.reader_payloads(results)
            answer = reader.answer(
                query["question"], payloads, conversation["turns"][-1]["timestamp"]
            )
            evidence = [conversation["dia_to_index"][value] for value in query["evidence"]]
            returned = {value.turn_index for value in results}
            f1.append(token_f1(answer.answer, query.get("answer")))
            recall.append(sum(value in returned for value in evidence) / max(len(evidence), 1))
            retrieved_counts.append(float(len(results)))
            reader_words.append(float(sum(word_count(value) for value in payloads)))
            qa_count += 1
    return BaselineResult(
        name="always_raw_naive_rag" if action == "raw" else f"always_{action}",
        fixed_action=action,
        conversations=len(conversations),
        qa_count=qa_count,
        mean_token_f1=_mean(f1),
        mean_evidence_recall_at_k=_mean(recall),
        mean_retrieved_turns=_mean(retrieved_counts),
        mean_reader_context_words=_mean(reader_words),
        total_write_words=write_words,
        total_index_words=index_words,
    )


def evaluate_none_baseline(conversations: list[dict], reader: ReaderAdapter) -> BaselineResult:
    """无热记忆诊断线；它不是 Naive RAG。"""
    f1, reader_words = [], []
    qa_count = 0
    for conversation in conversations:
        for query in conversation["qa"]:
            answer = reader.answer(query["question"], [], conversation["turns"][-1]["timestamp"])
            f1.append(token_f1(answer.answer, query.get("answer")))
            reader_words.append(0.0)
            qa_count += 1
    return BaselineResult(
        name="no_hot_memory",
        fixed_action=None,
        conversations=len(conversations),
        qa_count=qa_count,
        mean_token_f1=_mean(f1),
        mean_evidence_recall_at_k=0.0,
        mean_retrieved_turns=0.0,
        mean_reader_context_words=_mean(reader_words),
        total_write_words=0.0,
        total_index_words=0.0,
    )


def evaluate_full_context_baseline(conversations: list[dict], cfg: R2WConfig, reader: ReaderAdapter) -> BaselineResult:
    """完整历史直接给 reader 的长上下文参考线，不经过检索。"""
    f1, reader_words, retrieved_counts = [], [], []
    qa_count = 0
    for conversation in conversations:
        payloads = [build_representation(turn, "raw", None, cfg).payload for turn in conversation["turns"]]
        context_words = float(sum(word_count(value) for value in payloads))
        for query in conversation["qa"]:
            answer = reader.answer(query["question"], payloads, conversation["turns"][-1]["timestamp"])
            f1.append(token_f1(answer.answer, query.get("answer")))
            reader_words.append(context_words)
            retrieved_counts.append(float(len(payloads)))
            qa_count += 1
    return BaselineResult(
        name="full_history_context",
        fixed_action=None,
        conversations=len(conversations),
        qa_count=qa_count,
        mean_token_f1=_mean(f1),
        mean_evidence_recall_at_k=1.0,
        mean_retrieved_turns=_mean(retrieved_counts),
        mean_reader_context_words=_mean(reader_words),
        total_write_words=0.0,
        total_index_words=0.0,
    )
