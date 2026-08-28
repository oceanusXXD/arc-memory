"""R2W 的零参数反事实测量。

L1 只产出粗筛诊断；L2 的主标签严格是全 raw 背景下 ``p vs none``
的端到端 reader 结局差分。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import R2WConfig
from .costs import representation_costs
from .llm import LLMClient, LLMProtocolError
from .llm_prompts import READER
from .representations import Repr, STRUCT_ACTIONS, attach_hypothetical_queries, build_representation
from .retrieval import UnifiedIndex, representation_index_units
from .text import tokenize


def token_f1(prediction: object, gold: object) -> float:
    if gold is None:
        return float("nan")
    predicted_tokens, gold_tokens = tokenize(str(prediction)), tokenize(str(gold))
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted_tokens), overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class ReaderAnswer:
    answer: str
    citations: tuple[int, ...]


class ReaderAdapter(Protocol):
    def answer(self, question: str, payloads: list[str], current_date: str) -> ReaderAnswer: ...


class LLMReaderAdapter:
    """冻结 JSON reader；仅用于离线反事实结局。"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def answer(self, question: str, payloads: list[str], current_date: str) -> ReaderAnswer:
        context = "\n".join(f"[{index}] {text}" for index, text in enumerate(payloads))
        value = self.llm.json(
            READER.format(date=current_date, question=question, context=context),
            purpose="reader",
        )
        answer, citations = value.get("answer"), value.get("cites", [])
        if not isinstance(answer, (str, int, float)) or not isinstance(citations, list):
            raise LLMProtocolError("reader 必须返回 answer 与 cites 数组。")
        valid = tuple(dict.fromkeys(index for index in citations if isinstance(index, int) and 0 <= index < len(payloads)))
        return ReaderAnswer(str(answer), valid)


@dataclass
class UniformMeasure:
    representations: dict[str, dict[int, Repr]]
    coverage: dict[str, np.ndarray]
    ranks: dict[str, np.ndarray]
    qa: dict[str, np.ndarray]


@dataclass
class SwapMeasure:
    delta_e: np.ndarray
    delta_f: np.ndarray
    psi: np.ndarray
    hit_rate: np.ndarray
    costs: np.ndarray
    standard_error: np.ndarray
    n_rel: np.ndarray
    n_unrel: np.ndarray
    measured: np.ndarray
    relevant_delta: np.ndarray
    unrelated_delta: np.ndarray


@dataclass
class EffectTargets:
    effects: np.ndarray
    form_effects: np.ndarray
    hit_rates: np.ndarray
    costs: np.ndarray
    standard_errors: np.ndarray
    measured: np.ndarray
    psi: np.ndarray
    n_rel: np.ndarray
    n_unrel: np.ndarray


@dataclass
class PolicyMeasure:
    effects: np.ndarray
    measured: np.ndarray
    policy_actions: tuple[str, ...]


def _catalog(conversation: dict, constructor, qg, cfg: R2WConfig) -> dict[str, dict[int, Repr]]:
    turns = conversation["turns"]
    attach_hypothetical_queries(turns, qg)
    return {
        action: {
            index: build_representation(turn, action, constructor, cfg)
            for index, turn in enumerate(turns)
        }
        for action in STRUCT_ACTIONS
    }


def _index_for_action(turns: list[dict], representations: dict[int, Repr], embedder, cfg: R2WConfig) -> UnifiedIndex:
    units = [
        unit
        for index, turn in enumerate(turns)
        for unit in representation_index_units(index, representations[index], int(turn["session"]))
    ]
    return UnifiedIndex(units, embedder, cfg)


def _snapshot(index: UnifiedIndex, conversation: dict, query_index: int, target: int, cfg: R2WConfig, reader: ReaderAdapter | None) -> tuple[float, int, float]:
    query = conversation["qa"][query_index]
    results = index.retrieve_turns(query["question"], cfg.teacher_rank_limit)
    rank = next((position for position, value in enumerate(results, start=1) if value.turn_index == target), 0)
    evidence = [conversation["dia_to_index"][value] for value in query["evidence"]]
    coverage = sum(value in {result.turn_index for result in results[: cfg.retrieval_k]} for value in evidence) / max(len(evidence), 1)
    if reader is None:
        return float("nan"), rank, float(coverage)
    top = results[: cfg.retrieval_k]
    answer = reader.answer(
        query["question"], index.reader_payloads(top), conversation["turns"][-1]["timestamp"]
    )
    return token_f1(answer.answer, query.get("answer")), rank, float(coverage)


def run_uniform_measure(conversation: dict, embedder, constructor, qg, cfg: R2WConfig, reader: ReaderAdapter | None = None) -> UniformMeasure:
    """L1：全臂统一重放，仅保留粗筛/检索诊断，不进入训练标签。"""
    catalog = _catalog(conversation, constructor, qg, cfg)
    count, query_count = len(conversation["turns"]), len(conversation["qa"])
    coverage: dict[str, np.ndarray] = {}
    ranks: dict[str, np.ndarray] = {}
    qa: dict[str, np.ndarray] = {}
    for action, representations in catalog.items():
        index = _index_for_action(conversation["turns"], representations, embedder, cfg)
        action_ranks = np.zeros((query_count, count), dtype=np.int32)
        action_coverage = np.zeros(query_count, dtype=np.float32)
        action_qa = np.full(query_count, np.nan, dtype=np.float32)
        for q in range(query_count):
            query = conversation["qa"][q]
            results = index.retrieve_turns(query["question"], cfg.teacher_rank_limit)
            for rank, result in enumerate(results, start=1):
                action_ranks[q, int(result.turn_index)] = rank
            evidence = [conversation["dia_to_index"][value] for value in query["evidence"]]
            action_coverage[q] = sum(
                value in {result.turn_index for result in results[: cfg.retrieval_k]}
                for value in evidence
            ) / max(len(evidence), 1)
            if reader is not None:
                answer = reader.answer(query["question"], index.reader_payloads(results[: cfg.retrieval_k]), conversation["turns"][-1]["timestamp"])
                action_qa[q] = token_f1(answer.answer, query.get("answer"))
        coverage[action], ranks[action], qa[action] = action_coverage, action_ranks, action_qa
    return UniformMeasure(catalog, coverage, ranks, qa)


def _cost_components(representation: Repr, action: str) -> tuple[float, float, float]:
    if representation.arm != action:
        raise ValueError("表示 arm 与成本动作不一致。")
    return tuple(float(value) for value in representation_costs(representation))


def run_swap_measure(conversation: dict, embedder, cfg: R2WConfig, uniform: UniformMeasure, reader: ReaderAdapter | None = None, candidate_mask: np.ndarray | None = None) -> SwapMeasure:
    """L2：在全 raw 背景逐条测量 p 相对 none 的端到端效应。"""
    turns, queries = conversation["turns"], conversation["qa"]
    turns_count, actions_count, query_count = len(turns), len(STRUCT_ACTIONS), len(queries)
    if reader is None:
        raise ValueError("L2 主标签必须提供冻结 reader。")
    if candidate_mask is not None and candidate_mask.shape != (turns_count, actions_count):
        raise ValueError("candidate_mask 必须按 [turn, arm] 对齐。")
    delta_e = np.zeros((turns_count, actions_count), dtype=np.float32)
    psi = np.zeros_like(delta_e)
    hit_rate = np.zeros_like(delta_e)
    costs = np.zeros((turns_count, actions_count, 3), dtype=np.float32)
    standard_error = np.zeros_like(delta_e)
    n_rel = np.zeros_like(delta_e, dtype=np.int32)
    n_unrel = np.zeros_like(delta_e, dtype=np.int32)
    measured = np.zeros_like(delta_e, dtype=bool)
    relevant_delta = np.zeros_like(delta_e)
    unrelated_delta = np.zeros_like(delta_e)
    raw_index = _index_for_action(turns, uniform.representations["raw"], embedder, cfg)

    for target, turn in enumerate(turns):
        relevant = [q for q, query in enumerate(queries) if turn["dia_id"] in query["evidence"]]
        unrelated = [q for q in range(query_count) if q not in relevant]
        if len(unrelated) > cfg.interference_query_count:
            rng = np.random.default_rng(cfg.random_seed + target)
            unrelated = sorted(rng.choice(unrelated, cfg.interference_query_count, replace=False).tolist())
        old_raw = raw_index.remove_turn(target)
        none_scores = {
            q: _snapshot(raw_index, conversation, q, target, cfg, reader)[0]
            for q in [*relevant, *unrelated]
        }
        for action_index, action in enumerate(STRUCT_ACTIONS):
            if candidate_mask is not None and not candidate_mask[target, action_index]:
                continue
            representation = uniform.representations[action][target]
            raw_index.add_representation(target, representation, int(turn["session"]))
            try:
                rel_values, unrel_values, hits = [], [], []
                for q in range(query_count):
                    qa, rank, _ = _snapshot(raw_index, conversation, q, target, cfg, reader)
                    hits.append(float(0 < rank <= cfg.retrieval_k))
                    if q in none_scores:
                        difference = qa - none_scores[q]
                        if q in relevant:
                            rel_values.append(difference)
                        else:
                            unrel_values.append(difference)
                n_rel[target, action_index] = len(rel_values)
                n_unrel[target, action_index] = len(unrel_values)
                relevant_delta[target, action_index] = float(np.mean(rel_values)) if rel_values else 0.0
                unrelated_delta[target, action_index] = float(np.mean(unrel_values)) if unrel_values else 0.0
                hit_rate[target, action_index] = float(np.mean(hits)) if hits else 0.0
                costs[target, action_index] = _cost_components(representation, action)
                pi = len(relevant) / max(query_count, 1)
                rel_var = float(np.var(rel_values, ddof=1) / len(rel_values)) if len(rel_values) > 1 else 0.0
                unrel_var = float(np.var(unrel_values, ddof=1) / len(unrel_values)) if len(unrel_values) > 1 else 0.0
                standard_error[target, action_index] = float(np.sqrt(pi**2 * rel_var + (1.0 - pi) ** 2 * unrel_var))
                measured[target, action_index] = True
            finally:
                raw_index.remove_turn(target)
        raw_index.restore_turn(target, old_raw)

    # LoCoMo 没有 memory content type，所有样本属于同一个、不会泄漏的分组。
    global_means = np.zeros(actions_count, dtype=np.float32)
    for action_index in range(actions_count):
        valid = measured[:, action_index] & (n_rel[:, action_index] > 0)
        if valid.any():
            global_means[action_index] = float(relevant_delta[valid, action_index].mean())
    for target, turn in enumerate(turns):
        relevant_count = int(np.max(n_rel[target]))
        pi = relevant_count / max(query_count, 1)
        alpha = relevant_count / (relevant_count + cfg.empirical_bayes_n0) if relevant_count else 0.0
        for action_index in range(actions_count):
            if not measured[target, action_index]:
                continue
            shrunk = alpha * relevant_delta[target, action_index] + (1.0 - alpha) * global_means[action_index]
            psi[target, action_index] = -(1.0 - pi) * unrelated_delta[target, action_index]
            delta_e[target, action_index] = pi * shrunk - psi[target, action_index]
    raw = STRUCT_ACTIONS.index("raw")
    delta_f = delta_e - delta_e[:, raw : raw + 1]
    return SwapMeasure(delta_e, delta_f, psi, hit_rate, costs, standard_error, n_rel, n_unrel, measured, relevant_delta, unrelated_delta)


def build_effect_targets(conversation: dict, uniform: UniformMeasure, cfg: R2WConfig, swap: SwapMeasure) -> EffectTargets:
    return EffectTargets(swap.delta_e, swap.delta_f, swap.hit_rate, swap.costs, swap.standard_error, swap.measured, swap.psi, swap.n_rel, swap.n_unrel)


def run_policy_measure(conversation: dict, embedder, cfg: R2WConfig, uniform: UniformMeasure, policy_actions: list[str] | tuple[str, ...], reader: ReaderAdapter | None = None) -> PolicyMeasure:
    """一次性 L3 诊断：只报告策略背景，不回流为训练/校准标签。"""
    if len(policy_actions) != len(conversation["turns"]):
        raise ValueError("policy_actions 必须逐 turn 对齐。")
    if reader is None:
        raise ValueError("L3 审计需要 reader。")
    units = []
    for index, (turn, action) in enumerate(zip(conversation["turns"], policy_actions, strict=True)):
        if action != "none":
            units.extend(representation_index_units(index, uniform.representations[action][index], int(turn["session"])))
    index = UnifiedIndex(units, embedder, cfg)
    # 用 raw 背景 L2 标签的形状报告政策背景可测格；不把它作为主训练目标。
    effects = np.zeros((len(conversation["turns"]), len(STRUCT_ACTIONS)), dtype=np.float32)
    measured = np.zeros_like(effects, dtype=bool)
    for target, turn in enumerate(conversation["turns"]):
        relevant = [q for q, query in enumerate(conversation["qa"]) if turn["dia_id"] in query["evidence"]]
        original = index.remove_turn(target)
        baseline = [_snapshot(index, conversation, q, target, cfg, reader)[0] for q in relevant]
        for action_index, action in enumerate(STRUCT_ACTIONS):
            index.add_representation(target, uniform.representations[action][target], int(turn["session"]))
            values = [_snapshot(index, conversation, q, target, cfg, reader)[0] for q in relevant]
            effects[target, action_index] = float(np.mean(np.asarray(values) - np.asarray(baseline))) if values else 0.0
            measured[target, action_index] = True
            index.remove_turn(target)
        index.restore_turn(target, original)
    return PolicyMeasure(effects, measured, tuple(policy_actions))
