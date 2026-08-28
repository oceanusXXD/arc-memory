"""R2W 的十臂表示协议。

`build_representation` 是测量与在线执行共用的唯一构建入口。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import R2WConfig, default_config
from .constructor import StructuredConstructor
from .llm import LLMProtocolError
from .llm_prompts import EVENT, GRAPH, KV
from .query_generator import GeneratedQueries, QueryGenerator
from .text import QUESTION_WORDS, TOKEN_RE, date_count, relative_time_count, tokenize, word_count

COMPRESSION_AXIS = ("keep", "compress")
KEY_AXIS = ("none", "kv", "event", "graph", "hq")
STRUCT_ACTIONS = (
    "raw",
    "raw+kv",
    "raw+event",
    "raw+graph",
    "raw+hq",
    "sum",
    "sum+kv",
    "sum+event",
    "sum+graph",
    "sum+hq",
)
CERTIFICATE_ACTIONS = frozenset(
    {
        "raw+kv",
        "raw+event",
        "raw+graph",
        "sum",
        "sum+kv",
        "sum+event",
        "sum+graph",
        "sum+hq",
    }
)
ACTION_FACTORS = {
    action: (0 if index < len(KEY_AXIS) else 1, index % len(KEY_AXIS))
    for index, action in enumerate(STRUCT_ACTIONS)
}


@dataclass(frozen=True)
class MetaBlock:
    """所有 v2 存储动作共享的 M(t)。"""

    speaker: str
    timestamp: str
    session: int
    turn_id: str

    @classmethod
    def from_turn(cls, turn: dict) -> "MetaBlock":
        session = turn.get("session")
        if not isinstance(session, int) or session < 1:
            raise ValueError("turn.session 必须是从 1 开始的整数。")
        timestamp = _string(turn.get("timestamp"), "turn.timestamp")
        if date_count(timestamp) == 0:
            raise ValueError("turn.timestamp 必须包含可审计的绝对日期锚点。")
        return cls(
            speaker=_string(turn.get("speaker"), "turn.speaker"),
            timestamp=timestamp,
            session=session,
            turn_id=_string(turn.get("dia_id"), "turn.dia_id"),
        )

    def render(self) -> str:
        values = (self.speaker, self.timestamp, str(self.session), self.turn_id)
        speaker, timestamp, session, turn_id = (
            re.sub(r"\s+", " ", value).replace("|", "/").strip()
            for value in values
        )
        return (
            f"[speaker={speaker} | time={timestamp} | "
            f"session={session} | turn={turn_id}]"
        )


@dataclass(frozen=True)
class Repr:
    """协议 v2：payload 供 reader 阅读，keys 只参与检索。"""

    arm: str
    payload: str
    keys: tuple[str, ...] = ()
    key_meta: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    protocol_version: str = "r2w-repr-v3"

    def __post_init__(self) -> None:
        if self.arm not in STRUCT_ACTIONS:
            raise ValueError(f"未知表示动作: {self.arm}")
        if not self.payload.strip():
            raise ValueError("表示 payload 不能为空。")
        if len(self.keys) != len(self.key_meta):
            raise ValueError("keys 与 key_meta 必须一一对齐。")
        if any(not key.strip() for key in self.keys):
            raise ValueError("表示 key 不可为空。")

    @property
    def units(self) -> tuple[str, ...]:
        return (self.payload, *self.keys)


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LLMProtocolError(f"LLM JSON 缺少非空字符串字段 {field!r}。")
    return value.strip()


def _clip_words(text: str, limit: int) -> str:
    if limit < 1:
        raise ValueError("词数上限必须大于 0。")
    if word_count(text) <= limit:
        clipped = text.strip()
    else:
        matches = list(TOKEN_RE.finditer(text))
        clipped = text[: matches[limit - 1].end()].strip()
    if not clipped:
        raise LLMProtocolError("LLM 表示在长度约束后为空。")
    return clipped


def compress_summary(text: str, budget_words: int) -> str:
    """按局部内容显著性选择完整原句；确定性、抽取式且严格不超预算。"""
    source = _string(text, "turn.text")
    if budget_words < 1:
        return ""
    sentences = [
        match.group(0).strip()
        for match in re.finditer(r".+?(?:[.!?。！？]+|$)", source, flags=re.DOTALL)
        if match.group(0).strip()
    ]
    if not sentences:
        return ""
    token_sets = [set(tokenize(sentence)) for sentence in sentences]
    document_frequency: dict[str, int] = {}
    for tokens in token_sets:
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    count = len(sentences)
    scores = []
    for index, tokens in enumerate(token_sets):
        salience = sum(
            math.log(1.0 + (count - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5))
            for token in tokens
        )
        scores.append(salience + 0.01 * (1.0 - index / count))
    keep: list[int] = []
    used = 0
    for index in sorted(range(count), key=lambda value: (-scores[value], value)):
        size = word_count(sentences[index])
        if size and used + size <= budget_words:
            keep.append(index)
            used += size
    result = " ".join(sentences[index] for index in sorted(keep))
    if word_count(result) > budget_words:
        raise AssertionError("确定性摘要超出词预算。")
    return result
def raw_unit(turn: dict, cfg: R2WConfig | None = None) -> str:
    """返回带元数据契约的 raw reader payload。"""
    cfg = cfg or default_config()
    return f"{MetaBlock.from_turn(turn).render()} {_string(turn.get('text'), 'turn.text')}"


def _summary_body(turn: dict, constructor: StructuredConstructor | None, cfg: R2WConfig) -> str:
    cache_key = f"_r2w_summary_{cfg.repr_protocol_version}"
    if cache_key not in turn:
        text = _string(turn.get("text"), "turn.text")
        limit = max(1, math.floor(word_count(text) * cfg.summary_ratio))
        turn[cache_key] = compress_summary(text, limit)
    return str(turn[cache_key])


def _kv_keys(turn: dict, constructor: StructuredConstructor, cfg: R2WConfig) -> tuple[list[str], list[dict[str, Any]]]:
    cache_key = f"_r2w_kv_{cfg.repr_protocol_version}"
    if cache_key not in turn:
        payload = constructor.json(
            KV.format(max_keys=4, max_value_words=cfg.kv_value_words, speaker=turn["speaker"], text=turn["text"])
        )
        items = payload.get("items")
        if not isinstance(items, list) or len(items) > 4:
            raise LLMProtocolError("kv 响应必须包含不超过四项的 items。")
        parsed = []
        for item in items:
            if not isinstance(item, dict):
                raise LLMProtocolError("kv.items 的每项必须是 object。")
            parsed.append(
                (
                    f"{_string(item.get('key'), 'kv.key')} : "
                    f"{_clip_words(_string(item.get('value'), 'kv.value'), cfg.kv_value_words)}",
                    {"type": "kv", "provenance": turn["dia_id"]},
                )
            )
        turn[cache_key] = parsed
    rows = list(turn[cache_key])
    return [row[0] for row in rows], [dict(row[1]) for row in rows]


def _event_keys(turn: dict, constructor: StructuredConstructor, cfg: R2WConfig) -> tuple[list[str], list[dict[str, Any]]]:
    cache_key = f"_r2w_event_{cfg.repr_protocol_version}"
    if cache_key not in turn:
        fallback = _string(turn.get("timestamp"), "turn.timestamp")
        payload = constructor.json(
            EVENT.format(
                max_items=cfg.event_max_items,
                max_words=cfg.event_item_words,
                speaker=turn["speaker"],
                fallback_time=fallback,
                text=turn["text"],
            )
        )
        items = payload.get("items")
        if items is None and "time" in payload and "event" in payload:
            items = [{"time": payload["time"], "event": payload["event"]}]
        if not isinstance(items, list) or len(items) > cfg.event_max_items:
            raise LLMProtocolError("event 响应必须包含契约数量内的 items。")
        parsed = []
        for item in items:
            if not isinstance(item, dict):
                raise LLMProtocolError("event.items 的每项必须是 object。")
            normalised_time = _string(item.get("time"), "event.time")
            if relative_time_count(normalised_time) or date_count(normalised_time) == 0:
                raise LLMProtocolError("event.time 必须包含明确绝对时间锚点。")
            event = _clip_words(_string(item.get("event"), "event.event"), cfg.event_item_words)
            parsed.append(
                (
                    f"[{normalised_time}] {event}",
                    {
                        "type": "event",
                        "provenance": turn["dia_id"],
                        "valid_from": normalised_time,
                        "valid_to": None,
                    },
                )
            )
        turn[cache_key] = parsed
    rows = list(turn[cache_key])
    return [row[0] for row in rows], [dict(row[1]) for row in rows]


def _graph_keys(turn: dict, constructor: StructuredConstructor, cfg: R2WConfig) -> tuple[list[str], list[dict[str, Any]]]:
    cache_key = f"_r2w_graph_{cfg.repr_protocol_version}"
    if cache_key not in turn:
        payload = constructor.json(
            GRAPH.format(max_items=cfg.graph_max_items, speaker=turn["speaker"], text=turn["text"])
        )
        relations = payload.get("relations")
        if not isinstance(relations, list) or len(relations) > cfg.graph_max_items:
            raise LLMProtocolError("graph 响应必须包含契约数量内的 relations。")
        parsed = []
        for relation in relations:
            if isinstance(relation, list) and len(relation) == 3:
                relation = {"subject": relation[0], "relation": relation[1], "object": relation[2]}
            if not isinstance(relation, dict):
                raise LLMProtocolError("graph.relations 的每项必须是 object。")
            subject = _string(relation.get("subject"), "graph.subject")
            predicate = _string(relation.get("relation"), "graph.relation")
            obj = _string(relation.get("object"), "graph.object")
            value = relation.get("value")
            if value is not None and not isinstance(value, str):
                raise LLMProtocolError("graph.value 必须是字符串或 null。")
            rendered_value = value.strip() if isinstance(value, str) and value.strip() else obj
            text = _clip_words(f"{subject} {predicate} {obj} value={rendered_value}", cfg.graph_item_words)
            parsed.append(
                (
                    text,
                    {
                        "type": "graph",
                        "provenance": turn["dia_id"],
                        "valid_from": relation.get("valid_from") or turn["timestamp"],
                        "valid_to": relation.get("valid_to"),
                        "value": rendered_value,
                    },
                )
            )
        turn[cache_key] = parsed
    rows = list(turn[cache_key])
    return [row[0] for row in rows], [dict(row[1]) for row in rows]


def attach_hypothetical_queries(turns: list[dict], qg: QueryGenerator) -> None:
    for index, turn in enumerate(turns):
        if "_r2w_hq" not in turn:
            previous = turns[index - 1]["text"] if index else ""
            turn["_r2w_hq"] = qg.generate(previous, turn["text"])


def hypothetical_queries(turn: dict) -> GeneratedQueries:
    value = turn.get("_r2w_hq")
    if not isinstance(value, GeneratedQueries):
        raise TypeError("hq 表示和 Q/I 特征前必须先生成确定性假设 query。")
    return value


def answerable_hypothetical_query(
    question: str,
    text: str,
    overlap_threshold: float,
    entailment: Callable[[str, str], bool] | None = None,
) -> bool:
    """确定性 hq 过滤；可注入冻结 NLI，词面规则是可复测保守路径。"""
    if entailment is not None and entailment(text, question):
        return True
    q_tokens = {
        token
        for token in tokenize(question)
        if token not in QUESTION_WORDS
        and token not in {"is", "are", "was", "were", "did", "does", "do", "the", "a", "an"}
    }
    text_tokens = set(tokenize(text))
    return bool(q_tokens) and len(q_tokens & text_tokens) / len(q_tokens) >= overlap_threshold


def _hq_keys(
    turn: dict,
    cfg: R2WConfig,
    entailment: Callable[[str, str], bool] | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    generated = hypothetical_queries(turn)
    keys = []
    metas = []
    used = 0
    for question in generated.texts[: cfg.hq_max_items]:
        if not answerable_hypothetical_query(
            question,
            f"{turn['speaker']} {turn['text']}",
            cfg.hq_answerability_overlap,
            entailment,
        ):
            continue
        size = word_count(question)
        if size == 0 or used + size > cfg.hq_total_words:
            continue
        keys.append(question.strip())
        metas.append({"type": "hq", "provenance": turn["dia_id"], "answerable": True})
        used += size
    return keys, metas


def _fit_key_budget(
    prefix: str,
    keys: list[str],
    metadata: list[dict[str, Any]],
    budget: int,
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    if budget < 0:
        raise ValueError("key_budget_words 不能为负。")
    rendered: list[str] = []
    kept_meta: list[dict[str, Any]] = []
    used = 0
    for key, meta in zip(keys, metadata):
        candidate = f"{prefix} {key}".strip()
        size = word_count(candidate)
        if used + size <= budget:
            rendered.append(candidate)
            kept_meta.append(dict(meta))
            used += size
    return tuple(rendered), tuple(kept_meta)


def build_representation(
    turn: dict,
    action: str,
    constructor: StructuredConstructor | None,
    cfg: R2WConfig | None = None,
    entailment: Callable[[str, str], bool] | None = None,
) -> Repr:
    """按冻结契约构建表示；raw 不调用 LLM/QG/NLI。"""
    cfg = cfg or default_config()
    if action not in STRUCT_ACTIONS:
        raise ValueError(f"未知 FINAL 表示动作: {action}")
    prefix = MetaBlock.from_turn(turn).render()
    compression, key_axis = ACTION_FACTORS[action]
    body = _string(turn.get("text"), "turn.text") if compression == 0 else _summary_body(turn, constructor, cfg)
    payload = f"{prefix} {body}"
    keys: list[str] = []
    metadata: list[dict[str, Any]] = []
    if key_axis == 1:
        if constructor is None:
            raise RuntimeError("构建 kv 草稿需要结构化构建器。")
        keys, metadata = _kv_keys(turn, constructor, cfg)
    elif key_axis == 2:
        if constructor is None:
            raise RuntimeError("构建 event 草稿需要结构化构建器。")
        keys, metadata = _event_keys(turn, constructor, cfg)
    elif key_axis == 3:
        if constructor is None:
            raise RuntimeError("构建 graph 草稿需要结构化构建器。")
        keys, metadata = _graph_keys(turn, constructor, cfg)
    elif key_axis == 4:
        keys, metadata = _hq_keys(turn, cfg, entailment)
    fitted_keys, fitted_meta = _fit_key_budget(prefix, keys, metadata, cfg.key_budget_words)
    result = Repr(action, payload, fitted_keys, fitted_meta, cfg.repr_protocol_version)
    if sum(word_count(key) for key in result.keys) > cfg.key_budget_words:
        raise AssertionError("表示构建违反 key 总预算。")
    return result
