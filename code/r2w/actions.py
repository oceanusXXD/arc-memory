from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import re

from .config import R2WConfig

class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

class SimpleTokenCounter:
    """仅用于测试/无 tokenizer 的离线单元测试。正式实验应注入 reader tokenizer。"""
    def count(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

class StructuredConstructor(Protocol):
    def json(self, prompt: str) -> dict: ...

CANDIDATE_PROMPT = """你是 R2W 候选包构建器。只能使用当前轮次及给定的历史前缀做代词、省略和相对时间解析；不得复制历史前缀中的独立事实。严格输出 JSON object，字段为 summary, kv, event, hq{graph_suffix}。
summary: 忠实摘要字符串；kv: 至多 {kv_n} 个 object {{entity,attribute,value}}；event: 至多 {event_n} 个 object {{time,subject,event}}；hq: 至多 {hq_n} 个可由当前轮次回答的问题字符串{graph_rule}。
当前轮次时间: {timestamp}
说话人: {speaker}
历史前缀: {history}
当前轮次: {text}"""
RELATIVE_TIME_RE = re.compile(
    r"\b(?:today|yesterday|tomorrow|last\s+\w+|next\s+\w+)\b|今天|昨天|明天|上周|下周",
    re.I,
)

@dataclass(frozen=True)
class CandidateBundle:
    summary: str
    kv: tuple[dict[str, str], ...]
    event: tuple[dict[str, str], ...]
    hq: tuple[str, ...]
    graph: tuple[dict[str, str], ...] = ()
    protocol_version: str = "r2w-repr-v4"

@dataclass(frozen=True)
class Representation:
    action: str
    parent_id: str
    body: str
    keys: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = "r2w-repr-v4"

    def __post_init__(self):
        if not self.parent_id or not self.body.strip():
            raise ValueError("parent_id/body 不能为空。")
        if any(not k.strip() for k in self.keys):
            raise ValueError("检索键不能为空。")


def _string(v: Any, name: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{name} 必须是非空字符串。")
    if any(ord(ch)<32 and ch not in "\n\t" for ch in v):
        raise ValueError(f"{name} 含非法控制字符。")
    return v.strip()


def _list_of_dicts(value: Any, name: str, max_items: int, fields: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or len(value) > max_items:
        raise ValueError(f"{name} 必须是至多 {max_items} 项的数组。")
    out = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"{name} 每项必须是 object。")
        out.append({f: _string(row.get(f), f"{name}.{f}") for f in fields})
    return tuple(out)


def _candidate_prompt(turn: dict, history_prefix: list[dict], cfg: R2WConfig) -> str:
    history = "\n".join(f"{x.get('speaker','')}: {x.get('text','')}" for x in history_prefix)
    graph_suffix = ", graph" if cfg.use_extended_actions else ""
    graph_rule = f"；graph: 至多 {cfg.graph_max_items} 个 object {{subject,relation,object}}" if cfg.use_extended_actions else ""
    return CANDIDATE_PROMPT.format(
        graph_suffix=graph_suffix, graph_rule=graph_rule, kv_n=cfg.kv_max_items,
        event_n=cfg.event_max_items, hq_n=cfg.hq_max_items,
        timestamp=_string(turn.get("timestamp"), "timestamp"),
        speaker=_string(turn.get("speaker"), "speaker"), history=history,
        text=_string(turn.get("text"), "text"),
    )


def _validate_event_times(event: tuple[dict[str, str], ...]) -> None:
    for row in event:
        if RELATIVE_TIME_RE.search(row["time"]) or not re.search(r"\d{4}", row["time"]):
            raise ValueError("event.time 必须是可靠绝对时间锚；不得保留相对时间。")


def _parse_hq(value: dict, cfg: R2WConfig) -> tuple[str, ...]:
    hq_value = value.get("hq", [])
    if not isinstance(hq_value, list) or len(hq_value) > cfg.hq_max_items:
        raise ValueError("hq 数组超出契约。")
    hq = tuple(_string(x, "hq") for x in hq_value)
    if len(set(hq)) != len(hq):
        raise ValueError("hq 不得包含重复项。")
    return hq


def _validate_kv_conflicts(kv: tuple[dict[str, str], ...]) -> None:
    seen_kv = {}
    for row in kv:
        key = (row["entity"].casefold(), row["attribute"].casefold())
        old = seen_kv.get(key)
        if old is not None and old.casefold() != row["value"].casefold():
            raise ValueError("候选包内部存在冲突 KV 值。")
        seen_kv[key] = row["value"]


def _parse_candidate_value(value: dict, cfg: R2WConfig) -> CandidateBundle:
    summary = _string(value.get("summary"), "summary")
    kv = _list_of_dicts(value.get("kv", []), "kv", cfg.kv_max_items, ("entity", "attribute", "value"))
    event = _list_of_dicts(value.get("event", []), "event", cfg.event_max_items, ("time", "subject", "event"))
    _validate_event_times(event)
    hq = _parse_hq(value, cfg)
    _validate_kv_conflicts(kv)
    graph: tuple[dict[str, str], ...] = ()
    if cfg.use_extended_actions:
        graph = _list_of_dicts(value.get("graph", []), "graph", cfg.graph_max_items, ("subject", "relation", "object"))
    return CandidateBundle(summary, kv, event, hq, graph, cfg.repr_protocol_version)


def build_candidate_bundle(turn: dict, history_prefix: list[dict], constructor: StructuredConstructor, cfg: R2WConfig) -> CandidateBundle:
    value = constructor.json(_candidate_prompt(turn, history_prefix, cfg))
    return _parse_candidate_value(value, cfg)


def common_metadata(turn: dict) -> dict[str, Any]:
    return {
        "speaker": _string(turn.get("speaker"), "speaker"),
        "timestamp": _string(turn.get("timestamp"), "timestamp"),
        "normalized_time": _string(turn.get("normalized_time", turn.get("timestamp")), "normalized_time"),
        "original_time_expression": str(turn.get("original_time_expression", "")).strip(),
        "session": int(turn.get("session", 0)),
        "turn_id": _string(turn.get("dia_id"), "dia_id"),
    }


def _meta_prefix(meta: dict[str, Any]) -> str:
    original = f" | original_time={meta['original_time_expression']}" if meta.get("original_time_expression") else ""
    return f"[speaker={meta['speaker']} | time={meta['normalized_time']}{original} | session={meta['session']} | turn={meta['turn_id']}]"


def _representation_body(action: str, raw: str, bundle: CandidateBundle | None) -> str:
    if action == "raw" or action.startswith("raw"):
        return raw
    if bundle is None:
        raise ValueError("生成式动作必须复用同一个 CandidateBundle。")
    return bundle.summary


def _validate_summary_budget(action: str, raw: str, body: str, cfg: R2WConfig, counter: TokenCounter) -> None:
    if action.startswith("sum"):
        limit = min(cfg.summary_max_tokens, max(1, int(counter.count(raw) * cfg.summary_ratio)))
        if counter.count(body) > limit:
            raise ValueError("summary_contract_violation")  # reject; never truncate


def _additional_keys(action: str, bundle: CandidateBundle | None, cfg: R2WConfig, counter: TokenCounter) -> list[str]:
    if bundle is None:
        return []
    keys: list[str] = []
    if "+kv" in action:
        keys += [f"{x['entity']} | {x['attribute']} | {x['value']}" for x in bundle.kv]
    if "+event" in action:
        keys += [f"{x['time']} | {x['subject']} | {x['event']}" for x in bundle.event]
    if "+hq" in action:
        if any(counter.count(q)>cfg.hq_item_max_tokens for q in bundle.hq):
            raise ValueError("hq_item_contract_violation")
        keys += list(bundle.hq)
    if "+graph" in action:
        keys += [f"{x['subject']} | {x['relation']} | {x['object']}" for x in bundle.graph]
    return keys


def _representation_keys(meta: dict[str, Any], body: str, action: str, bundle: CandidateBundle | None, cfg: R2WConfig, counter: TokenCounter) -> tuple[str, ...]:
    base_key = _meta_prefix(meta) + " " + body
    cards = _additional_keys(action, bundle, cfg, counter)
    if sum(counter.count(card) for card in cards) > cfg.key_budget_tokens:
        raise ValueError("key_contract_violation")
    return tuple([base_key, *cards])


def build_representation(turn: dict, action: str, bundle: CandidateBundle | None, cfg: R2WConfig, counter: TokenCounter) -> Representation | None:
    if action == "none":
        return None
    if action not in cfg.storage_actions:
        raise ValueError(f"动作 {action} 不属于当前配置。")
    raw = _string(turn.get("text"), "text")
    body = _representation_body(action, raw, bundle)
    _validate_summary_budget(action, raw, body, cfg, counter)
    metadata = {**common_metadata(turn), "action": action, "protocol_version": cfg.repr_protocol_version}
    keys = _representation_keys(metadata, body, action, bundle, cfg, counter)
    return Representation(action, metadata["turn_id"], keys[0], keys, metadata, cfg.repr_protocol_version)
