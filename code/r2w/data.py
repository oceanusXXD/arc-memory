"""Strict benchmark adapters for the R2W v4 conversation schema.

The adapters only expose data available to the offline measurement protocol.
They retain benchmark evidence as sampling/audit metadata and never turn it
into an online feature.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空字符串。")
    return value.strip()


def _locomo_datetime(value: Any, field: str) -> str:
    original = _non_empty_string(value, field)
    try:
        return datetime.strptime(original, "%I:%M %p on %d %B, %Y").strftime("%Y-%m-%d %H:%M")
    except ValueError as exc:
        raise ValueError(f"{field} 不是 LoCoMo 的绝对时间格式。") from exc


def _longmemeval_datetime(value: Any, field: str) -> str:
    original = _non_empty_string(value, field)
    try:
        return datetime.strptime(original, "%Y/%m/%d (%a) %H:%M").strftime("%Y-%m-%d %H:%M")
    except ValueError as exc:
        raise ValueError(f"{field} 不是 LongMemEval 的绝对时间格式。") from exc


def _normalise_turns(value: Any) -> list[dict[str, Any]]:
    turns = [dict(turn) for turn in value]
    if not turns:
        raise ValueError("每段训练对话至少要有一个 turn。")
    for turn in turns:
        for field in ("dia_id", "speaker", "text", "timestamp"):
            turn[field] = _non_empty_string(turn.get(field), f"turn.{field}")
        if not isinstance(turn.get("session"), int) or turn["session"] < 1:
            raise ValueError("turn.session 必须是从 1 开始的整数。")
        turn.setdefault("normalized_time", turn["timestamp"])
        turn.setdefault("original_time_expression", "")
    return turns


def _build_dia_to_index(turns: list[dict[str, Any]]) -> dict[str, int]:
    dia_to_index = {turn["dia_id"]: index for index, turn in enumerate(turns)}
    if len(dia_to_index) != len(turns):
        raise ValueError("同一段对话中的 dia_id 必须唯一。")
    return dia_to_index


def _normalise_qa(value: Any, dia_to_index: dict[str, int]) -> list[dict[str, Any]]:
    qa = []
    for query in value:
        question = _non_empty_string(query.get("question"), "qa.question")
        evidence = query.get("evidence", [])
        if not isinstance(evidence, list):
            raise ValueError("qa.evidence 必须是数组。")
        mapped = list(dict.fromkeys(value for value in evidence if value in dia_to_index))
        # Benchmark evidence is optional sampling/audit metadata. Preserve any
        # schema mismatch explicitly instead of treating it as a negative label
        # or discarding the complete query from replay.
        missing = list(dict.fromkeys(value for value in evidence if value not in dia_to_index))
        record = {**query, "question": question, "evidence": mapped}
        if missing:
            record["unmapped_evidence"] = missing
        qa.append(record)
    return qa


def _normalise_conversation(raw: dict[str, Any]) -> dict[str, Any]:
    turns = _normalise_turns(raw.get("turns", []))
    dia_to_index = _build_dia_to_index(turns)
    qa = _normalise_qa(raw.get("qa", []), dia_to_index)
    return {
        "conversation_id": _non_empty_string(raw.get("conversation_id"), "conversation_id"),
        "turns": turns,
        "qa": qa,
        "dia_to_index": dia_to_index,
        "speaker_a": str(raw.get("speaker_a", "")),
        "speaker_b": str(raw.get("speaker_b", "")),
    }


def _locomo_evidence(values: Any) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("LoCoMo qa.evidence 必须是数组。")
    evidence = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("LoCoMo evidence 项必须是字符串。")
        for item in re.split(r"[;,\s]+", value.strip()):
            if not item:
                continue
            prefix, separator, suffix = item.partition(":")
            evidence.append(f"{prefix}:{int(suffix)}" if separator and suffix.isdigit() else item)
    return evidence


def _locomo_sessions(conversation: dict[str, Any]) -> list[tuple[int, str, list[Any]]]:
    sessions = []
    for key, value in conversation.items():
        match = re.fullmatch(r"session_(\d+)", key)
        if match and isinstance(value, list):
            sessions.append((int(match.group(1)), key, value))
    if not sessions:
        raise ValueError("LoCoMo conversation 不包含 session_N 数组。")
    return sorted(sessions)


def _flatten_locomo_turns(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    turns = []
    for session, key, values in _locomo_sessions(conversation):
        original_time = _non_empty_string(conversation.get(f"{key}_date_time"), f"{key}_date_time")
        timestamp = _locomo_datetime(original_time, f"{key}_date_time")
        for turn in values:
            if not isinstance(turn, dict):
                raise ValueError(f"{key} 中的 turn 必须是对象。")
            turns.append({
                **turn,
                "timestamp": timestamp,
                "normalized_time": timestamp,
                "original_time_expression": original_time,
                "session": session,
            })
    return turns


def _locomo_qa(value: Any) -> list[dict[str, Any]]:
    return [{**query, "evidence": _locomo_evidence(query.get("evidence", []))} for query in value]


def adapt_locomo_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten LoCoMo sessions into v4 turns with deterministic time anchors."""
    conversation = raw.get("conversation")
    if not isinstance(conversation, dict):
        raise ValueError("LoCoMo conversation 必须是对象。")
    return _normalise_conversation({
        "conversation_id": raw.get("sample_id"),
        "turns": _flatten_locomo_turns(conversation),
        "qa": _locomo_qa(raw.get("qa", [])),
        "speaker_a": conversation.get("speaker_a", ""),
        "speaker_b": conversation.get("speaker_b", ""),
    })


def _longmemeval_arrays(raw: dict[str, Any]) -> tuple[list[Any], list[Any], list[Any]]:
    sessions, session_ids, dates = raw.get("haystack_sessions"), raw.get("haystack_session_ids"), raw.get("haystack_dates")
    if not all(isinstance(value, list) for value in (sessions, session_ids, dates)):
        raise ValueError("LongMemEval 必须包含 session、session_id 与日期数组。")
    if not sessions or len(sessions) != len(session_ids) or len(sessions) != len(dates):
        raise ValueError("LongMemEval session、session_id 与日期数量必须相同且非空。")
    return sessions, session_ids, dates


def _longmemeval_turn(session: Any, session_id: Any, date: Any, position: int) -> dict[str, Any]:
    session_id = _non_empty_string(session_id, "LongMemEval session_id")
    if not isinstance(session, list) or not session:
        raise ValueError("LongMemEval 的每个 session 必须是非空数组。")
    lines = []
    for dialog_turn in session:
        if not isinstance(dialog_turn, dict):
            raise ValueError("LongMemEval session turn 必须是对象。")
        role = _non_empty_string(dialog_turn.get("role"), "LongMemEval role")
        content = _non_empty_string(dialog_turn.get("content"), "LongMemEval content")
        lines.append(f"{role}: {content}")
    return {
        "dia_id": session_id,
        "speaker": "session",
        "text": "\n".join(lines),
        "timestamp": _longmemeval_datetime(date, "haystack_dates"),
        "session": position,
    }


def _longmemeval_turns(raw: dict[str, Any]) -> list[dict[str, Any]]:
    turns = []
    sessions, session_ids, dates = _longmemeval_arrays(raw)
    for position, values in enumerate(zip(sessions, session_ids, dates, strict=True), start=1):
        turns.append(_longmemeval_turn(*values, position))
    evidence = raw.get("answer_session_ids")
    if not isinstance(evidence, list):
        raise ValueError("LongMemEval answer_session_ids 必须是数组。")
    return turns


def _longmemeval_qa(raw: dict[str, Any], question_id: str) -> list[dict[str, Any]]:
    evidence = raw.get("answer_session_ids")
    if not isinstance(evidence, list):
        raise ValueError("LongMemEval answer_session_ids 必须是数组。")
    return [{
        "id": question_id,
        "question": raw.get("question"),
        "answer": raw.get("answer"),
        "evidence": evidence,
        "timestamp": _longmemeval_datetime(raw.get("question_date"), "question_date"),
        "category": raw.get("question_type"),
    }]


def adapt_longmemeval_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Map each LongMemEval haystack session to a single R2W parent memory."""
    question_id = _non_empty_string(raw.get("question_id"), "LongMemEval question_id")
    return _normalise_conversation({
        "conversation_id": question_id,
        "turns": _longmemeval_turns(raw),
        "qa": _longmemeval_qa(raw, question_id),
        "speaker_a": "session_a",
        "speaker_b": "session_b",
    })


def load_benchmark_conversations(path: str | Path) -> list[dict[str, Any]]:
    """Load LoCoMo or LongMemEval JSON into the v4 offline-replay schema."""
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError("数据集根节点必须是对象数组。")
    conversations = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("数据集记录必须是对象。")
        if "conversation" in record:
            conversations.append(adapt_locomo_record(record))
        elif "haystack_sessions" in record:
            conversations.append(adapt_longmemeval_record(record))
        else:
            conversations.append(_normalise_conversation(record))
    return conversations
