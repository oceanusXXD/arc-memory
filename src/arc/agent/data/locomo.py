from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from arc.agent.config import load_config, processed_path, write_manifest
from arc.algorithm.memory import tokenizer_from_config
from arc.agent.io import read_jsonl, write_json, write_jsonl
from arc.agent.text import split_by_token_budget


SESSION_RE = re.compile(r"^session_(\d+)$")
EVIDENCE_ID_RE = re.compile(r"D:?\d+:\d+")
DEFAULT_SPLITS = {
    "train": ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43", "conv-44"],
    "dev": ["conv-47", "conv-48"],
    "final": ["conv-49", "conv-50"],
}


def load_records(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ordered_session_keys(conversation: dict[str, Any]) -> list[str]:
    keys: list[tuple[int, str]] = []
    for key, value in conversation.items():
        match = SESSION_RE.match(key)
        if match and isinstance(value, list):
            keys.append((int(match.group(1)), key))
    return [key for _, key in sorted(keys)]


def _speaker_name(conversation: dict[str, Any], speaker: str) -> str:
    if speaker == "speaker_a":
        return str(conversation.get("speaker_a") or speaker)
    if speaker == "speaker_b":
        return str(conversation.get("speaker_b") or speaker)
    return str(speaker)


def _canonical_dia_id(value: str) -> str | None:
    match = re.fullmatch(r"D:?([0-9]+):([0-9]+)", str(value).strip())
    if not match:
        return None
    return f"D{int(match.group(1))}:{int(match.group(2))}"


def _normalize_evidence(raw: Any, sample_id: str, known_dia_ids: dict[tuple[str, str], str]) -> list[str]:
    values = [raw] if isinstance(raw, str) else list(raw or [])
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        tokens = EVIDENCE_ID_RE.findall(text)
        if not tokens and text:
            tokens = [text]
        for token in tokens:
            canonical = _canonical_dia_id(token)
            resolved = known_dia_ids.get((sample_id, canonical), token) if canonical else token
            if resolved not in normalized:
                normalized.append(resolved)
    return normalized


def records_to_blocks(records: list[dict[str, Any]], max_block_tokens: int = 256, *, token_counter=None) -> tuple[list[dict], dict[str, Any]]:
    blocks: list[dict] = []
    audit: dict[str, Any] = {
        "missing_session_dates": [],
        "duplicate_source_ids": [],
        "duplicate_dia_ids": [],
        "empty_turns": [],
    }
    seen_sources: set[str] = set()
    seen_dia: set[tuple[str, str]] = set()
    global_turn_index = 0
    for sample_index, sample in enumerate(records):
        sample_id = str(sample.get("sample_id") or f"sample-{sample_index}")
        conversation = sample.get("conversation") or {}
        for session_key in ordered_session_keys(conversation):
            session_index = int(session_key.split("_")[1])
            date_key = f"{session_key}_date_time"
            session_datetime = str(conversation.get(date_key) or "")
            if not session_datetime:
                audit["missing_session_dates"].append({"sample_id": sample_id, "session": session_key})
            for turn_index, turn in enumerate(conversation.get(session_key) or []):
                dia_id = str(turn.get("dia_id") or f"D{session_index}:{turn_index + 1}")
                dia_key = (sample_id, dia_id)
                if dia_key in seen_dia:
                    audit["duplicate_dia_ids"].append({"sample_id": sample_id, "dia_id": dia_id})
                seen_dia.add(dia_key)
                text = str(turn.get("text") or "").strip()
                if not text:
                    audit["empty_turns"].append({"sample_id": sample_id, "dia_id": dia_id})
                chunks = split_by_token_budget(text, max_block_tokens)
                for chunk_index, chunk in enumerate(chunks):
                    source_id = f"{sample_id}:{dia_id}:{chunk_index}"
                    if source_id in seen_sources:
                        audit["duplicate_source_ids"].append(source_id)
                    seen_sources.add(source_id)
                    blocks.append({
                        "sample_id": sample_id,
                        "session_id": session_key,
                        "session_index": session_index,
                        "session_datetime": session_datetime,
                        "dia_id": dia_id,
                        "source_id": source_id,
                        "chunk_index": chunk_index,
                        "speaker": _speaker_name(conversation, str(turn.get("speaker") or "")),
                        "text": chunk,
                        "turn_index": global_turn_index,
                        "position_in_session": turn_index,
                        "token_count": int(token_counter(chunk)) if token_counter is not None else None,
                    })
                global_turn_index += 1
    return blocks, audit


def records_to_qas(records: list[dict[str, Any]]) -> tuple[list[dict], dict[str, Any]]:
    qas: list[dict] = []
    audit: dict[str, Any] = {"unknown_categories": [], "missing_evidence": [],
                             "normalized_evidence": [], "category_counts": {}}
    dia_ids: set[tuple[str, str]] = set()
    canonical_dia_ids: dict[tuple[str, str], str] = {}
    for sample in records:
        sample_id = str(sample.get("sample_id"))
        for key in ordered_session_keys(sample.get("conversation") or {}):
            for turn in (sample.get("conversation") or {}).get(key, []):
                dia_id = str(turn.get("dia_id") or "")
                dia_ids.add((sample_id, dia_id))
                canonical = _canonical_dia_id(dia_id)
                if canonical is not None and (sample_id, canonical) not in canonical_dia_ids:
                    canonical_dia_ids[(sample_id, canonical)] = dia_id
    for sample_index, sample in enumerate(records):
        sample_id = str(sample.get("sample_id") or f"sample-{sample_index}")
        for index, qa in enumerate(sample.get("qa") or []):
            category = int(qa.get("category", 0) or 0)
            if category not in {1, 2, 3, 4, 5}:
                audit["unknown_categories"].append({"sample_id": sample_id, "qa_index": index, "category": category})
            audit["category_counts"][str(category)] = audit["category_counts"].get(str(category), 0) + 1
            raw_evidence = [qa.get("evidence")] if isinstance(qa.get("evidence"), str) else list(qa.get("evidence") or [])
            evidence = _normalize_evidence(raw_evidence, sample_id, canonical_dia_ids)
            if evidence != [str(ev) for ev in raw_evidence]:
                audit["normalized_evidence"].append({"sample_id": sample_id, "qa_index": index,
                                                       "original": [str(ev) for ev in raw_evidence],
                                                       "normalized": evidence})
            missing = [ev for ev in evidence if (sample_id, ev) not in dia_ids]
            if missing:
                audit["missing_evidence"].append({"sample_id": sample_id, "qa_index": index, "evidence": missing})
            qas.append({
                "sample_id": sample_id,
                "qa_id": str(qa.get("id") or f"{sample_id}:qa:{index}"),
                "qa_index": index,
                "question": str(qa.get("question") or ""),
                "answer": "" if "answer" not in qa else str(qa.get("answer")),
                "category": category,
                "evidence": evidence,
                "adversarial_answer": None if "adversarial_answer" not in qa else str(qa.get("adversarial_answer")),
                "eligible_for_training": category in {1, 2, 3, 4},
            })
    return qas, audit


def load_blocks(config: dict[str, Any]) -> list[dict]:
    path = processed_path(config, "locomo_blocks.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run python -m arc.agent.data.locomo prepare first")
    return list(read_jsonl(path))


def load_qas(config: dict[str, Any]) -> list[dict]:
    path = processed_path(config, "locomo_qa.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run python -m arc.agent.data.locomo prepare first")
    return list(read_jsonl(path))


def split_qas(config: dict[str, Any], split: str) -> list[dict]:
    qas = load_qas(config)
    groups = set((config.get("splits") or {}).get(split) or [])
    if not groups:
        raise ValueError(f"unknown or empty split: {split}")
    return [qa for qa in qas if qa["sample_id"] in groups]


def experiment_qas(config: dict[str, Any], group: str) -> list[dict]:
    """Return QAs from one immutable history partition."""
    members = set(map(str, (config.get("experiment_groups") or {}).get(group) or []))
    if not members:
        raise ValueError(f"unknown or empty experiment group: {group}")
    return [qa for qa in load_qas(config) if str(qa["sample_id"]) in members]


def prepare(input_path: str | Path, split_manifest: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    records = load_records(input_path)
    # Block caps participate in source selection, so they must use the same
    # frozen tokenizer as the builder cost function.
    token_counter = tokenizer_from_config(config)
    blocks, block_audit = records_to_blocks(records, token_counter=token_counter)
    qas, qa_audit = records_to_qas(records)
    split = {name: list(values) for name, values in (config.get("splits") or DEFAULT_SPLITS).items()}
    split_path = Path(split_manifest)
    write_json(split_path, split)
    blocks_path = processed_path(config, "locomo_blocks.jsonl")
    qas_path = processed_path(config, "locomo_qa.jsonl")
    audit_path = processed_path(config, "locomo_audit.json")
    write_jsonl(blocks_path, blocks)
    write_jsonl(qas_path, qas)
    audit = {
        "samples": len(records),
        "blocks": len(blocks),
        "qas": len(qas),
        "splits": split,
        "block_audit": block_audit,
        "qa_audit": qa_audit,
    }
    write_json(audit_path, audit)
    write_manifest(config, "prepare", {"prepare": {"blocks": len(blocks), "qas": len(qas), "audit": str(audit_path)}})
    return {"blocks": str(blocks_path), "qas": str(qas_path), "audit": str(audit_path), "split_manifest": str(split_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LoCoMo blocks, QA records, split manifest, and audit.")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--config", default="configs/locomo.yaml")
    prepare_parser.add_argument("--input")
    prepare_parser.add_argument("--split-manifest")
    args = parser.parse_args()
    config = load_config(args.config)
    input_path = args.input or config["paths"]["raw_data"]
    split_manifest = args.split_manifest or config["paths"]["split_manifest"]
    print(json.dumps(prepare(input_path, split_manifest, config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
