from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import re

from arc.agent.config import cache_path, load_config, processed_path, write_manifest
from arc.agent.io import read_json, read_jsonl, write_json
from arc.agent.data.locomo import load_blocks, load_qas
from arc.agent.data.score import retrieval_recall


_CACHE_INDEX_VERSION = 1
_CACHE_ID_RE = re.compile(rb'"(sample_id|qa_id)"\s*:\s*"((?:\\.|[^"\\])*)"')
_CACHE_DIA_IDS_RE = re.compile(rb'"dia_ids"\s*:\s*(\[[^\]]*\])')
_CACHE_RECALL_RE = re.compile(rb'"evidence_recall"\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')


def _retrieval_index_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_name(f"{target.name}.index.json")


def _retrieval_file_signature(path: str | Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "inode": int(stat.st_ino)}


def _write_retrieval_index(path: str | Path, offsets: Mapping[tuple[str, str], int]) -> Path:
    target = _retrieval_index_path(path)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    rows = [{"sample_id": sample_id, "qa_id": qa_id, "offset": offset}
            for (sample_id, qa_id), offset in offsets.items()]
    payload = {"version": _CACHE_INDEX_VERSION, **_retrieval_file_signature(path), "rows": rows}
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _read_retrieval_index(path: str | Path) -> dict[tuple[str, str], int] | None:
    target = _retrieval_index_path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        signature = _retrieval_file_signature(path)
        if not isinstance(payload, dict) or int(payload.get("version", -1)) != _CACHE_INDEX_VERSION:
            return None
        if any(int(payload.get(key, -1)) != value for key, value in signature.items()):
            return None
        rows = payload.get("rows")
        if not isinstance(rows, list):
            return None
        offsets: dict[tuple[str, str], int] = {}
        for item in rows:
            if not isinstance(item, dict):
                return None
            key = (str(item["sample_id"]), str(item["qa_id"]))
            offset = int(item["offset"])
            if offset < 0 or offset >= signature["size"] or key in offsets:
                return None
            offsets[key] = offset
        return offsets
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def _validate_retrieval_row(row: dict[str, Any], location: str) -> None:
    required = {"sample_id", "qa_id", "source_ids", "scores", "backend",
                "query_vector", "source_vectors", "embedding_metadata"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"{location}: row missing required fields: {sorted(missing)}")
    if row["backend"] != "hybrid_rrf":
        raise ValueError(f"{location}: backend must be hybrid_rrf, got {row['backend']!r}")


def _cached_row_header(line: bytes) -> tuple[str, str, list[str], re.Match[bytes]] | None:
    identities: dict[str, str] = {}
    for match in _CACHE_ID_RE.finditer(line):
        identities.setdefault(match.group(1).decode("ascii"), json.loads(b'"' + match.group(2) + b'"'))
    dia_match = _CACHE_DIA_IDS_RE.search(line)
    recall_match = _CACHE_RECALL_RE.search(line)
    if set(identities) != {"sample_id", "qa_id"} or dia_match is None or recall_match is None:
        return None
    dia_ids = json.loads(dia_match.group(1))
    if not isinstance(dia_ids, list):
        return None
    return identities["sample_id"], identities["qa_id"], [str(value) for value in dia_ids], recall_match


def _group_blocks(blocks: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for block in blocks:
        grouped[str(block["sample_id"])].append(block)
    for values in grouped.values():
        values.sort(key=lambda b: (int(b.get("turn_index", 0)), int(b.get("chunk_index", 0))))
    return grouped


class _BM25Index:
    def __init__(self, blocks: list[dict]):
        self.documents = [_lexical_tokens(str(block.get("text", ""))) for block in blocks]
        self.source_ids = [str(block["source_id"]) for block in blocks]
        self.avg_len = sum(len(document) for document in self.documents) / max(1, len(self.documents))
        self.document_frequency: dict[str, int] = {}
        for document in self.documents:
            for token in set(document):
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1

    def top(self, question: str, top_k: int) -> list[str]:
        query = _lexical_tokens(str(question))
        total, k1, b = len(self.documents), 1.5, 0.75
        scores: list[float] = []
        for document in self.documents:
            length = len(document)
            score = 0.0
            for token in query:
                frequency = document.count(token)
                if not frequency:
                    continue
                df = self.document_frequency.get(token, 0)
                idf = np.log(1.0 + (total - df + 0.5) / (df + 0.5))
                score += float(idf) * frequency * (k1 + 1.0) / (
                    frequency + k1 * (1.0 - b + b * length / max(self.avg_len, 1e-9))
                )
            scores.append(score)
        order = sorted(range(len(self.documents)), key=lambda i: (-scores[i], self.source_ids[i]))[:top_k]
        return [self.source_ids[i] for i in order]


def blocks_by_source(blocks: list[dict]) -> dict[str, dict]:
    return {str(block["source_id"]): block for block in blocks}


def _top_bm25(question: str, blocks: list[dict], top_k: int) -> tuple[list[str], list[float]]:
    """Deterministic Okapi BM25 used by the frozen hybrid retriever."""
    query = _lexical_tokens(str(question))
    documents = [_lexical_tokens(str(block.get("text", ""))) for block in blocks]
    avg_len = sum(len(doc) for doc in documents) / max(1, len(documents))
    document_frequency: dict[str, int] = {}
    for doc in documents:
        for token in set(doc): document_frequency[token] = document_frequency.get(token, 0) + 1
    total, k1, b = len(documents), 1.5, 0.75
    scores: list[float] = []
    for doc in documents:
        length, score = len(doc), 0.0
        counts = {token: doc.count(token) for token in set(query)}
        for token in query:
            frequency = counts.get(token, 0)
            if not frequency: continue
            df = document_frequency.get(token, 0)
            idf = np.log(1.0 + (total - df + 0.5) / (df + 0.5))
            score += float(idf) * frequency * (k1 + 1.0) / (frequency + k1 * (1.0 - b + b * length / max(avg_len, 1e-9)))
        scores.append(score)
    order = sorted(range(len(blocks)), key=lambda i: (-scores[i], str(blocks[i]["source_id"])))[:top_k]
    return [str(blocks[i]["source_id"]) for i in order], [float(scores[i]) for i in order]


def _lexical_tokens(text: str) -> list[str]:
    """Deterministic Unicode tokenization for the BM25 component."""
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+(?:[-_.][a-z0-9]+)*", normalized)


def _top_hybrid(question: str, blocks: list[dict], top_k: int, config: dict[str, Any] | None = None, model: Any = None) -> tuple[list[str], list[float], str]:
    bm_ids, _ = _top_bm25(question, blocks, top_k)
    dense_ids, _ = _top_dense_one(question, blocks, top_k, config, model=model)
    rrf_k = int((config or {}).get("retrieval", {}).get("rrf_k", 60))
    scores: dict[str, float] = {}
    for rank, source_id in enumerate(bm_ids, 1): scores[source_id] = scores.get(source_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, source_id in enumerate(dense_ids, 1): scores[source_id] = scores.get(source_id, 0.0) + 1.0 / (rrf_k + rank)
    order = sorted(scores, key=lambda source_id: (-scores[source_id], source_id))[:top_k]
    return order, [scores[source_id] for source_id in order], "hybrid_rrf"


def _top_dense_one(question: str, blocks: list[dict], top_k: int, config: dict[str, Any] | None = None, model: Any = None) -> tuple[list[str], list[float]]:
    from sentence_transformers import SentenceTransformer
    spec = (config or {}).get("retrieval", {})
    model = model or SentenceTransformer(
        str(spec.get("embedding_model") or "Qwen/Qwen3-Embedding-0.6B"),
        device=str(spec.get("embedding_device") or "cpu"),
    )
    prompt_name = spec.get("query_prompt_name")
    encode_kwargs = {"prompt_name": prompt_name} if prompt_name else {}
    matrix = _normalize(np.asarray(model.encode([str(b.get("text", "")) for b in blocks], show_progress_bar=False), dtype="float32"))
    query = _normalize(np.asarray(model.encode([question], show_progress_bar=False, **encode_kwargs), dtype="float32"))[0]
    sims = matrix @ query
    order = sorted(range(len(blocks)), key=lambda i: (-float(sims[i]), str(blocks[i]["source_id"])))[:top_k]
    return [str(blocks[i]["source_id"]) for i in order], [float(sims[i]) for i in order]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return matrix / denom


def _top_dense(config: dict[str, Any], grouped_blocks: dict[str, list[dict]], qas: list[dict], top_k: int,
               *, model: Any = None) -> list[dict]:
    from sentence_transformers import SentenceTransformer

    if model is None:
        model_name = config["retrieval"]["embedding_model"]
        device = config["retrieval"].get("embedding_device", "cpu")
        model = SentenceTransformer(model_name, device=device)
    rows: list[dict] = []
    for sample_id, sample_blocks in grouped_blocks.items():
        texts = [str(block.get("text", "")) for block in sample_blocks]
        if not texts:
            continue
        block_emb = _normalize(np.asarray(model.encode(texts, show_progress_bar=False), dtype="float32"))
        source_ids = [str(block["source_id"]) for block in sample_blocks]
        sample_qas = [qa for qa in qas if qa["sample_id"] == sample_id]
        questions = [qa["question"] for qa in sample_qas]
        if not questions:
            continue
        query_emb = _normalize(np.asarray(model.encode(questions, show_progress_bar=False), dtype="float32"))
        sims = query_emb @ block_emb.T
        for qa, scores in zip(sample_qas, sims):
            order = sorted(range(len(source_ids)), key=lambda i: (-float(scores[i]), source_ids[i]))[:top_k]
            selected_ids = [source_ids[i] for i in order]
            rows.append(_retrieval_row(qa, sample_blocks, selected_ids, [float(scores[i]) for i in order], "dense",
                                      dense_ids=selected_ids))
    return rows


def _hybrid_rows(embedder: Any, sample_blocks: list[dict], sample_qas: list[dict], top_k: int,
                 rrf_k: int) -> Iterator[dict]:
    source_vectors = embedder.encode([str(block.get("text") or "") for block in sample_blocks])
    source_ids = [str(block["source_id"]) for block in sample_blocks]
    query_vectors = embedder.encode([str(qa["question"]) for qa in sample_qas], query=True)
    bm25 = _BM25Index(sample_blocks)
    source_vector_map = {
        source_id: vector.tolist() for source_id, vector in zip(source_ids, source_vectors)
    }
    for qa, query_vector in zip(sample_qas, query_vectors):
        bm_ids = bm25.top(str(qa["question"]), top_k)
        sims = source_vectors @ query_vector
        dense_order = sorted(range(len(source_ids)), key=lambda i: (-float(sims[i]), source_ids[i]))[:top_k]
        dense_ids = [source_ids[i] for i in dense_order]
        scores: dict[str, float] = {}
        for rank, source_id in enumerate(bm_ids, 1):
            scores[source_id] = scores.get(source_id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, source_id in enumerate(dense_ids, 1):
            scores[source_id] = scores.get(source_id, 0.0) + 1.0 / (rrf_k + rank)
        selected_ids = sorted(scores, key=lambda source_id: (-scores[source_id], source_id))[:top_k]
        row = _retrieval_row(
            qa,
            sample_blocks,
            selected_ids,
            [scores[source_id] for source_id in selected_ids],
            "hybrid_rrf",
            bm25_ids=bm_ids,
            dense_ids=dense_ids,
        )
        row["query_vector"] = query_vector.tolist()
        row["embedding_metadata"] = dict(embedder.metadata)
        row["source_vectors"] = {
            source_id: source_vector_map[source_id]
            for source_id in selected_ids
            if source_id in source_vector_map
        }
        yield row


def _retrieval_row(qa: dict, sample_blocks: list[dict], source_ids: list[str], scores: list[float], backend: str,
                   *, bm25_ids: list[str] | None = None, dense_ids: list[str] | None = None) -> dict:
    source_to_dia = {str(block["source_id"]): str(block["dia_id"]) for block in sample_blocks}
    dia_ids = [source_to_dia[source_id] for source_id in source_ids if source_id in source_to_dia]
    return {
        "sample_id": qa["sample_id"],
        "qa_id": qa["qa_id"],
        "qa_index": qa["qa_index"],
        "question": qa["question"],
        "category": qa["category"],
        "source_ids": source_ids,
        "scores": scores,
        "bm25_rank": {str(value): index for index, value in enumerate(bm25_ids or [], 1)},
        "dense_rank": {str(value): index for index, value in enumerate(dense_ids or [], 1)},
        "rrf_score": {str(value): float(score) for value, score in zip(source_ids, scores)},
        "dia_ids": dia_ids,
        "evidence_recall": retrieval_recall(list(qa.get("evidence") or []), dia_ids),
        "backend": backend,
    }


def _attach_cached_vectors(config: dict[str, Any], rows: list[dict], grouped: dict[str, list[dict]], embedder: Any = None) -> None:
    """Attach frozen query/source vectors needed by the online selector.

    Vectors are stored in the retrieval cache rather than recomputed inside H.
    """
    if not bool((config.get("retrieval") or {}).get("cache_vectors", True)):
        raise ValueError("retrieval.cache_vectors must be enabled for the selector")
    from .embeddings import FrozenEmbedder
    embedder = embedder or FrozenEmbedder(config)
    by_sample: dict[str, tuple[dict[str, list[float]], dict[str, Any]]] = {}
    for sample_id, blocks in grouped.items():
        vectors = embedder.encode([str(block.get("text") or "") for block in blocks])
        source_vectors = {str(block["source_id"]): vector.tolist() for block, vector in zip(blocks, vectors)}
        by_sample[sample_id] = (source_vectors, embedder.metadata)
    rows_by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in rows: rows_by_sample[str(row.get("sample_id"))].append(row)
    for sample_id, sample_rows in rows_by_sample.items():
        item = by_sample[sample_id]
        source_vectors, metadata = item
        queries = embedder.encode([str(row.get("question") or "") for row in sample_rows], query=True)
        for row, query_vector in zip(sample_rows, queries):
            row["query_vector"] = query_vector.tolist()
            row["embedding_metadata"] = dict(metadata)
            row["source_vectors"] = {source_id: source_vectors[source_id] for source_id in row.get("source_ids", []) if source_id in source_vectors}


def build_retrieval_cache(config: dict[str, Any], backend: str | None = None,
                          sample_ids: set[str] | None = None) -> dict[str, Any]:
    from .embeddings import FrozenEmbedder
    embedder = FrozenEmbedder(config)
    blocks = load_blocks(config)
    qas = load_qas(config)
    if sample_ids:
        selected = {str(value) for value in sample_ids}
        blocks = [block for block in blocks if str(block.get("sample_id")) in selected]
        qas = [qa for qa in qas if str(qa.get("sample_id")) in selected]
    grouped = _group_blocks(blocks)
    top_k = int(config["retrieval"].get("top_k", 128))
    requested = backend or str(config["retrieval"].get("backend", "auto"))
    if requested == "auto":
        actual = "hybrid"
    else:
        actual = requested
    if actual not in {"dense", "hybrid"}:
        raise ValueError("retrieval backend must be auto, dense, or hybrid")
    output = cache_path(config, "retrieval", "locomo_topk.jsonl")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    offsets: dict[tuple[str, str], int] = {}
    qas_by_sample: dict[str, list[dict]] = defaultdict(list)
    for qa in qas:
        qas_by_sample[str(qa["sample_id"])].append(qa)
    count = 0
    evidence_recall_sum = 0.0
    try:
        with temporary.open("wb") as handle:
            for sample_id, sample_blocks in grouped.items():
                sample_qas = qas_by_sample.get(sample_id, [])
                if not sample_qas or not sample_blocks:
                    continue
                if actual == "dense":
                    rows = _top_dense(config, {sample_id: sample_blocks}, sample_qas, top_k, model=embedder.model)
                    _attach_cached_vectors(config, rows, {sample_id: sample_blocks}, embedder=embedder)
                    iterator = iter(rows)
                else:
                    iterator = _hybrid_rows(
                        embedder,
                        sample_blocks,
                        sample_qas,
                        top_k,
                        int(config["retrieval"].get("rrf_k", 60)),
                    )
                for row in iterator:
                    key = (str(row["sample_id"]), str(row["qa_id"]))
                    if key in offsets:
                        raise ValueError(f"duplicate retrieval cache row: {key[0]} / {key[1]}")
                    offsets[key] = handle.tell()
                    handle.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
                    evidence_recall_sum += float(row["evidence_recall"])
                    count += 1
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if count != len(qas):
        raise ValueError(f"retrieval cache wrote {count} rows, expected {len(qas)}")
    temporary.replace(output)
    index_path = _write_retrieval_index(output, offsets)
    meta = {
        "backend": actual,
        "top_k": top_k,
        "embedding_model": config["retrieval"].get("embedding_model"),
        "items": count,
        "sample_ids": sorted({str(qa.get("sample_id")) for qa in qas}),
        "avg_evidence_recall": evidence_recall_sum / count if count else 0.0,
        "index": str(index_path),
    }
    write_json(cache_path(config, "retrieval", "index_manifest.json"), meta)
    write_manifest(config, "retrieval", {"retrieval": meta})
    return {"output": str(output), **meta}


def refresh_evidence_recall(config: dict[str, Any], input_path: str | Path | None = None,
                            manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Refresh cached evidence recall against the normalized processed QA file.

    The retrieval rankings and frozen vectors are left unchanged. Only the
    small identity, evidence, and recall fields are parsed from each line;
    large vector payloads are copied as bytes.
    """
    source_path = Path(input_path) if input_path else cache_path(config, "retrieval", "locomo_topk.jsonl")
    if not source_path.exists():
        raise FileNotFoundError(f"missing {source_path}; run retrieval index first")
    qas = load_qas(config)
    qas_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for qa in qas:
        key = (str(qa["sample_id"]), str(qa["qa_id"]))
        if key in qas_by_key:
            raise ValueError(f"duplicate processed QA: {key[0]} / {key[1]}")
        qas_by_key[key] = qa

    temporary = source_path.with_name(f".{source_path.name}.{os.getpid()}.tmp")
    seen: set[tuple[str, str]] = set()
    offsets: dict[tuple[str, str], int] = {}
    sample_ids: set[str] = set()
    count = 0
    evidence_recall_sum = 0.0
    try:
        with source_path.open("rb") as source, temporary.open("wb") as target:
            for line_number, line in enumerate(source, 1):
                line = line.rstrip(b"\r\n")
                if not line:
                    continue
                header = _cached_row_header(line)
                if header is not None:
                    sample_id, qa_id, dia_ids, recall_match = header
                    key = (sample_id, qa_id)
                else:
                    try:
                        row = json.loads(line)
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise ValueError(f"invalid retrieval cache JSON at line {line_number}") from exc
                    if not isinstance(row, dict):
                        raise ValueError(f"retrieval cache row at line {line_number} must be an object")
                    key = (str(row.get("sample_id")), str(row.get("qa_id")))
                    dia_ids = row.get("dia_ids")
                    if not isinstance(dia_ids, list):
                        raise ValueError(f"retrieval cache row missing dia_ids at line {line_number}")
                    output_row = row
                qa = qas_by_key.get(key)
                if qa is None:
                    raise ValueError(f"retrieval cache row has no processed QA: {key[0]} / {key[1]}")
                if key in seen:
                    raise ValueError(f"duplicate retrieval cache row: {key[0]} / {key[1]}")
                recall = retrieval_recall(list(qa.get("evidence") or []), list(dia_ids or []))
                if header is not None:
                    output_line = line[:recall_match.start(1)] + json.dumps(
                        recall, separators=(",", ":")
                    ).encode("ascii") + line[recall_match.end(1):] + b"\n"
                else:
                    output_row["evidence_recall"] = recall
                    output_line = (json.dumps(output_row, ensure_ascii=False) + "\n").encode("utf-8")
                target_offset = target.tell()
                target.write(output_line)
                seen.add(key)
                offsets[key] = target_offset
                sample_ids.add(key[0])
                count += 1
                evidence_recall_sum += recall
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    expected = {key for key in qas_by_key if key[0] in sample_ids}
    missing = expected - seen
    if missing:
        temporary.unlink(missing_ok=True)
        preview = sorted(missing)[:3]
        raise ValueError(f"retrieval cache is missing {len(missing)} processed QAs, e.g. {preview}")
    temporary.replace(source_path)
    index_path = _write_retrieval_index(source_path, offsets)

    target_manifest = Path(manifest_path) if manifest_path else source_path.with_name("index_manifest.json")
    metadata: dict[str, Any] = {}
    if target_manifest.exists():
        existing = read_json(target_manifest)
        if isinstance(existing, dict):
            metadata = existing
    metadata["items"] = count
    metadata["sample_ids"] = sorted(sample_ids)
    metadata["avg_evidence_recall"] = evidence_recall_sum / count if count else 0.0
    metadata["evidence_recall_source"] = str(processed_path(config, "locomo_qa.jsonl"))
    metadata["index"] = str(index_path)
    write_json(target_manifest, metadata)
    write_manifest(config, "retrieval_evidence", {"retrieval": metadata, "input": str(source_path)})
    return {
        "output": str(source_path),
        "manifest": str(target_manifest),
        "items": count,
        "sample_ids": sorted(sample_ids),
        "avg_evidence_recall": metadata["avg_evidence_recall"],
    }


class RetrievalCache(Mapping[tuple[str, str], dict]):
    """Indexed JSONL retrieval cache that loads one row per lookup."""

    required_fields = {"sample_id", "qa_id", "source_ids", "scores", "backend",
                       "query_vector", "source_vectors", "embedding_metadata"}

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._offsets = _read_retrieval_index(self.path)
        if self._offsets is None:
            self._offsets = {}
            with self.path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    try:
                        row = json.loads(line)
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise ValueError(f"invalid retrieval cache JSON at byte offset {offset}") from exc
                    if not isinstance(row, dict):
                        raise ValueError(f"retrieval cache row at byte offset {offset} must be an object")
                    _validate_retrieval_row(row, f"retrieval cache byte offset {offset}")
                    key = (str(row["sample_id"]), str(row["qa_id"]))
                    if key in self._offsets:
                        raise ValueError(f"duplicate retrieval cache row: {key[0]} / {key[1]}")
                    self._offsets[key] = offset
            _write_retrieval_index(self.path, self._offsets)
        if not self._offsets:
            raise ValueError(f"retrieval cache is empty: {self.path}")

    def __getitem__(self, key: tuple[str, str]) -> dict:
        offset = self._offsets[key]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            row = json.loads(handle.readline())
        if not isinstance(row, dict):
            raise ValueError(f"retrieval cache row at byte offset {offset} must be an object")
        _validate_retrieval_row(row, f"retrieval cache byte offset {offset}")
        actual_key = (str(row["sample_id"]), str(row["qa_id"]))
        if actual_key != key:
            raise ValueError(f"retrieval cache index mismatch at byte offset {offset}")
        return row

    def iter_rows(self) -> Iterator[dict]:
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                try:
                    row = json.loads(line)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid retrieval cache JSON at byte offset {offset}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"retrieval cache row at byte offset {offset} must be an object")
                _validate_retrieval_row(row, f"retrieval cache byte offset {offset}")
                yield row

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self._offsets)

    def __reversed__(self) -> Iterator[tuple[str, str]]:
        return reversed(self._offsets)

    def __len__(self) -> int:
        return len(self._offsets)


def load_retrieval(config: dict[str, Any]) -> RetrievalCache:
    path = cache_path(config, "retrieval", "locomo_topk.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run python -m arc.agent.data.retrieval index first")
    return RetrievalCache(path)


def _selection_block(raw: dict[str, Any], source_id: str, internal_id: int) -> dict[str, Any]:
    block = dict(raw)
    block["id"] = internal_id
    block["source_id"] = str(block.get("source_id", source_id))
    block.setdefault("session", block.get("session_id"))
    block.setdefault("position", block.get("position_in_session", internal_id))
    block.setdefault("retrieval_score", block.get("score", 0.0))
    block["text"] = str(block.get("text") or "")
    return block


def select_sources(retrieval_row: dict, source_lookup: dict[str, dict], cap_tokens: int,
                   max_blocks: int | None = None, *, cost_fn=None, input_limit: int | None = None) -> tuple[list[dict], list[str]]:
    if cost_fn is None:
        raise ValueError("exact cost_fn is required for source selection")
    selected: list[dict] = []
    dropped: list[str] = []
    source_vectors = retrieval_row.get("source_vectors") or {}
    query_vector = retrieval_row.get("query_vector")
    query_array = np.asarray(query_vector, dtype="float32")
    if query_array.ndim != 1 or not np.isfinite(query_array).all():
        raise ValueError(f"invalid frozen query vector for {retrieval_row.get('sample_id')} / {retrieval_row.get('qa_id')}")
    query_norm = float(np.linalg.norm(query_array))
    bm25_rank = retrieval_row.get("bm25_rank") or {}
    dense_rank = retrieval_row.get("dense_rank") or {}
    rrf_score = retrieval_row.get("rrf_score") or {}
    cost_cache: dict[tuple[str, ...], int] = {}
    used_body = 0
    for source_id in retrieval_row.get("source_ids", []):
        source_id = str(source_id)
        if max_blocks is not None and len(selected) >= max_blocks:
            dropped.append(source_id)
            continue
        block = source_lookup.get(source_id)
        if not block:
            dropped.append(source_id)
            continue
        block = _selection_block(block, source_id, len(selected) + 1)
        if source_id in source_vectors:
            block["embedding"] = source_vectors[source_id]
            block["query_vector"] = query_vector
            block["query_vector_question"] = retrieval_row.get("question")
            block["embedding_metadata"] = retrieval_row.get("embedding_metadata")
        block["bm25_rank"] = bm25_rank.get(source_id)
        block["dense_rank"] = dense_rank.get(source_id)
        block["rrf_score"] = rrf_score.get(source_id, 0.0)
        if block.get("embedding") is None:
            raise ValueError(f"missing frozen vectors for source {source_id}")
        source_array = np.asarray(block["embedding"], dtype="float32")
        if source_array.shape != query_array.shape or not np.isfinite(source_array).all():
            raise ValueError(f"invalid frozen vectors for source {source_id}")
        block["cosine_similarity"] = float(source_array @ query_array / max(float(np.linalg.norm(source_array)) * query_norm, 1e-12))
        if block.get("token_count") is None:
            raise ValueError(f"missing frozen token_count for source {source_id}; rerun locomo prepare")
        tokens = int(block["token_count"])
        body_cost = used_body + tokens
        candidate = [*selected, block]
        cost_key = tuple(str(item["source_id"]) for item in candidate)
        if cost_key not in cost_cache:
            cost_cache[cost_key] = int(cost_fn(candidate))
        candidate_cost = cost_cache[cost_key]
        if body_cost > cap_tokens or (input_limit is not None and candidate_cost > input_limit):
            dropped.append(source_id)
            continue
        selected.append(block)
        used_body = body_cost
    return selected, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed LoCoMo retrieval cache.")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index")
    index.add_argument("--config", default="configs/locomo.yaml")
    index.add_argument("--backend", choices=("hybrid", "auto", "dense"))
    index.add_argument("--sample-ids", nargs="+", help="restrict the cache to selected LoCoMo conversations")
    refresh = sub.add_parser("refresh-evidence")
    refresh.add_argument("--config", default="configs/locomo.yaml")
    refresh.add_argument("--input")
    refresh.add_argument("--manifest")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "index":
        print(json.dumps(build_retrieval_cache(config, args.backend, set(args.sample_ids or [])), ensure_ascii=False, indent=2))
    elif args.command == "refresh-evidence":
        print(json.dumps(refresh_evidence_recall(config, args.input, args.manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
