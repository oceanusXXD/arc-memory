"""Deterministic persistent-memory primitives used by the write/query path.

The method chapter separates a write update from a later query.  The original
adapter only built a packet for the current question, which made the selector
implicitly query-conditioned.  This module provides the small stateful layer
between those two phases: candidate generation sees a new history batch and
the current memory, ``Update`` stores the resulting packet, and ``Retrieve``
is the only operation used for later questions.

The store is deliberately JSON-serialisable.  It is not a second learned
model, and its ordering and replacement rules are deterministic so that an
offline compiler and deployment use the same state transition.
"""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .text import lexical_score, token_set


_TOKEN_RE = re.compile(r"\s+")


def _source_key(source: Mapping[str, Any], fallback: int) -> str:
    return str(source.get("source_id", source.get("id", fallback)))


def _copy_source(source: Mapping[str, Any], fallback: int) -> dict[str, Any]:
    item = dict(source)
    item["source_id"] = _source_key(item, fallback)
    item.setdefault("id", fallback)
    item.setdefault("session", item.get("session_id"))
    item.setdefault("position", item.get("position_in_session", fallback))
    item["text"] = str(item.get("text") or "")
    return item


def _memory_key(item: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    sources = tuple(sorted(str(value) for value in item.get("src", item.get("source_ids", ())) or ()))
    # Update semantics are provenance keyed: a later statement grounded in
    # the same source set replaces the prior version, while a statement with
    # different provenance remains an auditable entry.
    if sources:
        return "provenance", sources
    text = _TOKEN_RE.sub(" ", str(item.get("m") or item.get("text") or "").strip().lower())
    return "text", (text,)


@dataclass
class PersistentMemoryState:
    """A versioned, JSON-friendly memory state.

    ``sources`` contains the observable source units available to the fixed
    query retriever.  ``entries`` contains generated memory statements and
    their provenance.  Updating an existing provenance key replaces the old
    statement; otherwise the new statement is appended in insertion order.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    version: int = 0

    def clone(self) -> "PersistentMemoryState":
        return PersistentMemoryState(copy.deepcopy(self.entries), copy.deepcopy(self.sources), self.version)

    def update(
        self,
        memory_block: Iterable[Mapping[str, Any]],
        source_records: Iterable[Mapping[str, Any]] = (),
        *,
        architecture: str = "Flat",
        source_ids: Iterable[int | str] = (),
    ) -> "PersistentMemoryState":
        """Apply one deterministic ``Update`` transition and return a copy."""

        next_state = self.clone()
        known_sources = {str(item.get("source_id", item.get("id"))): item for item in next_state.sources}
        local_to_stable: dict[str, str] = {}
        for offset, source in enumerate(source_records, 1):
            item = _copy_source(source, len(known_sources) + offset)
            known_sources[item["source_id"]] = item
            local_to_stable[str(source.get("id", offset))] = item["source_id"]
        next_state.sources = [known_sources[key] for key in sorted(known_sources)]

        selected = tuple(sorted(str(value) for value in source_ids))
        now_entries: list[dict[str, Any]] = []
        for raw in memory_block:
            item = dict(raw)
            message = str(item.get("m") or item.get("text") or "").strip()
            if not message:
                continue
            raw_src = item.get("src", item.get("source_ids", selected)) or selected
            # Builder provenance uses the write-unit's local integer IDs;
            # persistent memory stores the stable source IDs so later Update
            # and Retrieve operations remain auditable across batches.
            src = tuple(sorted(local_to_stable.get(str(value), str(value)) for value in raw_src))
            entry = {
                "m": message,
                "src": list(src),
                "architecture": str(architecture),
                "version": next_state.version + 1,
            }
            now_entries.append(entry)

        # Replace an entry only when its provenance is the same.  A changed
        # statement with different provenance remains a separate auditable
        # memory rather than silently deleting evidence.
        by_key = {_memory_key(item): index for index, item in enumerate(next_state.entries)}
        for entry in now_entries:
            key = _memory_key(entry)
            if key in by_key:
                next_state.entries[by_key[key]] = entry
            else:
                by_key[key] = len(next_state.entries)
                next_state.entries.append(entry)
        next_state.version += 1
        return next_state

    def retrieve(self, query: str, *, top_k: int = 8) -> list[dict[str, Any]]:
        """Retrieve memory entries without invoking the write policy."""

        query = str(query or "")
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index, entry in enumerate(self.entries):
            text = str(entry.get("m") or "")
            source_text = " ".join(
                str(source.get("text") or "")
                for source in self.sources
                if str(source.get("source_id", source.get("id"))) in {str(v) for v in entry.get("src", ())}
            )
            score = lexical_score(query, f"{text} {source_text}")
            scored.append((float(score), -index, entry))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [copy.deepcopy(item[2]) for item in scored[: max(0, int(top_k))]]

    def render(self, query: str = "", *, top_k: int = 8) -> str:
        return self.render_entries(self.retrieve(query, top_k=top_k))

    @staticmethod
    def render_entries(entries: Iterable[Mapping[str, Any]]) -> str:
        """Render an already retrieved result without issuing a second read."""
        entries = list(entries)
        if not entries:
            return "Persistent memory: (empty)"
        lines = ["Persistent memory:"]
        for index, entry in enumerate(entries, 1):
            src = ",".join(str(value) for value in entry.get("src", ()))
            lines.append(f"[{index}] {entry.get('m', '')} [src:{src}]")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "arc.persistent_memory.v1", "version": self.version,
                "entries": copy.deepcopy(self.entries), "sources": copy.deepcopy(self.sources)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "PersistentMemoryState":
        if not value:
            return cls()
        if value.get("schema") not in (None, "arc.persistent_memory.v1"):
            raise ValueError("unsupported persistent memory schema")
        return cls(list(copy.deepcopy(value.get("entries") or [])),
                   list(copy.deepcopy(value.get("sources") or [])),
                   int(value.get("version") or 0))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "PersistentMemoryState":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def candidate_sources(
    history_batch: Iterable[Mapping[str, Any]],
    memory: PersistentMemoryState | Mapping[str, Any] | None = None,
    *,
    max_sources: int = 64,
    neighbourhood: int = 16,
) -> list[dict[str, Any]]:
    """Construct (E_i=C(X_i,M_i^-)) without looking at a query.

    New history is retained in its input order.  Existing source units are
    added only when their text overlaps the new batch or when they are among a
    deterministic recent neighbourhood.  No answer, query, or label is read.
    """

    state = memory if isinstance(memory, PersistentMemoryState) else PersistentMemoryState.from_dict(memory)
    fresh = [_copy_source(source, index) for index, source in enumerate(history_batch, 1)]
    fresh_text = " ".join(str(item.get("text") or "") for item in fresh)
    fresh_tokens = token_set(fresh_text)
    existing = list(state.sources)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, source in enumerate(existing):
        overlap = lexical_score(fresh_text, str(source.get("text") or "")) if fresh_text else 0.0
        # The recency term is observable memory state, not a query-dependent
        # score.  It only breaks ties when no related neighbourhood exists.
        recency = 1.0 / max(1, len(existing) - index)
        scored.append((overlap, recency, _copy_source(source, index + len(fresh) + 1)))
    scored.sort(key=lambda item: (-item[0], -item[1], str(item[2].get("source_id"))))
    selected_existing = [item[2] for item in scored[: max(0, int(neighbourhood))]]
    combined = fresh + selected_existing
    dedup: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(combined, 1):
        key = _source_key(source, index)
        dedup.setdefault(key, _copy_source(source, index))
    rows = list(dedup.values())[: max(0, int(max_sources))]
    # Selector IDs are local to this write unit and assigned in deterministic
    # candidate order.  Original source_id remains available for Update.
    for index, source in enumerate(rows, 1):
        source["id"] = index
    return rows


def Update(
    memory: PersistentMemoryState | Mapping[str, Any] | None,
    memory_block: Iterable[Mapping[str, Any]],
    source_records: Iterable[Mapping[str, Any]] = (),
    *,
    architecture: str = "Flat",
    source_ids: Iterable[int | str] = (),
) -> PersistentMemoryState:
    """Public method-level alias for the deterministic update interface."""

    state = memory if isinstance(memory, PersistentMemoryState) else PersistentMemoryState.from_dict(memory)
    return state.update(memory_block, source_records, architecture=architecture, source_ids=source_ids)


def Retrieve(
    query: str,
    memory: PersistentMemoryState | Mapping[str, Any] | None,
    *,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Public method-level alias for the fixed query-time retriever."""

    state = memory if isinstance(memory, PersistentMemoryState) else PersistentMemoryState.from_dict(memory)
    return state.retrieve(query, top_k=top_k)


__all__ = ["PersistentMemoryState", "candidate_sources", "Update", "Retrieve"]
