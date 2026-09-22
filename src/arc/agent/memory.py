"""Memory-stage contract shared by agents, runners, and independent plugins."""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Any, Protocol

@dataclass(frozen=True)
class MemoryPacket:
    text: str
    selected: dict[str, Any]
    usage: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    memories: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    errors: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    evidence_source_ids: tuple[str, ...] = field(default_factory=tuple)
    # A write-stage packet may carry the resulting persistent state to the
    # caller.  The field is optional so existing query-scoped arms remain
    # source-compatible.
    persistent_state: Any = None

    @property
    def mode(self) -> str:
        return str(self.selected.get("mode") or "raw")


@dataclass(frozen=True)
class MemoryRequest:
    question: str
    evidence: list[dict]
    arm: str = "our"
    sample_id: str = ""
    qa_id: str = ""
    # ``phase=write`` consumes ``write_batch`` and ``memory_state``.  Query
    # requests use the fixed Retrieve path and never call the selector.
    phase: str = "query"
    memory_state: Any = None
    write_batch: list[dict] | None = None
    query_time: Any = None
    top_k: int = 8


class MemoryStage(Protocol):
    def __call__(self, config: dict[str, Any], request: MemoryRequest) -> MemoryPacket: ...


def prepare_memory(config: dict[str, Any], request: MemoryRequest, stage: MemoryStage) -> MemoryPacket:
    # Plugins receive a private evidence copy, never reference labels or QA scores.
    packet = stage(copy.deepcopy(config), copy.deepcopy(request))
    if not isinstance(packet, MemoryPacket):
        raise TypeError("memory stage must return MemoryPacket")
    return packet


def render_sources(blocks: list[dict], title: str="Evidence") -> str:
    lines = [f"{title}:"]
    for rank, block in enumerate(blocks, 1): lines.append(f"[{rank}] source_id={block['source_id']} dia_id={block.get('dia_id','')} date={block.get('session_datetime') or 'unknown time'} speaker={block.get('speaker','')}: {block.get('text','')}")
    return "\n".join(lines)

def render_memory_text(evidence: list[dict], hints: str = "", prefix: str | None = None) -> str:
    return "\n".join([
        "Task-scoped memory packet.",
        "Use only this packet as supplied memory for the current task.",
        *([prefix.strip()] if prefix else []),
        render_sources(evidence, "Original evidence") if evidence else "Original evidence: (empty)",
        "Relationship hints:\n" + hints if hints else "Relationship hints: (none)",
    ])
