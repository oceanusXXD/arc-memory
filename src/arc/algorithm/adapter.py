"""Agent-facing memory adapter.

H may propose an ordered source sequence, but online legality is always
checked against the complete serialized-input cost.  The deterministic
rank-pack is the only fallback and performs the same exact cost checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from arc.agent.memory import MemoryPacket, MemoryRequest
from arc.agent.reader import ClaudeCodeClient
from .architecture import instantiate_structure, normalize_architecture
from .memory import build_once, configured_input_cost, complete_input_cost, memory_packet_from_build, normalize_sources, tokenizer_from_config, _preserve_or_normalize
from .selector import decode_joint, load_selector


def rank_pack(question: str, sources: list[dict[str, Any]], budget: int, *, tokenizer=None,
              config: dict[str, Any] | None = None, architecture: str = "Flat") -> list[dict[str, Any]]:
    """Greedily add sources in fixed retrieval order using complete T."""
    selected: list[dict[str, Any]] = []
    for source in sources:
        candidate = selected + [source]
        cost = (configured_input_cost(config, question, candidate, architecture=architecture)
                if config is not None else complete_input_cost(question, candidate, tokenizer, architecture=architecture))
        if cost <= budget:
            selected.append(source)
    return selected


@dataclass(frozen=True)
class OnlineSelection:
    source_ids: tuple[int, ...]
    budget: int
    method: str
    latency_ms: int = 0
    fallback: str | None = None
    architecture: str = "Flat"
    architecture_score: float = 0.0
    architecture_probabilities: tuple[tuple[str, float], ...] = ()


def select_online(config: dict[str, Any], question: str | None, sources: list[dict[str, Any]], budget: int) -> OnlineSelection:
    normalized, _ = _preserve_or_normalize(sources)
    if not normalized:
        return OnlineSelection((), budget, "content_first", architecture="Flat")
    selector_path = (config.get("selector") or {}).get("checkpoint")
    if not selector_path:
        raise FileNotFoundError("selector.checkpoint is required for the online path")
    selector = load_selector(selector_path)
    input_limit = int((config.get("budgets") or {}).get("builder_input_limit") or 15872)
    selector_cfg = config.get("selector") or {}
    # H is a write-stage policy.  A query is never supplied to its features;
    # ``question`` is retained in the signature for compatibility with old
    # callers and is used only by the later fixed Retrieve/Answer path.
    result = decode_joint(
        selector, None, normalized, budget,
        width=int(selector_cfg.get("beam_width", 4)),
        top_l=int(selector_cfg.get("top_l", selector_cfg.get("beam_width", 4))),
        schema=getattr(selector, "feature_schema", None),
        input_limit=input_limit,
        tokenizer=tokenizer_from_config(config),
        margin=int(selector_cfg.get("budget_margin", 0)),
    )
    chosen = [int(value) for value in result.source_ids]
    if any(value < 1 or value > len(normalized) for value in chosen):
        raise ValueError("selector returned an unknown source id")
    selected = [source for source in normalized if int(source["id"]) in chosen]
    if not result.stopped or configured_input_cost(config, None, selected, architecture=result.architecture) > budget or chosen != sorted(set(chosen)):
        raise ValueError("selector returned an illegal sequence")
    return OnlineSelection(tuple(chosen), budget, "content_first", result.latency_ms,
                           architecture=result.architecture,
                           architecture_score=result.architecture_score,
                           architecture_probabilities=result.architecture_probabilities)


class Memory:
    """MemoryStage implementation for the Agent boundary."""

    def snapshot(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema": "memory",
            "selector": dict(config.get("selector") or {}),
            "compiler": dict(config.get("compiler") or {}),
        }

    def __call__(self, config: dict[str, Any], request: MemoryRequest) -> MemoryPacket:
        if str(getattr(request, "phase", "query")).lower() == "write" or request.write_batch is not None:
            return self.write(config, request)
        # Query time is a fixed read path.  It may retrieve from the supplied
        # persistent state, but it never calls H, Build, or Update.
        from arc.agent.persistent import PersistentMemoryState, Retrieve
        state = request.memory_state if isinstance(request.memory_state, PersistentMemoryState) else PersistentMemoryState.from_dict(request.memory_state)
        entries = Retrieve(request.question, state, top_k=request.top_k)
        text = state.render_entries(entries)
        source_ids = tuple(str(value) for entry in entries for value in entry.get("src", ()))
        return MemoryPacket(
            text,
            {"mode": "read", "schema": "memory", "status": "ACTIVE", "query": request.question,
             "retrieved_entries": len(entries), "selector_called": False, "builder_called": False},
            (), tuple(entries), (), source_ids, state,
        )

    def write(self, config: dict[str, Any], request: MemoryRequest) -> MemoryPacket:
        """Write one query-independent history batch into persistent memory."""
        from arc.agent.persistent import PersistentMemoryState, Update, candidate_sources

        previous = request.memory_state
        state = previous if isinstance(previous, PersistentMemoryState) else PersistentMemoryState.from_dict(previous)
        batch = list(request.write_batch if request.write_batch is not None else request.evidence)
        candidates = candidate_sources(
            batch, state,
            max_sources=int((config.get("retrieval") or {}).get("max_blocks", 64)),
            neighbourhood=int((config.get("persistent_memory") or {}).get("neighbourhood", 16)),
        )
        budget = int((config.get("budgets") or {}).get("builder_input_tokens") or 4096)
        selection = select_online(config, None, candidates, budget)  # type: ignore[arg-type]
        selected = [source for source in candidates if int(source["id"]) in set(selection.source_ids)]
        if not selected:
            next_state = state.clone()
            selected_packet = MemoryPacket("", {"mode": "write", "status": "ACTIVE_EMPTY", "selection": selection.__dict__},
                                            persistent_state=next_state)
            return selected_packet
        client = ClaudeCodeClient(config)
        result = build_once(config, None, selected, client=client, sample_id=request.sample_id, qa_id=request.qa_id,
                            architecture=selection.architecture,
                            structure=instantiate_structure(selection.architecture, selected))
        packet = memory_packet_from_build(result, selected)
        # Persist the candidate records used by this write unit so local
        # selector IDs resolve to stable source IDs even when E_i includes a
        # neighbourhood from M_i^-.
        next_state = Update(state, packet.memories, candidates, architecture=selection.architecture,
                            source_ids=selection.source_ids)
        return MemoryPacket(
            packet.text,
            {**packet.selected, "mode": "write", "selection": selection.__dict__, "update_version": next_state.version},
            packet.usage, packet.memories, packet.errors, packet.evidence_source_ids, next_state,
        )

    def retrieve(self, request: MemoryRequest) -> list[dict[str, Any]]:
        """Fixed query-time retrieval; it never invokes the selector."""
        from arc.agent.persistent import Retrieve, PersistentMemoryState
        state = request.memory_state if isinstance(request.memory_state, PersistentMemoryState) else PersistentMemoryState.from_dict(request.memory_state)
        return Retrieve(request.question, state, top_k=request.top_k)


build_memory = Memory()
