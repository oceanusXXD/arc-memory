"""Content-first source pointer and content-conditioned architecture selector."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import nn

from .architecture import ARCHITECTURES, architecture_id, normalize_architecture, solution_key, solution_record
from .core import STOP, legal_actions, sequence_risk_torch, reachable
from .features import FeatureSchema, FeatureError


class Selector(nn.Module):
    """Two-layer set encoder, source pointer, and conditional architecture head.

    Source decoding intentionally has no architecture input.  The completed
    source beam is then scored by an architecture head conditioned on that
    completed content, implementing the method's ``S`` then ``g`` factorisation.
    """

    implementation = "content_first"

    def __init__(self, width: int = 128, heads: int = 4, ff: int = 512, layers: int = 2,
                 dropout: float = 0.0, input_size: int = 8):
        super().__init__()
        self.width = width
        self.input_size = int(input_size)
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        self.proj = nn.Linear(self.input_size, width)
        block = nn.TransformerEncoderLayer(width, heads, ff, dropout, activation="gelu", norm_first=True,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.state = nn.GRUCell(width, width)
        self.architecture_embedding = nn.Embedding(len(ARCHITECTURES), width)
        self.architecture_head = nn.Linear(width * 2 + 2, len(ARCHITECTURES))
        self.architecture_proj = nn.Linear(width, width, bias=False)
        self.query_init = nn.Linear(width + 2 + width, width)
        self.action_proj = nn.Linear(width, width, bias=False)
        self.state_proj = nn.Linear(width, width, bias=False)
        self.budget_proj = nn.Linear(2, width, bias=False)
        self.dynamic_proj = nn.Linear(4, width, bias=False)
        self.stop_embedding = nn.Parameter(torch.zeros(width))
        self.score = nn.Linear(width, 1, bias=False)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        batched = features.dim() == 3
        encoded = self.encoder(self.proj(features.unsqueeze(0) if not batched else features))
        return encoded if batched else encoded.squeeze(0)

    def architecture_logits(self, encoded: torch.Tensor, budget_features: torch.Tensor | None = None,
                            selected_ids: Sequence[int] | None = None) -> torch.Tensor:
        if budget_features is None:
            budget_features = encoded.new_zeros(2)
        budget_features = budget_features.to(device=encoded.device, dtype=encoded.dtype)
        all_context = encoded.mean(dim=0)
        if selected_ids:
            indices = [int(value) - 1 for value in selected_ids if 1 <= int(value) <= encoded.shape[0]]
            selected_context = encoded[indices].mean(dim=0) if indices else encoded.new_zeros(encoded.shape[1])
        else:
            selected_context = encoded.new_zeros(encoded.shape[1])
        return self.architecture_head(torch.cat((all_context, selected_context, budget_features)))

    def logits(self, encoded: torch.Tensor, state: torch.Tensor,
               actions: list[int | str], budget_features: torch.Tensor | None = None,
               dynamic_features: Mapping[int | str, torch.Tensor] | None = None,
               architecture: str | None = None) -> dict[int | str, torch.Tensor]:
        if budget_features is None:
            budget_features = encoded.new_zeros(2)
        budget_features = budget_features.to(device=encoded.device, dtype=encoded.dtype)
        state_term = self.state_proj(state)
        architecture_term = encoded.new_zeros(encoded.shape[1])
        if architecture is not None:
            architecture_term = self.architecture_proj(
                self.architecture_embedding.weight[architecture_id(architecture)].to(
                    device=encoded.device, dtype=encoded.dtype
                )
            )

        def score_action(representation: torch.Tensor) -> torch.Tensor:
            hidden = torch.tanh(self.action_proj(representation) + state_term +
                                self.budget_proj(budget_features) + architecture_term)
            return self.score(hidden).squeeze(-1)

        result: dict[int | str, torch.Tensor] = {}
        for action in actions:
            if action == STOP:
                result[action] = score_action(self.stop_embedding)
            else:
                value = score_action(encoded[int(action) - 1])
                if dynamic_features and action in dynamic_features:
                    dynamic = torch.tanh(self.dynamic_proj(dynamic_features[action].to(device=encoded.device, dtype=encoded.dtype)))
                    value = value + self.score(dynamic).squeeze(-1)
                result[action] = value
        return result


def _source_embedding_vectors(sources: list[Mapping[str, Any]]) -> list[torch.Tensor | None]:
    vectors: list[torch.Tensor | None] = []
    for index, item in enumerate(sources, 1):
        value = item.get("embedding")
        if value is None:
            vectors.append(None)
            continue
        try:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise FeatureError(f"invalid frozen embedding for source {index}") from exc
        if tensor.ndim != 1 or tensor.numel() == 0 or not torch.isfinite(tensor).all():
            raise FeatureError(f"invalid frozen embedding for source {index}")
        norm = float(torch.linalg.vector_norm(tensor))
        if norm == 0:
            raise FeatureError(f"zero frozen embedding for source {index}")
        vectors.append(tensor / norm)
    return vectors


def _dynamic_features(prefix: tuple[int, ...], source_id: int, sources: list[Mapping[str, Any]],
                      budget: int, question: str,
                      source_vectors: list[torch.Tensor | None] | None = None) -> torch.Tensor:
    chosen = [sources[index - 1] for index in prefix]
    source = sources[source_id - 1]
    vectors = source_vectors if source_vectors is not None else _source_embedding_vectors(sources)
    chosen_vectors = [vectors[index - 1] for index in prefix if vectors[index - 1] is not None]
    similarity = 0.0
    source_vec = vectors[source_id - 1]
    if source_vec is not None:
        similarity = max((float(torch.dot(source_vec, item)) for item in chosen_vectors), default=0.0)
    same_session = float(any(source.get("session") is not None and source.get("session") == item.get("session") for item in chosen))
    return torch.tensor([float(max(budget, 0)) / max(float(budget), 1.0), similarity, same_session,
                         float(len(prefix)) / max(len(sources), 1)], dtype=torch.float32)


@dataclass(frozen=True)
class DecodeResult:
    source_ids: tuple[int, ...]
    score: float
    stopped: bool
    latency_ms: int
    architecture: str = "Flat"
    architecture_score: float = 0.0
    architecture_probabilities: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ArchitectureDecodeResult:
    architecture: str
    score: float
    probabilities: tuple[tuple[str, float], ...]
    latency_ms: int


def _feature_tensor(question: str | None, sources: list[Mapping[str, Any]], budget: int,
                    schema: FeatureSchema | None = None) -> torch.Tensor:
    if schema is None:
        raise FeatureError("fitted_feature_schema_required")
    # Enforce the write-stage information boundary at the selector boundary,
    # even for legacy callers that still pass a question argument.
    features, _ = schema.transform(None, list(sources), budget=budget)
    return features


def _budget_features(prefix: tuple[int, ...], question: str, sources: list[Mapping[str, Any]], budget: int,
                     input_limit: int | None = None, tokenizer: Any = None,
                     cost_fn: Any = None) -> torch.Tensor:
    current = int(cost_fn(prefix)) if cost_fn is not None else _cost(prefix, question, sources, tokenizer)
    limit = max(int(input_limit or 15872), 1)
    return torch.tensor([float(budget) / limit, max(float(budget - current), 0.0) / max(float(budget), 1.0)], dtype=torch.float32)


def _initial_state(model: Selector, encoded: torch.Tensor, budget: int, input_limit: int,
                   architecture: str | None = None) -> torch.Tensor:
    budget_vec = encoded.new_tensor([float(budget) / max(input_limit, 1), 1.0])
    # The source stage is architecture agnostic.  A zero vector is used for
    # its optional embedding slot; a concrete architecture is only injected
    # when a caller explicitly requests a legacy architecture-conditioned
    # source decode.
    architecture_vec = encoded.new_zeros(encoded.shape[1])
    if architecture is not None:
        architecture_vec = model.architecture_embedding.weight[architecture_id(architecture)].to(
            device=encoded.device, dtype=encoded.dtype
        )
    return torch.tanh(model.query_init(torch.cat((encoded.mean(dim=0), budget_vec, architecture_vec))))


def _cached_complete_cost(question: str | None, sources: list[Mapping[str, Any]], tokenizer: Any = None,
                          architecture: str | None = "Flat") -> Callable[[Iterable[int]], int]:
    cache: dict[tuple[int, ...], int] = {}

    def cost(values: Iterable[int]) -> int:
        key = tuple(sorted(int(value) for value in values))
        if key not in cache:
            cache[key] = _cost(key, question, sources, tokenizer, architecture)
        return cache[key]

    return cost


def _prefix_state(model: Selector, encoded: torch.Tensor, initial_state: torch.Tensor,
                  prefix: tuple[int, ...], cache: dict[tuple[int, ...], torch.Tensor] | None = None) -> torch.Tensor:
    if cache is not None:
        cached = cache.get(prefix)
        if cached is not None:
            return cached
    if prefix:
        state = model.state(encoded[prefix[-1] - 1], _prefix_state(model, encoded, initial_state, prefix[:-1], cache))
    else:
        state = initial_state
    if cache is not None:
        cache[prefix] = state
    return state


def _candidate_logits(model: Selector, question: str | None, sources: list[Mapping[str, Any]], budget: int,
                      prefix: tuple[int, ...], schema: FeatureSchema | None = None,
                      input_limit: int | None = None, tokenizer: Any = None,
                      architecture: str | None = None) -> dict[int | str, float]:
    features = _feature_tensor(question, sources, budget, schema)
    encoded = model.encode(features)
    input_limit = int(input_limit or 15872)
    state = _prefix_state(model, encoded, _initial_state(model, encoded, budget, input_limit, architecture), prefix)
    cost_fn = _cached_complete_cost(question, sources, tokenizer, architecture)
    source_vectors = _source_embedding_vectors(sources)
    actions = legal_actions(prefix, range(1, len(sources) + 1), cost_fn, budget)
    with torch.no_grad():
        budget_features = _budget_features(prefix, question, sources, budget, input_limit, tokenizer, cost_fn=cost_fn)
        dynamic = {action: _dynamic_features(prefix, int(action), sources, budget, question, source_vectors)
                   for action in actions if action != STOP}
        return {action: float(value) for action, value in model.logits(encoded, state, actions, budget_features, dynamic, architecture).items()}


def _cost(values: Iterable[int], question: str | None, sources: list[Mapping[str, Any]], tokenizer: Any = None,
          architecture: str | None = "Flat") -> int:
    selected = [sources[index - 1] for index in values]
    if not selected:
        return 0
    from .memory import complete_input_cost
    if architecture is None:
        return max(complete_input_cost(question, selected, tokenizer, architecture=name) for name in ARCHITECTURES)
    return complete_input_cost(question, selected, tokenizer, architecture=architecture)


def conservative_input_cost(question: str | None, sources: list[Mapping[str, Any]], tokenizer: Any = None) -> int:
    r"""Return the architecture-independent source-stage upper bound \hat T."""
    return _cost(range(1, len(sources) + 1), question, sources, tokenizer, architecture=None)


def _domain_cost_lookup(row: Mapping[str, Any], architecture: str = "Flat") -> dict[tuple[int, ...], int]:
    architecture = normalize_architecture(architecture)
    return {
        tuple(sorted(int(value) for value in item.get("source_ids") or [])): int(item["cost"])
        for item in row.get("domain") or []
        if item.get("cost") is not None and normalize_architecture(item.get("architecture", "Flat")) == architecture
    }


def _record_cost(row: Mapping[str, Any], values: Iterable[int], question: str,
                 sources: list[Mapping[str, Any]], tokenizer: Any = None,
                 lookup: Mapping[tuple[int, ...], int] | None = None,
                 cache: dict[tuple[int, ...], int] | None = None,
                 architecture: str = "Flat") -> int:
    """Use the compiler's exact cached candidate cost when available.

    Prefixes that never appeared in the compilation record (for example the
    empty prefix or a singleton the compiler pruned) still need an exact
    complete-input cost, so the frozen builder tokenizer is threaded through
    from the training entry point.
    """
    key = tuple(sorted(int(value) for value in values))
    if cache is not None and key in cache:
        return cache[key]
    lookup = lookup if lookup is not None else _domain_cost_lookup(row, architecture)
    if key in lookup:
        value = lookup[key]
    else:
        value = _cost(key, question, sources, tokenizer, architecture)
    if cache is not None:
        cache[key] = value
    return value


def _feasible_architectures(question: str | None, sources: list[Mapping[str, Any]], selected_ids: Sequence[int],
                            budget: int, tokenizer: Any, input_limit: int) -> tuple[str, ...]:
    if not selected_ids:
        return ("Flat",)
    if tokenizer is None:
        return ARCHITECTURES
    selected = [sources[int(value) - 1] for value in selected_ids]
    return tuple(name for name in ARCHITECTURES
                 if _cost(selected_ids, question, sources, tokenizer, name) <= int(budget)
                 and _cost(selected_ids, question, sources, tokenizer, name) <= int(input_limit))


def decode_architecture(model: Selector, question: str | None, sources: list[Mapping[str, Any]], budget: int,
                        schema: FeatureSchema | None = None, input_limit: int | None = None,
                        selected_ids: Sequence[int] | None = None, tokenizer: Any = None) -> ArchitectureDecodeResult:
    """Score ``g`` conditional on a completed source set ``S``.

    ``selected_ids`` defaults to all sources for backward-compatible callers;
    the deployment path always supplies the source-beam completion.
    """
    started = time.perf_counter()
    input_limit = int(input_limit or 15872)
    selected = tuple(sorted(int(value) for value in (selected_ids if selected_ids is not None else range(1, len(sources) + 1))))
    with torch.no_grad():
        features = _feature_tensor(question, sources, budget, schema)
        encoded = model.encode(features)
        # The architecture head receives the same conservative source-stage
        # budget feature as Eq. (18), while exact per-architecture costs are
        # used only for its feasibility mask.
        source_cost = _cached_complete_cost(question, sources, tokenizer, None) if tokenizer is not None else None
        budget_features = _budget_features(selected, question, sources, budget, input_limit, tokenizer,
                                            cost_fn=source_cost) if source_cost is not None else encoded.new_tensor([float(budget) / max(input_limit, 1), 1.0])
        logits = model.architecture_logits(encoded, budget_features, selected)
        feasible = _feasible_architectures(question, sources, selected, budget, tokenizer, input_limit)
        mask = torch.full_like(logits, float("-inf"))
        for name in feasible:
            mask[architecture_id(name)] = logits[architecture_id(name)]
        log_probs = torch.log_softmax(mask, dim=0)
        probabilities = torch.softmax(mask, dim=0)
        index = int(torch.argmax(mask).item())
    return ArchitectureDecodeResult(
        architecture=ARCHITECTURES[index],
        score=float(log_probs[index].item()),
        probabilities=tuple((name, float(probabilities[i].item())) for i, name in enumerate(ARCHITECTURES)),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def decode_source_beam(model: Selector, question: str | None, sources: list[Mapping[str, Any]], budget: int,
                       width: int = 4, top_l: int | None = None, schema: FeatureSchema | None = None,
                       input_limit: int | None = None, tokenizer: Any = None, margin: int = 0,
                       architecture: str | None = None) -> list[DecodeResult]:
    """Decode source sets before any architecture is chosen."""
    started = time.perf_counter()
    input_limit = int(input_limit or 15872)
    effective_budget = max(0, int(budget) - max(0, int(margin)))
    active: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    complete: list[tuple[tuple[int, ...], float]] = []
    features = _feature_tensor(question, sources, effective_budget, schema)
    encoded = model.encode(features)
    # An explicit architecture is retained only for legacy diagnostics.  The
    # persistent deployment path passes None and therefore has no g signal.
    source_architecture = normalize_architecture(architecture) if architecture is not None else None
    initial_state = _initial_state(model, encoded, effective_budget, input_limit, source_architecture)
    state_cache: dict[tuple[int, ...], torch.Tensor] = {(): initial_state}
    cost_fn = _cached_complete_cost(question, sources, tokenizer, source_architecture)
    source_vectors = _source_embedding_vectors(sources)
    with torch.no_grad():
        for _ in range(len(sources) + 1):
            next_active: list[tuple[tuple[int, ...], float]] = []
            for prefix, score in active:
                state = _prefix_state(model, encoded, initial_state, prefix, state_cache)
                actions = legal_actions(prefix, range(1, len(sources) + 1), cost_fn, effective_budget)
                if not actions:
                    continue
                budget_features = _budget_features(prefix, question, sources, effective_budget, input_limit, tokenizer, cost_fn=cost_fn)
                dynamic = {action: _dynamic_features(prefix, int(action), sources, effective_budget, question or "", source_vectors)
                           for action in actions if action != STOP}
                raw = model.logits(encoded, state, actions, budget_features, dynamic, source_architecture)
                normalizer = torch.logsumexp(torch.stack(list(raw.values())), 0).item()
                for action, value in raw.items():
                    candidate_score = score + float(value) - normalizer
                    if action == STOP:
                        complete.append((prefix, candidate_score))
                    else:
                        next_active.append((prefix + (int(action),), candidate_score))
            active = sorted(next_active, key=lambda item: (-item[1], item[0]))[: max(1, int(width))]
            if not active:
                break
    complete = sorted(complete, key=lambda item: (-item[1], item[0]))[: max(1, int(top_l or width))]
    latency = int((time.perf_counter() - started) * 1000)
    return [DecodeResult(ids, score, True, latency, architecture="Flat") for ids, score in complete]


def decode_sources(model: Selector, question: str | None, sources: list[Mapping[str, Any]], budget: int, width: int = 4,
                   schema: FeatureSchema | None = None, input_limit: int | None = None,
                   tokenizer: Any = None, architecture: str | None = None, margin: int = 0) -> DecodeResult:
    """Compatibility wrapper returning the highest-scoring source completion."""
    results = decode_source_beam(model, question, sources, budget, width=width, top_l=1, schema=schema,
                                 input_limit=input_limit, tokenizer=tokenizer, architecture=architecture, margin=margin)
    if not results:
        return DecodeResult((), float("-inf"), False, 0, architecture=normalize_architecture(architecture) if architecture else "Flat")
    return results[0]


def decode_joint(model: Selector, question: str | None, sources: list[Mapping[str, Any]], budget: int, width: int = 4,
                 top_l: int | None = None, schema: FeatureSchema | None = None, input_limit: int | None = None,
                 tokenizer: Any = None, margin: int = 0) -> DecodeResult:
    """Decode Top-L source sets, then choose the highest joint ``S,g``."""
    started = time.perf_counter()
    candidates = decode_source_beam(model, question, sources, budget, width=width, top_l=top_l or width,
                                    schema=schema, input_limit=input_limit, tokenizer=tokenizer, margin=margin)
    scored: list[DecodeResult] = []
    for source_result in candidates:
        arch = decode_architecture(model, question, sources, budget, schema, input_limit,
                                   selected_ids=source_result.source_ids, tokenizer=tokenizer)
        scored.append(DecodeResult(source_result.source_ids,
                                   source_result.score + arch.score, True,
                                   int((time.perf_counter() - started) * 1000),
                                   architecture=arch.architecture,
                                   architecture_score=arch.score,
                                   architecture_probabilities=arch.probabilities))
    if not scored:
        return DecodeResult((), float("-inf"), False, int((time.perf_counter() - started) * 1000))
    return max(scored, key=lambda item: (item.score, tuple(-value for value in item.source_ids)))


def sequence_loss(logps: torch.Tensor, terminal_costs: Iterable[int], budget: int) -> torch.Tensor:
    return sequence_risk_torch(logps, list(terminal_costs), budget)


def _structured_solutions(row: Mapping[str, Any], budget_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read joint positive labels, with Flat as the canonical empty label."""
    values = budget_row.get("successful_solutions")
    if values is None:
        values = row.get("successful_solutions")
    if values is None:
        values = [solution_record("Flat", source_ids) for source_ids in
                  (budget_row.get("successful_sets") if "successful_sets" in budget_row
                   else row.get("successful_sets") or [])]
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for value in values or []:
        if isinstance(value, Mapping):
            item = solution_record(value.get("architecture", "Flat"), value.get("source_ids") or value.get("sources") or [],
                                   cost=value.get("cost"), status=value.get("status"), utility=value.get("utility"))
        else:
            item = solution_record("Flat", value)
        key = solution_key(item)
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _atomic_torch_save(record, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(record, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint(path: str | Path, model: Selector, schema: FeatureSchema | None = None, metadata=None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "implementation": Selector.implementation,
        "architectures": list(ARCHITECTURES),
        "width": model.width,
        "input_size": model.input_size,
        "heads": 4,
        "ff": 512,
        "layers": 2,
        "state_dict": model.state_dict(),
    }
    if schema is not None:
        record["feature_schema"] = schema.to_dict()
    if metadata:
        record["training"] = metadata
    _atomic_torch_save(record, target)


def load_selector(path: str | Path) -> Selector:
    record = torch.load(path, map_location="cpu", weights_only=False)
    if record.get("implementation") != Selector.implementation:
        raise RuntimeError("selector checkpoint is obsolete; retrain joint selector")
    if tuple(record.get("architectures") or ()) != ARCHITECTURES:
        raise RuntimeError("selector checkpoint has no matching architecture head")
    model = Selector(width=int(record.get("width", 128)), heads=int(record.get("heads", 4)),
                     ff=int(record.get("ff", 512)), layers=int(record.get("layers", 2)),
                     input_size=int(record.get("input_size", 8)))
    model.load_state_dict(record["state_dict"])
    schema_record = record.get("feature_schema")
    model.feature_schema = FeatureSchema.from_dict(schema_record) if schema_record else None
    model.eval()
    return model


def train_selector(records: Iterable[Mapping[str, Any]], output: str | Path, *, epochs: int = 30,
                   learning_rate: float = 3e-4, width: int = 128, weight_decay: float = 0.01,
                   gradient_clip: float = 1.0, schema: FeatureSchema | None = None,
                   tokenizer: Any = None, seed: int = 7, resume: bool = False,
                   input_sha256: str | None = None, progress_callback=None) -> dict[str, Any]:
    if iter(records) is records:
        records = list(records)
    random.seed(seed)
    torch.manual_seed(seed)
    if schema is None:
        schema_record = next((row.get("feature_schema") for row in records if row.get("feature_schema")), None)
        if schema_record is not None:
            schema = FeatureSchema.from_dict(schema_record)
    input_size = schema.input_size if schema is not None else 8
    model = Selector(width=width, input_size=input_size, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    updates = 0
    losses: list[float] = []
    settings = {"seed": seed, "epochs": epochs, "learning_rate": learning_rate, "width": width,
                "weight_decay": weight_decay, "gradient_clip": gradient_clip,
                "input_sha256": input_sha256, "feature_schema": schema.to_dict() if schema else None}
    progress_path = Path(str(output) + ".training.pt")
    cursor = (-1, -1, -1)
    if Path(output).exists():
        existing = torch.load(output, map_location="cpu", weights_only=True)
        training = existing.get("training") or {}
        if resume and training.get("settings") == settings and training.get("complete"):
            return {"implementation": Selector.implementation, "output": str(output), "updates": training["updates"], "mean_loss": training.get("mean_loss"), "skipped": True}
        raise FileExistsError("Existing checkpoint has different or unverified training provenance; preserved")
    if resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=True)
        if progress["settings"] != settings:
            raise ValueError("training resume settings differ")
        model.load_state_dict(progress["model"])
        optimizer.load_state_dict(progress["optimizer"])
        torch.set_rng_state(progress["rng_state"])
        updates, losses, cursor = progress["updates"], progress["losses"], tuple(progress["cursor"])
    for epoch in range(epochs):
        for row_index, row in enumerate(records):
            sources = list(row.get("sources") or [])
            budget_records = row.get("budgets") or {}
            if budget_records:
                budget_items = [(int(k), v) for k, v in budget_records.items()]
            else:
                budget_items = [(int(row.get("budget") or 0), {"successful_sets": row.get("successful_sets") or []})]
            if not sources:
                continue
            for budget_index, (budget, budget_row) in enumerate(budget_items):
                position = (epoch, row_index, budget_index)
                if position <= cursor:
                    continue
                # H is trained for write-time decisions and therefore never
                # receives the QA question or a cached query vector.
                question = None
                if budget <= 0:
                    continue
                solutions = _structured_solutions(row, budget_row)
                if not solutions:
                    continue
                features = _feature_tensor(question, sources, budget, schema)
                encoded = model.encode(features)
                source_vectors = _source_embedding_vectors(sources)
                logps = []
                terminal_costs = []
                seen = set()
                for solution in solutions:
                    architecture = normalize_architecture(solution["architecture"])
                    terminal = frozenset(int(value) for value in solution["source_ids"])
                    cost_lookup = _domain_cost_lookup(row, architecture)
                    cost_cache: dict[tuple[int, ...], int] = {}

                    def exact_cost(values: Iterable[int]) -> int:
                        return _record_cost(row, values, question, sources, tokenizer, cost_lookup, cost_cache, architecture)

                    source_cost_cache: dict[tuple[int, ...], int] = {}

                    def source_cost(values: Iterable[int]) -> int:
                        key = tuple(sorted(int(value) for value in values))
                        if key not in source_cost_cache:
                            if tokenizer is None:
                                # Small deterministic fixtures may carry the
                                # compiler's exact costs without a tokenizer.
                                # Production runs always take the exact frozen
                                # tokenizer branch.
                                costs = [
                                    _domain_cost_lookup(row, name).get(key)
                                    for name in ARCHITECTURES
                                ]
                                known = [value for value in costs if value is not None]
                                if not known and key:
                                    raise RuntimeError("frozen builder tokenizer is required for unseen prefix cost")
                                source_cost_cache[key] = max(known, default=0)
                            else:
                                source_cost_cache[key] = _cost(key, question, sources, tokenizer, architecture=None)
                        return source_cost_cache[key]

                    solution_key_value = (architecture, terminal)
                    if solution_key_value in seen or not reachable(terminal, exact_cost, budget):
                        continue
                    seen.add(solution_key_value)
                    initial_state = _initial_state(model, encoded, budget, 15872, None)
                    state_cache: dict[tuple[int, ...], torch.Tensor] = {(): initial_state}
                    transition_cache: dict[tuple[int, ...], dict[int | str, torch.Tensor]] = {}

                    def transition_logps(prefix: tuple[int, ...]) -> dict[int | str, torch.Tensor]:
                        cached = transition_cache.get(prefix)
                        if cached is not None:
                            return cached
                        state = _prefix_state(model, encoded, initial_state, prefix, state_cache)
                        actions = legal_actions(prefix, range(1, len(sources) + 1), source_cost, budget)
                        dynamic = {action: _dynamic_features(prefix, int(action), sources, budget, question, source_vectors)
                                   for action in actions if action != STOP}
                        logits = model.logits(
                            encoded, state, actions,
                            _budget_features(prefix, question, sources, budget, cost_fn=source_cost), dynamic,
                            None,
                        )
                        normalizer = torch.logsumexp(torch.stack(list(logits.values())), 0)
                        cached = {action: value - normalizer for action, value in logits.items()}
                        transition_cache[prefix] = cached
                        return cached

                    prefix = tuple(sorted(terminal))
                    arch_budget_features = _budget_features(prefix, question, sources, budget, cost_fn=source_cost)
                    arch_logits = model.architecture_logits(encoded, arch_budget_features, prefix)
                    feasible = _feasible_architectures(question, sources, prefix, budget, tokenizer, 15872)
                    arch_mask = torch.full_like(arch_logits, float("-inf"))
                    for name in feasible:
                        arch_mask[architecture_id(name)] = arch_logits[architecture_id(name)]
                    architecture_logps = torch.log_softmax(arch_mask, dim=0)
                    value = architecture_logps[architecture_id(architecture)]
                    for index in range(len(prefix) + 1):
                        current = prefix[:index]
                        value = value + transition_logps(current)[prefix[index] if index < len(prefix) else STOP]
                    logps.append(value)
                    terminal_costs.append(exact_cost(terminal))
                if not logps:
                    continue
                loss = sequence_loss(torch.stack(logps), terminal_costs, budget)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                updates += 1; losses.append(float(loss.detach()))
                if resume:
                    _atomic_torch_save({"settings": settings, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                        "rng_state": torch.get_rng_state(), "updates": updates, "losses": losses,
                                        "cursor": position}, progress_path)
                if progress_callback:
                    progress_callback(epoch=epoch + 1, row_index=row_index, budget=budget, updates=updates)
    metadata = {"settings": settings, "complete": True, "updates": updates, "mean_loss": sum(losses) / len(losses) if losses else None}
    _checkpoint(output, model, schema, metadata=metadata)
    return {"implementation": Selector.implementation, "output": str(output), "updates": updates, "mean_loss": sum(losses) / len(losses) if losses else None}


def train_selector_file(input_path: str | Path, output: str | Path, *, epochs: int = 30,
                        learning_rate: float = 3e-4, width: int = 128,
                        config: Mapping[str, Any] | None = None, resume: bool = False,
                        progress_callback=None) -> dict[str, Any]:
    from arc.agent.io import read_jsonl
    from .memory import tokenizer_from_config
    class Records:
        def __iter__(self):
            return iter(read_jsonl(input_path))
    records = Records()
    first = next(iter(records), None)
    if first is None:
        raise ValueError("selector training input is empty")
    # Training evaluates exact complete-input costs for every prefix it walks,
    # so it must use the same frozen builder tokenizer as deployment.
    dimension = int(((first.get("sources") or [{}])[0]
                     .get("embedding_metadata") or {}).get("dimension") or 1024)
    schema = FeatureSchema.fit(records, dimension=dimension)
    tokenizer = tokenizer_from_config(config) if config is not None else None
    result = train_selector(records, output, epochs=epochs, learning_rate=learning_rate,
                            width=width, schema=schema, tokenizer=tokenizer,
                            seed=int((config or {}).get("seed", 7)), resume=resume,
                            input_sha256=__import__('arc.agent.config', fromlist=['file_sha256']).file_sha256(input_path),
                            progress_callback=progress_callback)
    result["feature_schema"] = schema.to_dict()["schema"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the sequence selector from compilation records")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--config", default="configs/locomo.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from arc.agent.config import load_config
    print(json.dumps(train_selector_file(args.input, args.output, epochs=args.epochs,
                                         learning_rate=args.learning_rate, width=args.width,
                                         config=load_config(args.config), resume=args.resume),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
