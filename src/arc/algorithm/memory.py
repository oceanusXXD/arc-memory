"""Memory construction and tri-state auditing.

The online path receives only the query and retrieved sources.  Offline
annotation and audit records are represented by the same small dataclasses,
which keeps source provenance and measured builder usage explicit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from arc.agent.memory import MemoryPacket
from arc.agent.reader import ClaudeCodeClient, completion_error_row
from .architecture import architecture_contract, instantiate_structure, normalize_architecture, validate_structure
from .core import FAIL, PASS, UNKNOWN

SCHEMA = "memory"
MAX_MEMORIES = 6
MAX_OUTPUT_TOKENS = 1024
_TOKENIZER_CACHE: dict[tuple[str, str | None], Callable[[str], int] | None] = {}
_TOKENIZER_OBJECT_CACHE: dict[tuple[str, str | None], Any | None] = {}


@dataclass(frozen=True)
class RequirementAnnotation:
    requirements: tuple[dict[str, Any], ...]
    packages: tuple[tuple[frozenset[int], ...], ...]
    valid: bool = True
    reason: str | None = None
    usage: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AuditResult:
    status: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    requirement_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BuildResult:
    selected: frozenset[int]
    raw: str
    memories: tuple[dict[str, Any], ...]
    source_audit: AuditResult
    requirement_audit: AuditResult
    status: str
    usage: tuple[dict[str, Any], ...]
    error: str | None = None
    architecture: str = "Flat"
    structure: dict[str, Any] = field(default_factory=dict)
    # Kept at the end for positional compatibility with existing callers.
    # Structural validity is a separate certification layer from source
    # faithfulness and requirement completeness.
    structure_audit: AuditResult = field(default_factory=lambda: AuditResult(PASS))


def normalize_sources(sources: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Normalize a retrieved evidence list once at the retrieval boundary.

    The internal id is assigned in retrieval order and source_id remains the
    dataset id. Downstream subset builders pass these rows through unchanged.
    """
    normalized: list[dict[str, Any]] = []
    original: dict[int, str] = {}
    for index, source in enumerate(sources, 1):
        item = dict(source)
        internal = index
        original[internal] = str(item.get("source_id", item.get("id", internal)))
        item["id"] = internal
        item["source_id"] = original[internal]
        item.setdefault("session", item.get("session_id"))
        item.setdefault("position", item.get("position_in_session", index))
        item.setdefault("retrieval_score", item.get("score", 0.0))
        item["text"] = str(item.get("text") or "")
        normalized.append(item)
    if len(normalized) > 64:
        raise ValueError("at most 64 retrieved sources are supported")
    return normalized, original


def _preserve_or_normalize(sources: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, str]]:
    rows = [dict(source) for source in sources]
    ids: list[int] = []
    for row in rows:
        try:
            ids.append(int(row.get("id")))
        except (TypeError, ValueError):
            ids = []
            break
    if ids and len(ids) == len(set(ids)) and all(value > 0 for value in ids):
        original: dict[int, str] = {}
        preserved: list[dict[str, Any]] = []
        for index, row in zip(ids, rows):
            row["id"] = index
            row["source_id"] = str(row.get("source_id", index))
            row.setdefault("session", row.get("session_id"))
            row.setdefault("position", row.get("position_in_session", index))
            row.setdefault("retrieval_score", row.get("score", 0.0))
            row["text"] = str(row.get("text") or "")
            original[index] = row["source_id"]
            preserved.append(row)
        return preserved, original
    return normalize_sources(rows)


def serialize_input(question: str | None, sources: Iterable[Mapping[str, Any]], architecture: str = "Flat",
                    structure: Mapping[str, Any] | None = None) -> str:
    """Canonical G input envelope used for exact full-input token counting."""
    rows = []
    for source in sorted(sources, key=lambda item: int(item["id"])):
        rows.append({
            "id": int(source["id"]),
            "source_id": str(source.get("source_id", source["id"])),
            "session": source.get("session"),
            "speaker": source.get("speaker"),
            "time": source.get("time", source.get("session_datetime")),
            "position": source.get("position", source.get("position_in_session")),
            "source_offset": source.get("source_offset", source.get("offset")),
            "text": str(source.get("text") or ""),
        })
    kind = normalize_architecture(architecture)
    topology = dict(structure or instantiate_structure(kind, rows))
    envelope = {
        "schema": SCHEMA,
        "architecture": kind,
        "architecture_contract": architecture_contract(kind),
        "structure": topology,
        "system": (
            "Extract only source-supported task memory. Preserve dates, negation and exceptions. "
            "Return only a valid JSON array, with no explanation or markdown. Each item must be "
            "an object with exactly the keys m (string) and src (integer array)."
        ),
        "sources": rows,
        "output": {"type": "array", "item": {"m": "string", "src": "integer[]"}, "max_items": MAX_MEMORIES},
    }
    # The builder is a write-stage component.  ``question`` remains in the
    # compatibility signature for callers that also need it for auditing, but
    # it is never serialized into G's input.  This is the method's strict
    # query-independent write boundary.
    envelope["write_stage"] = True
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def complete_input_cost(question: str | None, sources: Iterable[Mapping[str, Any]], tokenizer: Callable[[str], int] | None = None,
                        architecture: str = "Flat", structure: Mapping[str, Any] | None = None) -> int:
    items = list(sources)
    if not items:
        return 0
    # Count exactly the content sent to G, including the instruction prefix.
    serialized = _builder_content(question, items, architecture=architecture, structure=structure)
    if tokenizer is None:
        raise RuntimeError("frozen builder tokenizer is required for exact input cost")
    return int(tokenizer(serialized))


def configured_input_cost(config: Mapping[str, Any], question: str | None,
                          sources: Iterable[Mapping[str, Any]], architecture: str = "Flat",
                          structure: Mapping[str, Any] | None = None) -> int:
    """Count a complete input with the configured builder tokenizer.

    Exact costs use the configured frozen builder tokenizer.
    """
    return complete_input_cost(question, sources, tokenizer_from_config(config), architecture, structure)


def tokenizer_from_config(config: Mapping[str, Any]) -> Callable[[str], int] | None:
    spec = dict(config.get("tokenizer") or {})
    if not bool(spec.get("enabled", False)):
        raise RuntimeError("tokenizer.enabled must be true for exact memory costs")
    model = str(spec.get("model") or (config.get("models") or {}).get("builder") or "")
    revision = spec.get("revision")
    key = (model, str(revision) if revision else None)
    if key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[key]
    tokenizer = _load_tokenizer(config, key, model, revision, spec)
    if tokenizer is not None:
        def count(text: str) -> int:
            kwargs = {"tokenize": True, "add_generation_prompt": True}
            template_kwargs = dict(spec.get("chat_template_kwargs") or {})
            if template_kwargs:
                kwargs["chat_template_kwargs"] = template_kwargs
            encoded = tokenizer.apply_chat_template([{"role": "user", "content": text}], **kwargs)
            # Hugging Face returns a BatchEncoding mapping rather than a
            # plain dict for tokenized chat templates. Count input_ids,
            # otherwise len(encoded) counts mapping keys (usually 2).
            if isinstance(encoded, Mapping) or hasattr(encoded, "input_ids"):
                encoded = encoded["input_ids"]
            return len(encoded)
        _TOKENIZER_CACHE[key] = count
    else:
        raise RuntimeError(f"unable to load frozen builder tokenizer: {model}")
    return _TOKENIZER_CACHE[key]


def _load_tokenizer(config: Mapping[str, Any], key: tuple[str, str | None], model: str,
                    revision: Any, spec: Mapping[str, Any]) -> Any | None:
    if key in _TOKENIZER_OBJECT_CACHE:
        return _TOKENIZER_OBJECT_CACHE[key]
    from transformers import AutoTokenizer
    kwargs = {"trust_remote_code": bool(spec.get("trust_remote_code", True))}
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
    _TOKENIZER_OBJECT_CACHE[key] = tokenizer
    return tokenizer


def builder_prompt(config: Mapping[str, Any], question: str | None,
                   sources: Iterable[Mapping[str, Any]], architecture: str = "Flat",
                   structure: Mapping[str, Any] | None = None) -> str:
    """Render the exact builder input when the configured tokenizer supports it."""
    envelope = _builder_content(question, sources, architecture=architecture, structure=structure)
    spec = dict(config.get("tokenizer") or {})
    if not bool(spec.get("enabled", False)):
        return envelope
    model = str(spec.get("model") or (config.get("models") or {}).get("builder") or "")
    revision = spec.get("revision")
    key = (model, str(revision) if revision else None)
    tokenizer = _load_tokenizer(config, key, model, revision, spec)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("builder tokenizer must provide apply_chat_template")
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    template_kwargs = dict(spec.get("chat_template_kwargs") or {})
    if template_kwargs:
        kwargs["chat_template_kwargs"] = template_kwargs
    return str(tokenizer.apply_chat_template([{"role": "user", "content": envelope}], **kwargs))


def _builder_content(question: str | None, sources: Iterable[Mapping[str, Any]], architecture: str = "Flat",
                     structure: Mapping[str, Any] | None = None) -> str:
    """Canonical user content shared by exact cost counting and G."""
    envelope = serialize_input(question, sources, architecture=architecture, structure=structure)
    # The example must not contain a literal memory sentence.  A concrete
    # sample such as {"m":"supported memory", ...} is copied verbatim by
    # small models, which would turn the builder into a constant echo instead
    # of an extractor.  Angle-bracket placeholders keep the shape explicit
    # while forcing the model to write real source-derived text.
    task_clause = "from the supplied sources only; do not use a query or future query"
    instruction = (
        f"Build a {normalize_architecture(architecture)} structured memory {task_clause}. Return ONLY the JSON array: no explanation, "
        "no Markdown, and no code fence. Each element must be an object with exactly "
        'two keys, "m" and "src". "m" is a short sentence written from the sources; '
        '"src" is the integer array of source ids supporting it. Replace the '
        'placeholders below with real extracted content: '
        '[{"m":"<sentence taken from the sources>","src":[...source ids...]}].'
    )
    return instruction + "\n" + envelope


def tokenizer_basis(config: Mapping[str, Any]) -> str:
    spec = dict(config.get("tokenizer") or {})
    if not bool(spec.get("enabled", False)) or tokenizer_from_config(config) is None:
        raise RuntimeError("frozen builder tokenizer is required")
    return "frozen_builder_tokenizer"


def _parse_json(text: str) -> Any:
    value = str(text).strip()
    # Qwen may wrap an otherwise valid JSON answer in a Markdown fence even
    # when the prompt says not to. Remove only one complete JSON fence; all
    # schema, field, and source-id checks still happen below.
    if "```" in value:
        lines = value.splitlines()
        for start, line in enumerate(lines):
            if line.strip().lower() not in {"```", "```json"}:
                continue
            end = next((index for index in range(start + 1, len(lines)) if lines[index].strip() == "```"), None)
            if end is not None:
                value = "\n".join(lines[start + 1:end]).strip()
            break
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = value.find(opener)
        if start < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(value)):
            char = value[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    candidate = value[start:index + 1]
                    try:
                        return json.loads(candidate)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        break
    raise ValueError("no valid JSON object or array found")


def structural_postprocess(raw: str, allowed_ids: Iterable[int]) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    """Validate raw G JSON without semantic repair or a second generation."""
    allowed = set(allowed_ids)
    try:
        value = _parse_json(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return (), (f"parse_error:{exc}",)
    if not isinstance(value, list):
        return (), ("root_not_array",)
    if len(value) > MAX_MEMORIES:
        return (), ("too_many_memories",)
    valid: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"m", "src"}:
            errors.append(f"item_{index}:invalid_fields")
            continue
        message = item.get("m")
        src = item.get("src")
        if not isinstance(message, str) or not message.strip():
            errors.append(f"item_{index}:empty_memory")
            continue
        if not isinstance(src, list) or not src or any(type(source_id) is not int for source_id in src):
            errors.append(f"item_{index}:invalid_src")
            continue
        if len(set(src)) != len(src) or not set(src) <= allowed:
            errors.append(f"item_{index}:src_out_of_input")
            continue
        valid.append({"m": message.strip(), "src": list(src)})
    return tuple(valid), tuple(errors)


def aggregate_audit(source: AuditResult, requirements: AuditResult, *,
                    structural_errors: Iterable[str] = (),
                    structure: AuditResult | None = None) -> str:
    """Combine the three independent certification layers.

    A clear failure is a measured FAIL; missing information is UNKNOWN.  This
    ordering is used both by the offline archive and by the deployment audit,
    so UNKNOWN is never silently converted into a negative training label.
    """
    structural = structure or AuditResult(
        FAIL if tuple(structural_errors) else PASS,
        tuple(structural_errors),
    )
    statuses = (structural.status, source.status, requirements.status)
    if FAIL in statuses:
        return FAIL
    if UNKNOWN in statuses:
        return UNKNOWN
    return PASS if all(status == PASS for status in statuses) else UNKNOWN


def build_once(
    config: dict[str, Any], question: str | None, sources: list[dict[str, Any]],
    *, client: ClaudeCodeClient | None = None, sample_id: str = "", qa_id: str = "",
    requirement_annotation: RequirementAnnotation | None = None,
    source_auditor: Callable[[str, list[dict[str, Any]], str, tuple[dict[str, Any], ...]], AuditResult] | None = None,
    requirement_auditor: Callable[[str, list[dict[str, Any]], RequirementAnnotation | None, str, tuple[dict[str, Any], ...]], AuditResult] | None = None,
    architecture: str = "Flat",
    structure: Mapping[str, Any] | None = None,
) -> BuildResult:
    """Run exactly one G call for a non-empty source set and retain all usage."""
    architecture = normalize_architecture(architecture)
    normalized, _ = _preserve_or_normalize(sources)
    structure_value = dict(structure or instantiate_structure(architecture, normalized))
    structure_ok, structure_errors = validate_structure(
        architecture, structure_value, [source["id"] for source in normalized]
    )
    if not structure_ok:
        structure_audit = AuditResult(FAIL, structure_errors)
        unknown = AuditResult(UNKNOWN, structure_errors)
        return BuildResult(
            frozenset(int(source["id"]) for source in normalized), "", (), unknown, unknown,
            FAIL, (), ";".join(structure_errors), architecture, structure_value,
            structure_audit,
        )
    structure_audit = AuditResult(PASS)
    selected = frozenset(int(source["id"]) for source in normalized)
    if not selected:
        source_audit = AuditResult(PASS)
        requirement_audit = AuditResult(PASS if not requirement_annotation or not requirement_annotation.requirements else FAIL,
                                        ("requirements cannot be represented by empty input",) if requirement_annotation and requirement_annotation.requirements else ())
        return BuildResult(selected, "[]", (), source_audit, requirement_audit,
                           aggregate_audit(source_audit, requirement_audit, structure=structure_audit), (), None,
                           architecture, structure_value, structure_audit)
    usage: list[dict[str, Any]] = []
    try:
        if client is None:
            provider = str((config.get("compiler") or {}).get("provider") or (config.get("agent") or {}).get("provider") or "claude_code")
            client = ClaudeCodeClient(config)
        # ``question`` is intentionally excluded from the builder prompt.  It
        # is retained below only for the semantic audit callbacks.
        prompt = builder_prompt(config, None, normalized, architecture, structure_value)
        completion = client.complete(
            "builder", prompt,
            max_tokens=int((config.get("decoder") or {}).get("constructor_max_tokens") or MAX_OUTPUT_TOKENS),
            temperature=float((config.get("decoder") or {}).get("temperature", 0.0)),
            # The contract requires a JSON array at the root.  Do not request
            # provider-specific ``json_object`` mode, which rejects arrays on
            # Claude Code receives the prompt and returns the required array.
            json_object=False,
        )
        usage.append(completion.cost_row(sample_id, qa_id, "builder"))
        raw = completion.text
        memories, errors = structural_postprocess(raw, selected)
        # A malformed root/truncated response is wholly undecidable.  Invalid
        # individual entries are dropped by structural_postprocess while valid
        # entries remain available to the target agent; the overall record is
        # still UNKNOWN because the original structure was not fully valid.
        if errors:
            # The builder returned a measurable schema violation.  It is a
            # failure of the construction contract, not an unobserved label.
            structure_audit = AuditResult(FAIL, tuple(errors))
            source_audit = AuditResult(PASS)
            requirement_audit = AuditResult(PASS)
            return BuildResult(selected, raw, memories, source_audit, requirement_audit, FAIL, tuple(usage), None,
                               architecture, structure_value, structure_audit)
        # Online construction has no semantic auditor. Offline compiler calls
        # pass both callbacks and replace these provisional records with the
        # two tri-state audits.
        audit_question = str(question or "")
        source_audit = source_auditor(audit_question, normalized, raw, memories) if source_auditor else AuditResult(PASS, ("audit_skipped",))
        requirement_audit = (requirement_auditor(audit_question, normalized, requirement_annotation, raw, memories)
                             if requirement_auditor else AuditResult(PASS, ("audit_skipped",)))
        status = aggregate_audit(source_audit, requirement_audit, structure=structure_audit)
        return BuildResult(selected, raw, memories, source_audit, requirement_audit, status, tuple(usage), None,
                           architecture, structure_value, structure_audit)
    except Exception as exc:
        if isinstance(exc, (ImportError, ModuleNotFoundError, FileNotFoundError)):
            raise
        if isinstance(exc, RuntimeError) and "required" in str(exc):
            raise
        usage.append(completion_error_row(sample_id, qa_id, "builder", "compiler", exc))
        error = str(exc)[:1000]
        unknown = AuditResult(UNKNOWN, (error,))
        return BuildResult(selected, "", (), unknown, unknown, UNKNOWN, tuple(usage), error,
                           architecture, structure_value, structure_audit)


def render_memory_text(memories: Iterable[Mapping[str, Any]], source_lookup: Mapping[int, Mapping[str, Any]] | None = None,
                       architecture: str = "Flat", structure: Mapping[str, Any] | None = None) -> str:
    rows = []
    for memory in memories:
        source_ids = ",".join(str(source_id) for source_id in memory.get("src", []))
        rows.append(f"- {memory.get('m', '')} [src:{source_ids}]")
    if not rows:
        return ""
    topology = dict(structure or {})
    return "\n".join([
        f"Architecture: {normalize_architecture(architecture)}",
        f"Topology: {json.dumps(topology, ensure_ascii=False, sort_keys=True)}",
        *rows,
    ])


def memory_packet_from_build(result: BuildResult, sources: list[dict[str, Any]]) -> MemoryPacket:
    lookup = {int(source["id"]): source for source in sources}
    # Keep valid entries after per-item validation failures.  A completely
    # unparsable/empty response has no entries and therefore yields an empty
    # packet naturally; the UNKNOWN status remains visible in metadata.
    text = render_memory_text(result.memories, lookup, result.architecture, result.structure)
    parse_status = "ok"
    if not result.raw:
        parse_status = "empty"
    elif result.status == UNKNOWN:
        parse_status = "invalid"
    return MemoryPacket(
        text,
        {"mode": "selected", "schema": SCHEMA, "status": result.status,
         "selected_ids": sorted(result.selected), "parse_status": parse_status,
         "raw_output": result.raw,
         "source_audit": result.source_audit.__dict__,
         "requirement_audit": result.requirement_audit.__dict__,
         "structure_audit": result.structure_audit.__dict__,
         "architecture": result.architecture, "structure": result.structure},
        result.usage,
        result.memories,
        tuple({"reason": reason} for reason in result.source_audit.reasons + result.requirement_audit.reasons),
        tuple(str(source.get("source_id", source["id"])) for source in sources if int(source["id"]) in result.selected),
    )
