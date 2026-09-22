"""Offline domain compilation command."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from arc.agent.config import load_config, run_dir, write_manifest
from arc.agent.io import read_jsonl, write_jsonl
from .architecture import ARCHITECTURES, architecture_utility, normalize_architecture, solution_record
from .core import PASS, FAIL, build_domain, compile_domain, context_order, bounds, reachable
from .memory import (RequirementAnnotation, build_once, configured_input_cost,
                     normalize_sources, structural_postprocess, _parse_json)
from .memory import AuditResult, aggregate_audit
from arc.agent.reader import ClaudeCodeClient
from arc.agent.runtime import question_answer_task, run_agent
from arc.agent.data.score import score_prediction
from arc.agent.persistent import PersistentMemoryState, Update


def _prompt_sources(sources: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep model-facing compiler prompts free of frozen vector payloads.

    Retrieval rows carry query/source embeddings for selector training.  Those
    vectors are useful in the compilation artifact but are not part of the
    annotation or audit task and can make a CLI prompt exceed the provider's
    context limit.  Preserve only the canonical source metadata and text.
    """
    fields = (
        "id", "source_id", "session", "session_id", "time", "session_datetime",
        "speaker", "position", "position_in_session", "source_offset", "offset", "text",
    )
    return [{key: source[key] for key in fields if key in source} for source in sources]


def annotation_from_row(row: dict[str, Any], source_map: dict[str, int] | None = None) -> RequirementAnnotation:
    if "requirements" not in row and "C" not in row:
        return RequirementAnnotation(tuple(), tuple(), False, "missing_requirement_annotation")
    requirements = tuple(dict(item) for item in row.get("requirements") or row.get("C") or [])
    if len(requirements) > 6:
        return RequirementAnnotation(requirements, tuple(), False, "too_many_requirements")
    packages = []
    try:
        for requirement in requirements:
            paths = requirement.get("packages") or requirement.get("support_packages") or []
            converted = []
            for path in paths:
                if not path:
                    raise ValueError("empty_support_package")
                values = []
                for value in path:
                    key = str(value)
                    if source_map and key in source_map:
                        values.append(source_map[key])
                    else:
                        values.append(int(value))
                if not values or any(value <= 0 or (source_map is not None and value not in set(source_map.values())) for value in values):
                    raise ValueError("support_package_source_out_of_range")
                converted.append(frozenset(values))
            if not 1 <= len(converted) <= 3:
                raise ValueError("support_package_count_must_be_1_to_3")
            packages.append(tuple(converted))
    except (TypeError, ValueError, KeyError) as exc:
        return RequirementAnnotation(requirements, tuple(), False, f"invalid_support_package:{exc}")
    valid = bool(row.get("annotation_valid", True)) and len(packages) == len(requirements)
    return RequirementAnnotation(requirements, tuple(packages), valid, row.get("annotation_reason"))


def annotate_requirements(config: dict[str, Any], question: str, sources: list[dict[str, Any]], full_raw: str,
                         source_map: dict[str, int], sample_id: str = "", qa_id: str = "") -> RequirementAnnotation:
    """Ask the compiler model for offline C/W labels when no labels are supplied."""
    prompt = json.dumps({"task": "annotate requirements and alternative support packages",
                         "question": question, "sources": _prompt_sources(sources), "full_output": full_raw,
                         "schema": {"requirements": [{"id": "r1", "text": "", "packages": [[1]]}]}}, ensure_ascii=False)
    try:
        provider = str((config.get("compiler") or {}).get("provider") or "claude_code")
        completion = ClaudeCodeClient(config).complete("compiler", prompt, max_tokens=512, temperature=float((config.get("decoder") or {}).get("temperature", 0.7)))
        usage = [completion.cost_row(sample_id, qa_id, "annotation")]
        try:
            body = _parse_json(completion.text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return RequirementAnnotation(tuple(), tuple(), False,
                                         f"annotation_parse_error:{getattr(exc, 'msg', str(exc))}", tuple(usage))
        row = {"requirements": body.get("requirements") if isinstance(body, dict) else None}
        annotation = annotation_from_row(row, source_map)
        # A separate validation pass checks that every package is supported by
        # the supplied source text.  It returns only a boolean verdict so it
        # cannot silently rewrite the teacher's labels.
        validation_prompt = json.dumps({
            "task": "validate requirement packages against the original sources",
            "question": question, "sources": _prompt_sources(sources),
            "requirements": list(annotation.requirements),
            "packages": [[sorted(path) for path in paths] for paths in annotation.packages],
            "schema": {"valid": True, "reason": ""},
        }, ensure_ascii=False)
        checked = ClaudeCodeClient(config).complete("auditor", validation_prompt, max_tokens=512, temperature=float((config.get("decoder") or {}).get("temperature", 0.7)))
        usage.append(checked.cost_row(sample_id, qa_id, "annotation_validation"))
        try:
            verdict = _parse_json(checked.text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return RequirementAnnotation(annotation.requirements, annotation.packages, False,
                                         f"annotation_validation_parse_error:{getattr(exc, 'msg', str(exc))}", tuple(usage))
        if not isinstance(verdict, dict) or verdict.get("valid") is not True:
            return RequirementAnnotation(annotation.requirements, annotation.packages, False,
                                         str((verdict or {}).get("reason") or "annotation_validation_failed"), tuple(usage))
        return RequirementAnnotation(annotation.requirements, annotation.packages, annotation.valid,
                                     annotation.reason, tuple(usage))
    except Exception:
        raise


def _future_queries(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen future-query sample for one write update.

    A compiler row may carry an explicit ``future_queries``/``queries`` list.
    The legacy per-QA format is accepted only when it also supplies an answer;
    rows without a query sample remain UNKNOWN rather than receiving a made-up
    utility label.
    """
    raw = row.get("future_queries")
    if raw is None:
        raw = row.get("queries")
    if raw is None and row.get("answer") is not None:
        raw = [{"question": row.get("question", ""), "answer": row.get("answer", ""),
                "category": row.get("category", 1)}]
    if not isinstance(raw, (list, tuple)):
        return []
    result: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            query = dict(item)
        else:
            query = {"question": str(item)}
        if query.get("question") is None:
            continue
        result.append(query)
    return result


def _query_metrics(config: dict[str, Any], row: dict[str, Any], result: Any,
                   selected: list[dict[str, Any]], architecture: str) -> dict[str, Any]:
    """Evaluate a candidate memory on all frozen future queries.

    Precomputed ``utility`` values are accepted for deterministic replay.  If
    they are absent, evaluation is performed only when the compiler config
    explicitly enables the frozen Answer model.  Missing query evidence is an
    UNKNOWN certification state, never a zero score.
    """
    queries = _future_queries(row)
    if not queries:
        return {"status": UNKNOWN, "reason": "missing_future_queries", "query_scores": [],
                "utility": None, "coverage": None, "lifecycle_cost": None}
    compiler_cfg = config.get("compiler") or {}
    evaluate_live = bool(compiler_cfg.get("evaluate_queries", False))
    query_scores: list[float] = []
    usage_rows: list[dict[str, Any]] = []
    state = Update(PersistentMemoryState(), result.memories, selected,
                   architecture=architecture, source_ids=result.selected)
    for query in queries:
        if query.get("utility") is not None:
            try:
                score = float(query["utility"])
            except (TypeError, ValueError):
                return {"status": UNKNOWN, "reason": "invalid_precomputed_utility", "query_scores": query_scores,
                        "utility": None, "coverage": None, "lifecycle_cost": None}
        elif not evaluate_live:
            return {"status": UNKNOWN, "reason": "future_query_answer_evaluation_disabled",
                    "query_scores": query_scores, "utility": None, "coverage": None, "lifecycle_cost": None}
        else:
            try:
                answer = run_agent(
                    question_answer_task(str(query.get("question") or "")),
                    state.render(str(query.get("question") or "")),
                    config=config,
                )
                usage = answer.usage or {}
                usage_rows.append(usage)
                if query.get("answer") is None or query.get("category") is None:
                    return {"status": UNKNOWN, "reason": "future_query_missing_reference_answer",
                            "query_scores": query_scores, "utility": None, "coverage": None,
                            "lifecycle_cost": None, "query_usage": usage_rows}
                score = score_prediction(answer.outcome, str(query.get("answer") or ""), int(query.get("category")))
            except Exception as exc:
                return {"status": UNKNOWN, "reason": f"future_query_execution:{exc}",
                        "query_scores": query_scores, "utility": None, "coverage": None,
                        "lifecycle_cost": None, "query_usage": usage_rows}
        if not 0.0 <= score <= 1.0:
            return {"status": UNKNOWN, "reason": "future_query_utility_out_of_range",
                    "query_scores": query_scores, "utility": None, "coverage": None,
                    "lifecycle_cost": None, "query_usage": usage_rows}
        query_scores.append(score)
    tau_q = float(compiler_cfg.get("tau_query", 0.5))
    tau_u = float(compiler_cfg.get("tau_utility", 0.5))
    tau_cov = float(compiler_cfg.get("tau_coverage", 0.5))
    utility = sum(query_scores) / len(query_scores)
    coverage = sum(score >= tau_q for score in query_scores) / len(query_scores)
    write_tokens = [int(item.get("total_tokens")) for item in result.usage
                    if item.get("total_tokens") is not None and item.get("usage_complete", True)]
    read_tokens = [int(item.get("total_tokens")) for item in usage_rows
                   if item.get("total_tokens") is not None and item.get("usage_complete", True)]
    lifecycle_cost = None
    if len(write_tokens) == len([item for item in result.usage if item.get("role") == "builder"]) and len(read_tokens) == len(queries):
        lifecycle_cost = sum(write_tokens) / len(queries) + sum(read_tokens) / len(queries)
    status = PASS if utility >= tau_u and coverage >= tau_cov else FAIL
    return {"status": status, "reason": None, "query_scores": query_scores,
            "utility": utility, "coverage": coverage, "lifecycle_cost": lifecycle_cost,
            "tau_query": tau_q, "tau_utility": tau_u, "tau_coverage": tau_cov,
            "query_usage": usage_rows}


def _certify_result(config: dict[str, Any], row: dict[str, Any], result: Any,
                    selected: list[dict[str, Any]], architecture: str) -> tuple[Any, dict[str, Any]]:
    """Apply the long-term value and coverage gate after the three audits."""
    if result.status != PASS:
        return result, {"status": result.status, "reason": "construction_audit_not_pass",
                        "query_scores": [], "utility": None, "coverage": None,
                        "lifecycle_cost": None}
    metrics = _query_metrics(config, row, result, selected, architecture)
    return replace(result, status=metrics["status"]), metrics


def compile_task(config: dict[str, Any], row: dict[str, Any], *, limit: int = 16, d: int = 8, full_result=None) -> dict[str, Any]:
    question = str(row.get("question") or "")
    raw_sources = list(row.get("sources") or row.get("evidence") or [])
    sources, _ = normalize_sources(raw_sources)
    source_map = {str(source.get("source_id", source["id"])): int(source["id"]) for source in sources}
    annotation = annotation_from_row(row, source_map)
    if row.get("teacher_model") not in (None, "grok-4.6"):
        raise ValueError("Formal compilation requires Grok-4.6 teacher annotations; Qwen is the frozen experiment model")
    if not annotation.valid:
        # Never substitute the Qwen compiler for the Codex/Grok teacher.
        return {"schema": "arc", "sample_id": row.get("sample_id"), "qa_id": row.get("qa_id"),
                "status": "annotation_error", "annotation": annotation.reason,
                "question": question, "sources": sources, "evaluations": [], "annotation_usage": [],
                "successful_solutions": [], "domain": []}
    if full_result is not None and full_result.selected != frozenset(int(s["id"]) for s in sources):
        raise ValueError("cached Full source IDs differ from compiler input")

    metadata = {int(source["id"]): source for source in sources}
    source_by_id = dict(metadata)
    audit_usage: list[dict[str, Any]] = []
    active_architecture = "Flat"

    def audit_json(prompt: str, role: str) -> AuditResult:
        completion = ClaudeCodeClient(config).complete(role, prompt, max_tokens=2048, temperature=float((config.get("decoder") or {}).get("temperature", 0.7)))
        audit_usage.append(completion.cost_row(str(row.get("sample_id", "")), str(row.get("qa_id", "")), f"{role}_audit"))
        body = _parse_json(completion.text)
        if not isinstance(body, dict):
            raise ValueError("auditor response must be a JSON object")
        status = str(body.get("status", "UNKNOWN")).upper()
        if status not in {PASS, FAIL, "UNKNOWN"}:
            raise ValueError(f"invalid auditor status: {status}")
        reasons = tuple(str(value) for value in body.get("reasons", []) if value is not None)
        requirements = tuple(str(value) for value in body.get("requirement_ids", []) if value is not None)
        return AuditResult(status, reasons, requirements)

    def source_auditor(q: str, selected: list[dict[str, Any]], raw: str, memories: tuple[dict[str, Any], ...]) -> AuditResult:
        prompt = json.dumps({"task": "You are a machine-readable auditor. Evaluate whether every generated memory is fully supported by the supplied sources. Do not answer the user question. Return ONLY one JSON object.",
                             "output_schema": {"status": "PASS if all memories are supported, FAIL if unsupported facts are present, or UNKNOWN if evidence is insufficient", "reasons": ["short reasons"]},
                             "question": q, "architecture": active_architecture,
                             "sources": _prompt_sources(selected),
                             "raw_output": raw, "memories": list(memories)}, ensure_ascii=False)
        return audit_json(prompt, "auditor")

    def requirement_auditor(q: str, selected: list[dict[str, Any]], ann: RequirementAnnotation | None,
                            raw: str, memories: tuple[dict[str, Any], ...]) -> AuditResult:
        prompt = json.dumps({"task": "You are a machine-readable auditor. Evaluate whether the generated memories satisfy every stated requirement for answering the question. Do not answer the user question. Return ONLY one JSON object.",
                             "output_schema": {"status": "PASS if all requirements are satisfied, FAIL if any requirement is missing, or UNKNOWN if it cannot be decided", "requirement_ids": ["failed or undecidable requirement ids"], "reasons": ["short reasons"]},
                             "question": q, "architecture": active_architecture,
                             "all_sources": _prompt_sources(sources), "requirements": list((ann or annotation).requirements),
                             "raw_output": raw, "memories": list(memories)}, ensure_ascii=False)
        return audit_json(prompt, "auditor")

    package_paths = annotation.packages
    ids = frozenset(range(1, len(sources) + 1))
    input_limit = int((config.get("budgets") or {}).get("builder_input_limit") or 8192)
    domains: dict[str, list[frozenset[int]]] = {}
    witnesses_by_architecture: dict[str, dict[frozenset[int], tuple[str, frozenset[int], frozenset[int]]]] = {}
    records_by_architecture: dict[str, dict[frozenset[int], str]] = {}
    results_by_architecture: dict[str, dict[frozenset[int], Any]] = {}
    metrics_by_architecture: dict[str, dict[frozenset[int], dict[str, Any]]] = {}
    full_by_architecture: dict[str, Any] = {}
    evaluations: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []

    for architecture in ARCHITECTURES:
        active_architecture = architecture
        cost_cache: dict[frozenset[int], int] = {}

        def cost(values: Iterable[int], _architecture=architecture) -> int:
            key = frozenset(int(value) for value in values)
            if key not in cost_cache:
                selected = [source_by_id[source_id] for source_id in sorted(key)]
                cost_cache[key] = configured_input_cost(config, question, selected, architecture=_architecture)
            return cost_cache[key]

        if architecture == "Flat" and full_result is not None:
            current_full = full_result
        else:
            # The compiler evaluates a write-stage memory.  The question is
            # available to the auditors and future-query evaluator, never to
            # the builder serialization itself.
            current_full = build_once(config, None, sources, sample_id=str(row.get("sample_id", "")),
                                      qa_id=str(row.get("qa_id", "")), architecture=architecture)
        if current_full.raw:
            parsed_memories, structural_errors = structural_postprocess(
                current_full.raw, [int(source["id"]) for source in sources]
            )
            structure_audit = AuditResult(
                FAIL if structural_errors else PASS,
                tuple(structural_errors),
            )
            if structural_errors:
                current_full = replace(
                    current_full, memories=parsed_memories,
                    source_audit=AuditResult(PASS), requirement_audit=AuditResult(PASS),
                    structure_audit=structure_audit, status=FAIL,
                )
            else:
                sa = source_auditor(question, sources, current_full.raw, parsed_memories)
                ra = requirement_auditor(question, sources, annotation, current_full.raw, parsed_memories)
                current_full = replace(current_full, memories=parsed_memories,
                                       source_audit=sa, requirement_audit=ra,
                                       structure_audit=structure_audit,
                                       status=aggregate_audit(sa, ra, structure=structure_audit),
                                       usage=tuple([*current_full.usage, *audit_usage]))
            audit_usage.clear()
        current_full, full_metrics = _certify_result(config, row, current_full, sources, architecture)
        full_by_architecture[architecture] = current_full
        metrics_by_architecture[architecture] = {}
        ids_for_full = frozenset(int(source["id"]) for source in sources)
        metrics_by_architecture[architecture][ids_for_full] = full_metrics
        domain, witnesses = build_domain(
            ids, package_paths, cost,
            lambda base, extras: context_order(base, extras, metadata),
            input_limit=input_limit, d=d,
        )
        if architecture == "Flat" and frozenset() not in domain:
            domain = sorted([*domain, frozenset()], key=lambda candidate: (cost(candidate), tuple(sorted(candidate))))
            witnesses[frozenset()] = ("empty", frozenset(), frozenset())
        domains[architecture] = domain
        witnesses_by_architecture[architecture] = witnesses
        cache: dict[frozenset[int], Any] = {ids: current_full}

        def evaluate(candidate: frozenset[int], _architecture=architecture) -> str:
            if candidate in cache:
                return cache[candidate].status
            selected = [source_by_id[source_id] for source_id in sorted(candidate)]
            result = build_once(config, None, selected, sample_id=str(row.get("sample_id", "")),
                                qa_id=str(row.get("qa_id", "")), requirement_annotation=annotation,
                                source_auditor=source_auditor, requirement_auditor=requirement_auditor,
                                architecture=_architecture)
            if audit_usage:
                result = replace(result, usage=tuple([*result.usage, *audit_usage]))
                audit_usage.clear()
            result, metrics = _certify_result(config, row, result, selected, _architecture)
            cache[candidate] = result
            metrics_by_architecture[_architecture][candidate] = metrics
            return result.status

        records = compile_domain(ids, domain, cost, evaluate, limit=limit)
        records_by_architecture[architecture] = records
        results_by_architecture[architecture] = cache
        for candidate, status in records.items():
            result = cache[candidate]
            combo = solution_record(architecture, candidate, cost=cost(candidate), status=status)
            combo.update({
                "raw_output": result.raw,
                "memories": list(result.memories),
                "source_audit": result.source_audit.__dict__,
                "requirement_audit": result.requirement_audit.__dict__,
                "structure_audit": result.structure_audit.__dict__,
                "usage": list(result.usage),
                "witness": [witnesses[candidate][0], sorted(witnesses[candidate][1]), sorted(witnesses[candidate][2])],
            })
            metrics = full_metrics if candidate == ids else metrics_by_architecture[architecture].get(candidate, {})
            combo.update({key: value for key, value in metrics.items()
                          if key in {"query_scores", "utility", "coverage", "lifecycle_cost", "query_usage", "reason",
                                     "tau_query", "tau_utility", "tau_coverage"}})
            evaluations.append(combo)
        domain_rows.extend(
            [solution_record(architecture, candidate, cost=cost(candidate)) | {
                "witness": [witnesses[candidate][0], sorted(witnesses[candidate][1]), sorted(witnesses[candidate][2])]
            } for candidate in domain]
        )
    budgets = [int(value) for value in ((config.get("budgets") or {}).get("deployment") or [input_limit])]
    budget_records = {}
    for budget in budgets:
        reachable_solutions: list[dict[str, Any]] = []
        successful_solutions: list[dict[str, Any]] = []
        architecture_bounds: dict[str, Any] = {}
        for architecture in ARCHITECTURES:
            cost_cache = {}

            def budget_cost(values: Iterable[int], _architecture=architecture) -> int:
                key = frozenset(int(value) for value in values)
                if key not in cost_cache:
                    budget_cost_value = configured_input_cost(config, question,
                                                              [source_by_id[source_id] for source_id in sorted(key)],
                                                              architecture=_architecture)
                    cost_cache[key] = budget_cost_value
                return cost_cache[key]

            domain = domains[architecture]
            records = records_by_architecture[architecture]
            feasible = [candidate for candidate in domain if reachable(candidate, budget_cost, budget)]
            architecture_bounds[architecture] = bounds(domain, records, budget_cost, budget)
            reachable_solutions.extend(solution_record(architecture, candidate, cost=budget_cost(candidate)) for candidate in feasible)
            successful_solutions.extend(
                solution_record(architecture, candidate, cost=budget_cost(candidate), status=records.get(candidate, "UNKNOWN"))
                for candidate in feasible if records.get(candidate) == PASS
            )
        budget_records[str(budget)] = {
            "budget": budget,
            "reachable_solutions": reachable_solutions,
            "successful_solutions": successful_solutions,
            "bounds": architecture_bounds,
            "reachable_sets": [item["source_ids"] for item in reachable_solutions if item["architecture"] == "Flat"],
            "successful_sets": [item["source_ids"] for item in successful_solutions if item["architecture"] == "Flat"],
        }
    flat_full = full_by_architecture["Flat"]
    flat_metrics = metrics_by_architecture["Flat"].get(frozenset(int(source["id"]) for source in sources), {})
    successful_solutions = [solution_record(item["architecture"], item["source_ids"], cost=item.get("cost"),
                                            status=item.get("status"), utility=item.get("utility"))
                            for item in evaluations if item["status"] == PASS]
    return {
        "schema": "arc", "sample_id": row.get("sample_id"), "qa_id": row.get("qa_id"), "domain_complete": True, "question": question,
        "sources": sources, "budget": input_limit, "budgets": budget_records,
        "full": {"architecture": "Flat", "status": flat_full.status, "raw_output": flat_full.raw,
                 "memories": list(flat_full.memories), "usage": list(flat_full.usage),
                 "source_audit": flat_full.source_audit.__dict__, "requirement_audit": flat_full.requirement_audit.__dict__,
                 "structure_audit": flat_full.structure_audit.__dict__,
                 **{key: value for key, value in flat_metrics.items()
                    if key in {"query_scores", "utility", "coverage", "lifecycle_cost", "query_usage", "reason",
                               "tau_query", "tau_utility", "tau_coverage"}}},
        "full_by_architecture": {
            architecture: {"architecture": architecture, "status": result.status, "raw_output": result.raw,
                           "memories": list(result.memories), "usage": list(result.usage),
                           "source_audit": result.source_audit.__dict__, "requirement_audit": result.requirement_audit.__dict__,
                           "structure_audit": result.structure_audit.__dict__}
            for architecture, result in full_by_architecture.items()
        },
        "annotation": {"requirements": list(annotation.requirements),
                       "packages": [[sorted(path) for path in paths] for paths in annotation.packages],
                       "valid": annotation.valid, "reason": annotation.reason, "usage": list(annotation.usage)},
        "domain": domain_rows,
        "evaluations": evaluations,
        "successful_solutions": successful_solutions,
        "successful_sets": [item["source_ids"] for item in successful_solutions if item["architecture"] == "Flat"],
    }


def compile_file(config: dict[str, Any], input_path: str | Path, output: str | Path | None = None, *, limit: int = 16, d: int = 8) -> dict[str, Any]:
    target = Path(output) if output else run_dir(config) / "compilation.jsonl"

    def rows() -> Iterable[dict[str, Any]]:
        for row in read_jsonl(input_path):
            yield compile_task(config, row, limit=limit, d=d)

    count = write_jsonl(target, rows())
    write_manifest(config, "compile_domain", {"input": str(input_path), "output": str(target), "items": count, "limit": limit, "d": d})
    return {"output": str(target), "items": count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the finite candidate domain")
    parser.add_argument("--config", default="configs/locomo.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--d", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(compile_file(load_config(args.config), args.input, args.output, limit=args.limit, d=args.d), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
