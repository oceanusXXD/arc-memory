"""Fixed architecture actions and deterministic topology construction.

Architecture is a first-class decision in arc.  The architecture action is
chosen once per query; source actions remain the sequential Pick/STOP process.
Topology is deliberately deterministic so no additional
topology policy is trained.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

ARCHITECTURES = ("Flat", "Chain", "Tree", "Graph", "Cluster")
ARCHITECTURE_TO_ID = {name: index for index, name in enumerate(ARCHITECTURES)}
ID_TO_ARCHITECTURE = {index: name for name, index in ARCHITECTURE_TO_ID.items()}
STATUS_UTILITY = {"PASS": 1.0, "UNKNOWN": 0.0, "FAIL": -1.0}


def architecture_contract(value: str | None) -> dict[str, Any]:
    """Return the frozen schema constraints supplied to the builder."""
    kind = normalize_architecture(value)
    contracts = {
        "Flat": ("independent source-backed facts or events", ("source-backed items", "no explicit inter-item edges")),
        "Chain": ("ordered event or state path", ("adjacent links only", "every link is source-traceable")),
        "Tree": ("single-root acyclic hierarchy", ("one root", "one parent per non-root node", "source-traceable parent relation")),
        "Graph": ("typed source-traceable relation graph", ("relations require source support", "no unsupported nodes or edges")),
        "Cluster": ("non-overlapping topic groups", ("each item belongs to one group", "group descriptions are source-backed")),
    }
    description, constraints = contracts[kind]
    return {"architecture": kind, "description": description, "constraints": list(constraints)}


def normalize_architecture(value: str | None) -> str:
    """Return one canonical architecture action or fail closed."""
    text = str(value or "Flat").strip()
    aliases = {name.lower(): name for name in ARCHITECTURES}
    result = aliases.get(text.lower())
    if result is None:
        raise ValueError(f"unknown architecture action: {value!r}")
    return result


def architecture_id(value: str) -> int:
    return ARCHITECTURE_TO_ID[normalize_architecture(value)]


def architecture_from_id(value: int) -> str:
    try:
        return ID_TO_ARCHITECTURE[int(value)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unknown architecture id: {value!r}") from exc


def _source_id(source: Mapping[str, Any]) -> int:
    return int(source["id"])


def instantiate_structure(architecture: str, sources: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Instantiate a deterministic structured context for ``(g, S)``.

    The returned object is part of the frozen builder input.  It describes the
    topology only; source text and source IDs remain supplied separately.
    """
    kind = normalize_architecture(architecture)
    contract = architecture_contract(kind)
    rows = sorted((dict(source) for source in sources), key=_source_id)
    ids = [_source_id(source) for source in rows]
    if kind == "Flat":
        return {"architecture": kind, "contract": contract, "nodes": ids, "groups": [ids] if ids else [], "edges": []}
    if kind == "Chain":
        return {
            "architecture": kind,
            "contract": contract,
            "nodes": ids,
            "groups": [ids] if ids else [],
            "edges": [[left, right] for left, right in zip(ids, ids[1:])],
        }
    if kind == "Tree":
        # The first retrieved source is the root.  Every later source attaches
        # to the nearest earlier source in the same session, or to the root.
        edges: list[list[int]] = []
        for index, source in enumerate(rows[1:], 1):
            same_session = [
                rows[parent]
                for parent in range(index)
                if rows[parent].get("session") == source.get("session")
            ]
            parent = same_session[-1] if same_session else rows[0]
            edges.append([_source_id(parent), _source_id(source)])
        return {"architecture": kind, "contract": contract, "nodes": ids, "groups": [ids] if ids else [], "edges": edges}
    if kind == "Graph":
        # Instantiate only observable, source-supported temporal links.  A
        # complete graph would fabricate relations between unrelated sources;
        # cross-session evidence remains represented as separate nodes unless
        # the builder/auditor can support an explicit relation in its output.
        edges = []
        for left, right in zip(rows, rows[1:]):
            if left.get("session") == right.get("session"):
                edges.append([_source_id(left), _source_id(right)])
        return {
            "architecture": kind,
            "contract": contract,
            "nodes": ids,
            "groups": [ids] if ids else [],
            "edges": edges,
        }
    groups: dict[str, list[int]] = defaultdict(list)
    for source in rows:
        # Session is the stable observable grouping key.  Missing sessions are
        # kept together instead of being silently discarded.
        key = str(source.get("session") or source.get("session_id") or "__unknown__")
        groups[key].append(_source_id(source))
    return {
        "architecture": kind,
        "contract": contract,
        "nodes": ids,
        "groups": [groups[key] for key in sorted(groups)],
        "edges": [],
    }


def validate_structure(architecture: str, structure: Mapping[str, Any], source_ids: Iterable[int]) -> tuple[bool, tuple[str, ...]]:
    """Validate topology metadata against the selected source set."""
    kind = normalize_architecture(architecture)
    expected = set(int(value) for value in source_ids)
    errors: list[str] = []
    try:
        raw_nodes = [int(value) for value in structure.get("nodes", ())]
        actual = set(raw_nodes)
    except (TypeError, ValueError):
        raw_nodes = []
        actual = set()
        errors.append("structure_nodes_not_integer")
    if actual != expected:
        errors.append("structure_nodes_do_not_match_sources")
    if len(raw_nodes) != len(actual):
        errors.append("structure_nodes_duplicate")
    if normalize_architecture(str(structure.get("architecture") or kind)) != kind:
        errors.append("structure_architecture_mismatch")
    edges = structure.get("edges", ()) or ()
    normalized_edges: list[tuple[int, int]] = []
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            errors.append("edge_invalid")
            continue
        try:
            parent, child = int(edge[0]), int(edge[1])
        except (TypeError, ValueError):
            errors.append("edge_not_integer")
            continue
        if parent == child:
            errors.append("edge_self_loop")
        if parent not in expected or child not in expected:
            errors.append("edge_out_of_sources")
        normalized_edges.append((parent, child))
    if len(set(normalized_edges)) != len(normalized_edges):
        errors.append("duplicate_edges")
    if kind == "Flat" and edges:
        errors.append("flat_has_edges")
    if kind == "Chain":
        expected_edges = list(zip(sorted(expected), sorted(expected)[1:]))
        if normalized_edges != expected_edges:
            errors.append("chain_edges_not_ordered")
    if kind == "Tree":
        parents: dict[int, int] = {}
        for parent, child in normalized_edges:
            if child in parents:
                errors.append("tree_multiple_parents")
            parents[child] = parent
        if expected and len(normalized_edges) != len(expected) - 1:
            errors.append("tree_must_have_one_root")
        roots = expected - set(parents)
        if len(roots) != (1 if expected else 0):
            errors.append("tree_root_count")
        # Following parent pointers must terminate at the unique root.
        for node in expected:
            seen: set[int] = set()
            current = node
            while current in parents:
                if current in seen:
                    errors.append("tree_cycle")
                    break
                seen.add(current)
                current = parents[current]
    groups = structure.get("groups", ()) or ()
    if kind == "Cluster":
        owners: set[int] = set()
        for group in groups:
            try:
                members = {int(value) for value in group}
            except (TypeError, ValueError):
                errors.append("cluster_group_not_integer")
                continue
            if not members:
                errors.append("cluster_empty_group")
            if owners & members:
                errors.append("cluster_groups_overlap")
            owners.update(members)
        if owners != expected:
            errors.append("cluster_groups_do_not_cover_sources")
    elif kind in {"Flat", "Chain", "Tree", "Graph"}:
        if groups and [int(v) for v in groups[0]] != sorted(expected):
            errors.append("non_cluster_group_mismatch")
    return not errors, tuple(dict.fromkeys(errors))


def architecture_utility(status: str) -> float:
    try:
        return STATUS_UTILITY[str(status).upper()]
    except KeyError as exc:
        raise ValueError(f"unknown compiler status: {status!r}") from exc


def solution_record(architecture: str, source_ids: Iterable[int], *, cost: int | None = None,
                    lifecycle_cost: float | None = None, status: str | None = None,
                    utility: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "architecture": normalize_architecture(architecture),
        "source_ids": sorted({int(value) for value in source_ids}),
    }
    if cost is not None:
        result["cost"] = int(cost)
    if lifecycle_cost is not None:
        result["lifecycle_cost"] = float(lifecycle_cost)
    if status is not None:
        result["status"] = str(status).upper()
        result["utility"] = architecture_utility(status) if utility is None else float(utility)
    elif utility is not None:
        result["utility"] = float(utility)
    return result


def solution_key(solution: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    return normalize_architecture(str(solution.get("architecture") or "Flat")), tuple(
        sorted(int(value) for value in solution.get("source_ids") or [])
    )
