"""Finite candidate compilation and sequence-risk primitives.

This module intentionally contains no model or dataset code.  It implements
the executable part of the arc specification: the finite joint evidence /
context domain, exact complete-input cost ordering, tri-state compilation
records, budget reachability, and complete stopping-sequence probabilities.
"""
from __future__ import annotations

from itertools import combinations, product
from math import exp, inf, isfinite, log
from typing import Any, Callable, Iterable, Mapping

STOP = "STOP"
PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"
STATUSES = frozenset({PASS, FAIL, UNKNOWN})


def canonical(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(values))


def powerset(values: Iterable[int]) -> Iterable[frozenset[int]]:
    sequence = canonical(values)
    for size in range(len(sequence) + 1):
        yield from (frozenset(part) for part in combinations(sequence, size))


def normalize(
    evidence: Iterable[int], packages: Iterable[Iterable[Iterable[int]]]
) -> tuple[frozenset[int], tuple[tuple[frozenset[int], ...], ...]]:
    """Validate the paper's n<=64, <=6 requirements and 1-3 packages rules."""
    universe = frozenset(evidence)
    if any(type(value) is not int or value < 1 for value in universe):
        raise ValueError("positive integer source IDs required")
    if universe != frozenset(range(1, len(universe) + 1)):
        raise ValueError("IDs must be consecutive positions 1..n")
    normalized: list[tuple[frozenset[int], ...]] = []
    for paths in packages:
        unique = {frozenset(path) for path in paths}
        normalized.append(tuple(sorted(unique, key=canonical)))
    if len(universe) > 64 or len(normalized) > 6:
        raise ValueError("source or requirement limit exceeded")
    if any(not paths or len(paths) > 3 for paths in normalized):
        raise ValueError("one to three paths per requirement required")
    if any(not path or not path <= universe for paths in normalized for path in paths):
        raise ValueError("each path must be a nonempty source subset")
    return universe, tuple(normalized)


def checked_cost(raw_cost: Callable[[frozenset[int]], int]) -> Callable[[Iterable[int]], int]:
    """Memoize and validate an exact complete serialized-input cost function."""
    cache: dict[frozenset[int], int] = {}

    def cost(values: Iterable[int]) -> int:
        key = frozenset(values)
        if key not in cache:
            value = raw_cost(key)
            if type(value) is not int or value < 0:
                raise ValueError("token count must be a nonnegative integer")
            cache[key] = value
        return cache[key]

    if cost(frozenset()) != 0:
        raise ValueError("empty input must use the zero-call adapter")
    return cost


def context_order(
    base: Iterable[int], extras: Iterable[int], metadata: Mapping[int, Mapping[str, Any]]
) -> list[int]:
    """Order context using only observable session, position and retrieval data."""
    base_set = frozenset(base)

    def key(source_id: int) -> tuple[int, float, float, int]:
        item = metadata[source_id]
        session = item.get("session")
        same = [
            other
            for other in base_set
            if session is not None and session == metadata[other].get("session")
        ]
        distance = min(
            (abs(float(item.get("position", 0)) - float(metadata[other].get("position", 0)))
             for other in same),
            default=inf,
        )
        return (
            0 if same else 1,
            distance,
            -float(item.get("retrieval_score", 0.0)),
            int(source_id),
        )

    return sorted((int(value) for value in extras), key=key)


def build_domain(
    evidence: Iterable[int],
    packages: Iterable[Iterable[Iterable[int]]],
    cost: Callable[[Iterable[int]], int],
    rank_extra: Callable[[frozenset[int], frozenset[int]], Iterable[int]],
    *,
    input_limit: int,
    d: int = 8,
) -> tuple[list[frozenset[int]], dict[frozenset[int], tuple[str, frozenset[int], frozenset[int]]]]:
    """Build the complete deduplicated domain from equations (5)-(7)."""
    universe, normalized = normalize(evidence, packages)
    if type(d) is not int or not 0 <= d <= 8:
        raise ValueError("d must be an integer between 0 and 8")
    if type(input_limit) is not int or input_limit <= 0:
        raise ValueError("input_limit must be a positive integer")
    full_cost = cost(universe)
    if full_cost > input_limit:
        raise ValueError("full input exceeds builder input limit")
    all_support = frozenset().union(*(path for paths in normalized for path in paths))
    outside = universe - all_support
    bases = {
        frozenset().union(*choice)
        for choice in product(*normalized)
    } if normalized else {frozenset()}
    witnesses: dict[frozenset[int], tuple[str, frozenset[int], frozenset[int]]] = {
        universe: ("full", universe, frozenset())
    }
    for base in sorted(bases, key=canonical):
        remaining = universe - base
        ordered = tuple(rank_extra(base, remaining))
        if len(ordered) != len(remaining) or set(ordered) != set(remaining):
            raise ValueError("rank_extra must return each extra exactly once")
        shortlist = frozenset(ordered[:d])
        for selected in powerset(shortlist):
            candidate = base | selected
            witnesses.setdefault(candidate, ("shortlist", base, selected))
        witnesses.setdefault(base | outside, ("outside_all", base, outside))
    cap = min(input_limit, full_cost)
    domain = sorted(
        (candidate for candidate in witnesses if cost(candidate) <= cap),
        key=lambda candidate: (cost(candidate), canonical(candidate)),
    )
    return domain, {candidate: witnesses[candidate] for candidate in domain}


def compile_domain(
    evidence: Iterable[int],
    domain: Iterable[Iterable[int]],
    cost: Callable[[Iterable[int]], int],
    evaluate: Callable[[frozenset[int]], str],
    limit: int = 16,
) -> dict[frozenset[int], str]:
    """Evaluate Full first, followed by the global exact-cost candidate order."""
    universe = frozenset(evidence)
    if type(limit) is not int or not 1 <= limit <= 16:
        raise ValueError("limit must be an integer between 1 and 16")
    ordered = sorted({frozenset(item) for item in domain}, key=lambda item: (cost(item), canonical(item)))
    if universe not in ordered:
        raise ValueError("domain must include full E")
    records: dict[frozenset[int], str] = {}
    for candidate in (universe, *(item for item in ordered if item != universe)):
        if len(records) >= limit:
            break
        status = evaluate(candidate)
        if status not in STATUSES:
            raise ValueError("evaluator must return PASS, FAIL, or UNKNOWN")
        records[candidate] = status
    return records


def reachable(values: Iterable[int], cost: Callable[[Iterable[int]], int], budget: int) -> bool:
    """Check every canonical prefix, including empty and complete prefixes."""
    sequence = canonical(values)
    return all(cost(frozenset(sequence[:size])) <= budget for size in range(len(sequence) + 1))


def bounds(
    domain: Iterable[Iterable[int]],
    records: Mapping[frozenset[int], str],
    cost: Callable[[Iterable[int]], int],
    budget: int,
) -> dict[str, Any]:
    """Return the unresolved [L,U] cost certificate from equation (9)."""
    if type(budget) is not int or budget <= 0:
        raise ValueError("positive budget required")
    feasible = [frozenset(item) for item in domain if reachable(item, cost, budget)]
    known = [cost(item) for item in feasible if records.get(item) == PASS]
    possible = [cost(item) for item in feasible if records.get(item) != FAIL]
    lower = min(possible, default=inf)
    upper = min(known, default=inf)
    return {
        "lower": lower,
        "upper": upper,
        "gap": upper - lower if isfinite(upper) else None,
        "cost_closed": isfinite(upper) and upper == lower,
        "unseen": sum(item not in records for item in feasible),
        "unknown": sum(records.get(item) == UNKNOWN for item in feasible),
    }


def legal_actions(
    prefix: Iterable[int],
    evidence: Iterable[int],
    cost: Callable[[Iterable[int]], int],
    budget: int,
) -> list[int | str]:
    sequence = tuple(prefix)
    last = sequence[-1] if sequence else 0
    actions = [
        source_id
        for source_id in sorted(set(evidence))
        if source_id > last and cost(frozenset(sequence) | {source_id}) <= budget
    ]
    return [*actions, STOP]


def logsumexp(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    if not values:
        raise ValueError("logsumexp requires at least one value")
    shift = max(values)
    return shift + log(sum(exp(value - shift) for value in values))


def sequence_logps(
    accepted: Iterable[Iterable[int]],
    evidence: Iterable[int],
    cost: Callable[[Iterable[int]], int],
    budget: int,
    logits_fn: Callable[[tuple[int, ...]], Mapping[int | str, float]],
) -> dict[frozenset[int], float]:
    """Compute complete canonical source-sequence probabilities including STOP."""
    terminals = sorted(
        {frozenset(item) for item in accepted if reachable(item, cost, budget)},
        key=canonical,
    )
    universe = frozenset(evidence)
    node_logps: dict[tuple[int, ...], dict[int | str, float]] = {}
    result: dict[frozenset[int], float] = {}
    for terminal in terminals:
        if not terminal <= universe:
            raise ValueError("terminal outside evidence")
        sequence = canonical(terminal)
        value = 0.0
        for index in range(len(sequence) + 1):
            prefix = sequence[:index]
            if prefix not in node_logps:
                legal = legal_actions(prefix, universe, cost, budget)
                raw = dict(logits_fn(prefix))
                if set(raw) != set(legal) or any(not isfinite(float(item)) for item in raw.values()):
                    raise ValueError("supply finite logits for ALL online-legal actions")
                normalizer = logsumexp(raw.values())
                node_logps[prefix] = {action: float(raw[action]) - normalizer for action in legal}
            action: int | str = sequence[index] if index < len(sequence) else STOP
            value += node_logps[prefix][action]
        result[terminal] = value
    return result


def sequence_risk(
    logps: Mapping[frozenset[int], float],
    cost: Callable[[Iterable[int]], int],
    budget: int,
) -> dict[str, Any]:
    """Numerical reference for equation (13); training uses the torch form."""
    if type(budget) is not int or budget <= 0 or not logps:
        raise ValueError("positive budget and nonempty archive required")
    if any(not isfinite(float(value)) for value in logps.values()):
        raise ValueError("finite log probabilities required")
    values = {frozenset(item): cost(item) for item in logps}
    if any(not 0 <= value <= budget for value in values.values()):
        raise ValueError("terminal cost outside budget")
    log_mass = logsumexp(logps.values())
    if log_mass > 1e-8:
        raise ValueError("terminal probability mass exceeds one")
    best = min(values.values())
    conditional = {item: exp(float(lp) - log_mass) for item, lp in logps.items()}
    # Equation (25) normalizes the soft cost preference by the cheapest
    # successful terminal cost.  A zero-cost empty terminal has no relative
    # cost spread, so its denominator is treated as one and contributes zero
    # regret when it is the only/cheapest solution.
    denominator = max(best, 1)
    regret = sum(conditional[item] * (values[item] - best) / denominator for item in values)
    return {
        "loss": -log_mass + regret,
        "fit": -log_mass,
        "risk": regret,
        "conditional": conditional,
    }


def sequence_risk_torch(logps: Any, terminal_costs: Any, budget: int) -> Any:
    """Autograd implementation of equation (13), kept optional at import time."""
    if budget <= 0 or getattr(logps, "numel", lambda: 0)() == 0:
        raise ValueError("positive budget and nonempty archive required")
    import torch

    costs = torch.as_tensor(terminal_costs, dtype=logps.dtype, device=logps.device)
    log_mass = torch.logsumexp(logps, dim=0)
    rho = torch.softmax(logps, dim=0)
    best = costs.min()
    denominator = torch.clamp(best, min=1.0)
    regret = ((costs - best) / denominator * rho).sum()
    return -log_mass + regret
