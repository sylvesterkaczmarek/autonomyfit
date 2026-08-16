from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from math import inf

from .models import Objective, Recommendation

_METRICS = ("latency", "throughput", "accuracy", "power", "memory")


def _costs(item: Recommendation) -> dict[str, float]:
    """Return known objective values as costs, where smaller is always better."""
    values: dict[str, float] = {}
    if item.estimated_memory_gb is not None:
        values["memory"] = item.estimated_memory_gb
    if item.latency_ms is not None:
        values["latency"] = item.latency_ms
    if item.fps is not None:
        values["throughput"] = -item.fps
    if item.model.accuracy is not None:
        accuracy = item.model.accuracy
        values["accuracy"] = -accuracy.value if accuracy.higher_is_better else accuracy.value
    if item.benchmark is not None and item.benchmark.power.mean_w is not None:
        values["power"] = item.benchmark.power.mean_w
    return values


def dominates(a: Recommendation, b: Recommendation, *, tolerance: float = 1e-9) -> bool:
    """Conservative Pareto dominance with explicit missing-data handling.

    Candidates are compared only when they expose the same known objective set. Missing
    evidence therefore never creates a Pareto advantage.
    """
    a_costs = _costs(a)
    b_costs = _costs(b)
    if not b_costs or set(a_costs) != set(b_costs):
        return False
    no_worse = all(a_costs[key] <= value + tolerance for key, value in b_costs.items())
    strictly_better = any(a_costs[key] < value - tolerance for key, value in b_costs.items())
    return no_worse and strictly_better


def pareto_layers(items: Iterable[Recommendation]) -> tuple[dict[str, int], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    candidates = list(items)
    dominates_map: dict[str, set[str]] = {item.model.id: set() for item in candidates}
    dominated_by_map: dict[str, set[str]] = {item.model.id: set() for item in candidates}
    for left in candidates:
        for right in candidates:
            if left.model.id == right.model.id:
                continue
            if dominates(left, right):
                dominates_map[left.model.id].add(right.model.id)
                dominated_by_map[right.model.id].add(left.model.id)

    remaining = {item.model.id for item in candidates}
    ranks: dict[str, int] = {}
    rank = 0
    while remaining:
        frontier = sorted(
            model_id
            for model_id in remaining
            if not (dominated_by_map[model_id] & remaining)
        )
        if not frontier:  # defensive cycle guard for floating-point edge cases
            frontier = sorted(remaining)
        for model_id in frontier:
            ranks[model_id] = rank
            remaining.remove(model_id)
        rank += 1
    return (
        ranks,
        {key: tuple(sorted(value)) for key, value in dominates_map.items()},
        {key: tuple(sorted(value)) for key, value in dominated_by_map.items()},
    )


def _normalized_utilities(items: list[Recommendation]) -> dict[str, float]:
    costs = {item.model.id: _costs(item) for item in items}
    ranges: dict[str, tuple[float, float]] = {}
    for metric in _METRICS:
        values = [value[metric] for value in costs.values() if metric in value]
        if values:
            ranges[metric] = (min(values), max(values))

    result: dict[str, float] = {}
    for item in items:
        values = costs[item.model.id]
        utilities: list[float] = []
        for metric in _METRICS:
            if metric not in values or metric not in ranges:
                continue
            lower, upper = ranges[metric]
            if abs(upper - lower) < 1e-12:
                utilities.append(1.0)
            else:
                utilities.append((upper - values[metric]) / (upper - lower))
        coverage = len(utilities) / len(_METRICS)
        result[item.model.id] = (sum(utilities) / len(utilities) if utilities else 0.0) * coverage
    return result


def _verdict_bucket(item: Recommendation) -> int:
    return {
        "VERIFIED_FIT": 0,
        "FEASIBLE": 1,
        "BENCHMARK_REQUIRED": 2,
        "CONSTRAINT_FAIL": 3,
        "NO_FIT": 4,
    }[item.verdict]


def _objective_value(item: Recommendation, objective: Objective) -> float:
    if objective == "latency":
        return item.latency_ms if item.latency_ms is not None else inf
    if objective == "throughput":
        return -item.fps if item.fps is not None else inf
    if objective == "accuracy":
        if item.model.accuracy is None:
            return inf
        metric = item.model.accuracy
        return -metric.value if metric.higher_is_better else metric.value
    if objective == "power":
        if item.benchmark is None or item.benchmark.power.mean_w is None:
            return inf
        return item.benchmark.power.mean_w
    if objective == "memory":
        return item.estimated_memory_gb if item.estimated_memory_gb is not None else inf
    raise ValueError(f"objective {objective!r} has no scalar value")


def rank_recommendations(items: list[Recommendation], objective: Objective) -> list[Recommendation]:
    if not items:
        return []
    feasible = [item for item in items if item.verdict not in {"NO_FIT", "CONSTRAINT_FAIL"}]
    ranks, dominates_map, dominated_by_map = pareto_layers(feasible)
    balanced = _normalized_utilities(feasible)
    objective_values = {
        item.model.id: _objective_value(item, objective)
        for item in feasible
        if objective != "balanced"
    }
    finite_values = [value for value in objective_values.values() if value != inf]
    objective_range = (min(finite_values), max(finite_values)) if finite_values else None

    enriched: list[Recommendation] = []
    for item in items:
        utility = balanced.get(item.model.id, 0.0)
        if objective == "balanced":
            score = 100.0 * utility
        else:
            value = objective_values.get(item.model.id, inf)
            if value == inf or objective_range is None:
                score = 0.0
            else:
                lower, upper = objective_range
                score = 100.0 if abs(upper - lower) < 1e-12 else 100.0 * (upper - value) / (upper - lower)
        enriched.append(
            replace(
                item,
                score=round(score, 3),
                pareto_rank=ranks.get(item.model.id),
                dominates=dominates_map.get(item.model.id, ()),
                dominated_by=dominated_by_map.get(item.model.id, ()),
                objective=objective,
            )
        )

    def key(item: Recommendation):
        confidence = item.confidence.score if item.confidence else 0.0
        pareto = item.pareto_rank if item.pareto_rank is not None else 10_000
        if objective == "balanced":
            return (
                _verdict_bucket(item),
                pareto,
                -balanced.get(item.model.id, 0.0),
                -confidence,
                item.model.id,
            )
        return (
            _verdict_bucket(item),
            pareto,
            _objective_value(item, objective),
            -confidence,
            item.model.id,
        )

    ordered = sorted(enriched, key=key)
    return [replace(item, objective_rank=index + 1) for index, item in enumerate(ordered)]
