"""Tests for the experimental size-pareto ranking policy."""

import numpy as np
import pytest

import misda
import misda.api as api


def _pareto(retention, *, full=10):
    intersection = int(round(retention * full))
    return api.ParetoMetrics(
        retention=float(retention),
        validity=1.0,
        jaccard=float(retention),
        full_front_size=full,
        reduced_front_size=intersection,
        intersection_size=intersection,
        union_size=full,
        exact_preservation=intersection == full,
        reduced_front_indices=tuple(range(intersection)),
    )


def _candidate(labels, indices, *, span, retention=None):
    return api.MISCandidate(
        objectives=tuple(labels),
        indices=tuple(indices),
        structural=api.StructuralMetrics(
            neighborhood=0,
            neighborhood_ratio=1.0,
            span=int(span),
            avg_external_degree=0.0,
            avg_internal_degree=0.0,
        ),
        pareto=None if retention is None else _pareto(retention),
    )


def _mis_set(*candidates):
    return api.MISSet(
        analysis=None,
        candidates=candidates,
        rank_groups=((0,), (1,), (2,)),
        data=np.arange(24, dtype=float).reshape(8, 3),
        labels=("f1", "f2", "f3"),
        seed=123,
    )


def test_size_pareto_orders_by_size_then_retention_without_mutating_mis_set():
    result = _mis_set(
        _candidate(("f1", "f2"), (0, 1), span=9, retention=0.4),
        _candidate(("f1", "f3"), (0, 2), span=4, retention=0.8),
        _candidate(("f3",), (2,), span=10, retention=1.0),
    )
    before = tuple(candidate.indices for candidate in result)

    ranking = misda.rank(result, policy=misda.SIZE_PARETO)

    assert tuple(candidate.indices for candidate in result) == before
    assert ranking.policy == "size_pareto"
    assert ranking.indices == (1, 0, 2)
    assert ranking.selected is result[1]


def test_size_pareto_scientific_ties_ignore_deterministic_label_order():
    result = _mis_set(
        _candidate(("z", "a"), (0, 1), span=9, retention=0.8),
        _candidate(("b", "c"), (0, 2), span=4, retention=0.8),
        _candidate(("d",), (2,), span=10, retention=1.0),
    )

    ranking = misda.rank(result, policy=misda.SIZE_PARETO)

    assert set(ranking.groups[0]) == {0, 1}
    assert ranking.groups[1] == (2,)


def test_size_pareto_requires_stored_evidence_without_cost_opt_in():
    result = _mis_set(
        _candidate(("f1", "f2"), (0, 1), span=9),
        _candidate(("f1", "f3"), (0, 2), span=4, retention=0.8),
        _candidate(("f3",), (2,), span=10, retention=1.0),
    )

    with pytest.raises(ValueError, match="requires Pareto evaluation"):
        misda.rank(result, policy=misda.SIZE_PARETO)


def test_size_pareto_accept_cost_authorizes_missing_pareto_evaluation(monkeypatch):
    result = _mis_set(
        _candidate(("f1", "f2"), (0, 1), span=9),
        _candidate(("f1", "f3"), (0, 2), span=4, retention=0.8),
        _candidate(("f3",), (2,), span=10, retention=1.0),
    )
    calls = []

    def fake_evaluate(mis_set, *, metrics, candidates, **kwargs):
        calls.append((tuple(metrics), tuple(candidates)))
        for index in candidates:
            object.__setattr__(mis_set[index], "pareto", _pareto(0.9))
        return mis_set

    monkeypatch.setattr(api, "evaluate", fake_evaluate)

    ranking = api.rank(
        result,
        policy=api.SIZE_PARETO,
        accept_cost=True,
    )

    assert calls == [(("pareto",), (0,))]
    assert ranking.selected is result[0]


def test_default_policy_remains_size_span():
    assert misda.SIZE_SPAN == "size_span"
    assert misda.SIZE_PARETO == "size_pareto"
