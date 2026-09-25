"""Tests for observed dominance-preservation ranking and trust annotation."""

from types import SimpleNamespace

import numpy as np

import misda
import misda.api as api


def _candidate(labels, indices, rate):
    return api.MISCandidate(
        objectives=tuple(labels),
        indices=tuple(indices),
        structural=api.StructuralMetrics(
            neighborhood=0,
            neighborhood_ratio=1.0,
            span=0,
            avg_external_degree=0.0,
            avg_internal_degree=0.0,
        ),
        dominance=api.DominanceMetrics(
            new_dominance_rate=float(rate),
            new_dominance_pairs=int(round(rate * 10)),
            original_no_dominance_pairs=10,
            exact_preservation=rate == 0.0,
        ),
    )


def _support(index, status, reasons=()):
    return api.CandidateSupport(
        candidate_index=index,
        status=status,
        reasons=tuple(reasons),
        transitivity_observed=0.0,
        transitivity_null=0.0,
        transitivity_excess=0.0,
        spectral_tested_dimension=1,
        spectral_observed_next_eigenvalue=0.0,
        spectral_null_next_eigenvalue=0.0,
        spectral_excess=0.0,
        n_permutations=8,
        seed=123,
    )


def _mis_set(candidates, *, original_dimension=3, supports=None):
    result = api.MISSet(
        analysis=SimpleNamespace(original_dimension=original_dimension),
        candidates=tuple(candidates),
        rank_groups=tuple((i,) for i in range(len(candidates))),
        data=np.zeros((4, original_dimension), dtype=float),
        labels=tuple(f'f{i + 1}' for i in range(original_dimension)),
        seed=123,
    )
    result._support_by_index = {
        item.candidate_index: item for item in (supports or ())
    }
    return result


def test_dominance_policy_orders_lower_new_dominance_first():
    result = _mis_set(
        [
            _candidate(('f1', 'f2'), (0, 1), 0.30),
            _candidate(('f1', 'f3'), (0, 2), 0.10),
            _candidate(('f2',), (1,), 0.20),
        ],
        supports=[
            _support(0, api.SUPPORTED),
            _support(1, api.SUPPORTED),
            _support(2, api.SUPPORTED),
        ],
    )

    ranking = misda.rank(result, policy=misda.DOMINANCE_PRESERVATION)

    assert ranking.indices == (1, 2, 0)
    assert ranking.selected is result[1]
    assert ranking.assessment.status == misda.SUPPORTED_REDUCTION


def test_dominance_policy_keeps_scientific_tie_but_orders_larger_mis_first():
    result = _mis_set(
        [
            _candidate(('f2',), (1,), 0.10),
            _candidate(('f1', 'f3'), (0, 2), 0.10),
        ],
        supports=[
            _support(0, api.SUPPORTED),
            _support(1, api.SUPPORTED),
        ],
    )

    ranking = misda.rank(result, policy=misda.DOMINANCE_PRESERVATION)

    assert ranking.indices == (1, 0)
    assert set(ranking.groups[0]) == {0, 1}


def test_unsupported_reduction_is_returned_and_annotated_not_trusted():
    result = _mis_set(
        [_candidate(('f1', 'f2'), (0, 1), 0.05)],
        supports=[_support(0, api.UNSUPPORTED, ('TRANSITIVE_CHAINING',))],
    )

    ranking = misda.rank(result, policy=misda.DOMINANCE_PRESERVATION)

    assert ranking.mis() is result[0]
    assert ranking.assessment.status == misda.UNSUPPORTED_REDUCTION
    assert ranking.assessment.trustworthy is False
    assert ranking.assessment.reasons == ('TRANSITIVE_CHAINING',)


def test_no_redundancy_is_reported_when_selected_mis_keeps_all_objectives():
    result = _mis_set(
        [_candidate(('f1', 'f2', 'f3'), (0, 1, 2), 0.0)],
        supports=[_support(0, api.SUPPORTED)],
    )

    ranking = misda.rank(result, policy=misda.DOMINANCE_PRESERVATION)

    assert ranking.mis() is result[0]
    assert ranking.assessment.status == misda.NO_REDUNDANCY
    assert ranking.assessment.trustworthy is True


def test_public_dominance_evaluation_matches_simple_pair_count():
    data = np.array(
        [
            [0.0, 2.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ]
    )
    prepared = api.prepare_dominance_pairs(data)
    raw = api.evaluate_dominance_preservation(prepared, (0,))

    assert raw['original_no_dominance_pairs'] == 3
    assert raw['new_dominance_pairs'] == 3
    assert raw['new_dominance_rate'] == 1.0
    assert raw['exact_preservation'] is False
