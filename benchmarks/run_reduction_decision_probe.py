# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end conservative reduction-decision probe.

The probe never feeds truth into MISDA. It first computes the complete
Y-only decision (candidate + trust annotation), then attaches known control
truth only for validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

import misda
import moeabench as mb
from misda.benchmarks import PROBLEM_BY_ID
from benchmarks.run_ranking_policy_robustness import SafePathMOP


M = 10
N_CONTROLLED = 300
SCREEN_POWER = 9
SAMPLE_SEED = 123
MISDA_SEED = 123


def _rank_frame(name, frame):
    result = misda.discover(frame, name=name, seed=MISDA_SEED)
    result.evaluate(metrics=("dominance", "pareto"), candidates="all")
    ranking = misda.rank(result, policy=misda.DOMINANCE_PRESERVATION)
    selected = ranking.mis()
    assessment = ranking.assessment
    return result, ranking, {
        "case": name,
        "original_dimension": int(result.analysis.original_dimension),
        "structural_dimension": int(result.analysis.structural_dimension),
        "selected_dimension": int(selected.size),
        "selected_indices": [int(i) for i in selected.indices],
        "selected_objectives": [str(x) for x in selected.objectives],
        "new_dominance_rate": float(selected.dominance.new_dominance_rate),
        "pareto_retention": float(selected.pareto.retention),
        "status": assessment.status,
        "trustworthy": bool(assessment.trustworthy),
        "support_status": assessment.support_status,
        "support_reasons": list(assessment.reasons),
    }


def _controlled_case(problem_id):
    problem = PROBLEM_BY_ID[problem_id]
    dataset = problem.generate(N=N_CONTROLLED, seed=SAMPLE_SEED, sigma=0.0)
    result, ranking, row = _rank_frame(problem_id, dataset.Y)
    row['truth_kind'] = 'controlled'
    row['structural_expected'] = int(dataset.truth['structural_expected'])
    row['dimension_matches_truth'] = (
        row['selected_dimension'] == row['structural_expected']
    )
    return result, ranking, row


def _mop_frame(mop):
    sampler = qmc.Sobol(d=mop.N, scramble=True, seed=SAMPLE_SEED)
    X = qmc.scale(
        sampler.random_base2(m=SCREEN_POWER),
        np.asarray(mop.xl, dtype=float),
        np.asarray(mop.xu, dtype=float),
    )
    F = np.asarray(mop.evaluation(X)['F'], dtype=float)
    return pd.DataFrame(F, columns=[f'f{i + 1}' for i in range(mop.M)])


def _mop_case(name, mop, safe_sets, *, all_reductions_unsafe=False):
    result, ranking, row = _rank_frame(name, _mop_frame(mop))
    selected = tuple(row['selected_indices'])
    row['truth_kind'] = 'optimization_control'
    row['selected_known_safe'] = bool(selected in safe_sets)
    row['all_reductions_unsafe'] = bool(all_reductions_unsafe)
    row['unsafe_trusted_reduction'] = bool(
        row['trustworthy']
        and row['selected_dimension'] < row['original_dimension']
        and (all_reductions_unsafe or selected not in safe_sets)
    )
    return result, ranking, row


def run_probe():
    rows = []
    for problem_id in (
        'independence',
        'total_redundancy',
        'blocks_4x5',
        'blocks_2x10',
    ):
        _result, _ranking, row = _controlled_case(problem_id)
        rows.append(row)

    _result, _ranking, row = _mop_case(
        'DTLZ2',
        mb.mops.DTLZ2(M=M),
        safe_sets=set(),
        all_reductions_unsafe=True,
    )
    rows.append(row)

    _result, _ranking, row = _mop_case(
        'DTLZ5',
        mb.mops.DTLZ5(M=M),
        safe_sets={(8, 9)},
    )
    rows.append(row)

    _result, _ranking, row = _mop_case(
        'DPF1',
        mb.mops.DPF1(M=M, D=2, K=5),
        safe_sets={(0, 1)},
    )
    rows.append(row)

    _result, _ranking, row = _mop_case(
        'SAFE_PATH',
        SafePathMOP(),
        safe_sets={(1,), (0, 2)},
    )
    rows.append(row)
    return pd.DataFrame(rows)


def validate_probe(table):
    checks = {}
    by_case = table.set_index('case')

    checks['independence_no_redundancy'] = bool(
        by_case.loc['independence', 'status'] == misda.NO_REDUNDANCY
        and by_case.loc['independence', 'selected_dimension']
        == by_case.loc['independence', 'original_dimension']
    )

    for case in ('total_redundancy', 'blocks_4x5', 'blocks_2x10'):
        checks[f'{case}_supported_reduction'] = bool(
            by_case.loc[case, 'status'] == misda.SUPPORTED_REDUCTION
            and by_case.loc[case, 'dimension_matches_truth']
        )

    checks['dtlz2_not_trusted'] = bool(
        by_case.loc['DTLZ2', 'status'] == misda.UNSUPPORTED_REDUCTION
        and not by_case.loc['DTLZ2', 'trustworthy']
    )
    for case in ('DTLZ5', 'DPF1', 'SAFE_PATH'):
        checks[f'{case.lower()}_candidate_known_safe'] = bool(
            by_case.loc[case, 'selected_known_safe']
        )
        checks[f'{case.lower()}_no_trusted_unsafe_reduction'] = bool(
            not by_case.loc[case, 'unsafe_trusted_reduction']
        )

    return checks


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args(argv)

    table = run_probe()
    checks = validate_probe(table)
    print('\n=== Conservative reduction-decision probe ===')
    print(table.to_string(index=False))
    print('\n=== Checks ===')
    for name, passed in checks.items():
        print(f'{name}: {"PASS" if passed else "FAIL"}')

    if args.output is not None:
        payload = {
            'checks': checks,
            'rows': [
                {key: _json_value(value) for key, value in row.items()}
                for row in table.to_dict(orient='records')
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding='utf-8',
        )

    if args.strict and not all(checks.values()):
        raise SystemExit('reduction-decision probe failed')


if __name__ == '__main__':
    main()
