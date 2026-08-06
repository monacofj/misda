# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""On-demand heavy evaluation for refactored static MISDA results."""

import time

import numpy as np

from ._reconstruction import (
    _derive_seed,
    evaluate_nonlinear_reconstruction,
    evaluate_null_reconstruction,
)
from ._validation import normalize_mis_selection
from .result import MISDAResult


def heavy(
    result,
    selection,
    *,
    null_reference=False,
    cancel_requested=None,
    _evaluator=None,
    _null_evaluator=None,
):
    """Complement selected MIS evaluations with nonlinear reconstruction.

    ``selection`` accepts one MIS index, a ``range``, or an explicit sequence
    of indices. Existing light and heavy metrics are preserved. The private
    evaluator hooks support deterministic unit tests without weakening the
    public, data-driven protocol.
    """

    if not isinstance(result, MISDAResult):
        raise TypeError("result must be a refactored MISDAResult.")
    if not isinstance(null_reference, (bool, np.bool_)):
        raise TypeError("null_reference must be a boolean.")
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None.")

    selected_mis = normalize_mis_selection(selection, len(result.mis))
    evaluator = _evaluator or evaluate_nonlinear_reconstruction
    null_evaluator = _null_evaluator or evaluate_null_reconstruction
    started = time.perf_counter()
    labels = tuple(
        result.analysis.structural_graph.nodes[index]["label"]
        for index in range(result.analysis.original_dimension)
    )

    for mis_index in selected_mis:
        candidate = result.mis[mis_index]
        nonlinear = candidate.evaluation.get("nonlinear_reconstruction")
        if nonlinear is None:
            nonlinear = evaluator(
                result._data,
                candidate.indices,
                labels,
                seed=_derive_seed(result.execution.seed, 8001, mis_index),
                cancel_requested=cancel_requested,
            )
            candidate.evaluation["nonlinear_reconstruction"] = nonlinear

        if nonlinear.get("cancelled", False):
            break
        if null_reference and "null_reference" not in nonlinear:
            nonlinear["null_reference"] = null_evaluator(
                result._data,
                candidate.indices,
                labels,
                nonlinear,
                seed=_derive_seed(result.execution.seed, 8002, mis_index),
                cancel_requested=cancel_requested,
                evaluator=evaluator,
            )

        null_result = nonlinear.get("null_reference")
        if null_result is not None and null_result.get("cancelled", False):
            break

    result.analysis.n_heavy_mis = sum(
        "nonlinear_reconstruction" in candidate.evaluation
        for candidate in result.mis
    )
    result.execution.timings["heavy"] = (
        result.execution.timings.get("heavy", 0.0)
        + time.perf_counter()
        - started
    )
    return result
