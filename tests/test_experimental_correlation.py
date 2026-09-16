# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import inspect

import numpy as np
import pytest

import misda
import misda.api as api
from misda._correlation_backend import (
    compute_correlation_statistics,
    correlation_backend,
)
from misda._validation import normalize_input_matrix


def _linear_data(n=24):
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack((x, 2.0 * x))


def test_discover_exposes_correlation_and_experimental_gate():
    parameters = inspect.signature(misda.discover).parameters
    assert parameters["correlation"].default == "pearson"
    assert parameters["experimental"].default is False
    assert api.discover is misda.discover


def test_default_and_explicit_pearson_are_identical():
    data = _linear_data()
    default = misda.discover(data, seed=17)
    explicit = misda.discover(
        data,
        seed=17,
        correlation="pearson",
        experimental=False,
    )
    opted_in = misda.discover(
        data,
        seed=17,
        correlation="pearson",
        experimental=True,
    )

    for result in (default, explicit, opted_in):
        assert result.correlation == "pearson"
        assert result.experimental is False

    assert default.analysis.latent_dimension == explicit.analysis.latent_dimension
    assert default.analysis.structural_dimension == explicit.analysis.structural_dimension
    assert tuple(default.analysis.structural_graph.edges()) == tuple(
        explicit.analysis.structural_graph.edges()
    )
    assert tuple(candidate.indices for candidate in default) == tuple(
        candidate.indices for candidate in explicit
    )
    assert tuple(candidate.indices for candidate in explicit) == tuple(
        candidate.indices for candidate in opted_in
    )


def test_spearman_requires_explicit_experimental_opt_in():
    with pytest.raises(ValueError, match="Spearman correlation is experimental"):
        misda.discover(_linear_data(), correlation="spearman")


def test_spearman_runs_when_explicitly_enabled():
    result = misda.discover(
        _linear_data(),
        seed=19,
        correlation="spearman",
        experimental=True,
    )
    assert result.correlation == "spearman"
    assert result.experimental is True
    assert result.analysis.latent_dimension == 1
    assert result.analysis.structural_dimension == 1


def test_invalid_correlation_and_experimental_types_are_rejected():
    data = _linear_data()
    with pytest.raises(ValueError, match="correlation must be"):
        misda.discover(data, correlation="kendall", experimental=True)
    with pytest.raises(TypeError, match="correlation must be"):
        misda.discover(data, correlation=None, experimental=True)
    with pytest.raises(TypeError, match="experimental must be a boolean"):
        misda.discover(data, experimental=1)


def test_backend_dispatch_really_changes_the_dependence_coefficient():
    x = np.linspace(0.0, 1.0, 300)
    y = np.exp(650.0 * (x - 1.0))
    normalized = normalize_input_matrix(np.column_stack((x, y)))

    with correlation_backend("pearson", False):
        pearson = compute_correlation_statistics(normalized).correlation[0, 1]
    with correlation_backend("spearman", True):
        spearman = compute_correlation_statistics(normalized).correlation[0, 1]

    assert pearson < 0.2
    assert spearman == pytest.approx(1.0, abs=1e-12)
