"""Characterization tests for the extracted legacy reconstruction layer."""

import numpy as np
import pandas as pd
import pytest

import misda
from misda import _reconstruction


PUBLIC_RECONSTRUCTION_OPERATIONS = (
    "_calculate_ses_core",
    "calculate_ses_linear",
    "calculate_ses",
    "calculate_ses_nonlinear",
)


@pytest.fixture
def reconstruction_example():
    rng = np.random.default_rng(20260806)
    source = rng.normal(size=120)
    target = 1.75 * source + rng.normal(scale=0.1, size=120)
    noise = rng.normal(size=120)
    return np.column_stack([source, target, noise])


@pytest.mark.parametrize("name", PUBLIC_RECONSTRUCTION_OPERATIONS)
def test_legacy_reconstruction_operations_are_reexported_from_package(name):
    assert getattr(misda, name) is getattr(_reconstruction, name)


def test_linear_reconstruction_values_are_unchanged(reconstruction_example):
    observed = misda.calculate_ses_linear(
        reconstruction_example,
        mis=[0, 2],
        n_perm=4,
        test_size=0.25,
        seed=17,
        return_details=True,
    )

    assert observed == {
        "ses": 0.9964603491700306,
        "F_real": 0.9964603491700306,
        "F_null": 0.0,
        "mis_size": 2,
        "M": 3,
        "N": 120,
        "targets_reconstructed": ["f2"],
        "r2_real": {"f2": 0.9964603491700306},
        "r2_null": {"f2": -0.10537488968189196},
        "status": "SUCCESS",
        "model_type": "linear",
        "settings": {
            "n_perm": 4,
            "test_size": 0.25,
            "seed": 17,
            "clip": True,
        },
    }


def test_calculate_ses_remains_exact_linear_alias(reconstruction_example):
    kwargs = {
        "mis": [0, 2],
        "n_perm": 4,
        "test_size": 0.25,
        "seed": 17,
        "return_details": True,
    }

    assert misda.calculate_ses(reconstruction_example, **kwargs) == (
        misda.calculate_ses_linear(reconstruction_example, **kwargs)
    )


def test_dataframe_labels_are_preserved(reconstruction_example):
    frame = pd.DataFrame(
        reconstruction_example,
        columns=["source", "target", "noise"],
    )

    observed = misda.calculate_ses_linear(
        frame,
        mis=["source", "noise"],
        n_perm=4,
        test_size=0.25,
        seed=17,
        return_details=True,
    )

    assert observed["targets_reconstructed"] == ["target"]
    assert observed["r2_real"] == {"target": 0.9964603491700306}
    assert observed["r2_null"] == {"target": -0.10537488968189196}


def test_no_reduction_schema_is_unchanged(reconstruction_example):
    observed = misda.calculate_ses_linear(
        reconstruction_example,
        mis=[0, 1, 2],
        return_details=True,
    )

    assert observed["status"] == "NO_REDUCTION"
    assert observed["ses"] is None
    assert observed["F_real"] is None
    assert observed["F_null"] is None
    assert observed["targets_reconstructed"] == []


@pytest.mark.parametrize(
    ("mis", "kwargs", "message"),
    [
        ([], {}, "mis cannot be empty"),
        ([3], {}, "mis contains index outside"),
        ([0], {"test_size": 0.0}, "test_size must be strictly between"),
        ([0], {"n_perm": 0}, "n_perm must be at least 1"),
    ],
)
def test_legacy_validation_errors_are_unchanged(
    reconstruction_example, mis, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        misda.calculate_ses_linear(reconstruction_example, mis=mis, **kwargs)


def test_nonlinear_reconstruction_remains_reproducible(reconstruction_example):
    kwargs = {
        "mis": [0, 2],
        "n_perm": 2,
        "test_size": 0.25,
        "seed": 17,
        "n_estimators": 8,
        "return_details": True,
    }

    first = misda.calculate_ses_nonlinear(reconstruction_example, **kwargs)
    second = misda.calculate_ses_nonlinear(reconstruction_example, **kwargs)

    assert first == second
    assert first["model_type"] == "nonlinear"
    assert first["status"] == "SUCCESS"
