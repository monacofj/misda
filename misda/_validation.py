# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Input normalization and argument validation for MISDA."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype


@dataclass(frozen=True)
class NormalizedInput:
    """Validated numeric input and metadata needed by later analysis stages."""

    data: np.ndarray
    labels: Tuple[Any, ...]
    constant_indices: Tuple[int, ...]

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def n_objectives(self) -> int:
        return self.data.shape[1]

    @property
    def constant_labels(self) -> Tuple[Any, ...]:
        return tuple(self.labels[index] for index in self.constant_indices)

    @property
    def constant_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_objectives, dtype=bool)
        mask[list(self.constant_indices)] = True
        return mask


def _format_columns(labels) -> str:
    return ", ".join(repr(label) for label in labels)


def _is_real_numeric_array(value: np.ndarray) -> bool:
    dtype = value.dtype
    return (
        np.issubdtype(dtype, np.number)
        and not np.issubdtype(dtype, np.complexfloating)
        and not np.issubdtype(dtype, np.bool_)
    )


def normalize_input_matrix(Y, *, require_fisher_z: bool = True) -> NormalizedInput:
    """Normalize an input matrix without discarding constant objectives.

    Only two-dimensional NumPy arrays and pandas DataFrames are accepted.  The
    returned array is an independent ``float`` copy; DataFrame column labels are
    preserved, while arrays receive the deterministic labels ``f1`` through
    ``fM``.  Constant objectives are recorded for explicit treatment by the
    statistical and evaluation layers.
    """

    if isinstance(Y, pd.DataFrame):
        if Y.ndim != 2:
            raise ValueError("Input data must be a two-dimensional matrix.")
        invalid_labels = [
            label
            for label, dtype in zip(Y.columns, Y.dtypes)
            if (
                not is_numeric_dtype(dtype)
                or is_bool_dtype(dtype)
                or is_complex_dtype(dtype)
            )
        ]
        if invalid_labels:
            raise TypeError(
                "Input data contains non-numeric columns: "
                f"{_format_columns(invalid_labels)}."
            )
        labels = tuple(Y.columns)
        data = Y.to_numpy(dtype=float, copy=True)
    elif isinstance(Y, np.ndarray):
        if Y.ndim != 2:
            raise ValueError("Input data must be a two-dimensional matrix.")
        if not _is_real_numeric_array(Y):
            raise TypeError("Input data must contain only real numeric values.")
        labels = tuple(f"f{index + 1}" for index in range(Y.shape[1]))
        data = np.array(Y, dtype=float, copy=True)
    else:
        raise TypeError("Input data must be a numpy.ndarray or pandas.DataFrame.")

    n_samples, n_objectives = data.shape
    if n_samples == 0 or n_objectives == 0:
        raise ValueError("Input data must not be empty.")
    if require_fisher_z and n_samples < 4:
        raise ValueError(
            "Input data requires at least four observations for Fisher-z analysis."
        )

    finite_by_column = np.all(np.isfinite(data), axis=0)
    if not np.all(finite_by_column):
        invalid_labels = [
            labels[index]
            for index in np.flatnonzero(~finite_by_column)
        ]
        raise ValueError(
            "Input data contains non-finite values in columns: "
            f"{_format_columns(invalid_labels)}."
        )

    constant_indices = tuple(
        int(index)
        for index in np.flatnonzero(np.ptp(data, axis=0) == 0)
    )
    return NormalizedInput(
        data=data,
        labels=labels,
        constant_indices=constant_indices,
    )


def _validate_input_matrix(Y):
    """Transitional adapter for the suspended adaptive implementation."""

    normalized = normalize_input_matrix(Y)
    return normalized.data, list(normalized.labels)


def validate_aggressiveness(value) -> float:
    """Return ``value`` as float when it belongs to the closed unit interval."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("aggressiveness must be a real number in [0, 1].")
    normalized = float(value)
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("aggressiveness must be in [0, 1].")
    return normalized


def validate_rank_policy(value) -> str:
    """Validate the currently supported ranking policy."""

    if not isinstance(value, str):
        raise TypeError("rank_policy must be a string.")
    if value != "default":
        raise ValueError("rank_policy must be 'default'.")
    return value


def validate_max_evaluated_mis(value) -> Optional[int]:
    """Validate the optional positive limit on normally evaluated MISs."""

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("max_evaluated_mis must be None or a positive integer.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError("max_evaluated_mis must be a positive integer.")
    return normalized


def normalize_mis_selection(selection, n_mis) -> Tuple[int, ...]:
    """Normalize an index, range, or index sequence for future ``heavy()`` use."""

    if isinstance(n_mis, (bool, np.bool_)) or not isinstance(n_mis, Integral):
        raise TypeError("n_mis must be a non-negative integer.")
    n_mis = int(n_mis)
    if n_mis < 0:
        raise ValueError("n_mis must be a non-negative integer.")

    if isinstance(selection, (bool, np.bool_)):
        raise TypeError("selection must be an index, range, or sequence of indices.")
    if isinstance(selection, Integral):
        indices = (int(selection),)
    elif isinstance(selection, range):
        indices = tuple(selection)
    elif isinstance(selection, np.ndarray):
        if selection.ndim != 1:
            raise ValueError("selection array must be one-dimensional.")
        indices = tuple(selection.tolist())
    elif isinstance(selection, (list, tuple)):
        indices = tuple(selection)
    else:
        raise TypeError("selection must be an index, range, or sequence of indices.")

    if not indices:
        raise ValueError("selection must contain at least one MIS index.")
    if any(
        isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
        for index in indices
    ):
        raise TypeError("selection indices must be integers.")

    normalized = tuple(int(index) for index in indices)
    if len(set(normalized)) != len(normalized):
        raise ValueError("selection must not contain duplicate MIS indices.")
    invalid = [index for index in normalized if index < 0 or index >= n_mis]
    if invalid:
        raise IndexError(
            f"selection indices out of range for {n_mis} MISs: {invalid}."
        )
    return normalized
