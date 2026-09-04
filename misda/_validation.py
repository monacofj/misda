# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Input normalization and discovery-argument validation for MISDA."""

from dataclasses import dataclass
from numbers import Real
from typing import Any, Tuple

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
    """Normalize an input matrix without discarding constant objectives."""

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


def validate_aggressiveness(value) -> float:
    """Return ``value`` as float when it belongs to the closed unit interval."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("aggressiveness must be a real number in [0, 1].")
    normalized = float(value)
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("aggressiveness must be in [0, 1].")
    return normalized
