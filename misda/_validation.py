# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Legacy input validation used by the MISDA analysis paths."""

import numpy as np


def _validate_input_matrix(Y):
    """
    Validates input matrix Y for finite values and constant objectives (zero range).
    Raises ValueError if data contains NaNs, Infs, or exact constant columns.
    """
    if hasattr(Y, "values"):
        data = np.asarray(Y.values, dtype=float)
        labels = list(Y.columns)
    else:
        data = np.asarray(Y, dtype=float)
        labels = [f"f{i+1}" for i in range(data.shape[1])]

    if not np.all(np.isfinite(data)):
        raise ValueError("Input data Y contains non-finite values (NaNs or Infs).")

    M = data.shape[1]
    for j in range(M):
        col = data[:, j]
        if np.ptp(col) == 0:
            raise ValueError(f"Objective '{labels[j]}' (column index {j}) has zero range (constant objective). Remove uninformative objectives before running MISDA.")
    return data, labels
