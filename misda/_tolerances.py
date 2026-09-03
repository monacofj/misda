# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Versioned numerical tolerances for portable regression gates."""

import numpy as np


GATE_TOLERANCE_VERSION = 1
GATE_RTOL = 0.0
GATE_ATOL = 1e-12


def gate_isclose(left, right) -> bool:
    """Compare floating-point gate metrics using the versioned policy."""

    return bool(np.isclose(left, right, rtol=GATE_RTOL, atol=GATE_ATOL))
