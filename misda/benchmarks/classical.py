"""Reproducible generators for classical DTLZ reference problems.

These problems are external/reference MOPs, not controlled MISDA diagnostics.
Known Pareto-front geometry is useful context but is not converted into MISDA
latent or structural ground truth.
"""

from __future__ import annotations

import math

import numpy as np


def _validate_dtlz_parameters(N: int, M: int, n_vars: int) -> tuple[int, int, int]:
    N = int(N)
    M = int(M)
    n_vars = int(n_vars)
    if N < 1:
        raise ValueError("N must be at least 1")
    if M < 2:
        raise ValueError("M must be at least 2")
    if n_vars < M:
        raise ValueError("n_vars must be at least M so k=n_vars-M+1 is positive")
    return N, M, n_vars


def generate_dtlz2(
    N: int = 1000,
    M: int = 3,
    n_vars: int = 12,
    on_front: bool = False,
    seed: int = 123,
):
    """Generate a reproducible DTLZ2 sample.

    When ``on_front=True`` the distance variables are fixed at 0.5, so ``g=0``
    and the sampled objectives lie on the positive-orthant unit hypersphere.
    """
    N, M, n_vars = _validate_dtlz_parameters(N, M, n_vars)
    rng = np.random.default_rng(int(seed))
    X = rng.uniform(0.0, 1.0, size=(N, n_vars))
    if on_front:
        X[:, M - 1 :] = 0.5

    g = np.sum((X[:, M - 1 :] - 0.5) ** 2, axis=1)
    F = np.empty((N, M), dtype=float)
    for objective in range(M):
        values = 1.0 + g
        for variable in range(M - 1 - objective):
            values = values * np.cos(X[:, variable] * math.pi / 2.0)
        if objective > 0:
            values = values * np.sin(
                X[:, M - 1 - objective] * math.pi / 2.0
            )
        F[:, objective] = values
    return F, X


def generate_dtlz5(
    N: int = 1000,
    M: int = 3,
    n_vars: int = 12,
    on_front: bool = False,
    seed: int = 123,
):
    """Generate a reproducible DTLZ5 sample.

    On the Pareto front (``g=0``), all angular coordinates except the first are
    fixed at pi/4. The objective front is therefore a one-dimensional curve even
    when ``M>2``.
    """
    N, M, n_vars = _validate_dtlz_parameters(N, M, n_vars)
    rng = np.random.default_rng(int(seed))
    X = rng.uniform(0.0, 1.0, size=(N, n_vars))
    if on_front:
        X[:, M - 1 :] = 0.5

    g = np.sum((X[:, M - 1 :] - 0.5) ** 2, axis=1)
    theta = np.empty((N, M - 1), dtype=float)
    theta[:, 0] = X[:, 0] * math.pi / 2.0
    for index in range(1, M - 1):
        theta[:, index] = (
            math.pi
            / (4.0 * (1.0 + g))
            * (1.0 + 2.0 * g * X[:, index])
        )

    F = np.empty((N, M), dtype=float)
    for objective in range(M):
        values = 1.0 + g
        for angle in range(M - 1 - objective):
            values = values * np.cos(theta[:, angle])
        if objective > 0:
            values = values * np.sin(theta[:, M - 1 - objective])
        F[:, objective] = values
    return F, X


CLASSICAL_MOPS = {
    "dtlz2": {
        "name": "DTLZ2",
        "generator": generate_dtlz2,
        "pareto_geometry": "positive-orthant unit hypersphere",
        "pareto_manifold_dimension": lambda M: int(M - 1),
    },
    "dtlz5": {
        "name": "DTLZ5",
        "generator": generate_dtlz5,
        "pareto_geometry": "degenerate one-dimensional curve",
        "pareto_manifold_dimension": lambda M: 1,
    },
}
