# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Internal correlation-backend selection for MISDA discovery.

Pearson remains the canonical MISDA backend.  Spearman is preserved as an
explicitly gated research backend.  A ContextVar selects the active backend so
all discovery calls share the same implementation pipeline without mutating
module globals per call, and concurrent calls can select different backends
safely.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

from ._statistics import (
    compute_correlation_statistics as _pearson_statistics,
    estimate_null_positive_correlation as _pearson_null,
)
from ._statistics_spearman import (
    compute_correlation_statistics as _spearman_statistics,
    estimate_null_positive_correlation as _spearman_null,
)


_CANONICAL = "pearson"
_EXPERIMENTAL = frozenset({"spearman"})
_ALLOWED = frozenset({_CANONICAL, *_EXPERIMENTAL})
_CURRENT = ContextVar("misda_correlation_backend", default=_CANONICAL)


def validate_correlation_backend(correlation, experimental):
    """Validate and return the requested correlation backend name."""

    if not isinstance(correlation, str):
        raise TypeError("correlation must be 'pearson' or 'spearman'.")
    if correlation not in _ALLOWED:
        raise ValueError("correlation must be 'pearson' or 'spearman'.")
    if not isinstance(experimental, bool):
        raise TypeError("experimental must be a boolean.")
    if correlation in _EXPERIMENTAL and not experimental:
        raise ValueError(
            "Spearman correlation is experimental and is not part of the "
            "canonical MISDA method. Pass experimental=True to enable it "
            "for research purposes."
        )
    return correlation


@contextmanager
def correlation_backend(correlation, experimental):
    """Temporarily select a validated discovery correlation backend."""

    name = validate_correlation_backend(correlation, experimental)
    token = _CURRENT.set(name)
    try:
        yield name
    finally:
        _CURRENT.reset(token)


def current_correlation_backend():
    """Return the backend selected in the current execution context."""

    return _CURRENT.get()


def compute_correlation_statistics(normalized):
    """Dispatch correlation statistics to the active backend."""

    if _CURRENT.get() == "spearman":
        return _spearman_statistics(normalized)
    return _pearson_statistics(normalized)


def estimate_null_positive_correlation(normalized, **kwargs):
    """Dispatch null-envelope estimation to the active backend."""

    if _CURRENT.get() == "spearman":
        return _spearman_null(normalized, **kwargs)
    return _pearson_null(normalized, **kwargs)
