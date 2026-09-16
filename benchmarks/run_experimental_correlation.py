"""Run an existing benchmark module with Spearman explicitly enabled.

This is a research harness, not a second MISDA implementation.  It wraps the
public ``misda.discover`` entry point so existing benchmark drivers exercise
the same pipeline with ``correlation='spearman', experimental=True``.

Usage:
    python -m benchmarks.run_experimental_correlation benchmarks.run_monotonic \
        --output /tmp/result.json
"""

from __future__ import annotations

import runpy
import sys

import misda
import misda.api as api


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_experimental_correlation MODULE [ARGS ...]")

    module = sys.argv[1]
    if not module.startswith("benchmarks."):
        raise SystemExit("MODULE must be inside the benchmarks package")

    canonical_discover = misda.discover

    def experimental_spearman_discover(Y, **kwargs):
        kwargs.setdefault("correlation", "spearman")
        kwargs.setdefault("experimental", True)
        return canonical_discover(Y, **kwargs)

    misda.discover = experimental_spearman_discover
    api.discover = experimental_spearman_discover
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
