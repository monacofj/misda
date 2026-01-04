<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

All notable changes to the **MISDA** (Maximal Independent Structural Dimensionality Analysis) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-04

### Changed (Breaking)
- **Separation of Concerns**: `misda.analyze()` no longer runs validation (SES, Pareto) by default. The `run_ses` argument has been removed.
  - **New Workflow**: Call `res = misda.analyze(...)` then `res.validate()`.
- **API Results**: `MISDAResult.mis_sets` and `res.get_mis_by_rank()` now return `MISCandidate` objects instead of dictionaries.
  - **Old**: `sol['mis_labels']`
  - **New**: `sol.labels` or `sol.indices`

### Added
- **Explicit Validation**: Added `MISDAResult.validate(check_linear=True, check_nonlinear=True, check_pareto=True)`.
- **Object-Oriented API**: Introduced `MISCandidate` class for cleaner access to solution details.
- **Flattened Metrics**: Added properties `res.separation_score`, `res.pareto_precision`, etc., to avoid dictionary lookups.
- **Pandas Export**: Added `res.to_pandas()` to export all findings to a DataFrame.
- **Rich Interaction**: Added informative `__repr__` for `MISDAResult` and `MISCandidate`.

## [0.1.0] - 2025-01-03

### Initial Release
- **Core Algorithm**: Released `misda` Python package implementing the MISDA algorithm for structural dimensionality reduction using Maximal Independent Sets on dependency graphs.
- **Metrics**: 
  - **SES (Structural Evidence Score)**: Implementation of linear reconstruction fidelity metric.
  - **Pareto Consistency**: Precision/Recall metrics for evaluating surrogate quality in Multi-Objective Optimization.
  - **Spectral Entropy**: Passive diagnostic tool to detect high global complexity (e.g., Sphere topologies) in reduced spaces.
- **Benchmarks**:
  - `benchmark.ipynb`: Comprehensive suite testing Canonical structures (Independence, Redundancy, Chains) and Synthetic MOPs.
  - `dtlz.ipynb`: Specialized benchmark for Many-Objective DTLZ2 (Irreducible) and DTLZ5 (Degenerate) problems, including high-dimensional (M=10) scaling tests.
- **Visualization**: 
  - 3D reconstruction plots for interpreting surrogate fidelity.
  - Graph visualization of dependency structures.
