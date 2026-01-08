<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

All notable changes to the **MISDA** (Maximal Independent Structural Dimensionality Analysis) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-01-08

### Added
- **Comparative Benchmark**: Added `comparative.ipynb` and `comparative.tex`, a rigorous study comparing MISDA against PCA and Clustering.
  - Demonstrates **Pareto Dominance** of MISDA in Efficiency (Dimension) vs Effectiveness (Fidelity) space.
  - Highlights MISDA's ability to preserve structural conflicts (MOP-D) that PCA collapses.
  - Quantifies the "Cost of Interpretability" (~2% fidelity loss) for selecting physical variables over abstract components.
- **Adaptive Improvements**: Validated `method='adaptive'` robustness on non-linear benchmarks (MOP-A, MOP-D).
- **MOP Definitions**: Added canonical MOP definitions to `mop_definitions.py` for reproducible research.

## [0.3.1] - 2026-01-07

### Fixed
- **High-Correlation Underflow**: Fixed a numerical stability issue where extremely high correlations ($r \approx 0.99+$) caused p-values to underflow to `0.0`, resulting in an infinite significance threshold and failure to detect obvious redundancies. Implemented `stats.norm.sf` (Survival Function) and `stats.norm.isf` for stable tail probability calculations.
- **Defaults**: Restored `caution` default to **1.0** (Conservative). The improved numerical stability makes this safest setting usable even for highly redundant data.

## [0.3.0] - 2026-01-05

### Added
- **Unified API**: `misda.analyze` now supports a **Strategy Pattern** via the `method` argument.
  - `method='static'` (Default): Standard heuristic execution.
  - `method='adaptive'`: Robust binary search optimization.
- **High-Dimensional Robustness**: 
  - Adaptive strategy solves the **"Sphere Paradox"** (M >= 10), preventing unsafe over-reduction by enforcing a fidelity target (`target_fidelity`).
  - Added robust fallback to Full Retention (safest option) when no reduction meets the fidelity target.

### Fixed
- **Infinite Loop**: Fixed a potential infinite loop in `repair_mis_coverage` when `alpha` approaches 0.0 (clamped `r_crit`).
- **Validation Metrics**: Corrected casing sensitivity in validation metric keys (`"F_real"`).
- **Defaults & Logic**: 
  - Ensured `caution` defaults to **1.0** (Conservative).
  - Fixed semantic inversion: `caution=1.0` now correctly selects `alpha_min` (Safe/Conservative), whereas previously it selected `alpha_max`.

## [0.2.1] - 2026-01-05

### Changed
- **Defaults**: Updated `misda.analyze` default `caution` to **1.0** (Conservative).
- **Documentation**: Renamed `docs/manual.md` to `docs/usage.md` and overhauled content.
  - Added comprehensive `MISDAResult` object breakdown.
  - Removed redundancy in examples.
  - Refined definitions of positive (redundant) vs negative (conflicting) correlation.
- **Reporting**: Added `MISDAResult.report()` for deep technical audits.
- **Benchmarks**: Neutralized tone in `benchmark.ipynb` conclusions.

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
