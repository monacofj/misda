<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

All notable changes to the **MISDA** (Maximal Independent Structural Dimensionality Analysis) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
