Maximal Independent Structural Dimensionality Analysis
=======
<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# MISDA: Maximal Independent Structural Dimensionality Analysis

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openscilabs/isda/blob/main/benchmark.ipynb)
[![REUSE status](https://api.reuse.software/badge/github.com/openscilabs/isda)](https://api.reuse.software/info/github.com/openscilabs/isda)

**MISDA** is a graph-theoretic framework for dimensionality reduction in **Multi-Objective Problems (MOPs)**.

Unlike projection-based methods (e.g., PCA) that create abstract components, MISDA identifies the **Maximal Independent Set (MIS)** of existing objectives. It builds a dependency graph from data correlations and mathematically extracts the largest subset of mutually independent features. This allows you to reduce the dimensionality of your problem while preserving the original meaning of your objectives and maintaining the topological structure of the solution space.

---

## Features

*   **Graph-Theoretic Reduction**: Uses the Bron-Kerbosch algorithm to find the Maximal Independent Set.
*   **Statistical Rigor**: Automatically estimates significance thresholds ($\alpha$) based on signal-to-noise ratio.
*   **Structure Preservation**: Ensures reduced sets "cover" the discarded objectives via strong correlation.
*   **Validation Metrics**: Includes **SES** ($R^2$ predictive fidelity) and **Homogeneity** (internal consistency) scores.
*   **Auto-Diagnosis**: Automatically flags potential topological issues like *Concept Drift* (Chains) or *Entanglement*.

## Installation

You can install `misda` directly from the source:

```bash
git clone https://github.com/openscilabs/isda.git
cd isda
pip install .
```

## Quick Start

The main entry point is `misda.analyze`, which handles the entire pipeline.

```python
import numpy as np
import misda

# 1. Load your data (N samples x M objectives)
Y = np.loadtxt("my_mop_data.csv", delimiter=",")

# 2. Run analysis
# caution=0.5 balances aggressive vs conservative reduction
# run_ses=True enables the Structural Evidence Score validation
result = misda.analyze(Y, caution=0.5, run_ses=True, name="Demo")

# 3. View results
print(result.summary())

# 4. Access the reduced set indices
print("Selected Objectives:", result.best_mis['mis_indices'])
```

## Documentation

For a detailed explanation of the theory, pipeline steps, and advanced usage, see the **[User Manual](docs/manual.md)**.

For performance benchmarks and validation against synthetic MOPs, see the **[Benchmark Suite](benchmark.ipynb)**.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contact the authors.

## References

> Souza, C. H., Monaco, F. J., Delbem, A. C. B., Kuruvilla, J. A.. *Maximal Independent Structural Dimensionality Analysis*, to be submitted, 2026.

## License

This project is licensed under the **GNU General Public License v3.0**. See [LICENSE](LICENSE) for details.

Copyright (C) 2025 Monaco F. J. <monaco@usp.br>
