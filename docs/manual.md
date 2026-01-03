<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

MISDA User Guide
================

Introduction
------------

**Maximal Independent Structural Dimensionality Analysis (MISDA)** constitutes a graph-theoretic framework designed for dimensionality reduction in **Multi-Objective Problems (MOPs)**. Its primary objective is to rationalize the optimization search space by eliminating redundant objectives while strictly preserving the underlying conflict structure of the Pareto frontier.

Fundamentally, MISDA operates by identifying the **Maximal Independent Set (MIS)** within an empirically derived objective dependency network. In contrast to projection-based techniques such as Principal Component Analysis (PCA)—which map attributes onto an abstract latent space—MISDA analyzes the topological properties of the correlation graph to isolate the largest subset of *original* decision variables that exhibit mutual statistical independence. Through the maximization of this independent set, the algorithm effectively recovers the intrinsic dimensionality of the problem, ensuring the retention of non-redundant information without semantic loss.

MISDA Workflow
-----

MISDA operates on the theoretical premise that the *essential* dimensionality of an MOP corresponds to the size of the largest set of conflicting (or independent) objectives. The framework proceeds in rigorous formal steps:

1.  **Dependency Modeling**: A correlation matrix is computed from the sample data.
2.  **Significance Testing**: Pairwise correlations are subjected to the Fisher Z-transformation to determine statistical significance at a given $\alpha$ level (e.g., $p < 0.05$).
3.  **Graph Construction**: An undirected graph $G=(V, E)$ is built where vertices $V$ represent objectives and edges $E$ represent significant dependencies (redundancies).
4.  **MIS Extraction**: The algorithm solves the **Maximum Independent Set** problem (NP-hard) on $G$ using an optimized **Bron-Kerbosch** algorithm. An independent set represents a group of objectives with no pairwise redundancies. The *Maximal* set (MIS) is the largest such group.
5.  **Coverage Repair**: The initial MIS is iteratively refined to ensure that every discarded objective is "covered" (strongly correlated) by at least one objective retained in the MIS, guaranteeing representativeness.
6.  **Metric Selection**: If multiple MIS solutions of the same size exist, MISDA ranks them using secondary topological metrics:
    *   **Neighborhood**: Maximizing the number of covered external variables.
    *   **Span**: Maximizing the total correlation strength with external variables.

**Validation & Diagnosis**
To assess the quality of the reduction, MISDA computes:

*   **SES (Structural Evidence Score)**: Computes the predictive power ($R^2$) of the reduced set (MIS) against the full set, using a linear model compared to a permutation null model.
    *   $SES \approx 1.0$: Perfect reconstruction (Lossless).
    *   $SES \approx 0.0$: No better than noise (High Information Loss).

*   **Homogeneity Ratio**: Measures the internal consistency of connected components by comparing the weakest correlation to the strongest correlation within the component ($min/max$).
    *   $Ratio < 0.6$ warns of "transitive chains" (A-B-C) where A and C are independent but linked by B.

*   **Auto-Diagnosis**: Automatically categorizes the topology based on the intersection of Fidelity ($F$) and Homogeneity ($H$):
    *   **Ideal (Clique)** ($F>0.9, H>0.8$): Perfect dense groups. Safe reduction.
    *   **Good (Robust)** ($F>0.9$): Reliable reduction.
    *   **Entangled** ($F>0.9, H<0.2$): High fidelity but messy topology (mixed positive/negative correlations).
    *   **Drift (Chain)** ($F<0.8, H \ge 0.6$): Warning state. Potential loss of transitivity.
    *   **Fragmented (Bridge)** ($F<0.6, H<0.6$): Failure state. Graph is held together by weak links (bridges).

Usage
-----

The library is designed for ease of use with a single high-level entry point.

### Main Function

The primary function for end-users is `analyze`. It handles the entire pipeline: automatic alpha estimation, regime diagnosis, graph execution, and validation.

```python
import misda

# Y is a (N, M) numpy array or pandas DataFrame of objective values
result = misda.analyze(Y, caution=0.5, run_ses=True, name="MyExperiment")

print(result.summary())
```

*   **`caution`** (float, 0.0 - 1.0): Controls the conservatism of the significance test.
    *   `0.0`: Aggressive reduction (uses $\alpha_{max}$).
    *   `1.0`: Conservative (uses $\alpha_{min}$), retaining more dimensions if in doubt.
*   **`run_ses`** (bool): If `True`, performs the computationally expensive but valuable Structural Evidence Score validation.

### Result Object

The `analyze` function returns a `MISDAResult` object with useful properties:

*   **`result.summary()`**: Returns a formatted string report (Diagnosis, Action, Quality, Validation).
*   **`result.plot()`**: Returns a matplotlib figure visualizing the dependency graph and the selected MIS independent set (in green).
*   **`result.best_mis`**: Dictionary containing the indices and labels of the best selected subset.
*   **`result.diagnosis`**: Auto-diagnosis string (e.g., "Ideal (Clique)", "Drift (Chain)").

Other user functions
--------------------

For advanced users requiring granular control, the component functions are accessible:

*   **`estimate_alpha_interval(Y)`**:
    Calculates the lower and upper bounds for the significance level $\alpha$ based on the signal-to-noise ratio of the dataset.

*   **`misda_significance(Y, alpha, ...)`**:
    The core engine. Runs the graph construction and Bron-Kerbosch algorithm for a *specific* manual $\alpha$ value. Returns the raw dictionary of graph artifacts (adjacency, components, all MIS sets).

*   **`calculate_ses(Y, mis_indices)`**:
    Runs the validation procedure independently. Useful if you want to test a specific subset of objectives `mis_indices` against the dataset `Y` without running the full graph discovery.

*   **`diagnose_alpha_regime(alpha_min, alpha_max)`**:
    Returns the statistical regime (e.g., `SIGNAL_BELOW_NOISE` or `IMMEDIATE_SEPARATION`) describing how distinguishable the dependencies are from random noise.

References
----------

1. *Souza, C. H., Monaco, F. J., Delbem, A. C. B., Kuruvilla, J. A.* . *Maximal Independent Structural Dimensionality Analysis*, to be submitted, 2026.
2.  **Bron, C., & Kerbosch, J.** (1973). Algorithm 457: finding all cliques of an undirected graph. *Communications of the ACM*, 16(9), 575-577.
3.  **Deb, K., & Saxena, D. K.** (2005). On finding pareto-optimal solutions through dimensionality reduction for certain large-dimensional multi-objective optimization problems. *KanGAL Report*, 2005011.
