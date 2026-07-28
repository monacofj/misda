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

MISDA accepts input as an $(N, M)$ dataset, provided as a NumPy array or pandas DataFrame, where $N$ represents samples and $M$ represents objectives (columns). The core philosophy of MISDA is based on the nature of objective correlations:

*   **Positive Correlation (Redundancy)**: Objectives that vary together convey similar information. Stronger positive correlation implies higher redundancy, allowing for dimensionality reduction by removing one of the pair.
*   **Negative Correlation (Conflict)**: Objectives that vary inversely represent the trade-offs (Pareto frontier) that define the problem structure. These must be retained to preserve the essential conflict information. 

MISDA Workflow
-----

MISDA operates on the theoretical premise that the *essential* dimensionality of an MOP corresponds to the size of the largest set of conflicting (or independent) objectives. The framework proceeds in rigorous formal steps:

1.  **Dependency Modeling**: A collumn-wise correlation matrix is computed from the sample data.
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

# 1. Analyse
res = misda.analyze(df)

# 2. Inspect (Standard summary)
print(res.summary())

# or Visualise
res.plot()

# 3. Export reduced dataset
reduced_df = df[res.best_mis.labels]
reduced_df.to_csv("reduced_mop.csv")
```

### 2. Deep Validation (Scientific)

```python
# Run analysis and force validation
res = misda.analyze(df)
res.validate() # Computes F_real, F_null, SES

# Get full audit report (includes summary details)
print(res.report())

# Check specific candidates
all_candidates = res.to_pandas()
print(all_candidates.head())
```

The `misda.analyze()` function accepts the following key arguments:

*   **`caution`** (float, 0.0 - 1.0): Controls the conservatism of the significance test.
    *   `0.0`: Aggressive reduction (uses $\alpha_{max}$).
    *   `1.0`: Conservative (uses $\alpha_{min}$), retaining more dimensions if in doubt.
*   **`name`** (str, optional): A label for the analysis case (e.g., "Experiment 1"), used in reports.
*   **`ensure_coverage`** (bool): If `True` (default), ensures the final set covers all variables locally.

### Adaptive Analysis (Strategy Pattern)

For high-dimensional ($M \ge 10$) or challenging problems ("Hyper-spheres"), you can switch to the adaptive strategy:

```python
# Adaptive search for optimal alpha
# 'method' switches strategy. 'target_fidelity' (0.95) sets the goal.
res = misda.analyze(df, method='adaptive', target_fidelity=0.95, name="HighDim_Case")

# Validate is already run internally during adaptive search, but you can run it again
print(res.summary())
```

Strategies:
1.  **`'static'`** (Default): Uses `caution` to pick a single `alpha`. Fast ($O(1)$ run).
2.  **`'adaptive'`**: Ignores `caution`. Performs a binary search on `alpha` to find the maximal reduction that maintains `target_fidelity`. Slower ($O(\log N)$ runs), but robust.

### Result Object (`MISDAResult`)

The `analyze` function returns a `MISDAResult` object providing rich access to the analysis state:

#### 1. Core Inputs & Decision Parameters
*   **`result.Y`**: Original input data.
*   **`result.name`**: Analysis name.
*   **`result.caution`**: The user-provided caution level (0.0 - 1.0).
*   **`result.alpha_min`**: Lower bound of the estimated alpha interval.
*   **`result.alpha_max`**: Upper bound of the estimated alpha interval.
*   **`result.alpha`**: The specific alpha value used for execution (alias for `alpha_exec`).
*   **`result.regime`**: The `AlphaRegime` object (diagnosis of the correlation structure).
*   **`result.metrics`**: Dictionary of raw diagnosis metrics (e.g., `r_max`, `r_null`).

#### 2. Execution Results (The Answers)
*   **`result.best_mis_labels`**: The variable names of the best solution.
*   **`result.best_mis_indices`**: The list of column indices of the best solution.
*   **`result.best_mis`**: The full `MISCandidate` object for the best solution.
*   **`result.mis_sets`**: List of all `MISCandidate` solutions found.
*   **`result.ranked_mis_sets`**: Dictionary mapping rank to list of candidates (access via `ranks_available`).
*   **`result.reduction_applied`**: Boolean (`True` if dimensionality was actually reduced).
*   **`result.get_mis_by_rank(k)`**: Method to retrieve all solutions for rank `k`.

#### 3. Quality & Validation
*   **`result.diagnosis`**: Short diagnostic string summarizing graph topology and surrogate fidelity:
    *   `"Valid (No Reduction Required)"`: When all objectives are retained ($S = M, T = \emptyset$).
    *   `"Ideal (Clique)"`: High fidelity ($F_{real} \ge 0.9$), high homogeneity ($h \ge 0.8$), 1 component, strictly verified clique ($\text{min\_compactness} \ge \alpha$).
    *   `"Ideal (Disjoint Cliques)"`: High fidelity ($F_{real} \ge 0.9$), high homogeneity ($h \ge 0.8$), $>1$ components, strictly verified cliques ($\text{min\_compactness} \ge \alpha$).
    *   `"Ideal (Multiple Components)"`: High fidelity ($F_{real} \ge 0.9$), high homogeneity ($h \ge 0.8$), $>1$ components, but not all components are complete cliques ($\text{min\_compactness} < \alpha$).
    *   `"Entangled (Mixed)"`, `"Good (Robust)"`, `"Drift (Chain)"`, `"Fragmented (Bridge)"`.
*   **`result.homogeneity_ratio`**: Score indicating cluster correlation consistency ($\text{min\_corr}/\text{max\_corr}$).
*   **`result.min_compactness`**: Lowest internal correlation strength across all components.
*   **`result.validation_metrics`**: Dictionary storing SES/Pareto results (populated by `validate()`).
*   **`result.validation_status`**: String indicating validation checks run (e.g., "Linear, Non-Linear, Pareto").
*   **`result.ses_results`**: Detailed dictionary of linear SES metrics (`F_real`, `F_null`, `ses`, `status`). Returns `status: "NO_REDUCTION"` and `ses: None` (`N/A`) if no reduction occurred.
*   **`result.ses_nonlinear`**: Scalar Non-Linear SES score (`float` or `None`).
*   **`result.ses_nonlinear_results`**: Detailed dictionary of non-linear Random Forest SES metrics (`F_real`, `F_null`, `ses`, `status`).

#### 4. Visuals & Reports
*   **`result.summary()`**: Returns formatted text summary.
*   **`result.report(top_k=5)`**: Returns comprehensive technical audit report, including per-component internal correlation range, homogeneity ratio, and compactness status (`Tight` vs `Loose`).
*   **`result.plot()`**: Returns network graph figure.
*   **`result.correlations`**: Returns textual correlation report.
*   **`result.to_pandas()`**: Returns DataFrame of all candidate solutions.

Other user functions
--------------------

For advanced users requiring granular control, the component functions are accessible:

*   **`estimate_alpha_interval(Y)`**:
    Calculates the lower and upper bounds for the significance level $\alpha$ based on dataset signal-to-noise ratio.

*   **`misda_significance(Y, alpha, ...)`**:
    Core engine. Runs graph construction and Bron-Kerbosch algorithm for a specific manual $\alpha$ value.

*   **`calculate_ses_linear(Y, mis, return_details=True)`**:
    Runs out-of-sample linear OLS reconstruction on eliminated targets $T$. Returns dict with `ses`, `F_real`, `F_null`. Returns `None` (`N/A`) when $T = \emptyset$.

*   **`calculate_ses_nonlinear(Y, mis, n_estimators=100, return_details=False)`**:
    Runs out-of-sample multi-output Random Forest non-linear reconstruction on eliminated targets $T$. Computes null baseline $F_{\text{null}}^{NL}$ via joint row permutation. By default returns scalar `ses` (or `None`). Set `return_details=True` for complete dictionary.

*   **`calculate_ses(Y, mis)`**:
    Alias for `calculate_ses_linear`.

*   **`diagnose_alpha_regime(alpha_min, alpha_max)`**:
    Returns the statistical regime (e.g., `SIGNAL_BELOW_NOISE` or `IMMEDIATE_SEPARATION`) describing how distinguishable the dependencies are from random noise.

References
----------

1. *Souza, C. H., Monaco, F. J., Delbem, A. C. B., Kuruvilla, J. A.* . *Maximal Independent Structural Dimensionality Analysis*, (in print), 2026.
2.  **Bron, C., & Kerbosch, J.** (1973). Algorithm 457: finding all cliques of an undirected graph. *Communications of the ACM*, 16(9), 575-577.
3.  **Deb, K., & Saxena, D. K.** (2005). On finding pareto-optimal solutions through dimensionality reduction for certain large-dimensional multi-objective optimization problems. *KanGAL Report*, 2005011.
