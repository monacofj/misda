# ADR 0001 — MISDA methodological model

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

MISDA estimates objective-space dimensional structure directly from observed objective values. The central idea is to transform statistically supported pairwise dependence into graph structure and then reason about mutually non-redundant objective subsets through graph independence.

The method must remain data-driven, reproducible, and explicit about the difference between structural redundancy, signed dependence, candidate reduction sets, and evidence attached to those candidates.

## Definitions

Let `Y in R^(N x M)` contain `N` observations of `M` objectives.

For nonconstant objectives `i` and `j`, MISDA computes the Pearson correlation

```text
r_ij = sum_k (y_ki-ybar_i)(y_kj-ybar_j)
       / sqrt(sum_k (y_ki-ybar_i)^2 sum_k (y_kj-ybar_j)^2).
```

Statistical evidence is represented by a one-tailed Fisher-z probability for the magnitude of the observed correlation:

```text
z_ij = atanh(|r_ij|) sqrt(N-3)
log p_ij = log P(Z >= z_ij),  Z ~ N(0,1).
```

MISDA derives two data-dependent probability endpoints:

- `alpha_onset`: the strongest observed positive structural event;
- `alpha_null`: a permutation-null endpoint derived from the expected maximum positive correlation under independently permuted objective columns.

For `a = aggressiveness in [0,1]`, the effective threshold is interpolated in probability space while evaluated stably in log space:

```text
alpha(a) = (1-a) alpha_onset + a alpha_null.
```

At the selected threshold MISDA constructs:

- `G+`: statistically supported positive dependence;
- `G±`: statistically supported positive or negative dependence.

The structural candidate universe is the set of all maximal independent sets of `G+`.

## Decision

The normative MISDA pipeline is:

```text
Y
 -> validate and normalize
 -> signed pairwise correlation statistics
 -> alpha_onset and sequential alpha_null
 -> aggressiveness interpolation
 -> G+ and G±
 -> graph-derived structural and latent dimensions
 -> complete maximal-independent-set discovery on G+
 -> structural candidate characterization
 -> dimensional-support diagnostics
 -> optional candidate evaluation
 -> optional ranking views
 -> external benchmark validation
```

The graph construction, dimensions, and structural candidate universe are discovery outputs. Reconstruction, Pareto evidence, and ranking are downstream and must not feed back into discovery.

## Rationale

A graph makes the reduction problem explicit: an edge encodes a supported pairwise relation and an independent set is a subset whose vertices contain no such pairwise redundancy relation. Maximal independent sets represent structurally irreducible alternatives in the sense that no additional objective can be added without violating independence.

The two-graph construction separates positive redundancy from signed dependence, allowing structural and latent dimensionality to answer different questions.

The threshold is not a fixed user-selected correlation cutoff. It is calibrated from the observed data and an empirical permutation null, with `aggressiveness` selecting a point between conservative onset and null-calibrated endpoints.

## Invariants

A conforming implementation must preserve all of the following:

- the analysis is derived only from `Y` and explicit method controls;
- the effective threshold is derived from `alpha_onset`, `alpha_null`, and `aggressiveness`;
- `G+` and `G±` are constructed from the same selected threshold but interpret correlation sign differently;
- graph-derived dimensions and the structural MIS universe are determined before optional candidate evaluation or user ranking;
- all structural MISs of `G+` are retained;
- benchmark truth never influences discovery.

## Current implementation

The current implementation is split across validation, statistics, graph construction, support, evaluation, ranking, and API modules. Correlation probabilities are kept in the log domain for numerical stability. Constant objectives remain explicit isolated graph vertices because pairwise Fisher-z evidence involving them is undefined.

Exact MIS enumeration is implemented by maximal-clique enumeration on the complement graph using a pivoted Bron-Kerbosch procedure. Exact graph independence numbers are also computed through maximal cliques in the complement graph.

## Permitted implementation variations

Implementations may change internal data structures, numerical kernels, exact graph algorithms, caching, vectorization, or parallelization as long as the invariants above and the more specific ADRs remain satisfied.

## Forbidden shortcuts / regression risks

Performance work must not silently replace data-derived thresholds with fixed cutoffs, graph independence with component counts, complete MIS discovery with bounded enumeration, or optional downstream evidence with inputs to structural discovery.

## Verification

Verification is distributed across the static test suite, synthetic benchmark battery, notebook executions, and the dedicated invariants described by ADRs 0002–0014.

## References

- Fisher, R. A. (1921). On the probable error of a coefficient of correlation deduced from a small sample. *Metron*, 1, 3–32.
- Bron, C.; Kerbosch, J. (1973). Algorithm 457: finding all cliques of an undirected graph. *Communications of the ACM*, 16(9), 575–577.
- Efron, B.; Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. (Permutation/resampling rationale.)