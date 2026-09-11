# ADR 0010 — Decision-stable sequential alpha-null estimation

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

`alpha_null` is estimated from a permutation null and therefore carries Monte Carlo uncertainty. Pursuing arbitrary numerical precision is wasteful when remaining uncertainty can no longer change any public structural conclusion of discovery.

## Formal specification

For each null permutation, objective columns are permuted independently and the maximum positive pairwise correlation is recorded. Let these maxima be `m_1,...,m_B`, with

```text
r_null = mean(m_b)
SE_MC  = sd(m_b) / sqrt(B).
```

The current uncertainty interval in correlation space is

```text
[r_low, r_high] = [max(0,r_null-SE_MC), min(1,r_null+SE_MC)].
```

Both endpoints are transformed through the same Fisher-z probability mapping used by discovery to obtain threshold endpoints.

Define the discovery signature

```text
Sigma(alpha) = (
  structural_dimension,
  latent_dimension,
  complete structural_coverage scientific tie groups
).
```

Convergence occurs when

```text
Sigma(alpha_low) == Sigma(alpha_high).
```

Sampling begins with at least `B=N` null permutations and is bounded by

```text
B_max = 10N.
```

If the signatures have not stabilized by the cap, MISDA returns the current estimate with `converged=False`, reason `MAX_PERMUTATIONS_REACHED`, and emits one runtime warning per public call.

## Decision

Sequential null estimation stops on stability of public structural decisions, not on an arbitrary numerical tolerance for `alpha_null`.

## Rationale

Dimensions alone are too weak because the candidate universe or its order may still change. Literal graph equality is too strong because edge changes need not alter any public structural conclusion. Raw metric equality similarly demands precision without decision consequence. Scientific tie groups are included to stabilize downstream prefix selections without treating deterministic within-tie order as scientific information.

## Invariants

- null columns are permuted independently;
- the statistic is the maximum positive null correlation;
- stopping compares both structural and latent dimensions and the complete structural tie-group ordering;
- at least `N` permutations are observed before convergence can be declared;
- public estimation cannot run beyond `10N` permutations;
- nonconvergence at the cap is explicit, not silently treated as convergence.

## Current implementation

`misda._statistics.estimate_null_positive_correlation()` pre-standardizes columns once, uses `numpy.random.default_rng(seed)`, permutes each standardized column independently, and computes correlation matrices as matrix products. `estimate_null_from_maxima()` evaluates the sequential rule. The final RNG state and seed are retained in the estimate.

## Permitted implementation variations

Vectorization, parallel generation, batching, pre-standardization, caching, and equivalent numerically stable transformations are allowed if they preserve the null distribution, RNG/reproducibility contract, stopping rule, and public outputs.

## Forbidden shortcuts / regression risks

Do not replace the stopping rule by threshold-difference tolerance, dimensions-only stability, graph equality, a fixed small `B`, or a wall-clock timeout reported as converged.

## Verification

Controlled null-maxima sequences should test early convergence, cap nonconvergence, cancellation, and signature sensitivity. Public tests should verify the `10N` cap and propagated reason/status.

## References

- Fisher, R. A. (1921). On the probable error of a coefficient of correlation deduced from a small sample. *Metron*, 1, 3–32.
- Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, 3rd ed. Springer.