# ADR 0015 — Empirical null envelope for alpha-null estimation

- Status: Accepted
- Recorded: 2026-09-15
- Supersedes: ADR 0010
- Tracking issue: #44

## Context

MISDA constructs its dependence graphs by comparing observed correlations with a permutation-derived null reference. ADR 0010 defined that reference from the **mean of the maximum positive correlations** observed across null permutations, together with a sequential `mean ± MC-SE` stopping rule.

Repeated clean runs of the fully independent benchmark exposed a weakness in that design. An independent observed sample is itself another draw from the same no-dependence regime represented by the permutations. Its largest accidental correlation can therefore exceed the *mean* null maximum with substantial frequency. A single such exceedance creates a spurious graph edge and can reduce the graph independence number even though no dependence exists.

The methodological question is therefore not the typical size of the largest null correlation, but whether an observed correlation exceeds the upper empirical range produced by the null experiment.

## Decision

For a data matrix with `N` observations, public `alpha_null` estimation uses exactly `B=N` independent-column permutations. If `m_b` is the maximum **positive** pairwise correlation produced by null permutation `b`, define

```text
r_null = max(m_1, ..., m_N).
```

`alpha_null` is then obtained by applying the same one-tailed Fisher-z probability mapping used by discovery:

```text
log_alpha_null = positive_correlation_log_p(r_null, N).
```

The same threshold remains shared by `G+` and `G±`; negative observed correlations may enter `G±`, but a second absolute-correlation null calibration is not introduced.

The sequential `mean ± MC-SE` convergence rule and its `10N` cap are no longer part of the public estimator.

## Rationale

The old estimator asked whether an observed correlation exceeded the **typical null maximum**. For graph construction this was too permissive: values that are uncommon but still routinely attainable under independence could become edges.

The empirical envelope asks the stricter question: did the observed correlation exceed every maximum produced by the fixed null experiment? This directly targets the false-edge failure mode seen in the independence case while remaining entirely data-derived and hyperparameter-free.

The fixed budget `B=N` is itself derived from the dataset size rather than supplied by the user. Experimental validation found no dimensional changes when the envelope budget was increased from `N` to `2N`, `5N`, or `10N` in the tested battery.

## Experimental basis

Before adoption, the candidate rule was evaluated on the controlled benchmark suite:

- 50 clean replicates of `independence`: exact latent and structural dimension recovery in 100% of runs;
- 11 regular clean cases × 20 replicates: exact latent and structural dimension recovery in 100% of runs;
- regular robustness cases across `sigma ∈ {0, 0.05, 0.10, 0.20, 0.40}`: exact latent and structural recovery in all tested replicates;
- the known adversarial `transitive_chain` and `regime_switching` cases remained adversarial rather than being hidden by the new threshold;
- shared positive-null calibration and separate positive/absolute null calibration produced the same dimensional conclusions in the tested battery.

These experiments motivate the decision but do not redefine benchmark truth.

## Metadata compatibility

`NullAlphaEstimate` and the public analysis object retain legacy fields to avoid an unnecessary API break.

Under the envelope estimator:

- `n_permutations` is exactly `N` after normal completion;
- `converged=True` means that the fixed-budget envelope was completed, not that a Monte Carlo mean converged;
- `reason=None` after normal completion;
- `se_mc=NaN` because a standard error of a Monte Carlo mean is not defined for this estimator;
- `r_interval=(r_null, r_null)` and `log_alpha_interval=(log_alpha_null, log_alpha_null)` are degenerate compatibility fields, not uncertainty intervals;
- `lower_r_signature` and `upper_r_signature` are equal to the signature at the selected envelope threshold;
- explicit cancellation remains represented by `converged=False` and `reason="CANCELLED"`.

A future API revision may rename or remove these compatibility fields, but they must not be given the old ADR 0010 interpretation.

## Invariants

- objective columns are permuted independently;
- every null replicate contributes the maximum positive pairwise correlation;
- normal public estimation uses exactly `N` null replicates;
- `r_null` is the maximum of those null maxima;
- the Fisher-z mapping remains the probability transformation used by discovery;
- the null threshold is shared by `G+` and `G±`;
- the estimator remains reproducible from the public seed/RNG contract;
- no user-tunable percentile, confidence level, permutation count, or numerical tolerance is introduced.

## Forbidden shortcuts / regression risks

Do not replace the envelope by a fixed percentile, a hard-coded correlation threshold, a user-selected `B`, the old mean-of-maxima statistic, or a separate signed threshold without a new methodological decision.

Do not interpret the retained `se_mc` or interval fields as uncertainty estimates for the envelope.

## Verification

Tests must verify that public estimation consumes exactly `N` permutations, that `r_null == max(samples)`, that repeated runs with the same seed are identical, that constant inputs remain well-defined, and that the obsolete `10N` non-convergence warning is absent.

The clean diagnostic battery and regular benchmark cases must not regress when this decision is implemented.
