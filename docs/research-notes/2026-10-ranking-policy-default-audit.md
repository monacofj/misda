# Ranking-policy default audit

- Period: 2026-09-30 to 2026-10-01
- Status: **Rejected for default promotion; follow-up open**
- Tracking: issue #78, PR #79; follow-up issue #80
- Normative reference: ADR 0021

## Question

The optimization controls had made `dominance_preservation` look substantially better than the structural `size_span` order for choosing among admissible MISs. Before changing the public default, the project asked a stricter regression question:

> If the default selection policy were changed, would the benchmark behavior already validated under `size_span` be preserved?

The audit was intentionally observational. It did not change `discover()`, the public default, threshold inference, graph construction, or the MIS universe.

## Policy-impact audit

For each generated `Y`, discovery was performed once. Both policies operated on the same discovered candidates. Dominance evidence was evaluated for all candidates because the candidate policy requires it; benchmark truth was attached only after both policies had selected an MIS.

The audit covered `controlled`, `controlled_noisy`, `sampling_robustness`, `noisy_robustness`, and `comparison`.

### Clean results

Selection was completely preserved on clean data:

```text
controlled             13/13 same selected MIS
comparison               5/5 same selected MIS
sampling robustness    260/260 same selected MIS
```

This established that the new policy did not disturb the already validated clean controlled behavior.

### Observation noise

The fixed `sigma=0.10` reference diverged in 8 of 13 cases. The selected dimension was preserved in all 13; seven divergences improved declared Pareto truth and one slightly regressed it.

The broader noise sweep was the discriminating experiment:

```text
runs                                  200
same selected MIS                     123
policy divergences                     77
same selected dimension               200/200
trust-status changes                    0/200
declared Pareto-truth improvements     32
declared Pareto-truth regressions      40
neutral comparisons                   128
```

Every divergence occurred at nonzero observation noise. In every divergent run, `dominance_preservation` improved the observed new-dominance rate by construction, yet that observed-Y advantage did not reliably improve clean-truth Pareto agreement. Across the 77 divergences, the mean clean-truth Pareto-Jaccard change was slightly negative (about `-0.0057`). `nonlinear_blocks_4x5` was the strongest negative contributor.

The regressions were not structural-dimension failures: no tested divergence changed selected dimension, declared structural-unit adequacy, or reduction trust status.

## Hypothesis: is `aggressiveness=1` the real problem?

Because the first audit used the default `aggressiveness=1.0`, a follow-up tested whether the noisy representative-selection problem appeared only near the maximally aggressive endpoint.

The 77 noisy cases that diverged at `aggressiveness=1.0` were rescanned on

```text
0.00, 0.05, 0.10, ..., 1.00
```

For each point the experiment tracked structural dimension, MIS universe, selections under both policies, support/trust, observed dominance evidence, and benchmark Pareto truth.

### Result

All 77 cases exhibited exactly two tested profile regimes:

- at `aggressiveness=0.0`, every case moved to a much more conservative structural dimension of 19;
- from `aggressiveness=0.05` through `1.0`, each case retained the same structural dimension, the same `size_span` MIS, the same `dominance_preservation` MIS, the same trust status, and the same truth outcome.

Among the positive-aggressiveness regimes, 40 cases had selected dimension 4 and 37 had selected dimension 2. All 77 candidate reductions remained `SUPPORTED_REDUCTION` throughout the tested positive range.

For the 40 regressions seen at `aggressiveness=1.0`:

- 34 ceased to be regressions at `aggressiveness=0.0`, but only because the selected dimension jumped to 19;
- 0 avoided the regression at a lower aggressiveness while preserving the 2D/4D selected dimension;
- 6 remained regressions even at the conservative endpoint;
- all 40 were unchanged from `0.05` through `1.0`.

## Interpretation

The follow-up rejected the hypothesis that the observed ranking regressions were primarily caused by operating at `aggressiveness=1.0`.

A future `profile.report()` would reveal the structural discontinuity at the conservative endpoint, but the previously considered rule

```text
profile.selected = most aggressive supported point
```

would still select `1.0` in these cases because support remains positive throughout the stable 2D/4D regime. The profile therefore does not diagnose the representative-selection problem inside that regime.

This separates two axes:

- `profile()` studies sensitivity of graph structure to the threshold/aggressiveness path;
- ranking robustness studies whether one admissible MIS is preferred consistently enough over another under sample/noise variation.

## Decision

The evidence does **not** justify promoting `dominance_preservation` wholesale to the public default yet. `size_span` remains the default structural order and `dominance_preservation` remains experimental.

## Next hypothesis: resampling the ranking evidence

Issue #80 tests the next question directly. For a fixed observed `Y`, let
`S_dom` be the candidate selected by `dominance_preservation` and `S_struct` the
candidate selected by `size_span`. Holding those candidate subsets fixed, row
resamples of `Y` will be used to estimate

```text
DeltaD = D(S_dom) - D(S_struct)
```

where `D(S)` is the observed `new_dominance_rate` after projection to `S`.

The working hypothesis is:

- a genuinely strong observed-order advantage should keep `DeltaD < 0` across most resamples;
- a noise-driven representative preference should show weak support, ties, or frequent sign reversals around zero.

The primary experiment deliberately does **not** rediscover graph structure in
each bootstrap sample. That would mix uncertainty in discovery with uncertainty
in ranking evidence. The first test isolates only the latter by comparing the
same two candidate subsets on resampled rows.

Analytical controls such as DTLZ5 and DPF1 are expected to provide the strong
signal side of the experiment. Noisy controlled divergences from PR #79,
especially block and antagonistic cases, provide the uncertain/noise-sensitive
side. Benchmark truth remains external and is attached only after the Y-only
resampling statistic has been computed.

No public threshold, API change, or default-policy change is part of issue #80.
The purpose is to determine whether resampling uncertainty can distinguish the
robust dominance advantages previously seen in optimization controls from the
noisy representative choices that blocked default promotion in PR #79.

## Reproducibility policy

PR #79 retains the audit runners and a GitHub Actions workflow, but these experiments are scientific validation rather than ordinary acceptance tests. The workflow is manual-only (`workflow_dispatch`); `core`, `sampling`, `noisy`, and `aggressiveness` batteries are all opt-in and disabled by default, and a dispatch with no explicit selection fails immediately.

This note is the scientific reason PR #79 exists: it preserves the experiment that prevented a premature default-policy change and now points explicitly to issue #80 for the ranking-resampling follow-up.
