# Ranking policy, noise scale, and revised default hypothesis

- Period: 2026-10-01
- Tracking: issues #83 and #84
- Branch: `issue-84-noisy-ranking-safety`
- Related notes: `2026-10-ranking-policy-default-audit.md`, `2026-10-noisy-ranking-safety.md`, `2026-10-noisy-ranking-supported-stress.md`
- Status: **design hypothesis recorded; no public default changed in this branch**

## Why revisit the default question

The earlier default-policy audit correctly rejected a premature promotion of `dominance_preservation`: noisy finite-sample Pareto diagnostics could favor a representative that disagreed with clean-sample truth, and bootstrap stability did not solve that identifiability problem.

Subsequent end-to-end optimization evidence changed the interpretation of that result.

Issue #83 showed that, when discovery data are clean/representative, `dominance_preservation` can choose a substantially better optimization-relevant MIS than `size_span`. Issue #84 then showed that moderate additive observation noise did not immediately destabilize the DTLZ5/DPF1 choices; operational failures appeared only in deliberately high-noise stress regimes.

The central question is therefore no longer simply whether dominance-based ranking is more sensitive to noise. It is whether that sensitivity becomes practically relevant inside a data-quality regime in which objective-reduction inference is itself still reasonable.

## Noise scale used in the experiments

The controlled observation model is

\[
Y_j = Z_j + \sigma\,s_j\,\varepsilon_j,
\qquad \varepsilon_j \sim N(0,1),
\]

where `s_j` is the clean sample standard deviation of objective `j`.

Under the idealized assumptions of independent additive error and using the clean variance as signal variance,

\[
\mathrm{NSR}=\frac{\mathrm{Var}(\text{noise})}{\mathrm{Var}(Z)}=\sigma^2,
\]

and the fraction of observed variance attributable to the clean signal is

\[
R=\frac{\mathrm{Var}(Z)}{\mathrm{Var}(Y)}=\frac{1}{1+\sigma^2}.
\]

`R` is the classical reliability ratio for this additive-error model. This interpretation is specific to the experiment's noise model; MISDA does not estimate `R` from an arbitrary observed `Y`.

| sigma | noise/signal variance (NSR) | signal fraction / reliability R |
|---:|---:|---:|
| 0.05 | 0.25% | 99.75% |
| 0.10 | 1% | 99.01% |
| 0.20 | 4% | 96.15% |
| 0.40 | 16% | 86.21% |
| 0.80 | 64% | 60.98% |
| 1.00 | 100% | 50.00% |
| 2.00 | 400% | 20.00% |

Thus `sigma=0.40` is already a substantial perturbation: the noise variance is 16% of the clean-signal variance and about 14% of the total observed variance is noise. It should not be described as nearly clean.

## Empirical boundary observed in DTLZ5/DPF1

For the DTLZ5 and DPF1 clean controls used in issue #83, neither ranking policy changed its selected MIS for any of five observation seeds throughout the ordinary grid up to `sigma=0.40`.

The high-noise stress probe showed the first DTLZ5 selection instability only beyond that ordinary range. At approximately `sigma=0.8`, selection changes began to appear; at `sigma=1.0` a supported DTLZ5 condition produced a clear end-to-end ranking failure for `dominance_preservation`, while `size_span` remained operationally strong. DPF1 remained less discriminating and did not establish a universal noisy-data advantage for `size_span`.

The important scale observation is that the demonstrated DTLZ5 failure occurred around

\[
\mathrm{NSR}=1,
\qquad R=0.5,
\]

that is, a regime in which noise variance is as large as signal variance. The more extreme support-boundary counterexample at `sigma=2` corresponds to noise variance four times the signal variance.

These stress results remain scientifically useful because they expose failure modes, but they should not be interpreted as evidence that `dominance_preservation` is fragile under small or moderate measurement noise.

## Revised interpretation of the two policies

The accumulated evidence supports the following functional distinction.

- `dominance_preservation` is **optimization-oriented**: when observed `Y` is representative of the underlying objective geometry, it has direct evidence of selecting better downstream reductions than `size_span`.
- `size_span` is **structural/conservative**: it can be substantially safer when the observed dominance geometry is strongly corrupted, but it is not universally superior under noise.

The stress experiments therefore move the default-policy question away from the earlier formulation

```text
noise exists -> keep size_span by default
```

and toward

```text
objective-reduction inference assumes Y is sufficiently informative;
within that intended regime, dominance_preservation is the better-motivated default;
size_span remains an explicit conservative structural alternative.
```

This is a **design hypothesis**, not yet a normative API decision. A public default change should be recorded separately in an ADR/implementation issue if adopted.

## Responsibility boundary

MISDA should continue to detect and report internal evidence that a reduction is unsupported. However, it cannot be required to recover the clean objective geometry from an arbitrarily corrupted single `Y` without additional information.

In particular, a sufficiently high observation-noise regime is fundamentally a data-quality problem. Because support, graph structure, and ranking are all inferred from `Y`, severe corruption can mislead all three. No ranking policy derived only from that same matrix can provide an unconditional safety guarantee.

This does not mean "bad data are the user's fault". It means the methodological contract must state the assumption clearly: objective-reduction recommendations are conditional on `Y` being sufficiently informative about the system being optimized.

## Consequence for `profile()`

This conclusion should not turn `profile()` into a supposed noise detector.

The intended role of `profile()` is structural: characterize how the discovered objective geometry behaves across the threshold/aggressiveness path, including dimensions, graph regimes, separation, candidate support, and structural discontinuities.

A profile can say, in effect,

> "Given this observed `Y`, the inferred reduction is structurally well or poorly supported."

It cannot, from one arbitrary matrix alone, generally say

> "This `Y` has reliability 0.86" or "the observed dominance geometry is the clean geometry."

The experiments in issue #84 explicitly demonstrate this distinction: a stable/supported structure can still be misleading when the observations themselves are severely distorted.

## Current design hypothesis

A concise current position is:

```text
poor structural support             -> abstain
adequate structural support         -> ranking is meaningful
adequate support + informative Y    -> dominance_preservation is the leading default candidate
known poor/uncertain data quality    -> size_span may be preferred, or abstain
```

The last branch requires external or additional evidence about data quality; `profile()` alone cannot generally supply it.

No default is changed by this note.