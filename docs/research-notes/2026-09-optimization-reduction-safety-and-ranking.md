# Optimization reduction safety and ranking signals

- Period: September 2026
- Status: **Partly adopted; optimization validation remains open**
- Tracking: optimization benchmark work in PR #76; ranking-robustness and reduction-decision probes
- Normative references: ADR 0020 and ADR 0021

## Question

The investigation asked whether an objective subset selected from observed `Y` remains a useful proxy for optimizing the original objective set, and which ranking evidence best distinguishes candidate MISs.

Benchmark truth and optimizer outcomes were kept external to runtime MISDA.

## Experimental separation

The work converged on three distinct layers:

```text
observed Y
  -> discovery / diagnostics / ranking
  -> selected candidate and trust annotation

known benchmark mathematics
  -> external interpretation of the reduction

MOEA experiment
  -> consequence of optimizing Full versus Reduced
     measured again in the original objective space
```

A runtime `SUPPORTED` label is therefore not a proof of global Pareto equivalence, and `UNSUPPORTED` does not by itself establish that a selected reduction fails globally.

## Initial optimization pilot

The first paired NSGA-III experiments showed that reconstruction quality alone was not enough. DTLZ2 (`M=10`) provided a negative analytical control: every objective is indispensable on the true Pareto front, and the pilot reduced-run quality degraded in the original objective space. DPF1 (`M=10`, base dimension `D=2`) showed the complementary problem: a compact candidate could reconstruct the remaining objectives well while still receiving an internal support warning, and early optimizer comparisons were not interpretable until the Full baseline itself was calibrated.

The resulting rule was to calibrate Full convergence before attributing Full-versus-Reduced differences to the reduction.

## Pareto-retention ranking probe

A first ranking experiment compared canonical `size_span` with empirical Pareto-front retention on independent Sobol screening samples. Across eight seeds and `N in {256, 512, 1024}`:

- DPF1's analytical base pair `f1,f2` was selected by `pareto_retention` in all 24 runs, while `size_span` consistently preferred the earlier `f2,f8` candidate;
- DTLZ5's analytical pair `f9,f10` was always discovered and the explicit counterexample `f1,f10` was consistently rejected by `pareto_retention`, but `f9,f10` was not always the unique top-retention candidate;
- DTLZ2 never produced exact empirical-front preservation for a proper subset.

Increasing the generic Sobol sample did not remove the DTLZ5 ambiguity. Empirical Pareto retention was therefore useful but incomplete. ADR 0020 records the resulting experimental policy.

## Global dominance-distortion probe

The next hypothesis was that the full observed cloud contains useful order information that front-only metrics discard. For a candidate `S`, the probe measured the fraction of unordered row pairs that are incomparable in full `Y` but become strictly comparable after projection to `Y_S`. Lower is better.

The global signal was markedly more stable:

- DTLZ5 `f9,f10` was the unique minimum-global-distortion candidate in all 24 runs;
- the explicit DTLZ5 counterexample was never best;
- DPF1 `f1,f2` had zero global distortion in every run and was best in all 24 runs;
- the previous DPF1 `f2,f8` candidate was never best;
- DTLZ2 again provided the negative control: no proper reduction had zero distortion.

The front-only version was much less stable for DTLZ5. This signal became the basis of the experimental `dominance_preservation` policy in ADR 0021.

## Variable-cardinality safe control

A custom analytical control (`SAFE_PATH`) tested whether observed-order criteria implicitly favor larger MISs even when a smaller MIS is equally valid for the continuous optimization problem.

For

```text
f1 = x + y
f2 = y
f3 = 1 - x + y
```

with `(x,y) in [0,1]^2`, the positive-dependence graph is the path `f1--f2--f3`, with maximal independent sets `{f2}` and `{f1,f3}`. Both preserve the true Pareto set `y=0`.

Across 24 Sobol runs, both Pareto retention and global dominance distortion preferred the larger MIS `{f1,f3}` every time.

This established that observed-cloud preservation is a conservative signal, not an estimator of the smallest globally equivalent objective subset.

## Conservative reduction-decision probe

The end-to-end Y-only pipeline was then exercised on exact controls and analytical optimization cases:

```text
Y
 -> discover complete MIS universe
 -> evaluate candidate dominance/Pareto evidence
 -> rank by dominance preservation
 -> always return the selected MIS
 -> annotate NO_REDUNDANCY / SUPPORTED_REDUCTION / UNSUPPORTED_REDUCTION
```

Observed outcomes included:

- independence: all objectives retained, `NO_REDUNDANCY`;
- exact redundancy and block controls: expected supported reductions;
- DTLZ2: a proper reduction returned with `UNSUPPORTED_REDUCTION`, agreeing with the negative analytical control;
- DTLZ5: `f9,f10` selected but marked unsupported because of `TRANSITIVE_CHAINING`;
- DPF1: `f1,f2` selected but likewise marked unsupported;
- SAFE_PATH: `{f1,f3}` selected and supported.

The tested controls contained no case in which a known failing reduction was marked trustworthy. DTLZ5 and DPF1 instead exposed conservative false negatives.

## Overall conclusion

The investigation established that reconstruction, empirical-front retention, observed dominance preservation, internal support, analytical equivalence, and optimizer performance are different estimands. `dominance_preservation` was the strongest tested observed-Y ranking signal for the DTLZ5/DPF1 controls, but it remained conservative and did not constitute a proof of global optimization equivalence. `TRANSITIVE_CHAINING` could reduce recall, and unsupported candidates should still be returned so benchmarks can measure their actual consequence.

The later default-policy audit in `2026-10-ranking-policy-default-audit.md` tested whether this strong clean-control evidence justified replacing `size_span` as the public default.
