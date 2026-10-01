# Sampling and observation-noise validation

- Period: September 2026
- Status: **Adopted as validation evidence**
- Tracking: GitHub Actions run `35020551271`; controlled, controlled-noisy, sampling-robustness, and noisy-robustness batteries
- Normative references: ADR 0013 and ADR 0014

## Question

How stable are MISDA's graph-derived dimensions, support annotations, and empirical Pareto diagnostics under ordinary finite-sample variation and explicit observation noise?

The experiment was designed to keep three sources of evidence separate:

1. clean resampling, which changes the sampled observations but adds no observation noise;
2. a fixed noisy reference condition, used as a reproducible diagnostic snapshot;
3. a multi-sigma observation-noise sweep, used to study degradation rather than define a pass/fail threshold.

## Protocol

The fixed-noise controlled reference used `N=300`, sample seed `123`, observation seed `456`, and scale-relative `sigma=0.10` across the 13 controlled diagnostics.

The clean sampling study used `N=300`, fixed MISDA seed `123`, `sigma=0`, and 20 independent sample seeds (`1001` through `1020`) for all 13 diagnostics.

The observation-noise study used `N=300`, fixed MISDA seed `123`, five replicate seeds, and `sigma in {0, 0.05, 0.10, 0.20, 0.40}`. Within a problem/replicate, clean `Z` and the standardized perturbation realization were reused across sigma so only noise intensity changed.

No sigma was interpreted as a scientific acceptance threshold.

## Dimensional recovery

At the fixed `sigma=0.10` reference, Cases 1--11 recovered the declared latent and structural dimensions exactly. The two previously known failure modes remained visible rather than being silently corrected:

- `transitive_chain` remained `UNSUPPORTED` with `TRANSITIVE_CHAINING`;
- `regime_switching` remained `UNSUPPORTED` with `HIDDEN_SPECTRAL_STRUCTURE`.

Across 20 independent clean samples, Cases 1--11 recovered both dimensions in every replicate and were `SUPPORTED` in every replicate. `transitive_chain` and `regime_switching` missed their declared dimensions consistently while their corresponding support diagnostics fired consistently.

In the focused observation-noise sweep, the regular controlled diagnostics retained exact latent and structural recovery through the tested `sigma=0.40`. At the highest noise level the support layer became more cautious in some nonlinear cases without changing the recovered graph dimensions; this was treated as diagnostic evidence, not as a correction rule.

## Pareto-membership interpretation

The validation separated:

```text
observation agreement   P_Z versus P_Y
reduction agreement     P_Y versus P_R
end-to-end agreement    P_Z versus P_R
```

where `P_Z` is the sampled clean-truth front, `P_Y` the observed full-space front, and `P_R` the reduced-space front.

This distinction explained apparently poor exact-membership scores without implying a dimensional-estimation bug.

### Nonlinear monotonic redundancy

Independent observation errors can break the common ordering of many monotonic transforms and inflate the observed full-space front before reduction. At `sigma=0.10`, averaged over five replicates, dimensional recovery remained exact while observation and reduction Jaccard were both low. The low exact-membership agreement was therefore primarily observation-driven.

### Antagonistic groups

For the antagonistic nonlinear control, the clean and observed full-space fronts contained all sampled rows across the tested noise grid, while the two-objective reduced projection allowed some noisy observations to dominate others. Here the membership loss was reduction-driven rather than observation-driven. The same qualitative effect appeared in the antagonistic linear control.

## Conclusion

The experiments did not indicate a Pareto implementation defect. They established a boundary that remains important for later work:

- dimensional recovery/support and Pareto-membership stability answer different questions;
- exact Pareto Jaccard can degrade even when graph-derived dimensions remain correct;
- observation-driven front inflation and reduction-driven dominance changes must be distinguished;
- no fixed Pareto-stability or noise threshold was justified by these runs.

This validation later motivated the separate investigation of whether observation-noise susceptibility could be inferred from a single `Y`; that question and its negative result are recorded in `2026-09-pareto-noise-susceptibility.md`.
