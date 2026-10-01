# Noisy ranking-policy safety audit

## Question

Issue #83 established that, under clean discovery, `dominance_preservation` can produce a better end-to-end reduction than `size_span` when the two policies select different MISs. In DTLZ5 the difference was very large; in DPF1 it was smaller but still favorable on average.

The unresolved question is reliability when the discovery observations are noisy:

> If ranking is performed on a distorted `Y`, which policy is less likely to produce a harmful reduction, and when should MISDA abstain from reducing at all?

This is issue #84 on branch `issue-84-noisy-ranking-safety`.

## Operational distinction

The audit separates two decisions that should not be conflated:

1. **Gate** — is there enough support to reduce at all?
2. **Ranking** — conditional on reducing, which MIS should be selected?

The primary operational safety criterion is Reduced-versus-Full degradation in the original objective space. A common calibrated GT remains useful as an absolute diagnostic reference, but a reduction is unsafe if it is materially worse than the corresponding Full treatment even when Full itself is imperfect.

## Important limitation of the first controls

DTLZ5 and DPF1 are useful ranking controls because `size_span` and `dominance_preservation` select different two-objective MISs under clean discovery. However, both clean selections are already annotated `UNSUPPORTED_REDUCTION` because of `TRANSITIVE_CHAINING`.

Therefore these two cases cannot by themselves validate a policy-independent rule such as `low support -> abstain`: that gate would already stop before ranking even in the clean condition. They remain valuable for the narrower question:

> If reduction is forced despite the warning, how robust is each ranking policy to noisy discovery?

A later gate-validation stage will need cases in which clean discovery is actually supported, so that observation noise can be tested for transitions from supported to unsupported and for false-positive reduction recommendations.

## Noise model

The first stage uses the repository's established scale-relative additive Gaussian observation model:

\[
Y_j = Z_j + \sigma\,\operatorname{sd}(Z_j)\,\varepsilon_j,
\qquad \varepsilon_j \sim N(0,1).
\]

Only the discovery matrix is perturbed. The underlying MOP remains clean, and any downstream optimizer output will be re-evaluated on the clean original objectives.

Initial levels reuse the existing robustness grid:

- `sigma = 0.00, 0.05, 0.10, 0.20, 0.40`;
- observation seeds `101, 202, 303, 404, 505`;
- discovery sample: the same 512-point scrambled Sobol sample used by #83;
- MISDA seed `123`.

Using the same standard-noise realization across sigma levels within a seed gives a nested perturbation path rather than unrelated noisy samples.

## Stage 1 — screening map

`benchmarks/run_noisy_ranking_safety.py` performs discovery and both rankings for every problem/noise/observation-seed combination without running the MOEA. It records:

- selected MIS for each policy;
- whether the selection changed relative to clean discovery;
- policy disagreement;
- observed dominance score;
- structural dimensions and separation status;
- candidate support/reduction assessment.

The purpose is to identify informative noisy conditions before paying for the end-to-end MOEA runs. In particular, the expensive stage should focus on conditions where noise changes a policy selection, makes the two policies disagree in a new way, or changes support/trust assessment.

## Planned Stage 2 — end-to-end noisy discovery

For selected informative conditions:

1. generate one fixed clean discovery sample `Z` and noisy observation `Y`;
2. select MISs from `Y` using both policies;
3. optimize Full and the two Reduced formulations on the clean MOP under matched initial populations, optimizer seeds, budgets, and NSGA-III reference directions;
4. re-evaluate every population in the full clean objective space;
5. compare Reduced against Full as the primary preservation question and against the common GT as an absolute diagnostic;
6. repeat across independent optimizer seeds only after the noisy screening condition is fixed.

## Hypotheses to test, not assume

- Low-noise representative `Y` may favor `dominance_preservation` because its ranking criterion is more aligned with optimization.
- `size_span` may be less exposed to noisy dominance geometry, but this has not yet been established end-to-end.
- At sufficiently poor data quality, the correct decision may be abstention rather than switching ranking policy.
- Bootstrap stability of a dominance preference is not sufficient evidence of correctness under systematic observation noise.

## Status

Issue #83 is closed as complete. Issue #84 and branch `issue-84-noisy-ranking-safety` are active. Stage-1 screening instrumentation and helper tests have been added; the next step is to run the screening grid and use the observed transitions to choose the smallest informative end-to-end experiment.
