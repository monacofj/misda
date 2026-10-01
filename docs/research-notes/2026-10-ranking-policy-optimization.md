# Ranking-policy optimization audit

## Question

The ranking-policy experiments in Revisão 09 compared `size_span` and `dominance_preservation` using properties measured on the observed objective sample `Y`. In several cases, notably DTLZ5 and DPF1, `dominance_preservation` selected a different MIS and showed a strong, bootstrap-stable advantage in observed dominance preservation. That result does not by itself establish that the selected reduction is better when used by a multiobjective optimizer.

The end-to-end question is therefore:

> When `size_span` and `dominance_preservation` select different MISs from the same discovery sample, which reduction better preserves the outcome of optimization relative to the full objective problem?

## Experiment

Use the existing optimization protocol and compare three conditions under matched budgets and seeds:

1. **Full** — optimize the complete objective set.
2. **Reduced / size_span** — optimize only the objectives selected by the default structural ranking policy.
3. **Reduced / dominance_preservation** — optimize only the objectives selected by the experimental dominance-based policy.

All resulting solutions are reevaluated in the original full objective space and compared against the same MoeaBench-calibrated ground truth using GD+, IGD+ and relative HV.

The first targets are cases where the policies are known to disagree, especially DTLZ5 and DPF1. A comparison is only interpretable after inspecting convergence of the corresponding Full run; otherwise a difference between reduced variants may reflect a weak full baseline rather than the effect of ranking policy.

## Implementation

Issue #83 is implemented on branch `issue-83-ranking-policy-optimization` in `benchmarks/run_ranking_policy_optimization.py`.

The experiment deliberately discovers the MIS universe only once for each MOP and then constructs both ranking views from that same `MISSet`. Both selected candidates receive the same screening diagnostics. For each MOEA seed, Full and both Reduced treatments receive the same explicit initial decision population, optimization budget, optimizer seed and independently fixed NSGA-III reference-direction seed. Reduced solutions are never judged in their projected objective spaces: each checkpoint is re-evaluated in the original M-objective space before computing quality metrics.

The default scientific run uses:

- `M=10`;
- a 512-point scrambled Sobol discovery sample;
- population 60;
- checkpoints at 50, 100, 200 and 400 generations;
- one initial optimizer seed for the first audit, with support for repeated `--moea-seed` arguments in follow-up runs;
- a fixed Monte-Carlo seed for relative-HV measurement so metric noise is shared across treatments rather than coupled to the optimizer seed.

No automatic numerical threshold is used to declare the Full run converged. Its checkpoint trajectory must be inspected before policy differences are interpreted.

A dedicated workflow, `.github/workflows/ranking-policy-optimization.yml`, runs helper-contract smoke tests on the branch and can launch the scientific audit. The initial smoke run passed before the scientific execution was triggered.

## First scientific run — 2026-10-01

The first audit used MOEA seed `321`, population `60`, and 400 generations. It is a single-seed diagnostic run, not a statistical comparison.

### DTLZ5

The two policies selected different two-objective reductions from the same discovery result:

- `size_span`: `f1, f10`, observed new-dominance rate `0.271413`;
- `dominance_preservation`: `f9, f10`, observed new-dominance rate `0.200307`.

Both were internally annotated `UNSUPPORTED_REDUCTION` because of `TRANSITIVE_CHAINING`. The dominance-based selection is the independently known optimization-safe DTLZ5 pair used in the analytical control.

At generation 400, after reevaluation in the original 10-objective space:

| treatment | GD+ | IGD+ | relative HV |
|---|---:|---:|---:|
| Full | 2.023461 | 0.880806 | 0.000000 |
| Reduced / size_span | 1.356477 | 0.743161 | 0.000000 |
| Reduced / dominance_preservation | 0.000319 | 0.005501 | 0.993282 |

The dominance-based reduction is dramatically closer to the calibrated GT than the size-span reduction in this run. The direction already appears at generations 50, 100 and 200.

However, the Full trajectory is not a converged reference: its IGD+ moved `0.6689 -> 0.5561 -> 0.3599 -> 0.8808` at generations 50, 100, 200 and 400, and its relative HV remained essentially zero. Therefore the result cannot be interpreted as "Reduced is better than a converged Full." It instead shows that, for this many-objective NSGA-III configuration, the analytically safe two-objective reduction is much easier to optimize and reaches the known GT while the Full treatment does not.

This is strong evidence that the ranking choice has a real end-to-end consequence in DTLZ5, and in this case the observed-Y dominance signal points to the analytically safe and optimizer-effective reduction. It is not yet evidence for a general default-policy change.

### DPF1

The two policies again selected different two-objective reductions:

- `size_span`: `f2, f8`, observed new-dominance rate `0.335535`;
- `dominance_preservation`: `f1, f2`, observed new-dominance rate `0.0` with exact observed dominance preservation.

Both candidates have perfect linear reconstruction on the screening sample, but only `f1, f2` exactly preserves observed Pareto/dominance relations there. Both are again annotated `UNSUPPORTED_REDUCTION` because of `TRANSITIVE_CHAINING`.

The optimizer comparison was not stable across checkpoints. The IGD+ difference `dominance_preservation - size_span` was approximately `-17.86`, `+29.18`, `-21.08`, and `+10.06` at generations 50, 100, 200 and 400 respectively. At generation 400:

| treatment | GD+ | IGD+ | relative HV |
|---|---:|---:|---:|
| Full | 20.851960 | 20.760362 | 0.000000 |
| Reduced / size_span | 42.442734 | 41.734896 | 0.000000 |
| Reduced / dominance_preservation | 52.636475 | 51.790460 | 0.000000 |

The Full treatment is also clearly not converged: although GD+/IGD+ improve overall with budget, they remain very large and relative HV is zero at every checkpoint. The two Reduced trajectories cross repeatedly. Consequently this run gives no defensible ranking-policy conclusion for DPF1 and demonstrates directly that a better static dominance score on `Y` does not guarantee a better MOEA outcome at an arbitrary finite budget.

## Interpretation

The first end-to-end run sharpens the role of the Revisão 09 metric rather than settling the default-policy question.

For DTLZ5, static dominance preservation, independent analytical safety, and optimizer performance align very strongly: `dominance_preservation` chooses `f9,f10`, and that reduction reaches the calibrated GT while `size_span` does not. For DPF1, the static signal is even cleaner, but the optimizer trajectories do not preserve a stable ordering between the two candidate reductions under the current configuration.

A second point is now explicit: **the sampled Full optimization run is a treatment, not the ground truth**. In many-objective problems a reduced formulation may legitimately be much easier for the optimizer than the Full formulation. End-to-end quality should therefore be judged primarily against the common calibrated Pareto GT; Full is useful as a baseline only when its own convergence is adequate.

The current data support neither promoting `dominance_preservation` nor dismissing it. They do show that the ranking policy can have a large downstream effect and that the observed-Y dominance metric can be predictive in at least one important analytical control.

## Next step

Before drawing a policy conclusion:

1. calibrate the Full optimization protocol more carefully for DTLZ5 and DPF1, rather than treating 400 generations as automatically sufficient;
2. repeat the comparison across independent MOEA seeds once a defensible optimization configuration/budget is established;
3. interpret Reduced variants against the common GT, using Full as an additional treatment rather than as truth;
4. only then ask whether the sign/magnitude of the static dominance advantage predicts the distribution of end-to-end optimization outcomes.

## Status

Implementation and smoke tests pass. The first scientific workflow run completed successfully and produced informative but not yet definitive results. No ranking-policy default change is justified at this stage.
