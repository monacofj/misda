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

## Interpretation

This experiment tests whether the Revisão 09 dominance-preservation score is a useful *proxy* for the property that ultimately matters to MISDA: preserving optimization outcomes after objective reduction.

Possible outcomes include:

- dominance-preservation advantage on `Y` also predicts a better Reduced optimization result;
- both policies optimize equally well despite different static scores;
- the structurally ranked MIS optimizes better even when its observed dominance-preservation score is worse.

Any of these outcomes is informative. The experiment should not change the default ranking policy by itself; policy conclusions should follow from the end-to-end evidence rather than from the static score alone.

## Status

Implementation complete and smoke-tested. Scientific DTLZ5/DPF1 audit launched from this commit; results and interpretation are still pending and must be recorded here before drawing a policy conclusion.
