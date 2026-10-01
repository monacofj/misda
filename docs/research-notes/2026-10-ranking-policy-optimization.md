# Ranking-policy optimization audit

## Question

The ranking-policy experiments in Revisão 09 compared `size_span` and `dominance_preservation` using properties measured on the observed objective sample `Y`. In several cases, notably DTLZ5 and DPF1, `dominance_preservation` selected a different MIS and showed a strong, bootstrap-stable advantage in observed dominance preservation. That result does not by itself establish that the selected reduction is better when used by a multiobjective optimizer.

The end-to-end question is therefore:

> When `size_span` and `dominance_preservation` select different MISs from the same discovery sample, which reduction better preserves the outcome of optimization relative to the full objective problem?

## Experiment

Use the existing optimization benchmark and compare three conditions under matched budgets and seeds:

1. **Full** — optimize the complete objective set.
2. **Reduced / size_span** — optimize only the objectives selected by the default structural ranking policy.
3. **Reduced / dominance_preservation** — optimize only the objectives selected by the experimental dominance-based ranking policy.

All resulting solutions must be reevaluated in the original full objective space and compared against the same ground truth/reference using the existing MoeaBench metrics (including GD+, IGD+ and the standard benchmark diagnostics).

The first targets should be cases where the policies actually disagree, especially DTLZ5 and DPF1. A comparison is only interpretable after verifying that the corresponding Full run is sufficiently converged; otherwise a difference between reduced variants may reflect a weak full baseline rather than the effect of ranking policy.

## Interpretation

This experiment tests whether the Revisão 09 dominance-preservation score is a useful *proxy* for the property that ultimately matters to MISDA: preserving optimization outcomes after objective reduction.

Possible outcomes include:

- dominance-preservation advantage on `Y` also predicts a better Reduced optimization result;
- both policies optimize equally well despite different static scores;
- the structurally ranked MIS optimizes better even when its observed dominance-preservation score is worse.

Any of these outcomes is informative. The experiment should not change the default ranking policy by itself; policy conclusions should follow from the end-to-end evidence rather than from the static score alone.

## Status

Planned on branch `issue-83-ranking-policy-optimization`. No conclusion yet.
