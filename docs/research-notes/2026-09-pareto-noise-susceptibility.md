# Pareto susceptibility to unknown observation noise

- Period: September 2026
- Status: **Rejected as a generic Y-only warning**
- Tracking: issues #46, #47, #49; PRs #48 and #50

## Question

The noisy controlled studies showed that some problems preserve Pareto membership under observation perturbations much better than others, even when dimensional recovery remains correct. The practical question was whether MISDA could warn a real user about this susceptibility using only the observed matrix `Y`, without access to clean latent objectives `Z` or a known observation model.

The target was deliberately narrower than “detect noise”. A legitimate Y-only diagnostic would have to say something about susceptibility of the observed Pareto structure without claiming that noise actually exists or estimating its magnitude.

Two principled approaches were investigated.

## Attempt 1 — worst-case Pareto membership stability radius

Issue #47 / PR #48 investigated the smallest range-normalized symmetric `L_inf` perturbation capable of changing exact Pareto membership.

The attraction was mathematical cleanliness: the quantity depends only on `Y`, requires no benchmark truth, and measures how close the observed front is to *some* membership-changing perturbation.

The implementation derived both loss and gain directions and passed the full acceptance gate. It was then evaluated against the existing noisy-robustness evidence without tuning the formula after seeing benchmark outcomes.

### Result

The radius was a poor predictor of susceptibility to random observation noise. Across the canonical robustness cases, rank association between the clean radius and later membership degradation was weak and approached zero as perturbation scale increased. Antagonistic cases supplied the clearest counterexample: they can have an extremely small adversarial radius while remaining highly stable under the tested random Gaussian perturbations.

### Interpretation

The radius measures **worst-case adversarial sensitivity**, not the volume or probability of dangerous perturbations. A very small exceptional direction may exist even when a random perturbation is overwhelmingly unlikely to follow it.

### Decision

PR #48 was closed without merge. The mathematical quantity was not promoted to the public method because it did not answer the intended practical question.

## Attempt 2 — stochastic Pareto perturbation-response curve

Issue #49 / PR #50 investigated a different estimand. For an explicit perturbation law, define a curve

```text
S_Y(sigma) = E[J(P(Y), P(Y + sigma * s(Y) * E))]
```

where `s(Y)` is an objective-wise empirical scale, `E` is a standardized random perturbation, and `J` is exact-membership Jaccard.

This asks not for the smallest possible harmful perturbation, but for the expected response of Pareto membership under a specified perturbation model.

### Result under independent perturbations

For iid Gaussian perturbations scaled in the same way as the noisy benchmark observation model, the Y-only curve closely reproduced independent noisy-robustness outcomes. Across the tested problems, Spearman association was approximately `0.976–0.994` over the tested positive sigma values, with small mean absolute Jaccard error.

Changing only the one-dimensional marginal law to equal-variance Laplace or uniform perturbations preserved very similar qualitative ordering and curve values.

### Failure under dependence misspecification

The strong result did not survive arbitrary changes in cross-objective dependence. Moderate Gaussian correlation between objective errors degraded agreement, and fully correlated per-row perturbations could reverse the cross-problem ordering.

Thus the curve is useful only **conditional on a perturbation dependence model**. A single observed matrix `Y` does not in general identify the magnitude, marginal law, or dependence structure of the unknown observation process.

### Decision

PR #50 was closed without merge. The curve was not promoted to an automatic warning.

## Overall conclusion

The original #46 goal — a generic, reproducible warning for real observation-noise Pareto risk from a single `Y` and without observation-model assumptions — was not achieved.

The negative conclusion is stronger than “we did not find the right scalar”. The experiments separate two fundamentally different quantities:

- worst-case geometric susceptibility can be computed from `Y` but need not predict random perturbation behavior;
- stochastic susceptibility can be estimated accurately when a perturbation model is supplied, but the model itself is not identifiable from one observed `Y` in general.

The current MISDA therefore does **not** infer observation-noise Pareto risk from `Y` alone. `controlled_noisy` and `noisy_robustness` remain the appropriate tools for documenting behavior under explicit observation models. A conditional perturbation-response analysis could become meaningful if external information supplies the error model, for example replicated measurements or justified instrumental uncertainty.

## Why keep this note

Both investigated approaches are plausible enough that another researcher could spend substantial effort rediscovering them. The first fails because adversarial distance is the wrong estimand for random noise; the second succeeds conditionally but cannot become a model-free warning. Those are reusable scientific results even though no code was merged.
