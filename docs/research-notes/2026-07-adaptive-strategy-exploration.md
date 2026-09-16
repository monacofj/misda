# Adaptive strategy exploration

- Period: July–August 2026
- Status: **Suspended**
- Current contract: static MISDA is the active scientific path

## Question

An adaptive formulation was explored as an alternative to the static thresholded-graph method. The goal was to let the data determine not only graph structure but also a preferred operating point along a reduction/fidelity trade-off.

## Historical direction

The adaptive work explored a substantially richer pipeline than the current static method. Historical code and planning included:

- discrete candidate thresholds rather than an unconstrained continuous search;
- bootstrap/OOB validation;
- a Pareto trade-off between reduction and Pareto preservation/recall;
- candidate recommendation using a knee-point criterion;
- adaptive result/candidate objects and separate OOB summaries.

The motivation was reasonable: a single fixed structural operating point might miss useful trade-offs that become visible when reduction quality is evaluated over multiple thresholds.

## What became problematic

The adaptive path accumulated several layers of semantics at once: threshold selection, candidate generation, out-of-bag validation, Pareto criteria, recommendation, and stochastic stopping. This made it difficult to tell which quantity was an estimator, which was validation evidence, and which was merely a preference rule.

At the same time, the static formulation was being clarified around a smaller set of explicit estimands: signed dependence graphs, graph independence numbers, complete MIS discovery, separate candidate evaluation, and separate ranking.

The resulting methodological priority was to stabilize and validate the static path before reopening adaptive selection.

## Outcome

Adaptive analysis was suspended and removed from the active acceptance gate. Current notebooks and public guidance use the static method. The historical adaptive implementation is not treated as an alternative estimator whose conclusions should be combined with current static results.

This was a suspension, not a proof that adaptive analysis is impossible or useless. Reopening it would require a new specification that cleanly separates:

1. structural estimation;
2. candidate evaluation;
3. uncertainty/validation;
4. preference or recommendation.

It should also be judged against the now-established static API and ADRs rather than resumed from old implementation details by default.

## Why keep this note

The repository history contains substantial adaptive machinery, so a future contributor could reasonably assume it is merely unfinished code waiting to be re-enabled. It is not. It represents an explored research direction that was intentionally suspended while the static model was made scientifically explicit.
