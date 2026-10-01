# Pareto projection semantics correction

- Period: September 2026
- Status: **Adopted later**
- Tracking: final cleanup in PR #76
- Normative reference: ADR 0020

## Question

What set relation, if any, must hold between the empirical nondominated set in full objective space and the empirical nondominated set after projecting the same sampled rows onto a retained objective subset?

## Initial conjecture

During the optimization-ranking work, an overly strong claim was briefly used:

```text
dropping objectives only creates dominance
ND(Y_S) subseteq ND(Y)
```

Under that conjecture, Pareto validity would be identically one and Jaccard agreement would collapse to retention.

## Counterexample

Consider minimization with

```text
Y = [[0, 1],
     [0, 2]]
```

In the full two-objective space, the first row strictly dominates the second. If only the first objective is retained, both projected rows equal `0`. The strict dominance relation disappears because the projection creates an exact tie, and both rows are nondominated in the reduced one-dimensional sample.

Thus objective projection can do both of the following:

1. create new strict dominance among rows that were incomparable in full `Y`;
2. erase an existing strict dominance relation when all coordinates carrying strict improvement are removed and the projected rows tie.

There is therefore no universal subset relation between the two empirical nondominated sets.

## Consequence

The implementation and documentation were corrected so that:

- `pareto_retention`, `pareto_validity`, and `pareto_jaccard` remain distinct quantities;
- no reporting text describes validity as identically one;
- tests include a deterministic projected-tie counterexample;
- benchmark probes report observed subset relations only when they happen in that finite sample;
- ADR 0020 states the corrected projection semantics.

In generic continuous samples exact projected ties may be rare, so validity can empirically equal one and Jaccard can then equal retention. That is a contingent sample property, not a theorem of objective projection.

## Why keep this note

The false subset claim is intuitive and easy to reintroduce. The counterexample records why the current three-metric Pareto contract is deliberate rather than redundant.
