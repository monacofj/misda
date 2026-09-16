# Signed correlation and objective conflict

- Period: July 2026
- Status: **Adopted later**
- Normative outcome: ADR 0003 — Signed dependence graph model

## Question

Early formulations had to decide whether dependence should be represented by signed correlation or by correlation magnitude. The tempting simplification was to treat large `|rho|` as a generic indication that one objective could stand in for another.

## Investigation

The key counterexample is a strongly anti-correlated objective pair. A pair with `rho ~= +1` and one with `rho ~= -1` are both strongly dependent statistically, but they have opposite interpretations in a multiobjective problem.

Positive association can support structural substitutability: if two objectives move together, retaining one may preserve much of the structural information carried by the other. Strong negative association instead represents conflict or trade-off. Removing an objective because it strongly opposes another would erase precisely the type of distinction that a multiobjective formulation is intended to preserve.

This led to separating two uses of the same signed evidence:

- a positive-redundancy graph `G+`, containing only statistically supported positive associations;
- a signed-dependence graph `G±`, containing statistically supported positive and negative associations.

The statistical evidence can still depend on `|rho|`; the sign is restored when graph membership is interpreted.

## Outcome

Using `|rho|` directly as a redundancy criterion was rejected. Negative dependence must not create a positive-redundancy edge.

The distinction later became normative in ADR 0003. It also became the basis for separating structural and latent dimensionality rather than trying to express both with one graph.

## Why keep this note

The absolute-correlation alternative is superficially attractive and easy to reintroduce during simplification or optimization. This note records that it was not omitted accidentally: it was considered and rejected because dependence strength and structural redundancy are not the same concept in a multiobjective setting.
