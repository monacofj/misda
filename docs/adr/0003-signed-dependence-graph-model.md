# ADR 0003 — Signed dependence graph model

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

Positive and negative association have different meanings for objective reduction. Positive association can support redundancy: two objectives move together and one may be structurally substitutable for the other. Negative association instead expresses dependence/conflict and must not be converted into a positive-redundancy relation.

## Formal specification

For every valid pair `(i,j)`, MISDA computes signed Pearson correlation `r_ij` and one-tailed Fisher-z evidence based on `|r_ij|`:

```text
z_ij = atanh(|r_ij|) sqrt(N-3)
p_ij = P(Z >= z_ij),  Z ~ N(0,1).
```

At selected threshold `alpha`, the pair is statistically supported when `p_ij <= alpha`.

Two undirected graphs on the same objective vertices are then defined:

```text
(i,j) in E(G+)  iff p_ij <= alpha and r_ij > 0
(i,j) in E(G±) iff p_ij <= alpha and r_ij != 0
```

Numerically exact zero correlation creates no signed edge. Constant-objective pairs are invalid for Fisher-z evidence and therefore create no edge.

## Decision

`G+` is the structural redundancy graph. `G±` is the signed dependence graph. Both use the same statistical threshold; only the interpretation of correlation sign differs.

Negative dependence contributes to latent dependence and therefore to `G±`, but never creates a structural-redundancy edge in `G+`.

## Rationale

Treating negative conflict as redundancy would permit reduction to remove an objective precisely because it strongly opposes another objective, which is inappropriate in a multiobjective setting. The signed graph therefore captures dependence broadly, while the positive graph captures substitutability more narrowly.

## Invariants

- `G+` is a subgraph of `G±` on the same vertex set.
- every edge of `G+` has positive correlation;
- a statistically supported negative pair may appear only in `G±`;
- one common selected threshold governs both graphs;
- constants remain explicit vertices and are not silently discarded.

## Current implementation

`misda._statistics.correlation_edge_masks()` builds positive and signed masks from `log_p <= log_alpha`. `misda._graph.build_dependency_graphs()` materializes NetworkX graphs and stores correlation, `log_p`, and sign as edge attributes.

The implementation computes probability evidence using `abs(r)` but uses the original sign only when deciding graph membership. This separation is intentional.

## Permitted implementation variations

The graph library, sparse/dense representation, and vectorized edge-construction strategy may change. A conforming implementation must produce the same vertex and edge sets for the same numerical inputs and tolerance policy.

## Forbidden shortcuts / regression risks

Do not merge `G+` and `G±` into one graph, use negative edges as positive redundancy, or remove constant objectives from the vertex universe merely to simplify calculations.

## Verification

Reference tests should include positive, negative, null, constant, and mixed-sign pairs and verify the edge-set inclusion relation `E(G+) subset E(G±)`.

## References

- Fisher, R. A. (1921). On the probable error of a coefficient of correlation deduced from a small sample. *Metron*, 1, 3–32.
- Pearson, K. (1896). Mathematical contributions to the theory of evolution. III. Regression, heredity, and panmixia. *Philosophical Transactions of the Royal Society A*, 187, 253–318.