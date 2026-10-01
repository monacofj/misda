# ADR 0021 — Conservative dominance-preservation reduction selection

- Status: Accepted for the public ranking default
- Recorded: 2026-09-25
- Promoted: 2026-10-01 after the clean/noisy default-policy audit consolidated in #85
- Does not supersede: ADR 0017 structural-order semantics or ADR 0020

## Context

MISDA is used on observed objective data Y without access to the unknown
generator or benchmark truth. Optimization benchmarks may use external truth
only to validate the method; truth must never enter discovery, evaluation,
ranking, or the runtime trust annotation.

The optimization controls showed that empirical Pareto retention is useful but
can be unstable on DTLZ5. A stronger observed-data diagnostic measures how
many row pairs with no dominance relation in full Y acquire a strict dominance
relation after projecting to a candidate MIS.

Across the tested DTLZ5 and DPF1 controls this global dominance distortion
selected the known safe candidate more consistently than Pareto retention.
A variable-cardinality safe control showed that the metric is deliberately
conservative: it may prefer a larger safe MIS over a smaller safe MIS. This is
acceptable because MISDA prioritizes safe reduction, not maximal reduction.

The later clean/noisy audit established two additional points. First,
`dominance_preservation` is the best-supported optimization-oriented public
default when observed Y is representative. Second, severe corruption of Y can
mislead both dominance geometry and support inferred from the same observations;
this is a data-quality/identifiability boundary rather than a reason to fold
ranking back into discovery.

## Decision

The ranking policy `dominance_preservation` is the public default of
`misda.rank(mis_set)`.

For a candidate retained set S, let U be the unordered row pairs for which
neither row dominates the other in full Y. Let N_S be the members of U for
which one row dominates the other after projection to Y_S. Define

```text
new_dominance_rate(S) = |N_S| / |U|
```

with value zero when U is empty. Lower values rank first.

Scientific tie groups are defined only by equal `new_dominance_rate`. Inside
an exact tie, larger MISs are ordered first as a conservative operational
tie-break, followed by deterministic objective labels. The tie-break does not
split the scientific group.

The canonical structural order stored by discovery remains `size_span` under
ADR 0017. Making dominance preservation the default of the separate `rank()`
operation does not change discovery, structural dimensions, candidate identity,
or canonical structural ordering.

## Trust annotation does not suppress the answer

MISDA always returns the ranking-selected MIS. It never replaces an
unsupported answer with the full objective set and never silently filters an
unsupported candidate out of the ranking universe.

`Ranking.assessment` annotates the selected answer with one of:

```text
NO_REDUNDANCY
SUPPORTED_REDUCTION
UNSUPPORTED_REDUCTION
```

`NO_REDUNDANCY` means the selected MIS contains all original objectives.
`SUPPORTED_REDUCTION` means the selected proper subset has no contradiction
from the current candidate-specific support diagnostics.
`UNSUPPORTED_REDUCTION` means MISDA still returns that proper subset, but the
current internal diagnostics say it should not be trusted. This behavior is
intentional so validation can measure how wrong unsupported answers become.

## Candidate-specific support

Support evidence is computed and stored for every discovered MIS using the
shared permutation engine. The historical aggregate `MISSet.support` remains
defined over the canonical first `size_span` group for compatibility and for
regime-level discovery support. `MISSet.support_for(candidate)` exposes the
stored support of any candidate, including one selected by dominance ranking.

## Evaluation contract

`dominance_preservation` requires dominance evidence for every ranked MIS.
The canonical user flow may therefore use:

```python
ranking = misda.rank(mis_set)
candidate = ranking.mis()
assessment = ranking.assessment
```

The default public call authorizes computation of missing dominance evidence as
part of the ranking operation. This is evidence enrichment owned by ranking and
must not alter the discovered MIS universe, graph structure, graph-derived
dimensions, or canonical order.

Callers that need strict control of computation may pre-evaluate evidence or
forbid implicit ranking-time evaluation:

```python
mis_set.evaluate(metrics=("dominance",), candidates="all")
ranking = misda.rank(
    mis_set,
    policy="dominance_preservation",
    accept_cost=False,
)
```

Benchmark truth is never used.

## Interpretation

The policy is conservative observed-data evidence, not a proof of global
optimization equivalence. It answers: among structurally admissible MISs,
which projection introduces the fewest new dominance relations in observed Y?

The method is not required to find the smallest safe reduction. Retaining
extra objectives is acceptable when that better preserves the observed order.

## Invariants

- `dominance_preservation` is the public default of `rank()`.
- `size_span` remains the canonical structural order owned by discovery.
- discovery still enumerates the same complete MIS universe.
- structural and latent dimensions are unchanged by ranking policy.
- benchmark truth never enters runtime selection or trust annotation.
- an unsupported selected MIS is still returned and remains fully inspectable.
- unsupported candidates are not filtered from ranking merely because support
  contradicts them.
- `NO_REDUNDANCY` must be possible when no objective reduction is discovered.
- support attached to a selected MIS is candidate-specific rather than borrowed
  from the canonical first structural group.

## Research history

The experiments that motivated this policy, including DTLZ5, DPF1, DTLZ2,
SAFE_PATH, Pareto-retention comparison, and the conservative decision probe, are
recorded chronologically in
`docs/research-notes/2026-09-optimization-reduction-safety-and-ranking.md`.

The pre-default audit and the subsequent clean/noisy ranking-policy work are
recorded in the October 2026 research notes and were consolidated before the
public-default decision in #85.
