# ADR 0021 — Conservative dominance-preservation reduction selection

- Status: Experimental
- Recorded: 2026-09-25
- Does not supersede: ADR 0017 or ADR 0020

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

## Decision

Add an experimental ranking policy named `dominance_preservation`.

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

## Trust annotation does not suppress the answer

MISDA always returns the ranking-selected MIS. It never replaces an
unsupported answer with the full objective set and never stops the analysis.
Instead, `Ranking.assessment` annotates the selected answer with one of:

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
defined over the canonical first `size_span` group for compatibility.
`MISSet.support_for(candidate)` exposes the stored support of any candidate,
including one selected by an alternative ranking.

## Evaluation contract

`dominance_preservation` requires stored dominance evidence:

```python
mis_set.evaluate(metrics=("dominance",), candidates="all")
ranking = misda.rank(mis_set, policy="dominance_preservation")
candidate = ranking.mis()
assessment = ranking.assessment
```

Alternatively, `accept_cost=True` may explicitly authorize the missing
dominance evaluation. Ranking and reporting do not silently consult truth.

## Interpretation

The policy is conservative observed-data evidence, not a proof of global
optimization equivalence. It answers: among structurally admissible MISs,
which projection introduces the fewest new dominance relations in observed Y?

The method is not required to find the smallest safe reduction. Retaining
extra objectives is acceptable when that better preserves the observed order.

## Invariants

- `size_span` remains the canonical structural default.
- discovery still enumerates the same complete MIS universe.
- structural and latent dimensions are unchanged by ranking policy.
- benchmark truth never enters runtime selection or trust annotation.
- an unsupported selected MIS is still returned and remains fully inspectable.
- `NO_REDUNDANCY` must be possible when no objective reduction is discovered.
- support attached to an alternative selected MIS is candidate-specific rather
  than borrowed from the canonical first structural group.
