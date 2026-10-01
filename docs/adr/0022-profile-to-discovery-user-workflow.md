# ADR 0022 — Profile-to-discovery user workflow

- Status: Accepted
- Recorded: 2026-10-01
- Supersedes: ADR 0019 as the canonical user-facing workflow
- Partially supersedes: ADR 0011 public-flow section
- Refines but does not supersede: ADR 0017 and ADR 0021

## Context

The low-level MISDA API deliberately separated structural discovery,
candidate evaluation, and ranking:

```python
mis_set = misda.discover(Y)
mis_set.evaluate(...)
ranking = misda.rank(mis_set, policy=...)
```

That separation remains valuable for benchmarks, validation, and expert
instrumentation. It is, however, too low-level as the canonical workflow for a
user who wants a defensible reduction recommendation.

Two logically distinct decisions were being exposed as if they were one:

1. which alpha/aggressiveness regime should be used for the observed `Y`;
2. which MIS should represent the chosen regime.

The ranking-policy experiments in issues #83 and #84 further showed that these
questions should remain distinct. Structural support and threshold sensitivity
are properties of the observed-data profile. Representative selection is a
ranking problem within a selected regime.

## Decision

The canonical user-facing workflow is:

```python
profile = misda.profile(Y)
profile.report()

result = misda.discovery(profile)
result.report()
```

The convenience form

```python
result = misda.discovery(Y)
```

is equivalent to profiling first and then discovering from the resulting
profile.

The low-level `discover`, `evaluate`, and `rank` functions remain public and
retain their existing semantics for scientific and advanced use.

## Profile semantics

`profile(Y)` is a diagnostic map over the calibrated alpha/aggressiveness path.
It enumerates distinct structural regimes induced by threshold changes and
shows their consequences, including:

- alpha/aggressiveness;
- latent and structural dimensions;
- structural/latent topology and components through the stored regime result;
- MIS universe and structural ranking;
- dimensional-support assessment and reasons;
- `alpha_onset`, `alpha_null`, and separation/null-envelope context.

Regimes are represented at their **most aggressive boundary**. Critical
thresholds are probed on both sides and structurally equivalent consecutive
states are compressed. This lets the stored aggressiveness of a regime mean the
most aggressive point at which that same structural state still holds.

`profile.report()` must remain observational: reporting does not trigger
additional candidate evaluation.

## Selected regime

`profile.selected` is the **most aggressive acceptable regime**.

A regime is acceptable when the canonical structural selection at that regime
has assessment:

```text
NO_REDUNDANCY
SUPPORTED_REDUCTION
```

A regime whose canonical structural selection is
`UNSUPPORTED_REDUCTION` is not acceptable for automatic selection.

If no acceptable regime exists, `profile.selected` is `None` and the high-level
workflow abstains rather than silently using `aggressiveness=1`.

Selection does not hide alternatives: all regimes remain available for
inspection in the profile.

## Discovery semantics

`discovery(profile)` operates only on `profile.selected` and then chooses a
representative MIS within that regime.

The default high-level ranking policy is:

```text
dominance_preservation
```

because the clean end-to-end experiments in #83 showed that it is more aligned
with downstream optimization than the structural `size_span` order when the
observed geometry is representative. Issue #84 established the epistemic
boundary: severe corruption of `Y` can mislead dominance ranking and even
support inferred from the same data. The default is therefore an
observed-data recommendation, not an unconditional noise-robustness claim.

Users may request the structural policy explicitly:

```python
result = misda.discovery(profile, rank_policy="size_span")
```

The low-level call `misda.rank(mis_set)` continues to default to `size_span`.
This preserves ADR 0017: `size_span` remains the canonical zero-cost structural
ordering. The new high-level default applies only to the recommendation layer.

## Abstention and assessment

If `profile.selected is None`, `discovery(profile)` returns an abstaining result
rather than inventing a reduction.

If a selected regime exists, the ranking-selected MIS retains its
candidate-specific `ReductionAssessment`. A high-level result therefore remains
inspectable even when alternative ranking evidence exposes a candidate-specific
warning.

## Data-quality boundary

`profile()` diagnoses structure **conditional on the observed `Y`**. It does not
claim to estimate an external measurement-noise variance, reliability ratio, or
systematic distortion from a single matrix.

Known data-quality information may inform how a user interprets the profile or
chooses a ranking policy. X-aware or replicate-aware diagnostics remain separate
research questions.

## Compatibility

- `misda.discover(Y, aggressiveness=...)` remains available.
- `MISSet.evaluate(...)` remains available.
- `misda.rank(...)` remains available and defaults to `size_span`.
- benchmark runners may continue using the low-level API where exact control of
  regime, candidate scope, and evaluation cost is scientifically necessary.
- notebook and user-document migration is tracked separately after this API
  contract stabilizes.

## Research basis

The default-policy and data-quality rationale is recorded in:

- `docs/research-notes/2026-10-ranking-policy-optimization.md`;
- `docs/research-notes/2026-10-noisy-ranking-safety.md`;
- `docs/research-notes/2026-10-noisy-ranking-supported-stress.md`;
- `docs/research-notes/2026-10-ranking-policy-noise-scale-and-default-hypothesis.md`.
