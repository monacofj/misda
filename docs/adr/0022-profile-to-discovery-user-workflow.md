# ADR 0022 — Profile-to-discovery user workflow

- Status: Accepted
- Recorded: 2026-10-01
- Corrected: 2026-10-01 under issue #91
- Supersedes: ADR 0019 as the canonical user-facing workflow
- Partially supersedes: ADR 0011 public-flow section and the public-default clause of ADR 0017
- Refines but does not supersede: ADR 0007, ADR 0014, ADR 0017, and ADR 0021

## Correction note

The first version of this ADR, merged in PR #90, incorrectly made
`discovery(profile)` apply a ranking policy and select a representative MIS.
That contradicted ADR 0007, whose central invariant is that discovery,
evaluation, and ranking are separate operations and that ranking policy is not
an input to structural discovery.

Issue #91 corrects that regression explicitly rather than silently redefining
the older architectural contract. The corrected workflow below restores ADR
0007 as a binding invariant.

## Context

The low-level MISDA API deliberately separates structural discovery,
candidate evidence, and preference among already discovered candidates:

```python
mis_set = misda.discover(Y)
mis_set.evaluate(...)
ranking = misda.rank(mis_set, policy=...)
```

That separation remains scientifically important. A higher-level user workflow
is still useful because the calibrated alpha path may contain several distinct
structural regimes and the user should not need to choose an aggressiveness
value manually.

There are therefore three logically distinct questions:

1. which alpha/aggressiveness regime is admissible for the observed `Y`;
2. what complete MIS universe is discovered in that regime;
3. which discovered MIS is preferred under a declared ranking policy.

These questions must remain separate in the public API.

## Decision

The canonical user-facing workflow is:

```python
profile = misda.profile(Y)
profile.report()

mis_set = misda.discovery(profile)
mis_set.report()

ranking = misda.rank(mis_set)
ranking.report()

mis = ranking.mis()
mis.report()
```

The convenience form

```python
mis_set = misda.discovery(Y)
```

is equivalent to profiling first and then returning the MISSet associated with
the selected profile regime.

`profile`, `discovery`, and `rank` have separate ownership:

```text
profile    -> selects the threshold regime / alpha
 discovery -> returns the complete MISSet for that regime
 rank       -> orders the already discovered MISs and selects a representative
```

## Profile semantics

`profile(Y)` is a diagnostic map over the calibrated alpha/aggressiveness path.
It enumerates distinct structural regimes induced by threshold changes and
shows their consequences, including:

- alpha/aggressiveness;
- latent and structural dimensions;
- structural/latent topology and components through the stored regime result;
- complete MIS universe and canonical structural order;
- dimensional-support evidence;
- `alpha_onset`, `alpha_null`, and separation/null-envelope context.

Profile construction may internally build the MISSet for each structural regime
because support is defined on discovered MISs. This does not make profile a
ranking operation: no ranking policy is accepted or consulted when selecting
the regime.

Candidate-specific support is stored for every MIS in each regime, matching the
low-level `discover()` contract. The historical aggregate `MISSet.support`
remains defined over the complete first canonical `size_span` scientific tie
group as required by ADR 0014. Deterministic ordering inside that tie group may
not decide regime support.

Regimes are represented at their most aggressive boundary. Critical thresholds
are probed on both sides and structurally equivalent consecutive states are
compressed.

### Regime status and automatic selection

A regime has one of the following profile-level interpretations:

```text
NO_REDUNDANCY        no structural reduction is present
SUPPORTED_REDUCTION  aggregate discovery support is SUPPORTED
PARTIALLY_SUPPORTED  the first structural tie group has mixed support
UNSUPPORTED_REDUCTION aggregate discovery support is UNSUPPORTED
```

`profile.selected` is the most aggressive regime that is fully admissible for
automatic use. The admissible states are:

```text
NO_REDUNDANCY
SUPPORTED_REDUCTION
```

`PARTIALLY_SUPPORTED` remains visible but is not auto-selected because a
scientific tie must not be resolved by deterministic candidate order.
`UNSUPPORTED_REDUCTION` is likewise not auto-selected.

If no admissible regime exists, `profile.selected` is `None`. The profile
remains inspectable, but high-level discovery refuses to invent a fallback
regime.

`profile.report()` remains a compact table. Its final column may include a
short human-readable interpretation of the status, and the report must state
that selecting a regime does not select an MIS. Reporting remains a view over
stored state and triggers no scientific computation.

## Discovery semantics

`discovery(profile)` returns exactly the `MISSet` stored for
`profile.selected`.

It does not:

- accept a ranking policy;
- rank candidates;
- select a representative MIS;
- filter unsupported candidates;
- compute dominance or Pareto preference evidence.

If `profile.selected is None`, high-level discovery raises rather than silently
falling back to `aggressiveness=1` or another regime.

`discovery(Y)` is only the convenience composition
`profile(Y) -> discovery(profile)`.

## Ranking semantics

`rank()` is the only public step that applies a ranking policy.

The public default is:

```text
dominance_preservation
```

because the clean end-to-end experiments consolidated in #85 showed that this
policy is better aligned with downstream optimization than structural
`size_span` when the observed geometry is representative.

The canonical structural ordering stored by discovery remains `size_span`.
Users may request it explicitly:

```python
ranking = misda.rank(mis_set, policy="size_span")
```

The default dominance ranking may compute missing dominance evidence as part of
the ranking operation. This is ranking-time evidence enrichment, not discovery.
Callers that require precomputed evidence may pass `accept_cost=False`.

Ranking considers the complete requested candidate set. Unsupported candidates
are not filtered out. If the ranking-selected MIS is contradicted by its stored
support diagnostics, `Ranking.assessment` reports
`UNSUPPORTED_REDUCTION` while preserving the selected candidate for inspection
and validation.

## Reporting non-regression

This ADR does not redesign the established report formats of:

- `MISSet.report()`;
- `Ranking.report()`;
- `MISCandidate.report()`.

Those formats remain protected by ADR 0016 and their existing reporting
contract tests. Only `profile.report()` is new to this workflow and may be
refined within the compact tabular design described above.

## Data-quality boundary

`profile()` and ranking diagnose the observed `Y`. They do not claim to infer
an arbitrary external measurement-noise variance, reliability ratio, or
systematic distortion from one matrix. Severe corruption of observed geometry
can still mislead support and dominance evidence; the noisy-safety experiments
record that epistemic boundary.

## Compatibility

- `misda.discover(Y, aggressiveness=...)` remains available for exact low-level
  structural control.
- `MISSet.evaluate(...)` remains available.
- `misda.rank(...)` remains a view over an existing `MISSet`; its public default
  is now `dominance_preservation`.
- `size_span` remains the canonical structural ordering owned by discovery and
  remains explicitly selectable as a ranking policy.
- benchmark runners may use explicit policies and low-level primitives where
  scientific instrumentation requires them.
- notebook and user-document migration remains tracked separately in #87.

## Verification

Contract tests must verify at least that:

1. `discovery()` has no ranking-policy parameter;
2. `discovery(profile)` returns an `MISSet` and does not compute ranking
   evidence;
3. `rank()` is the policy-bearing operation and defaults to
   `dominance_preservation`;
4. ranking/evaluation do not change candidate membership, graph structure,
   graph-derived dimensions, or canonical structural order;
5. profile reporting has no scientific side effects;
6. profile regimes retain candidate-specific support for all discovered MISs;
7. existing `MISSet`, `Ranking`, and `MISCandidate` report contracts are
   unchanged.

## Research basis

The default-policy and data-quality rationale is recorded in:

- `docs/research-notes/2026-10-ranking-policy-optimization.md`;
- `docs/research-notes/2026-10-noisy-ranking-safety.md`;
- `docs/research-notes/2026-10-noisy-ranking-supported-stress.md`;
- `docs/research-notes/2026-10-ranking-policy-noise-scale-and-default-hypothesis.md`.
