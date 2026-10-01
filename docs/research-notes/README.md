# MISDA Research Notebook

`docs/research-notes/` is the chronological, non-normative laboratory notebook of MISDA.
It preserves the scientific path: questions, hypotheses, experimental designs,
probes, negative results, corrections, suspended ideas, and conclusions.

This directory is the **single home for research history**. Results should not be
maintained in a parallel validation ledger or copied into design summaries.

The documentation layers answer different questions:

- `docs/research-notes/` — **what did we investigate, how, and what did we learn?**
- `docs/adr/` — **what decisions currently define the method and public software contract?**
- `docs/userguide.md` — **how does a user operate the current implementation?**
- specialized documents such as `docs/visualization.md` and
  `docs/experimental-correlation.md` — **how does a current specialized capability behave?**

Research notes are evidence and memory, not specification. If a note conflicts
with an accepted ADR, the ADR is authoritative. A research result that becomes a
methodological or architectural decision should be referenced by the relevant
ADR rather than duplicated there in full.

## Entry convention

A substantive investigation should normally have one dated/topic note. A note
should preserve enough of the chronological reasoning to prevent the same work
from being rediscovered:

```text
Question / hypothesis
Experimental design or analytical test
Result
Interpretation
Conclusion / decision boundary
Tracking references (issue, PR, workflow, ADR) when available
```

Routine implementation debugging, ordinary refactors, and decisions already
fully explained by an ADR do not automatically deserve a research note. The
criterion is scientific provenance: would losing this path make it likely that a
future researcher repeats a substantive dead end, misreads a known limitation,
or loses an unresolved hypothesis?

## Status vocabulary

Each note should make its current status explicit. Useful states are:

- **Adopted later** — the investigation informed a decision now normative elsewhere;
- **Adopted as validation evidence** — the result remains empirical evidence but did not itself redefine the method;
- **Rejected** — the investigated approach was tested or reasoned through and deliberately not adopted;
- **Suspended** — work was explored but intentionally removed from the active scientific path;
- **Open** — the problem remains an active research question;
- **Unvalidated hypothesis** — a conjecture was proposed but never established strongly enough to guide the method;
- **Partly adopted** — some conclusions entered the method while the broader question remains open.

## Chronological index

### July 2026

- [`2026-07-signed-correlation-and-objective-conflict.md`](2026-07-signed-correlation-and-objective-conflict.md) — signed dependence and objective conflict.
- [`2026-07-adaptive-strategy-exploration.md`](2026-07-adaptive-strategy-exploration.md) — exploration and suspension of the adaptive strategy.

### August 2026

- [`2026-08-bic-reliability-hypothesis.md`](2026-08-bic-reliability-hypothesis.md) — BIC-based reliability hypothesis.

### September 2026

- [`2026-09-alpha-null-estimator-evolution.md`](2026-09-alpha-null-estimator-evolution.md) — evolution of the `alpha_null` estimator.
- [`2026-09-complete-mis-enumeration-and-performance.md`](2026-09-complete-mis-enumeration-and-performance.md) — complete MIS enumeration and performance trade-offs.
- [`2026-09-transitive-positive-chains.md`](2026-09-transitive-positive-chains.md) — transitive chaining as a structural failure mode.
- [`2026-09-hidden-structure-under-regime-switching.md`](2026-09-hidden-structure-under-regime-switching.md) — hidden spectral structure under regime switching.
- [`2026-09-sampling-and-observation-noise-validation.md`](2026-09-sampling-and-observation-noise-validation.md) — dimensional and Pareto behavior under clean resampling and explicit observation noise.
- [`2026-09-pareto-noise-susceptibility.md`](2026-09-pareto-noise-susceptibility.md) — attempts to infer Pareto susceptibility to unknown observation noise from `Y` alone.
- [`2026-09-pearson-vs-spearman-investigation.md`](2026-09-pearson-vs-spearman-investigation.md) — Pearson versus Spearman discovery evidence.
- [`2026-09-optimization-reduction-safety-and-ranking.md`](2026-09-optimization-reduction-safety-and-ranking.md) — optimization-equivalence question, ranking probes, dominance preservation, and analytical controls.
- [`2026-09-pareto-projection-semantics-correction.md`](2026-09-pareto-projection-semantics-correction.md) — correction of the false empirical-front subset assumption under objective projection.

### October 2026

- [`2026-10-ranking-policy-default-audit.md`](2026-10-ranking-policy-default-audit.md) — pre-default audit of `dominance_preservation`, noisy representative selection, and the aggressiveness follow-up from issue #78 / PR #79. This note is the scientific provenance for PR #79: it records the hypothesis that motivated the PR, the experiments it preserves, and why the default was deliberately left unchanged.
- [`2026-10-ranking-resampling-uncertainty.md`](2026-10-ranking-resampling-uncertainty.md) — issue #80 / PR #81 follow-up testing whether row-bootstrap stability of `DeltaD` can distinguish robust dominance evidence from noise-driven representative choices; rejected as a sufficient general decision rule.

## Maintenance rule

When an investigation concludes:

1. update or create its research note;
2. create/update an ADR only if a normative decision changes;
3. update user-facing documentation only if current behavior or usage changes;
4. do not leave a second historical copy elsewhere in `docs/`.

This keeps the repository readable in both directions: the research notebook
explains **why we arrived here**, while the ADR set defines **what here means**.
