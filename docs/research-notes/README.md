# MISDA Research Notes

`docs/research-notes/` preserves the non-normative scientific history of MISDA: questions investigated, alternative hypotheses, experiments, negative results, and lines of inquiry that were suspended or superseded.

These notes answer a different question from the ADRs:

- `docs/adr/` records decisions that define the current method or software contract;
- `docs/research-notes/` records what was explored and what was learned, including approaches that were rejected or never adopted.

A research note is therefore evidence and memory, not specification. If a note conflicts with an accepted ADR, the ADR is normative. A note may point to issues, pull requests, benchmark artifacts, ADRs, or validation records that provide the primary implementation trail.

## Status vocabulary

Each note should make its current status explicit. Useful states are:

- **Adopted later** — the investigation informed a decision that is now normative elsewhere;
- **Rejected** — the investigated approach was tested or reasoned through and deliberately not adopted;
- **Suspended** — work was explored but intentionally removed from the active scientific path;
- **Open** — the problem remains an active research question;
- **Unvalidated hypothesis** — a conjecture was proposed but never established strongly enough to guide the method.

## Current notes

- `2026-07-signed-correlation-and-objective-conflict.md`
- `2026-07-adaptive-strategy-exploration.md`
- `2026-08-bic-reliability-hypothesis.md`
- `2026-09-alpha-null-estimator-evolution.md`
- `2026-09-complete-mis-enumeration-and-performance.md`
- `2026-09-transitive-positive-chains.md`
- `2026-09-hidden-structure-under-regime-switching.md`
- `2026-09-pareto-noise-susceptibility.md`

The collection is intentionally selective. Routine implementation debugging, ordinary refactors, and decisions already fully explained by an ADR do not automatically deserve a research note. The criterion is whether preserving the path of investigation can prevent future researchers from repeating a substantive dead end or losing a useful unresolved hypothesis.
