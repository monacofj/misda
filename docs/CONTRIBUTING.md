# Contributing to MISDA

MISDA is research software. Changes must preserve both the public software
contract and the scientific provenance of methodological changes.

## Public behavior

Before changing discovery, evaluation, ranking, reporting, visualization, or
benchmark behavior:

1. identify the relevant ADRs in `docs/adr/`;
2. preserve their invariants unless the change intentionally supersedes them;
3. update tests that protect the public contract;
4. do not silently remove report fields, metrics, selectors, or reproducibility
   controls.

A change that intentionally alters a normative methodological or architectural
decision requires a new ADR or an explicit superseding update to the ADR set.

## Scientific provenance

`docs/research-notes/` is the chronological laboratory notebook. Substantive
investigations should preserve the question, experimental or analytical test,
result, interpretation, and conclusion, including negative or rejected results.

Do not create parallel historical ledgers for validation results or design
rationale. The documentation roles are:

- `docs/adr/`: current normative method and public contract;
- `docs/research-notes/`: scientific history and open/rejected hypotheses;
- `docs/userguide.md`: current user workflow;
- specialized documents: current capability-specific instructions.

If an investigation changes the method, record the history in a research note
and the adopted decision in an ADR, linking the two rather than copying the
full experimental narrative into both.

## Benchmarks and expensive validation

Benchmark truth must remain outside runtime MISDA discovery, evaluation, and
ranking. Scientific validation workflows may be heavier than ordinary CI and
should not become implicit pull-request costs unless they are genuine regression
gates. Manual validation workflows must make their selected batteries explicit
and preserve reproducible artifacts when appropriate.

## Reproducibility

Stochastic experiments must use explicit seeds and record enough configuration
to reproduce the result. When a benchmark conclusion depends on a particular
commit or workflow run, record that provenance in the corresponding research
note.
