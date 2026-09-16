# ADR 0016 — Public capability non-regression contract

- Status: Accepted
- Recorded: 2026-09-16

## Context

During the September 2026 API refactor, the rich stored-result renderer was removed in commit `3fa9050` (`refactor(report): remove legacy result reporting`) and its regression tests were removed in commit `dce9519` (`test: remove obsolete result-report tests`). The replacement `MISSet.report()` preserved only a small summary. The underlying scientific state remained available, but an important user-facing audit capability silently disappeared.

This failure is architectural rather than cosmetic. A refactor that preserves internal computations but removes a mature public diagnostic surface is still a regression. Removing the tests together with the implementation eliminated the mechanism that should have exposed the loss.

## Decision

User-visible capabilities that are part of normal scientific use are treated as public contracts even when their exact formatting is not an API guarantee. Refactoring may replace their implementation, but it must preserve their informational capability unless a deliberate breaking change is separately approved and documented.

For `MISSet.report()`, the minimum informational contract includes:

- original, latent, structural, and ranking-selected dimensions;
- dependence and structural graph topology;
- threshold/null-envelope state and separation regime;
- structural ranking policy, MIS count, and tie-group information;
- dimensional-support status and diagnostics;
- evaluation scope for each requested metric family;
- representative candidates from the leading structural tie groups;
- stored structural candidate metrics;
- stored linear reconstruction metrics and jackknife uncertainty;
- stored Pareto-preservation metrics;
- stored nonlinear reconstruction and null-reference metrics when available;
- observed-data Pareto-stability diagnostics when available;
- no hidden scientific computation triggered by reporting.

The benchmark report must embed the native `MISSet.report()` output verbatim in its data-derived MISDA section and keep declared-truth validation separate.

## Refactor rule

A public capability may not be removed merely because its implementation is classified as `legacy`.

Before deleting or replacing an implementation, the same change set must establish one of the following:

1. replacement behavior covered by non-regression tests that preserve the existing public capability; or
2. an explicit superseding ADR documenting the intentional breaking change, migration path, and removed behavior.

Deleting a regression test because the implementation it protects is being deleted is forbidden unless equivalent replacement coverage is already present. Tests are evidence of a contract, not disposable implementation debris.

## Verification rule

Stable user-facing capabilities must have dedicated contract tests independent of implementation details. The reporting contract tests must fail when a refactor removes an evidence family, hides partial evaluation, triggers new scientific computation from `report()`, or causes benchmark and native MISDA reporting to diverge.

The CI acceptance workflow must run the public-reporting contract as an explicit named gate in addition to the full test suite.

## Integration rule

`main` should be protected so changes cannot be integrated unless the acceptance gate succeeds. Direct pushes that bypass required checks defeat the purpose of the contract tests and should not be used for substantive refactors.

## Invariants

- refactoring is behavior-preserving by default;
- user-visible scientific diagnostics are first-class behavior;
- old implementation code may disappear only after equivalent public behavior is protected;
- tests protecting public behavior may not disappear without explicit replacement coverage;
- benchmark reporting never becomes a second independent implementation of MISDA reporting;
- reporting remains a view over stored state and has no hidden scientific side effects;
- intentional breaking changes require an explicit decision record rather than being smuggled into cleanup/refactor work.

## Current implementation

`misda._reporting` renders the rich stored-state audit for `MISSet.report()`. `tests/test_reporting_contract.py` protects its minimum information families, nonlinear/null evidence, side-effect-free behavior, and exact embedding in benchmark reports. The GitHub Actions acceptance workflow contains a named `Public reporting contract` step before the complete test suite.

## Forbidden shortcuts / regression risks

Do not delete a renderer and its tests in the same cleanup merely because both are called legacy; do not replace rich diagnostics with a summary without explicit approval; do not move MISDA-only evidence into benchmark-only code; do not rely on code review memory as the sole safeguard; and do not treat a green generic unit-test suite as sufficient when no test encodes the user-visible capability being preserved.
