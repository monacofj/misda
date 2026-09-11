# ADR 0012 — Reproducibility and stochastic computation

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

MISDA contains stochastic procedures, principally permutation-based null estimation and nonlinear model evaluation. Scientific reproducibility requires that stochasticity be explicit, deterministic under a fixed seed, and isolated from unrelated global RNG state.

## Decision

Public stochastic procedures must accept or derive from explicit deterministic seeds. A fixed input dataset, fixed public controls, fixed seed, and fixed implementation/version must reproduce the same stochastic stream and therefore the same public result, subject only to the centralized numerical tolerance contract.

Stochastic procedures must use local generator state rather than implicitly consuming process-global random state.

## Invariants

- seeds are explicit at the public boundary where stochastic discovery occurs;
- permutation-null estimation uses a dedicated local RNG;
- independent objective-column permutations are generated from that RNG;
- deterministic seed derivation is used for nested nonlinear evaluation;
- adding unrelated random calls elsewhere in an application must not change MISDA results;
- stored convergence/evidence state must not falsely imply deterministic certainty when stochastic estimation did not converge.

## Current implementation

Structural null estimation uses `numpy.random.default_rng(seed)` and stores both the seed and final bit-generator state in `NullAlphaEstimate`. The current public discovery default seed is deterministic. Nonlinear evaluation uses deterministic seed derivation for nested Random Forest fitting and model-selection operations.

Numerical acceptance gates centralize floating comparison tolerances at

```text
GATE_RTOL = 0
GATE_ATOL = 1e-12.
```

Discrete graph/candidate quantities require exact equality.

## Permitted implementation variations

A different RNG implementation or parallel generation strategy may be used only if its reproducibility semantics are explicitly versioned and tests are updated deliberately. Parallelism must not make results scheduling-dependent.

## Forbidden shortcuts / regression risks

Do not use module-global `numpy.random` state, derive seeds from unstable object hashes or wall-clock time, allow thread scheduling to choose random streams implicitly, or conceal stochastic nonconvergence behind deterministic-looking outputs.

## Verification

Tests should verify repeated equality under fixed seeds, independence from unrelated global RNG calls, distinct outputs where seed sensitivity is expected, and deterministic replay of benchmark/notebook reference runs.