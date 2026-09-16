# Hidden dimensional structure under regime switching

- Period: August–September 2026
- Status: **Open**
- Tracking: issue #40

## Question

Can MISDA recover dimensional structure that is real in the generator but hidden from a global pairwise-correlation graph by regime-dependent or nonlinear mixing?

## Case that exposed the problem

The controlled `regime_switching` problem (historical MOP-F) has theoretical `dl=2` and `ds=2`. A smooth switching mechanism mixes latent factors so strongly that the global pairwise correlation structure can appear effectively one-dimensional. Under the current thresholded graph semantics, both `G±` and `G+` can collapse to complete structures and yield `1/1`.

The failure is therefore different from ordinary observation noise. Even clean data generated exactly from the model can hide the second dimension from the current pairwise graph.

## Investigations and clarifications

A spectral clue was observed: after the graph-based latent signal dimension is exhausted, the next rank-correlation eigenvalue can remain far above a column-permutation null reference. This motivated the `HIDDEN_SPECTRAL_STRUCTURE` diagnostic.

The important methodological constraint is that this diagnostic does not itself tell us what the corrected dimension should be. Converting “extra spectral structure exists” directly into a guessed `dl` or `ds` would introduce a new estimator without defining its estimand or proving its relationship to the graph quantities.

Alternative directions discussed include spectral and nonlinear extensions, but no threshold-free rule has yet been established that:

- determines the number of hidden dimensions;
- distinguishes regime mixing from harmless nonlinear redundancy;
- preserves the existing regular controlled cases;
- keeps latent and structural dimension conceptually distinct.

## Outcome

The current method deliberately reports the graph-derived dimensions and marks the result `UNSUPPORTED` with `HIDDEN_SPECTRAL_STRUCTURE`. Issue #40 remains open.

A future solution requires a new mathematical estimator or model extension, followed by clean/noisy multi-seed validation and an ADR if the dimensional definitions change.

## Why keep this note

The spectral residual is a clue, not a solved estimator. Recording that distinction prevents a future implementation from turning an anomaly detector into a dimension counter merely because the known benchmark happens to have one missing dimension.
