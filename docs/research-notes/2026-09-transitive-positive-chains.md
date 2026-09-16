# Dimensional recovery under transitive positive chains

- Period: July–September 2026
- Status: **Open**
- Tracking: issue #39

## Question

Can thresholded pairwise dependence distinguish genuine high-dimensional structure from a strongly transitive chain in which adjacent/cumulative objectives are all highly positively correlated?

## Case that exposed the problem

The controlled `transitive_chain` generator is triangular/cumulative. Its theoretical latent and structural dimensions are both 20, but pairwise positive correlations propagate strongly enough that the thresholded graphs can collapse to complete or nearly complete structures. Under the current graph definitions, the resulting independence numbers can therefore collapse to `1/1`.

This is not a benchmark-definition mistake. The clean generator genuinely has many independent innovations even though pairwise association is dominated by the cumulative chain.

## Investigations and clarifications

Several conceptual corrections emerged from this case:

- connected components are topology diagnostics, not structural dimensions;
- chains are not cliques in the conceptual sense of a single redundant structural unit, even when thresholding can make the observed graph appear clique-like;
- simply counting connected components cannot recover the hidden innovations;
- `UNSUPPORTED` must remain a diagnostic state, not be reinterpreted as an alternate dimension estimate.

The eventual support diagnostic `TRANSITIVE_CHAINING` was designed to make this failure visible. In repeated clean and noisy validation, the case remained dimensionally unrecovered while the diagnostic fired consistently. This is useful because the current method fails loudly rather than silently, but it does not solve the estimator problem.

## Outcome

No accepted correction to `dl` or `ds` has been found. Issue #39 remains open.

Any future solution must explain mathematically how to recover independent innovations hidden by transitive pairwise association without using benchmark truth or an arbitrary chain-breaking threshold. It must also avoid changing regular block/redundancy cases that the current graph estimator handles correctly.

## Why keep this note

This is a canonical negative result for the present estimator. It prevents two recurring mistakes: treating the failure as merely an implementation bug, and “fixing” it by substituting a topology count or benchmark-specific rule that changes the estimand without justification.
