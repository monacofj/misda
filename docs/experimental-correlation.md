# Experimental correlation backends

MISDA uses Pearson correlation as the canonical dependence measure in structural discovery. The public default remains unchanged:

```python
import misda

result = misda.discover(Y)
```

This is equivalent to:

```python
result = misda.discover(
    Y,
    correlation="pearson",
    experimental=False,
)
```

## Spearman research backend

A Spearman backend is retained for reproducible investigation of monotonic dependence. It is intentionally gated and must be enabled explicitly:

```python
result = misda.discover(
    Y,
    correlation="spearman",
    experimental=True,
)
```

Calling `correlation="spearman"` without `experimental=True` raises an error explaining that Spearman is not part of the canonical MISDA method.

The `experimental` switch is an opt-in barrier, not a second algorithm. Pearson and Spearman use the same discovery pipeline, graph construction, MIS enumeration, dimensional-support diagnostics, ranking, evaluation, reporting objects, and visualization facilities. Only the correlation statistic and its corresponding permutation-null calculation are selected at the statistics boundary.

The result records the selected backend:

```python
result.correlation     # "pearson" or "spearman"
result.experimental    # False for Pearson, True for Spearman
```

Passing `experimental=True` while keeping `correlation="pearson"` is permitted and still produces the canonical Pearson result. The flag exists to authorize non-canonical backends; it does not itself change the method.

## Scientific status

Spearman is preserved as a research option because the September 2026 investigation found a genuine advantage for strongly nonlinear monotonic relations, especially under clean resampling, but also a marked loss of robustness under even small observation noise once the monotonic transformation becomes sufficiently steep. The evidence therefore did not support replacing Pearson as the default.

The complete experimental rationale, formulas, benchmark design, intermediate conjectures, corrections, resampling/noise studies, the `k x sigma` sweep, and the final interpretation are recorded in:

`docs/research-notes/2026-09-pearson-vs-spearman-investigation.md`

The experimental option exists so those results can be reproduced and extended as the rest of MISDA evolves, without maintaining a separate frozen implementation branch.
