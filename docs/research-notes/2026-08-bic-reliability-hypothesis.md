# BIC as a reliability signal

- Period: August 2026
- Status: **Unvalidated hypothesis**

## Question

Once MISDA began reporting latent/structural dimensions without access to benchmark truth, a separate question emerged: can the method flag results that are internally suspicious even when no external ground truth is available?

The target was not another dimension estimator. It was a confidence/reliability diagnostic capable of distinguishing ordinary supported cases from known pathological cases such as the transitive chain and regime-switching examples.

## Hypothesis considered

A Bayesian Information Criterion (BIC) derived from a correlation-model fit was proposed as a possible internal signal. The working conjecture was that a sign-based condition such as `BIC < 0` might identify cases in which the graph-derived dimensional description was inadequate.

The appeal was that BIC is data-derived and penalizes model complexity, apparently fitting the project's preference for reproducible diagnostics without an arbitrary user-selected threshold.

## What was not established

The conjecture was never validated strongly enough to become part of the method. In particular, no accepted evidence established that:

- the proposed BIC formulation had a unique, well-defined model comparison interpretation for MISDA;
- its zero point had the desired invariant meaning across the controlled cases;
- it separated pathological from regular cases without false positives;
- it provided information not already captured more directly by later diagnostics such as `TRANSITIVE_CHAINING` and `HIDDEN_SPECTRAL_STRUCTURE`.

No current ADR defines BIC as a MISDA support criterion, and the public support model does not depend on it.

## Outcome

The BIC idea remains historical conjecture only. It should not be reintroduced as an established reliability rule merely because it appeared promising in early discussion.

If revisited, the first task is theoretical: specify exactly which likelihood/model comparison BIC represents and why the relevant comparison has a data-independent interpretation. Only then should it be evaluated across the full clean/noisy multi-seed battery.

## Why keep this note

This is precisely the type of hypothesis that is easy to rediscover: it sounds principled, produces a scalar, and seems to offer a threshold-free quality signal. Recording that it was considered but never validated prevents a future implementation from silently promoting an old conjecture into a scientific guarantee.
