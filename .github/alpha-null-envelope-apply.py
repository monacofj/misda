from pathlib import Path
import re

statistics = Path("misda/_statistics.py")
text = statistics.read_text(encoding="utf-8")
text = text.replace("import warnings\n", "")
text = text.replace(
    '    """Sequential estimate of the expected maximum positive null correlation."""',
    '    """Permutation estimate of the positive null envelope and diagnostic metadata."""',
    1,
)

marker = "\n\ndef estimate_null_from_maxima(\n"
assert marker in text
envelope_code = '''


def _null_envelope_snapshot(
    samples,
    n_samples,
    signature: Callable[[float], Any],
    completed,
    reason=None,
) -> NullAlphaEstimate:
    """Build metadata for a fixed-budget empirical null envelope."""

    values = np.asarray(samples, dtype=float)
    count = len(values)
    if count:
        r_null = float(np.max(values))
        log_alpha_null = positive_correlation_log_p(r_null, n_samples)
        threshold_signature = signature(log_alpha_null)
        r_interval = (r_null, r_null)
        log_alpha_interval = (log_alpha_null, log_alpha_null)
    else:
        r_null = math.nan
        log_alpha_null = math.nan
        threshold_signature = None
        r_interval = (math.nan, math.nan)
        log_alpha_interval = (math.nan, math.nan)

    return NullAlphaEstimate(
        r_null=r_null,
        se_mc=math.nan,
        r_interval=r_interval,
        log_alpha_null=log_alpha_null,
        log_alpha_interval=log_alpha_interval,
        n_permutations=count,
        converged=bool(completed),
        lower_r_signature=threshold_signature,
        upper_r_signature=threshold_signature,
        samples=tuple(float(value) for value in values),
        reason=reason,
    )


def estimate_null_envelope_from_maxima(
    maxima: Iterable[float],
    *,
    n_samples: int,
    signature: Callable[[float], Any],
    cancel_requested: Optional[Callable[[int], bool]] = None,
) -> NullAlphaEstimate:
    """Estimate the empirical upper null envelope from exactly ``N`` maxima.

    Each null permutation contributes the maximum positive pairwise correlation.
    The estimator consumes exactly ``n_samples`` such maxima and uses their
    maximum as ``r_null``. The legacy Monte-Carlo standard-error metadata is
    therefore not applicable and is reported as ``NaN``.
    """

    if (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, Integral)
        or int(n_samples) < 4
    ):
        raise ValueError("n_samples must be an integer greater than or equal to 4.")
    if not callable(signature):
        raise TypeError("signature must be callable.")
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None.")

    samples = []
    for maximum in maxima:
        value = float(maximum)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("null maxima must be finite values in [0, 1].")
        samples.append(value)

        if len(samples) >= int(n_samples):
            return _null_envelope_snapshot(
                samples,
                int(n_samples),
                signature,
                completed=True,
            )
        if cancel_requested is not None and cancel_requested(len(samples)):
            return _null_envelope_snapshot(
                samples,
                int(n_samples),
                signature,
                completed=False,
                reason="CANCELLED",
            )

    return _null_envelope_snapshot(
        samples,
        int(n_samples),
        signature,
        completed=False,
        reason="SEQUENCE_EXHAUSTED",
    )
'''
text = text.replace(marker, envelope_code + marker, 1)

text = text.replace(
    '    """Estimate the expected maximum positive correlation under permutation."""',
    '    """Estimate the empirical upper envelope of positive null correlation."""',
    1,
)
old_public = '''    result = estimate_null_from_maxima(
        maxima(),
        n_samples=normalized.n_samples,
        signature=signature,
        cancel_requested=cancel_requested,
        max_permutations=10 * normalized.n_samples,
    )
    if not result.converged and result.reason == "MAX_PERMUTATIONS_REACHED":
        warnings.warn(
            "Structural alpha_null estimation did not converge by B_max=10N; "
            "returning the current null estimate with converged=False.",
            RuntimeWarning,
            stacklevel=2,
        )
'''
new_public = '''    result = estimate_null_envelope_from_maxima(
        maxima(),
        n_samples=normalized.n_samples,
        signature=signature,
        cancel_requested=cancel_requested,
    )
'''
assert old_public in text
statistics.write_text(text.replace(old_public, new_public, 1), encoding="utf-8")

positive_tests = Path("tests/test_positive_statistics.py")
text = positive_tests.read_text(encoding="utf-8")
old = '''    assert observed.r_null == 0.0
    assert observed.se_mc == 0.0
    assert observed.log_alpha_null == pytest.approx(math.log(0.5))
'''
new = '''    assert observed.r_null == 0.0
    assert np.isnan(observed.se_mc)
    assert observed.r_interval == (0.0, 0.0)
    assert observed.log_alpha_interval == pytest.approx(
        (math.log(0.5), math.log(0.5))
    )
    assert observed.log_alpha_null == pytest.approx(math.log(0.5))
'''
assert old in text
positive_tests.write_text(text.replace(old, new, 1), encoding="utf-8")

cap_tests = Path("tests/test_null_cap.py")
text = cap_tests.read_text(encoding="utf-8")
text = text.replace("import math\n", "import math\nimport warnings\n", 1)
pattern = re.compile(
    r"def test_public_null_estimator_warns_once_at_10n_cap\(\):\n.*?(?=\n\ndef test_cancellation_remains_distinct_from_autonomous_cap)",
    re.S,
)
replacement = '''def test_public_null_estimator_uses_fixed_n_envelope_without_cap_warning():
    rng = np.random.default_rng(91)
    normalized = _validation.normalize_input_matrix(rng.normal(size=(8, 3)))

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        observed = _statistics.estimate_null_positive_correlation(
            normalized,
            signature=lambda log_alpha: log_alpha,
            seed=123,
        )

    assert recorded == []
    assert observed.converged
    assert observed.n_permutations == normalized.n_samples
    assert observed.reason is None
    assert observed.r_null == max(observed.samples)
    assert np.isnan(observed.se_mc)
    assert observed.r_interval == pytest.approx((observed.r_null, observed.r_null))
'''
text, count = pattern.subn(replacement, text, count=1)
assert count == 1
cap_tests.write_text(text, encoding="utf-8")

Path("tests/test_null_envelope.py").write_text('''"""Contracts for the fixed-budget empirical alpha-null envelope."""

import math

import numpy as np
import pytest

from misda import _statistics, _validation


def test_controlled_envelope_uses_maximum_of_exactly_n_null_maxima():
    observed = _statistics.estimate_null_envelope_from_maxima(
        [0.10, 0.40, 0.30, 0.20, 0.90],
        n_samples=4,
        signature=lambda log_alpha: ("threshold", log_alpha),
    )

    assert observed.converged
    assert observed.reason is None
    assert observed.n_permutations == 4
    assert observed.samples == pytest.approx((0.10, 0.40, 0.30, 0.20))
    assert observed.r_null == pytest.approx(0.40)
    assert np.isnan(observed.se_mc)
    assert observed.r_interval == pytest.approx((0.40, 0.40))
    assert observed.log_alpha_null == pytest.approx(
        _statistics.positive_correlation_log_p(0.40, 4)
    )
    assert observed.log_alpha_interval == pytest.approx(
        (observed.log_alpha_null, observed.log_alpha_null)
    )
    assert observed.lower_r_signature == observed.upper_r_signature


def test_controlled_envelope_preserves_explicit_cancellation():
    observed = _statistics.estimate_null_envelope_from_maxima(
        [0.10, 0.30, 0.20, 0.90, 0.80],
        n_samples=5,
        signature=lambda log_alpha: log_alpha,
        cancel_requested=lambda count: count == 3,
    )

    assert not observed.converged
    assert observed.reason == "CANCELLED"
    assert observed.n_permutations == 3
    assert observed.r_null == pytest.approx(0.30)
    assert observed.samples == pytest.approx((0.10, 0.30, 0.20))


def test_public_envelope_is_reproducible_and_uses_exactly_n_permutations():
    rng = np.random.default_rng(91)
    normalized = _validation.normalize_input_matrix(rng.normal(size=(12, 4)))

    first = _statistics.estimate_null_positive_correlation(
        normalized,
        signature=lambda log_alpha: log_alpha,
        seed=123,
    )
    second = _statistics.estimate_null_positive_correlation(
        normalized,
        signature=lambda log_alpha: log_alpha,
        seed=123,
    )

    assert first == second
    assert first.converged
    assert first.reason is None
    assert first.n_permutations == normalized.n_samples
    assert len(first.samples) == normalized.n_samples
    assert first.r_null == max(first.samples)
    assert np.isnan(first.se_mc)
    assert first.r_interval == pytest.approx((first.r_null, first.r_null))
    assert first.log_alpha_interval == pytest.approx(
        (first.log_alpha_null, first.log_alpha_null)
    )
    assert first.seed == 123
    assert first.rng_state["bit_generator"] == "PCG64"


def test_all_constant_input_has_zero_empirical_envelope():
    normalized = _validation.normalize_input_matrix(np.ones((6, 3)))
    observed = _statistics.estimate_null_positive_correlation(
        normalized,
        signature=lambda log_alpha: log_alpha,
        seed=7,
    )

    assert observed.converged
    assert observed.n_permutations == 6
    assert observed.r_null == 0.0
    assert np.isnan(observed.se_mc)
    assert observed.r_interval == (0.0, 0.0)
    assert observed.log_alpha_null == pytest.approx(math.log(0.5))
''', encoding="utf-8")

adr10 = Path("docs/adr/0010-decision-stable-alpha-null.md")
text = adr10.read_text(encoding="utf-8")
assert "- Status: Accepted" in text
adr10.write_text(text.replace("- Status: Accepted", "- Status: Superseded by ADR 0015", 1), encoding="utf-8")

Path("docs/adr/0015-empirical-null-envelope-alpha-null.md").write_text('''# ADR 0015 — Empirical null envelope for alpha-null estimation

- Status: Accepted
- Recorded: 2026-09-15
- Supersedes: ADR 0010
- Tracking issue: #44

## Context

MISDA constructs its dependence graphs by comparing observed correlations with a permutation-derived null reference. ADR 0010 defined that reference from the **mean of the maximum positive correlations** observed across null permutations, together with a sequential `mean ± MC-SE` stopping rule.

Repeated clean runs of the fully independent benchmark exposed a weakness in that design. An independent observed sample is itself another draw from the same no-dependence regime represented by the permutations. Its largest accidental correlation can therefore exceed the *mean* null maximum with substantial frequency. A single such exceedance creates a spurious graph edge and can reduce the graph independence number even though no dependence exists.

The methodological question is therefore not the typical size of the largest null correlation, but whether an observed correlation exceeds the upper empirical range produced by the null experiment.

## Decision

For a data matrix with `N` observations, public `alpha_null` estimation uses exactly `B=N` independent-column permutations. If `m_b` is the maximum **positive** pairwise correlation produced by null permutation `b`, define

```text
r_null = max(m_1, ..., m_N).
```

`alpha_null` is then obtained by applying the same one-tailed Fisher-z probability mapping used by discovery:

```text
log_alpha_null = positive_correlation_log_p(r_null, N).
```

The same threshold remains shared by `G+` and `G±`; negative observed correlations may enter `G±`, but a second absolute-correlation null calibration is not introduced.

The sequential `mean ± MC-SE` convergence rule and its `10N` cap are no longer part of the public estimator.

## Rationale

The old estimator asked whether an observed correlation exceeded the **typical null maximum**. For graph construction this was too permissive: values that are uncommon but still routinely attainable under independence could become edges.

The empirical envelope asks the stricter question: did the observed correlation exceed every maximum produced by the fixed null experiment? This directly targets the false-edge failure mode seen in the independence case while remaining entirely data-derived and hyperparameter-free.

The fixed budget `B=N` is itself derived from the dataset size rather than supplied by the user. Experimental validation found no dimensional changes when the envelope budget was increased from `N` to `2N`, `5N`, or `10N` in the tested battery.

## Experimental basis

Before adoption, the candidate rule was evaluated on the controlled benchmark suite:

- 50 clean replicates of `independence`: exact latent and structural dimension recovery in 100% of runs;
- 11 regular clean cases × 20 replicates: exact latent and structural dimension recovery in 100% of runs;
- regular robustness cases across `sigma ∈ {0, 0.05, 0.10, 0.20, 0.40}`: exact latent and structural recovery in all tested replicates;
- the known adversarial `transitive_chain` and `regime_switching` cases remained adversarial rather than being hidden by the new threshold;
- shared positive-null calibration and separate positive/absolute null calibration produced the same dimensional conclusions in the tested battery.

These experiments motivate the decision but do not redefine benchmark truth.

## Metadata compatibility

`NullAlphaEstimate` and the public analysis object retain legacy fields to avoid an unnecessary API break.

Under the envelope estimator:

- `n_permutations` is exactly `N` after normal completion;
- `converged=True` means that the fixed-budget envelope was completed, not that a Monte Carlo mean converged;
- `reason=None` after normal completion;
- `se_mc=NaN` because a standard error of a Monte Carlo mean is not defined for this estimator;
- `r_interval=(r_null, r_null)` and `log_alpha_interval=(log_alpha_null, log_alpha_null)` are degenerate compatibility fields, not uncertainty intervals;
- `lower_r_signature` and `upper_r_signature` are equal to the signature at the selected envelope threshold;
- explicit cancellation remains represented by `converged=False` and `reason="CANCELLED"`.

A future API revision may rename or remove these compatibility fields, but they must not be given the old ADR 0010 interpretation.

## Invariants

- objective columns are permuted independently;
- every null replicate contributes the maximum positive pairwise correlation;
- normal public estimation uses exactly `N` null replicates;
- `r_null` is the maximum of those null maxima;
- the Fisher-z mapping remains the probability transformation used by discovery;
- the null threshold is shared by `G+` and `G±`;
- the estimator remains reproducible from the public seed/RNG contract;
- no user-tunable percentile, confidence level, permutation count, or numerical tolerance is introduced.

## Forbidden shortcuts / regression risks

Do not replace the envelope by a fixed percentile, a hard-coded correlation threshold, a user-selected `B`, the old mean-of-maxima statistic, or a separate signed threshold without a new methodological decision.

Do not interpret the retained `se_mc` or interval fields as uncertainty estimates for the envelope.

## Verification

Tests must verify that public estimation consumes exactly `N` permutations, that `r_null == max(samples)`, that repeated runs with the same seed are identical, that constant inputs remain well-defined, and that the obsolete `10N` non-convergence warning is absent.

The clean diagnostic battery and regular benchmark cases must not regress when this decision is implemented.
''', encoding="utf-8")

readme = Path("docs/adr/README.md")
text = readme.read_text(encoding="utf-8")
anchor = "- [0014 — Public diagnostics and reporting semantics](0014-public-diagnostics-and-reporting-semantics.md)\n"
assert anchor in text
text = text.replace(anchor, anchor + "- [0015 — Empirical null envelope for alpha-null estimation](0015-empirical-null-envelope-alpha-null.md)\n", 1)
text = text.replace(
    "They document the current MISDA design as of 2026-09-11.",
    "They document the current MISDA design, with later records superseding earlier decisions where explicitly stated.",
    1,
)
readme.write_text(text, encoding="utf-8")
