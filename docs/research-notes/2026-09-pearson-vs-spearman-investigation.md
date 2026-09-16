# Pearson versus Spearman in MISDA: laboratory notebook of the 2026-09 investigation

- Period: September 2026
- Status: **Experimental investigation completed; no default change adopted**
- Branch: `spearman`
- Production baseline: `main`
- Main question: should MISDA continue to use signed Pearson correlation in discovery, or would signed Spearman correlation be a better structural dependence coefficient?
- Current outcome: **keep Pearson as the production default for now; Spearman is scientifically informative as a complementary monotonic-sensitivity diagnostic, but the experiments do not support replacing Pearson.**

> This note is intentionally written as a laboratory notebook rather than as a compact research summary. It records the sequence of questions, conjectures, experimental controls, corrections to the experimental truth, observed results, intermediate interpretations, and the final conclusion reached from this investigation. It is non-normative. The production method remains defined by the code and accepted ADRs on `main`.

## 1. Why this question was reopened

MISDA had been using signed Pearson correlation as its basic pairwise dependence statistic. This was not merely an accidental implementation detail. The earlier rationale for Pearson was that the structural graph was being interpreted in terms of quantitative, metric association between objectives: strong positive linear association indicated redundancy, while strong negative association indicated an important conflict rather than removable redundancy.

That rationale also aligned naturally with the surrounding diagnostics. A linear reconstruction diagnostic asks whether one objective can be approximately reconstructed from another through a linear model. Pearson measures the same kind of linear association. Under that interpretation, Spearman could seem too permissive: it maps any exact strictly monotonic relation to magnitude 1 after replacing values by their ranks, even when the metric geometry of the relation is highly nonlinear.

A different conceptual question later reopened the choice. In multiobjective minimization, a strictly increasing transformation of an objective preserves the ordering of candidate solutions along that objective and therefore preserves dominance relations. If

\[
f_2 = g(f_1)
\]

with strictly increasing \(g\), then the two objectives may be structurally redundant from an order/Pareto perspective even when the numerical relation is far from linear. An example is

\[
f_2 = \exp(f_1).
\]

Spearman correlation is exactly sensitive to this order-preserving monotonic dependence. Pearson is not invariant to such transformations.

This produced the central scientific tension of the investigation:

- **Pearson hypothesis:** MISDA should measure strong metric/linear association because this is more robust and corresponds to the quantitative geometry actually observed in objective space.
- **Spearman hypothesis:** MISDA should measure monotonic order dependence because order is closer to the semantics of Pareto dominance and because strictly increasing transformations do not change the ordering information relevant to minimization.

The correct question was therefore not initially framed as “which coefficient is larger or more modern?” It was:

> What notion of redundancy should MISDA recognize, and what happens empirically if the discovery coefficient is changed from Pearson to Spearman while the rest of the method is held fixed?

## 2. Statistical definitions relevant to the experiment

### 2.1 Pearson correlation

For two observed objective vectors \(x\) and \(y\), Pearson correlation is

\[
r_P =
\frac{\sum_i (x_i-\bar x)(y_i-\bar y)}
{\sqrt{\sum_i(x_i-\bar x)^2}\sqrt{\sum_i(y_i-\bar y)^2}}.
\]

It is exactly \(+1\) for an exact positive affine relation and \(-1\) for an exact negative affine relation. It can be much smaller in magnitude for exact nonlinear dependence.

### 2.2 Spearman correlation

Spearman correlation is Pearson correlation applied to the ranks:

\[
r_S = r_P(R(x),R(y)).
\]

When there are no ties, it can also be written as

\[
r_S =
1-\frac{6\sum_i d_i^2}{n(n^2-1)},
\]

where \(d_i\) is the difference between the two ranks for observation \(i\).

Therefore:

- any exact strictly increasing relation gives \(r_S=+1\);
- any exact strictly decreasing relation gives \(r_S=-1\);
- a deterministic but globally non-monotonic relation need not yield a large \(|r_S|\).

The experimental implementation uses average ranks for ties.

### 2.3 Signed interpretation in MISDA

The sign remains essential. The experiment did **not** switch MISDA to absolute correlation.

Positive dependence contributes to the positive structural graph \(G^+\), where edges represent candidate positive redundancy. Negative dependence remains part of the signed/dependence graph \(G^\pm\) but does not become a positive redundancy edge. Thus an exact decreasing monotonic relation is expected to be latent dependence without collapsing the positive structural dimension in the same way as an increasing monotonic relation.

### 2.4 Calibration was not held at the Pearson threshold

A critical experimental requirement was identified before changing the coefficient: it would be invalid to compute Spearman correlations and then apply a null threshold calibrated from Pearson permutations.

The branch therefore recalibrates the permutation null with the same coefficient used in the observed data. In the experimental Spearman implementation, each column is converted to centered, unit-norm average ranks; observed signed correlations are dot products of these standardized ranks. During null calibration, each rank column is independently permuted and the maximum positive Spearman correlation is fed into the same fixed-budget null-envelope machinery used by production MISDA.

The experiment deliberately preserves MISDA's existing monotone mapping from correlation magnitude to log-alpha through the Fisher-z based function. This is important to interpret correctly. The branch does **not** claim that the usual Pearson Fisher-z parametric significance theory is an exact sampling theory for Spearman. The mapping is retained as a common monotone scale while the empirical null is consistently recalibrated with Spearman. This was done to isolate the effect of changing the dependence coefficient as closely as possible.

## 3. Experimental isolation: the `spearman` branch

The investigation was performed on a dedicated branch named `spearman`; `main` was kept as the Pearson production baseline.

The experiment intentionally avoided introducing a public correlation-choice parameter at this stage. Instead, the public `discover()` pipeline on the branch is routed to experimental Spearman implementations of the two discovery-stage statistical operations:

1. observed correlation statistics;
2. positive-correlation permutation-null estimation.

Everything downstream is left unchanged: graph construction semantics, aggressiveness interpolation, MIS enumeration, candidate ranking, dimensional estimators, reporting, and benchmark truth.

This makes the comparison an A/B experiment:

\[
\text{main} = \text{Pearson},
\qquad
\text{spearman branch} = \text{Spearman}.
\]

The branch comparison shows that the experimental work is additive: the production Pearson code on `main` remains untouched, while the branch adds `_statistics_spearman.py`, benchmark runners, a notebook, and an experimental workflow.

## 4. First control: existing `controlled` and `controlled_noisy`

### 4.1 Question

Before constructing cases favorable to Spearman, the first test was deliberately conservative:

> Does simply replacing Pearson by Spearman alter the already established controlled behavior of MISDA?

The same sample size, seeds, discovery pipeline, thresholds, aggressiveness, graph semantics, ranking, and benchmark truth were retained. Only the dependence coefficient and its corresponding permutation null were changed.

The runs used the existing controlled suites with \(N=300\), seed `123`; the noisy controlled experiment used the same established observation-noise configuration, including `observation_seed=456` and `sigma=0.1`.

### 4.2 Result

The serialized outputs of Pearson and Spearman were recursively compared. For both `controlled` and `controlled_noisy`, the comparison yielded **zero differences** in the inspected serialized results.

This was stronger than equality of the final dimensional estimates. The observed equivalence covered the graph structures, number of edges, MIS universe size, selected candidate, separation status, declared-truth status, linear evidence, and Pareto-related serialized evidence.

The dimensional outcomes were:

| Caso | Pearson d_l / d_s | Spearman d_l / d_s |
| --- | --- | --- |
| Independentes | 20 / 20 | 20 / 20 |
| Redundância completa | 1 / 1 | 1 / 1 |
| 4 blocos | 4 / 4 | 4 / 4 |
| 2 blocos | 2 / 2 | 2 / 2 |
| Mistura | 12 / 12 | 12 / 12 |
| Antagônicos lineares | 1 / 2 | 1 / 2 |
| MOP-A, monotônico não linear | 1 / 1 | 1 / 1 |
| MOP-B | 2 / 2 | 2 / 2 |
| MOP-C | 4 / 4 | 4 / 4 |
| MOP-D | 1 / 2 | 1 / 2 |
| MOP-E | 2 / 2 | 2 / 2 |
| Cadeia | 1 / 1 | 1 / 1 |
| MOP-F | 1 / 1 | 1 / 1 |

### 4.3 Interpretation

This was a useful negative result. It showed that the existing canonical controlled suites did **not** discriminate Pearson from Spearman.

That did not establish that the coefficients were equivalent. It established something narrower and more important for experiment design: in those cases, any numerical differences between Pearson and Spearman did not cross the data-driven thresholds in a way that changed the inferred structure.

Therefore a new benchmark had to be designed specifically to separate the two hypotheses.

## 5. Construction of a discriminating `monotonic` benchmark

### 5.1 Design principle

The new benchmark was named `monotonic`, rather than `pearson_vs_spearman`, because a benchmark should describe the scientific property under study rather than the current methods being compared.

The benchmark was designed around controlled relations for which the latent and structural truth are known. It includes:

- an exact positive linear control;
- a mild positive monotonic nonlinear relation;
- an extremely nonlinear positive monotonic relation;
- the corresponding negative monotonic relation;
- a positive monotonic redundant pair embedded with an independent objective;
- a deterministic globally non-monotonic dependence;
- an independence control.

The baseline parameters are:

- \(N=300\);
- seed `123`;
- steep exponential parameter \(k=650\).

The extreme exponential is written as

\[
y = \exp(650(x-1)), \qquad x\in[0,1],
\]

instead of \(\exp(650x)\). Multiplication by the constant \(\exp(-650)\) does not change either Pearson or Spearman correlation, while the form above keeps the values representable and bounded in \((0,1]\).

### 5.2 The seven cases and their semantic truth

| Caso | Construção / interpretação | d_l esperado | d_s esperado | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- |
| linear_positive_control | Exact positive linear redundancy: y = 2x. | 1 | 1 | 1.0000 | 1.0000 |
| mild_monotonic_control | Smooth monotonic nonlinear redundancy: y = x^3 on [0,1]. | 1 | 1 | 0.9162 | 1.0000 |
| steep_positive_monotonic | Strictly increasing but strongly nonlinear redundancy: y = exp(650(x-1)). | 1 | 1 | 0.1119 | 1.0000 |
| steep_negative_monotonic | Strictly decreasing strong nonlinear dependence: y = -exp(650(x-1)). It is latent dependence but not positive redundancy. | 1 | 2 | -0.1119 | -1.0000 |
| steep_positive_plus_independent | One steep positive monotonic redundant pair plus one independent objective. | 2 | 2 | 0.1119 | 1.0000 |
| nonmonotonic_quadratic_limit | Deterministic but globally non-monotonic latent dependence: y = x^2 on [-1,1]. Latent truth is one dimension, but there is no global positive monotonic redundancy, so structural truth is two. | 1 | 2 | -0.0000 | -0.0016 |
| independence_control | Independent Gaussian objectives. | 2 | 2 | 0.0268 | 0.0441 |

### 5.3 Correction made during the experiment: the quadratic truth

An important correction occurred while constructing the benchmark.

The case

\[
y=x^2,\qquad x\in[-1,1]
\]

was initially at risk of being treated as if failure to find a monotonic relation implied two latent dimensions. That would have been conceptually wrong.

The correct distinction is:

- latent dimension: \(d_l=1\), because \(y\) is deterministically reconstructed from \(x\);
- structural positive-monotonic dimension: \(d_s=2\), because the global relation is not monotonic and therefore does not define a global positive redundancy edge of the kind tested by either Pearson or Spearman in this experiment.

The benchmark truth was corrected accordingly. This case was renamed `nonmonotonic_quadratic_limit` to make clear that it is a deliberate limit case for the tested dependence families, not a positive control.

That correction matters because it prevents the benchmark from “rewarding” either coefficient for failing to detect genuine nonlinear latent dependence.

## 6. Clean discriminating benchmark: Pearson versus Spearman

The clean `monotonic` benchmark produced 5/7 exact cases for Pearson and 6/7 for Spearman.

The detailed discovery outcomes were:

| Caso | P d_l/d_s | P exato | P separação | P α_onset | P α_null | S d_l/d_s | S exato | S separação | S α_onset | S α_null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear_positive_control | 1/1 | yes | NULL_SEPARATION | 0 | 0.00478 | 1/1 | yes | NULL_SEPARATION | 0 | 0.00478 |
| mild_monotonic_control | 1/1 | yes | NULL_SEPARATION | 1.879e-160 | 0.01095 | 1/1 | yes | NULL_SEPARATION | 0 | 0.00478 |
| steep_positive_monotonic | 1/1 | yes | NULL_SEPARATION | 0.02644 | 0.03322 | 1/1 | yes | NULL_SEPARATION | 0 | 0.00478 |
| steep_negative_monotonic | 1/2 | yes | NO_NULL_SEPARATION | — | 0.03272 | 1/2 | yes | NO_NULL_SEPARATION | — | 0.00153 |
| steep_positive_plus_independent | 3/3 | no | NO_NULL_SEPARATION | 0.02644 | 0.00515 | 2/2 | yes | NULL_SEPARATION | 0 | 0.00490 |
| nonmonotonic_quadratic_limit | 2/2 | no | NO_NULL_SEPARATION | — | 0.00044 | 2/2 | no | NO_NULL_SEPARATION | — | 0.00092 |
| independence_control | 2/2 | yes | NO_NULL_SEPARATION | 0.32177 | 3.237e-05 | 2/2 | yes | NO_NULL_SEPARATION | 0.22349 | 9.861e-06 |

Here `NULL_SEPARATION` means the observed positive onset is separated from the permutation-null reference; `NO_NULL_SEPARATION` means it is not.

### 6.1 Cases in which both methods agree

The linear positive control is trivial for both methods: \(r_P=r_S=1\), and both infer \(d_l=d_s=1\).

For \(y=x^3\) on \([0,1]\), Pearson is already strong (\(r_P\approx0.9162\)), while Spearman is exactly 1. Both are strong enough relative to their own data-driven null envelopes, and both recover \(1/1\).

For the isolated extreme exponential pair, Pearson correlation falls to approximately

\[
r_P \approx 0.111866,
\]

whereas Spearman remains

\[
r_S = 1.
\]

Nevertheless, with only the two objectives present, Pearson still crosses its own calibrated threshold and recovers \(1/1\). This was a first warning against drawing conclusions from the correlation coefficient alone: what matters in MISDA is the coefficient **relative to the empirically calibrated multivariate null threshold**.

For the negative extreme monotonic case,

\[
y=-\exp(650(x-1)),
\]

both methods recover the intended distinction \(d_l=1,d_s=2\). The signed graph recognizes dependence, but \(G^+\) contains no positive redundancy edge.

The independent Gaussian control is also correctly left disconnected by both methods.

### 6.2 The first genuinely discriminating case

The key clean case is `steep_positive_plus_independent`:

\[
Y = \left[x,\exp(650(x-1)),z\right],
\]

with \(z\) an independent Gaussian objective.

The monotonic pair is semantically redundant, and the third objective is independent, so the truth is

\[
d_l=d_s=2.
\]

Pearson still sees only \(r_P\approx0.111866\) for the monotonic pair. In this three-objective context, however, the permutation-null envelope is different. The observed Pearson onset is approximately

\[
\alpha_{\text{onset}} \approx 0.0264358,
\]

while the null reference is

\[
\alpha_{\text{null}} \approx 0.00515276.
\]

The required separation is absent. No dependence edge is accepted, and Pearson returns

\[
d_l=d_s=3.
\]

Spearman sees the pair as exactly monotonic:

\[
r_S=1.
\]

Its onset is effectively zero on the alpha scale, and its Spearman-calibrated null reference is approximately

\[
\alpha_{\text{null}} \approx 0.00490444.
\]

The pair is connected, and Spearman returns the correct

\[
d_l=d_s=2.
\]

This is the clean result that first supplied genuine empirical evidence in favor of Spearman: there exists a known positive monotonic redundancy that Pearson misses in a multivariate context while Spearman recovers it.

### 6.3 The shared non-monotonic limit

For \(y=x^2\) on \([-1,1]\), Pearson is essentially zero and Spearman is also essentially zero:

\[
r_P\approx -2.0\times10^{-16},
\qquad
r_S\approx -0.00159.
\]

Both methods return \(d_l=2,d_s=2\). Structural dimension is correct; latent dimension is not. This confirms a precise limitation:

> Spearman extends sensitivity from linear relations to monotonic relations. It does not solve general nonlinear dependence.

At this point the clean-data interpretation was:

> Spearman appears to correct a real Pearson blind spot for strong monotonic nonlinear redundancy, without changing the established controlled benchmarks, but this is not yet enough to justify replacing the default. Robustness must be tested.

## 7. Robustness experiment: resampling and observation noise

A second runner, `run_monotonic_robustness.py`, was added to distinguish two different perturbation questions.

### 7.1 Bootstrap resampling

For each clean case, ten bootstrap samples of rows were drawn with replacement using seeds `2001` through `2010`.

This tests sensitivity to the empirical sample: the semantic relation is preserved, but the realized leverage, density, and empirical distribution change.

MISDA's own discovery seed was held fixed at `123` so that observed variation would be attributable to the resampled data rather than to a changing internal randomization.

### 7.2 Observation noise

The same clean cases were also perturbed with additive Gaussian noise:

\[
Y_{\sigma}
=
Y + \sigma\, s(Y)\odot E,
\]

where \(s(Y)\) is the per-objective sample standard deviation and \(E\) is a standardized Gaussian noise matrix.

Five noise realizations were used, with seeds `3001` through `3005`, and

\[
\sigma\in\{0,0.01,0.02,0.05,0.10\}.
\]

Common random numbers were used across the sigma levels within each replicate. That is, increasing \(\sigma\) scales the same underlying noise field instead of drawing a new field. This makes deterioration along the sigma axis interpretable as an effect of noise magnitude rather than a side effect of unrelated random draws.

### 7.3 Bootstrap result

| Caso | P exato | S exato | P d_l | S d_l | P d_s | S d_s | P moda d_l/d_s | S moda d_l/d_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear_positive_control | 100% | 100% | 100% | 100% | 100% | 100% | 1/1 | 1/1 |
| mild_monotonic_control | 100% | 100% | 100% | 100% | 100% | 100% | 1/1 | 1/1 |
| steep_positive_monotonic | 100% | 100% | 100% | 100% | 100% | 100% | 1/1 | 1/1 |
| steep_negative_monotonic | 70% | 100% | 70% | 100% | 100% | 100% | 1/2 | 1/2 |
| steep_positive_plus_independent | 10% | 100% | 10% | 100% | 10% | 100% | 3/3 | 2/2 |
| nonmonotonic_quadratic_limit | 0% | 10% | 0% | 10% | 100% | 100% | 2/2 | 2/2 |
| independence_control | 100% | 100% | 100% | 100% | 100% | 100% | 2/2 | 2/2 |

The important pattern is immediate.

For the difficult positive monotonic pair embedded with an independent objective, Pearson exact recovery is only **10%**, whereas Spearman exact recovery is **100%**.

The negative extreme monotonic case also improves under Spearman, from **70%** exact recovery with Pearson to **100%** with Spearman.

Thus the clean Spearman advantage is not a one-seed artifact. Under bootstrap resampling, Spearman is substantially more stable in the exact monotonic regime that was designed to expose Pearson's lack of monotonic invariance.

The quadratic case remains a limit case. A small fraction of bootstrap samples can incidentally cause the inferred dimensions to coincide with the declared truth, but this is not evidence that either coefficient has learned global non-monotonic dependence. The edge-level behavior confirms that interpretation.

### 7.4 Observation-noise result

The full robustness table is reproduced below. Rates are over five noise realizations for each \(\sigma\).

| Caso | σ | P exato | S exato | P d_l | S d_l | P d_s | S d_s | P G+ | S G+ | P G± | S G± |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear_positive_control | 0.00 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| linear_positive_control | 0.01 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| linear_positive_control | 0.02 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| linear_positive_control | 0.05 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| linear_positive_control | 0.10 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| mild_monotonic_control | 0.00 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| mild_monotonic_control | 0.01 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| mild_monotonic_control | 0.02 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| mild_monotonic_control | 0.05 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| mild_monotonic_control | 0.10 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| steep_positive_monotonic | 0.00 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| steep_positive_monotonic | 0.01 | 100% | 0% | 100% | 0% | 100% | 0% | 100% | 0% | 100% | 0% |
| steep_positive_monotonic | 0.02 | 100% | 0% | 100% | 0% | 100% | 0% | 100% | 0% | 100% | 0% |
| steep_positive_monotonic | 0.05 | 40% | 0% | 40% | 0% | 40% | 0% | 40% | 0% | 40% | 0% |
| steep_positive_monotonic | 0.10 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| steep_negative_monotonic | 0.00 | 100% | 100% | 100% | 100% | 100% | 100% | 0% | 0% | 100% | 100% |
| steep_negative_monotonic | 0.01 | 100% | 0% | 100% | 0% | 100% | 100% | 0% | 0% | 100% | 0% |
| steep_negative_monotonic | 0.02 | 100% | 0% | 100% | 0% | 100% | 100% | 0% | 0% | 100% | 0% |
| steep_negative_monotonic | 0.05 | 40% | 0% | 40% | 0% | 100% | 100% | 0% | 0% | 40% | 0% |
| steep_negative_monotonic | 0.10 | 20% | 0% | 20% | 0% | 100% | 100% | 0% | 0% | 20% | 0% |
| steep_positive_plus_independent | 0.00 | 0% | 100% | 0% | 100% | 0% | 100% | 0% | 100% | 0% | 100% |
| steep_positive_plus_independent | 0.01 | 0% | 20% | 0% | 20% | 0% | 20% | 0% | 20% | 0% | 20% |
| steep_positive_plus_independent | 0.02 | 0% | 20% | 0% | 20% | 0% | 20% | 0% | 20% | 0% | 20% |
| steep_positive_plus_independent | 0.05 | 0% | 20% | 0% | 20% | 0% | 20% | 0% | 20% | 0% | 20% |
| steep_positive_plus_independent | 0.10 | 0% | 20% | 0% | 20% | 0% | 20% | 0% | 20% | 0% | 20% |
| nonmonotonic_quadratic_limit | 0.00 | 0% | 0% | 0% | 0% | 100% | 100% | 0% | 0% | 0% | 0% |
| nonmonotonic_quadratic_limit | 0.01 | 0% | 0% | 0% | 0% | 100% | 100% | 0% | 0% | 0% | 0% |
| nonmonotonic_quadratic_limit | 0.02 | 0% | 0% | 0% | 0% | 100% | 100% | 0% | 0% | 0% | 0% |
| nonmonotonic_quadratic_limit | 0.05 | 0% | 0% | 0% | 0% | 100% | 100% | 0% | 0% | 0% | 0% |
| nonmonotonic_quadratic_limit | 0.10 | 0% | 0% | 0% | 0% | 100% | 100% | 0% | 0% | 0% | 0% |
| independence_control | 0.00 | 100% | 100% | 100% | 100% | 100% | 100% | 0% | 0% | 0% | 0% |
| independence_control | 0.01 | 100% | 100% | 100% | 100% | 100% | 100% | 0% | 0% | 0% | 0% |
| independence_control | 0.02 | 100% | 100% | 100% | 100% | 100% | 100% | 0% | 0% | 0% | 0% |
| independence_control | 0.05 | 100% | 100% | 100% | 100% | 100% | 100% | 0% | 0% | 0% | 0% |
| independence_control | 0.10 | 100% | 100% | 100% | 100% | 100% | 100% | 0% | 0% | 0% | 0% |

### 7.5 Unexpected result: the Spearman advantage reverses under noise

The observation-noise results refuted the simple conjecture that “Spearman is strictly better because it recognizes monotonic structure.”

For the extreme positive exponential pair, Spearman is perfect at \(\sigma=0\) but collapses immediately at \(\sigma=0.01\). Pearson, by contrast, remains exact at low noise and degrades later.

The same phenomenon appears in the negative extreme monotonic case. Spearman preserves the structural dimension \(d_s=2\), because no positive edge should exist, but loses the signed dependence needed for \(d_l=1\). Pearson remains much more robust at the smaller noise levels.

For the `steep_positive_plus_independent` case, the comparison is more nuanced:

- at \(\sigma=0\), Pearson fails and Spearman succeeds;
- at positive noise, Pearson remains failed for this extreme \(k=650\) multivariate case;
- Spearman retains only 20% recovery at the tested positive noise levels.

This is not a general win for either coefficient. It reveals two different failure modes.

### 7.6 Why an extremely steep monotonic relation is fragile for ranks

The family

\[
y=\exp(k(x-1))
\]

becomes extremely compressed near zero for large \(k\). For \(k=650\), most observations have very small \(y\) values, with only a narrow portion near \(x=1\) occupying a substantial numeric range.

Spearman relies entirely on the ordering of observations. Even observation noise that is small relative to the global standard deviation of the objective can reorder many of the densely compressed values. Once ranks are shuffled, Spearman's perfect clean monotonicity can disappear rapidly.

Pearson behaves differently. In this particular geometry, the high-amplitude tail has substantial leverage on covariance. That can make Pearson more resistant to the same standard-deviation-scaled perturbation, even though its clean correlation is much smaller.

This produced a new working hypothesis:

> The apparent Spearman fragility may be concentrated in extremely steep monotonic transformations, not in ordinary monotonic nonlinearity. We therefore need to vary the nonlinearity continuously rather than reason from the single \(k=650\) case.

## 8. Nonlinearity-by-noise sweep

The next experiment introduced the family

\[
y=\exp(k(x-1)),\qquad x\in[0,1],
\]

with

\[
k\in\{1,2,5,10,20,50,100,200,400,650\}.
\]

Two contexts were tested:

1. `pair`: only \((x,y)\), with expected \(d_l=d_s=1\);
2. `plus_independent`: \((x,y,z)\), where \(z\) is independent, with expected \(d_l=d_s=2\).

Noise levels were refined to

\[
\sigma\in\{0,0.001,0.002,0.005,0.01,0.02,0.05,0.10\},
\]

with ten noise realizations (`4001` through `4010`) for every \(k,\sigma,\) and context.

Again, noise was scaled by each objective's sample standard deviation. Common random numbers were used across both \(k\) and \(\sigma\) within each context and replicate. The discovery seed remained fixed.

This design directly tests where, in the two-dimensional plane of nonlinearity and observation noise, each coefficient succeeds or fails.

### 8.1 Clean correlations along the \(k\) axis

| k | Pearson clean | Spearman clean |
| --- | --- | --- |
| 1 | 0.9918 | 1.0000 |
| 2 | 0.9689 | 1.0000 |
| 5 | 0.8574 | 1.0000 |
| 10 | 0.6920 | 1.0000 |
| 20 | 0.5189 | 1.0000 |
| 50 | 0.3392 | 1.0000 |
| 100 | 0.2432 | 1.0000 |
| 200 | 0.1752 | 1.0000 |
| 400 | 0.1305 | 1.0000 |
| 650 | 0.1119 | 1.0000 |

Spearman is exactly 1 for every clean member of this strictly increasing family. Pearson decays smoothly from almost linear at \(k=1\) to approximately 0.1119 at \(k=650\).

This confirms that \(k\) is a useful controlled axis for moving from nearly linear dependence to an extreme monotonic transformation without changing the semantic ordering relation.

### 8.2 Exact recovery matrices: pair context

Each cell below is the proportion of ten noise replicates with exact recovery of both \(d_l\) and \(d_s\).

#### Pearson, pair only

| k | σ=0 | σ=0.001 | σ=0.002 | σ=0.005 | σ=0.01 | σ=0.02 | σ=0.05 | σ=0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 2 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 5 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 10 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 20 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 50 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 100 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 200 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 400 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 650 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 60% |

#### Spearman, pair only

| k | σ=0 | σ=0.001 | σ=0.002 | σ=0.005 | σ=0.01 | σ=0.02 | σ=0.05 | σ=0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 2 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 5 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 10 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 20 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 50 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 100 | 100% | 100% | 100% | 90% | 90% | 90% | 70% | 70% |
| 200 | 100% | 70% | 60% | 40% | 20% | 20% | 20% | 20% |
| 400 | 100% | 20% | 20% | 20% | 10% | 10% | 10% | 10% |
| 650 | 100% | 10% | 10% | 10% | 10% | 10% | 10% | 10% |

In the two-objective context, Pearson is extraordinarily robust over the tested grid: it remains at 100% exact recovery through \(k=400,\sigma=0.10\), and only drops to 60% at \(k=650,\sigma=0.10\).

Spearman matches Pearson through \(k=50\). At \(k=100\), it begins to degrade at larger noise levels. At \(k=200\) and beyond, the degradation becomes abrupt even for very small positive noise.

### 8.3 Exact recovery matrices: monotonic pair plus independent objective

#### Pearson, plus independent

| k | σ=0 | σ=0.001 | σ=0.002 | σ=0.005 | σ=0.01 | σ=0.02 | σ=0.05 | σ=0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 2 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 5 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 10 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 20 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 50 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 100 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 200 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 90% |
| 400 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 650 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

#### Spearman, plus independent

| k | σ=0 | σ=0.001 | σ=0.002 | σ=0.005 | σ=0.01 | σ=0.02 | σ=0.05 | σ=0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 2 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 5 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 10 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 20 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 50 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 100 | 100% | 100% | 90% | 90% | 60% | 60% | 50% | 20% |
| 200 | 100% | 40% | 20% | 10% | 10% | 10% | 0% | 0% |
| 400 | 100% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 650 | 100% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

The additional independent objective exposes the multivariate threshold effect that motivated the benchmark.

Pearson remains perfectly robust through \(k=100\), and nearly perfect through \(k=200\). But at \(k=400\) and \(k=650\), it fails even at \(\sigma=0\): the clean Pearson association has become too weak relative to the multivariate null envelope.

Spearman shows the complementary behavior. At \(\sigma=0\), it remains exact for every \(k\), including \(400\) and \(650\), because strict monotonicity is preserved exactly in ranks. But once positive noise is introduced, its recovery collapses much sooner than Pearson as \(k\) becomes large.

### 8.4 80% recovery tolerance summary

For each \(k\) and context, the experiment also reports the largest tested \(\sigma\) at which exact recovery is at least 80%.

| Contexto | k | Pearson: maior σ com ≥80% | Spearman: maior σ com ≥80% |
| --- | --- | --- | --- |
| pair | 1 | ≥ 0.100 | ≥ 0.100 |
| pair | 2 | ≥ 0.100 | ≥ 0.100 |
| pair | 5 | ≥ 0.100 | ≥ 0.100 |
| pair | 10 | ≥ 0.100 | ≥ 0.100 |
| pair | 20 | ≥ 0.100 | ≥ 0.100 |
| pair | 50 | ≥ 0.100 | ≥ 0.100 |
| pair | 100 | ≥ 0.100 | 0.020 |
| pair | 200 | ≥ 0.100 | 0 only |
| pair | 400 | ≥ 0.100 | 0 only |
| pair | 650 | 0.050 | 0 only |
| plus_independent | 1 | ≥ 0.100 | ≥ 0.100 |
| plus_independent | 2 | ≥ 0.100 | ≥ 0.100 |
| plus_independent | 5 | ≥ 0.100 | ≥ 0.100 |
| plus_independent | 10 | ≥ 0.100 | ≥ 0.100 |
| plus_independent | 20 | ≥ 0.100 | ≥ 0.100 |
| plus_independent | 50 | ≥ 0.100 | ≥ 0.100 |
| plus_independent | 100 | ≥ 0.100 | 0.005 |
| plus_independent | 200 | ≥ 0.100 | 0 only |
| plus_independent | 400 | none | 0 only |
| plus_independent | 650 | none | 0 only |

The decisive crossover is now clear.

For \(k\le 50\), the choice of Pearson versus Spearman is practically irrelevant over the tested noise range: both maintain at least 80% exact recovery through \(\sigma=0.10\).

Around \(k=100\), Spearman begins to lose noise tolerance while Pearson remains robust. In the `plus_independent` context, Spearman reaches the 80% criterion only through \(\sigma=0.005\), whereas Pearson remains at or above the criterion through \(\sigma=0.10\).

At \(k\ge200\), the rank-based method becomes highly fragile to positive observation noise.

At \(k=400\) and \(650\) in the multivariate context, the methods exchange strengths:

- Pearson fails even in the exact clean case because the nonlinear monotonic relation is too weak in linear correlation relative to the null threshold;
- Spearman succeeds perfectly in the exact clean case because ranks preserve strict monotonicity;
- Spearman's advantage disappears immediately under the smallest tested positive noise for these extreme transformations.

## 9. Conjectures tested and what happened to them

### 9.1 “The current controlled benchmarks may already favor one coefficient”

**Refuted.**

`controlled` and `controlled_noisy` did not discriminate the methods. The serialized outcomes were the same under the comparison that was run. A dedicated discriminating benchmark was necessary.

### 9.2 “Spearman will recognize monotonic nonlinear redundancy that Pearson can miss”

**Confirmed.**

`steep_positive_plus_independent` at \(k=650\) is a clean counterexample to the adequacy of Pearson as a general monotonic-dependence detector. Pearson returns \(3/3\); Spearman returns the correct \(2/2\).

The \(k\)-sweep generalizes this result: in the multivariate context, Pearson eventually loses the clean monotonic pair at sufficiently high nonlinearity.

### 9.3 “If Spearman recognizes the correct clean relation, it should also be a more robust default”

**Refuted.**

Recognition of exact monotonic order does not imply robustness to noisy observation of that order. In strongly compressed monotonic transformations, small additive perturbations can reorder many observations, degrading Spearman much faster than Pearson.

### 9.4 “The failure at \(k=650\) may be an artifact of an absurdly extreme example”

**Partially confirmed and partially refuted.**

It is indeed concentrated in the high-nonlinearity regime: up to roughly \(k=50\), Spearman and Pearson behave essentially the same over the tested grid.

However, the fragility begins already around \(k=100\), especially in the multivariate context. Therefore it is not unique to the single \(k=650\) example, although the most dramatic collapse is associated with the most extreme transformations.

### 9.5 “Spearman solves MISDA's nonlinear-dependence limitation”

**Refuted.**

The quadratic limit case demonstrates that a globally non-monotonic deterministic relation can have one latent dimension while neither Pearson nor Spearman identifies it. Spearman broadens the detectable class from linear to monotonic dependence; it does not become a general nonlinear dependence estimator.

## 10. Scientific interpretation

The investigation separates three concepts that can easily be conflated.

### 10.1 Exact order redundancy

If two objectives are related by an exact strictly increasing transformation, Spearman expresses a strong and conceptually attractive invariance:

\[
r_S=1
\]

regardless of the curvature of the transformation.

For a Pareto-oriented interpretation of redundancy, this is meaningful evidence in Spearman's favor.

### 10.2 Empirical detectability under a multivariate null

MISDA does not infer graph edges from a raw correlation alone. It calibrates an empirical null envelope from the observed sample size and objective count.

Therefore a relation can have a nonzero Pearson correlation and still fail to become an edge when the multivariate null reference is more demanding. The `steep_positive_plus_independent` case is exactly such an example.

### 10.3 Robustness to observation perturbation

Order invariance in clean data is not the same as robustness of the observed order.

When many transformed values are numerically compressed, additive observation error can produce many rank inversions. Spearman then loses the property that made it attractive in the clean data.

Pearson, because it depends on numeric covariance rather than only ranks, can retain substantial evidence in those same cases.

These are not contradictory observations. They show that Pearson and Spearman answer related but different questions.

## 11. Final conclusion of this investigation

The experiments do **not** support replacing Pearson by Spearman as the default discovery coefficient in MISDA.

The case for Spearman is real but narrower than initially conjectured:

- it detects exact positive monotonic redundancy that Pearson can miss;
- this advantage is robust to bootstrap resampling of a clean exact monotonic relation;
- it preserves the correct signed interpretation for exact decreasing monotonic dependence;
- it does not introduce changes in the existing controlled benchmarks.

However, the counterevidence is also strong:

- in strongly nonlinear monotonic relations with observation noise, Spearman can become substantially less robust than Pearson;
- the loss of robustness begins before the most extreme \(k=650\) case;
- in the \(k\times\sigma\) grid, Pearson dominates much of the noisy high-nonlinearity region;
- Spearman does not solve non-monotonic nonlinear latent dependence.

The most defensible conclusion at the end of this experiment is therefore:

> Keep signed Pearson correlation as the production default. Treat signed Spearman correlation as a scientifically useful complementary monotonic-sensitivity diagnostic or alternative hypothesis, not as a demonstrated replacement.

This is not a claim that Pearson is universally more correct. The investigation explicitly found clean monotonic cases where it is wrong relative to the declared structural truth. The decision not to replace it is based on the full tradeoff observed across exact monotonic invariance, multivariate thresholding, bootstrap stability, and observation-noise robustness.

## 12. What was deliberately *not* done

Several tempting extensions were intentionally not folded into the first investigation because they would have obscured the A/B question.

- No public `correlation="pearson"|"spearman"` parameter was introduced.
- No hybrid Pearson/Spearman decision rule was designed.
- No threshold was tuned after seeing the benchmark outcomes.
- No attempt was made to use nonlinear machine-learning dependence estimators as the structural graph statistic.
- No attempt was made to make Spearman detect globally non-monotonic dependence.
- No production change was made to `main`.

These remain separate research questions if the project later wants to pursue them.

## 13. Reproducibility record

### 13.1 Branch and implementation

- experimental branch: `spearman`;
- production baseline: `main`;
- current branch comparison at the end of this investigation: branch ahead of `main`, with the production baseline unchanged by the experiment;
- experimental statistic implementation: `misda/_statistics_spearman.py`;
- experimental routing: `misda/__init__.py`.

The Spearman implementation changes only the discovery-stage correlation statistic and its permutation-null maxima. Average ranks are centered and normalized; the signed rank-correlation matrix is computed by dot products of these standardized rank columns. Null permutations independently permute each rank column.

### 13.2 Benchmark files

- `benchmarks/run_monotonic.py`: clean discriminating cases;
- `benchmarks/monotonic.ipynb`: notebook form of the clean benchmark;
- `benchmarks/run_monotonic_robustness.py`: bootstrap and scaled observation-noise study;
- `benchmarks/run_monotonic_sweep.py`: \(k\times\sigma\) map;
- `.github/workflows/spearman-experiment.yml`: paired Pearson/Spearman execution and artifact preservation.

### 13.3 Important commits in the experimental path

- `50b6377cb20a2e92f2568c13a854f08ed6d60b8b` — corrected the semantic latent truth of the non-monotonic quadratic case to \(d_l=1\) and renamed it as a limit case;
- `39a1961e30175beb2e2effea9d606f0e86647e41` — added the monotonic robustness experiment;
- `2053b2e07fbd97dddaa66df5f36023c34d1b5c9c` — added the nonlinearity-by-noise sweep.

### 13.4 Workflow evidence

The final sweep workflow run was GitHub Actions run `35113292716`. The Pearson and Spearman monotonic sweep jobs both completed successfully, and their JSON artifacts were preserved along with the clean and robustness artifacts.

The experiment should be reproducible from the branch itself; the artifacts preserve the concrete outputs used in the tables above.

## 14. Current research status

This notebook closes the immediate question “should Pearson simply be replaced by Spearman?” with a negative answer.

It leaves a more nuanced open scientific possibility:

> A future MISDA may benefit from reporting disagreement between Pearson and Spearman as evidence about the *type* of dependence present, without allowing one coefficient to silently substitute for the other.

For example, a pair with high Spearman but modest Pearson may indicate strong monotonic order dependence with substantial metric deformation. Whether that should become a user-facing diagnostic requires a separate investigation, because this notebook shows that the same pattern can be fragile under observation noise.

Until such an investigation is carried out, this branch should be treated as experimental evidence rather than as a new method specification.
