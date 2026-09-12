"""Executable controlled diagnostic problems with an explicit X -> Z -> Y flow.

R5 separates three notions that legacy benchmark generators often mixed in one
function:

- ``X``: sampled generating/decision variables;
- ``Z = F(X)``: clean theoretical objective values;
- ``Y``: observed values supplied to MISDA after an optional observation model.

The historical generators remain unchanged during migration and continue to
serve as regression fixtures.  New scientific experiments should use the
problem objects defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .diagnostics import DIAGNOSTIC_BY_ID, DiagnosticScenario


FrameFn = Callable[[int, int], pd.DataFrame]
EvalFn = Callable[[pd.DataFrame], pd.DataFrame]


def _objective_frame(columns: list[np.ndarray] | tuple[np.ndarray, ...]) -> pd.DataFrame:
    matrix = np.column_stack(columns)
    return pd.DataFrame(matrix, columns=[f"f{i}" for i in range(1, matrix.shape[1] + 1)])


def _normal_columns(rng: np.random.Generator, N: int, count: int) -> pd.DataFrame:
    values = np.column_stack([rng.normal(size=N) for _ in range(count)])
    return pd.DataFrame(values, columns=[f"x{i}" for i in range(1, count + 1)])


def _family_names(sizes: tuple[int, ...]) -> list[list[str]]:
    families: list[list[str]] = []
    start = 1
    for size in sizes:
        families.append([f"f{i}" for i in range(start, start + size)])
        start += size
    return families


@dataclass(frozen=True)
class DiagnosticDataset:
    """One sampled diagnostic problem and its clean/observed representations."""

    problem_id: str
    X: pd.DataFrame
    Z: pd.DataFrame
    Y: pd.DataFrame
    truth: dict
    sigma: float
    sample_seed: int
    observation_seed: int


@dataclass(frozen=True)
class DiagnosticProblem:
    """Executable theoretical diagnostic problem independent of MISDA."""

    scenario: DiagnosticScenario
    sampler: FrameFn
    evaluator: EvalFn

    @property
    def id(self) -> str:
        return self.scenario.id

    def sample(self, N: int = 1000, seed: int = 123) -> pd.DataFrame:
        if N < 1:
            raise ValueError("N must be at least 1")
        X = self.sampler(int(N), int(seed))
        if len(X) != N:
            raise RuntimeError(f"{self.id}: sampler returned the wrong row count")
        return X

    def evaluate(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self.evaluator(X.copy())
        expected_m = sum(self.scenario.family_sizes)
        if Z.shape != (len(X), expected_m):
            raise RuntimeError(
                f"{self.id}: evaluator returned {Z.shape}; expected {(len(X), expected_m)}"
            )
        expected_columns = [f"f{i}" for i in range(1, expected_m + 1)]
        if list(Z.columns) != expected_columns:
            raise RuntimeError(f"{self.id}: evaluator returned unexpected objective names")
        return Z

    def truth(self) -> dict:
        """Return generator truth; never inspect observed data or MISDA output."""
        truth = {
            "name": self.scenario.historical_name,
            "problem_id": self.id,
            "latent_expected": self.scenario.latent_expected,
            "structural_expected": self.scenario.structural_expected,
            "families_expected": _family_names(self.scenario.family_sizes),
            "pareto_expected": None,
            "tags": sorted(self.scenario.tags),
        }
        if self.scenario.structural_unit_sizes is not None:
            truth["blocks_expected"] = _family_names(
                self.scenario.structural_unit_sizes
            )
        return truth

    def observe(
        self,
        Z: pd.DataFrame,
        *,
        sigma: float = 0.0,
        seed: int = 123,
        standard_noise: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Observe clean objectives with scale-relative additive Gaussian noise.

        ``sigma=0`` is exactly the identity observation.  For ``sigma>0``, each
        objective j receives ``sigma * std(Z_j) * epsilon_j``.  Sampling and
        observation use separate RNGs.  An explicit ``standard_noise`` matrix
        can be supplied so robustness studies scale the same realization across
        several sigma values.
        """
        sigma = float(sigma)
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        values = Z.to_numpy(dtype=float)
        if sigma == 0.0:
            return Z.copy()

        if standard_noise is None:
            rng = np.random.default_rng(int(seed))
            epsilon = rng.normal(size=values.shape)
        else:
            epsilon = np.asarray(standard_noise, dtype=float)
            if epsilon.shape != values.shape:
                raise ValueError("standard_noise must have the same shape as Z")

        scales = np.std(values, axis=0, ddof=1) if len(Z) > 1 else np.zeros(values.shape[1])
        observed = values + sigma * scales[np.newaxis, :] * epsilon
        return pd.DataFrame(observed, columns=Z.columns, index=Z.index)

    def generate(
        self,
        N: int = 1000,
        *,
        seed: int = 123,
        sigma: float = 0.0,
        observation_seed: int | None = None,
        standard_noise: np.ndarray | None = None,
    ) -> DiagnosticDataset:
        """Generate X, clean Z, and observed Y in one explicit pipeline."""
        X = self.sample(N=N, seed=seed)
        Z = self.evaluate(X)
        obs_seed = int(seed if observation_seed is None else observation_seed)
        Y = self.observe(
            Z,
            sigma=sigma,
            seed=obs_seed,
            standard_noise=standard_noise,
        )
        return DiagnosticDataset(
            problem_id=self.id,
            X=X,
            Z=Z,
            Y=Y,
            truth=self.truth(),
            sigma=float(sigma),
            sample_seed=int(seed),
            observation_seed=obs_seed,
        )


# ---------------------------------------------------------------------------
# Historical Cases 1--7, expressed as theoretical generators.


def _sample_case1(N: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(N, 20)), columns=[f"x{i}" for i in range(1, 21)]
    )


def _eval_case1(X: pd.DataFrame) -> pd.DataFrame:
    return _objective_frame(tuple(X[f"x{i}"].to_numpy() for i in range(1, 21)))


def _sample_normal_1(N: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(size=N)})


def _eval_case2(X: pd.DataFrame) -> pd.DataFrame:
    x = X["x"].to_numpy()
    return _objective_frame(tuple(x for _ in range(20)))


def _sample_normal_4(N: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(N, 4))
    return pd.DataFrame(values, columns=["x1", "x2", "x3", "x4"])


def _eval_case3(X: pd.DataFrame) -> pd.DataFrame:
    return _objective_frame(
        tuple(X[f"x{block}"].to_numpy() for block in range(1, 5) for _ in range(5))
    )


def _sample_normal_2(N: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(N, 2))
    return pd.DataFrame(values, columns=["x1", "x2"])


def _eval_case4(X: pd.DataFrame) -> pd.DataFrame:
    x1 = X["x1"].to_numpy()
    x2 = X["x2"].to_numpy()
    return _objective_frame(tuple(x1 for _ in range(10)) + tuple(x2 for _ in range(10)))


def _sample_case5(N: int, seed: int) -> pd.DataFrame:
    # Column-wise draws intentionally mirror the historical innovation stream.
    rng = np.random.default_rng(seed)
    return _normal_columns(rng, N, 20)


def _eval_case5(X: pd.DataFrame) -> pd.DataFrame:
    x1 = X["x1"].to_numpy()
    cumulative = x1.copy()
    objectives = [cumulative.copy()]
    for j in range(2, 21):
        cumulative = cumulative + 0.2 * X[f"x{j}"].to_numpy()
        objectives.append(cumulative.copy())
    return _objective_frame(tuple(objectives))


def _sample_case6(N: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    independent = rng.normal(size=(N, 10))
    latent1 = rng.normal(size=N)
    latent2 = rng.normal(size=N)
    values = np.column_stack((independent, latent1, latent2))
    return pd.DataFrame(values, columns=[f"x{i}" for i in range(1, 13)])


def _eval_case6(X: pd.DataFrame) -> pd.DataFrame:
    objectives = [X[f"x{i}"].to_numpy() for i in range(1, 11)]
    objectives.extend(X["x11"].to_numpy() for _ in range(5))
    objectives.extend(X["x12"].to_numpy() for _ in range(5))
    return _objective_frame(tuple(objectives))


def _eval_case7(X: pd.DataFrame) -> pd.DataFrame:
    x = X["x"].to_numpy()
    return _objective_frame(tuple(x for _ in range(10)) + tuple(-x for _ in range(10)))


# ---------------------------------------------------------------------------
# Historical MOP-A--F, now with clean F(X) separated from observation.


def _sample_uniform_1(N: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.uniform(0.0, 1.0, size=N)})


def _eval_mop_a(X: pd.DataFrame) -> pd.DataFrame:
    x = X["x"].to_numpy()
    feats = (
        x,
        2.0 * x + 0.1,
        np.log(1.0 + 9.0 * x),
        x**2,
        np.sqrt(np.maximum(x, 0.0)),
        x**3,
        np.exp(0.5 * x) - 1.0,
        1.0 / (1.0 + np.exp(-10.0 * (x - 0.5))),
        (x + 0.2) ** 2,
        np.log(1.0 + 3.0 * x),
        np.tanh(2.0 * x),
        (1.0 + x) ** 1.5,
        np.clip(x + 0.05, 0.0, 1.0),
        np.clip(1.2 * x, 0.0, 1.0),
        np.log1p(20.0 * x) / np.log1p(20.0),
        (x + 1e-6) ** 0.25,
        (x + 0.1) ** 3,
        np.sqrt(np.maximum(0.1 + x, 0.0)),
        np.exp(x) - 1.0,
        (x + 0.3) ** 2,
    )
    return _objective_frame(feats)


def _sample_uniform_ab(N: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"a": rng.uniform(0.0, 1.0, size=N), "b": rng.uniform(0.0, 1.0, size=N)}
    )


def _eval_mop_b(X: pd.DataFrame) -> pd.DataFrame:
    a = X["a"].to_numpy()
    b = X["b"].to_numpy()
    C = 0.6 * a + 0.8 * b
    E = b + 0.3 * (1.0 - a)
    P = np.clip(a * (1.0 - b) + 0.2 * a, 0.0, 1.0)
    Q = 1.0 - P
    cost = (C, C, 1.0 + 2.0 * C, np.log1p(9.0 * C), np.sqrt(np.maximum(C, 0.0)), C**2, (C + 0.1) ** 1.5)
    consumption = (E, E, np.sqrt(np.maximum(E, 0.0)), np.log1p(9.0 * E), E**2, E + 0.05, (E + 0.2) ** 1.3)
    performance = (Q, Q, Q**2, np.sqrt(np.maximum(Q, 0.0)), np.log1p(9.0 * Q), (Q + 0.1) ** 1.2)
    return _objective_frame(cost + consumption + performance)


def _sample_mop_c(N: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.0, 1.0, size=(4, N)).T
    return pd.DataFrame(values, columns=["u", "v", "w", "z"])


def _eval_mop_c(X: pd.DataFrame) -> pd.DataFrame:
    u = X["u"].to_numpy()
    v = X["v"].to_numpy()
    w = X["w"].to_numpy()
    z = X["z"].to_numpy()
    b1 = (u, 2.0 * u, u**2, np.sqrt(np.maximum(u, 0.0)), np.log1p(9.0 * u))
    b2 = (v, v + 0.5, np.log1p(9.0 * v), v**2, np.sqrt(np.maximum(v, 0.0)))
    b3 = (w, w, np.sqrt(np.maximum(w, 0.0)), np.log1p(9.0 * w), (w + 0.1) ** 2)
    b4 = (z, (1.0 + z) ** 2, np.exp(z) - 1.0, np.log1p(9.0 * z), np.sqrt(np.maximum(z, 0.0)))
    return _objective_frame(b1 + b2 + b3 + b4)


def _eval_mop_d(X: pd.DataFrame) -> pd.DataFrame:
    x = X["x"].to_numpy()
    y = 1.0 - x

    def transforms(value: np.ndarray) -> tuple[np.ndarray, ...]:
        return (
            value,
            2.0 * value + 0.1,
            np.log1p(9.0 * value),
            value**2,
            np.sqrt(np.maximum(value, 0.0)),
            value**3,
            np.tanh(2.0 * value),
            np.log1p(3.0 * value),
            (value + 0.2) ** 2,
            (1.0 + value) ** 1.5,
        )

    return _objective_frame(transforms(x) + transforms(y))


def _eval_mop_e(X: pd.DataFrame) -> pd.DataFrame:
    a = X["a"].to_numpy()
    b = X["b"].to_numpy()
    A = (
        a,
        a,
        a,
        2.0 * a + 0.1,
        a**2,
        np.sqrt(np.maximum(a, 0.0)),
        np.log1p(9.0 * a),
        (a + 0.2) ** 2,
        np.tanh(2.0 * a),
        (1.0 + a) ** 1.2,
    )
    B = (b, b + 0.5, np.sqrt(np.maximum(b, 0.0)), np.log1p(9.0 * b))
    s = a + b
    C = (
        s,
        s**2,
        np.sqrt(np.maximum(s, 0.0)),
        np.log1p(9.0 * s),
        (s + 0.1) ** 1.5,
        1.0 / (1.0 + np.exp(-10.0 * (s - 1.0))),
    )
    return _objective_frame(A + B + C)


def _eval_mop_f(X: pd.DataFrame) -> pd.DataFrame:
    a = X["a"].to_numpy()
    b = X["b"].to_numpy()
    switch = 1.0 / (1.0 + np.exp(-20.0 * (a - 0.5)))
    L = (1.0 - switch) * a + switch * b

    def transforms(value: np.ndarray) -> tuple[np.ndarray, ...]:
        return (
            value,
            value**2,
            np.log1p(9.0 * value),
            np.sqrt(np.maximum(value, 0.0)),
            (value + 0.1) ** 1.5,
            np.tanh(2.0 * value),
            np.exp(0.5 * value) - 1.0,
            (value + 0.2) ** 2,
            np.log1p(3.0 * value),
            value,
        )

    return _objective_frame(transforms(L) + transforms(b))


PROBLEMS = (
    DiagnosticProblem(DIAGNOSTIC_BY_ID["independence"], _sample_case1, _eval_case1),
    DiagnosticProblem(DIAGNOSTIC_BY_ID["total_redundancy"], _sample_normal_1, _eval_case2),
    DiagnosticProblem(DIAGNOSTIC_BY_ID["blocks_4x5"], _sample_normal_4, _eval_case3),
    DiagnosticProblem(DIAGNOSTIC_BY_ID["blocks_2x10"], _sample_normal_2, _eval_case4),
    DiagnosticProblem(DIAGNOSTIC_BY_ID["transitive_chain"], _sample_case5, _eval_case5),
    DiagnosticProblem(DIAGNOSTIC_BY_ID["mixed_independent_and_blocks"], _sample_case6, _eval_case6),
    DiagnosticProblem(DIAGNOSTIC_BY_ID["antagonistic_linear_groups"], _sample_normal_1, _eval_case7),
    DiagnosticProblem(DIAGNOSTIC_BY_ID["monotonic_redundancy"], _sample_uniform_1, _eval_mop_a),
    DiagnosticProblem(DIAGNOSTIC_BY_ID["tradeoff_redundancies"], _sample_uniform_ab, _eval_mop_b),
    DiagnosticProblem(DIAGNOSTIC_BY_ID["nonlinear_blocks_4x5"], _sample_mop_c, _eval_mop_c),
    DiagnosticProblem(DIAGNOSTIC_BY_ID["antagonistic_nonlinear_groups"], _sample_uniform_1, _eval_mop_d),
    DiagnosticProblem(DIAGNOSTIC_BY_ID["overlapping_factors"], _sample_uniform_ab, _eval_mop_e),
    DiagnosticProblem(DIAGNOSTIC_BY_ID["regime_switching"], _sample_uniform_ab, _eval_mop_f),
)

PROBLEM_BY_ID = {problem.id: problem for problem in PROBLEMS}

if len(PROBLEM_BY_ID) != len(PROBLEMS):
    raise RuntimeError("Diagnostic problem ids must be unique")
