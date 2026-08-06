# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Linear, nonlinear, and legacy reconstruction metrics for MISDA."""

from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def _explicit_loo_predictions(data, selected, eliminated):
    n_samples = data.shape[0]
    predictions = np.empty((n_samples, len(eliminated)), dtype=float)
    for held_out in range(n_samples):
        training = np.arange(n_samples) != held_out
        design_train = np.column_stack(
            (np.ones(np.sum(training)), data[training][:, selected])
        )
        design_test = np.concatenate(
            ([1.0], data[held_out, selected])
        )
        coefficients = np.linalg.pinv(design_train) @ data[training][:, eliminated]
        predictions[held_out] = design_test @ coefficients
    return predictions


def _press_predictions(data, selected, eliminated):
    design = np.column_stack((np.ones(data.shape[0]), data[:, selected]))
    design_pinv = np.linalg.pinv(design)
    targets = data[:, eliminated]
    fitted = design @ (design_pinv @ targets)
    residuals = targets - fitted
    leverage = np.einsum("ij,ji->i", design, design_pinv)
    denominator = 1.0 - leverage
    tolerance = np.finfo(float).eps * max(10.0, float(design.shape[1]))
    if np.any(np.abs(denominator) <= tolerance):
        return _explicit_loo_predictions(data, selected, eliminated)
    return targets - residuals / denominator[:, np.newaxis]


def _r2_metrics(data, selected, eliminated, labels):
    predictions = _press_predictions(data, selected, eliminated)
    values = {}
    reasons = {}
    defined = []
    for position, objective in enumerate(eliminated):
        label = labels[objective]
        observed = data[:, objective]
        total = float(np.sum((observed - np.mean(observed)) ** 2))
        if total <= np.finfo(float).eps:
            values[label] = None
            reasons[label] = "CONSTANT_TARGET"
            continue
        residual = float(np.sum((observed - predictions[:, position]) ** 2))
        value = float(1.0 - residual / total)
        values[label] = value
        defined.append(value)
    return {
        "r2_by_objective": values,
        "r2_reason_by_objective": reasons,
        "mean_r2": float(np.mean(defined)) if defined else None,
        "worst_r2": float(np.min(defined)) if defined else None,
    }


def _jackknife_standard_error(values):
    if not values or any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=float)
    center = float(np.mean(array))
    return float(
        np.sqrt((len(array) - 1) / len(array) * np.sum((array - center) ** 2))
    )


def evaluate_linear_reconstruction(data, selected_indices, labels):
    """Evaluate eliminated objectives by external PRESS/LOO and jackknife."""

    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("data must be a two-dimensional matrix.")
    n_samples, n_objectives = matrix.shape
    selected = tuple(sorted(set(int(index) for index in selected_indices)))
    if not selected:
        raise ValueError("selected_indices must not be empty.")
    if any(index < 0 or index >= n_objectives for index in selected):
        raise IndexError("selected_indices contains an out-of-range objective.")
    if len(labels) != n_objectives:
        raise ValueError("labels must contain one value per objective.")

    eliminated = tuple(
        index for index in range(n_objectives) if index not in selected
    )
    if not eliminated:
        return {
            "r2_by_objective": None,
            "r2_reason_by_objective": {},
            "mean_r2": None,
            "worst_r2": None,
            "reason_by_metric": {
                "r2_by_objective": "NO_ELIMINATED_OBJECTIVES",
                "mean_r2": "NO_ELIMINATED_OBJECTIVES",
                "worst_r2": "NO_ELIMINATED_OBJECTIVES",
            },
            "jackknife": {
                "r2_se_by_objective": None,
                "mean_r2_se": None,
                "worst_r2_se": None,
                "n_replicates": 0,
                "reason": "NO_ELIMINATED_OBJECTIVES",
            },
        }

    full = _r2_metrics(matrix, selected, eliminated, labels)
    replicates = [
        _r2_metrics(
            np.delete(matrix, omitted, axis=0),
            selected,
            eliminated,
            labels,
        )
        for omitted in range(n_samples)
    ]
    r2_se = {
        labels[objective]: _jackknife_standard_error(
            [
                replicate["r2_by_objective"][labels[objective]]
                for replicate in replicates
            ]
        )
        for objective in eliminated
    }
    reason_by_metric = {}
    if full["mean_r2"] is None:
        reason_by_metric["mean_r2"] = "NO_DEFINED_TARGETS"
    if full["worst_r2"] is None:
        reason_by_metric["worst_r2"] = "NO_DEFINED_TARGETS"
    return {
        **full,
        "reason_by_metric": reason_by_metric,
        "jackknife": {
            "r2_se_by_objective": r2_se,
            "mean_r2_se": _jackknife_standard_error(
                [replicate["mean_r2"] for replicate in replicates]
            ),
            "worst_r2_se": _jackknife_standard_error(
                [replicate["worst_r2"] for replicate in replicates]
            ),
            "n_replicates": n_samples,
            "reason": None,
        },
    }


def _derive_seed(seed, *coordinates):
    """Derive a stable uint32 seed without sharing mutable RNG state."""

    sequence = np.random.SeedSequence([int(seed), *map(int, coordinates)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _r2_from_predictions(observed, predicted):
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    if total <= np.finfo(float).eps:
        return None
    residual = float(np.sum((observed - predicted) ** 2))
    return float(1.0 - residual / total)


def _prediction_jackknife_se(observed, predicted):
    values = [
        _r2_from_predictions(
            np.delete(observed, omitted),
            np.delete(predicted, omitted),
        )
        for omitted in range(len(observed))
    ]
    return _jackknife_standard_error(values)


def _summarize_external_predictions(data, eliminated, labels, predictions):
    values = {}
    reasons = {}
    sample_se = {}
    defined = []
    for position, objective in enumerate(eliminated):
        label = labels[objective]
        observed = data[:, objective]
        predicted = predictions[:, position]
        value = _r2_from_predictions(observed, predicted)
        values[label] = value
        if value is None:
            reasons[label] = "CONSTANT_TARGET"
            sample_se[label] = None
        else:
            defined.append(value)
            sample_se[label] = _prediction_jackknife_se(observed, predicted)

    mean_r2 = float(np.mean(defined)) if defined else None
    worst_r2 = float(np.min(defined)) if defined else None
    mean_replicates = []
    worst_replicates = []
    for omitted in range(data.shape[0]):
        replicate_values = []
        for position, objective in enumerate(eliminated):
            value = _r2_from_predictions(
                np.delete(data[:, objective], omitted),
                np.delete(predictions[:, position], omitted),
            )
            if value is not None:
                replicate_values.append(value)
        mean_replicates.append(
            float(np.mean(replicate_values)) if replicate_values else None
        )
        worst_replicates.append(
            float(np.min(replicate_values)) if replicate_values else None
        )

    return {
        "r2_by_objective": values,
        "r2_reason_by_objective": reasons,
        "mean_r2": mean_r2,
        "worst_r2": worst_r2,
        "reason_by_metric": (
            {
                "mean_r2": "NO_DEFINED_TARGETS",
                "worst_r2": "NO_DEFINED_TARGETS",
            }
            if not defined
            else {}
        ),
        "jackknife": {
            "r2_se_by_objective": sample_se,
            "mean_r2_se": _jackknife_standard_error(mean_replicates),
            "worst_r2_se": _jackknife_standard_error(worst_replicates),
            "n_replicates": data.shape[0],
            "reason": None if defined else "NO_DEFINED_TARGETS",
        },
    }


def _rf_configuration_domain(n_predictors, n_training):
    """Return every discrete admissible RF flexibility configuration."""

    max_leaf = max(1, (n_training - 1) // 2)
    return tuple(
        (max_features, min_samples_leaf)
        for max_features in range(1, n_predictors + 1)
        for min_samples_leaf in range(1, max_leaf + 1)
    )


def _select_rf_configuration(
    predictors,
    target,
    *,
    n_trees,
    seed,
    cancel_requested=None,
    model_factory=RandomForestRegressor,
):
    """Select RF flexibility by deterministic inner leave-one-out MSE."""

    best = None
    for config_index, (max_features, min_samples_leaf) in enumerate(
        _rf_configuration_domain(predictors.shape[1], predictors.shape[0])
    ):
        predictions = np.empty(predictors.shape[0], dtype=float)
        for held_out in range(predictors.shape[0]):
            if cancel_requested is not None and cancel_requested():
                return None
            training = np.arange(predictors.shape[0]) != held_out
            model = model_factory(
                n_estimators=n_trees,
                criterion="squared_error",
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                random_state=_derive_seed(
                    seed,
                    config_index,
                    held_out,
                ),
                n_jobs=1,
            )
            model.fit(predictors[training], target[training])
            predictions[held_out] = model.predict(
                predictors[held_out : held_out + 1]
            )[0]
        mse = float(np.mean((target - predictions) ** 2))
        simplicity = (-min_samples_leaf, max_features)
        if best is None:
            best = (mse, simplicity, max_features, min_samples_leaf)
            continue
        if mse < best[0] and not np.isclose(
            mse,
            best[0],
            rtol=1e-12,
            atol=1e-15,
        ):
            best = (mse, simplicity, max_features, min_samples_leaf)
        elif np.isclose(mse, best[0], rtol=1e-12, atol=1e-15):
            if simplicity < best[1]:
                best = (mse, simplicity, max_features, min_samples_leaf)
    return {
        "max_features": best[2],
        "min_samples_leaf": best[3],
        "validation_mse": best[0],
    }


def _tree_standard_errors(data, eliminated, labels, tree_predictions):
    errors = {}
    for position, objective in enumerate(eliminated):
        label = labels[objective]
        observed = data[:, objective]
        values = [
            _r2_from_predictions(observed, prediction[:, position])
            for prediction in tree_predictions
        ]
        if any(value is None for value in values):
            errors[label] = None
        elif len(values) < 2:
            errors[label] = 0.0
        else:
            errors[label] = float(
                np.std(values, ddof=1) / np.sqrt(len(values))
            )
    return errors


def _tree_stopping_reached(tree_se, sample_se):
    defined = [
        label
        for label, value in sample_se.items()
        if value is not None and tree_se.get(label) is not None
    ]
    return not defined or all(
        tree_se[label] <= sample_se[label]
        or np.isclose(
            tree_se[label],
            sample_se[label],
            rtol=1e-12,
            atol=0.0,
        )
        for label in defined
    )


def _no_reduction_nonlinear_result():
    return {
        "r2_by_objective": None,
        "r2_reason_by_objective": {},
        "mean_r2": None,
        "worst_r2": None,
        "reason_by_metric": {
            "r2_by_objective": "NO_ELIMINATED_OBJECTIVES",
            "mean_r2": "NO_ELIMINATED_OBJECTIVES",
            "worst_r2": "NO_ELIMINATED_OBJECTIVES",
        },
        "jackknife": {
            "r2_se_by_objective": None,
            "mean_r2_se": None,
            "worst_r2_se": None,
            "n_replicates": 0,
            "reason": "NO_ELIMINATED_OBJECTIVES",
        },
        "tree_se_by_objective": None,
        "n_trees": 0,
        "configuration_counts": {},
        "configuration_by_outer_fold": {},
        "converged": True,
        "cancelled": False,
        "convergence_reason": "NO_ELIMINATED_OBJECTIVES",
    }


def _cancelled_nonlinear_result(reason):
    result = _no_reduction_nonlinear_result()
    result["reason_by_metric"] = {
        "r2_by_objective": reason,
        "mean_r2": reason,
        "worst_r2": reason,
    }
    result["jackknife"] = {
        "r2_se_by_objective": None,
        "mean_r2_se": None,
        "worst_r2_se": None,
        "n_replicates": 0,
        "reason": reason,
    }
    result["converged"] = False
    result["cancelled"] = True
    result["convergence_reason"] = reason
    return result


def evaluate_nonlinear_reconstruction(
    data,
    selected_indices,
    labels,
    *,
    seed=123,
    cancel_requested=None,
    model_factory=RandomForestRegressor,
):
    """Evaluate nonlinear reconstruction by nested external LOO Random Forest."""

    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("data must be a two-dimensional matrix.")
    n_samples, n_objectives = matrix.shape
    selected = tuple(sorted(set(int(index) for index in selected_indices)))
    if not selected:
        raise ValueError("selected_indices must not be empty.")
    if any(index < 0 or index >= n_objectives for index in selected):
        raise IndexError("selected_indices contains an out-of-range objective.")
    if len(labels) != n_objectives:
        raise ValueError("labels must contain one value per objective.")
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None.")

    eliminated = tuple(
        index for index in range(n_objectives) if index not in selected
    )
    if not eliminated:
        return _no_reduction_nonlinear_result()
    if cancel_requested is not None and cancel_requested():
        return _cancelled_nonlinear_result("CANCELLED_BEFORE_EVALUATION")

    n_trees = n_samples
    configurations = {}
    models = {}
    for held_out in range(n_samples):
        training = np.arange(n_samples) != held_out
        predictors = matrix[training][:, selected]
        for target_position, objective in enumerate(eliminated):
            configuration = _select_rf_configuration(
                predictors,
                matrix[training, objective],
                n_trees=n_trees,
                seed=_derive_seed(seed, held_out, target_position),
                cancel_requested=cancel_requested,
                model_factory=model_factory,
            )
            if configuration is None:
                return _cancelled_nonlinear_result(
                    "CANCELLED_DURING_MODEL_SELECTION"
                )
            key = (held_out, target_position)
            configurations[key] = configuration
            model = model_factory(
                n_estimators=n_trees,
                criterion="squared_error",
                max_features=configuration["max_features"],
                min_samples_leaf=configuration["min_samples_leaf"],
                random_state=_derive_seed(seed, 1, held_out, target_position),
                n_jobs=1,
                warm_start=True,
            )
            model.fit(predictors, matrix[training, objective])
            models[key] = model

    while True:
        predictions = np.empty((n_samples, len(eliminated)), dtype=float)
        per_tree = np.empty(
            (n_trees, n_samples, len(eliminated)),
            dtype=float,
        )
        for held_out in range(n_samples):
            held_predictors = matrix[held_out : held_out + 1, selected]
            for target_position, _objective in enumerate(eliminated):
                model = models[(held_out, target_position)]
                tree_values = np.asarray(
                    [
                        estimator.predict(held_predictors)[0]
                        for estimator in model.estimators_
                    ],
                    dtype=float,
                )
                per_tree[:, held_out, target_position] = tree_values
                predictions[held_out, target_position] = float(
                    np.mean(tree_values)
                )

        summary = _summarize_external_predictions(
            matrix,
            eliminated,
            labels,
            predictions,
        )
        tree_se = _tree_standard_errors(
            matrix,
            eliminated,
            labels,
            per_tree,
        )
        undefined_sample_uncertainty = [
            label
            for label, value in summary["r2_by_objective"].items()
            if value is not None
            and summary["jackknife"]["r2_se_by_objective"][label] is None
        ]
        if undefined_sample_uncertainty:
            converged = False
            cancelled = False
            convergence_reason = "SAMPLE_UNCERTAINTY_UNDEFINED"
            break
        if _tree_stopping_reached(
            tree_se,
            summary["jackknife"]["r2_se_by_objective"],
        ):
            converged = True
            cancelled = False
            convergence_reason = None
            break
        if cancel_requested is not None and cancel_requested():
            converged = False
            cancelled = True
            convergence_reason = "EXTERNAL_CANCELLATION"
            break
        n_trees += 1
        for (held_out, target_position), model in models.items():
            training = np.arange(n_samples) != held_out
            model.set_params(n_estimators=n_trees)
            model.fit(
                matrix[training][:, selected],
                matrix[training, eliminated[target_position]],
            )

    counts = Counter(
        (
            configuration["max_features"],
            configuration["min_samples_leaf"],
        )
        for configuration in configurations.values()
    )
    return {
        **summary,
        "tree_se_by_objective": tree_se,
        "n_trees": n_trees,
        "configuration_counts": {
            f"max_features={max_features},min_samples_leaf={min_leaf}": count
            for (max_features, min_leaf), count in sorted(counts.items())
        },
        "configuration_by_outer_fold": {
            f"{held_out}:{labels[eliminated[target_position]]}": dict(
                configuration
            )
            for (held_out, target_position), configuration in sorted(
                configurations.items()
            )
        },
        "converged": converged,
        "cancelled": cancelled,
        "convergence_reason": convergence_reason,
    }


def evaluate_null_reconstruction(
    data,
    selected_indices,
    labels,
    observed,
    *,
    seed=123,
    cancel_requested=None,
    evaluator=evaluate_nonlinear_reconstruction,
):
    """Calibrate nonlinear mean R² against a sequential null reference."""

    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("data must be a non-empty two-dimensional matrix.")
    selected = tuple(sorted(set(int(index) for index in selected_indices)))
    if not selected:
        raise ValueError("selected_indices must not be empty.")
    if any(index < 0 or index >= matrix.shape[1] for index in selected):
        raise IndexError("selected_indices contains an out-of-range objective.")
    if len(labels) != matrix.shape[1]:
        raise ValueError("labels must contain one value per objective.")
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None.")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable.")
    eliminated = tuple(
        index for index in range(matrix.shape[1]) if index not in selected
    )
    observed_mean = observed.get("mean_r2")
    observed_se = observed.get("jackknife", {}).get("mean_r2_se")
    if observed_mean is None or observed_se is None:
        return {
            "mean_null_r2": None,
            "above_null_r2": None,
            "incidental_reconstruction_rate": None,
            "n_permutations": 0,
            "mc_se_mean_null_r2": None,
            "above_null_r2_se": None,
            "incidental_reconstruction_rate_se": None,
            "converged": True,
            "cancelled": False,
            "reason": "OBSERVED_RECONSTRUCTION_UNDEFINED",
        }

    rng = np.random.default_rng(_derive_seed(seed, 7001))
    null_values = []
    converged = False
    cancelled = False
    mc_se = None
    while True:
        if cancel_requested is not None and cancel_requested():
            cancelled = True
            break
        permutation = rng.permutation(matrix.shape[0])
        permuted = matrix.copy()
        permuted[:, eliminated] = matrix[permutation][:, eliminated]
        null_result = evaluator(
            permuted,
            selected,
            labels,
            seed=_derive_seed(seed, 7002, len(null_values)),
            cancel_requested=cancel_requested,
        )
        null_mean = null_result.get("mean_r2")
        if not null_result.get("converged", True):
            cancelled = bool(null_result.get("cancelled", False))
            if null_mean is not None:
                null_values.append(float(null_mean))
            break
        if null_mean is None:
            raise RuntimeError(
                "The null evaluator returned an undefined mean_r2 for a "
                "defined observed reconstruction."
            )
        null_values.append(float(null_mean))
        if len(null_values) < matrix.shape[0]:
            continue
        mc_se = (
            float(np.std(null_values, ddof=1) / np.sqrt(len(null_values)))
            if len(null_values) > 1
            else 0.0
        )
        if mc_se <= observed_se or np.isclose(
            mc_se,
            observed_se,
            rtol=1e-12,
            atol=0.0,
        ):
            converged = True
            break

    if not null_values:
        return {
            "mean_null_r2": None,
            "above_null_r2": None,
            "incidental_reconstruction_rate": None,
            "n_permutations": 0,
            "mc_se_mean_null_r2": None,
            "above_null_r2_se": None,
            "incidental_reconstruction_rate_se": None,
            "converged": False,
            "cancelled": cancelled,
            "reason": "CANCELLED_BEFORE_NULL_EVALUATION",
        }

    mean_null = float(np.mean(null_values))
    if mc_se is None:
        mc_se = (
            float(np.std(null_values, ddof=1) / np.sqrt(len(null_values)))
            if len(null_values) > 1
            else 0.0
        )
    incidental = float(
        (1 + sum(value >= observed_mean for value in null_values))
        / (len(null_values) + 1)
    )
    return {
        "mean_null_r2": mean_null,
        "above_null_r2": float(observed_mean - mean_null),
        "incidental_reconstruction_rate": incidental,
        "n_permutations": len(null_values),
        "mc_se_mean_null_r2": mc_se,
        "above_null_r2_se": mc_se,
        "incidental_reconstruction_rate_se": float(
            np.sqrt(incidental * (1.0 - incidental) / (len(null_values) + 1))
        ),
        "converged": converged,
        "cancelled": cancelled,
        "reason": None,
    }


def _calculate_ses_core(
    Y,
    mis,
    model_type="linear",
    n_perm=20,
    test_size=0.3,
    seed=123,
    clip=True,
    n_estimators=100,
):
    """
    Unified core engine for Linear (OLS) and Non-Linear (Random Forest) SES.
    Predicts ONLY eliminated targets T from kept predictors S.
    """
    if hasattr(Y, "values") and hasattr(Y, "columns"):
        cols = list(Y.columns)
        Ymat = np.asarray(Y.values, dtype=float)
        names = cols
    else:
        Ymat = np.asarray(Y, dtype=float)
        if Ymat.ndim != 2:
            raise ValueError("Y must be 2D matrix (N x M).")
        cols = None
        names = [f"f{i+1}" for i in range(Ymat.shape[1])]

    N, M = Ymat.shape
    if N < 2:
        raise ValueError("Y must have at least 2 samples.")
    if M < 1:
        raise ValueError("Y must have at least 1 feature.")

    # Process mis (kept indices / labels)
    if isinstance(mis, dict) and "mis_indices" in mis:
        mis_list = mis["mis_indices"]
    else:
        mis_list = mis

    if len(mis_list) == 0:
        raise ValueError("mis cannot be empty.")

    if cols is not None and isinstance(mis_list[0], str):
        S_idx = [cols.index(c) for c in mis_list]
    else:
        S_idx = list(map(int, mis_list))

    S_idx = sorted(set(S_idx))
    if any(i < 0 or i >= M for i in S_idx):
        raise ValueError("mis contains index outside of range [0, M).")

    # Validate parameters
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be strictly between 0 and 1.")
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1.")

    T_idx = [j for j in range(M) if j not in S_idx]

    # Edge Case: No Reduction (All objectives kept: T is empty)
    if len(T_idx) == 0:
        return {
            "ses": None,
            "F_real": None,
            "F_null": None,
            "mis_size": len(S_idx),
            "M": int(M),
            "N": int(N),
            "targets_reconstructed": [],
            "r2_real": {},
            "r2_null": {},
            "status": "NO_REDUCTION",
            "model_type": model_type,
            "settings": {
                "n_perm": int(n_perm),
                "test_size": float(test_size),
                "seed": int(seed),
                "clip": bool(clip),
            },
        }

    if model_type == "nonlinear":
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            raise ImportError(
                "scikit-learn is required to calculate non-linear SES (RandomForestRegressor)."
            )

    # Train / Test split
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_test = int(np.round(test_size * N))
    n_test = min(max(n_test, 1), N - 1)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    def compute_F_and_r2dict(X_tr, X_te, base_seed):
        Y_tr = Ymat[train_idx, :][:, T_idx]
        Y_te = Ymat[test_idx, :][:, T_idx]

        if model_type == "linear":
            Xtr_b = np.column_stack([np.ones((X_tr.shape[0], 1)), X_tr])
            Xte_b = np.column_stack([np.ones((X_te.shape[0], 1)), X_te])
            beta, *_ = np.linalg.lstsq(Xtr_b, Y_tr, rcond=None)
            Y_hat = Xte_b @ beta
        elif model_type == "nonlinear":
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(
                n_estimators=n_estimators, random_state=base_seed, n_jobs=-1
            )
            Y_tr_fit = Y_tr.ravel() if Y_tr.shape[1] == 1 else Y_tr
            rf.fit(X_tr, Y_tr_fit)
            Y_hat = rf.predict(X_te)
            if Y_hat.ndim == 1:
                Y_hat = Y_hat[:, np.newaxis]
        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

        r2 = {}
        vals = []
        for idx_k, j in enumerate(T_idx):
            y_test_j = Y_te[:, idx_k]
            y_hat_j = Y_hat[:, idx_k]
            ss_res = float(np.sum((y_test_j - y_hat_j) ** 2))
            y_mean = float(np.mean(y_test_j))
            ss_tot = float(np.sum((y_test_j - y_mean) ** 2))
            if ss_tot <= 1e-15:
                r2[names[j]] = None
            else:
                r2_j = float(1.0 - (ss_res / ss_tot))
                r2[names[j]] = r2_j
                vals.append(max(0.0, r2_j))

        if len(vals) == 0:
            return None, r2
        return float(np.mean(vals)), r2

    X_real = Ymat[:, S_idx]
    X_tr_real = X_real[train_idx, :]
    X_te_real = X_real[test_idx, :]
    F_real, r2_real = compute_F_and_r2dict(X_tr_real, X_te_real, seed)

    if F_real is None:
        return {
            "ses": None,
            "F_real": None,
            "F_null": None,
            "mis_size": len(S_idx),
            "M": int(M),
            "N": int(N),
            "targets_reconstructed": [names[j] for j in T_idx],
            "r2_real": r2_real,
            "r2_null": {},
            "status": "UNDEFINED_TARGETS",
            "model_type": model_type,
            "settings": {
                "n_perm": int(n_perm),
                "test_size": float(test_size),
                "seed": int(seed),
                "clip": bool(clip),
            },
        }

    # Permutation null model: permute within train and test independently
    r2_null_acc = {names[j]: [] for j in T_idx}
    F_null_vals = []

    for b in range(int(n_perm)):
        perm_seed_tr = seed + 1000 + b * 2
        perm_seed_te = seed + 1000 + b * 2 + 1
        rng_tr = np.random.default_rng(perm_seed_tr)
        rng_te = np.random.default_rng(perm_seed_te)

        # Joint row permutation: permute rows of S in block to preserve internal multivariate structure
        p_tr = rng_tr.permutation(len(train_idx))
        X_tr_perm = X_tr_real[p_tr, :].copy()

        p_te = rng_te.permutation(len(test_idx))
        X_te_perm = X_te_real[p_te, :].copy()

        b_seed = seed + 5000 + b * 100
        Fb, r2b = compute_F_and_r2dict(X_tr_perm, X_te_perm, b_seed)
        if Fb is not None:
            F_null_vals.append(Fb)
            for k, v in r2b.items():
                if v is not None:
                    r2_null_acc[k].append(v)

    if len(F_null_vals) > 0:
        F_null = float(np.mean(F_null_vals))
    else:
        F_null = 0.0

    r2_null = {
        k: (float(np.mean(vs)) if len(vs) > 0 else None)
        for k, vs in r2_null_acc.items()
    }

    denom = 1.0 - F_null
    if denom <= 0:
        ses = 0.0 if (F_real <= F_null) else 1.0
    else:
        ses = (F_real - F_null) / denom

    if clip:
        ses = float(np.clip(ses, 0.0, 1.0))
    else:
        ses = float(ses)

    return {
        "ses": ses,
        "F_real": float(F_real),
        "F_null": float(F_null),
        "mis_size": len(S_idx),
        "M": int(M),
        "N": int(N),
        "targets_reconstructed": [names[j] for j in T_idx],
        "r2_real": r2_real,
        "r2_null": r2_null,
        "status": "SUCCESS",
        "model_type": model_type,
        "settings": {
            "n_perm": int(n_perm),
            "test_size": float(test_size),
            "seed": int(seed),
            "clip": bool(clip),
        },
    }


def calculate_ses_linear(
    Y,
    mis,
    *,
    n_perm=20,
    test_size=0.3,
    seed=123,
    clip=True,
    return_details=True,
):
    """
    Calculates Linear SES (Structural Evidence Score) using OLS Linear Regression.
    """
    out = _calculate_ses_core(
        Y,
        mis,
        model_type="linear",
        n_perm=n_perm,
        test_size=test_size,
        seed=seed,
        clip=clip,
    )
    return out if return_details else out["ses"]


def calculate_ses(
    Y,
    mis,
    *,
    n_perm=20,
    test_size=0.3,
    seed=123,
    clip=True,
    return_details=True,
):
    """
    calculate_ses(Y, mis) -> ses (0..1) + details

    SES = Structural Evidence Score (Linear OLS).
    Alias for calculate_ses_linear.
    """
    return calculate_ses_linear(
        Y,
        mis,
        n_perm=n_perm,
        test_size=test_size,
        seed=seed,
        clip=clip,
        return_details=return_details,
    )


def calculate_ses_nonlinear(
    Y,
    mis,
    *,
    n_perm=20,
    test_size=0.3,
    seed=123,
    clip=True,
    n_estimators=100,
    return_details=False,
):
    """
    Calculates Non-Linear SES using Random Forest Regression.
    Returns scalar SES by default (or detailed dict if return_details=True).

    Args:
        Y: (N, M) matrix or DataFrame
        mis: list of indices/labels (or dict from result)
        n_perm: number of permutations for null model
        test_size: fraction for test set (default 0.3)
        seed: random seed for reproducibility
        clip: clip SES score to [0, 1]
        n_estimators: trees in RF
        return_details: if True, returns full dict instead of float

    Returns:
        float or None (if return_details=False) or dict (if return_details=True)
    """
    out = _calculate_ses_core(
        Y,
        mis,
        model_type="nonlinear",
        n_perm=n_perm,
        test_size=test_size,
        seed=seed,
        clip=clip,
        n_estimators=n_estimators,
    )
    return out if return_details else out["ses"]
