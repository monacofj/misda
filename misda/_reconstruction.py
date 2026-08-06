# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Legacy reconstruction metrics used by MISDA validation."""

import numpy as np


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
