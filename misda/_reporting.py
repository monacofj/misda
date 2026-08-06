# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Legacy textual reporting helpers."""

import numpy as np


def explain_ses(out, top_k=8, name=None, show_all=False):
    """
    Explains the result of calculate_ses(out). Returns string report.
    SES = Structural Evidence Score.
    """
    if out is None or not isinstance(out, dict):
        return "explain_ses: 'out' is invalid (expected dict)."

    lines = []
    def _p(x): lines.append(str(x))

    title = f"Structural Evidence Score for {name}" if name else "Structural Evidence Score"
    _p("\n" + " " * 72)
    _p(title)
    _p("-" * 72)

    status = out.get("status", None)
    mis = out.get("mis_size", None)
    if mis is not None:
        _p(f"Surrogate size (mis): {mis}")

    if status == "NO_REDUCTION":
        _p("Status: NO_REDUCTION (All objectives kept; reconstruction N/A).")
        _p("SES = N/A  |  F_real = N/A  |  F_null = N/A")
        return "\n".join(lines)

    ses = out.get("ses", None)
    F_real = out.get("F_real", None)
    F_null = out.get("F_null", None)
    r2_by_target = out.get("r2_real", None)

    if ses is None or F_real is None or F_null is None:
        _p("Status: UNDEFINED or missing metrics ('ses', 'F_real', 'F_null').")
        _p("SES = N/A  |  F_real = N/A  |  F_null = N/A")
        return "\n".join(lines)

    gap = F_real - F_null
    denom = max(1e-15, (1.0 - F_null))
    ses_recalc = np.clip(gap / denom, 0.0, 1.0)

    _p(f"SES = {ses:.4f}  (recalc = {ses_recalc:.4f})")
    _p(f"F_real = {F_real:.4f}  |  F_null = {F_null:.4f}  |  gap = {gap:.4f}")
    _p("Operational interpretation (Structural Evidence Score):")
    _p("  - SES≈1: surrogate reconstructs others very well, far above null.")
    _p("  - SES≈0: surrogate does not reconstruct better than null; suspicious reduction.")
    _p("  - intermediate values: some reconstruction, but there is relevant loss.")

    if ses >= 0.9:
        _p("Short read: strong SES (reduction tends to be safe for reconstruction).")
    elif ses >= 0.7:
        _p("Short read: moderate SES (reduction may work, but deserves checking).")
    else:
        _p("Short read: low SES (high risk of surrogate being too small).")

    if isinstance(r2_by_target, dict) and len(r2_by_target) > 0:
        items = list(r2_by_target.items())
        items_sorted = sorted(items, key=lambda kv: (-(np.inf) if kv[1] is None else kv[1]))
        items_sorted = [(k, (-np.inf if v is None else float(v))) for k, v in items_sorted]
        items_sorted = sorted(items_sorted, key=lambda kv: kv[1])

        worst = items_sorted[:min(top_k, len(items_sorted))]
        best = items_sorted[-min(top_k, len(items_sorted)):] if len(items_sorted) > 1 else []

        def _fmt_r2(v):
            if v is None or np.isneginf(v):
                return "N/A"
            return f"{v:.4f}"

        _p("\nWorst targets (lowest R² in test):")
        for k, v in worst:
            _p(f"  {k}: R² = {_fmt_r2(v)}")

        if best:
            _p("\nBest targets (highest R² in test):")
            for k, v in reversed(best):
                _p(f"  {k}: R² = {_fmt_r2(v)}")

        if show_all:
            _p("\nR² by target (all):")
            for k, v in items_sorted:
                _p(f"  {k}: R² = {_fmt_r2(v)}")
    else:
        _p("\nR² by target is not available.")

    return "\n".join(lines)
