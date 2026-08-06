# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Legacy result objects returned by the MISDA static pipeline."""

import math

from ._metadata import __version__
from ._pareto import evaluate_pareto_consistency
from ._plotting import plot_custom_misda_graph
from ._reconstruction import calculate_ses, calculate_ses_nonlinear
from ._reporting import explain_ses
from ._statistics import calculate_spectral_entropy, describe_alpha_regime

class MISCandidate:
    """
    Represents a single Maximum Independent Set (MIS) solution found by the algorithm.
    Wrapper around the internal dictionary to provide object-oriented access.
    """
    def __init__(self, data: dict):
        self._data = data

    @property
    def indices(self):
        """List of column indices corresponding to the selected variables."""
        return self._data.get('mis_indices', [])

    @property
    def labels(self):
        """List of variable names (column headers) of the selected variables."""
        return self._data.get('mis_labels', [])

    @property
    def rank(self):
        """Rank of this solution (1 = Best)."""
        return self._data.get('rank', 999)

    @property
    def size(self):
        """Number of variables in this solution."""
        return len(self.indices)

    @property
    def total_correlation(self):
        """Sum of internal pair-wise correlations (lower is better)."""
        return self._data.get('total_correlation', float('inf'))

    @property
    def max_correlation(self):
        """Maximum single pair-wise correlation within this set (lower is better)."""
        return self._data.get('max_correlation', float('inf'))

    def __repr__(self):
        return f"<MISCandidate: {self.labels} (Size={self.size}, Rank={self.rank})>"


class MISDAResult:
    """
    Encapsulates the complete result of an MISDA analysis.
    Stores input parameters, diagnostic regimes, execution results (MIS),
    and validation metrics (SES).
    """
    def __init__(self, Y, caution, alpha_min, alpha_max, metrics, regime, alpha_exec, isda_res, name=None):
        self.Y = Y
        self.name = name
        self.caution = caution
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.metrics = metrics
        self.regime = regime
        self.alpha = alpha_exec # effectively used alpha
        self.isda_results = isda_res
        self.validation_metrics = {}

    def validate(self, check_linear=True, check_nonlinear=True, check_pareto=True):
        """
        Runs post-hoc validation metrics.
        Args:
            check_linear (bool): Run calculate_ses (Linear Regression).
            check_nonlinear (bool): Run calculate_ses_nonlinear (Random Forest).
            check_pareto (bool): Run evaluate_pareto_consistency.
        """
        # Linear SES
        if check_linear:
             if self.best_mis:
                best_ids = self.best_mis.indices
                Y_val = self.Y.values if hasattr(self.Y, "values") else self.Y
                self.validation_metrics['linear'] = calculate_ses(Y_val, best_ids, return_details=True)

        # Non-Linear SES
        if check_nonlinear:
             if self.best_mis:
                best_ids = self.best_mis.indices
                # Safeguard: prevent massive RF runs on huge data unless explicit
                if self.Y.shape[0] <= 10000:
                    try:
                        self.validation_metrics['nonlinear'] = calculate_ses_nonlinear(self.Y, best_ids, return_details=True)
                    except ImportError:
                        pass

        # Pareto
        if check_pareto:
             if self.best_mis:
                try:
                    p, r = evaluate_pareto_consistency(self)
                    self.validation_metrics['pareto'] = (p, r)
                except Exception:
                    pass

    @property
    def ses_results(self):
        """Backward compatibility for linear SES results."""
        return self.validation_metrics.get('linear')

    @property
    def correlations(self):
        """Returns the correlation report string from the ISDA execution."""
        return self.isda_results.get('corr_report')

    @property
    def min_compactness(self):
        """Returns the minimum component compactness found (worst internal correlation)."""
        return self.isda_results.get('min_component_compactness', 1.0)

    @property
    def homogeneity_ratio(self):
        """Returns the global homogeneity ratio (worst Min/Max within a component)."""
        stats = self.isda_results.get('homogeneity_stats', {})
        return stats.get('min_ratio', 1.0)

    @property
    def mis_sets(self):
        """
        Returns a list of all MISCandidate objects found, sorted by rank.
        """
        raw_sets = self.isda_results.get('mis_ranked', [])
        return [MISCandidate(d) for d in raw_sets]

    @property
    def best_mis(self):
        """Returns the top-ranked MISCandidate or None."""
        if self.isda_results.get('mis_ranked'):
            return MISCandidate(self.isda_results['mis_ranked'][0])
        return None

    @property
    def best_mis_indices(self):
        """Returns the list of indices of the best MIS."""
        mis = self.best_mis
        return mis.indices if mis else []

    @property
    def best_mis_labels(self):
        """Returns the list of labels of the best MIS."""
        mis = self.best_mis
        return mis.labels if mis else []

    @property
    def ranked_mis_sets(self):
        """
        Returns a dictionary mapping rank (int) -> list of MISCandidate objects.
        """
        raw_groups = self.isda_results.get('rank_groups', {})
        return {r: [MISCandidate(d) for d in l] for r, l in raw_groups.items()}

    def get_mis_by_rank(self, rank):
        """
        Returns the list of MISCandidate objects for the specified rank.
        Returns empty list if rank not found.
        """
        return self.ranked_mis_sets.get(rank, [])

    @property
    def reduction_applied(self):
        """Boolean: True if dim(MIS) < dim(Y)."""
        mis = self.best_mis
        if mis:
            return mis.size < self.Y.shape[1]
        return False

    # --- Flattened Metrics ---
    @property
    def separation_score(self):
        """The Separation Score (S). Higher is better."""
        return float(self.metrics.get("S", float('nan')))

    @property
    def normalized_separation_score(self):
        """Normalized S-score (S_norm). Closer to 1.0 is better."""
        return float(self.metrics.get("S_norm", float('nan')))

    # --- Flattened Validation ---
    @property
    def ses_nonlinear_results(self):
        """Detailed Non-Linear SES dict result. None if not run."""
        val = self.validation_metrics.get('nonlinear')
        return val if isinstance(val, dict) else None

    @property
    def ses_nonlinear(self):
        """Non-Linear SES scalar metric score (float or None)."""
        val = self.validation_metrics.get('nonlinear')
        if isinstance(val, dict):
            return val.get('ses')
        return val

    @property
    def pareto_precision(self):
        """Pareto Precision (Safety). None if not run."""
        p_r = self.validation_metrics.get('pareto')
        return p_r[0] if p_r else None

    @property
    def pareto_recall(self):
        """Pareto Recall (Coverage). None if not run."""
        p_r = self.validation_metrics.get('pareto')
        return p_r[1] if p_r else None

    def to_pandas(self):
        """
        Exports all found independent sets to a pandas DataFrame.
        Columns: ['rank', 'size', 'max_corr', 'total_corr', 'labels', 'indices']
        """
        import pandas as pd
        data = []
        for m in self.mis_sets:
            data.append({
                'rank': m.rank,
                'size': m.size,
                'max_corr': m.max_correlation,
                'total_corr': m.total_correlation,
                'labels': m.labels,
                'indices': m.indices
            })
        if not data:
            return pd.DataFrame(columns=['rank', 'size', 'max_corr', 'total_corr', 'labels', 'indices'])
        return pd.DataFrame(data)

    def __repr__(self):
        n_start = self.Y.shape[1]
        n_end = self.best_mis.size if self.best_mis else "?"
        name_str = f"'{self.name}'" if self.name else "Untitled"
        return f"<MISDAResult: {name_str} (Dim {n_start}->{n_end}, Rank={self.best_mis.rank if self.best_mis else '?'})>"

    @property
    def diagnosis(self):
        """Returns a short diagnostic string based on Fidelity and Homogeneity."""
        if not self.reduction_applied:
            return "Valid (No Reduction Required)"

        f = None
        status = None
        if self.ses_results and isinstance(self.ses_results, dict):
            f = self.ses_results.get('F_real', None)
            status = self.ses_results.get('status', None)

        if status == "NO_REDUCTION":
            return "Valid (No Reduction Required)"

        h = self.homogeneity_ratio

        if f is None or math.isnan(f):
            return "Unvalidated (Missing SES)"

        comps = self.isda_results.get('components_labels', [])
        num_comps = len(comps)

        # Strict clique completeness check (min_compactness >= alpha)
        is_true_clique = (self.min_compactness >= self.alpha)

        # Heuristic Decision Tree
        if f >= 0.9 and h >= 0.8:
            if num_comps > 1:
                return "Ideal (Disjoint Cliques)" if is_true_clique else "Ideal (Multiple Components)"
            return "Ideal (Clique)" if is_true_clique else "Good (Robust)"
        if f >= 0.9 and h < 0.2:
             return "Entangled (Mixed)"
        if f >= 0.9:
             return "Good (Robust)"

        if f < 0.8 and h >= 0.6:
             return "Drift (Chain)"

        if f < 0.6 and h < 0.6:
             return "Fragmented (Bridge)"

        return "Ambiguous/Warn"

    @property
    def validation_status(self):
        """Returns string describing what has been validated."""
        validated = []
        if 'linear' in self.validation_metrics: validated.append("Linear")
        if 'nonlinear' in self.validation_metrics: validated.append("Non-Linear")
        if 'pareto' in self.validation_metrics: validated.append("Pareto")
        return ", ".join(validated) if validated else "None"

    def summary(self):
        """Returns a textual summary of the analysis."""
        lines = []
        lines.append("\n" + "" * 70)
        title = f"MISDA Analysis Summary: {self.name}" if self.name else "MISDA Analysis Summary"
        lines.append(title)
        lines.append("-" * 70)

        # Ground Truth / Inputs
        lines.append(f"Input: [N={self.Y.shape[0]}, M={self.Y.shape[1]}]")
        lines.append(f"Caution: {self.caution}")

        # Diagnosis
        lines.append("\n--- 1. Diagnosis ---")
        lines.append(describe_alpha_regime(self.metrics))
        lines.append(f"Regime: {self.regime.name}")
        lines.append(f"Validation: {self.validation_status}")

        # Decision
        lines.append("\n--- 2. Decision ---")
        if self.reduction_applied:
            lines.append("Action: Reduction APPLIED")
        else:
            lines.append("Action: Full Dimension Kept (No Reduction)")
        lines.append(f"Alpha Used: {self.alpha:.6g} (Range: [{self.alpha_min:.6g}, {self.alpha_max:.6g}])")

        # Results
        lines.append("\n--- 3. Results ---")
        mis = self.best_mis
        if mis:
             lines.append(f"Best MIS Size: {mis.size}")
             lines.append(f"Best MIS Labels: {mis.labels}")
        else:
             lines.append("No independent set found (or execution failed).")

        # Quality
        lines.append("\n--- 4. Quality ---")
        ratio = self.homogeneity_ratio
        diag = self.diagnosis

        def _fmt_ratio(r):
            if math.isnan(r): return "N/A"
            return f"{r:.4f}"

        lines.append(f"Homogeneity Ratio: {_fmt_ratio(ratio)}")
        lines.append(f"Auto-Diagnosis: {diag}")

        if not math.isnan(ratio) and ratio < 0.6:
            lines.append("WARNING: Low homogeneity ratio (< 0.6). Possible over-reduction due to transitive chains or bridges.")
        else:
            lines.append("Status: OK (Components are internally homogeneous)")

        # Global Complexity Warning (Sphere Paradox)
        if self.reduction_applied:
            se_norm = calculate_spectral_entropy(self.Y)
            if se_norm > 0.75:
                # User-requested warning
                lines.append("WARNING: High global complexity detected (SE={:.2f}) despite aggressive reduction. Suspected Latent Conflict (Sphere-like topology).".format(se_norm))

        # SES (Linear)
        if 'linear' in self.validation_metrics:
             lines.append("\n--- 5. Validation (SES - Linear) ---")
             lines.append(explain_ses(self.validation_metrics['linear'], name=self.name))

        # SES (Non-Linear)
        if 'nonlinear' in self.validation_metrics:
             nl_res = self.validation_metrics['nonlinear']
             if isinstance(nl_res, dict):
                 ses_nl = nl_res.get('ses')
                 f_real_nl = nl_res.get('F_real')
                 nl_str = f"{ses_nl:.4f}" if ses_nl is not None else "N/A"
                 f_str = f"{f_real_nl:.4f}" if f_real_nl is not None else "N/A"
                 lines.append(f"Non-Linear SES (RF): {nl_str} (F_real = {f_str})")
             else:
                 nl_str = f"{nl_res:.4f}" if nl_res is not None else "N/A"
                 lines.append(f"Non-Linear SES (RF): {nl_str}")

        # Pareto Consistency
        if 'pareto' in self.validation_metrics:
             lines.append("\n--- 6. Pareto Consistency ---")
             prec, rec = self.validation_metrics['pareto']
             lines.append(f"Precision (Safety):   {prec:.4f}  (Prob. that Surrogate Optimum is True Optimum)")
             lines.append(f"Recall    (Coverage): {rec:.4f}  (Prob. that True Optimum is retained)")
             if prec < 1.0:
                 lines.append("WARN: Surrogate introduces false optima (Precision < 1.0).")
             if rec < 0.8:
                 lines.append("WARN: Surrogate misses significant portion of Pareto front (Recall < 0.8).")

        return "\n".join(lines)


    def report(self, top_k=5):
        """
        Returns a comprehensive technical report of the analysis.
        Combines the standard summary with deep inspection of internal state.

        Args:
            top_k (int): Number of candidates to show per rank (default: 5).
        """
        # Start with standard summary
        base_report = self.summary()

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("                    DETAILED INSPECTION REPORT")
        lines.append("=" * 70)

        # 1. Statistical Foundation
        lines.append("\n--- A. Statistical Foundation ---")
        lines.append(f"MISDA Version: {__version__}")
        lines.append(f"Sample Size (N): {self.isda_results.get('N')} | Objectives (M): {self.isda_results.get('M')}")
        lines.append(f"Fisher Transform Error (sigma_z): {self.isda_results.get('sigma_z', 0):.6f}")
        lines.append(f"Alpha Range: [min={self.alpha_min:.3g} (Signal), max={self.alpha_max:.3g} (Noise)]")
        lines.append(f"Caution setting: {self.caution:.2f}")
        lines.append(f"Effective Alpha: {self.alpha:.6g} (based on regime={self.regime.name})")
        lines.append(f"Critical Z-score: {self.isda_results.get('z_crit', 0):.4f}")

        # 2. Graph & Component Details
        lines.append("\n--- B. Graph Topology Details ---")
        comps = self.isda_results.get('components_labels', [])
        homog_stats = self.isda_results.get('homogeneity_stats', {})

        lines.append(f"Connected Components: {len(comps)}")
        for i, c in enumerate(comps):
            # Try to get specific stats for this component if available
            c_stat = homog_stats.get('details', {}).get(i, {})
            min_r = c_stat.get('min_r', float('nan'))
            max_r = c_stat.get('max_r', float('nan'))
            ratio = c_stat.get('ratio', float('nan'))

            def _fmt_nan(v, default="N/A"):
                if math.isnan(v): return default
                return f"{v:.4f}"

            status = "Tight" if (not math.isnan(ratio) and ratio > 0.8) else "Loose"
            lines.append(f"  C{i+1}: {c}")
            lines.append(f"      Internal Correlation: [{_fmt_nan(min_r)} ... {_fmt_nan(max_r)}] | Homogeneity: {_fmt_nan(ratio)} ({status})")

        # 3. Solution Space (All Candidates)
        lines.append("\n--- C. Solution Space (All Candidates) ---")
        rank_groups = self.ranked_mis_sets

        if not rank_groups:
             lines.append("  No solutions found.")

        for r in sorted(rank_groups.keys()):
            cands = rank_groups[r]
            n_cands = len(cands)
            lines.append(f"  Rank {r} ({n_cands} candidates):")

            # Smart Truncation
            show_cands = cands[:top_k]
            for c in show_cands:
                lines.append(f"    - {c.labels} (Size={c.size})")
                lines.append(f"      Criteria: TotalCorr={c.total_correlation:.4f} | MaxCorr={c.max_correlation:.4f}")

            if n_cands > top_k:
                lines.append(f"      ... (+ {n_cands - top_k} more candidates. Use `res.to_pandas()` to view all.)")

        # 4. Extended Verification
        lines.append("\n--- D. Verification Details ---")
        if self.ses_results:
            ses = self.ses_results
            f_real = ses.get('F_real')
            f_null = ses.get('F_null')
            s_val = ses.get('ses')

            def _fmt_v(val):
                return f"{val:.4f}" if val is not None else "N/A"

            lines.append("  Linear SES Breakdown:")
            lines.append(f"    Fidelity (Real): {_fmt_v(f_real)}")
            lines.append(f"    Fidelity (Null): {_fmt_v(f_null)}")
            lines.append(f"    Raw SES Score:   {_fmt_v(s_val)}")
        else:
            lines.append("  Linear SES: Not run (or failed).")

        return base_report + "\n".join(lines)

    def plot(self, show=True):
        """
        Plots the ISDA graph.

        Args:
            show (bool): If True, calls plt.show() to display the plot immediately.

        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        ret = plot_custom_misda_graph(
            self.isda_results,
            title=f"{self.name or 'MISDA'} — alpha={self.alpha:.3g} — regime={self.regime.name}",
            show_removed=False
        )
        fig = ret['fig']

        if show:
            import matplotlib.pyplot as plt
            plt.show()

        return fig
