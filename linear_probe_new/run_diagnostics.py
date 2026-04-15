#!/usr/bin/env python3
"""
run_diagnostics.py — Standalone IID Mass-Mean assumption diagnostics for CRE data.

Checks whether activation data satisfies the theoretical assumptions required
by the IID mass-mean probing method (Marks & Tegmark, 2024).

Tests: Homoscedasticity, Gaussianity, Confound correlation, Linear separability,
       Bidirectional balance.

Usage:
    python -u run_diagnostics.py \
        --act-dir /workspace/activations/meta-llama_Llama-3.1-8B-Instruct \
        --probe-dir ./results/llama-8b \
        --out-dir ./results/llama-8b/diagnostics_assumptions \
        --all-positions
"""

import argparse
import json
import gc
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# AssumptionDiagnostics class (inlined)
# ═══════════════════════════════════════════════════════════════════════════════

class AssumptionDiagnostics:
    def __init__(self, activations, labels, metadata=None):
        assert activations.ndim == 2
        assert labels.ndim == 1
        assert len(activations) == len(labels)

        self.X = activations.astype(np.float64)
        self.y = labels.astype(int)
        self.metadata = metadata or {}
        self.n, self.d = self.X.shape

        self.X_pos = self.X[self.y == 1]
        self.X_neg = self.X[self.y == 0]
        self.n_pos = len(self.X_pos)
        self.n_neg = len(self.X_neg)

        self.mu_pos = self.X_pos.mean(axis=0)
        self.mu_neg = self.X_neg.mean(axis=0)

        self.theta_mm = self.mu_pos - self.mu_neg
        norm = np.linalg.norm(self.theta_mm)
        self.theta_mm_unit = self.theta_mm / norm if norm > 0 else self.theta_mm

        self._results = {}

    def test_homoscedasticity(self, n_components=50):
        k = min(n_components, self.n_pos - 1, self.n_neg - 1, self.d)
        X_pos_c = self.X_pos - self.mu_pos
        X_neg_c = self.X_neg - self.mu_neg

        def top_k_eigen(Xc, k):
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            eigenvalues = (S[:k] ** 2) / (len(Xc) - 1)
            eigenvectors = Vt[:k].T
            return eigenvalues, eigenvectors

        evals_pos, evecs_pos = top_k_eigen(X_pos_c, k)
        evals_neg, evecs_neg = top_k_eigen(X_neg_c, k)

        eval_ratio = evals_pos[:k] / (evals_neg[:k] + 1e-10)
        eval_ratio_summary = {
            "mean_ratio": float(np.mean(eval_ratio)),
            "max_ratio": float(np.max(eval_ratio)),
            "min_ratio": float(np.min(eval_ratio)),
            "std_ratio": float(np.std(eval_ratio)),
        }

        overlap_svs = np.linalg.svd(evecs_pos.T @ evecs_neg, compute_uv=False)
        subspace_overlap = float(np.mean(overlap_svs[:k]))

        proj_pos = X_pos_c @ self.theta_mm_unit
        proj_neg = X_neg_c @ self.theta_mm_unit
        var_pos = float(np.var(proj_pos))
        var_neg = float(np.var(proj_neg))
        var_ratio_along_theta = var_pos / (var_neg + 1e-10)

        X_all_c = self.X - self.X.mean(axis=0)
        _, _, Vt_all = np.linalg.svd(X_all_c, full_matrices=False)
        P = Vt_all[:k].T
        Cov_pos_proj = (X_pos_c @ P).T @ (X_pos_c @ P) / (self.n_pos - 1)
        Cov_neg_proj = (X_neg_c @ P).T @ (X_neg_c @ P) / (self.n_neg - 1)
        Cov_pool_proj = (Cov_pos_proj * self.n_pos + Cov_neg_proj * self.n_neg) / self.n
        diff_frob = np.linalg.norm(Cov_pos_proj - Cov_neg_proj, "fro")
        pool_frob = np.linalg.norm(Cov_pool_proj, "fro")
        relative_frob = float(diff_frob / (pool_frob + 1e-10))

        issues = []
        if abs(var_ratio_along_theta - 1.0) > 0.5:
            issues.append(
                f"Variance ratio along theta_mm is {var_ratio_along_theta:.2f} "
                f"(ideal: ~1.0). Different spread along key direction."
            )
        if relative_frob > 0.3:
            issues.append(
                f"Relative Frobenius norm of cov_diff is {relative_frob:.2f} "
                f"(>0.3 suggests meaningful heteroscedasticity)."
            )
        if subspace_overlap < 0.7:
            issues.append(
                f"Subspace overlap is {subspace_overlap:.2f} — classes occupy "
                f"different subspaces (strong violation)."
            )

        result = {
            "eigenvalue_ratio": eval_ratio_summary,
            "subspace_overlap_mean_cos": subspace_overlap,
            "variance_ratio_along_theta_mm": var_ratio_along_theta,
            "var_follow_system_along_theta": var_pos,
            "var_follow_user_along_theta": var_neg,
            "relative_frobenius_norm": relative_frob,
            "n_components_used": k,
            "issues": issues,
            "passes": len(issues) == 0,
        }
        self._results["homoscedasticity"] = result
        return result

    def test_gaussianity(self):
        proj_pos = self.X_pos @ self.theta_mm_unit
        proj_neg = self.X_neg @ self.theta_mm_unit

        def compute_moments(proj):
            skew = float(np.mean(((proj - proj.mean()) / (proj.std() + 1e-10)) ** 3))
            kurt = float(np.mean(((proj - proj.mean()) / (proj.std() + 1e-10)) ** 4) - 3)
            info = {"skewness": skew, "excess_kurtosis": kurt}
            if len(proj) <= 5000:
                stat, p = sp_stats.shapiro(proj[:5000])
                info["shapiro_wilk_stat"] = float(stat)
                info["shapiro_wilk_p"] = float(p)
            return info

        pos_moments = compute_moments(proj_pos)
        neg_moments = compute_moments(proj_neg)

        issues = []
        for name, moments in [("follow_system", pos_moments), ("follow_user", neg_moments)]:
            if abs(moments["skewness"]) > 1.0:
                issues.append(f"{name}: skewness = {moments['skewness']:.2f} (|skew| > 1).")
            if abs(moments["excess_kurtosis"]) > 2.0:
                issues.append(f"{name}: excess kurtosis = {moments['excess_kurtosis']:.2f} (|kurt| > 2).")

        result = {
            "follow_system_projection": pos_moments,
            "follow_user_projection": neg_moments,
            "issues": issues,
            "passes": len(issues) == 0,
        }
        self._results["gaussianity"] = result
        return result

    def test_confound_correlation(self):
        if not self.metadata:
            return {"issues": ["No metadata provided."], "passes": None}

        results_per_field = {}
        issues = []

        for field_name, field_values in self.metadata.items():
            if len(field_values) != self.n:
                continue

            if field_values.dtype.kind in ("U", "S", "O"):
                unique_vals = np.unique(field_values)
                contingency = np.zeros((len(unique_vals), 2), dtype=int)
                for i, val in enumerate(unique_vals):
                    mask = field_values == val
                    contingency[i, 0] = np.sum(self.y[mask] == 0)
                    contingency[i, 1] = np.sum(self.y[mask] == 1)

                n_total = contingency.sum()
                chi2, p, dof, _ = sp_stats.chi2_contingency(contingency)
                min_dim = min(contingency.shape) - 1
                cramers_v = float(np.sqrt(chi2 / (n_total * max(min_dim, 1))))

                balance = {}
                for i, val in enumerate(unique_vals):
                    total = contingency[i].sum()
                    balance[str(val)] = {
                        "n": int(total),
                        "pct_follow_system": float(contingency[i, 1] / max(total, 1)),
                    }

                category_cos_sims = {}
                for val in unique_vals:
                    mask = field_values == val
                    if mask.sum() < 5:
                        continue
                    mu_cat = self.X[mask].mean(axis=0) - self.X[~mask].mean(axis=0)
                    cos = float(np.dot(mu_cat, self.theta_mm_unit) / (np.linalg.norm(mu_cat) + 1e-10))
                    category_cos_sims[str(val)] = cos

                field_result = {
                    "type": "categorical",
                    "chi2": float(chi2),
                    "p_value": float(p),
                    "cramers_v": cramers_v,
                    "class_balance_per_category": balance,
                    "cos_sim_category_vs_theta_mm": category_cos_sims,
                }

                if cramers_v > 0.3:
                    issues.append(
                        f"'{field_name}': Cramer's V = {cramers_v:.3f} (> 0.3) — "
                        f"theta_mm may be contaminated by '{field_name}'."
                    )
                for cat_val, cos in category_cos_sims.items():
                    if abs(cos) > 0.5:
                        issues.append(
                            f"'{field_name}={cat_val}' direction has cos_sim "
                            f"{cos:.3f} with theta_mm."
                        )
            else:
                corr, p = sp_stats.pointbiserialr(self.y, field_values.astype(float))
                field_result = {
                    "type": "numerical",
                    "point_biserial_r": float(corr),
                    "p_value": float(p),
                }
                if abs(corr) > 0.3:
                    issues.append(f"'{field_name}': r = {corr:.3f} (|r| > 0.3).")

            results_per_field[field_name] = field_result

        result = {"fields": results_per_field, "issues": issues, "passes": len(issues) == 0}
        self._results["confound_correlation"] = result
        return result

    def test_linearity(self, cv_folds=5):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr_scores = cross_val_score(lr, X_scaled, self.y, cv=cv, scoring="roc_auc")

        n_pca = min(100, self.d, self.n - 1)
        X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X_scaled)
        gbt = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        gbt_scores = cross_val_score(gbt, X_pca, self.y, cv=cv, scoring="roc_auc")

        lr_mean = float(np.mean(lr_scores))
        gbt_mean = float(np.mean(gbt_scores))
        gap = gbt_mean - lr_mean

        issues = []
        if gap > 0.05:
            issues.append(f"Nonlinear outperforms linear by {gap:.3f} AUC — boundary may be nonlinear.")

        result = {
            "linear_probe_auc": lr_mean,
            "linear_probe_std": float(np.std(lr_scores)),
            "nonlinear_probe_auc": gbt_mean,
            "nonlinear_probe_std": float(np.std(gbt_scores)),
            "gap": gap,
            "issues": issues,
            "passes": len(issues) == 0,
        }
        self._results["linearity"] = result
        return result

    def test_bidirectional_balance(self, direction_field="direction"):
        if direction_field not in self.metadata:
            return {"issues": [f"No '{direction_field}' field in metadata."], "passes": None}

        directions = self.metadata[direction_field]
        unique_dirs = np.unique(directions)
        balance_info = {}
        for d in unique_dirs:
            mask = directions == d
            n_d = int(mask.sum())
            pct_sys = float(self.y[mask].mean()) if n_d > 0 else None
            balance_info[str(d)] = {"n": n_d, "pct_follow_system": pct_sys}

        issues = []
        if len(unique_dirs) < 2:
            issues.append("Only one direction — dataset is not bidirectional.")
        for d_name, info in balance_info.items():
            if info["pct_follow_system"] is not None:
                imbalance = abs(info["pct_follow_system"] - 0.5)
                if imbalance > 0.3:
                    issues.append(f"Direction '{d_name}': {info['pct_follow_system']:.1%} follow system (imbalanced).")

        result = {"direction_balance": balance_info, "issues": issues, "passes": len(issues) == 0}
        self._results["bidirectional_balance"] = result
        return result

    def run_all(self, save_dir=None):
        print("=" * 70)
        print(f"  ASSUMPTION DIAGNOSTICS FOR IID MASS-MEAN PROBING")
        print(f"  n={self.n}, d={self.d}, n_sys={self.n_pos}, n_usr={self.n_neg}")
        print("=" * 70)

        tests = [
            ("1. Homoscedasticity", self.test_homoscedasticity),
            ("2. Gaussianity along theta_mm", self.test_gaussianity),
            ("3. Confound correlation", self.test_confound_correlation),
            ("4. Linear separability", self.test_linearity),
        ]
        if "direction" in self.metadata:
            tests.append(("5. Bidirectional balance", self.test_bidirectional_balance))

        all_results = {}
        for name, test_fn in tests:
            print(f"\n{'─' * 60}")
            print(f"  {name}")
            print(f"{'─' * 60}")
            try:
                result = test_fn()
                all_results[name] = result
                status = "PASS" if result.get("passes") else (
                    "ISSUES FOUND" if result.get("passes") is not None else "INFO ONLY"
                )
                print(f"  Status: {status}")
                if result.get("issues"):
                    for issue in result["issues"]:
                        print(f"  >> {issue}")
                else:
                    for k, v in result.items():
                        if k in ("issues", "passes", "note"):
                            continue
                        if isinstance(v, (int, float)):
                            print(f"  {k}: {v:.4f}")
                        elif isinstance(v, dict) and all(isinstance(vv, (int, float)) for vv in v.values()):
                            for kk, vv in v.items():
                                print(f"    {kk}: {vv:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[name] = {"error": str(e)}

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            self._plot_diagnostics(save_path)
            print(f"\n  Plots saved to {save_path}")

        print(f"\n{'=' * 70}")
        print("  SUMMARY")
        print(f"{'=' * 70}")
        n_pass = sum(1 for r in all_results.values() if isinstance(r, dict) and r.get("passes") is True)
        n_fail = sum(1 for r in all_results.values() if isinstance(r, dict) and r.get("passes") is False)
        print(f"  Passed: {n_pass}  |  Issues: {n_fail}")

        return all_results

    def _plot_diagnostics(self, save_path):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        proj_pos = self.X_pos @ self.theta_mm_unit
        proj_neg = self.X_neg @ self.theta_mm_unit
        axes[0].hist(proj_neg, bins=40, alpha=0.6, label="follow_user", color="#e74c3c", density=True)
        axes[0].hist(proj_pos, bins=40, alpha=0.6, label="follow_system", color="#2980b9", density=True)
        axes[0].set_xlabel("Projection onto theta_mm")
        axes[0].set_ylabel("Density")
        axes[0].set_title("1D Projections onto theta_mm")
        axes[0].legend(fontsize=8)

        X_pos_c = self.X_pos - self.mu_pos
        X_neg_c = self.X_neg - self.mu_neg
        k = min(30, self.n_pos - 1, self.n_neg - 1)
        _, S_pos, _ = np.linalg.svd(X_pos_c, full_matrices=False)
        _, S_neg, _ = np.linalg.svd(X_neg_c, full_matrices=False)
        evals_pos = (S_pos[:k] ** 2) / (self.n_pos - 1)
        evals_neg = (S_neg[:k] ** 2) / (self.n_neg - 1)
        axes[1].plot(range(k), evals_pos, "o-", label="follow_system", color="#2980b9", markersize=3)
        axes[1].plot(range(k), evals_neg, "s-", label="follow_user", color="#e74c3c", markersize=3)
        axes[1].set_xlabel("Component index")
        axes[1].set_ylabel("Eigenvalue")
        axes[1].set_title("Eigenvalue Spectra (top-30)")
        axes[1].legend(fontsize=8)
        axes[1].set_yscale("log")

        X_centered = self.X - self.X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        X_pca = X_centered @ Vt[:2].T
        axes[2].scatter(X_pca[self.y == 0, 0], X_pca[self.y == 0, 1],
                        c="#e74c3c", alpha=0.4, s=10, label="follow_user")
        axes[2].scatter(X_pca[self.y == 1, 0], X_pca[self.y == 1, 1],
                        c="#2980b9", alpha=0.4, s=10, label="follow_system")
        axes[2].set_xlabel("PC1")
        axes[2].set_ylabel("PC2")
        axes[2].set_title("PCA (top 2 components)")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path / "assumption_diagnostics.png", dpi=150, bbox_inches="tight")
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading for CRE pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(act_dir, probe_dir, method="iid_mm", position="last_prompt"):
    act_dir = Path(act_dir)
    probe_dir = Path(probe_dir)

    with open(probe_dir / "metrics" / "peak_layers.json") as f:
        peak_layers = json.load(f)
    peak_layer = peak_layers[method][position]

    act_path = None
    for f in act_dir.glob("activations_*.npy"):
        if f.stem.endswith(f"_{position}"):
            act_path = f
            break
    if act_path is None:
        raise FileNotFoundError(f"No activation file for position '{position}' in {act_dir}")

    X_full = np.load(act_path)
    X_peak = X_full[:, peak_layer, :]
    del X_full
    gc.collect()

    y_files = sorted(act_dir.glob("y_*.npy"))
    labels_file = act_dir / "labels.npz"
    if labels_file.exists():
        y = np.load(labels_file)["y"]
    elif y_files:
        y = np.load(y_files[0])
    else:
        raise FileNotFoundError(f"No label file in {act_dir}")

    if (act_dir / "metadata.csv").exists():
        df = pd.read_csv(act_dir / "metadata.csv")
    else:
        meta_files = sorted(act_dir.glob("meta_*.csv"))
        df = pd.read_csv(meta_files[0]) if meta_files else pd.DataFrame()

    return X_peak, y, df, peak_layer


def build_metadata_dict(df):
    metadata = {}
    for col in df.columns:
        if col in ("label", "y"):
            continue
        metadata[col] = df[col].values
    return metadata


def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IID Mass-Mean assumption diagnostics for CRE data.")
    parser.add_argument("--act-dir", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", default="iid_mm")
    parser.add_argument("--position", default="last_prompt")
    parser.add_argument("--all-positions", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(args.probe_dir) / "metrics" / "peak_layers.json") as f:
        peak_layers = json.load(f)

    positions = list(peak_layers.get(args.method, {}).keys()) if args.all_positions else [args.position]

    for position in positions:
        print(f"\n{'#' * 70}")
        print(f"  Position: {position} | Method: {args.method}")
        print(f"{'#' * 70}")

        X_peak, y, df, peak_layer = load_data(args.act_dir, args.probe_dir, args.method, position)
        X_scaled = StandardScaler().fit_transform(X_peak)
        metadata = build_metadata_dict(df)

        print(f"  Loaded: {X_scaled.shape[0]} samples, d={X_scaled.shape[1]}, peak_layer={peak_layer}")
        print(f"  Metadata fields: {list(metadata.keys())}")

        pos_out_dir = out_dir / f"{position}_L{peak_layer}"
        diag = AssumptionDiagnostics(X_scaled, y, metadata)
        results = diag.run_all(save_dir=str(pos_out_dir))

        report_path = pos_out_dir / "diagnostics_report.json"
        with open(report_path, "w") as f:
            json.dump(make_serializable(results), f, indent=2)
        print(f"\n  Report saved to {report_path}")

        del X_peak, X_scaled
        gc.collect()

    print(f"\n{'=' * 70}")
    print(f"  All diagnostics complete. Results in {out_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()