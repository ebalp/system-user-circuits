#!/usr/bin/env python3
"""
run_probing.py — Unified probing & analysis pipeline for conflict resolution.

Reads activations from extract_activations.py output, runs all analyses
(×3 direction methods × N positions × all layers), saves figures + metrics + directions.

Usage:
    python run_probing.py --act-dir ./activations/meta-llama_Llama-3.1-8B-Instruct
    python run_probing.py --act-dir ./activations/meta-llama_Llama-3.1-8B-Instruct --skip-conceptors
    python run_probing.py --act-dir ./activations/meta-llama_Llama-3.1-8B-Instruct --positions last_prompt first_generated
"""

import argparse
import json
import gc
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, confusion_matrix,
)
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, f_oneway
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProbingConfig:
    act_dir: Path
    out_dir: Path | None = None     # defaults to results/{model_name}
    positions: list[str] | None = None  # None = use all from extraction
    n_cv_folds: int = 5
    test_size: float = 0.2
    bal_per_class: int = 2500       # max samples per class for LR subset
    n_permutations: int = 3         # kept small for speed; 3 is enough for chance-level check
    conceptor_alpha: float = 0.1    # aperture for conceptor computation
    skip_conceptors: bool = False
    device: str = "cuda"
    save_formats: list[str] = None  # defaults to ["png", "pdf"]

    def __post_init__(self):
        if self.save_formats is None:
            self.save_formats = ["png", "pdf"]


DIRECTION_METHODS = ["dim", "iid_mm", "lr"]
METHOD_LABELS = {"dim": "Diff-in-Mean", "iid_mm": "IID Mass-Mean", "lr": "Logistic Reg."}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _detect_activation_files(act_dir: Path):
    """
    Auto-detect activation file naming convention.
    Supports two layouts:
      1. Script output:   activations_{pos}.npy
      2. Notebook output: activations_{model_stem}_{pos}.npy
    Returns: dict mapping position_name -> file path
    """
    all_npy = sorted(act_dir.glob("activations_*.npy"))
    if not all_npy:
        raise FileNotFoundError(f"No activations_*.npy files found in {act_dir}")

    KNOWN_POSITIONS = [
        "last_prompt", "last_system", "last_user",
        "first_generated", "mid_generated", "last_generated",
    ]

    found = {}
    for path in all_npy:
        name = path.stem  # e.g. "activations_meta-llama_Llama-3.1-8B-Instruct_last_prompt"
        for pos in KNOWN_POSITIONS:
            if name.endswith(f"_{pos}"):
                found[pos] = path
                break
        else:
            # Try simple format: activations_{pos}
            simple_pos = name.replace("activations_", "")
            if simple_pos in KNOWN_POSITIONS:
                found[simple_pos] = path

    if not found:
        raise FileNotFoundError(
            f"Could not match any activation files to known positions.\n"
            f"Files found: {[p.name for p in all_npy]}"
        )

    return found


def _detect_labels(act_dir: Path):
    """
    Auto-detect labels. Supports:
      1. labels.npz (script output)
      2. y_{model_stem}.npy (notebook output)
    """
    # Try script format
    if (act_dir / "labels.npz").exists():
        data = np.load(act_dir / "labels.npz")
        return data["y"]

    # Try notebook format: y_*.npy
    y_files = sorted(act_dir.glob("y_*.npy"))
    if y_files:
        return np.load(y_files[0])

    raise FileNotFoundError(f"No label file found in {act_dir}. Expected labels.npz or y_*.npy")


def _detect_metadata(act_dir: Path):
    """
    Auto-detect metadata CSV. Supports:
      1. metadata.csv (script output)
      2. meta_*.csv (notebook output)
    """
    if (act_dir / "metadata.csv").exists():
        return pd.read_csv(act_dir / "metadata.csv")

    meta_files = sorted(act_dir.glob("meta_*.csv"))
    if meta_files:
        return pd.read_csv(meta_files[0])

    raise FileNotFoundError(f"No metadata file found in {act_dir}. Expected metadata.csv or meta_*.csv")


def _detect_or_create_config(act_dir: Path, activations: dict, y: np.ndarray, positions: list):
    """
    Load extraction_config.json if it exists, otherwise infer from data.
    """
    config_path = act_dir / "extraction_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    # Infer from activation shape
    sample_pos = positions[0]
    shape = activations[sample_pos].shape  # (n_samples, n_layers, d_model)

    config = {
        "model": act_dir.name,
        "n_samples": int(shape[0]),
        "n_layers": int(shape[1]),
        "d_model": int(shape[2]),
        "token_positions": positions,
        "scr": float(y.mean()),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
    }

    # Save for future runs
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Generated extraction_config.json from data shapes")

    return config


def load_activations(cfg: ProbingConfig):
    """
    Load activations, labels, metadata.
    Auto-detects both script output and notebook output file formats.
    """
    act_dir = cfg.act_dir

    # Auto-detect activation files
    pos_to_path = _detect_activation_files(act_dir)
    available_positions = sorted(pos_to_path.keys())
    print(f"  Detected positions: {available_positions}")

    # Filter to requested positions
    if cfg.positions:
        positions = [p for p in cfg.positions if p in available_positions]
        if not positions:
            raise ValueError(
                f"None of requested positions {cfg.positions} found. "
                f"Available: {available_positions}"
            )
    else:
        positions = available_positions

    # Load activations
    activations = {}
    for pos in positions:
        activations[pos] = np.load(pos_to_path[pos])
        print(f"  Loaded {pos}: {activations[pos].shape}  ({pos_to_path[pos].name})")

    # Load labels
    y = _detect_labels(act_dir)
    print(f"  Labels: {len(y)} samples")

    # Load metadata
    df = _detect_metadata(act_dir)
    print(f"  Metadata: {len(df)} rows, cols={list(df.columns)}")

    # Load or create config
    ext_config = _detect_or_create_config(act_dir, activations, y, positions)

    n_layers = ext_config["n_layers"]
    d_model = ext_config["d_model"]

    print(f"  n_samples={len(y)}, n_layers={n_layers}, d_model={d_model}")
    print(f"  SCR={y.mean():.3f}, positions={positions}")

    return activations, y, df, positions, n_layers, d_model, ext_config


def build_lr_subset(y, bal_per_class, rng_seed=99):
    """Build balanced subset for LR probing (speed)."""
    bal_n = min(y.sum(), (y == 0).sum(), bal_per_class)
    rng = np.random.RandomState(rng_seed)
    pos_idx = rng.choice(np.where(y == 1)[0], size=bal_n, replace=False)
    neg_idx = rng.choice(np.where(y == 0)[0], size=bal_n, replace=False)
    lr_idx = np.sort(np.concatenate([pos_idx, neg_idx]))
    return lr_idx


# ─────────────────────────────────────────────────────────────────────────────
# Direction methods (GPU-accelerated)
# ─────────────────────────────────────────────────────────────────────────────

def diff_in_mean_probe_gpu(X, y, scale=True, device="cuda"):
    """
    Diff-in-Mean direction finding. GPU-accelerated.
    X: (n_samples, n_layers, d_model)
    Returns: DataFrame of bal_acc per layer, list of direction vectors per layer.
    """
    n_layers = X.shape[1]
    rows = []
    directions = []
    y_t = torch.tensor(y, dtype=torch.bool, device=device)

    for layer in range(n_layers):
        X_t = torch.as_tensor(X[:, layer, :], device=device)
        if scale:
            mu = X_t.mean(0)
            std = X_t.std(0).clamp(min=1e-12)
            X_t = (X_t - mu) / std

        d = X_t[y_t].mean(0) - X_t[~y_t].mean(0)
        d_unit = d / (d.norm() + 1e-12)
        proj = X_t @ d_unit

        threshold = (proj[y_t].mean() + proj[~y_t].mean()) / 2
        preds = (proj > threshold).int()
        tp = preds[y_t].float().mean()
        tn = (1 - preds[~y_t].float()).mean()
        acc = ((tp + tn) / 2).item()

        rows.append({"layer": layer, "bal_acc": acc})
        directions.append(d_unit.cpu().numpy())

    return pd.DataFrame(rows), directions


def iid_mass_mean_probe_gpu(X, y, scale=True, device="cuda"):
    """
    IID Mass-Mean (Fisher-like) direction finding. GPU-accelerated.
    linalg.solve goes to CPU to avoid CUDA numerical instability.
    """
    n_layers = X.shape[1]
    rows = []
    directions = []
    y_t = torch.tensor(y, dtype=torch.bool, device=device)

    for layer in range(n_layers):
        X_t = torch.as_tensor(X[:, layer, :], device=device)
        if scale:
            mu = X_t.mean(0)
            std = X_t.std(0).clamp(min=1e-12)
            X_t = (X_t - mu) / std

        mu_pos = X_t[y_t].mean(0)
        mu_neg = X_t[~y_t].mean(0)
        theta_mm = mu_pos - mu_neg

        X_centered = X_t.clone()
        X_centered[y_t] -= mu_pos
        X_centered[~y_t] -= mu_neg

        Sigma = X_centered.T @ X_centered / (X_centered.shape[0] - 1)
        Sigma[torch.arange(Sigma.shape[0]), torch.arange(Sigma.shape[0])] += 1e-6

        # Solve on CPU — CUDA linalg.solve can be numerically unstable
        # on certain driver versions. This is intentional.
        eff = torch.linalg.solve(Sigma.cpu(), theta_mm.cpu()).to(device)
        proj = X_t @ eff

        threshold = (proj[y_t].mean() + proj[~y_t].mean()) / 2
        preds = (proj > threshold).int()
        tp = preds[y_t].float().mean()
        tn = (1 - preds[~y_t].float()).mean()
        acc = ((tp + tn) / 2).item()

        eff_unit = eff / (eff.norm() + 1e-12)
        rows.append({"layer": layer, "bal_acc": acc})
        directions.append(eff_unit.cpu().numpy())

    return pd.DataFrame(rows), directions


def lr_fit_gpu(X_np, y_np, C=1.0, max_iter=100, device="cuda"):
    """
    GPU-accelerated logistic regression via PyTorch LBFGS.
    Avoids sklearn ConvergenceWarning issues.
    Returns weight vector (on CPU, as numpy).
    """
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    w = torch.zeros(X.shape[1], device=device, requires_grad=True)
    reg = 1.0 / C

    optimizer = torch.optim.LBFGS([w], max_iter=20, line_search_fn="strong_wolfe")

    for _ in range(max_iter):
        def closure():
            optimizer.zero_grad()
            logits = X @ w
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, reduction="mean"
            )
            loss = loss + 0.5 * reg * (w * w).sum() / X.shape[0]
            loss.backward()
            return loss
        optimizer.step(closure)

    return w.detach().cpu().numpy()


def lr_probe_cv(X, y, n_folds=5, scale=True):
    """
    LR probe with cross-validation. Uses sklearn for CV logic,
    but lr_fit_gpu for direction extraction.
    Returns: DataFrame of (layer, mean_acc, std_acc), list of AUROC per layer,
             list of direction vectors per layer.
    """
    n_layers = X.shape[1]
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    rows = []
    aurocs = []
    directions = []

    for layer in tqdm(range(n_layers), desc="LR Probe", leave=False):
        X_layer = X[:, layer, :]

        # CV for balanced accuracy
        if scale:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000, C=1.0, solver="lbfgs",
                    class_weight="balanced", fit_intercept=False,
                )),
            ])
        else:
            pipe = LogisticRegression(
                max_iter=2000, C=1.0, solver="lbfgs",
                class_weight="balanced", fit_intercept=False,
            )
        scores = cross_val_score(
            pipe, X_layer, y, cv=cv, scoring="balanced_accuracy"
        )
        rows.append({"layer": layer, "bal_acc": scores.mean(), "std": scores.std()})

        # AUROC on held-out split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_layer, y, test_size=0.2, stratify=y, random_state=42
        )
        if scale:
            pipe_auc = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000, C=1.0, solver="lbfgs",
                    class_weight="balanced", fit_intercept=False,
                )),
            ])
        else:
            pipe_auc = LogisticRegression(
                max_iter=2000, C=1.0, solver="lbfgs",
                class_weight="balanced", fit_intercept=False,
            )
        pipe_auc.fit(X_tr, y_tr)
        proba = pipe_auc.predict_proba(X_te)[:, 1]
        aurocs.append(roc_auc_score(y_te, proba))

        # Direction via GPU LBFGS (on full data, scaled)
        X_dir = X_layer.copy()
        if scale:
            X_dir = StandardScaler().fit_transform(X_dir)
        w = lr_fit_gpu(X_dir, y)
        w_unit = w / (np.linalg.norm(w) + 1e-12)
        directions.append(w_unit)

    return pd.DataFrame(rows), aurocs, directions


# ─────────────────────────────────────────────────────────────────────────────
# Controls
# ─────────────────────────────────────────────────────────────────────────────

def metadata_only_control(df, y, n_folds=5, position_maps=None):
    """
    Train a classifier on metadata features only (no activations).
    Tests whether surface features predict the outcome.
    """
    all_constraint_types = sorted(df["constraint_type"].unique())
    all_sys_styles = sorted(df["system_style"].unique()) if "system_style" in df.columns else []
    all_usr_styles = sorted(df["user_style"].unique()) if "user_style" in df.columns else []

    def one_hot(val, categories):
        return [int(val == c) for c in categories]

    features = []
    for i, row in df.iterrows():
        f = []
        f.extend(one_hot(row["constraint_type"], all_constraint_types))
        if all_sys_styles:
            f.extend(one_hot(row["system_style"], all_sys_styles))
        if all_usr_styles:
            f.extend(one_hot(row["user_style"], all_usr_styles))
        if "direction" in df.columns:
            f.append(1 if row["direction"] == "forward" else 0)
        features.append(f)

    X_meta = np.array(features, dtype=np.float32)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs",
            class_weight="balanced", fit_intercept=True,
        )),
    ])
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_meta, y, cv=cv, scoring="balanced_accuracy")
    return scores.mean(), scores.std()


def permutation_control(X_peak, y, n_permutations=3, n_folds=5):
    """
    Permuted-label control at peak layer. Verifies probe is above chance.
    Kept to few permutations (3-5) since we only need to confirm ≈0.50.
    """
    rng = np.random.default_rng(0)
    perm_accs = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, C=1.0, solver="lbfgs",
                class_weight="balanced", fit_intercept=False,
            )),
        ])
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X_peak, y_perm, cv=cv, scoring="balanced_accuracy")
        perm_accs.append(scores.mean())
    return np.mean(perm_accs), np.std(perm_accs)


def condition_level_pca(activations_pos, y, df, peak_layer):
    """
    Condition-level PCA: group by (constraint_type, system_style, user_style),
    compute mean activations, run PCA, check correlation with SCR.
    """
    condition_cols = ["constraint_type"]
    if "system_style" in df.columns:
        condition_cols.append("system_style")
    if "user_style" in df.columns:
        condition_cols.append("user_style")

    X_peak = activations_pos[:, peak_layer, :]
    X_scaled = StandardScaler().fit_transform(X_peak)

    groups = df.groupby(condition_cols)
    cond_means = []
    cond_scrs = []
    cond_labels = []
    for name, idx_group in groups:
        indices = idx_group.index.values
        if len(indices) < 5:
            continue
        cond_means.append(X_scaled[indices].mean(axis=0))
        cond_scrs.append(y[indices].mean())
        cond_labels.append(name)

    X_cond = np.array(cond_means)
    cond_scr = np.array(cond_scrs)

    pca = PCA(n_components=min(5, len(X_cond) - 1), random_state=42)
    X_pca = pca.fit_transform(X_cond)

    results = {
        "n_conditions": len(X_cond),
        "variance_explained": pca.explained_variance_ratio_.tolist(),
    }

    for i in range(min(3, X_pca.shape[1])):
        r, p = pearsonr(X_pca[:, i], cond_scr)
        results[f"pc{i+1}_scr_corr"] = r
        results[f"pc{i+1}_scr_pval"] = p

    return results, X_pca, cond_scr, cond_labels


# ─────────────────────────────────────────────────────────────────────────────
# Direction Analysis
# ─────────────────────────────────────────────────────────────────────────────

def cross_method_cosine_sim(directions_by_method, n_layers):
    """Cosine similarity between direction methods at each layer."""
    methods = list(directions_by_method.keys())
    results = {}
    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            key = f"{m1}_vs_{m2}"
            sims = []
            for layer in range(n_layers):
                d1 = directions_by_method[m1][layer]
                d2 = directions_by_method[m2][layer]
                sim = abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12))
                sims.append(sim)
            results[key] = sims
    return results


def cross_position_transfer(activations, y, directions, peak_layer, positions, test_size=0.2):
    """
    Train direction at one position, test balanced accuracy at another.
    Returns transfer matrix (positions × positions).
    """
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=test_size, stratify=y, random_state=42
    )
    matrix = np.zeros((len(positions), len(positions)))
    for i, train_pos in enumerate(positions):
        d = directions[train_pos][peak_layer]
        for j, test_pos in enumerate(positions):
            X_te = activations[test_pos][test_idx, peak_layer, :]
            X_te_s = StandardScaler().fit_transform(X_te)
            proj = X_te_s @ d
            threshold = np.median(proj)
            preds = (proj > threshold).astype(int)
            matrix[i, j] = balanced_accuracy_score(y[test_idx], preds)
    return pd.DataFrame(matrix, index=positions, columns=positions)


def cross_category_transfer(activations_pos, y, df, directions, peak_layer):
    """
    Train on one constraint category, test on another.
    Categories: content, format, language, style, other.
    """
    constraint_categories = {}
    for ct in sorted(df["constraint_type"].unique()):
        ct_lower = ct.lower()
        if any(kw in ct_lower for kw in ["json", "bullet", "numbered", "section", "prose",
                                          "paragraph", "line", "caps", "markdown"]):
            constraint_categories[ct] = "format"
        elif any(kw in ct_lower for kw in ["english", "spanish", "french", "language",
                                            "loanword", "leetspeak", "vowel"]):
            constraint_categories[ct] = "language"
        elif any(kw in ct_lower for kw in ["keyword", "word", "alliteration", "repetition",
                                            "sentence", "starting", "connector", "number"]):
            constraint_categories[ct] = "content"
        elif any(kw in ct_lower for kw in ["tone", "voice", "hedging", "formal", "casual",
                                            "active", "passive", "tense"]):
            constraint_categories[ct] = "style"
        else:
            constraint_categories[ct] = "other"

    df_cat = df.copy()
    df_cat["category"] = df_cat["constraint_type"].map(constraint_categories)
    categories = sorted(df_cat["category"].unique())

    X_peak = activations_pos[:, peak_layer, :]
    X_scaled = StandardScaler().fit_transform(X_peak)

    matrix = np.zeros((len(categories), len(categories)))
    for i, train_cat in enumerate(categories):
        train_mask = df_cat["category"] == train_cat
        X_tr = X_scaled[train_mask]
        y_tr = y[train_mask]
        if y_tr.sum() < 2 or (1 - y_tr).sum() < 2:
            matrix[i, :] = np.nan
            continue
        d = X_tr[y_tr == 1].mean(0) - X_tr[y_tr == 0].mean(0)
        d_unit = d / (np.linalg.norm(d) + 1e-12)

        for j, test_cat in enumerate(categories):
            test_mask = df_cat["category"] == test_cat
            X_te = X_scaled[test_mask]
            y_te = y[test_mask]
            if y_te.sum() < 1 or (1 - y_te).sum() < 1:
                matrix[i, j] = np.nan
                continue
            proj = X_te @ d_unit
            threshold = (proj[y_te == 1].mean() + proj[y_te == 0].mean()) / 2
            preds = (proj > threshold).astype(int)
            matrix[i, j] = balanced_accuracy_score(y_te, preds)

    return pd.DataFrame(matrix, index=categories, columns=categories), categories


def projection_gap_per_layer(activations_pos, y, directions, n_layers, device="cuda"):
    """Mean projection gap (system − user) per layer using given directions."""
    y_t = torch.tensor(y, dtype=torch.bool, device=device)
    gaps = []
    for layer in range(n_layers):
        X_t = torch.as_tensor(activations_pos[:, layer, :], device=device)
        mu = X_t.mean(0)
        std = X_t.std(0).clamp(min=1e-12)
        X_s = (X_t - mu) / std
        d = torch.tensor(directions[layer], device=device, dtype=torch.float32)
        d_unit = d / (d.norm() + 1e-12)
        proj = X_s @ d_unit
        gap = proj[y_t].mean() - proj[~y_t].mean()
        gaps.append(gap.item())
    return gaps


# ─────────────────────────────────────────────────────────────────────────────
# Deep diagnostics (from Llama extra block)
# ─────────────────────────────────────────────────────────────────────────────

def residual_pca_analysis(activations_pos, y, df, direction, peak_layer):
    """
    Project out the conflict direction, then PCA on the residual.
    Checks what structure remains after removing the probe signal.
    """
    X_t = torch.as_tensor(activations_pos[:, peak_layer, :], device="cuda")
    mu = X_t.mean(0)
    std = X_t.std(0).clamp(min=1e-12)
    X_scaled = ((X_t - mu) / std).cpu().numpy()

    d = direction
    proj = X_scaled @ d
    residual = X_scaled - np.outer(proj, d)

    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(residual).ravel()

    results = {"variance_explained_residual_pc1": float(pca.explained_variance_ratio_[0])}

    # Check residual PC1 vs metadata
    for col in ["constraint_type", "system_style", "user_style", "direction"]:
        if col not in df.columns:
            continue
        groups = df.groupby(col).apply(lambda g: pc1[g.index].mean())
        spread = float(groups.max() - groups.min())
        results[f"residual_pc1_spread_{col}"] = spread

    # Correlation with label
    r, p = pearsonr(pc1, y)
    results["residual_pc1_label_corr"] = float(r)
    results["residual_pc1_label_pval"] = float(p)

    return results, residual, pc1


def pc_label_correlation(activations_pos, y, peak_layer, n_components=5):
    """PCA at peak layer, check which PCs correlate with outcome."""
    X_t = torch.as_tensor(activations_pos[:, peak_layer, :], device="cuda")
    mu = X_t.mean(0)
    std = X_t.std(0).clamp(min=1e-12)
    X_scaled = ((X_t - mu) / std).cpu().numpy()

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    results = {"variance_explained": pca.explained_variance_ratio_.tolist()}
    for i in range(n_components):
        r, p = pearsonr(X_pca[:, i], y)
        gap = X_pca[y == 1, i].mean() - X_pca[y == 0, i].mean()
        results[f"pc{i+1}_label_corr"] = float(r)
        results[f"pc{i+1}_label_pval"] = float(p)
        results[f"pc{i+1}_label_gap"] = float(gap)

    return results, X_pca


def top_neuron_identification(direction, n_top=20):
    """Identify top pro-system and pro-user neurons in a direction vector."""
    top_pos = np.argsort(direction)[::-1][:n_top]
    top_neg = np.argsort(direction)[:n_top]
    return {
        "pro_system_neurons": top_pos.tolist(),
        "pro_system_weights": direction[top_pos].tolist(),
        "pro_user_neurons": top_neg.tolist(),
        "pro_user_weights": direction[top_neg].tolist(),
    }


def accuracy_by_scr_group(activations_pos, y, df, direction, peak_layer):
    """Probe accuracy stratified by constraint SCR level."""
    scr_by_ct = df.groupby("constraint_type").apply(lambda g: y[g.index].mean())
    high = scr_by_ct[scr_by_ct >= 0.3].index.tolist()
    mid = scr_by_ct[(scr_by_ct >= 0.1) & (scr_by_ct < 0.3)].index.tolist()
    low = scr_by_ct[scr_by_ct < 0.1].index.tolist()

    X_peak = activations_pos[:, peak_layer, :]
    X_scaled = StandardScaler().fit_transform(X_peak)
    d = direction / (np.linalg.norm(direction) + 1e-12)

    results = {}
    for group_name, cts in [("high", high), ("mid", mid), ("low", low)]:
        mask = df["constraint_type"].isin(cts)
        if mask.sum() < 10:
            results[group_name] = {"n": 0, "acc": None}
            continue
        proj = X_scaled[mask] @ d
        y_grp = y[mask]
        threshold = (proj[y_grp == 1].mean() + proj[y_grp == 0].mean()) / 2 if y_grp.sum() > 0 and (1 - y_grp).sum() > 0 else 0
        preds = (proj > threshold).astype(int)
        acc = balanced_accuracy_score(y_grp, preds) if len(np.unique(y_grp)) > 1 else float("nan")
        results[group_name] = {
            "n": int(mask.sum()),
            "n_pos": int(y_grp.sum()),
            "acc": float(acc),
            "n_constraint_types": len(cts),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Conceptor computation & analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_conceptor(X, alpha=0.1):
    """
    Compute conceptor matrix C = R(R + α⁻²I)⁻¹ where R = XᵀX/n.
    X: (n_samples, d_model) — activations for one class at one layer.
    Returns: C (d_model, d_model) as numpy array.
    """
    n = X.shape[0]
    # Compute on CPU to avoid CUDA linalg instability
    X_t = torch.tensor(X, dtype=torch.float64)
    R = (X_t.T @ X_t) / n
    reg = (1.0 / alpha**2) * torch.eye(R.shape[0], dtype=torch.float64)
    C = R @ torch.linalg.solve(R + reg, torch.eye(R.shape[0], dtype=torch.float64))
    return C.float().numpy()


def conceptor_eigenspectrum(C, n_top=50):
    """Analyze eigenspectrum of a conceptor matrix."""
    eigvals = np.linalg.eigvalsh(C)[::-1]  # sorted descending
    eff_dim_90 = int(np.searchsorted(np.cumsum(eigvals) / eigvals.sum(), 0.9)) + 1
    eff_dim_99 = int(np.searchsorted(np.cumsum(eigvals) / eigvals.sum(), 0.99)) + 1
    return {
        "top_eigenvalues": eigvals[:n_top].tolist(),
        "effective_dim_90": eff_dim_90,
        "effective_dim_99": eff_dim_99,
        "total_energy": float(eigvals.sum()),
    }


def conceptor_subspace_overlap(C_sys, C_usr):
    """Measure overlap between system and user conceptor subspaces."""
    # Frobenius inner product normalized
    overlap = np.trace(C_sys @ C_usr) / (
        np.sqrt(np.trace(C_sys @ C_sys)) * np.sqrt(np.trace(C_usr @ C_usr)) + 1e-12
    )
    return float(overlap)


def conceptor_vs_probe_alignment(C, probe_direction):
    """
    Check alignment between top eigenvector of conceptor and probe direction.
    """
    eigvals, eigvecs = np.linalg.eigh(C)
    top_eigvec = eigvecs[:, -1]  # largest eigenvalue
    cos_sim = abs(np.dot(top_eigvec, probe_direction) / (
        np.linalg.norm(top_eigvec) * np.linalg.norm(probe_direction) + 1e-12
    ))
    return float(cos_sim)


def run_conceptor_analysis(activations, y, directions_by_method, positions, n_layers,
                           alpha=0.1, save_dir=None):
    """
    Full conceptor computation & analysis pipeline.
    Computes, analyzes, and saves each conceptor per layer then discards it
    to avoid accumulating ~17GB of matrices in RAM.
    """
    print("\n── Phase 6: Conceptor computation & analysis ──")
    results = {"alpha": alpha}

    if save_dir:
        cdir = save_dir / "conceptors"
        cdir.mkdir(parents=True, exist_ok=True)

    for pos in positions:
        print(f"  Position: {pos}")
        pos_results = {}

        for layer in tqdm(range(n_layers), desc=f"  Conceptors {pos}", leave=False):
            X_layer = activations[pos][:, layer, :]
            X_sys = X_layer[y == 1]
            X_usr = X_layer[y == 0]

            C_sys = compute_conceptor(X_sys, alpha=alpha)
            C_usr = compute_conceptor(X_usr, alpha=alpha)

            # Save matrices to disk immediately
            if save_dir:
                np.save(cdir / f"C_system_{pos}_L{layer}.npy", C_sys)
                np.save(cdir / f"C_user_{pos}_L{layer}.npy", C_usr)

            # Analyze
            spec_sys = conceptor_eigenspectrum(C_sys)
            spec_usr = conceptor_eigenspectrum(C_usr)
            overlap = conceptor_subspace_overlap(C_sys, C_usr)

            layer_results = {
                "system_eff_dim_90": spec_sys["effective_dim_90"],
                "user_eff_dim_90": spec_usr["effective_dim_90"],
                "system_eff_dim_99": spec_sys["effective_dim_99"],
                "user_eff_dim_99": spec_usr["effective_dim_99"],
                "subspace_overlap": overlap,
            }

            # Alignment with each probe direction
            for method, dirs in directions_by_method.items():
                if pos in dirs:
                    d = dirs[pos][layer]
                    align_sys = conceptor_vs_probe_alignment(C_sys, d)
                    align_usr = conceptor_vs_probe_alignment(C_usr, d)
                    layer_results[f"align_sys_{method}"] = align_sys
                    layer_results[f"align_usr_{method}"] = align_usr

            pos_results[f"L{layer}"] = layer_results

            # Discard — don't accumulate in memory (each is 67MB)
            del C_sys, C_usr

        results[pos] = pos_results

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "last_prompt":     "rgba(31,119,180,1)",
    "last_system":     "rgba(44,160,44,1)",
    "last_user":       "rgba(148,103,189,1)",
    "first_generated": "rgba(255,127,14,1)",
    "mid_generated":   "rgba(227,119,194,1)",
    "last_generated":  "rgba(214,39,40,1)",
}

METHOD_COLORS = {
    "dim":    "rgba(31,119,180,1)",
    "iid_mm": "rgba(255,127,14,1)",
    "lr":     "rgba(44,160,44,1)",
}


def save_fig(fig, path_stem, formats=("png", "pdf")):
    """Save plotly figure to multiple formats."""
    for fmt in formats:
        fig.write_image(f"{path_stem}.{fmt}", scale=2)


def plot_layer_metric(metrics_by_pos, n_layers, title, ylabel, path_stem, formats):
    """Generic layer-wise metric plot across positions."""
    fig = go.Figure()
    for pos_name, values in metrics_by_pos.items():
        fig.add_trace(go.Scatter(
            x=list(range(n_layers)), y=values,
            name=pos_name, mode="lines+markers",
            line=dict(color=COLORS.get(pos_name, "gray"), width=2),
            marker=dict(size=3),
        ))
    fig.update_layout(
        title=title, xaxis_title="Layer", yaxis_title=ylabel,
        template="plotly_white", width=900, height=400,
    )
    save_fig(fig, path_stem, formats)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(cfg: ProbingConfig):
    print(f"\n{'═' * 70}")
    print(f"  run_probing.py — {cfg.act_dir.name}")
    print(f"{'═' * 70}")

    # ── Load data ──
    print("\n── Loading activations ──")
    activations, y, df, positions, n_layers, d_model, ext_config = load_activations(cfg)

    model_name = ext_config.get("model", cfg.act_dir.name)
    if cfg.out_dir is None:
        out_dir = Path("results") / cfg.act_dir.name
    else:
        out_dir = cfg.out_dir
    fig_dir = out_dir / "figures"
    met_dir = out_dir / "metrics"
    art_dir = out_dir / "artifacts" / "directions"
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    fmts = cfg.save_formats

    # ── Build LR subset ──
    lr_idx = build_lr_subset(y, cfg.bal_per_class)
    y_lr = y[lr_idx]
    activations_lr = {pos: activations[pos][lr_idx] for pos in positions}
    print(f"  LR subset: {len(lr_idx)} samples ({y_lr.sum()} pos, {(1-y_lr).sum()} neg)")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: EDA
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Phase 1: EDA ──")
    eda = {
        "model": model_name,
        "n_samples": len(y),
        "n_layers": n_layers,
        "d_model": d_model,
        "scr": float(y.mean()),
        "positions": positions,
    }

    scr_breakdown = df.copy()
    scr_breakdown["y"] = y
    scr_by_ct = scr_breakdown.groupby("constraint_type").agg(
        n=("y", "count"), scr=("y", "mean")
    ).sort_values("scr", ascending=False)
    scr_by_ct.to_csv(met_dir / "scr_breakdown.csv")
    eda["scr_by_constraint_type"] = scr_by_ct.to_dict()

    with open(met_dir / "eda_report.json", "w") as f:
        json.dump(eda, f, indent=2, default=str)
    print(f"  Saved EDA report")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: Direction finding (×3 methods × N positions)
    # ═══════════════════════════════════════════════════════════════════════

    # Storage: directions[method][pos] = list of direction vectors (one per layer)
    # metrics:  probe_metrics[method][pos] = DataFrame
    all_directions = {}   # method -> pos -> [dir_per_layer]
    all_metrics = {}      # method -> pos -> DataFrame
    all_aurocs = {}       # method -> pos -> list

    # Check if Phase 2 already completed (resume support)
    phase2_done = (
        (met_dir / "probe_metrics.csv").exists()
        and (met_dir / "peak_layers.json").exists()
        and (art_dir / f"dim_{positions[0]}_L0.npy").exists()
    )

    if phase2_done:
        print("\n── Phase 2: Direction finding (RESUMING from saved results) ──")

        # Reload metrics
        metrics_df = pd.read_csv(met_dir / "probe_metrics.csv")
        for method in DIRECTION_METHODS:
            all_metrics[method] = {}
            for pos in positions:
                mask = (metrics_df["method"] == method) & (metrics_df["position"] == pos)
                df_m = metrics_df[mask].copy().reset_index(drop=True)
                if "bal_acc" not in df_m.columns and "mean" in df_m.columns:
                    df_m["bal_acc"] = df_m["mean"]
                all_metrics[method][pos] = df_m
                peak = df_m["bal_acc"].max()
                peak_l = df_m["bal_acc"].idxmax()
                print(f"  {method} @ {pos}: peak {peak:.3f} @ L{peak_l}")

        # Reload directions
        for method in DIRECTION_METHODS:
            all_directions[method] = {}
            for pos in positions:
                dirs = []
                for layer in range(n_layers):
                    d_path = art_dir / f"{method}_{pos}_L{layer}.npy"
                    dirs.append(np.load(d_path))
                all_directions[method][pos] = dirs

        # Reload peak layers
        with open(met_dir / "peak_layers.json") as f:
            peak_layers = json.load(f)

        print("  Resumed successfully.")

    else:
        print("\n── Phase 2: Direction finding ──")

        for pos in positions:
            print(f"\n  Position: {pos}")

            # DiM
            print(f"    DiM...")
            dim_df, dim_dirs = diff_in_mean_probe_gpu(activations[pos], y, scale=True, device=device)
            all_directions.setdefault("dim", {})[pos] = dim_dirs
            all_metrics.setdefault("dim", {})[pos] = dim_df
            print(f"      Peak: {dim_df['bal_acc'].max():.3f} @ L{dim_df['bal_acc'].idxmax()}")

            # IID-MM
            print(f"    IID-MM...")
            iid_df, iid_dirs = iid_mass_mean_probe_gpu(activations[pos], y, scale=True, device=device)
            all_directions.setdefault("iid_mm", {})[pos] = iid_dirs
            all_metrics.setdefault("iid_mm", {})[pos] = iid_df
            print(f"      Peak: {iid_df['bal_acc'].max():.3f} @ L{iid_df['bal_acc'].idxmax()}")

            # LR (on balanced subset)
            print(f"    LR Probe...")
            lr_df, lr_aurocs, lr_dirs = lr_probe_cv(
                activations_lr[pos], y_lr, n_folds=cfg.n_cv_folds, scale=True
            )
            all_directions.setdefault("lr", {})[pos] = lr_dirs
            all_metrics.setdefault("lr", {})[pos] = lr_df
            all_aurocs.setdefault("lr", {})[pos] = lr_aurocs
            print(f"      Peak: {lr_df['bal_acc'].max():.3f} @ L{lr_df['bal_acc'].idxmax()}")

        # Save directions
        for method in DIRECTION_METHODS:
            for pos in positions:
                dirs = all_directions[method][pos]
                for layer, d in enumerate(dirs):
                    np.save(art_dir / f"{method}_{pos}_L{layer}.npy", d)

        # Save metrics
        probe_metrics_flat = []
        for method in DIRECTION_METHODS:
            for pos in positions:
                df_m = all_metrics[method][pos].copy()
                df_m["method"] = method
                df_m["position"] = pos
                probe_metrics_flat.append(df_m)
        pd.concat(probe_metrics_flat).to_csv(met_dir / "probe_metrics.csv", index=False)

        # Peak layers
        peak_layers = {}
        for method in DIRECTION_METHODS:
            peak_layers[method] = {}
            for pos in positions:
                peak_layers[method][pos] = int(all_metrics[method][pos]["bal_acc"].idxmax())
        with open(met_dir / "peak_layers.json", "w") as f:
            json.dump(peak_layers, f, indent=2)

    # Plots
    for method in DIRECTION_METHODS:
        plot_layer_metric(
            {pos: all_metrics[method][pos]["bal_acc"].tolist() for pos in positions},
            n_layers,
            f"Balanced accuracy — {METHOD_LABELS[method]}",
            "Balanced accuracy",
            str(fig_dir / f"layer_acc_{method}"),
            fmts,
        )

    print(f"\n  Saved directions, metrics, plots")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3: Controls
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Phase 3: Controls ──")
    controls = {}

    # Metadata-only
    meta_acc, meta_std = metadata_only_control(df, y, n_folds=cfg.n_cv_folds)
    controls["metadata_only"] = {"mean": meta_acc, "std": meta_std}
    print(f"  Metadata-only control: {meta_acc:.3f} ± {meta_std:.3f}")

    # Permuted labels (at each method's peak layer, for best position)
    for method in DIRECTION_METHODS:
        best_pos = max(positions, key=lambda p: all_metrics[method][p]["bal_acc"].max())
        pk = peak_layers[method][best_pos]
        X_peak = activations_lr[best_pos][:, pk, :]
        perm_mean, perm_std = permutation_control(
            X_peak, y_lr, n_permutations=cfg.n_permutations, n_folds=cfg.n_cv_folds
        )
        controls[f"permuted_{method}"] = {
            "mean": perm_mean, "std": perm_std,
            "position": best_pos, "layer": pk,
        }
        print(f"  Permuted ({method}, {best_pos}@L{pk}): {perm_mean:.3f} ± {perm_std:.3f}")

    # Condition PCA
    best_pos_overall = max(
        positions, key=lambda p: max(
            all_metrics[m][p]["bal_acc"].max() for m in DIRECTION_METHODS
        )
    )
    best_layer_overall = max(
        (peak_layers[m][best_pos_overall] for m in DIRECTION_METHODS),
        key=lambda l: max(
            all_metrics[m][best_pos_overall]["bal_acc"].iloc[l]
            for m in DIRECTION_METHODS
        )
    )
    cond_results, X_pca_cond, cond_scr, cond_labels = condition_level_pca(
        activations[best_pos_overall], y, df, best_layer_overall
    )
    controls["condition_pca"] = cond_results
    print(f"  Condition PCA: PC1↔SCR r={cond_results.get('pc1_scr_corr', 'N/A'):.3f}")

    with open(met_dir / "controls.json", "w") as f:
        json.dump(controls, f, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 4: Direction analysis (×3 methods)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Phase 4: Direction analysis ──")
    direction_analysis = {}

    for method in DIRECTION_METHODS:
        print(f"  Method: {method}")
        m_results = {}

        # Cross-method cosine sim (only meaningful to compute once, but we
        # store it per method for the "this method vs others" perspective)
        if method == DIRECTION_METHODS[0]:  # compute once
            for pos in positions:
                dirs_at_pos = {m: all_directions[m][pos] for m in DIRECTION_METHODS}
                cos_sims = cross_method_cosine_sim(dirs_at_pos, n_layers)
                direction_analysis[f"cross_method_cos_{pos}"] = {
                    k: v for k, v in cos_sims.items()
                }

        # Cross-position transfer
        pk = peak_layers[method][positions[0]]  # use first position's peak
        transfer = cross_position_transfer(
            activations, y, all_directions[method], pk, positions, cfg.test_size
        )
        m_results["cross_position_transfer"] = transfer.to_dict()
        print(f"    Cross-position transfer computed (L{pk})")

        # Cross-category transfer (at best position)
        best_pos = max(positions, key=lambda p: all_metrics[method][p]["bal_acc"].max())
        pk_best = peak_layers[method][best_pos]
        cat_transfer, categories = cross_category_transfer(
            activations[best_pos], y, df, all_directions[method][best_pos], pk_best
        )
        m_results["cross_category_transfer"] = cat_transfer.to_dict()

        # Projection gap
        for pos in positions:
            gaps = projection_gap_per_layer(
                activations[pos], y, all_directions[method][pos], n_layers, device
            )
            m_results[f"projection_gap_{pos}"] = gaps

        direction_analysis[method] = m_results

    with open(met_dir / "direction_analysis.json", "w") as f:
        json.dump(direction_analysis, f, indent=2, default=str)

    # Plots for Phase 4
    for method in DIRECTION_METHODS:
        # Projection gap
        plot_layer_metric(
            {pos: direction_analysis[method][f"projection_gap_{pos}"] for pos in positions},
            n_layers,
            f"Projection gap — {METHOD_LABELS[method]}",
            "Gap (system − user)",
            str(fig_dir / f"projection_gap_{method}"),
            fmts,
        )
    print(f"  Saved direction analysis")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 5: Deep diagnostics (×3 methods)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Phase 5: Deep diagnostics ──")
    diagnostics = {}

    for method in DIRECTION_METHODS:
        print(f"  Method: {method}")
        m_diag = {}

        best_pos = max(positions, key=lambda p: all_metrics[method][p]["bal_acc"].max())
        pk = peak_layers[method][best_pos]
        direction = all_directions[method][best_pos][pk]

        # Residual PCA
        res_pca, residual, pc1 = residual_pca_analysis(
            activations[best_pos], y, df, direction, pk
        )
        m_diag["residual_pca"] = res_pca

        # PC↔label correlation
        pc_corr, X_pca_peak = pc_label_correlation(activations[best_pos], y, pk)
        m_diag["pc_label_correlation"] = pc_corr

        # Top neurons
        m_diag["top_neurons"] = top_neuron_identification(direction)

        # Accuracy by SCR group
        scr_acc = accuracy_by_scr_group(activations[best_pos], y, df, direction, pk)
        m_diag["accuracy_by_scr_group"] = scr_acc

        diagnostics[method] = m_diag
        print(f"    residual_pc1↔label r={res_pca['residual_pc1_label_corr']:.3f}")
        print(f"    SCR groups: " + ", ".join(
            f"{g}={v['acc']:.3f}" for g, v in scr_acc.items() if v.get('acc') is not None
        ))

    with open(met_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    print(f"  Saved diagnostics")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 6: Conceptors
    # ═══════════════════════════════════════════════════════════════════════
    if not cfg.skip_conceptors:
        conceptor_results = run_conceptor_analysis(
            activations, y, all_directions, positions, n_layers,
            alpha=cfg.conceptor_alpha,
            save_dir=out_dir / "artifacts",
        )
        with open(met_dir / "conceptor_analysis.json", "w") as f:
            json.dump(conceptor_results, f, indent=2, default=str)
        print(f"  Saved conceptor analysis")
    else:
        print("\n── Phase 6: Conceptors SKIPPED ──")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 7: Summary & export
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Phase 7: Summary ──")
    summary = {
        "model": model_name,
        "n_samples": len(y),
        "n_layers": n_layers,
        "d_model": d_model,
        "scr": float(y.mean()),
        "positions": positions,
        "peak_layers": peak_layers,
        "controls": controls,
    }

    # Best accuracy per method
    for method in DIRECTION_METHODS:
        best_pos = max(positions, key=lambda p: all_metrics[method][p]["bal_acc"].max())
        pk = peak_layers[method][best_pos]
        acc = all_metrics[method][best_pos]["bal_acc"].max()
        summary[f"best_{method}"] = {
            "position": best_pos,
            "layer": pk,
            "bal_acc": float(acc),
        }

    with open(out_dir / "summary_report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Bundle all directions
    dir_bundle = {}
    for method in DIRECTION_METHODS:
        for pos in positions:
            for layer, d in enumerate(all_directions[method][pos]):
                dir_bundle[f"{method}_{pos}_L{layer}"] = d
    np.savez_compressed(out_dir / "artifacts" / "all_directions.npz", **dir_bundle)

    print(f"\n  Summary saved to {out_dir / 'summary_report.json'}")
    print(f"  All outputs in {out_dir}/")
    print(f"\n{'═' * 70}")
    print(f"  Pipeline complete.")
    print(f"{'═' * 70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run probing & analysis pipeline on extracted activations."
    )
    parser.add_argument(
        "--act-dir", type=str, required=True,
        help="Directory with activations from extract_activations.py",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: results/{model_name})",
    )
    parser.add_argument(
        "--positions", nargs="+", default=None,
        help="Token positions to analyze (default: all from extraction)",
    )
    parser.add_argument("--n-cv-folds", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=3)
    parser.add_argument("--bal-per-class", type=int, default=2500)
    parser.add_argument("--conceptor-alpha", type=float, default=0.1)
    parser.add_argument("--skip-conceptors", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--formats", nargs="+", default=["png", "pdf"],
        help="Figure output formats",
    )
    args = parser.parse_args()

    cfg = ProbingConfig(
        act_dir=Path(args.act_dir),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        positions=args.positions,
        n_cv_folds=args.n_cv_folds,
        n_permutations=args.n_permutations,
        bal_per_class=args.bal_per_class,
        conceptor_alpha=args.conceptor_alpha,
        skip_conceptors=args.skip_conceptors,
        device=args.device,
        save_formats=args.formats,
    )

    if not cfg.act_dir.exists():
        print(f"Error: activation directory not found: {cfg.act_dir}")
        sys.exit(1)

    run_pipeline(cfg)


if __name__ == "__main__":
    main()