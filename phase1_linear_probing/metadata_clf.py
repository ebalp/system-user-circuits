"""Metadata-only control classifiers for linear probing analysis.

Provides linear (LogisticRegression) and non-linear (HistGradientBoosting)
classifiers trained on surface-level prompt features — no activations used.

The boosted classifier uses nested cross-validation: inner loop tunes
hyperparameters via RandomizedSearchCV, outer loop produces unbiased scores
directly comparable to the activation probe's CV scores.
"""

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Feature construction ─────────────────────────────────────────────────────

def _one_hot(val, categories):
    return [int(val == c) for c in categories]


def build_category_lists(df):
    """Extract sorted unique values for each categorical column."""
    return {
        "constraint_type": sorted(df["constraint_type"].unique()),
        "strength": sorted(df["strength"].unique()),
        "user_style": sorted(df["user_style"].unique()),
        "task_id": sorted(df["task_id"].unique()),
    }


def build_metadata_features(df, position_maps, cats):
    """Build the metadata feature matrix from dataframe rows and token position maps.

    Parameters
    ----------
    df : pd.DataFrame
        Condition C data with columns: constraint_type, strength, user_style,
        task_id, direction.
    position_maps : list[dict]
        Per-sample token position dicts (from find_token_positions).
    cats : dict
        Output of build_category_lists.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features), dtype float32
    """
    rows = []
    for pm, (_, row) in zip(position_maps, df.iterrows()):
        rows.append([
            pm["last_prompt"] + 1,
            pm["mean_system"][1],
            pm["mean_user"][1] - pm["mean_user"][0],
            pm["mean_user"][0],
            *_one_hot(row["constraint_type"], cats["constraint_type"]),
            *_one_hot(row["strength"], cats["strength"]),
            *_one_hot(row["user_style"], cats["user_style"]),
            *_one_hot(row["task_id"], cats["task_id"]),
            int(row["direction"] == "b_to_a"),
        ])
    return np.array(rows, dtype=np.float32)


def get_feature_names(cats):
    """Return ordered list of feature names matching build_metadata_features columns."""
    return (
        ["total_tokens", "sys_len", "user_len", "user_start"]
        + [f"ctype_{c}" for c in cats["constraint_type"]]
        + [f"strength_{s}" for s in cats["strength"]]
        + [f"style_{s}" for s in cats["user_style"]]
        + [f"task_{t}" for t in cats["task_id"]]
        + ["dir_b_to_a"]
    )


def get_feature_groups(cats):
    """Return dict mapping group name -> list of column indices."""
    n_ctype = len(cats["constraint_type"])
    n_str = len(cats["strength"])
    n_sty = len(cats["user_style"])
    n_task = len(cats["task_id"])

    i0_ctype = 4
    i0_str = i0_ctype + n_ctype
    i0_sty = i0_str + n_str
    i0_task = i0_sty + n_sty
    i0_dir = i0_task + n_task

    return {
        "length_feats": list(range(0, 4)),
        "constraint_type": list(range(i0_ctype, i0_str)),
        "strength": list(range(i0_str, i0_sty)),
        "user_style": list(range(i0_sty, i0_task)),
        "task_id": list(range(i0_task, i0_dir)),
        "direction": [i0_dir],
    }


# ── Linear control ───────────────────────────────────────────────────────────

def run_linear_control(X, y, *, n_folds=5, metric="roc_auc", random_state=42):
    """Run linear logistic regression CV on metadata features.

    Returns
    -------
    dict with keys: mean, std, scores (per-fold array), estimator_name
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=metric, n_jobs=-1)
    return {
        "mean": scores.mean(),
        "std": scores.std(),
        "scores": scores,
        "estimator_name": "LogisticRegression",
    }


# ── Boosted control (shared helpers) ─────────────────────────────────────────

def _get_boosted_param_distributions():
    """Shared hyperparam search space for HistGradientBoosting."""
    return {
        "learning_rate": loguniform(0.01, 0.3),
        "max_leaf_nodes": randint(15, 80),
        "min_samples_leaf": randint(5, 50),
        "l2_regularization": loguniform(1e-3, 10.0),
    }


def _make_boosted_search(*, metric, n_search_iter, inner_folds, random_state):
    """Create a RandomizedSearchCV wrapping HistGradientBoostingClassifier."""
    inner_cv = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=random_state
    )
    base_estimator = HistGradientBoostingClassifier(
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.15,
        random_state=random_state,
    )
    return RandomizedSearchCV(
        base_estimator,
        _get_boosted_param_distributions(),
        n_iter=n_search_iter,
        cv=inner_cv,
        scoring=metric,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )


def run_boosted_control(
    X, y, *,
    n_folds=5,
    metric="roc_auc",
    n_search_iter=30,
    inner_folds=3,
    random_state=42,
):
    """Run nested-CV boosted tree classifier on metadata features.

    Outer loop: n_folds StratifiedKFold (same splits as probe).
    Inner loop: RandomizedSearchCV with inner_folds and n_search_iter draws.

    HistGradientBoostingClassifier uses built-in early stopping, so max_iter
    is set high and n_iter_no_change controls when to stop adding trees.

    Returns
    -------
    dict with keys: mean, std, scores (per-fold array), estimator_name
    """
    outer_cv = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )

    search = _make_boosted_search(
        metric=metric,
        n_search_iter=n_search_iter,
        inner_folds=inner_folds,
        random_state=random_state,
    )

    scores = cross_val_score(
        search, X, y, cv=outer_cv, scoring=metric, n_jobs=1,
    )

    return {
        "mean": scores.mean(),
        "std": scores.std(),
        "scores": scores,
        "estimator_name": "HistGradientBoosting (nested CV)",
    }


def fit_boosted_importances(
    X, y, *,
    metric="roc_auc",
    n_search_iter=30,
    inner_folds=3,
    n_repeats=5,
    random_state=42,
):
    """Tune a HistGradientBoosting on all data, return permutation importances.

    Uses RandomizedSearchCV to select hyperparams, then computes
    permutation_importance on the fitted model (shuffles each feature
    and measures the score drop).

    Returns
    -------
    dict with keys:
        importances_mean (np.ndarray), importances_std (np.ndarray),
        best_params (dict)
    """
    search = _make_boosted_search(
        metric=metric,
        n_search_iter=n_search_iter,
        inner_folds=inner_folds,
        random_state=random_state,
    )
    search.fit(X, y)
    perm = permutation_importance(
        search.best_estimator_, X, y,
        scoring=metric,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return {
        "importances_mean": perm.importances_mean,
        "importances_std": perm.importances_std,
        "best_params": search.best_params_,
    }


# ── Group ablation ───────────────────────────────────────────────────────────

def run_group_ablation(
    X, y, feature_groups, *,
    classifier="linear",
    n_folds=5,
    metric="roc_auc",
    baseline_mean=None,
    baseline_std=None,
    random_state=42,
    **boosted_kwargs,
):
    """Leave-one-group-out ablation for either linear or boosted classifier.

    Parameters
    ----------
    classifier : "linear" or "boosted"
    baseline_mean, baseline_std : float or None
        If provided, used as the "none (baseline)" row. Otherwise computed.
    **boosted_kwargs : forwarded to run_boosted_control (n_search_iter, etc.)

    Returns
    -------
    pd.DataFrame with columns: group_dropped, mean, std, drop
    """
    run_fn = run_linear_control if classifier == "linear" else run_boosted_control
    run_kw = dict(n_folds=n_folds, metric=metric, random_state=random_state)
    if classifier == "boosted":
        run_kw.update(boosted_kwargs)

    if baseline_mean is None:
        res = run_fn(X, y, **run_kw)
        baseline_mean = res["mean"]
        baseline_std = res["std"]

    rows = [{
        "group_dropped": "none (baseline)",
        "mean": baseline_mean,
        "std": baseline_std,
        "drop": 0.0,
    }]

    for grp_name, grp_idx in feature_groups.items():
        mask = np.ones(X.shape[1], dtype=bool)
        mask[grp_idx] = False
        res = run_fn(X[:, mask], y, **run_kw)
        rows.append({
            "group_dropped": grp_name,
            "mean": res["mean"],
            "std": res["std"],
            "drop": baseline_mean - res["mean"],
        })

    return pd.DataFrame(rows)
