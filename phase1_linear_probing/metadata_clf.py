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
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from probe import make_cv_splitter


_SCORING = ["roc_auc", "balanced_accuracy"]


# ── Feature construction ─────────────────────────────────────────────────────

def _one_hot(val, categories):
    return [int(val == c) for c in categories]


def build_category_lists(df, categorical_cols=["constraint_type", "strength", "user_style", "task_id"]):
    """Extract sorted unique values for each categorical column."""
    return {col: sorted(df[col].unique()) for col in categorical_cols}


def build_metadata_features(df, position_maps, cats, *, exclude_groups=None):
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
    exclude_groups : set[str] or None
        Feature group names to exclude. Valid names: ``"length_feats"``,
        ``"constraint_type"``, ``"strength"``, ``"user_style"``, ``"task_id"``,
        ``"direction"``.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features), dtype float32
    """
    excl = exclude_groups or set()
    rows = []
    for pm, (_, row) in zip(position_maps, df.iterrows()):
        feats = []
        if "length_feats" not in excl:
            feats += [
                pm["last_prompt"] + 1,
                pm["mean_system"][1],
                pm["mean_user"][1] - pm["mean_user"][0],
                pm["mean_user"][0],
            ]
        if "constraint_type" not in excl:
            feats += _one_hot(row["constraint_type"], cats["constraint_type"])
        if "strength" not in excl:
            feats += _one_hot(row["strength"], cats["strength"])
        if "user_style" not in excl:
            feats += _one_hot(row["user_style"], cats["user_style"])
        if "task_id" not in excl:
            feats += _one_hot(row["task_id"], cats["task_id"])
        if "direction" not in excl:
            feats += [int(row["direction"] == "b_to_a")]
        rows.append(feats)
    return np.array(rows, dtype=np.float32)


def get_feature_names(cats, *, exclude_groups=None):
    """Return ordered list of feature names matching build_metadata_features columns."""
    excl = exclude_groups or set()
    names = []
    if "length_feats" not in excl:
        names += ["total_tokens", "sys_len", "user_len", "user_start"]
    if "constraint_type" not in excl:
        names += [f"ctype_{c}" for c in cats["constraint_type"]]
    if "strength" not in excl:
        names += [f"strength_{s}" for s in cats["strength"]]
    if "user_style" not in excl:
        names += [f"style_{s}" for s in cats["user_style"]]
    if "task_id" not in excl:
        names += [f"task_{t}" for t in cats["task_id"]]
    if "direction" not in excl:
        names += ["dir_b_to_a"]
    return names


def get_feature_groups(cats, *, exclude_groups=None):
    """Return dict mapping group name -> list of column indices.

    If *exclude_groups* is given, the excluded groups are omitted and indices
    are recomputed so they match the matrix returned by
    ``build_metadata_features(..., exclude_groups=exclude_groups)``.
    """
    excl = exclude_groups or set()
    groups = {}
    idx = 0

    def _add(name, size):
        nonlocal idx
        if name not in excl:
            groups[name] = list(range(idx, idx + size))
            idx += size

    _add("length_feats", 4)
    _add("constraint_type", len(cats["constraint_type"]))
    _add("strength", len(cats["strength"]))
    _add("user_style", len(cats["user_style"]))
    _add("task_id", len(cats["task_id"]))
    _add("direction", 1)

    return groups


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_multi_metric_result(cv_out, estimator_name):
    """Build a multi-metric result dict from cross_validate output.

    Includes backward-compatible aliases (mean, std, scores) pointing to
    roc_auc values.
    """
    result = {
        "roc_auc_mean": cv_out["test_roc_auc"].mean(),
        "roc_auc_std": cv_out["test_roc_auc"].std(),
        "roc_auc_scores": cv_out["test_roc_auc"],
        "balanced_accuracy_mean": cv_out["test_balanced_accuracy"].mean(),
        "balanced_accuracy_std": cv_out["test_balanced_accuracy"].std(),
        "balanced_accuracy_scores": cv_out["test_balanced_accuracy"],
        "estimator_name": estimator_name,
    }
    # Backward compatibility aliases
    result["mean"] = result["roc_auc_mean"]
    result["std"] = result["roc_auc_std"]
    result["scores"] = result["roc_auc_scores"]
    return result


# ── Linear control ───────────────────────────────────────────────────────────

def run_linear_control(
    X, y, *,
    cv_mode="stratified",
    groups=None,
    n_folds=5,
    random_state=42,
):
    """Run linear logistic regression CV on metadata features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Binary labels of shape (n_samples,).
    cv_mode : str
        ``"stratified"`` or ``"grouped"``.
    groups : np.ndarray or None
        Group labels for GroupKFold (required if cv_mode="grouped").
    n_folds : int
        Number of CV folds.
    random_state : int
        Seed for StratifiedKFold shuffle.

    Returns
    -------
    dict
        Keys: roc_auc_mean, roc_auc_std, roc_auc_scores,
        balanced_accuracy_mean, balanced_accuracy_std,
        balanced_accuracy_scores, estimator_name.
        Backward-compat aliases: mean, std, scores (point to roc_auc values).
    """
    cv = make_cv_splitter(cv_mode, n_folds, random_state)
    cv_kwargs = {"groups": groups} if cv_mode == "grouped" else {}
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
    ])
    cv_out = cross_validate(
        pipe, X, y, cv=cv, scoring=_SCORING, n_jobs=-1, **cv_kwargs,
    )
    return _build_multi_metric_result(cv_out, "LogisticRegression")


# ── Boosted control (shared helpers) ─────────────────────────────────────────

def _get_boosted_param_distributions():
    """Shared hyperparam search space for HistGradientBoosting."""
    return {
        "learning_rate": loguniform(0.01, 0.3),
        "max_leaf_nodes": randint(15, 80),
        "min_samples_leaf": randint(5, 50),
        "l2_regularization": loguniform(1e-3, 10.0),
    }


def _make_boosted_search(*, n_search_iter, inner_folds, random_state):
    """Create a RandomizedSearchCV wrapping HistGradientBoostingClassifier.

    Inner CV always uses StratifiedKFold (for hyperparameter tuning).
    Scoring is hardcoded to ``"roc_auc"`` for refit selection.
    """
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
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )


def run_boosted_control(
    X, y, *,
    cv_mode="stratified",
    groups=None,
    n_folds=5,
    n_search_iter=30,
    inner_folds=3,
    random_state=42,
):
    """Run nested-CV boosted tree classifier on metadata features.

    Outer loop: configurable CV splitter (stratified or grouped).
    Inner loop: RandomizedSearchCV with StratifiedKFold and n_search_iter draws.

    HistGradientBoostingClassifier uses built-in early stopping, so max_iter
    is set high and n_iter_no_change controls when to stop adding trees.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Binary labels of shape (n_samples,).
    cv_mode : str
        ``"stratified"`` or ``"grouped"``.
    groups : np.ndarray or None
        Group labels for GroupKFold (required if cv_mode="grouped").
    n_folds : int
        Number of outer CV folds.
    n_search_iter : int
        Number of hyperparameter search iterations.
    inner_folds : int
        Number of inner CV folds for hyperparameter tuning.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Keys: roc_auc_mean, roc_auc_std, roc_auc_scores,
        balanced_accuracy_mean, balanced_accuracy_std,
        balanced_accuracy_scores, estimator_name.
        Backward-compat aliases: mean, std, scores (point to roc_auc values).
    """
    outer_cv = make_cv_splitter(cv_mode, n_folds, random_state)
    cv_kwargs = {"groups": groups} if cv_mode == "grouped" else {}

    search = _make_boosted_search(
        n_search_iter=n_search_iter,
        inner_folds=inner_folds,
        random_state=random_state,
    )

    cv_out = cross_validate(
        search, X, y, cv=outer_cv, scoring=_SCORING, n_jobs=1, **cv_kwargs,
    )

    return _build_multi_metric_result(
        cv_out, "HistGradientBoosting (nested CV)"
    )


def fit_boosted_importances(
    X, y, *,
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
        n_search_iter=n_search_iter,
        inner_folds=inner_folds,
        random_state=random_state,
    )
    search.fit(X, y)
    perm = permutation_importance(
        search.best_estimator_, X, y,
        scoring="roc_auc",
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
    cv_mode="stratified",
    groups=None,
    n_folds=5,
    baseline_roc_auc_mean=None,
    baseline_roc_auc_std=None,
    random_state=42,
    **boosted_kwargs,
):
    """Leave-one-group-out ablation for either linear or boosted classifier.

    Parameters
    ----------
    classifier : "linear" or "boosted"
    cv_mode : str
        ``"stratified"`` or ``"grouped"``.
    groups : np.ndarray or None
        Group labels for GroupKFold (required if cv_mode="grouped").
    n_folds : int
        Number of CV folds.
    baseline_roc_auc_mean, baseline_roc_auc_std : float or None
        If provided, used as the "none (baseline)" row. Otherwise computed.
    **boosted_kwargs : forwarded to run_boosted_control (n_search_iter, etc.)

    Returns
    -------
    pd.DataFrame with columns: group_dropped,
        roc_auc_mean, roc_auc_std, roc_auc_drop,
        balanced_accuracy_mean, balanced_accuracy_std, balanced_accuracy_drop
    """
    run_fn = run_linear_control if classifier == "linear" else run_boosted_control
    run_kw = dict(
        cv_mode=cv_mode, groups=groups, n_folds=n_folds,
        random_state=random_state,
    )
    if classifier == "boosted":
        run_kw.update(boosted_kwargs)

    if baseline_roc_auc_mean is None:
        res = run_fn(X, y, **run_kw)
        baseline_roc_auc_mean = res["roc_auc_mean"]
        baseline_roc_auc_std = res["roc_auc_std"]
        baseline_bacc_mean = res["balanced_accuracy_mean"]
        baseline_bacc_std = res["balanced_accuracy_std"]
    else:
        baseline_bacc_mean = None
        baseline_bacc_std = None

    rows = [{
        "group_dropped": "none (baseline)",
        "roc_auc_mean": baseline_roc_auc_mean,
        "roc_auc_std": baseline_roc_auc_std,
        "roc_auc_drop": 0.0,
        "balanced_accuracy_mean": baseline_bacc_mean,
        "balanced_accuracy_std": baseline_bacc_std,
        "balanced_accuracy_drop": 0.0,
    }]

    for grp_name, grp_idx in feature_groups.items():
        mask = np.ones(X.shape[1], dtype=bool)
        mask[grp_idx] = False
        res = run_fn(X[:, mask], y, **run_kw)
        rows.append({
            "group_dropped": grp_name,
            "roc_auc_mean": res["roc_auc_mean"],
            "roc_auc_std": res["roc_auc_std"],
            "roc_auc_drop": baseline_roc_auc_mean - res["roc_auc_mean"],
            "balanced_accuracy_mean": res["balanced_accuracy_mean"],
            "balanced_accuracy_std": res["balanced_accuracy_std"],
            "balanced_accuracy_drop": (
                baseline_bacc_mean - res["balanced_accuracy_mean"]
                if baseline_bacc_mean is not None
                else None
            ),
        })

    return pd.DataFrame(rows)
