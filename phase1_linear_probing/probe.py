"""Unified linear probing: CV scoring, full-data fitting, and persistence.

Merges the old ``probe_all_positions`` and ``fit_probe_directions`` notebook
functions into a single :func:`probe_and_fit` call that returns both CV
evaluation scores and fitted classifiers at every layer.  Supports both
stratified and grouped cross-validation, always reports ``roc_auc`` and
``balanced_accuracy``, and provides joblib-based classifier persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


# ── CV Splitter ──────────────────────────────────────────────────────────────


def make_cv_splitter(
    cv_mode: Literal["stratified", "grouped"] = "stratified",
    n_folds: int = 5,
    random_state: int = 42,
):
    """Create a cross-validation splitter.

    Parameters
    ----------
    cv_mode : "stratified" or "grouped"
        ``"stratified"`` -> :class:`StratifiedKFold` (balanced class ratios
        per fold).
        ``"grouped"`` -> :class:`GroupKFold` (entire groups held out per fold).
    n_folds : int
        Number of folds.  For ``"grouped"``, must be <= n_unique_groups.
    random_state : int
        Only used for :class:`StratifiedKFold` shuffle.
    """
    if cv_mode == "stratified":
        return StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
    elif cv_mode == "grouped":
        return GroupKFold(n_splits=n_folds)
    else:
        raise ValueError(
            f"Unknown cv_mode: {cv_mode!r}. Use 'stratified' or 'grouped'."
        )


# ── Probe Result ─────────────────────────────────────────────────────────────


@dataclass
class ProbeResult:
    """Results from probing all layers at one token position.

    Combines CV evaluation scores with full-data fitted classifiers.
    """

    # CV scores -- always contains both metrics
    cv_scores: pd.DataFrame
    # Columns: layer, roc_auc_mean, roc_auc_std,
    #          balanced_accuracy_mean, balanced_accuracy_std

    # Full-data fit results (for direction analysis and persistence)
    weights: np.ndarray  # (n_layers, d_model) unit-norm direction vectors
    weights_raw: np.ndarray  # (n_layers, d_model) raw coefficient vectors
    biases: np.ndarray  # (n_layers,) intercept values
    scalers: list  # list[StandardScaler | None] per layer
    classifiers: list  # list[LogisticRegression] per layer

    # Metadata
    pos_name: str
    cv_mode: str
    use_scaler: bool

    # Per-fold data (populated when return_estimator=True)
    fold_weights: np.ndarray | None = None  # (n_layers, n_folds, d_model) unit-norm
    fold_scores: np.ndarray | None = None  # (n_layers, n_folds) ROC AUC per fold
    fold_group_names: list[str] | None = None  # (n_folds,) held-out group per fold


# ── Helpers ──────────────────────────────────────────────────────────────────

_SCORING = ["roc_auc", "balanced_accuracy"]


def _make_pipeline(use_scaler: bool) -> Pipeline:
    """Build a logistic regression pipeline with optional scaling."""
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        ("clf", LogisticRegression(max_iter=1000, C=0.005, solver="lbfgs"))
    )
    return Pipeline(steps)


# ── Core Probing ─────────────────────────────────────────────────────────────


def probe_and_fit(
    activations: dict[str, np.ndarray],
    y: np.ndarray,
    token_positions: list[str],
    *,
    cv_mode: Literal["stratified", "grouped"] = "stratified",
    groups: np.ndarray | None = None,
    n_folds: int = 5,
    use_scaler: bool = False,
    random_state: int = 42,
) -> dict[str, ProbeResult]:
    """Run linear probe at every layer: CV scoring + full-data fit.

    For each token position and layer:

    1. :func:`~sklearn.model_selection.cross_validate` with
       ``scoring=["roc_auc", "balanced_accuracy"]`` produces CV metrics.
    2. A fresh pipeline is fit on all data to obtain classifier weights,
       directions, and scalers.

    Parameters
    ----------
    activations : dict mapping position name -> (n_samples, n_layers, d_model)
    y : binary labels, shape (n_samples,)
    token_positions : which positions to probe
    cv_mode : "stratified" or "grouped"
    groups : group labels for GroupKFold (required if cv_mode="grouped")
    n_folds : number of CV folds
    use_scaler : whether to include StandardScaler in the pipeline
    random_state : seed for StratifiedKFold

    Returns
    -------
    dict mapping position name -> ProbeResult
    """
    cv = make_cv_splitter(cv_mode, n_folds, random_state)
    cv_kwargs: dict = {"groups": groups} if cv_mode == "grouped" else {}

    results: dict[str, ProbeResult] = {}

    for pos_name in token_positions:
        X = activations[pos_name]
        n_layers = X.shape[1]
        d_model = X.shape[2]

        # Accumulators
        rows: list[dict] = []
        weights = np.zeros((n_layers, d_model))
        weights_raw = np.zeros((n_layers, d_model))
        biases = np.zeros(n_layers)
        scalers: list = []
        classifiers: list = []

        # Determine fold group names once (same split for all layers)
        fold_group_names: list[str] | None = None
        if cv_mode == "grouped" and groups is not None:
            fold_group_names = []
            X_dummy = np.zeros((len(y), 1))
            for _, val_idx in cv.split(X_dummy, y, groups):
                fold_group_names.append(str(groups[val_idx[0]]))

        actual_n_folds = n_folds
        fold_weights_all = np.zeros((n_layers, n_folds, d_model))
        fold_scores_all = np.zeros((n_layers, n_folds))

        for layer in tqdm(
            range(n_layers),
            desc=f"Probing {pos_name}",
            leave=False,
        ):
            X_layer = X[:, layer, :]

            # -- CV scoring --
            pipe_cv = _make_pipeline(use_scaler)
            cv_out = cross_validate(
                pipe_cv,
                X_layer,
                y,
                cv=cv,
                scoring=_SCORING,
                n_jobs=-1,
                return_estimator=True,
                **cv_kwargs,
            )

            rows.append(
                {
                    "layer": layer,
                    "roc_auc_mean": cv_out["test_roc_auc"].mean(),
                    "roc_auc_std": cv_out["test_roc_auc"].std(),
                    "balanced_accuracy_mean": cv_out[
                        "test_balanced_accuracy"
                    ].mean(),
                    "balanced_accuracy_std": cv_out[
                        "test_balanced_accuracy"
                    ].std(),
                }
            )

            # -- Per-fold weights and scores --
            actual_n_folds = len(cv_out["estimator"])
            if layer == 0:
                fold_weights_all = np.zeros(
                    (n_layers, actual_n_folds, d_model)
                )
                fold_scores_all = np.zeros((n_layers, actual_n_folds))

            fold_scores_all[layer] = cv_out["test_roc_auc"]
            for fi, pipe in enumerate(cv_out["estimator"]):
                w_fold = pipe.named_steps["clf"].coef_[0]
                norm_fold = np.linalg.norm(w_fold)
                fold_weights_all[layer, fi] = (
                    w_fold / norm_fold if norm_fold > 0 else w_fold
                )

            # -- Full-data fit --
            pipe_fit = _make_pipeline(use_scaler)
            pipe_fit.fit(X_layer, y)

            clf = pipe_fit.named_steps["clf"]
            w = clf.coef_[0]
            weights_raw[layer] = w
            norm = np.linalg.norm(w)
            weights[layer] = w / norm if norm > 0 else w
            biases[layer] = clf.intercept_[0]

            if use_scaler:
                scalers.append(pipe_fit.named_steps["scaler"])
            else:
                scalers.append(None)
            classifiers.append(clf)

        results[pos_name] = ProbeResult(
            cv_scores=pd.DataFrame(rows),
            weights=weights,
            weights_raw=weights_raw,
            biases=biases,
            scalers=scalers,
            classifiers=classifiers,
            pos_name=pos_name,
            cv_mode=cv_mode,
            use_scaler=use_scaler,
            fold_weights=fold_weights_all,
            fold_scores=fold_scores_all,
            fold_group_names=fold_group_names,
        )

    return results


# ── Fold-Level CMD ───────────────────────────────────────────────────────────


def compute_fold_cmds(
    activations: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    pos_name: str,
    layer: int,
    *,
    n_folds: int = 8,
) -> np.ndarray:
    """Compute CMD (class-mean difference) on each fold's training set.

    For each fold of a GroupKFold split, compute
    ``mean(X[y==1]) - mean(X[y==0])`` on the training indices, then
    unit-normalize.

    Parameters
    ----------
    activations : dict mapping position name -> (n_samples, n_layers, d_model)
    y : binary labels, shape (n_samples,)
    groups : group labels for GroupKFold
    pos_name : which token position to use
    layer : which layer to extract
    n_folds : number of folds (must match n_unique_groups for LOGO)

    Returns
    -------
    dict[str, np.ndarray]
        Maps held-out group name -> unit-norm CMD vector of shape ``(d_model,)``.
    """
    X_layer = activations[pos_name][:, layer, :]
    cv = GroupKFold(n_splits=n_folds)

    cmds: dict[str, np.ndarray] = {}
    for train_idx, val_idx in cv.split(X_layer, y, groups):
        held_out = str(groups[val_idx[0]])
        X_train = X_layer[train_idx]
        y_train = y[train_idx]
        cmd = X_train[y_train == 1].mean(axis=0) - X_train[y_train == 0].mean(axis=0)
        norm = np.linalg.norm(cmd)
        cmds[held_out] = cmd / norm if norm > 0 else cmd

    return cmds


def compute_constraint_cmds(
    activations: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    pos_name: str,
    layer: int,
) -> dict[str, np.ndarray]:
    """Compute CMD using only samples from each individual constraint type.

    For each unique group, restrict to that group's samples and compute
    ``mean(X[y==1]) - mean(X[y==0])``, then unit-normalize.

    Parameters
    ----------
    activations : dict mapping position name -> (n_samples, n_layers, d_model)
    y : binary labels, shape (n_samples,)
    groups : group labels (constraint types)
    pos_name : which token position to use
    layer : which layer to extract

    Returns
    -------
    dict[str, np.ndarray]
        Maps group name -> unit-norm CMD vector of shape ``(d_model,)``.
        Groups where one class is absent are omitted.
    """
    X_layer = activations[pos_name][:, layer, :]
    cmds: dict[str, np.ndarray] = {}

    for group_name in np.unique(groups):
        mask = groups == group_name
        X_g = X_layer[mask]
        y_g = y[mask]
        if y_g.sum() == 0 or y_g.sum() == len(y_g):
            continue
        cmd = X_g[y_g == 1].mean(axis=0) - X_g[y_g == 0].mean(axis=0)
        norm = np.linalg.norm(cmd)
        cmds[str(group_name)] = cmd / norm if norm > 0 else cmd

    return cmds


# ── Control Probe ────────────────────────────────────────────────────────────


def probe_control(
    activations: dict[str, np.ndarray],
    y: np.ndarray,
    token_positions: list[str],
    *,
    n_permutations: int = 1,
    cv_mode: Literal["stratified", "grouped"] = "stratified",
    groups: np.ndarray | None = None,
    n_folds: int = 5,
    use_scaler: bool = True,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Probe with permuted labels to establish chance-level baseline.

    Now supports both stratified and grouped CV modes.

    Returns
    -------
    dict mapping position name -> DataFrame with columns:
        layer, roc_auc_mean, roc_auc_std,
        balanced_accuracy_mean, balanced_accuracy_std
    """
    cv = make_cv_splitter(cv_mode, n_folds)
    cv_kwargs: dict = {"groups": groups} if cv_mode == "grouped" else {}
    rng = np.random.default_rng(seed)

    results: dict[str, pd.DataFrame] = {}

    for pos_name in token_positions:
        X = activations[pos_name]
        n_layers = X.shape[1]

        # (n_permutations, n_layers) for each metric
        auc_scores = np.zeros((n_permutations, n_layers))
        bacc_scores = np.zeros((n_permutations, n_layers))

        for p in range(n_permutations):
            y_perm = rng.permutation(y)
            for layer in range(n_layers):
                X_layer = X[:, layer, :]
                pipe = _make_pipeline(use_scaler)
                cv_out = cross_validate(
                    pipe,
                    X_layer,
                    y_perm,
                    cv=cv,
                    scoring=_SCORING,
                    n_jobs=-1,
                    **cv_kwargs,
                )
                auc_scores[p, layer] = cv_out["test_roc_auc"].mean()
                bacc_scores[p, layer] = cv_out[
                    "test_balanced_accuracy"
                ].mean()

        rows = [
            {
                "layer": layer,
                "roc_auc_mean": auc_scores[:, layer].mean(),
                "roc_auc_std": auc_scores[:, layer].std(),
                "balanced_accuracy_mean": bacc_scores[:, layer].mean(),
                "balanced_accuracy_std": bacc_scores[:, layer].std(),
            }
            for layer in range(n_layers)
        ]
        results[pos_name] = pd.DataFrame(rows)

    return results


# ── Persistence ─────────────────────────────────────────────────────────────


def save_results(
    results: dict[str, ProbeResult],
    path: Path,
    *,
    model_name: str = "",
) -> None:
    """Save complete probe results (CV scores + fitted classifiers) via joblib.

    Parameters
    ----------
    results : dict mapping position name -> ProbeResult
        The full output of :func:`probe_and_fit`.
    path : Path
        File path to write (typically ``*.joblib``).
    model_name : str
        Stored in metadata for provenance.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": results,
        "model_name": model_name,
    }
    joblib.dump(payload, path, compress=3)
    positions = list(results.keys())
    n_layers = len(next(iter(results.values())).classifiers)
    print(f"Saved results: {path.name} ({n_layers} layers, positions={positions})")


def load_results(path: Path) -> dict[str, ProbeResult]:
    """Load complete probe results saved by :func:`save_results`.

    Returns
    -------
    dict mapping position name -> ProbeResult
    """
    path = Path(path)
    payload = joblib.load(path)
    results = payload["results"]
    positions = list(results.keys())
    n_layers = len(next(iter(results.values())).classifiers)
    print(f"Loaded results: {path.name} ({n_layers} layers, positions={positions})")
    return results


def results_path(
    run_dir: Path,
    cv_mode: str,
    use_scaler: bool,
) -> Path:
    """Canonical file path for probe results within a run directory."""
    scaler_label = "scaled" if use_scaler else "unscaled"
    return run_dir / f"results_{cv_mode}_{scaler_label}.joblib"


def save_classifiers(
    probe_result: ProbeResult,
    path: Path,
    *,
    model_name: str = "",
) -> None:
    """Save fitted probe classifiers and metadata via joblib.

    .. deprecated:: Use :func:`save_results` to persist the full ProbeResult
       including CV scores.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_layers = len(probe_result.classifiers)
    d_model = probe_result.weights.shape[1]
    payload = {
        "classifiers": probe_result.classifiers,
        "scalers": probe_result.scalers,
        "weights": probe_result.weights,
        "weights_raw": probe_result.weights_raw,
        "biases": probe_result.biases,
        "metadata": {
            "pos_name": probe_result.pos_name,
            "cv_mode": probe_result.cv_mode,
            "use_scaler": probe_result.use_scaler,
            "model_name": model_name,
            "n_layers": n_layers,
            "d_model": d_model,
        },
    }
    joblib.dump(payload, path, compress=3)
    print(f"Saved classifiers: {path} ({n_layers} layers, d_model={d_model})")


def load_classifiers(path: Path) -> dict:
    """Load persisted probe classifiers.

    Returns dict with: classifiers, scalers, weights, weights_raw, biases,
    metadata.

    .. deprecated:: Use :func:`load_results` for full ProbeResult objects.
    """
    path = Path(path)
    payload = joblib.load(path)
    meta = payload["metadata"]
    print(
        f"Loaded: {path.name} ({meta['n_layers']} layers, "
        f"pos={meta['pos_name']})"
    )
    return payload
