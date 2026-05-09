# Phase 1: Linear Probing

Linear probes on LLM residual-stream activations to classify whether a model follows system or user instructions under conflicting prompts (Condition C from `phase0_v2`). A logistic regression classifier is trained at each transformer layer to find the direction in activation space that separates "followed system" from "followed user" responses, with grouped cross-validation to prevent constraint-type leakage.

## Module Map

| File | Description |
|---|---|
| `data.py` | `ProbeConfig` dataclass, env loading (`load_sync_env`), data loading, tokenization, activation extraction (TransformerLens + nnsight) |
| `probe.py` | `ProbeResult` dataclass, cross-validated probing (`probe_and_fit`), permuted-label control (`probe_control`), persistence (joblib save/load) |
| `viz.py` | Plotly-based visualization: probe curves, cosine heatmaps, direction agreement, bias analysis, metadata importance, summary table |
| `metadata_clf.py` | Surface-feature baselines: linear and boosted classifiers on metadata, group ablation, with `exclude_groups` support to drop leaky feature groups under grouped CV |
| `linear_probe.ipynb` | End-to-end notebook orchestrating all modules |
| `tests/` | Pytest suite using synthetic data (no GPU or model needed) |

## Configuration (`ProbeConfig`)

Defined in `data.py`. All experiment parameters live here; derived paths are computed in `__post_init__`.

| Field | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"meta-llama/Llama-3.1-8B-Instruct"` | HuggingFace model identifier |
| `label_mode` | `"binary" \| "one_vs_rest"` | `"binary"` | `"binary"` keeps only `followed_system` / `followed_user` |
| `token_positions` | `list[str]` | `["last_prompt"]` | Positions to probe: `last_prompt`, `last_system`, `last_user`, `mean_all`, `mean_system`, `mean_user` |
| `cv_mode` | `"stratified" \| "grouped"` | `"grouped"` | Cross-validation strategy (see below) |
| `n_cv_folds` | `int` | `8` | Number of CV folds (capped to n_groups for grouped mode) |
| `max_samples` | `int \| None` | `None` | Subsample cap for fast debugging |
| `use_scaler` | `bool` | `False` | Whether to apply `StandardScaler` before logistic regression |
| `device` | `str` | auto (`"cuda"` / `"cpu"`) | PyTorch device for activation extraction |
| `repo_root` | `Path` | auto (walks up to `pyproject.toml`) | Repository root path |
| `run_id` | `str \| None` | auto (12-char SHA-256 hash) | Unique identifier for this configuration |

### `cv_mode`

- **`"grouped"`** (default): Uses `GroupKFold` with `constraint_type` as the group key. Entire constraint types are held out per fold, preventing the probe from memorizing type-specific patterns. This gives honest generalization estimates.
- **`"stratified"`**: Uses `StratifiedKFold` with balanced class ratios per fold. Samples from the same constraint type can appear in both train and test, which inflates accuracy. Useful for comparison to quantify leakage.

### `run_id`

Auto-generated as a 12-character hex hash of `(model_name, label_mode, token_positions, cv_mode, n_cv_folds, use_scaler, max_samples, data_hash)`. The `data_hash` component is a 16-character SHA-256 hash of the source JSONL file contents (computed by `_data_hash()`), so the `run_id` changes whenever the underlying behavioral data changes, even if config parameters stay the same. Changing any of these parameters produces a different `run_id` and therefore a different output directory. You can set `run_id` explicitly to override.

### Derived Paths

- `data_dir`: `{repo_root}/phase0_v2/data/results` -- source behavioral data
- `run_dir`: `{repo_root}/phase1_linear_probing/data/runs/{run_id}/`
- `reports_dir`: `{run_dir}/reports/`

## File Layout / Paths

```
phase1_linear_probing/
  data/
    activations2/                          # legacy activation storage
      act_nn_{safe_model_name}.npz         # nnsight-extracted activations
      act_tl_{safe_model_name}.npz         # TransformerLens-extracted activations
    runs/
      {run_id}/
        config.json                        # serialized ProbeConfig for reproducibility
        act_nn_{safe_model_name}.npz       # activations (when extracted into run dir)
        results_grouped_unscaled.joblib    # full ProbeResult dicts (grouped CV, no scaler)
        results_grouped_scaled.joblib      # full ProbeResult dicts (grouped CV, with scaler)
        results_stratified_unscaled.joblib # full ProbeResult dicts (stratified CV, no scaler)
        results_stratified_scaled.joblib   # full ProbeResult dicts (stratified CV, with scaler)
        reports/
          probe_{safe_model_name}_*.html   # interactive Plotly visualizations
          cosine_similarity_heatmap.html
          cosine_similarity_curves.html
          bias_analysis.html
          direction_agreement_*.html
```

## Workflow

Matches the notebook section order:

1. **Configure** -- Instantiate `ProbeConfig`, call `load_sync_env()` to load env vars, then `ensure_dirs()` and `save_config()`
2. **Load data** -- `load_results()` reads `phase0_v2` JSONL results; `prepare_condition_c()` filters to Condition C and assigns binary labels
3. **Extract/load activations** -- Either `extract_activations_nn()` / `extract_activations_tl()` for fresh extraction, or `load_activations()` for cached `.npz` files
4. **Probe** -- `probe_and_fit()` runs per-layer logistic regression with CV scoring (`roc_auc`, `balanced_accuracy`) and full-data fitting (unit-norm directions, raw weights, biases); returns `dict[str, ProbeResult]`
5. **Direction analysis** -- Compare scaled vs unscaled probe directions using `plot_cosine_heatmaps()`, `plot_cosine_curves()`, `plot_bias_analysis()`, and `plot_direction_agreement()` (with optional ROC AUC subplot when CV scores are passed)
6. **Control baselines** -- `probe_control()` (permuted labels), `run_linear_control()` / `run_boosted_control()` (metadata classifiers). Under grouped CV the notebook excludes leaky feature groups (`length_feats`, `constraint_type`, `direction`) via `exclude_groups` before building the feature matrix
7. **Visualize** -- `plot_probe_curves()`, `build_summary_table()`, and other `viz.py` functions; all save interactive HTML to `reports_dir`

## Key Functions

### `data.py`

| Function | Description |
|---|---|
| `ProbeConfig(...)` | Central config dataclass; auto-generates `run_id` (including source data hash via `_data_hash()`) and derived paths |
| `ProbeConfig.save_config() -> Path` | Serialize config as JSON in `run_dir` for reproducibility |
| `ProbeConfig.ensure_dirs()` | Create `run_dir` and `reports_dir` if they don't exist |
| `ProbeConfig.safe_model_name -> str` | Model name with `/` replaced by `_` (property) |
| `find_repo_root() -> Path` | Walk up from cwd until `pyproject.toml` is found |
| `load_sync_env(repo_root) -> Path \| None` | Auto-load first `*.sync.env` file into `os.environ` via `setdefault` |
| `load_results(data_dir, model_name) -> DataFrame` | Load `phase0_v2` JSONL results for a model |
| `prepare_condition_c(df, label_mode, max_samples) -> DataFrame` | Filter to Condition C, assign binary labels, optionally subsample |
| `build_formatted_prompt(tokenizer, system_text, user_text) -> str` | Apply chat template to system/user messages |
| `find_token_positions(tokenizer, system_text, user_text) -> dict` | Compute token-position map (last_prompt, last_system, last_user, mean_*) |
| `precompute_prompts(df, tokenizer) -> (prompts, position_maps, input_ids)` | Batch tokenization and position mapping for all rows |
| `extract_activations_tl(input_ids, position_maps, cfg) -> dict` | Extract residual-stream activations via TransformerLens |
| `extract_activations_nn(input_ids, position_maps, cfg) -> dict` | Extract residual-stream activations via nnsight |
| `save_activations(activations, path)` | Save activation arrays as compressed `.npz` |
| `load_activations(path) -> dict` | Load activation arrays from `.npz` |
| `filter_activations(activations, indices) -> dict` | Subset activations by sample indices |
| `compare_backends(act_tl, act_nn, positions)` | Print per-position cosine similarity between TL and nnsight activations |

### `probe.py`

| Function | Description |
|---|---|
| `ProbeResult` | Dataclass holding per-position results: `cv_scores` (DataFrame), `weights` (unit-norm), `weights_raw`, `biases`, `scalers`, `classifiers`, plus metadata (`pos_name`, `cv_mode`, `use_scaler`) |
| `probe_and_fit(activations, y, token_positions, ...) -> dict[str, ProbeResult]` | Per-layer CV scoring (`roc_auc`, `balanced_accuracy`) + full-data logistic regression fit; stores both unit-norm and raw weight vectors |
| `probe_control(activations, y, token_positions, ...) -> dict[str, DataFrame]` | Permuted-label baseline (chance level); supports multiple permutations via `n_permutations` and both CV modes; defaults to `use_scaler=True` |
| `make_cv_splitter(cv_mode, n_folds, ...) -> StratifiedKFold \| GroupKFold` | Create `StratifiedKFold` (with shuffle) or `GroupKFold` |
| `save_results(results, path, *, model_name)` | Persist full `dict[str, ProbeResult]` via joblib (compress=3); `model_name` is keyword-only, stored in payload metadata |
| `load_results(path) -> dict[str, ProbeResult]` | Load persisted probe results |
| `results_path(run_dir, cv_mode, use_scaler) -> Path` | Canonical file path: `results_{cv_mode}_{scaled\|unscaled}.joblib` |
| `save_classifiers(probe_result, path, ...)` | *Deprecated.* Save fitted classifiers, scalers, weights, weights_raw, biases, and metadata only |
| `load_classifiers(path) -> dict` | *Deprecated.* Load fitted classifiers |

### `viz.py`

| Function | Description |
|---|---|
| `plot_probe_curves(probe_results, positions, model_name, n_folds, ...) -> Figure` | Layer-wise AUC/accuracy curves with optional control baselines |
| `plot_probe_comparison(results_strat, results_grouped, pos, model, ...) -> Figure` | Side-by-side stratified vs grouped CV curves for one position |
| `plot_cosine_heatmaps(weights_scaled, weights_unscaled, pos, ...) -> Figure` | Cross-layer cosine similarity heatmaps (scaled vs unscaled) |
| `plot_cosine_curves(weights_scaled, weights_unscaled, pos, ...) -> Figure` | Consecutive and final-layer cosine similarity line plots |
| `plot_scaled_vs_unscaled(auc_scaled, auc_unscaled, agreement, ...) -> Figure` | Three-panel scaled vs unscaled comparison |
| `plot_bias_analysis(biases_scaled, biases_unscaled, pos, ...) -> Figure` | Probe bias (intercept) by layer; renders a second subplot with `StandardScaler` statistics when optional `scalers` list is provided, otherwise a single panel |
| `plot_direction_agreement(weights_scaled, weights_unscaled, pos, ...) -> Figure` | Per-layer cosine similarity between scaled and unscaled directions; when optional `cv_scores_scaled` and `cv_scores_unscaled` DataFrames are provided, renders a second subplot showing ROC AUC by layer for both variants |
| `plot_metadata_importance(linear_coefs, names, boosted_imps, ...) -> Figure` | Side-by-side linear coefficient and boosted importance bars |
| `plot_metadata_ablation(df_abl_lin, df_abl_bst, ...) -> Figure` | Group ablation bar charts for linear and boosted classifiers |
| `build_summary_table(probe_results, positions, n_layers, ...) -> DataFrame` | Summary table with peak metrics per backend and position |

### `metadata_clf.py`

| Function | Description |
|---|---|
| `build_category_lists(df, categorical_cols=...) -> dict` | Extract sorted unique values for each categorical column. Default `categorical_cols`: `["constraint_type", "system_style", "user_style", "task_id"]` |
| `build_metadata_features(df, position_maps, cats, *, exclude_groups=None) -> ndarray` | Build feature matrix: one-hot categoricals + token length features. `exclude_groups` is a set of group names to omit (e.g. `{"length_feats", "constraint_type", "direction"}`) |
| `get_feature_names(cats, *, exclude_groups=None) -> list[str]` | Ordered feature names matching the feature matrix columns. Respects `exclude_groups` to stay aligned with the matrix |
| `get_feature_groups(cats, *, exclude_groups=None) -> dict[str, list[int]]` | Map group name to column indices (for ablation). When `exclude_groups` is given, excluded groups are omitted and indices are recomputed to match the reduced matrix |
| `run_linear_control(X, y, ...) -> dict` | Logistic regression CV on metadata features |
| `run_boosted_control(X, y, ...) -> dict` | Nested-CV `HistGradientBoostingClassifier` on metadata features |
| `fit_boosted_importances(X, y, ...) -> dict` | Full-data boosted fit with permutation importance. Returns dict with keys `importances_mean`, `importances_std`, `best_params` |
| `run_group_ablation(X, y, feature_groups, ...) -> DataFrame` | Leave-one-feature-group-out ablation |

## Persistence / Caching

All probe results are cached per `(run_dir, cv_mode, use_scaler)` combination:

- **Canonical path**: `results_path(run_dir, cv_mode, use_scaler)` returns `{run_dir}/results_{cv_mode}_{scaled|unscaled}.joblib`
- **Cache check**: The notebook checks `rpath.exists()` before calling `probe_and_fit`; if the file exists, it calls `load_results(rpath)` instead
- **Save/load**: `save_results(results, path, *, model_name)` wraps the full `dict[str, ProbeResult]` in a payload dict (with `model_name` for provenance) and persists it via joblib with compress=3. Each `ProbeResult` contains CV scores, unit-norm weights, raw weights, biases, scalers, and classifiers. `load_results(path)` unwraps the payload and returns the `dict[str, ProbeResult]`
- **Both scaler variants**: The direction analysis section needs both scaled and unscaled weights, so both `results_{cv_mode}_scaled.joblib` and `results_{cv_mode}_unscaled.joblib` are computed and cached within the same run
- **Deprecated**: `save_classifiers` / `load_classifiers` persist only the fitted classifiers without CV scores. Still available but superseded by `save_results` / `load_results`

## Running Tests

```bash
uv run pytest phase1_linear_probing/tests/ -v
```

All tests use synthetic data generated in `tests/conftest.py` (50 samples, 4 layers, 16 dims). No GPU, model downloads, or real experiment data required.

Test coverage by module:
- `test_data.py` -- `ProbeConfig` fields, derived paths, `run_id` determinism, data loading/filtering, activation slicing and I/O
- `test_probe.py` -- CV splitters, `probe_and_fit` output shapes and metrics, scaler behavior, permuted-label control, save/load roundtrips, `results_path`
- `test_viz.py` -- All plot functions return `go.Figure`, file saving, `build_summary_table` columns and row counts
- `test_metadata_clf.py` -- Linear and boosted controls return both metrics, grouped CV support, group ablation output, backward-compat aliases
