# Steering Analysis Report

Linear probe directions for activation steering on Llama-3.1-8B-Instruct instruction hierarchy conflicts.

**Date**: 2026-04-03
**Run ID**: `curated4-8b-v001`
**Model**: `meta-llama/Llama-3.1-8B-Instruct`
**Conflicts**: `json_only_vs_plain`, `list_bullets_vs_numbered`, `past_vs_present_tense`, `starting_word_hello_greetings`
**Samples**: 9,939 Condition C (hierarchy conflict), counterbalanced (a\_to\_b and b\_to\_a directions)

## 1. Probe Validity

A logistic regression probe trained on last-prompt-token residual stream activations separates `followed_system` vs `followed_user` with high accuracy.

**Cross-validated AUC** (grouped CV, leave-one-conflict-out, 4 folds):

| Layer | AUC mean | AUC std |
|-------|----------|---------|
| 16    | 0.867    | 0.119   |
| 25    | 0.870    | 0.106   |

**Full-data probe AUC** (refit on all 4 conflicts):

| Layer | AUC   | Cohen's d | \|Δmean\| |
|-------|-------|-----------|-----------|
| 16    | 0.983 | 2.94      | 0.81      |
| 25    | 0.990 | 3.38      | 1.21      |

The probe was trained and evaluated using `probe_and_fit_gpu` (custom batched LBFGS on GPU). Stored fold weights reproduce the stored fold AUC scores exactly.

### Counterbalancing rules out constraint identity

Each conflict is counterbalanced: `followed_system` includes samples where the system prompt requested constraint A and the user requested B, *and* samples where the system requested B and the user requested A. The probe cannot learn "JSON-style activations = class 1" — it must learn something about which instruction source (system vs user) the model is following.

### Data alignment matters

Activations are stored in `conflict_id`-sorted order (the notebook does `df.sort_values("conflict_id")` before extraction). Any analysis must apply the same sort to the label dataframe. Without this, projections appear random (AUC ≈ 0.5, d ≈ 0.13). With correct alignment: AUC > 0.98, d ≈ 3.

## 2. Two Steering Directions

### Probe direction

The logistic regression weight vector (unit-normalized). Optimized to maximize classification margin — achieves higher Cohen's d than the CMD by compressing within-class variance.

### CMD (class-mean difference)

`mean(X[y=1]) - mean(X[y=0])` averaged across CV folds, then unit-normalized. Maximizes absolute mean separation but does not account for variance. Raw CMD norms per conflict range from 1.8 to 5.0 (layer 16) and 2.5 to 10.2 (layer 25).

### Relationship between directions

| Layer | cos(probe, CMD) |
|-------|-----------------|
| 16    | 0.287           |
| 25    | 0.165           |

The directions are weakly aligned — they capture related but distinct aspects of the sys/usr distinction.

## 3. Within-Conflict Separation

The probe finds strong separation *within* each conflict. This is not obvious from pooled statistics because each conflict occupies a different region of activation space.

### Layer 16

**Along probe direction (unit-norm):**

| Conflict | \|Δmean\| | Cohen's d | AUC |
|----------|-----------|-----------|-----|
| json\_only\_vs\_plain | 0.961 | 3.74 | 0.997 |
| list\_bullets\_vs\_numbered | 0.918 | 3.93 | 0.990 |
| past\_vs\_present\_tense | 0.362 | 2.27 | 0.934 |
| starting\_word\_hello\_greetings | 1.014 | 3.93 | 0.992 |

**Along CMD direction (unit-norm):**

| Conflict | \|Δmean\| | Cohen's d | AUC |
|----------|-----------|-----------|-----|
| json\_only\_vs\_plain | 1.355 | 1.56 | 0.883 |
| list\_bullets\_vs\_numbered | 3.958 | 3.47 | 0.976 |
| past\_vs\_present\_tense | 0.811 | 0.85 | 0.719 |
| starting\_word\_hello\_greetings | 4.056 | 3.68 | 0.984 |

### Layer 25

**Along probe direction (unit-norm):**

| Conflict | \|Δmean\| | Cohen's d | AUC |
|----------|-----------|-----------|-----|
| json\_only\_vs\_plain | 1.716 | 4.95 | 0.999 |
| list\_bullets\_vs\_numbered | 1.143 | 4.19 | 0.995 |
| past\_vs\_present\_tense | 0.679 | 2.39 | 0.961 |
| starting\_word\_hello\_greetings | 1.413 | 4.29 | 0.996 |

**Along CMD direction (unit-norm):**

| Conflict | \|Δmean\| | Cohen's d | AUC |
|----------|-----------|-----------|-----|
| json\_only\_vs\_plain | 4.477 | 2.26 | 0.964 |
| list\_bullets\_vs\_numbered | 9.352 | 3.30 | 0.977 |
| past\_vs\_present\_tense | 2.482 | 0.77 | 0.707 |
| starting\_word\_hello\_greetings | 10.175 | 3.44 | 0.979 |

`past_vs_present_tense` is consistently the weakest conflict across both directions and both layers.

## 4. Distribution Geometry: Stripes vs Clean Split

### Probe direction: clean global split

Projecting all samples onto the probe direction and sorting by group mean:

**Layer 25:**

```
Group          p10      p50      p90     mean
json_onl_U  -1.584   -1.203   -0.751   -1.179
starting_U  -1.571   -1.128   -0.477   -1.075
list_bul_U  -1.217   -1.005   -0.344   -0.877
past_vs__U  -0.954   -0.472   -0.047   -0.483
                                               ← gap
past_vs__S  -0.052    0.205    0.460    0.196
list_bul_S   0.056    0.290    0.464    0.266
starting_S   0.054    0.336    0.575    0.339
json_onl_S   0.091    0.541    1.039    0.536
```

All U distributions sit on the left, all S on the right. No interleaving. The gap between the rightmost U (past\_vs\_present p90 = -0.047) and leftmost S (past\_vs\_present p10 = -0.052) is essentially zero — the two classes just barely touch, and only for the weakest conflict.

### CMD direction: interleaved stripes

```
Group          p10      p50      p90     mean
json_onl_U  -8.009   -6.375   -4.455   -6.204
starting_U  -7.683   -5.401    2.013   -4.122
list_bul_U  -5.969   -3.835    3.906   -2.332
json_onl_S  -3.964   -3.030    2.047   -1.727   ← S interleaved with U
past_vs__U  -3.553    0.453    5.693    0.689
past_vs__S  -1.760    3.746    6.841    3.172
starting_S   2.397    6.526    8.356    6.053
list_bul_S   5.670    7.096    8.412    7.021
```

The CMD direction shows interleaving: `json_onl_S` (mean=-1.73) sits between `list_bul_U` (-2.33) and `past_vs__U` (0.69). U < S holds within each conflict, but not globally. The distributions are also much wider (CMD std ≈ 1-4 vs probe std ≈ 0.2-0.4).

### Interpretation

The probe direction finds a low-variance projection where the sys/usr boundary is consistent across conflicts. The CMD direction maximizes raw mean separation but doesn't control variance, resulting in wide, overlapping distributions. This is why the probe has higher Cohen's d despite smaller absolute separation: it compresses within-class scatter.

## 5. Cross-Validated Fold Analysis

Each CV fold trains on 3 conflicts and tests on the held-out 4th. The fold-specific probe directions differ substantially:

**Fold direction cosines (layer 16):**

| | Fold 0 | Fold 1 | Fold 2 | Fold 3 |
|------|--------|--------|--------|--------|
| Fold 0 | 1.00 | 0.39 | 0.60 | 0.54 |
| Fold 1 | | 1.00 | 0.77 | 0.74 |
| Fold 2 | | | 1.00 | 0.88 |
| Fold 3 | | | | 1.00 |

Each fold direction achieves AUC 0.93-0.99 on its training conflicts (within-conflict) and transfers to the held-out conflict with degraded but still meaningful performance:

| Held-out conflict | Fold AUC |
|-------------------|----------|
| past\_vs\_present\_tense | 0.678 |
| json\_only\_vs\_plain | 0.854 |
| list\_bullets\_vs\_numbered | 0.976 |
| starting\_word\_hello\_greetings | 0.959 |

This transfer — probe trained on other conflicts separating sys/usr in an unseen conflict — is the core finding. It suggests a shared "system-following" representation that varies by conflict but has a common linear component.

## 6. Alpha Calibration for Steering

### Method

Steering adds `alpha * unit_direction` to the hidden state at the target layer via a PyTorch forward hook, at every token position during generation. Alpha selection must balance:

1. **Farthest U must reach S**: U\_p10 + alpha should reach at least S\_p50
2. **Closest U must enter S**: U\_p90 + alpha should exceed S\_p10
3. **S must not overshoot**: S\_p90 + alpha should not go far beyond the S range

### Layer 25 — Probe direction

| Conflict | U \[p10..p90\] | S \[p10..p90\] | Gap | min α | sweet spot | max α |
|----------|---------------|---------------|-----|-------|------------|-------|
| json\_only | \[-1.58, -0.75\] | \[0.09, 1.04\] | +0.84 | 0.84 | 2.12 | 0.95 |
| list\_bullets | \[-1.22, -0.34\] | \[0.06, 0.46\] | +0.40 | 0.40 | 1.51 | 0.41 |
| past\_vs\_present | \[-0.95, -0.05\] | \[-0.05, 0.46\] | -0.01 | ~0 | 1.16 | 0.51 |
| starting\_word | \[-1.57, -0.48\] | \[0.05, 0.58\] | +0.53 | 0.53 | 1.91 | 0.52 |

- **min α ≈ max α** for most conflicts: the alpha needed to start flipping the closest U samples is approximately the same as the alpha at which S starts overshooting.
- No single alpha perfectly flips all U→S without side effects.

### Layer 25 — CMD direction

| Conflict | U \[p10..p90\] | S \[p10..p90\] | Gap | min α | sweet spot | max α |
|----------|---------------|---------------|-----|-------|------------|-------|
| json\_only | \[-8.01, -4.46\] | \[-3.96, 2.05\] | +0.49 | 0.49 | 4.98 | 6.01 |
| list\_bullets | \[-5.97, 3.91\] | \[5.67, 8.41\] | +1.76 | 1.76 | 13.06 | 2.74 |
| past\_vs\_present | \[-3.55, 5.69\] | \[-1.76, 6.84\] | -7.45 | N/A | 7.30 | 8.60 |
| starting\_word | \[-7.68, 2.01\] | \[2.40, 8.36\] | +0.38 | 0.38 | 14.21 | 5.96 |

CMD has wider distributions and more variation across conflicts, making uniform alpha selection harder.

### Layer comparison

| | Layer 16 | Layer 25 |
|---|---|---|
| \|\|h\|\| | 9.6 | 22.7 |
| Probe \|Δmean\| | 0.81 | 1.21 |
| Headroom (alpha=1 as % \|\|h\|\|) | 10.4% | 4.4% |
| S distribution width (probe) | 0.3–0.5 | 0.4–0.9 |

Layer 25 chosen for steering: higher AUC, larger separation, 2.4× more headroom, wider S distributions (more room before overshoot).

### Selected alphas

```python
ALPHAS = {
    "probe": [0.5, 1, 1.5],
    "cmd_overall": [2, 5, 10],
}
STEER_LAYER = 25
```

## 7. Implementation

### Steering mechanism

Steered generation uses a PyTorch `register_forward_hook` on the target decoder layer. The hook adds `alpha * direction_vector` to the layer output at every forward pass during `model.generate()`.

```python
def _steering_hook(module, input, output):
    if isinstance(output, tuple):
        return (output[0] + alpha * steer_vec, *output[1:])
    return output + alpha * steer_vec

handle = _get_decoder_layers(hf_model)[layer].register_forward_hook(_steering_hook)
```

The `isinstance(output, tuple)` check handles architecture differences: Llama/Qwen2/Mistral decoder layers return a plain tensor, Gemma2/Gemma3 return a tuple `(hidden_states, attn_weights)`.

`_get_decoder_layers()` resolves the decoder layer list for different architectures:
- Standard (Llama, Gemma2, Qwen2, Mistral): `model.model.layers`
- Multimodal wrappers (Gemma3): `model.model.language_model.layers`

### Why not nnsight

nnsight was upgraded from 0.5.15 to 0.6.3. The `tracer.all()` API works in top-level scripts but fails when called from functions in imported modules due to source analysis issues. PyTorch hooks are simpler (~10 lines), have no nnsight dependency for generation, and enable true batching.

### Three baselines

| Condition | Backend | Hook | Purpose |
|-----------|---------|------|---------|
| `phase0` | Stored responses from Phase 0 | None | Original behavioral baseline |
| `unsteered` | Raw HF `generate()` | None | Re-generation baseline |
| `unsteered_hooked` | HF `generate()` | Registered, alpha=0 | Controls for hook overhead |

### Per-direction alphas

`run_condition_comparison()` accepts `alphas` as either a `list[float]` (same for all directions) or `dict[str, list[float]]` (per-direction). This was necessary because the probe and CMD directions have very different scales of separation.

### Caching

Results are cached as per-conflict JSONL files in `{run_dir}/steering/{condition}/`. Steered results are cached per alpha: `{conflict_id}_alpha_{alpha}.jsonl`. The pipeline skips cached files on restart.

### Files

| File | Role |
|------|------|
| `steer.py` | Steering directions, batched generation, scoring, `run_condition_comparison()` |
| `run_steering_pipeline.py` | Headless script: loads model, runs all conditions, prints summary |
| `steering_experiment.ipynb` | Visualization notebook (loads from cache) |
| `tests/test_steer.py` | 26 unit tests (synthetic data, no GPU) |
| `tests/test_steer_gpu.py` | 7 GPU integration tests |

## 8. Open Questions

- **Behavioral results**: Does the SCR actually shift with steering? Pipeline running as of 2026-04-03.
- **Per-conflict directions**: The global direction is a compromise. Per-conflict steering vectors might work better, especially for `past_vs_present_tense`.
- **Why is `past_vs_present_tense` weak?** It has the smallest CMD norm, most overlap between U and S, and lowest within-conflict separation on both directions. The constraint might be harder for the model to distinguish at the representation level.
- **Nonlinearity**: The probe achieves d=2.9–3.4 with a linear direction, but maybe a nonlinear probe could separate even better and suggest more nuanced interventions.
- **Multi-layer steering**: Intervening at multiple layers simultaneously might be more effective than single-layer.
- **More conflicts**: Would training on more than 4 conflicts improve the global direction's quality?

---

## Appendix A: Loading Data and Directions

All analysis scripts share this preamble. The critical detail is
`sort_values("conflict_id")` — activations are stored in this order, and
without it labels are misaligned (producing spurious AUC ≈ 0.5).

```python
import sys, numpy as np
sys.path.insert(0, "phase1_linear_probing")
from pathlib import Path
from probe import load_results, results_path
from data import load_results as load_data, prepare_condition_c, load_sync_env
from sklearn.metrics import roc_auc_score

run_dir = Path("phase1_linear_probing/data/runs/curated4-8b-v001")

# Probe results (weights, fold weights, cv scores)
rpath = results_path(run_dir, cv_mode="grouped", use_scaler=False)
results = load_results(rpath)
pr = results["last_prompt"]

# Activations: shape (9939, 32, 4096) — (samples, layers, d_model)
act_data = np.load(
    run_dir / "act_nn_meta-llama_Llama-3.1-8B-Instruct.npz"
)

# Labels — MUST sort by conflict_id to align with activations
load_sync_env(run_dir.parent.parent)
df_all = load_data(
    Path("phase0_v2/data/results"),
    "meta-llama/Llama-3.1-8B-Instruct",
)
df_c = prepare_condition_c(
    df_all, "binary",
    conflict_ids=[
        "json_only_vs_plain",
        "list_bullets_vs_numbered",
        "past_vs_present_tense",
        "starting_word_hello_greetings",
    ],
)
df_c = df_c.sort_values("conflict_id").reset_index(drop=True)  # critical!
y = df_c["y"].values  # 1 = followed_system, 0 = followed_user

# Extract layer activations and directions
layer = 25  # or 16
X = act_data["last_prompt"][:, layer, :]  # (9939, 4096)
w_probe = pr.weights[layer]               # unit-norm probe direction

# CMD direction (mean of fold CMDs, then unit-normalized)
fold_cmds = np.load(run_dir / f"fold_cmds_L{layer}.npz")
vecs = np.stack([fold_cmds[k] for k in fold_cmds.files])
cmd_raw = vecs.mean(axis=0)
cmd_unit = cmd_raw / np.linalg.norm(cmd_raw)
```

## Appendix B: Global Separation and Cohen's d

```python
for name, d_vec in [("probe", w_probe), ("cmd_overall", cmd_unit)]:
    proj = X @ d_vec
    ps, pu = proj[y == 1], proj[y == 0]
    sep = ps.mean() - pu.mean()
    pooled = np.sqrt((ps.std() ** 2 + pu.std() ** 2) / 2)
    d = sep / pooled
    auc = roc_auc_score(y, proj)
    print(f"{name:15s}: |Δmean|={abs(sep):.4f}  d={d:.3f}  AUC={auc:.3f}")
```

## Appendix C: Within-Conflict Separation

```python
for name, d_vec in [("probe", w_probe), ("cmd_overall", cmd_unit)]:
    print(f"=== {name} ===")
    for cid in sorted(df_c["conflict_id"].unique()):
        mask = (df_c["conflict_id"] == cid).values
        proj = X[mask] @ d_vec
        y_cid = y[mask]
        ps, pu = proj[y_cid == 1], proj[y_cid == 0]
        sep = abs(ps.mean() - pu.mean())
        pooled = np.sqrt((ps.std() ** 2 + pu.std() ** 2) / 2)
        d = sep / pooled
        auc = roc_auc_score(y_cid, proj)
        print(f"  {cid:40s}  d={d:.3f}  AUC={auc:.3f}")
```

## Appendix D: Stripe Diagram (p10/p50/p90 Ranges)

Produces the distribution range tables and checks for U/S interleaving.

```python
for name, d_vec in [("probe", w_probe), ("cmd_overall", cmd_unit)]:
    proj = X @ d_vec
    print(f"=== {name} direction (layer {layer}) ===")

    entries = []
    for cid in sorted(df_c["conflict_id"].unique()):
        mask = (df_c["conflict_id"] == cid).values
        for label, lname in [(0, "U"), (1, "S")]:
            m = mask & (y == label)
            p = proj[m]
            short = cid[:8]
            entries.append((p.mean(), f"{short}_{lname}", p))

    entries.sort(key=lambda x: x[0])
    print(f"{'group':>12s}  {'p10':>7s}  {'p25':>7s}  {'p50':>7s}  "
          f"{'p75':>7s}  {'p90':>7s}  {'mean':>7s}")
    for _, ename, p in entries:
        print(f"{ename:>12s}  {np.percentile(p, 10):7.3f}  "
              f"{np.percentile(p, 25):7.3f}  {np.percentile(p, 50):7.3f}  "
              f"{np.percentile(p, 75):7.3f}  {np.percentile(p, 90):7.3f}  "
              f"{p.mean():7.3f}")

    # Check ordering
    print("\n  U < S within each conflict?")
    for cid in sorted(df_c["conflict_id"].unique()):
        mask = (df_c["conflict_id"] == cid).values
        mean_s = proj[mask & (y == 1)].mean()
        mean_u = proj[mask & (y == 0)].mean()
        ok = "yes" if mean_u < mean_s else "NO — REVERSED"
        print(f"    {cid:40s}  mean_U={mean_u:.3f}  mean_S={mean_s:.3f}  {ok}")
```

## Appendix E: Alpha Calibration (Range-Based)

Computes per-conflict min alpha (closest U enters S), sweet-spot alpha
(farthest U reaches S center), and max alpha (S overshoot threshold) using
p10/p90 ranges rather than means.

```python
for name, d_vec in [("probe", w_probe), ("cmd_overall", cmd_unit)]:
    proj = X @ d_vec
    print(f"=== {name} (layer {layer}) ===")
    print(f"  {'conflict':>12s}  {'min alpha':>10s}  {'sweet spot':>11s}  "
          f"{'max alpha':>10s}")
    print(f"  {'':>12s}  {'(close U→S)':>10s}  {'(far U→S50)':>11s}  "
          f"{'(overshoot)':>10s}")
    print(f"  " + "-" * 50)

    for cid in sorted(df_c["conflict_id"].unique()):
        mask = (df_c["conflict_id"] == cid).values
        pu = proj[mask & (y == 0)]
        ps = proj[mask & (y == 1)]

        u_p10 = np.percentile(pu, 10)  # farthest U from boundary
        u_p90 = np.percentile(pu, 90)  # closest U to boundary
        s_p10 = np.percentile(ps, 10)  # closest S to boundary
        s_p50 = np.median(ps)          # S center
        s_p90 = np.percentile(ps, 90)  # farthest S from boundary
        s_range = s_p90 - s_p10        # width of S distribution

        # min alpha: push closest U (p90) just past S entry (p10)
        min_alpha = s_p10 - u_p90

        # sweet spot: push farthest U (p10) to S center (p50)
        sweet = s_p50 - u_p10

        # max alpha: pushing S_p90 by more than S_range sends it
        # 1× the S width beyond the S distribution
        max_alpha = s_range

        print(f"  {cid[:12]:>12s}  {min_alpha:10.3f}  "
              f"{sweet:11.3f}  {max_alpha:10.3f}")

    # Also print ranges
    print()
    print(f"  {'conflict':>12s}  {'U [p10..p90]':>20s}  "
          f"{'S [p10..p90]':>20s}  {'gap':>6s}")
    print(f"  " + "-" * 55)
    for cid in sorted(df_c["conflict_id"].unique()):
        mask = (df_c["conflict_id"] == cid).values
        pu = proj[mask & (y == 0)]
        ps = proj[mask & (y == 1)]
        u10, u90 = np.percentile(pu, 10), np.percentile(pu, 90)
        s10, s90 = np.percentile(ps, 10), np.percentile(ps, 90)
        gap = s10 - u90
        print(f"  {cid[:12]:>12s}  [{u10:7.3f} .. {u90:7.3f}]  "
              f"[{s10:7.3f} .. {s90:7.3f}]  {gap:+6.3f}")
```

## Appendix F: Cross-Validated Fold Reproducibility

Verifies that stored fold weights reproduce stored fold AUC scores,
confirming data integrity.

```python
from sklearn.model_selection import GroupKFold

groups = df_c["constraint_type"].values
gkf = GroupKFold(n_splits=4)

print(f"Fold group names: {pr.fold_group_names}")
for fi, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    held_out = np.unique(groups[val_idx])[0]
    w_fold = pr.fold_weights[layer, fi]  # unit-norm

    # Score using fold weight (no bias needed for AUC — only ranking matters)
    score = X[val_idx] @ w_fold
    auc = roc_auc_score(y[val_idx], score)
    stored = pr.fold_scores[layer, fi]

    print(f"Fold {fi} ({held_out:40s}): "
          f"stored={stored:.3f}  reproduced={auc:.3f}  "
          f"match={np.isclose(stored, auc, atol=1e-3)}")
```

## Appendix G: Cross-Conflict AUC Matrix

Shows how each fold's direction performs on every conflict (not just its
held-out one). Reveals that directions trained on 3 conflicts transfer to
the 4th.

```python
from sklearn.model_selection import GroupKFold

groups = df_c["constraint_type"].values
conflicts = sorted(df_c["conflict_id"].unique())
gkf = GroupKFold(n_splits=4)

print("Row = fold direction, Col = conflict tested on")
header = "                       " + "  ".join(
    f"{c[:12]:>12s}" for c in conflicts
)
print(header)

for fi, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    held_out = np.unique(groups[val_idx])[0]
    w = pr.fold_weights[layer, fi]

    row = f"  Fold {fi} (out={held_out[:12]:12s})"
    for cid in conflicts:
        mask = (df_c["conflict_id"] == cid).values
        score = X[mask] @ w
        y_cid = y[mask]
        auc = roc_auc_score(y_cid, score)
        star = "*" if cid == held_out else " "
        row += f"  {auc:11.3f}{star}"
    print(row)
```

## Appendix H: Per-Conflict Raw CMD Norms

Shows that per-conflict CMDs are much larger than the overall CMD (which
averages directions that partially cancel), and how they relate to the
probe direction.

```python
cdata = np.load(run_dir / f"constraint_cmds_L{layer}.npz")

for cid in sorted(df_c["conflict_id"].unique()):
    mask = (df_c["conflict_id"] == cid).values
    X_cid = X[mask]
    y_cid = y[mask]

    # Raw CMD for this conflict
    mean_sys = X_cid[y_cid == 1].mean(axis=0)
    mean_usr = X_cid[y_cid == 0].mean(axis=0)
    cmd_raw = mean_sys - mean_usr
    cmd_norm = np.linalg.norm(cmd_raw)
    cmd_unit_c = cmd_raw / cmd_norm

    # Separation along this conflict's own CMD vs the probe
    proj_cmd = X_cid @ cmd_unit_c
    proj_probe = X_cid @ w_probe

    d_cmd = abs(
        proj_cmd[y_cid == 1].mean() - proj_cmd[y_cid == 0].mean()
    ) / np.sqrt(
        (proj_cmd[y_cid == 1].std() ** 2
         + proj_cmd[y_cid == 0].std() ** 2) / 2
    )
    d_probe = abs(
        proj_probe[y_cid == 1].mean() - proj_probe[y_cid == 0].mean()
    ) / np.sqrt(
        (proj_probe[y_cid == 1].std() ** 2
         + proj_probe[y_cid == 0].std() ** 2) / 2
    )

    cos_probe = np.dot(w_probe, cmd_unit_c)
    cos_cmd_ov = np.dot(cmd_unit, cmd_unit_c)

    print(f"{cid}:")
    print(f"  ||CMD_raw|| = {cmd_norm:.3f}")
    print(f"  d along own CMD = {d_cmd:.3f}")
    print(f"  d along probe   = {d_probe:.3f}")
    print(f"  cos(probe, this CMD) = {cos_probe:.3f}")
    print(f"  cos(cmd_overall, this CMD) = {cos_cmd_ov:.3f}")
```
