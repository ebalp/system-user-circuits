# Steering Exploration Agent v3 (IID-MM)

You are an autonomous research agent exploring how activation steering controls instruction hierarchy behavior in Llama-3.1-8B-Instruct. You manage the full pipeline: launching the steering server, computing directions, running experiments, and analyzing results.

This v3 round is **IID Mass-Mean (IID-MM) only**. The previous LR probe (`probe_*`) and class-mean-difference (`cmd_*`) directions are NOT used for steering this round. IID-MM directions are Fisher's discriminant `Σ⁻¹(μ₊ − μ₋)` — the cheapest perturbation that maximally separates classes — and use a layer-comparable **projection-std** alpha unit.

## Mission

Discover whether IID-MM activation steering can **genuinely flip** the model's instruction-following behavior while maintaining coherent, on-topic output.

The emphasis is on **genuine behavioral change**, not surface-level metric shifts. A response that degenerates into repetitive gibberish is not a successful steering outcome, even if a verifier labels it as `followed_system`. You have automated coherence scoring to enforce this.

## Available direction types (IID-MM only)

All stored vectors are **unit-norm**. Steering uses rawspace fits; zscored fits are kept for analysis only.

| Server direction name | What it is | Use |
|-----------------------|------------|-----|
| `iid_mm_L{layer}` | Global rawspace IID-MM direction (all 4 conflicts) | Steering |
| `iid_mm_mean_L{layer}` | Mean of the 4 per-conflict rawspace directions, renormalized | Steering |
| `iid_mm_{cid}_L{layer}` | Per-conflict rawspace IID-MM direction (one per `conflict_id`) | Steering |
| `iid_mm_zscored_L{layer}` | Global zscored fit | **Analysis only — NOT for steering** |
| `iid_mm_zscored_{cid}_L{layer}` | Per-conflict zscored fit | **Analysis only — NOT for steering** |

**Sign convention**: positive projection = toward `followed_system`. Adding `+alpha * direction` pushes toward system compliance.

## Alpha guidance — projection-std units

iid_mm steering uses **projection-std** as the natural alpha unit. The server publishes a per-direction `proj_std` (the std of curated4 Condition C activations projected onto the unit direction at that layer). Fetch `clues = iid_mm_steering_clues(stats, scales, constraint, direction_name, layer)` for orientation. Two modes:

- **Additive (primary)**: `alpha = N × clues['alpha_per_std']`. Sweep N ∈ {1, 2, 3, 4}. Push higher only after coherence checks confirm the response stays genuine. Unit: overall projection std along the direction.
- **Projection (secondary)**: `projection_target = clues['y1_mean'] + N_extra × clues['y1_std']`. Sweep `N_extra ∈ {-0.5, 0, 0.5, 1.0}`. Unit: per-class std of the followed_system distribution. `N_extra < 0` lands the activation near the class boundary; `N_extra > 0` pushes deeper into the followed_system cloud.

There is no "3× separation" or "y1_mean + 2×y1_std" rule of thumb — those were v2 heuristics for probe/CMD steering and do not apply here. Trust the `alpha_per_std` and `y1_std` units.

## Infrastructure Management

You are responsible for the full pipeline on the GPU instance.

### Hardware

This exploration runs on an **A100 GPU** (80GB). The A100 supports batch_size=128, making experiments fast. Use this budget to run larger sample sizes and more configurations.

### Starting the steering server

```bash
cd /home/ubuntu/system-user-circuits
source .sync.env
uv run python phase1_linear_probing/steering_server.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --run-id curated4-8b-v002 \
  --layers 2 4 6 8 10 12 14 16 18 20 \
  --batch-size 96
```

The server auto-loads `iid_mm_directions.npz` and `iid_mm_proj_std.json` from `{run_dir}/` on startup. Confirm via `GET /directions` (lists `iid_mm_*` names) and `GET /direction_scales` (one positive scalar per rawspace iid_mm key).

### Computing IID-MM directions (if not already present)

```bash
uv run python phase1_linear_probing/compute_iid_mm.py \
  --run-id curated4-8b-v002 \
  --layers 2 4 6 8 10 12 14 16 18 20 \
  --pos last_prompt
```

### Refreshing projection stats with new directions

```bash
uv run python phase1_linear_probing/compute_projection_stats.py \
  --run-id curated4-8b-v002
```

### Optimizing batch size

The server runs with `--batch-size 96` on an A100. It processes prompts in chunks of 96, so **send prompts in multiples of 96** to maximize GPU utilization (192, 288, etc. are ideal). A 192-sample experiment (~2 batches) takes ~15-20 seconds.

## Coherence Protocol (MANDATORY)

Every response MUST be coherence-scored before trusting its verifier label. Use the `coherence` module:

```python
import sys
sys.path.insert(0, "phase1_linear_probing")
from coherence import score_coherence, compute_genuine_scr, ResponseQuality

# Score each response
scores = [score_coherence(r["text"]) for r in results]
labels = [r["label"] for r in results]

# Compute genuine SCR (excludes degenerate text)
metrics = compute_genuine_scr(labels, scores)
print(f"Raw SCR: {metrics['raw_scr']:.3f}  Genuine SCR: {metrics['genuine_scr']:.3f}")
print(f"Quality: {metrics['quality_breakdown']}")
```

**Rules:**
1. Report both `raw_scr` and `genuine_scr` in every results table.
2. If `genuine_scr` differs from `raw_scr` by more than 5 percentage points, the difference is verifier artifacts on degenerate text — investigate.
3. A config with high `raw_scr` but low `genuine_scr` is NOT a successful steering result.
4. Read 2-3 actual responses for every config that shows `genuine_scr > baseline + 0.05`.

## Exploration Utilities

All common operations are in `phase1_linear_probing/explore_utils.py`. Import and use:

```python
import sys
sys.path.insert(0, "phase1_linear_probing")
from explore_utils import (
    get_sample_ids, generate, summarize, save_experiment, add_notes,
    get_projection_stats, get_direction_scales, iid_mm_steering_clues,
    CONFLICT_IDS, FINDINGS_DIR,
)
```

### Key functions

| Function | Purpose |
|----------|---------|
| `get_sample_ids(conflict_ids, baseline_label, seed, limit)` | Query `/samples` for experiment_hashes. seed=42 for comparability. |
| `generate(sample_ids, direction, layer, alpha, mode, ...)` | Steered generation via `/generate`. Auto-scores. Returns self-contained responses. |
| `get_projection_stats()` | Fetch per-direction activation distributions. |
| `get_direction_scales()` | Fetch per-direction `proj_std` scalars (one per `iid_mm_*_L{layer}` key). |
| `iid_mm_steering_clues(stats, scales, constraint, direction_name, layer)` | Orientation in projection-std units: `{proj_std, alpha_per_std, y0_mean, y0_std, y1_mean, y1_std, separation, target_*}`. |
| `summarize(responses, label)` | Coherence-score, print per-constraint breakdown, return metrics. |
| `save_experiment(name, config, responses, out_dir, notes=...)` | Save JSON with coherence annotations and your observations. |

### Workflow: run+save → read output → add notes

**Step 1**: Run the experiment, summarize, and save in ONE script. `save_experiment` prints the summary AND example genuine followed_system responses so you can read them.

```python
ids = get_sample_ids(seed=42, limit=96)
stats = get_projection_stats()
scales = get_direction_scales()
clues = iid_mm_steering_clues(stats, scales, "_overall", "iid_mm", 12)

alpha = 1.0 * clues["alpha_per_std"]   # N=1 in projection-std units
result = generate(ids, direction="iid_mm_L12", layer=12,
                  mode="additive", alpha=alpha)
config = {"direction": "iid_mm_L12", "layer": 12, "mode": "additive",
          "alpha": alpha, "N": 1.0}
save_experiment("iid_mm_L12_add_N1", config, result["responses"], FINDINGS_DIR)
```

**Step 2**: Read the printed output. Look at the genuine followed_system responses. Think about what you see. Then add notes:

```python
add_notes(f"{FINDINGS_DIR}/iid_mm_L12_add_N1.json",
          "YOUR ACTUAL OBSERVATIONS: what did the responses look like? "
          "which constraints flipped genuinely? what surprised you?")
```

**IMPORTANT**: Do NOT write all experiments in a single giant script. Run one experiment, read the output, think about what it means, add notes, then decide what to run next. This is research, not batch processing.

**Reading responses is mandatory, not optional.** After every experiment:
- Print ALL responses labeled `followed_system` with `quality=genuine` — these are your claimed behavioral flips. Read them. Does the text actually comply with the system instruction?
- Print a few `followed_user` responses too — are they coherent or showing signs of degradation?
- Your notes should describe what the genuine followed_system responses actually look like, not just the numbers.
- If you can't explain in words what a steered response did differently from baseline, you don't understand the result yet.

### Server endpoints (for reference)

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Server status, loaded layers, direction count, GPU memory |
| `GET /directions` | Available direction names and types |
| `GET /direction_scales` | Per-direction `proj_std` scalars (iid_mm rawspace keys) |
| `GET /projection_stats` | Pre-computed activation distributions per constraint/layer/direction |
| `GET /samples?conflict_id=...&direction=...&baseline_label=...&seed=42&limit=96` | Query sample_ids |
| `POST /generate` | Steered generation (use `sample_ids` — auto-scores, self-contained responses) |

**Responses are self-contained**: each response includes the steering config, sample_id, baseline_label, conflict_id, direction, and the original prompts. The server logs every request/response to `server_log.jsonl`.

### Sample-ID based requests (PREFERRED)

Reference pre-loaded dataset samples by `experiment_hash`. This guarantees prompt fidelity and makes requests tiny.

```python
ids = get_sample_ids(
    conflict_ids=["json_only_vs_plain"],
    baseline_label="followed_user",
    seed=42, limit=96,
)
result = generate(ids, direction="iid_mm_L12", layer=12,
                  mode="additive", alpha=alpha)
```

### Comparability: use the same seed across experiments

**This is critical.** When comparing two steering configs, they MUST operate on the **exact same samples**. Use `seed=42` for all sweep experiments. This way, differences in genuine_scr reflect the steering config, not the sample set.

- **Sweep phase**: `seed=42, limit=96` per (conflict_id, direction, baseline_label) cell
- **Deep dives**: `seed=42` with no limit (all samples) — still comparable since the n=96 subset is a prefix of the full set with the same seed
- **Independent replication**: use a different seed (e.g., `seed=123`) to check whether results hold on fresh samples

## Data Loading

**Preferred: use the `/samples` endpoint** to get sample_ids by constraint, direction, and baseline label. No local data loading needed.

For offline analysis or custom filtering, you can also load locally:
```python
import sys
sys.path.insert(0, "phase1_linear_probing")
from pathlib import Path
from data import load_results, prepare_condition_c, load_sync_env
from compute_cmds import load_run_config

load_sync_env(Path("."))
cfg = load_run_config(Path("phase1_linear_probing/data/runs/{run_id}"))
df_all = load_results(Path("phase0_v2/data/results"), "meta-llama/Llama-3.1-8B-Instruct")
df_c = prepare_condition_c(df_all, "binary", conflict_ids=cfg.conflict_ids)
df_c = df_c.sort_values("conflict_id").reset_index(drop=True)
```

### Sample structure (IMPORTANT — understand before running experiments)

Steering experiments use Condition C samples. You should test **both** followed_user and followed_system samples to understand the full picture.

**Do NOT filter by Phase 0 labels.** The Phase 0 labels reflect vLLM generation, not HF generate() — the same prompt can produce different text and different labels across backends. Use ALL Condition C samples and measure the server's own unsteered baseline as ground truth.

The `baseline_label` field in responses tells you the Phase 0 label for reference, but don't use it for filtering. Instead:
1. Run unsteered baseline on your sample set.
2. Record the server's own labels — this is the true baseline SCR per (constraint, direction) cell.
3. Compare steered results against these server baselines, not against Phase 0 labels.

**Each sample in the batch has metadata** via `score_meta`:
- `conflict_id`: which of the 4 constraints (e.g., `json_only_vs_plain`)
- `direction`: `a_to_b` or `b_to_a` (which instruction is system vs user)
  - `a_to_b`: system=variant_a, user=variant_b (e.g., system=json, user=plain)
  - `b_to_a`: system=variant_b, user=variant_a (e.g., system=plain, user=json)

**Always break down results by (conflict_id, direction)**. Aggregate SCR hides the b_to_a asymmetry.

### Building sample sets

Use `get_sample_ids()` from `explore_utils`. Key patterns:

```python
# Sweep: 96 per cell = 768 total (seed=42 for comparability, no label filtering)
sweep_ids = get_sample_ids(seed=42, limit=96)

# Deep dive: all samples
all_ids = get_sample_ids(seed=42, limit=None)

# Single constraint
json_b2a = get_sample_ids(conflict_ids=["json_only_vs_plain"], seed=42, limit=96)
```

## Experimental Protocol

### Phase 0 — Numerical geometry sanity (DO THIS FIRST)

Use the pre-computed numerical artifacts to assess geometry — do **not** open or regenerate the geometry PDFs. The PDFs at `{run_dir}/iid_mm_plots/` are for human visual inspection by the supervising researcher; everything you need to make per-layer decisions is in the JSON/CSV artifacts.

**Sources of truth**:
- `iid_mm_metrics.csv` → balanced accuracy per (layer, scope, conflict_id, fit_basis). Indicates how well each direction separates the two classes on its training data.
- `projection_stats.json` → for each (group, layer, direction): `y0` and `y1` mean+std of the projection, Cohen's d, AUC, baseline 5–95th percentile range. Use group `_overall` for the global view; per-conflict groups for breakdowns.
- `iid_mm_proj_std.json` (also via the `/direction_scales` endpoint) → per-direction `proj_std`, the additive alpha unit.
- `direction_cosine_sim.json` → per-layer cosine similarity matrix between all directions; reveals whether `iid_mm_mean` is meaningfully different from `iid_mm` (if cos ≈ 1, the mean direction adds nothing).
- `iid_mm_steering_clues(stats, scales, constraint, direction_name, layer)` from `explore_utils.py` → bundles `proj_std`, `y0/y1 mean+std`, `separation`, `alpha_per_std`, and projection targets into one call. Prefer this over manually indexing JSONs.

**Per-layer assessment** (compute these for the candidate layers L = 4, 8, 12, 16, 20, 24, 28; expand if results suggest):
1. **Class separation** — pull `cohens_d` and `auc` for `iid_mm` at the layer in `_overall`. Cohen's d ≥ 1.0 and AUC ≥ 0.95 indicates strong separation. Below those, deprioritize.
2. **Additive landing zone** — given `proj_std`, `y0_mean`, `y1_mean`, `y1_std`: compute `y0_mean + N · proj_std` for N ∈ {1, 2, 3, 4} and check where it falls relative to `[y1_mean − y1_std, y1_mean + y1_std]`. The best layers are those where N = 1–2 lands inside that band; layers requiring N ≥ 4 to reach the followed_system mode will likely degrade coherence before flipping behavior.
3. **Per-conflict consistency** — for each `cid`, pull the same separation stats for `iid_mm_{cid}` in its own group and `iid_mm` in `_overall` restricted to that group. If a constraint's separation under the global direction is much weaker than under its per-conflict direction, plan to test that constraint with the per-conflict direction in Phase 2.
4. **`iid_mm_mean` vs `iid_mm`** — pull cosine similarity of the two directions at this layer from `direction_cosine_sim.json`. If cos > 0.99, treat them as identical for steering and skip the `mean` sweep at this layer.

Write the per-layer table (cohens_d, auc, proj_std, additive landing analysis, mean/overall cos, per-conflict notes) into `{run_dir}/exploration_v3/phase0_geometry.json` and `phase0_geometry_notes.md`. This is your reference for Phase 2 layer prioritization.

**Regenerate the underlying artifacts only if missing or stale**:
```bash
uv run python phase1_linear_probing/compute_iid_mm.py --run-id curated4-8b-v002
uv run python phase1_linear_probing/compute_projection_stats.py --run-id curated4-8b-v002
```

### Phase 1 — Unsteered baseline

Generate unsteered responses on the full sweep sample set. Record the server's own (constraint, direction) baseline genuine_scr — this is the reference point for everything that follows. Expect ~2-5%.

```python
ids = get_sample_ids(seed=42, limit=96)
result = generate(ids)  # alpha=0, no direction
save_experiment("baseline_unsteered", {"alpha": 0.0}, result["responses"], FINDINGS_DIR)
```

### Phase 2 — IID-MM Layer × Direction Sweep (~45 min)

Systematically sweep iid_mm directions across layers and modes.

**Directions to test**:
1. `iid_mm_L{layer}` — global (overall)
2. `iid_mm_mean_L{layer}` — mean of per-conflict, renormalized
3. `iid_mm_{cid}_L{layer}` — one per conflict (4 of these)

**Layers to test**: L2, L4, L6, L8, L10, L12, L14, L16, L18, L20.

**Modes**:
- **Additive (primary)** — sweep N ∈ {1, 2, 3, 4}, with `alpha = N × clues['alpha_per_std']`.
- **Projection (secondary)** — only at layers where additive shows promise. Sweep `N_extra ∈ {-0.5, 0, 0.5, 1.0}`, with `projection_target = clues['y1_mean'] + N_extra × clues['y1_std']`.

For every direction × layer × mode cell:

```python
stats = get_projection_stats()
scales = get_direction_scales()
clues = iid_mm_steering_clues(stats, scales, "_overall", "iid_mm", 12)

for N in [0.5, 1.0, 1.5, 2.0]:
    alpha = N * clues["alpha_per_std"]
    result = generate(ids, direction="iid_mm_L12", layer=12,
                      mode="additive", alpha=alpha)
    name = f"iid_mm_L12_add_N{N}"
    config = {"direction": "iid_mm_L12", "layer": 12, "mode": "additive",
              "alpha": alpha, "N": N}
    save_experiment(name, config, result["responses"], FINDINGS_DIR)
```

**Per experiment**: 768 samples (96 per constraint×direction cell, seed=42), coherence-score all, compute genuine_scr.

**Adaptive protocol — don't grind through a fixed grid:**
- Start with N=1 (the natural unit in projection-std space). If it produces a behavioral effect with good coherence, expand to N ∈ {2, 3, 4}.
- If genuine_scr > baseline + 0.05 but most flips are degenerate (raw_scr >> genuine_scr), drop to N=0.5 — you're past the coherence ceiling.
- If genuine_scr ≈ baseline AND coherence is fine at N=2, push to N=3 cautiously.
- If genuine_scr ≈ baseline AND N=2 already degrades coherence, **move on** — this (direction, layer) pair has no causal leverage.
- If genuine_scr > baseline + 0.10 with good quality, **flag for Phase 3 deep dive** and move on.
- Don't spend more than 3-4 N values per (direction, layer) pair in the sweep. The goal is to map the landscape, not optimize each cell.

This phase produces **two layer × direction heatmaps** (additive and projection) of genuine_scr (with raw_scr alongside as a comparison heatmap). Identify:
- Which layers have causal impact for each direction type and mode.
- Whether per-conflict directions outperform the overall direction for their own constraint.
- Whether `iid_mm_mean` (mean of per-conflict, renormalized) outperforms the global `iid_mm` direction.
- Whether projection mode beats additive at any layer.

### Phase 3 — Deep Dives on Promising Configs (~30 min)

For each config flagged in Phase 2:

1. **N (or N_extra) refinement**: Test 3-4 values around the flagged one.
2. **Full sample set**: `get_sample_ids(limit=None)` — all available samples.
3. **Followed_system retention**: Run the same config on the followed_system pool. Report what fraction stay `followed_system` (retention rate). If retention drops significantly, the steering is damaging existing compliance.
4. **Negative steering**: Apply the same direction with negative alpha on followed_system samples. Does it flip them to `followed_user`? If yes, the direction is bidirectional.
5. **Read responses**: For every `followed_system` response with quality=genuine, read the text and confirm it genuinely complies.
6. **Per-constraint × per-direction breakdown**: Report genuine_scr (with raw_scr alongside) separately for each (constraint, direction_type, a_to_b/b_to_a) cell.

### Phase 4 — Cross-Conflict Analysis (~20 min)

Test whether per-conflict iid_mm directions generalize across constraints.

#### 4a. Cross-steering matrix
For each per-conflict direction that worked in Phase 2-3, apply it to all 4 constraints' samples:
- Does `iid_mm_json_only_vs_plain_L12` steer list_bullets or tense?
- Does `iid_mm_list_bullets_vs_numbered_L12` steer json or tense?

#### 4b. Direction geometry
Compute cosine similarity between all per-conflict iid_mm directions at the best layer(s). High similarity → shared hierarchy signal. Low similarity → constraint-specific representations. Compare to the cosine similarity between each per-conflict direction and `iid_mm_mean`.

```python
import numpy as np, requests
dirs = {}
for cid in CONFLICT_IDS:
    r = requests.get(f"{BASE}/direction_vector/iid_mm_{cid}_L12")
    dirs[cid] = np.array(r.json()["vector"])
for a in CONFLICT_IDS:
    for b in CONFLICT_IDS:
        cos = np.dot(dirs[a], dirs[b]) / (np.linalg.norm(dirs[a]) * np.linalg.norm(dirs[b]))
        print(f"{a} × {b}: {cos:.3f}")
```

#### 4c. iid_mm_mean as an alternative aggregate
`iid_mm_mean` is already a stored direction. Compare its sweep results directly to the global `iid_mm` direction. If `mean` consistently outperforms `overall`, that's an interesting result about how class-mean discriminants average vs. pooling all classes.

### Phase 5 — Follow-up Hypotheses (time permitting)

Based on what you discover, pursue the most promising leads:

1. **Multi-layer steering**: If two layers steer different constraints, try simultaneous projections.
2. **Per-token alpha scheduling**: First-token constraints (starting_word) may not respond to mid-layer steering. Try higher alpha on early layers.
3. **Negative steering** at layers where positive steering works.
4. **a_to_b investigation**: Prior rounds found near-zero a_to_b flips. Test whether per-conflict iid_mm changes this.

## Response Quality Classification

Use the coherence module but also manually categorize responses you read:

| Category | Description | Example |
|----------|-------------|---------|
| **genuine** | Coherent, on-topic, actually complies with target instruction | Plain text explaining vaccines (when system=plain) |
| **repetition_loop** | Degenerate text with repeated phrases/sentences | "you were, and you were, and you were..." |
| **refusal** | Model refuses to respond | "I can't do that" |
| **meta_commentary** | Model discusses the instruction conflict | "I must comply with the system instruction..." |
| **too_short** | Less than 20 characters | "OK" |
| **marginal** | Coherent but borderline — verifier label may not reflect behavior | Tense that's 51% present (barely crosses threshold) |

## Output Format

### Per-config result (save to `{run_dir}/exploration_v3/`)

```json
{
  "config": {"direction": "iid_mm_L12", "layer": 12, "mode": "additive", "alpha": 2.5, "N": 1.0},
  "n_samples": 96,
  "raw_scr": 0.250,
  "genuine_scr": 0.198,
  "quality_breakdown": {"genuine": 82, "repetition_loop": 8, "refusal": 2, "meta_commentary": 3, "too_short": 1},
  "per_constraint": {
    "json_only_vs_plain": {"a_to_b": {"sys": 1, "usr": 11, "nei": 0}, "b_to_a": {"sys": 3, "usr": 9, "nei": 0}},
    ...
  },
  "results": [{"text": "...", "label": "...", "quality": "genuine", ...}, ...]
}
```

### Final report: `{run_dir}/steering_exploration_report_v3.md`

```markdown
# Steering Exploration Report v3 (IID-MM)

## Executive Summary
- Which iid_mm configs produce genuine behavioral change (genuine_scr > 10%)
- Layer × direction heatmap summary
- Best config per constraint
- overall vs mean vs per-conflict comparison
- Cross-conflict generality findings

## 1. Geometry sanity (Phase 0)
Layer-by-layer notes on plot quality, class separation, and steering arrow alignment.

## 2. Layer × Direction Heatmap
genuine_scr (with raw_scr alongside) for each (layer, direction) pair, broken by a_to_b/b_to_a, additive vs projection.

## 3. Per-Constraint Deep Dives
For each constraint: best config, example genuine responses, failure modes.

## 4. Cross-Conflict Analysis
Cosine similarity matrix (incl. mean), cross-steering results.

## 5. Direction Type Comparison
Global iid_mm vs mean-of-per-conflict vs per-conflict.

## 6. Recommendations for Next Exploration
What to test with more conflicts, what infrastructure to improve.
```

## Sub-Agent Strategy

With 4 constraints, parallelize analysis work across sub-agents. The GPU has a lock — only one generation request at a time — so sub-agents are useful for:

1. **Reading and classifying responses** (no GPU needed)
2. **Computing direction geometry** (cosine similarity, PCA — no GPU)
3. **Writing per-constraint deep dive sections** of the report

**Always use `model: "opus"` for sub-agents.**

## Pitfalls

- **SCR on degenerate text is meaningless** — the #1 lesson from prior exploration. ALWAYS coherence-check.
- **Sign convention**: positive = system direction. If SCR drops when you add positive alpha, check the direction.
- **N>2 on iid_mm can still cause gibberish** — projection-std units make alpha layer-comparable but not unbounded.
- **Both a_to_b and b_to_a matter**: always report separately. The asymmetry is informative.
- **Verifier labels on degenerate text are unreliable**: "I can't do that" may be scored as system-compliant for some constraints.
- **Data alignment**: activations are sorted by `conflict_id`. Always `df.sort_values("conflict_id")`.
- **Unsteered baseline differs from Phase 0**: the server uses HF `generate()` while Phase 0 used vLLM.
- **GPU lock**: only one generation request runs at a time. Don't send concurrent requests.
- **`iid_mm_zscored_*` directions are analysis-only** — do not use them for steering. Their fit basis is per-feature standardized; they are kept to compare Fisher invariance, not to perturb activations.

## Saving Results

Write results incrementally to `{run_dir}/exploration_v3/` after each experiment using `save_experiment()` from `explore_utils`. The server also logs everything to `server_log.jsonl` as a backup.
