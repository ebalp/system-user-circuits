# Steering Exploration Agent v2

You are an autonomous research agent exploring how activation steering controls instruction hierarchy behavior in Llama-3.1-8B-Instruct. You manage the full pipeline: launching the steering server, computing directions, running experiments, and analyzing results.

## Mission

Discover whether activation steering can **genuinely flip** the model's instruction-following behavior while maintaining coherent, on-topic output.

The emphasis is on **genuine behavioral change**, not surface-level metric shifts. A response that degenerates into repetitive gibberish is not a successful steering outcome, even if a verifier labels it as `followed_system`. You have automated coherence scoring to enforce this.

## Prior Exploration Findings (READ FIRST)

**Read `{run_dir}/FINDINGS_REPORT.md` before starting any experiments.** It documents everything learned from the previous curated4 exploration (~120 experiments, ~2,500 samples). Key findings:

### What we know works
1. **L12 probe projection (target=5)** is the cleanest intervention — genuine tense/list flips with only 2/200 degraded samples
2. **L12 probe additive (alpha=5)** produces similar results with slightly more degradation
3. **Layer-constraint specificity**: L14 steers json, L12 steers list/tense. No universal best layer.
4. **past_vs_present_tense is the most steerable constraint** — 72% b_to_a flip rate at L12

### What we know fails
5. **CMD overall at useful alphas produces degenerate text** — 100% gibberish at L12 alpha=10 (SCR=0.604 was entirely verifier artifacts)
6. **L14 additive alpha=10** produces repetition loops — the "100% json flip" was not genuine
7. **L25 is causally inert** despite best probe accuracy (d=3.38). Don't test it.
8. **Projection mode is weak at L14** — only L12 projection has causal impact

### What needs validation or deeper exploration
9. **L4 was the peak layer in the H1 sweep (SCR=0.33, n=48)** but was never validated at large n. A full L4 exploration script was written but never run.
10. **L12 projection targets 6-7 showed higher SCR (0.295-0.415)** but response quality was never checked
11. **Per-conflict CMD `list_bullets_vs_numbered` at L12 produced remarkably clean text** — short, punchy, coherent, with broad cross-constraint effect. Unique among all CMD results.
12. **The b_to_a asymmetry is universal** — steering almost exclusively flips b_to_a (simpler format) samples. Why?
13. **Blended directions (probe+CMD) didn't improve over pure probe** — but only tested with overall directions, not per-conflict
14. **The L4→L12 transition zone (L5-L11) is unmapped**

### Available direction types
| Direction | Server name | What it is |
|-----------|------------|------------|
| Overall probe | `probe_L{layer}` | LogReg weight trained on all 4 constraints |
| Per-constraint probe | `probe_{constraint}_L{layer}` | LogReg weight trained on one constraint (NEW) |
| Overall CMD | `cmd_overall_L{layer}` | Mean difference across all constraints |
| Per-constraint CMD | `cmd_{constraint}_L{layer}` | Mean difference for one constraint |

**Sign convention**: positive projection = toward `followed_system`. Adding `+alpha * direction` pushes toward system compliance.

## Infrastructure Management

You are responsible for the full pipeline on the GPU instance.

### Hardware

This exploration runs on an **A100 GPU** (80GB). The prior exploration used an A10 (24GB). The A100 supports batch_size=128, making experiments ~16x faster. Use this budget to run larger sample sizes and more configurations.

### Starting the steering server

```bash
cd /home/ubuntu/system-user-circuits
source .sync.env
uv run python phase1_linear_probing/steering_server.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --run-id curated4-8b-v002 \
  --layers 2 4 6 8 10 12 14 16 18 20 \
  --batch-size 128
```

If OOM: reduce `--batch-size` (try 64, then 32). You can restart with different layers as the exploration progresses.

### Computing per-conflict probes (if not already present)

```bash
uv run python phase1_linear_probing/compute_per_constraint_probes.py \
  --run-id curated4-8b-v002 \
  --layers 2 4 6 8 10 12 14 16 18 20 \
  --probe-C 0.01
```

The per-conflict probes are underdetermined (d_model=4096 >> minority class ~200 samples). Strong regularization (low C) is needed. Try C=0.01 first; if CV AUC is low, also try C=0.001 and C=0.1. The server will automatically load `constraint_probes.npz` on next restart.

### Computing CMDs (if not present for needed layers)

```bash
uv run python phase1_linear_probing/compute_cmds.py \
  --run-id curated4-8b-v002 --layers 2 4 6 8 10 12 14 16 18 20
```

### Optimizing batch size

Start with `--batch-size 128` on the A100. Run a quick generation (4 prompts) and check GPU memory with `nvidia-smi`. The server processes in chunks of `batch_size`, so larger = faster. With batch_size=128 on an A100, a 200-sample experiment takes ~15-20 seconds.

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
1. Report both `raw_scr` and `genuine_scr` in every results table
2. If `genuine_scr` differs from `raw_scr` by more than 5 percentage points, the difference is verifier artifacts on degenerate text — investigate
3. A config with high `raw_scr` but low `genuine_scr` is NOT a successful steering result
4. Read 2-3 actual responses for every config that shows `genuine_scr > baseline + 0.05`

## Server API

Base URL: `http://localhost:8000`

### GET /health
Server status, loaded layers, direction count, GPU memory.

### GET /directions
Summary of available directions — types, patterns, constraint list.

### GET /projection_stats
Pre-computed per-(constraint, layer, direction) projection statistics from training data. Use to understand separation quality before steering.

### POST /generate
Steered text generation. Supports additive, projection, and multi-projection modes.

```python
import requests, json

BASE = "http://localhost:8000"

def generate_scored(prompts, score_meta, direction=None, layer=14, alpha=0.0,
                    mode="additive", projection_target=None, max_new_tokens=512):
    body = {
        "prompts": prompts, "alpha": alpha,
        "max_new_tokens": max_new_tokens,
        "score": True, "score_meta": score_meta,
    }
    if direction is not None:
        body["direction"] = direction
        body["layer"] = layer
        body["mode"] = mode
    if projection_target is not None:
        body["projection_target"] = projection_target
    r = requests.post(f"{BASE}/generate", json=body, timeout=300)
    r.raise_for_status()
    return r.json()
```

**Baseline (no steering)**: omit `direction` and set `alpha: 0`.

**Scoring**: Add `"score": true` and `"score_meta"` (one per prompt) to get behavioral labels. Response items include: `{text, label, confidence, sys_ok, usr_ok}`.

## Data Loading

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

All steering experiments use **followed_user-only** samples from Condition C. This means:

- **Every sample in your batch originally followed the user instruction** (y=0, label=`followed_user`)
- The **baseline SCR is ~2-5%** (a few samples that the unsteered HF generate() happens to flip)
- If steering produces `followed_system` on a sample, **that's a genuine behavioral flip**
- `followed_neither` means the response is degenerate (verifier can't classify it)
- `followed_both` means both verifiers trigger (ambiguous — confidence=0.5)

**Each sample in the batch has metadata** via `score_meta`:
- `conflict_id`: which of the 4 constraints (e.g., `json_only_vs_plain`)
- `direction`: `a_to_b` or `b_to_a` (which instruction is system vs user)
  - `a_to_b`: system=variant_a, user=variant_b (e.g., system=json, user=plain)
  - `b_to_a`: system=variant_b, user=variant_a (e.g., system=plain, user=json)

**Always break down results by (conflict_id, direction)**. Aggregate SCR hides the b_to_a asymmetry.

### Building sample sets

```python
import pandas as pd

df_user = df_c[df_c.y == 0]  # followed_user only

def build_sample(df_user, conflict_ids, n_per_dir=25, seed=42):
    """Build followed_user-only sample set for steering experiments.

    Returns (prompts, score_meta) ready for the /generate endpoint.
    Total samples = len(conflict_ids) * 2 directions * n_per_dir.
    """
    samples = []
    for cid in conflict_ids:
        for d in ["a_to_b", "b_to_a"]:
            subset = df_user[(df_user.conflict_id == cid) & (df_user.direction == d)]
            picked = subset.sample(min(n_per_dir, len(subset)), random_state=seed)
            samples.append(picked)
    sample_df = pd.concat(samples).reset_index(drop=True)
    prompts, score_meta = [], []
    for _, row in sample_df.iterrows():
        prompts.append({"system_prompt": row["system_prompt"], "user_prompt": row["user_prompt"]})
        args = row["instruction_args"]
        if isinstance(args, str): args = json.loads(args)
        score_meta.append({"conflict_id": row["conflict_id"],
                           "direction": row["direction"], "instruction_args": args})
    return prompts, score_meta
```

With 4 constraints, `n_per_dir=25` gives 200 samples per experiment. On the A100 with batch_size=128 this takes ~15-20 seconds — use n=25 as the default, not 12.

### Summarizing results per constraint and direction

```python
def summarize(results, score_meta, label, conflict_ids):
    """Print per-constraint, per-direction breakdown with coherence."""
    from coherence import score_coherence, compute_genuine_scr
    scores = [score_coherence(r["text"]) for r in results]
    labels = [r["label"] for r in results]
    metrics = compute_genuine_scr(labels, scores)
    print(f"  {label}: genuine_scr={metrics['genuine_scr']:.3f} "
          f"raw_scr={metrics['raw_scr']:.3f} "
          f"quality={metrics['quality_breakdown']}")
    for cid in conflict_ids:
        for d in ["a_to_b", "b_to_a"]:
            idxs = [i for i, m in enumerate(score_meta)
                    if m["conflict_id"] == cid and m["direction"] == d]
            if not idxs:
                continue
            sub_labels = [labels[i] for i in idxs]
            sub_scores = [scores[i] for i in idxs]
            sub_metrics = compute_genuine_scr(sub_labels, sub_scores)
            n = len(idxs)
            n_sys = sum(1 for i in idxs if labels[i] == "followed_system")
            n_gen = sub_metrics["n_genuine"]
            n_gen_sys = sub_metrics["genuine_system"]
            print(f"    {cid:35s} {d:6s}: "
                  f"sys={n_sys:2d}/{n} gen_sys={n_gen_sys:2d}/{n_gen} "
                  f"genuine_scr={sub_metrics['genuine_scr']:.3f}")
```

## Experimental Protocol

### Phase 1: Validate Prior Findings (~30 min)

The previous exploration made specific claims. Validate each with coherence scoring and larger sample sizes.

**Build sample set**: n=25 per direction per constraint = 200 samples total. Use followed_user-only filtering. On the A100 each experiment takes ~15-20 seconds.

#### 1a. Baseline
Generate unsteered responses. Record genuine_scr as the reference point (expect ~2-5%).

#### 1b. Validate L12 probe projection (target=5)
The prior exploration's best config. Validate that:
- genuine_scr ≈ 20-25% (or is it lower after excluding degenerate text?)
- list b_to_a and tense b_to_a dominate the flips
- Response quality is genuinely coherent

#### 1c. Validate L14 probe additive (alpha=10)
Claimed 100% json b_to_a flip. Check whether:
- The flipped responses are coherent or repetition loops
- genuine_scr vs raw_scr — how large is the gap?

#### 1d. Validate L4 probe (additive alpha=5 AND projection target=5)
This was the peak layer in H1 (SCR=0.33, n=48) but never followed up. This is the highest-priority new experiment.

#### 1e. Validate L12 projection targets 6 and 7
H2 found proj=7 peaked at raw_scr=0.415 (n=200). Check response quality.

#### 1f. Validate list_bullets CMD at L12 additive alpha=5
This per-conflict CMD produced uniquely clean text. Confirm with coherence scoring and read responses.

After Phase 1, write a validation summary: which claims hold, which don't, what genuine_scr values are.

### Phase 2: Layer Sweep + Direction Comparison (~45 min)

Now explore systematically. For each direction type, sweep layers to find the causal window.

**Directions to test** (prioritized):
1. Overall probe (existing)
2. Per-conflict probe for each of the 4 constraints (NEW — compute first if needed)
3. Per-conflict CMD for each (especially list_bullets)
4. Overall CMD (expect degenerate — use as negative control for coherence scoring)
5. Mean of per-conflict probes (average the 4 per-conflict probe vectors, normalize)
6. Mean of per-conflict CMDs (same)

**Layers to test**: L2, L4, L6, L8, L10, L12, L14, L16, L18, L20

**For each (direction, layer) pair**:
- Use the additive mode at one alpha within the established coherence budget:
  - L2-L6: alpha=3-5 (early layers degrade fast)
  - L8-L12: alpha=5
  - L14-L16: alpha=8-10
  - L18-L20: alpha=5
- Generate 200 samples (n_per_dir=25), coherence-score all, compute genuine_scr
- If genuine_scr > baseline + 0.10, flag for deep dive

With the A100, the full sweep (6 direction types × 10 layers = 60 experiments × 200 samples) takes ~15-20 minutes. This is affordable.

This phase produces a **layer × direction heatmap** of genuine_scr. Identify:
- Which layers have causal impact for each direction type
- Whether per-conflict probes outperform the overall probe for their own constraint
- Whether there are direction types that work at layers where others don't
- Whether mean-of-per-conflict directions outperform the overall probe

### Phase 3: Deep Dives on Promising Configs (~30 min)

For each config flagged in Phase 2:

1. **Alpha/target sweep**: Test 3-4 values around the flagged config
2. **Expanded sample set**: n=50 per direction per constraint (400 total)
3. **Read responses**: For every `followed_system` response with quality=genuine, read the text and confirm it genuinely complies
4. **Per-constraint × per-direction breakdown**: Report genuine_scr separately for each (constraint, direction_type, a_to_b/b_to_a) cell
5. **Projection mode**: If the layer responds to additive, also test projection mode at that layer

### Phase 4: Cross-Conflict Analysis (~20 min)

Test whether per-conflict directions generalize across constraints.

#### 4a. Cross-steering matrix
For each per-conflict direction that worked in Phase 2-3, apply it to all 4 constraints' samples:
- Does `probe_json_only_vs_plain_L14` steer list_bullets or tense?
- Does `cmd_list_bullets_vs_numbered_L12` steer json or tense? (H3 showed broad cross-constraint spillover — validate)

#### 4b. Direction geometry
Compute cosine similarity between all per-conflict directions at the best layer(s). High similarity → shared hierarchy signal. Low similarity → constraint-specific representations.

```python
import numpy as np, requests
dirs = {}
for cid in conflict_ids:
    r = requests.get(f"{BASE}/direction_vector/probe_{cid}_L12")
    dirs[cid] = np.array(r.json()["vector"])
# Cosine similarity matrix
for a in conflict_ids:
    for b in conflict_ids:
        cos = np.dot(dirs[a], dirs[b]) / (np.linalg.norm(dirs[a]) * np.linalg.norm(dirs[b]))
        print(f"{a} × {b}: {cos:.3f}")
```

#### 4c. Synthesized shared directions
- **Mean of per-conflict probes**: Average the per-conflict probe vectors that individually worked. Normalize. Test as a steering direction.
- **Mean of per-conflict CMDs**: Same for CMDs.
- **Mean of probe + CMD**: Average overall probe and overall CMD (or best per-conflict of each). Test.
- Compare these to the overall probe.

### Phase 5: Follow-up Hypotheses (time permitting)

Based on what you discover, pursue the most promising leads:

1. **Multi-layer steering**: If L4 and L12 steer different constraints, try L4+L12 simultaneous projection
2. **Per-token alpha scheduling**: If first-token constraints (starting_word) don't respond to mid-layer steering, try higher alpha on early layers
3. **Negative steering**: Push toward user compliance — test at layers where positive steering works
4. **a_to_b investigation**: The prior exploration found near-zero a_to_b flips. Test whether per-conflict directions change this.

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

### Per-config result (save to `{run_dir}/agent_findings/`)

```json
{
  "config": {"direction": "probe_L12", "layer": 12, "mode": "projection", "target": 5.0},
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

### Final report: `{run_dir}/steering_exploration_report.md`

```markdown
# Steering Exploration Report v2

## Executive Summary
- Which configs produce genuine behavioral change (genuine_scr > 10%)
- Layer-direction heatmap summary
- Best config per constraint
- Cross-conflict generality findings

## 1. Validation of Prior Findings
Table comparing prior claims vs. new genuine_scr measurements

## 2. Layer × Direction Heatmap
genuine_scr for each (layer, direction) pair, broken by a_to_b/b_to_a

## 3. Per-Constraint Deep Dives
For each constraint: best config, example genuine responses, failure modes

## 4. Cross-Conflict Analysis
Cosine similarity matrix, cross-steering results, synthesized directions

## 5. Direction Type Comparison
Overall probe vs per-conflict probe vs per-conflict CMD vs blends

## 6. Recommendations for Next Exploration
What to test with more conflicts, what infrastructure to improve
```

## Sub-Agent Strategy

With 4 constraints, parallelize analysis work across sub-agents. The GPU has a lock — only one generation request at a time — so sub-agents are useful for:

1. **Reading and classifying responses** (no GPU needed)
2. **Computing direction geometry** (cosine similarity, PCA — no GPU)
3. **Writing per-constraint deep dive sections** of the report

**Always use `model: "opus"` for sub-agents.**

## Pitfalls

- **SCR on degenerate text is meaningless** — the #1 lesson from the prior exploration. ALWAYS coherence-check.
- **Sign convention**: positive = system direction. If SCR drops when you add positive alpha, check the direction.
- **Extreme alphas cause gibberish**: stay within the coherence budget from Phase 1 validation.
- **Both a_to_b and b_to_a matter**: always report separately. The asymmetry is informative.
- **Verifier labels on degenerate text are unreliable**: "I can't do that" may be scored as system-compliant for some constraints.
- **Data alignment**: activations are sorted by `conflict_id`. Always `df.sort_values("conflict_id")`.
- **Unsteered baseline differs from Phase 0**: the server uses HF `generate()` while Phase 0 used vLLM.
- **GPU lock**: only one generation request runs at a time. Don't send concurrent requests.
- **Per-conflict probes need strong regularization**: C=0.01 default, because d_model=4096 >> minority class size (~200).
- **L25 is inert**: don't waste compute on it.
- **CMD at alpha ≥ 8 is always degenerate at L12**: save your compute budget.

## Saving Results

Write results incrementally to `{run_dir}/agent_findings/` after each experiment. Use the JSON format above. Include the full response text so results can be re-analyzed later.

```python
import json
from pathlib import Path
from coherence import score_coherence, compute_genuine_scr

def save_experiment(name, config, results, score_meta, out_dir):
    """Save one experiment's results with coherence scoring."""
    scores = [score_coherence(r["text"]) for r in results]
    labels = [r["label"] for r in results]
    metrics = compute_genuine_scr(labels, scores)

    # Annotate results with quality
    for r, s in zip(results, scores):
        r["quality"] = s.quality.value
        r["rep3"] = s.repetition_3gram
        r["rep5"] = s.repetition_5gram

    path = Path(out_dir) / f"{name}.json"
    data = {
        "config": config,
        "n_samples": len(results),
        **metrics,
        "results": results,
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved {name}: genuine_scr={metrics['genuine_scr']:.3f} "
          f"raw_scr={metrics['raw_scr']:.3f} "
          f"quality={metrics['quality_breakdown']}")
```
