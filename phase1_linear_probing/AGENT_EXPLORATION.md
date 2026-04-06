# Steering Exploration Agent

You are an autonomous research agent exploring how activation steering controls instruction hierarchy behavior in Llama-3.1-8B-Instruct. A steering server is running at `http://localhost:8000` with the model loaded and precomputed direction vectors.

## Mission

Discover whether activation steering can **genuinely flip** the model's instruction-following behavior — making it follow the system instruction instead of the user instruction (or vice versa) — while maintaining coherent, on-topic output.

The emphasis is on **genuine behavioral change**, not surface-level metric shifts. A response that says "I can't do that" or degenerates into repetitive gibberish is not a successful steering outcome, even if a verifier labels it as `followed_system`.

### Success criteria

A steering intervention is successful when:
1. The model produces **coherent, on-topic text** (not refusals, gibberish, or repetition loops)
2. The text **genuinely complies** with the target instruction (not meta-commentary about it)
3. The effect is **consistent** across both counterbalancing directions (a_to_b and b_to_a)
4. The verifier label **correctly reflects** the behavioral change (not a verifier artifact)

### What to report

- Which configurations produce genuine behavioral change, and for which constraints
- The qualitative character of steered responses (with examples)
- Which failure modes appear at different steering strengths
- Specific recommendations for follow-up experiments
- Proposals for new server features, direction types, or methodology improvements

## Background

### Instruction hierarchy conflicts

When a system prompt and user message give conflicting instructions (e.g., system says "respond in JSON only", user says "respond in plain text"), the model must choose which to follow. We call this **Condition C**.

- **Labels**: `followed_system` (y=1) vs `followed_user` (y=0), determined by automated verifiers
- **Counterbalancing**: each conflict is tested in both directions (a_to_b and b_to_a), so the probe can't just learn one side of the conflict
- **System Compliance Rate (SCR)**: fraction of responses labeled `followed_system`
- **SCR is often highly asymmetric** between a_to_b and b_to_a directions. Always test and report both.

### Understanding the verifier system (CRITICAL)

Before interpreting any steering results, you **must** understand how verifiers assign labels. Read the conflict definition code for each constraint you're working with (`phase0_v2/conflicts/definitions/{conflict_id}.py`).

**Key verifier properties to understand:**

1. **Boolean vs float verifiers.** Boolean verifiers (emoji presence, language detection) produce unambiguous labels. Float verifiers (formality score, keyword density) apply thresholds — edge cases and refusals can produce unexpected labels.

2. **How refusals are classified.** "I can't do that" contains a contraction → scores as casual (followed_user for formal_vs_casual_tone). Empty/short responses trivially satisfy avoidance constraints. A mass-refusal steering output can appear to have high SCR for constraints where refusal happens to match the system instruction's surface pattern.

3. **The four-label quadrant:**
   - `followed_system`: sys_verify=True, usr_verify=False
   - `followed_user`: sys_verify=False, usr_verify=True
   - `followed_both`: both True (confidence=0.5, inherently ambiguous)
   - `followed_neither`: both False (often indicates degenerate output)

4. **Meta-commentary interactions.** Some verifiers strip meta-commentary before scoring (via `extract_content()`); others don't. A response saying "I won't use emojis" while containing emojis is correctly classified by the emoji verifier (it checks for Unicode chars, not the word "emoji").

**Reference:** Read the calibration audit methodology at `phase0_v2/calibration/audit_agent_instructions.md` for a thorough treatment of verifier quality assessment. Condition C audit reports (when available) at `phase0_v2/calibration/output/condition_c_audit/` show what "good verification" looks like — behavioral taxonomies, failure mode quantification, and rubric justification.

### Direction families

Linear probes and class-mean differences (CMDs) trained on residual stream activations find directions that separate system-following from user-following behavior:

| Direction | What it is | Properties |
|-----------|-----------|------------|
| `probe` | Logistic regression weight (unit-norm) | Clean global split, higher Cohen's d |
| `probe_raw` | Same, unnormalized | Preserves magnitude |
| `cmd_overall` | mean(X\|y=1) - mean(X\|y=0), unit-norm | Higher raw separation, interleaved distributions |
| `cmd_overall_raw` | Same, unnormalized | Preserves per-constraint magnitude |
| `cmd_{constraint}` | Per-constraint CMD, unit-norm | Tuned to specific constraint |
| `cmd_raw_{constraint}` | Same, unnormalized | Preserves per-constraint magnitude |

**Sign convention**: positive projection = toward `followed_system`. Adding `+alpha * direction` pushes toward system compliance.

### Prior findings from curated35 exploration

The curated35-8b-v001 exploration revealed important patterns that should guide your work:

1. **Layer 25 is causally inert.** Despite having the highest probe accuracy (AUC=0.99, Cohen's d=3.14), steering at L25 — additive or projection, any alpha — produces zero behavioral change. L25 is a "read-only" layer for this signal.

2. **Layer 15 showed the only causal impact** with additive mode (probe, alpha=10): SCR shifted from 8.6% to 20.0%. But only ~2/35 constraints showed genuine compliance; the rest were meta-commentary, verifier artifacts, or unchanged.

3. **CMD directions cause mass refusals.** `cmd_overall` at L15/alpha=10 produces "I can't do that" on ~25/35 constraints. The SCR metric is misleading because verifiers score these refusals as system-compliant for some constraints.

4. **Projection mode had zero effect at all layers and targets.** Pinning the 1D projection to specific values doesn't change behavior. The causal mechanism likely requires perturbing dimensions beyond the probe direction.

5. **The coherence window is narrow.** At L15: alpha=5 is subtle, alpha=10 shows effects, alpha=15 produces gibberish. The useful range is approximately alpha 8-12.

6. **Layers 12, 14, 16 remain untested** and are high priority for the next exploration.

### Layers

The model has 32 layers (0-31). Directions exist for all layers with loaded directions. Based on prior findings:
- **Priority layers: 12, 14, 16** — Neighboring L15 which showed causal impact. These may reveal the causal window more precisely.
- **L25** — Confirmed non-causal for additive/projection in curated35, but include as a control to verify this holds for the curated4 constraints.
- Do NOT test L15 or L20 — L15 is already characterized; L20 showed no causal impact.
- Early layers (0-10) typically have weak separation and are unlikely to steer.

## Server API

Base URL: `http://localhost:8000`

### GET /health
Server status and configuration.
```json
{"status": "ok", "model": "meta-llama/Llama-3.1-8B-Instruct", "device": "cuda",
 "gpu_memory_gb": 15.2, "n_directions": 2304, "layers": [0,1,...,31],
 "d_model": 4096, "batch_size": 4}
```

### GET /directions
Summary of available directions — types, patterns, constraint list.

### GET /direction_vector/{name}
Fetch the actual vector for a named direction.

### GET /projection_stats
Pre-computed per-(constraint, layer, direction) projection statistics from training data.

**Filter by constraint and/or layer**:
```bash
curl 'http://localhost:8000/projection_stats?constraint_type=json_only_vs_plain&layer=14'
```

Returns nested dict: `{constraint: {L{layer}: {direction: {y0: stats, y1: stats, cohens_d, auc, baseline_range}}}}`.

**Use this to**: identify separation quality per constraint/layer, set reasonable targets, understand baseline ranges.

### POST /project
**Readout-only**: measure where prompts sit in direction space without any steering. Fast (~0.4s).

```json
POST /project
{
  "prompts": [{"system_prompt": "Be formal.", "user_prompt": "Tell me a joke"}],
  "directions": ["probe_L14", "cmd_overall_L14"],
  "layer": 14
}
```

### POST /generate
Steered text generation. Three modes:

**1. Additive steering** — `h' = h + alpha * direction`
```json
{
  "prompts": [{"system_prompt": "...", "user_prompt": "..."}],
  "direction": "probe_L14",
  "layer": 14,
  "mode": "additive",
  "alpha": 8.0,
  "max_new_tokens": 512
}
```

**2. Projection mode** — pin h·direction to a target value
```json
{
  "prompts": [...],
  "direction": "probe_L14",
  "layer": 14,
  "mode": "projection",
  "projection_target": 0.2,
  "max_new_tokens": 512
}
```

**3. Multi-projection** — pin multiple directions simultaneously
```json
{
  "prompts": [...],
  "projections": [
    {"direction": "cmd_json_only_vs_plain_L14", "layer": 14, "target": 5.0},
    {"direction": "probe_L16", "layer": 16, "target": 0.2}
  ],
  "max_new_tokens": 512
}
```

**Scoring**: Add `"score": true` and `"score_meta"` (one per prompt) to get behavioral labels:
```json
{
  "prompts": [...],
  "direction": "probe_L14", "layer": 14, "alpha": 8.0,
  "score": true,
  "score_meta": [{"conflict_id": "json_only_vs_plain", "direction": "a_to_b", "instruction_args": {"sys_value": "json", "usr_value": "plain"}}]
}
```
Response items include: `{text, label, confidence, sys_ok, usr_ok}`.

**Baseline (no steering)**: omit `direction` and set `alpha: 0`.

## Available Data

### Loading the dataset

```python
import sys
sys.path.insert(0, "phase1_linear_probing")
from pathlib import Path
from data import load_results, prepare_condition_c, load_sync_env

load_sync_env(Path("."))
df_all = load_results(Path("phase0_v2/data/results"), "meta-llama/Llama-3.1-8B-Instruct")
df_c = prepare_condition_c(df_all, "binary")
df_c = df_c.sort_values("conflict_id").reset_index(drop=True)
```

To filter to the run's constraints:
```python
from compute_cmds import load_run_config
cfg = load_run_config(Path("phase1_linear_probing/data/runs/{run_id}"))
df_c = prepare_condition_c(df_all, "binary", conflict_ids=cfg.conflict_ids)
```

### Important: DO NOT load the full activations NPZ
Use the `/project` endpoint for real-time readout or `/projection_stats` for pre-computed distributions.

## Methodology

### Principles

1. **Read responses first, compute metrics second.** Never trust an SCR number without reading the actual text. A 50% SCR could be genuine behavioral change or verifier artifacts on garbage.

2. **Classify responses, not just labels.** For every config you test, categorize responses into:
   - **Genuine compliance** — coherent, on-topic text that actually follows the target instruction
   - **Meta-commentary** — model discusses the instruction conflict without resolving it ("I will follow the rule, but...")
   - **Refusal/shutdown** — "I can't do that" or very short non-responsive text
   - **Gibberish/loops** — degenerate repetitive or incoherent output
   - **Unchanged** — same behavior as baseline

3. **Test both directions.** Always include both a_to_b and b_to_a samples. SCR asymmetry is common and informative.

4. **Find the coherence ceiling first.** Before looking for behavioral effects, determine the maximum steering strength that preserves coherent text. Then search for effects below that threshold.

5. **Chase what works.** Don't grind through a predetermined grid. Run a few configs, read responses, form hypotheses, test targeted follow-ups. Spend your compute budget on the most promising configurations.

6. **Save incrementally.** Write results to `agent_findings/` after each configuration, not at the end of a batch.

### Phase 1: Orientation (~10 min)

1. Fetch `/health` to confirm server status, note loaded layers and directions.
2. Fetch `/projection_stats` for all constraints in this run.
3. For each constraint, note: Cohen's d at each loaded layer, y0/y1 distribution stats, class balance.
4. **Read the verifier code** for each constraint (`phase0_v2/conflicts/definitions/{conflict_id}.py`). Understand: what it measures, bool vs float, how refusals are classified, edge cases.
5. Build a sample set: **4 prompts per constraint** (2 a_to_b, 2 b_to_a). This is your core test set for the entire exploration.

### Phase 2: Coherence ceiling (~15 min)

Before searching for behavioral effects, find where coherence breaks.

1. Run baseline (unsteered) on all samples. Read 1-2 responses per constraint. Note the model's default behavior.
2. Pick the probe direction at your highest-priority layer. Test additive alphas: 5, 8, 10, 12, 15.
3. For each alpha, **read all responses** (not just SCR). Identify:
   - At what alpha does gibberish/repetition start appearing?
   - At what alpha do refusals start appearing?
   - Where is the coherence boundary?
4. The useful exploration range is between "no visible effect" and "coherence breaks." This is your **steering budget**.

### Phase 3: Directed exploration (~30-60 min)

Within your steering budget, explore systematically but adaptively:

1. **Layer comparison:** Test the best alpha (from Phase 2) at each available layer. Read responses. Which layers produce qualitatively different behavior?

2. **Direction comparison:** At the best layer, compare:
   - `probe` vs `cmd_overall` vs per-constraint CMDs
   - Look for: does CMD cause more refusals? Does per-constraint CMD produce more genuine compliance for its target constraint?

3. **Per-constraint deep dives (use sub-agents — one per constraint):** Each constraint gets its own focused exploration:
   - What does the model do at baseline? (Read actual responses, both directions)
   - What changes under steering with the general probe direction?
   - **Test the per-constraint CMD** (`cmd_{constraint}_L{layer}`). This direction is specifically optimized for this constraint and may produce genuine compliance where the general probe fails.
   - Test 2-3 alphas with the per-constraint CMD (within the coherence budget)
   - **Try multi-projection**: pin both probe AND per-constraint CMD simultaneously using the `/generate` multi-projection endpoint. This lets you steer the general hierarchy signal and the constraint-specific signal at the same time — potentially more effective than either alone.
   - Example multi-projection for `json_only_vs_plain`:
     ```json
     {
       "prompts": [...],
       "projections": [
         {"direction": "probe_L14", "layer": 14, "target": 0.2},
         {"direction": "cmd_json_only_vs_plain_L14", "layer": 14, "target": 3.0}
       ],
       "score": true, "score_meta": [...]
     }
     ```
   - Report: what worked, what didn't, example responses, recommended config

4. **Chase interesting observations.** If you notice that one constraint flips genuinely at a certain config, try:
   - Varying alpha in small increments (±1) around that point
   - Testing at neighboring layers
   - Multi-projection: combining probe + per-constraint CMD at same or different layers
   - Cross-layer multi-projection: e.g., probe at L12 + constraint CMD at L16

### Phase 4: Validation (if genuine effects found)

If you find configurations that produce genuine compliance:

1. Expand the sample set to 8-10 per constraint (both directions)
2. Rerun the best config on the expanded set
3. Compute SCR per constraint, per direction
4. Read all responses that the verifier labels as `followed_system` — confirm they are genuinely compliant
5. Check for collateral damage: did other constraints break?

### Phase 5: Advanced experiments (if time permits)

1. **Multi-projection combinations**: systematically explore probe + CMD combinations:
   - Same layer: probe_L14 + cmd_{constraint}_L14
   - Cross-layer: probe_L12 + cmd_{constraint}_L16
   - Vary targets for each direction independently
   - Try additive on one direction + projection on another (use additive for the general probe, multi-projection for constraint CMD)
2. **Custom directions**: compute novel vectors in Python (PCA of CMDs, blended directions, difference vectors) and pass as raw float lists
3. **Adaptive targeting**: use `/project` to measure each prompt's baseline projection, then set per-prompt steering strength proportional to how far it is from the target
4. **Negative steering**: push toward user compliance — this may be easier/harder than pushing toward system compliance

## Sub-Agent Strategy

With a small number of constraints (e.g., 4), the natural parallelization is **one sub-agent per constraint**. Each sub-agent owns the full exploration for its constraint:

1. **Per-constraint sub-agents (Phase 3):** Each sub-agent:
   - Loads samples for its constraint only (both a_to_b and b_to_a)
   - Reads the verifier code for that constraint
   - Runs baseline, reads responses
   - Tests general probe at each layer within the coherence budget
   - Tests per-constraint CMD (`cmd_{constraint}_L{layer}`) at each layer
   - Tests multi-projection (probe + constraint CMD simultaneously)
   - Writes detailed findings to `agent_findings/{constraint_type}/`
   - Reports: best config, example responses, failure modes, SCR per direction

2. **Qualitative review sub-agents:** After initial exploration, have a sub-agent read all steered responses for a batch of configs and categorize them (genuine/meta/refusal/gibberish/unchanged).

3. **Verifier sub-agent (Phase 1):** One sub-agent reads all verifier code and audit reports, produces a summary of how each constraint's verifier works and what edge cases to watch for.

**Important**: Always use `model: "opus"` when spawning sub-agents. The server has a GPU lock, so sub-agents cannot make concurrent API calls — coordinate via the orchestrator or accept that sub-agents will queue behind each other. Sub-agents are most useful for parallelizing analysis/reading work, not API calls.

Each sub-agent should write structured results (JSON or markdown) to `{run_dir}/agent_findings/` for the orchestrator to aggregate.

## Python Patterns

Use Python `requests` for API calls, not curl.

```python
import requests, json

BASE = "http://localhost:8000"

def generate_scored(prompts, score_meta, direction=None, layer=14, alpha=0.0,
                    mode="additive", projection_target=None, max_new_tokens=512):
    """Generate with scoring. Returns full response JSON."""
    body = {
        "prompts": prompts,
        "alpha": alpha,
        "max_new_tokens": max_new_tokens,
        "score": True,
        "score_meta": score_meta,
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

### Building prompts with score_meta from DataFrame rows

```python
def df_to_request(df_subset):
    """Convert DataFrame rows to prompts + score_meta for the server."""
    prompts, score_meta = [], []
    for _, row in df_subset.iterrows():
        prompts.append({
            "system_prompt": row["system_prompt"],
            "user_prompt": row["user_prompt"],
        })
        args = row["instruction_args"]
        if isinstance(args, str):
            args = json.loads(args)
        score_meta.append({
            "conflict_id": row["conflict_id"],
            "direction": row["direction"],
            "instruction_args": args,
        })
    return prompts, score_meta
```

### Classifying response quality

After each generation, classify responses:

```python
def classify_response_quality(text, label):
    """Categorize a steered response beyond the verifier label."""
    text_lower = text.lower().strip()
    if len(text_lower) < 20:
        return "refusal_short"
    if text_lower.startswith("i can't") or text_lower.startswith("i'm not able"):
        return "refusal"

    # Check for repetition loops
    words = text.split()
    if len(words) > 20:
        last_20 = words[-20:]
        unique_ratio = len(set(last_20)) / len(last_20)
        if unique_ratio < 0.3:
            return "gibberish_loop"

    # Check for meta-commentary about rules/instructions
    meta_phrases = ["i will follow", "the rule is", "it is a rule",
                    "i cannot follow", "my instructions", "system instruction"]
    if any(p in text_lower for p in meta_phrases):
        return "meta_commentary"

    return "genuine"  # coherent, on-topic — still verify manually
```

### Saving results incrementally

```python
import json
from pathlib import Path

def save_config_result(name, config, results, out_dir):
    """Save one config's results immediately after running."""
    path = Path(out_dir) / f"explore_{name}.json"
    scr = sum(1 for r in results if r["label"] == "followed_system") / len(results)
    data = {
        "config": {k: v for k, v in config.items()
                   if k != "direction" or isinstance(v, str)},
        "scr": scr,
        "n_samples": len(results),
        "results": results,
    }
    path.write_text(json.dumps(data, indent=2))
```

### Computing and testing custom directions

You can compute novel steering vectors in Python and pass them directly:

```python
import numpy as np, requests

probe = np.array(requests.get(f"{BASE}/direction_vector/probe_L14").json()["vector"])
cmd = np.array(requests.get(f"{BASE}/direction_vector/cmd_overall_L14").json()["vector"])
blended = 0.7 * probe + 0.3 * cmd
blended = (blended / np.linalg.norm(blended)).tolist()

r = requests.post(f"{BASE}/generate", json={
    "prompts": [...],
    "direction": blended,  # raw vector instead of name
    "layer": 14, "alpha": 8.0, "max_new_tokens": 512,
})
```

## Output: Complete Research Report

Your final deliverable is a single comprehensive report at:

**`{run_dir}/steering_exploration_report.md`**

Working data and sub-agent outputs go in `agent_findings/`.

### Report structure

```markdown
# Steering Exploration Report
## Executive Summary
- Best configurations found (with response quality assessment)
- Which constraints showed genuine behavioral change
- Which constraints showed only meta-commentary, refusals, or no change
- Coherence budget: alpha range where output stays coherent

## 1. Verifier Assessment
- For each constraint: what the verifier measures, how refusals are classified
- Known edge cases relevant to steering

## 2. Baseline Analysis
- Unsteered behavior per constraint (with response examples)
- Distribution stats from projection_stats

## 3. Coherence Ceiling
- Alpha at which gibberish/repetition/refusals begin (per layer)
- Useful steering range

## 4. Exploration Results
For each configuration tested:
- Direction, layer, mode, alpha/target
- Per-constraint response categorization: genuine / meta / refusal / gibberish / unchanged
- 1-2 example responses with commentary
- Per-constraint SCR in both directions

## 5. Per-Constraint Deep Dives
For each constraint:
- Baseline behavior (both directions)
- Best steering result (with response examples)
- Per-constraint CMD vs general direction comparison
- Why this constraint does/doesn't respond to steering

## 6. Key Findings
- Causal layer analysis (which layers affect behavior)
- Direction type comparison (probe vs CMD vs per-constraint)
- Additive vs projection mode comparison
- Failure mode taxonomy with examples

## 7. Recommendations
- Recommended configurations for each constraint
- Suggested next experiments
- Server/direction proposals

## Appendix
- Full per-constraint tables
- Example responses (good and bad)
- Scripts used
```

## Pitfalls

- **Sign convention**: positive = system direction. If SCR drops when you add positive alpha, check you're using the right direction.
- **Extreme alphas cause gibberish**: start within the coherence budget from Phase 2.
- **SCR is misleading without qualitative review**: always read responses before trusting numbers.
- **Verifier labels on degenerate text are unreliable**: "I can't do that" may be labeled `followed_user` or `followed_system` depending on the constraint's verifier logic.
- **Data alignment**: activations are sorted by `conflict_id`. Always `df.sort_values("conflict_id")`.
- **Unsteered baseline differs from Phase 0**: the server uses HF `generate()` while Phase 0 used vLLM.
- **GPU lock**: only one request runs at a time. Don't send concurrent requests.
- **Both directions matter**: a_to_b and b_to_a SCR can be very different. Never draw conclusions from one direction only.
- **Refusals game float verifiers**: "I can't do that" has contractions → casual score → followed_user for formal_vs_casual_tone. This is the verifier working correctly on bad text, not a steering signal.
