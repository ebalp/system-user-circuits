# Phase 0: Behavioral Analysis

Evaluates how LLMs handle conflicting instructions between system prompts and user messages across 4 experimental conditions.

## Experimental Design

| Condition | System prompt | User message | `strength` | `user_style` | Purpose |
|-----------|--------------|--------------|-----------|--------------|---------|
| **A** | `{instruction}. {negative}.` | Task only | `weak` | `task_only` | Baseline: does the model follow a system constraint? |
| **B** | *(empty)* | `Please {instruction}. {task}` | `null` | `with_instruction` | Baseline: does the model follow a user constraint? |
| **C** | strength template | style template | varies | varies | Conflict: system vs user — measures hierarchy compliance |
| **D** | *(empty)* | `{instruction}. {negative}. Please {instruction2}. {task}` | `weak` | `with_instruction` | Recency: same-level conflict — first vs second instruction |

Conditions A and B are capability baselines (no conflict). Condition C is the main hierarchy test. Condition D isolates recency effects: if SCR in C is much higher than first-instruction compliance in D, the hierarchy is real rather than a positional artifact.

Counterbalancing (both `a_to_b` and `b_to_a` directions) is applied to C and D to control for capability bias between options.

## Prompt Counts

Per experiment pair, per model (current config: 8 pairs, 16 tasks, 1 instance, counterbalancing on):

| Condition | Formula | Count (per pair) | Total (8 pairs) |
|-----------|---------|-----------------|-----------------|
| A | 2 dirs × T × I | 2 × 16 × 1 = **32** | 256 |
| B | 2 dirs × T × I | 2 × 16 × 1 = **32** | 256 |
| C | 2 dirs × S × U × T × I | 2 × 3 × 4 × 16 × 1 = **384** | 3,072 |
| D | 2 dirs × T × I | 2 × 16 × 1 = **32** | 256 |
| **Total** | | **480** | **3,840** |

Variables: T = tasks, I = instances per cell, S = system strengths (C only), U = user styles (C only).

## Setup

From the repo root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync
```

## HuggingFace API Token

Experiments run against the HuggingFace Inference API. Set `HF_TOKEN` in your `<name>.sync.env` and source it before running:

```bash
source <name>.sync.env
```

## Configuring Experiments

Edit `config/experiment.yaml` to configure:

- **`models`** — List of HuggingFace model IDs to evaluate
- **`constraint_types`** — Instruction types to test (language, format, starting word, etc.) with their option pools
- **`system_templates`** — System prompt templates at different strength levels (weak/medium/strong)
- **`user_templates`** — User message styles (with_instruction/polite/jailbreak)
- **`generation`** — `instances_per_cell` controls how many samples per condition

## Running Experiments

```bash
uv run python run_experiments.py
```

Results are saved as JSONL files in `data/results/`. The runner uses SHA-256 hashing for deduplication, so re-running skips already-completed experiments.

## Generating Reports

```bash
uv run python generate_report.py
```

Produces an interactive HTML report at `reports/report.html`. Options:

```bash
uv run python generate_report.py --results-dir data/results --output reports/report.html
```

## Tests

```bash
uv run pytest              # all tests
uv run pytest -k "test_name"  # single test by name
```
