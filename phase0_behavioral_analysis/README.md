# Phase 0: Behavioral Analysis

Evaluates how LLMs handle conflicting instructions between system prompts and user messages across 4 experimental conditions:

- **Condition A** — System-only baseline: system prompt has a constraint, user has only a task
- **Condition B** — User-only baseline: generic system prompt, user has constraint + task
- **Condition C** — Hierarchy conflict: system and user have conflicting constraints
- **Condition D** — Recency conflict: user message contains two contradictory constraints

## Setup

From the repo root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync
```

## HuggingFace API Token

Experiments run against the HuggingFace Inference API. Provide your token in one of two ways:

- **File**: Create `hf_token.txt` in this directory with your token
- **Environment variable**: `export HF_API_KEY=hf_...`

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
