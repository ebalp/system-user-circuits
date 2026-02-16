# Phase 0: Behavioral Analysis

Evaluates how LLMs handle conflicting instructions between system prompts and user messages across 4 experimental conditions:

- **Condition A** — System-only baseline: system prompt has a constraint, user has only a task
- **Condition B** — User-only baseline: generic system prompt, user has constraint + task
- **Condition C** — Hierarchy conflict: system and user have conflicting constraints
- **Condition D** — Recency conflict: user message contains two contradictory constraints

## Setup

```bash
cd phase0_behavioral_analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
python run_experiments.py
```

Results are saved as JSONL files in `data/results/`. The runner uses SHA-256 hashing for deduplication, so re-running skips already-completed experiments.

## Generating Reports

```bash
python generate_report.py
```

Produces an interactive HTML report at `reports/report.html`. Options:

```bash
python generate_report.py --results-dir data/results --output reports/report.html
```

## Tests

```bash
pytest              # all tests
pytest -k "test_name"  # single test by name
```
