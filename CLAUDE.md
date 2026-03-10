# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Lambda AI Instance

When running on a Lambda instance, `.sync.env` is auto-sourced on login (credentials, tokens, etc.).

### Data sync

Data sync uses the `lambda-gpu` CLI from `lambda-cloud-toolkit`:

```bash
# Upload results to bucket
lambda-gpu sync upload phase0_v2/data/results

# Download data from bucket
lambda-gpu sync download phase0_v2/data

# List bucket contents
lambda-gpu sync ls
```

Before running, explain to the user what will be synced and the direction (upload overwrites bucket, download overwrites local). Wait for confirmation before running. Always remind the user to upload before terminating an instance.

## Git Conventions

Do not add `Co-Authored-By` trailers to commit messages.

## Project Overview

**Instruction Hierarchy Evaluation System** — a research platform studying how LLMs handle conflicting instructions between system prompts and user messages.

### Phases

| Phase | Directory | What it does |
|-------|-----------|-------------|
| **Lambda Cloud** | `lambda-cloud-toolkit` (external package) | GPU instance lifecycle, SSH operations, vLLM deployment. Installed via `lambda-cloud-toolkit`. Config: `lambda-cloud.yaml`. |
| **Phase 0 v2** | `phase0_v2/` | Class-based behavioral experiments: 33 conflicts, 4 conditions (A-D), 5 system/user styles, counterbalancing. Calibration system for verifier quality analysis and tier assignment. |
| **Phase 0** | `phase0_behavioral_analysis/` | Original behavioral experiments (legacy). |
| **Phase 1** | `phase1_linear_probing/` | Mechanistic analysis: trains per-layer linear probes on residual-stream activations to find directions separating "followed system" vs "followed user". Includes metadata baselines, grouped CV, and direction analysis. |

Each phase has its own `README.md` with detailed module maps, function references, and workflows. Read the relevant README when working on that phase.

### Data Flow Across Phases

Phase 0 produces `data/results/{model}_results.jsonl` → Phase 1 reads these via `load_results()`, filters to Condition C (hierarchy conflict), extracts activations, and trains probes.

## Commands

### Environment Setup

The project uses `uv`. The `.venv` lives at the repo root.

```bash
uv python install 3.12
uv sync
```

To run scripts: `uv run python <script.py>` or `uv run pytest`. Or activate: `source .venv/bin/activate`.

### Running Tests

```bash
# Phase 0 v2
uv run pytest phase0_v2/tests/ -v

# Phase 0 legacy (from phase0_behavioral_analysis/)
uv run pytest

# Phase 1 (from repo root)
uv run pytest phase1_linear_probing/tests/ -v
```

### Running Calibration Analysis

```bash
uv run python -m phase0_v2.calibration.analyze \
  phase0_v2/data/results/meta-llama_Llama-3.1-8B-Instruct_results.jsonl \
  --output-dir phase0_v2/calibration/output/
```

### Lambda Cloud Scripts

```bash
# Snatch a GPU instance and bootstrap it
lambda-gpu snatch --setup

# Bootstrap an existing instance
lambda-gpu setup --ip IP

# Launch vLLM on an instance with SSH tunnel
lambda-gpu vllm --ip IP --model meta-llama/Llama-3.1-8B-Instruct --tunnel
```

### Running Phase 0 Experiments

```bash
# Source env for HF_TOKEN, then from phase0_behavioral_analysis/
source <config>.sync.env
uv run python run_experiments.py
```

### Generating Phase 0 Reports

```bash
# From phase0_behavioral_analysis/
uv run python generate_report.py --results-dir data/results --output reports/report.html
```

## Key Concepts

- **4 Conditions**: A (system baseline), B (user baseline), C (hierarchy conflict — main test), D (recency control)
- **Constraint types**: 33 conflicts across language, format, starting_word, capitalization, emoji, disclaimer, list_format, self_reference, person, word count, etc.
- **Counterbalancing**: both `a_to_b` and `b_to_a` directions for conditions C and D
- **Grouped CV** (Phase 1): `GroupKFold` by `constraint_type` prevents leakage; `stratified` mode available for comparison
- **Experiment hashing**: SHA-256 deduplication for resumable Phase 0 runs; Phase 1 `run_id` hashes config + data contents

## Testing

Phase 0 uses `pytest` + `hypothesis` for property-based testing. Phase 1 tests use synthetic data (no GPU or model needed). Test files mirror source modules in both phases.
