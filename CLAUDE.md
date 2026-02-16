# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Instruction Hierarchy Evaluation System** — a research platform for evaluating how LLMs handle conflicting instructions between system prompts and user messages. The project is organized in phases; currently only **Phase 0 (Behavioral Analysis)** is implemented under `phase0_behavioral_analysis/`.

## Commands

All commands should be run from the `phase0_behavioral_analysis/` directory.

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Running Tests
```bash
# All tests
pytest

# Single test file
pytest tests/test_config.py

# Single test by name
pytest tests/test_config.py -k "test_name"

# Verbose output
pytest -v
```

### Running Experiments
```bash
# Requires HF API token in hf_token.txt or HF_API_KEY env var
python run_experiments.py
```

### Generating Reports
```bash
python generate_report.py --results-dir data/results --output reports/report.html
```

## Architecture

### Experiment Design (4 Conditions)

The system tests models under four experimental conditions:

- **Condition A** (System-only baseline): System prompt has a constraint, user has only a task. Measures SBR (System Baseline Rate).
- **Condition B** (User-only baseline): Generic system prompt, user has constraint + task. Measures UCR (User Compliance Rate).
- **Condition C** (Hierarchy conflict): System and user have *conflicting* constraints. Measures SCR (System Compliance Rate) and Hierarchy Index.
- **Condition D** (Recency conflict): User message contains two contradictory constraints in sequence. Measures recency effect.

Counterbalancing tests both directions (a_to_b and b_to_a) for conditions C and D to detect capability bias.

### Data Flow

1. **Config** (`config/experiment.yaml`) → `src/config.py` loads and validates YAML into typed dataclasses (`ExperimentConfig`, `ConstraintType`, `ExperimentPair`, etc.)
2. **Prompt generation** → `src/prompts.py` (`PromptGenerator`) expands config into all prompt combinations, or `src/experiment.py` (`ExperimentRunner.generate_experiment_keys()`) generates `ExperimentKey` objects for the dedup-based runner
3. **API calls** → `src/api_client.py` (`HFClient`) calls HuggingFace Inference API with retry/backoff. Strips `<think>` blocks from Qwen3 models.
4. **Classification** → `src/classifiers.py` classifies responses: `LanguageClassifier` (langdetect), `FormatClassifier` (JSON/YAML/plain), `StartingWordClassifier`. `compute_label()` determines compliance label (`followed_system`, `followed_user`, `followed_both`, `followed_neither`).
5. **Results** → Stored as JSONL in `data/results/{model_safe_name}_results.jsonl` with full metadata per record
6. **Metrics** → `src/metrics.py` (`MetricsCalculator`) computes SCR, UCR, SBR, Hierarchy Index, Conflict Resolution Rate, Recency Effect, with Wilson CIs and directional breakdowns
7. **Reporting** → `src/reporting.py` (`ReportGenerator`) produces matplotlib charts and markdown; `src/report/` generates interactive HTML reports

### Key Concepts

- **Constraint types**: Defined in config with templates (`instruction_template`, `negative_template`) and an options pool. Supported classifiers: `language`, `format`, `yaml`, `starting_word`.
- **System templates** define strength levels (weak/medium/strong) with `{instruction}` and `{negative}` placeholders.
- **User templates** define styles (with_instruction/polite/jailbreak) with `{instruction}` and `{task}` placeholders.
- **Experiment hashing**: Each experiment is uniquely identified via SHA-256 hash of all parameters for deduplication and resumability.
- **Go/no-go thresholds**: Hierarchy Index > 0.7, Conflict Resolution > 0.8, adjusted asymmetry ≤ 0.15.

### Testing

Tests use `pytest` with `hypothesis` for property-based testing (especially in `test_config.py`). Test files mirror source modules. Tests import from `src.*` using relative package imports.
