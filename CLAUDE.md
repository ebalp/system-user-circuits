# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Lambda AI Instance Setup

### New instance bootstrap

When a user asks to "set up the instance" or "clone the repo and set up", follow these steps:

1. **Find the filesystem mount and clone the repo:**
   ```bash
   ls /lambda/nfs/
   cd /lambda/nfs/<filesystem-name>
   git clone https://github.com/ebalp/system-user-circuits.git
   cd system-user-circuits
   ```

2. **Locate the config file:**
   ```bash
   ls *.sync.env
   ```
   If none exists, ask the user: "Do you have a `.sync.env` config file? You can upload it through Jupyter to the repo directory. Otherwise I can create one from the template if you give me the values."

   - **If they can upload it**: Wait for them to upload it into the repo directory, then continue.
   - **If they need to create it**: Ask for the values listed in `sync.env.template` and create `config.sync.env` from them.

3. **Run setup** to configure git:
   ```bash
   ./lambda-sync.sh <config>.sync.env setup
   ```

4. **Download from bucket** (if the user has previous work): follow the sync protocol below.

5. **Set up the Python environment** from the repo root. The `.venv` is never synced to the bucket, so this step is always needed on a new instance:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.12
   uv sync
   ```

### Running sync commands (upload / download)

The sync script has one confirmation prompt for both upload and download. Claude Code cannot handle interactive prompts natively, so the protocol is:

1. **State clearly what will happen** before running anything:
   - For **upload**: "This will overwrite bucket data with local `*/data/` and `*/reports/` from the repo."
   - For **download**: "This will overwrite local `*/data/` and `*/reports/` with bucket data. Local changes not uploaded will be lost."

2. **Wait for the user to confirm** in the conversation.

3. **Only after confirmation**, pipe the response:
   ```bash
   printf "y\n" | bash ./lambda-sync.sh <config>.sync.env upload
   printf "y\n" | bash ./lambda-sync.sh <config>.sync.env download
   ```

### Before shutting down

Always remind the user to upload before terminating. Follow the sync protocol above.

## Git Conventions

Do not add `Co-Authored-By` trailers to commit messages.

## Project Overview

This is the **Instruction Hierarchy Evaluation System** — a research platform for evaluating how LLMs handle conflicting instructions between system prompts and user messages. The project is organized in phases; currently only **Phase 0 (Behavioral Analysis)** is implemented under `phase0_behavioral_analysis/`.

## Commands

### Environment setup

The project uses `uv`. The `.venv` lives at the repo root and is stored on the NFS filesystem, so it persists across instance restarts within a region. When switching regions, recreate it with `uv sync` (fast).

From the repo root:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync
```

To run scripts or tools:
```bash
uv run python <script.py>
uv run pytest
```

Or activate for interactive work:
```bash
source .venv/bin/activate
```

### Running Tests

All test commands from `phase0_behavioral_analysis/`:
```bash
# All tests
uv run pytest

# Single test file
uv run pytest tests/test_config.py

# Single test by name
uv run pytest tests/test_config.py -k "test_name"

# Verbose output
uv run pytest -v
```

### Running Experiments
```bash
# From phase0_behavioral_analysis/
# Source your config first to load HF_API_KEY: source <config>.sync.env
uv run python run_experiments.py
```

### Generating Reports
```bash
# From phase0_behavioral_analysis/
uv run python generate_report.py --results-dir data/results --output reports/report.html
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
