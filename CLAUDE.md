# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Lambda AI Instance Setup

This project runs on Lambda AI cloud instances. Each team member has their own filesystem and bucket for data persistence across instances and regions. See `SYNC.md` for full documentation.

### New instance bootstrap

When a user asks to "set up the instance" or "clone the repo and set up", follow these steps:

1. **Detect the filesystem name** from the current mount:
   ```bash
   ls /lambda/nfs/
   ```

2. **Ask the user for their name** (used for the config file name).

3. **Clone the repo:**
   ```bash
   cd /lambda/nfs/<filesystem-name>
   git clone https://github.com/ebalp/system-user-circuits.git
   cd system-user-circuits
   ```

4. **Check if `<name>.sync.env` already exists** (it would if they previously uploaded to their bucket and are downloading onto a new filesystem). If it does not exist, **ask the user**: "Do you have your `<name>.sync.env` file? You can upload it through Jupyter to the repo directory. Otherwise I can create one if you give me the values."

   - **If they can upload it**: Wait for them to upload `<name>.sync.env` into the repo directory, then continue.
   - **If they need to create it**: Ask for these values:
     - BUCKET_NAME (their personal Lambda AI filesystem bucket UUID)
     - LAMBDA_ACCESS_KEY_ID (from Lambda Cloud console → Filesystem → S3 Adapter Keys)
     - LAMBDA_SECRET_ACCESS_KEY (same source)
     - LAMBDA_REGION (default: us-east-2)
     - LAMBDA_ENDPOINT_URL (default: https://files.us-east-2.lambda.ai)
     - GIT_USER_NAME (their full name for git commits)
     - GIT_USER_EMAIL (their email for git commits)
     - GITHUB_TOKEN (classic token from GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic), with repo scope)

     Then create `<name>.sync.env` from the template with those values.

5. **Run setup** to configure git:
   ```bash
   ./lambda-sync.sh <name>.sync.env setup
   ```

6. **Download from bucket** (if the user has previous work): follow the sync protocol below.

7. **Set up the Python environment** from the repo root. The `.venv` is never synced to the bucket (rebuilding with `uv sync` is faster than transferring it), so this step is always needed on a new instance:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.12
   uv sync
   ```

### Running sync commands (upload / download)

The sync script uses interactive confirmation prompts that Claude Code cannot handle natively. **Never bypass these silently** — the overwrite warning on `download` is a real safeguard.

**Protocol for every upload or download:**

1. **Run the bucket preview** as a standalone command and show the output to the user in the conversation:
   ```bash
   source <name>.sync.env && \
   AWS_ACCESS_KEY_ID=$LAMBDA_ACCESS_KEY_ID \
   AWS_SECRET_ACCESS_KEY=$LAMBDA_SECRET_ACCESS_KEY \
   AWS_DEFAULT_REGION=$LAMBDA_REGION \
   aws s3 ls s3://$BUCKET_NAME/ --endpoint-url $LAMBDA_ENDPOINT_URL
   ```

2. **Show the user the bucket contents and explicitly state what will happen** before asking anything:
   - For **download**: "Bucket contains X. This will overwrite everything in `/lambda/nfs/<filesystem>/` with those contents. Local changes not previously uploaded will be lost."
   - For **upload**: "Bucket currently contains X. This will sync `/lambda/nfs/<filesystem>/` to your bucket, overwriting older bucket contents."

3. **Wait for the user to explicitly confirm** in the conversation. Do not proceed until they do.

4. **Only after user confirmation**, run the command with piped responses (the script prompts twice for download — once to confirm the bucket, once to confirm the overwrite):
   ```bash
   # download — two prompts
   printf "y\ny\n" | bash ./lambda-sync.sh <name>.sync.env download

   # upload — one prompt
   printf "y\n" | bash ./lambda-sync.sh <name>.sync.env upload
   ```

### Before shutting down

Always remind the user to upload before terminating, following the sync protocol above:
```bash
printf "y\n" | bash ./lambda-sync.sh <name>.sync.env upload
```

### Key files

- `lambda-sync.sh` — sync script (setup/upload/download)
- `sync.env.template` — config template for new team members
- `<your-name>.sync.env` — personal config (gitignored, syncs with bucket)
- `SYNC.md` — full sync documentation

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
# Requires HF API token in hf_token.txt or HF_API_KEY env var
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
