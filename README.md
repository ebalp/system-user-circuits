# System-User Circuits

Code companion for the workshop paper on instruction hierarchy in language models: how models resolve direct conflicts between system and user instructions, how that behavior is measured, and how the resulting conflict outcome can be probed and steered from internal activations.

This branch is intentionally trimmed. It contains only the code and tracked artifacts relevant to the workshop submission:

- `phase0_v2/`
- `linear_probe_new/`
- `phase1_linear_probing/`
- minimal repo files: `README.md`, `pyproject.toml`, `uv.lock`

## What This Branch Covers

The tracked code supports the workshop pipeline in three layers:

1. **Behavioral evaluation**
   `phase0_v2/` generates the conflict prompts, runs model evaluation, applies deterministic verifiers, calibrates thresholds, and produces behavioral summaries such as system compliance under conflict.

2. **Detailed probing experiments**
   `linear_probe_new/` contains the more exploratory probing analyses used to inspect separability, layer structure, style effects, token-level projections, and diagnostic comparisons across probe constructions.

3. **Main probing and steering pipeline**
   `phase1_linear_probing/` loads the behavioral outputs from `phase0_v2`, extracts residual-stream activations, fits linear probes and steering directions, and runs the steering analyses used in the workshop study.

The branch also intentionally keeps:

- `phase0_v2/calibration/output/` for saved verifier-audit artifacts
- `phase0_v2/reports/*.html` for saved behavioral reports

## Repository Map

### `phase0_v2/`

Behavioral pipeline for system-vs-user instruction conflicts.

Key contents:

- `run_experiments.py`: generate prompts and collect model responses
- `generate_report.py`, `generate_model_report.py`: build summary reports
- `config/`: experiment definitions, conflict metadata, thresholds
- `conflicts/`: registered conflict definitions and verifier logic
- `calibration/`: threshold selection, verifier auditing, rescoring, refusal tagging
- `src/`: prompt generation, experiment orchestration, metrics, API clients
- `tests/`: behavioral and verifier test suite
- `reports/`: saved HTML reports

This is the code corresponding to the workshop paper's benchmark, verifier, and behavioral-regime sections.

### `linear_probe_new/`

Detailed probing-analysis scripts used between the behavioral pipeline and the final steering-focused pipeline.

Key contents:

- `run_probing.py`: broad probing runs across methods, layers, and token positions
- `run_diagnostics.py`: diagnostic analyses for probe behavior and separability
- `plot_results.py`: plotting utilities for probing outputs
- `analyze_per_category.py`: per-category probing analysis
- `analyze_style_scatter.py`: style-conditioned probing visualizations
- `extract_activations.py`, `extract_token_projections.py`: activation and token-level projection extraction

This directory is best thought of as the detailed probing-experiment layer: more exploratory and analysis-heavy than `phase1_linear_probing/`, and useful for reproducing the intermediate probing investigations behind the workshop results.

### `phase1_linear_probing/`

Main probing and steering pipeline.

Key contents:

- `data.py`: run configuration, behavioral-data loading, token positions, activation I/O
- `probe.py`: probe fitting and persistence
- `metadata_clf.py`: metadata-only baselines and ablations
- `steer.py`: steering utilities and evaluation
- `compute_iid_mm.py`, `compute_cmds.py`, `compute_per_constraint_probes.py`: direction construction scripts
- `run_steering_pipeline.py`, `steering_server.py`: steering workflows
- `tests/`: probing and steering test suite

This is the code corresponding to the workshop paper's probing and steering sections.

## Expected Workflow

At a high level:

1. Run `phase0_v2` to generate or load behavioral results.
2. Use `phase0_v2/calibration/` to audit verifiers and finalize thresholds if needed.
3. Use `linear_probe_new` for detailed probing experiments and diagnostic analysis.
4. Use `phase1_linear_probing` to load the behavioral results, extract activations, fit probes, and run steering experiments.

The default handoff is:

- `phase0_v2/data/results/*.jsonl` -> behavioral results
- `linear_probe_new/` -> detailed probing experiments on those behavioral outputs
- `phase1_linear_probing/data/` -> activations, probe runs, and steering artifacts

## Running Tests

From the repo root:

```bash
uv sync
uv run pytest phase0_v2/tests/ -v
uv run pytest phase1_linear_probing/tests/ -v
```

## Scope Notes

This branch is a code-focused workshop snapshot, not the full research repository.

- It does **not** include the earlier legacy behavioral pipeline.
- It does **not** include the earlier legacy linear probing directory.
- It does **not** include the dataset-construction directory that was removed from this branch.
- It does include tracked generated artifacts under `phase0_v2/` because they document verifier calibration and behavioral outputs used during the workshop cycle.

For implementation details inside each subsystem, use:

- [phase0_v2/README.md](/Users/enrique/system-user-circuits/phase0_v2/README.md)
- [phase1_linear_probing/README.md](/Users/enrique/system-user-circuits/phase1_linear_probing/README.md)
