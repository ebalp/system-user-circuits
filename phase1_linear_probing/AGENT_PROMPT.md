# Steering Exploration Agent — Session Prompt

You are a steering exploration agent. Your job is to run experiments on the steering server for hours, deepening your understanding of how activation steering controls instruction hierarchy behavior in Llama-3.1-8B-Instruct.

## Setup

1. Read `phase1_linear_probing/AGENT_EXPLORATION.md` for the full experimental protocol.
2. Read `phase1_linear_probing/data/runs/curated4-8b-v002/FINDINGS_REPORT.md` for prior findings from the previous exploration (~120 experiments).
3. Check `phase1_linear_probing/data/runs/curated4-8b-v002/exploration_v2/` for any results already saved in this session.
4. The steering server should be running at `http://localhost:8000`. Verify with `/health`. If it's not running, tell the user.

## Utilities

All experiment helpers are in `phase1_linear_probing/explore_utils.py`:

```python
import sys
sys.path.insert(0, "phase1_linear_probing")
from explore_utils import (
    get_sample_ids, generate, summarize, save_experiment,
    add_notes, steering_clues, get_projection_stats,
    CONFLICT_IDS, FINDINGS_DIR,
)
```

Working directory is `/home/ubuntu/system-user-circuits`. Save results to `FINDINGS_DIR`.

## Critical Design Decisions

### Coherence detector: only filters gibberish
The `genuine_scr` metric excludes ONLY gibberish (`repetition_loop` and `too_short`). Refusals and meta-commentary are coherent text and ARE counted. Rationale:
- A bare refusal like "I can't fulfill that request" IS genuinely `followed_system` for absence constraints (e.g., `json_b_to_a` where system says "plain text" — no JSON was produced).
- Meta-commentary responses that discuss the instruction conflict are still coherent, on-topic text.
- The only thing we want to filter is degenerate garbage that inflates verifier labels meaninglessly.

### No Phase 0 label filtering
Use ALL Condition C samples. Do NOT filter by `baseline_label`. The Phase 0 labels reflect vLLM generation, not HF `generate()`. Measure the server's own unsteered baseline as ground truth.

### past_vs_present_tense has a noisy float verifier
Its unsteered baseline is already ~15-20% `followed_system` (vs ~0-9% for other constraints). Weight conclusions about tense steering accordingly — small SCR lifts may be noise.

## How to Work

### Continuous exploration for hours
- Set a 30-minute recurring reminder (CronCreate) to wake yourself up and continue.
- After each experiment: save results, read the printed output (especially genuine followed_system responses), write notes via `add_notes()`, then decide what to run next.
- Think between experiments. Form hypotheses. Check if results make sense. Don't just grind through a fixed grid.

### Make good use of GPU time
- Run experiments in the background.
- Set sleep N bash background processes to keep track.
- While waiting for results, plan the next experiment or analyze prior results.
- Send prompts in multiples of 96 (the batch size) to maximize GPU utilization.
- Each 768-sample experiment takes ~5-7 minutes on the A100 at batch_size=96.

### One experiment at a time
Do NOT batch multiple experiments into one script. Run one, read the output, add notes, then decide what's next. This is research, not batch processing.

## Experimental Priorities

### Phase 1: Validate prior findings with coherence scoring
1. **Unsteered baseline** — record genuine_scr per (constraint, direction) cell. This is ground truth.
2. **L12 probe projection target=5** — prior best config (SCR=0.235 at n=200). Validate with coherence.
3. **L12 probe additive alpha=5** — compare to projection.
4. **L4 probe (additive alpha=5 AND projection target=5)** — peak layer in prior H1 sweep (SCR=0.33, n=48) but NEVER validated at large n. Highest priority new experiment.
5. **L12 projection targets 6 and 7** — prior H2 found proj=7 peaked at raw_scr=0.415. Check response quality.
6. **L14 probe additive alpha=10** — claimed 100% json b_to_a flip but suspected degenerate text.

### Phase 2: Layer sweep with informed parameters
Use `steering_clues()` to get informed alpha/target per layer. Sweep layers 2-20 with probe directions. Map the causal window.

### Phase 3: Deep dives on promising configs
Alpha/target sweeps, full sample sets, bidirectional steering, per-constraint breakdowns.

### Phase 4: Cross-conflict analysis
Per-conflict probes, cosine similarity, cross-steering matrix.

## Key Reminders

- **Sign convention**: positive = toward followed_system. Adding `+alpha * direction` pushes toward system compliance.
- **b_to_a asymmetry**: steering almost exclusively flips b_to_a samples. Always report a_to_b and b_to_a separately.
- **L25 is inert**: don't test it despite high probe accuracy.
- **CMD at alpha >= 8 at L12 is always degenerate**: don't waste compute.
- **GPU lock**: only one generation request at a time. Don't send concurrent requests.
- The server has per-model thresholds activated.
