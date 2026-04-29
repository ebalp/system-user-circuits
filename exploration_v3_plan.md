# Plan: IID Mass-Mean Steering Directions for `curated4-8b-v002`

> **For the executing Claude Code session.** This plan was authored in a prior session that built up the rationale interactively. Treat the rationale below as load-bearing context — read it before delegating any tasks. The plan is structured for orchestration: an onboarding phase, then four parallel opus subagent tasks, each self-testing.

---

## Context

Phase 1 steering exploration in this repo currently uses LR probe weights (`probe_*`) and class-mean differences (`cmd_*`) as steering directions. Two limitations motivated this round:

1. **Direction geometry.** LR weights are decision-boundary normals shaped by an isotropic L2 penalty (basis-dependent); CMDs are unweighted mean differences with no whitening. Neither is the "cheapest perturbation that maximally separates classes." That is Fisher's discriminant / IID Mass-Mean (IID-MM): `w = Σ⁻¹(μ_+ − μ_-)`. The Σ⁻¹ explicitly suppresses high-variance noise dimensions — exactly the steering objective.

2. **Alpha semantics across layers.** Existing alphas (e.g., 5, 10) bear no consistent relationship to per-layer activation magnitudes. We want a layer-comparable scale where an alpha of 1 means "shift activations by 1 standard deviation in the projection along the direction."

A separate analysis pipeline at `linear_probe_new/run_probing.py:276` already implements `iid_mass_mean_probe_gpu`, but it fits in z-scored space (per-feature standardization). The per-neuron z-score is not principled — neuron axes are not a privileged basis (downstream linear maps mix them freely). Mathematically, Fisher's LDA is invariant under invertible linear transforms; the standardization only shifts the basis in which the small `+εI` regularization sits. So we fit IID-MM in **rawspace** (no per-feature standardization), keeping a z-scored version available for analysis only.

**Independence note:** `phase1_linear_probing/` owns its own iid_mm implementation. We **do not** vendor or import from `linear_probe_new/`; we re-derive the algorithm in phase1 (it's ~20 lines). `linear_probe_new/run_probing.py` is referenced once for understanding only. If `linear_probe_new` later changes its regularizer or signature, phase1 is unaffected.

**Naming convention — important to avoid confusion:**
- "**rawspace**" vs "**zscored**" describes the **fit basis** (whether features were standardized before solving for `w`).
- "**raw**" (as already used in `steer.py` for `probe_raw`, `cmd_overall_raw`) describes a **stored vector that is not unit-normalized** — i.e., a normalization status, not a fit basis.
- These are orthogonal axes. All iid_mm vectors stored on disk are unit-norm (matching the existing `probe`/`cmd_overall` non-`_raw` convention). The rawspace-vs-zscored distinction lives in the key suffix only.

The next exploration round is **IID-MM-only** (no LR probe, no CMD), with global + per-conflict + mean-of-per-conflict variants. Per-conflict averaging (mean of unit per-conflict directions, renormalized) is included because in the prior CMD round it outperformed the global probe for steering — likely worth testing with IID-MM too.

The existing `AGENT_EXPLORATION.md` workflow stays in place; only direction names, alpha semantics, and output folder change.

## Locked-in design decisions

- **Algorithm:** IID-MM only, re-implemented in `phase1_linear_probing/` (no dependency on `linear_probe_new`). Both rawspace fit (`scale=False`, used for steering) and zscored fit (`scale=True`, kept for analysis).
- **Variants:** global; per-conflict (4, one per `conflict_id` in curated4); mean-of-per-conflict (average of 4 unit rawspace per-conflict directions, then renormalize).
- **Token position:** `last_prompt`.
- **Sample set:** existing `phase1_linear_probing/data/runs/curated4-8b-v002/` activations. No re-extraction — reuse `act_nn_*.npz` that drives the existing probe/CMD pipeline.
- **Storage:** all stored steering vectors are **unit-norm** (matches existing probe/CMD convention; keeps `/projection_stats` units coherent). A separate JSON sidecar holds per-direction-per-layer projection-std multipliers.
- **Alpha convention:** stored vectors stay unit-norm. A new `iid_mm_steering_clues()` helper surfaces `proj_std`, `y0_mean`, `y0_std`, `y1_mean`, `y1_std`. Two modes:
  - **Additive:** `alpha = N × clues["alpha_per_std"]` (= `N × proj_std`). Sweep N ∈ {0.5, 1, 1.5, 2}. Unit: overall projection std.
  - **Projection:** `projection_target = clues["y1_mean"] + N_extra × clues["y1_std"]`. Sweep N_extra ∈ {-0.5, 0, 0.5, 1.0}. Unit: per-class std of the followed_system class. N_extra<0 lands near the boundary; N_extra>0 pushes deeper into followed_system.
- **Experiment output folder:** `{run_dir}/exploration_v3/`.

---

## Phase 1 — Onboarding exploration agents (orchestrator runs first)

Launch the following Explore agents **in parallel** (single message, multiple tool calls). They establish context that subsequent implementation subagents need; the orchestrator should read their findings and pass relevant excerpts into each implementation subagent's prompt rather than expecting subagents to re-explore.

### Explore agent 1 — Steering server direction loading

**Goal:** produce a precise reference document that Task C can edit against without re-reading these files. Output should be quoted code snippets + `file:line` citations, not prose summaries.

Required outputs:
1. **`steer.py:load_steering_directions()`** (around lines 26-114): paste the full return-dict shape and the order of NPZ-file reads. Note specifically whether each key is a single ndarray or a nested `{constraint_id: ndarray}` dict, and whether `_raw` (unnormalized) variants are returned alongside unit-norm versions. Confirm exact filenames read: `constraint_probes.npz`, `constraint_cmds.npz`, plus any legacy fallbacks (`fold_cmds_L*.npz`, `constraint_cmds_L*.npz`).
2. **`steering_server.py` startup loop** (around lines 280-400, the loop that calls `load_steering_directions`): paste the prefix-mapping logic that turns dict keys into server names. Specifically: how does it produce `probe_L12`, `probe_{constraint}_L12`, `cmd_overall_L12`? What are the exact prefix strings used for unpacking nested dicts? **This is critical for Task C to extend without breaking existing dispatch.**
3. **`compute_per_constraint_probes.py`**: NPZ key naming convention (`{constraint}_probe_L{layer}`, `{constraint}_probe_raw_L{layer}`), CLI args, and the activation-loading pattern around line 68-73 (`act_path = cfg.run_dir / f"act_nn_{cfg.safe_model_name}.npz"`).
4. **`compute_cmds.py`**: same — NPZ key naming for `constraint_cmds.npz`, plus how it loads activations.
5. **`compute_projection_stats.py:_select_directions()`** (around lines 139-164): paste the function. Note how it currently filters out `_raw` keys (so projection stats only cover unit-norm directions). Confirm output JSON schema around lines 213-214 — specifically the nesting `{constraint_or_overall: {Lk: {direction_name: {y0: {mean,std}, y1: {mean,std}}}}}` so Task D's `iid_mm_steering_clues` can index it correctly.
6. **`ls phase1_linear_probing/data/runs/curated4-8b-v002/`** at top level: list the NPZ/JSON artifacts already present (we expect `constraint_probes.npz`, `constraint_cmds.npz`, `projection_stats.json`, run config). Note: `act_nn_*.npz` may not exist locally — that's expected, activations live on Lambda.

### Explore agent 2 — IID-MM reference algorithm and curated4 metadata

**Goal:** give Task A everything it needs to (a) re-implement IID-MM in phase1 from first principles, and (b) recover `(X_row, y_row, conflict_id_row)` triples for per-conflict subsetting. Phase1 will not import from `linear_probe_new`, so the reference implementation is for understanding only.

Required outputs:
1. **Reference algorithm** at `linear_probe_new/run_probing.py:iid_mass_mean_probe_gpu` (lines 276-319): paste the full function. Note the exact computation steps (class means, pooled covariance, `Σ⁻¹(μ_+ − μ_-)`), the `+εI` regularization (currently `1e-6`), and the scaling branch (`scale=True` z-scores per-feature before solving). Task A will re-derive this from scratch in phase1 — do **not** suggest vendoring or importing.
2. **`prepare_condition_c`** at `phase1_linear_probing/data.py` (lines ~225-259): paste the full function. Confirm two facts Task A relies on:
   - The returned DataFrame **preserves `conflict_id` as a column** (it's carried through from `df_all` — the function only `.copy()`s and adds `y`).
   - The function accepts a `conflict_ids: list[str] | None` arg and filters when provided.
   Task A can therefore subset per-conflict either by reading `df["conflict_id"]` after a single call, or by calling `prepare_condition_c(df_all, conflict_ids=[cid])` four times. **No edits to `prepare_condition_c` are required.**
3. **Run config** at `phase1_linear_probing/data/runs/curated4-8b-v002/config.json` (or equivalent): paste the full file. List the `conflict_id`s (we expect 4), the model name, the loaded layers, and the positions list. Confirm `last_prompt` is among positions.
4. **Activation loader pattern** at `compute_per_constraint_probes.py:68-73`: paste the snippet. Task A will mirror this exactly: `act_path = cfg.run_dir / f"act_nn_{cfg.safe_model_name}.npz"; data = np.load(act_path); X = data[f"L{layer}_{pos}"]` (or whatever the actual key format is — confirm).
5. **`load_run_config`** wherever it lives (likely `data.py` or a config module): paste its signature so Task A knows how to discover layers and `safe_model_name` from a `--run-id` arg.

### Explore agent 3 — Test infrastructure

**Goal:** give Tasks A/B/C/D a concrete, paste-ready template for new test files. We've already confirmed there are no existing tests for `compute_cmds.py` or `compute_per_constraint_probes.py`, so the closest precedent is `test_steer.py`.

Required outputs:
1. **`tests/test_steer.py`** — paste the `tmp_path`-based fixture(s) that build a synthetic run dir with fake NPZ files. Specifically, find the helper that constructs a fake `{run_dir}/...npz` and any minimal `config.json`-equivalent the loaders expect. Cite line numbers. This is the template all four task subagents will mirror.
2. **Synthetic-data conventions used in `test_steer.py`**: typical d_model size (16? 32? 64?), number of layers, number of samples, how `y` is generated, how positions are simulated. Tasks A/B/D need to pick consistent sizes for their synthetic fixtures.
3. **List every file in `phase1_linear_probing/tests/`** and one-line summary of what each tests. Helps the four subagents avoid filename collisions.
4. **Test command** confirmed from CLAUDE.md: `uv run pytest phase1_linear_probing/tests/test_compute_iid_mm.py -v` (per-file) and `uv run pytest phase1_linear_probing/tests/ -v` (full suite). Confirm no special env vars or fixtures are required at session-level (no GPU, no model download).
5. **Anti-patterns to avoid**: any test in the directory that shells out to subprocess instead of calling main as a function, or that depends on real model artifacts. Flag them so subagents don't accidentally mirror them.

After all three return, the orchestrator writes a short context bundle (file paths, key signatures, NPZ key conventions, test patterns) and pastes it into each implementation subagent's prompt below.

---

## Phase 2 — Implementation subagents (parallel, opus)

All four subagents run in parallel with `subagent_type=general-purpose, model="opus"`. Each receives:
- The relevant context excerpts from Phase 1.
- The full "Locked-in design decisions" section above.
- Their task spec (below).
- An explicit instruction: **"Write tests and run them with `uv run pytest <test_file> -v`. Report test output. Do not mark the task complete until tests pass."**

Subagents should write code that compiles/imports cleanly even though Lambda-side end-to-end verification (which needs a GPU and the full server) happens later, by the user.

### Task A — `compute_iid_mm.py` (probing artifact generation)

**Deliverables:**
- New file `phase1_linear_probing/compute_iid_mm.py`.
- New test file `phase1_linear_probing/tests/test_compute_iid_mm.py`.

**Implementation spec:**

CLI:
```
--run-id <id>             # required, e.g. curated4-8b-v002
--layers L1 L2 ... [opt]  # default: all layers from run config
--pos last_prompt         # default
--reg-eps-rawspace 1e-3   # rawspace-fit Σ regularization, scaled by mean(diag(Σ))
--reg-eps-zscored 1e-6    # zscored-fit Σ regularization
--min-class-samples 50    # skip per-conflict fit if minority class smaller
```

Pipeline per layer:
1. Load `X = activations[args.pos]` from `{run_dir}/act_nn_{safe_model}.npz` (reuse the loader pattern from `compute_per_constraint_probes.py:68-73`).
2. Recover `y` and `conflict_id` per row from `prepare_condition_c(df_all)` — the returned DF already carries `conflict_id` as a column. Align rows with `X` by index.
3. Fit **global** IID-MM twice — once in **rawspace** (no per-feature standardization), once **zscored** (per-feature standardization first). Implement the algorithm directly in `compute_iid_mm.py`; do **not** import from `linear_probe_new`. The rawspace fit uses ridge `reg_eps_rawspace × mean(diag(Σ))` (scale-invariant). The zscored fit uses constant ridge `reg_eps_zscored`.
4. Fit **per-conflict** IID-MM (rawspace + zscored) for each of the 4 `conflict_id` values. Skip with a warning if either class < `min-class-samples`.
5. Compute **mean-of-per-conflict (rawspace)**: stack the 4 unit per-conflict rawspace directions, element-wise mean, renormalize.
6. Compute **per-direction projection std (rawspace only)**: project all curated4 Condition C activations at this layer (in raw activation space, no z-scoring) onto each unit rawspace direction; record `std(projections)`.
7. Record balanced accuracy per fit.

Outputs into `{run_dir}/`:
- **`iid_mm_directions.npz`** with keys (all unit-norm):
  - `overall_iid_mm_L{layer}` — global, rawspace fit
  - `overall_iid_mm_zscored_L{layer}` — global, zscored fit
  - `mean_iid_mm_L{layer}` — mean-of-per-conflict rawspace
  - `{conflict_id}_iid_mm_L{layer}` — per-conflict rawspace
  - `{conflict_id}_iid_mm_zscored_L{layer}` — per-conflict zscored
- **`iid_mm_proj_std.json`** — `{ direction_key: float }` for rawspace direction keys only (no `_zscored` keys).
- **`iid_mm_metrics.csv`** — columns `layer, scope (overall|per_conflict|mean), conflict_id (str|null), fit_basis (rawspace|zscored), bal_acc`.

**Tests** (`tests/test_compute_iid_mm.py`):
- **Style template:** mirror `tests/test_steer.py`'s `tmp_path` + synthetic-NPZ pattern (specifically the helper that builds a fake run dir — Explore agent 3 will surface its exact name and line range). Do not shell out; call the script's main as a function.
- Synthetic dataset with two clearly separable Gaussian classes in d_model=64, 2 layers, 2 fake conflict_ids. Construct fake `act_nn_*.npz` and a fake run dir with the minimum config the loader expects. Use `tmp_path` fixture.
- Assert: NPZ produced with all expected keys; every direction is unit-norm to 1e-5; `iid_mm_proj_std.json` exists with one positive scalar per rawspace key (and no `_zscored` keys); metrics CSV present with expected rows.
- **Fisher-invariance sanity:** rawspace vs zscored BA on the same data should match within ~2 percentage points for the global fit (small numerical differences from regularization basis).
- Edge case: per-conflict fit with one class artificially shrunk below `min-class-samples` — assert it's skipped with a warning, no key produced for that conflict.

### Task B — `plot_iid_mm_geometry.py` (pre-experiment visual sanity)

**Deliverables:**
- New file `phase1_linear_probing/plot_iid_mm_geometry.py`.
- New test file `phase1_linear_probing/tests/test_plot_iid_mm_geometry.py`.

**Implementation spec:**

CLI:
```
--run-id <id>
--layers L1 L2 ...        # default: all layers present in iid_mm_directions.npz
--pos last_prompt
--out-dir <path>          # default: {run_dir}/iid_mm_plots/
```

Per layer, build figure (matplotlib or plotly — match repo convention from Phase 1 context):
1. **Axes** (raw activation projection space, no z-scoring):
   - X = projection onto `overall_iid_mm_L{layer}`.
   - Y = projection onto **orthogonal component** of `mean_iid_mm_L{layer}` w.r.t. the global. That is: `d_perp = mean − (mean · overall) × overall`, renormalize to unit, project onto it.
2. **Scatter** of all curated4 Condition C samples at `last_prompt`. Color by binary label (`followed_user` blue, `followed_system` red). Marker shape per `conflict_id` (4 shapes).
3. **Vector overlays as arrows from origin:**
   - Unit `overall` direction (length 1 along X).
   - Unit `mean_perp` direction (length 1 along Y).
   - Steering vector at α=1 = `proj_std × unit_direction` for the global iid_mm direction (arrow of length `proj_std_x` along X — shows "1 std worth of perturbation").
   - Same for the mean direction (arrow lives in the X+Y plane since `mean = component_along_overall + component_along_perp`).
4. **Reference annotations**: dashed vertical lines at `y0_mean_x` and `y1_mean_x`. Title carries `proj_std_x`, Cohen's d, and BA.
5. **Side panels** (2×2 grid): per-conflict scatters using the same axes. Confirms the global direction generalizes per conflict.

Outputs: `{out_dir}/iid_mm_geometry_L{layer}.png` (and `.pdf` if matplotlib path is used).

**Tests** (`tests/test_plot_iid_mm_geometry.py`):
- **Style template:** mirror `tests/test_steer.py`'s `tmp_path` + synthetic-NPZ pattern.
- Build a tiny synthetic `iid_mm_directions.npz` + activations + metadata in `tmp_path`. Two layers, two conflicts, d_model=32, ~200 samples.
- Call the script's main function. Assert output files exist and are non-empty.
- Assert no exceptions raised when the `mean` direction collapses onto `overall` (degenerate case): the script should still produce a plot, with `mean_perp` flagged in the title or replaced by a fallback (e.g., principal residual direction).

### Task C — Server integration (`steer.py`, `steering_server.py`, `compute_projection_stats.py`)

**Deliverables:**
- Edits to `phase1_linear_probing/steer.py` (extend `load_steering_directions()`).
- Edits to `phase1_linear_probing/steering_server.py` (extend startup unpacking, add `/direction_scales` endpoint, update `/directions` patterns).
- Edits to `phase1_linear_probing/compute_projection_stats.py` (extend `_select_directions()` to include iid_mm names).
- New tests in `phase1_linear_probing/tests/test_iid_mm_loading.py` (no live server required).

**Implementation spec:**

`steer.py:load_steering_directions()` — append a block after the existing `constraint_probes.npz` loader that reads `iid_mm_directions.npz` and adds to the returned dict:
```python
{
  ...existing keys...,
  "iid_mm_overall": ndarray | None,
  "iid_mm_overall_zscored": ndarray | None,
  "iid_mm_mean": ndarray | None,
  "iid_mm_per_constraint": {cid: ndarray, ...},
  "iid_mm_per_constraint_zscored": {cid: ndarray, ...},
}
```

`steering_server.py` startup unpacking loop — extend the prefix logic to map:

| `load_steering_directions` key | Server direction name |
|--------------------------------|----------------------|
| `iid_mm_overall` | `iid_mm_L{layer}` |
| `iid_mm_overall_zscored` | `iid_mm_zscored_L{layer}` |
| `iid_mm_mean` | `iid_mm_mean_L{layer}` |
| `iid_mm_per_constraint[{cid}]` | `iid_mm_{cid}_L{layer}` |
| `iid_mm_per_constraint_zscored[{cid}]` | `iid_mm_zscored_{cid}_L{layer}` |

All stored unit-norm. **No pre-scaling.**

**Prefix-collision rule (explicit):** the existing unpacking loop dispatches on top-level keys with prefixes `"probe"` and `"cmd"` (see Explore agent 1's report for exact line numbers). The new `iid_mm_*` keys must be handled by a **separate dispatch branch keyed on the full prefix `"iid_mm"`**, not by the existing `"probe"`/`"cmd"` branches. Specifically:

- `iid_mm_overall`, `iid_mm_overall_zscored`, `iid_mm_mean` are flat ndarrays — handle directly, do **not** route through nested-dict unpacking.
- `iid_mm_per_constraint`, `iid_mm_per_constraint_zscored` are nested `{cid: ndarray}` dicts — unpack with `iid_mm` (or `iid_mm_zscored`) as the prefix when forming server names.
- The dispatch must explicitly check `key.startswith("iid_mm")` **before** any `key.startswith("probe")` / `key.startswith("cmd")` check, so `iid_mm_*` is never accidentally caught by a more permissive earlier branch.
- Add a unit test (in Task C's test file) asserting that introducing an `iid_mm_*` key does not cause it to be served under a `probe_*` or `cmd_*` name.

Add a one-time `iid_mm_proj_std.json` load into a new module-global `_direction_scales: dict[str, float]` keyed by server direction names (e.g., `iid_mm_L12`).

New endpoint:
```python
@app.get("/direction_scales")
async def direction_scales(): return _direction_scales
```

Update `/directions` endpoint Pydantic response with new pattern entries describing the iid_mm families.

`compute_projection_stats.py:_select_directions()` — add the new server direction names so `projection_stats.json` includes them (no schema change, just more keys per layer).

**Tests** (`tests/test_iid_mm_loading.py`):
- **Style template:** mirror `tests/test_steer.py`'s `tmp_path` + synthetic-NPZ pattern (Explore agent 3 will surface the helper to copy from).
- Build a fake `{run_dir}/iid_mm_directions.npz` and `iid_mm_proj_std.json` in `tmp_path` with a couple of layers and conflicts.
- Call `load_steering_directions(tmp_run_dir, "last_prompt", layer=12)` and assert all five new keys are present with correct shapes/types.
- Test the unpacking helper(s) used by `steering_server.py` directly (factor out the unpacking into a pure function if not already, so it can be unit-tested without spinning up FastAPI). Assert the produced server-name → vector dict matches the table above.
- **Prefix-isolation test:** seed the run dir with both `constraint_probes.npz` (existing) and `iid_mm_directions.npz` (new). Run unpacking. Assert (a) every `iid_mm_*` server name starts with `iid_mm` and (b) no `iid_mm_*` server name is also produced under a `probe_*` or `cmd_*` server name (i.e., no double-registration).
- Test that `_direction_scales` keys match server direction names.

### Task D — `explore_utils.py` helpers + `AGENT_EXPLORATION.md` rewrite

**Deliverables:**
- Edits to `phase1_linear_probing/explore_utils.py` (add `get_direction_scales()` and `iid_mm_steering_clues()`).
- Rewrite of `phase1_linear_probing/AGENT_EXPLORATION.md` for v3 round.
- New tests in `phase1_linear_probing/tests/test_iid_mm_clues.py`.

**Implementation spec:**

`explore_utils.py` — add next to existing `steering_clues()` (lines 138-182). Do **not** modify the existing helper.

```python
def get_direction_scales() -> dict:
    """Fetch per-direction projection-std multipliers from the server."""
    r = requests.get(f"{BASE}/direction_scales", timeout=30)
    r.raise_for_status()
    return r.json()

def iid_mm_steering_clues(
    stats: dict,
    scales: dict,           # from get_direction_scales()
    constraint: str,        # conflict_id or "_overall"
    direction_name: str,    # e.g. "iid_mm", "iid_mm_mean", f"iid_mm_{cid}"
    layer: int,
) -> dict:
    proj_std = scales[f"{direction_name}_L{layer}"]
    s = stats[constraint][f"L{layer}"][direction_name]
    return {
        "proj_std": proj_std,
        "y0_mean": s["y0"]["mean"],
        "y0_std":  s["y0"]["std"],
        "y1_mean": s["y1"]["mean"],
        "y1_std":  s["y1"]["std"],
        "separation": s["y1"]["mean"] - s["y0"]["mean"],
        "alpha_per_std": proj_std,
        "target_y1_minus_half_class_std": s["y1"]["mean"] - 0.5 * s["y1"]["std"],
        "target_y1":                       s["y1"]["mean"],
        "target_y1_plus_half_class_std":   s["y1"]["mean"] + 0.5 * s["y1"]["std"],
        "target_y1_plus_one_class_std":    s["y1"]["mean"] + 1.0 * s["y1"]["std"],
    }
```

`AGENT_EXPLORATION.md` rewrite:
- Replace "Available direction types" table with iid_mm-only entries (overall, mean, per-conflict; raw is steering, zscored is analysis).
- Replace alpha guidance: "iid_mm steering uses projection-std units. Fetch `clues = iid_mm_steering_clues(...)`. Two modes:
  - **Additive**: `alpha = N × clues['alpha_per_std']`, sweep N ∈ {0.5, 1, 1.5, 2}. Push higher only after coherence checks. Unit: overall projection std.
  - **Projection**: `projection_target = clues['y1_mean'] + N_extra × clues['y1_std']`, sweep N_extra ∈ {-0.5, 0, 0.5, 1}. Unit: per-class std of followed_system. N_extra<0 lands near class boundary; N_extra>0 pushes deeper into followed_system."
- Remove the "3× separation" / "y1_mean + 2×y1_std" heuristics throughout.
- Update output directory references: `exploration_v3/` (not `exploration_v2/`).
- Add a "Phase 0 — sanity visualization" step at the top of the protocol: run `plot_iid_mm_geometry.py` and inspect plots before any generation. Plots must show clean class separation along X and steering vector pointing into the followed_system cloud.
- Drop Phase 1 validation of prior LR/CMD claims (no longer the steering primitives). Keep coherence protocol unchanged — still mandatory.
- Restructure Phase 2 sweep around: layers × {overall, mean, 4 per-conflict} × {additive primarily, projection as secondary}.

**Tests** (`tests/test_iid_mm_clues.py`):
- **Style template:** mirror existing pure-function tests in `tests/test_steer.py` (no fixtures needed for this one — the helper takes plain dicts).
- Pure-function test of `iid_mm_steering_clues`: pass synthetic `stats` and `scales` dicts, assert all returned fields equal the expected math (e.g., `target_y1_plus_half_class_std == y1_mean + 0.5 × y1_std`).
- Test that `KeyError` is raised cleanly when the direction is missing from `scales` or `stats` (clear error message naming the missing key).
- Skip live HTTP testing (server isn't running in unit tests).

---

## Phase 3 — Orchestrator review and integration check

After all four subagents report passing tests:

1. Orchestrator reads each modified file directly (don't trust subagent summaries — verify changes).
2. Confirm cross-task contracts are aligned:
   - Task A's NPZ key naming matches what Task C reads.
   - Task A's `iid_mm_proj_std.json` keys match what Task C registers as `_direction_scales` keys (without the `_L{layer}` strip), and what Task D's `iid_mm_steering_clues()` looks up.
   - Task D's `AGENT_EXPLORATION.md` references match the actual server names produced by Task C.
3. Run the full test suite once at repo root: `uv run pytest phase1_linear_probing/tests/ -v`.
4. If everything passes, the orchestrator reports to the user and provides the Lambda execution recipe (Phase 4 below). Lambda-side execution is the user's responsibility — the orchestrator does **not** ssh, sync, or run anything on Lambda.

---

## Phase 4 — Lambda execution recipe (handed to the user)

```bash
# 1. Generate IID-MM artifacts
uv run python phase1_linear_probing/compute_iid_mm.py \
  --run-id curated4-8b-v002 \
  --layers 2 4 6 8 10 12 14 16 18 20 \
  --pos last_prompt

# 2. Generate geometry sanity plots — INSPECT BEFORE PROCEEDING
uv run python phase1_linear_probing/plot_iid_mm_geometry.py \
  --run-id curated4-8b-v002 \
  --layers 2 4 6 8 10 12 14 16 18 20

# 3. Refresh projection stats with new directions
uv run python phase1_linear_probing/compute_projection_stats.py \
  --run-id curated4-8b-v002

# 4. Restart steering server (auto-loads new artifacts)

# 5. Server sanity:
#    GET /directions  → lists iid_mm_L*, iid_mm_mean_L*, iid_mm_{cid}_L*, _zscored variants
#    GET /direction_scales  → one positive scalar per raw iid_mm key

# 6. Kick off agent-driven exploration writing into {run_dir}/exploration_v3/
```

End-to-end smoke (also in the user's hands):
- `iid_mm_steering_clues(...)` returns sensible numbers; `alpha = clues["alpha_per_std"]` at L12 gives meaningful behavioral shift in `summarize(...)` output.
- N=0 matches unsteered baseline; N=1 moves above baseline with minimal degeneration; N=3 degrades — confirms the std-units scale.
- Cross-layer (L4, L12, L20) at N=1: comparable magnitude of behavioral effect, no layer requiring N=0.1 or N=10.

---

## Critical files (for the orchestrator's mental map)

| File | Role | Touched by |
|------|------|-----------|
| `linear_probe_new/run_probing.py:276` | Reference algorithm only (read-only, do NOT import or modify) | A re-derives independently |
| `phase1_linear_probing/compute_per_constraint_probes.py` | Reference pattern + activation loader | A mirrors |
| `phase1_linear_probing/compute_cmds.py` | Reference pattern for unified NPZ output | A mirrors |
| `phase1_linear_probing/data.py:prepare_condition_c` | Recovers `y` and `conflict_id` per sample | A reuses |
| `phase1_linear_probing/steer.py:26-114` | `load_steering_directions()` | C extends |
| `phase1_linear_probing/steering_server.py:280-400, 436-463` | Startup loading + endpoints | C extends |
| `phase1_linear_probing/compute_projection_stats.py:139-164` | `_select_directions()` | C extends |
| `phase1_linear_probing/explore_utils.py:131-182` | `get_projection_stats`, `steering_clues` | D adds siblings |
| `phase1_linear_probing/AGENT_EXPLORATION.md` | Agent-facing protocol | D rewrites |
| `phase1_linear_probing/data/runs/curated4-8b-v002/` | Existing artifact dir | A writes new files in |

## Verification summary (post-implementation, pre-Lambda)

- `uv run pytest phase1_linear_probing/tests/ -v` → all green, including the four new test files.
- Manual reading of each modified file confirms cross-task contracts align (NPZ keys, JSON keys, server names, docs).
- No edits to or imports from `linear_probe_new/` — phase1 owns its iid_mm code independently.
- No new dependencies added beyond what's already in the repo.
