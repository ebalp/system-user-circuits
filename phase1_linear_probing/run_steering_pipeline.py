"""Full steering pipeline — equivalent to the notebook without visualization.

Runs all 4 conflicts, 3 alphas, all 2500 samples/conflict through
run_condition_comparison(). Results are cached to {run_dir}/steering/ as
per-conflict JSONL files.

After this completes, open steering_experiment.ipynb and run cell 8 onward —
everything will load from cache (0 pending).

Usage:  uv run python phase1_linear_probing/run_steering_pipeline.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

_phase1_dir = Path(__file__).resolve().parent
_repo_root = _phase1_dir.parent
sys.path.insert(0, str(_phase1_dir))
sys.path.insert(0, str(_repo_root))

from data import (
    ProbeConfig,
    _cleanup_nn_model,
    _load_nn_model,
    load_results,
    load_sync_env,
    prepare_condition_c,
)
from steer import (
    load_steering_directions,
    run_condition_comparison,
    save_experiment_manifest,
)

# ── Config ────────────────────────────────────────────────────────────────────
PROBE_CONFLICTS_4 = [
    "json_only_vs_plain",
    "list_bullets_vs_numbered",
    "past_vs_present_tense",
    "starting_word_hello_greetings",
]

BATCH_SIZE = 48
SEED = 42
ALPHAS = {
    "probe": [0.5, 1, 1.5],
    "cmd_overall": [2, 5, 10],
}
MAX_NEW_TOKENS = 512
STEER_LAYER = 25


def main():
    cfg = ProbeConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        label_mode="binary",
        token_positions=["last_prompt"],
        cv_mode="grouped",
        n_cv_folds=4,
        use_scaler=False,
        probe_C=1.0,
        batch_size=1,
        conflict_ids=PROBE_CONFLICTS_4,
        run_id="curated4-8b-v001",
    )
    load_sync_env(cfg.repo_root)
    cfg.ensure_dirs()

    print(f"Run ID       : {cfg.run_id}")
    print(f"Run dir      : {cfg.run_dir}")
    for dname, als in ALPHAS.items():
        print(f"Alphas ({dname:12s}): {als}")
    print(f"Batch size   : {BATCH_SIZE}")

    # ── Steering layer ─────────────────────────────────────────────────────
    pos = cfg.token_positions[0]
    steer_layer = STEER_LAYER
    print(f"Steer layer  : {steer_layer}")

    # ── Load steering directions ──────────────────────────────────────────
    directions = load_steering_directions(cfg.run_dir, pos, steer_layer)
    steer_directions = {}
    if "probe" in directions:
        steer_directions["probe"] = directions["probe"]
    if "cmd_overall" in directions:
        steer_directions["cmd_overall"] = directions["cmd_overall"]
    print(f"Directions   : {list(steer_directions.keys())}")

    # ── Load Condition C samples (all — no subsampling) ───────────────────
    df_all = load_results(cfg.data_dir, cfg.model_name)
    df_c = prepare_condition_c(
        df_all, cfg.label_mode, conflict_ids=cfg.conflict_ids
    )
    df_c = df_c.sort_values("conflict_id").reset_index(drop=True)
    print(
        f"Condition C  : {len(df_c)} samples across "
        f"{df_c['conflict_id'].nunique()} conflicts"
    )
    print(f"Phase 0 SCR  : {(df_c['label'] == 'followed_system').mean():.3f}")

    # ── Load model ────────────────────────────────────────────────────────
    model_nn, _n_layers = _load_nn_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── Run all conditions ────────────────────────────────────────────────
    steer_dir = cfg.run_dir / "steering"

    try:
        run_results = run_condition_comparison(
            model_nn,
            tokenizer,
            df_c,
            steering_directions=steer_directions,
            layer=steer_layer,
            alphas=ALPHAS,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=BATCH_SIZE,
            output_dir=steer_dir,
            seed=SEED,
        )

        # Save manifest
        save_experiment_manifest(
            steer_dir,
            seed=SEED,
            model_name=cfg.model_name,
            conflict_ids=list(cfg.conflict_ids),
            direction_names=list(steer_directions.keys()),
            alphas=ALPHAS,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=BATCH_SIZE,
            layer=steer_layer,
        )

        # ── Baseline analysis ─────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("BASELINE ANALYSIS")
        print("=" * 70)

        baselines = ["phase0", "unsteered", "unsteered_hooked"]
        for name in baselines:
            scr = (run_results[name]["label"] == "followed_system").mean()
            print(f"{name:20s} SCR: {scr:.3f}")

        # Pairwise agreement
        for a, b in [
            ("phase0", "unsteered"),
            ("unsteered", "unsteered_hooked"),
        ]:
            agree = (
                run_results[a]["label"].values
                == run_results[b]["label"].values
            ).mean()
            scr_a = (run_results[a]["label"] == "followed_system").mean()
            scr_b = (run_results[b]["label"] == "followed_system").mean()
            print(
                f"\n{a} → {b}:  "
                f"ΔSCR={scr_b - scr_a:+.3f}  agreement={agree:.1%}"
            )

        # Per-conflict breakdown
        print(
            f"\n{'conflict_id':40s}  {'phase0':>7s} / "
            f"{'unsteered':>9s} / {'hooked':>6s}"
        )
        print("-" * 70)
        for cid in sorted(df_c["conflict_id"].unique()):
            row = []
            for name in baselines:
                mask = run_results[name]["conflict_id"] == cid
                scr = (
                    run_results[name].loc[mask, "label"] == "followed_system"
                ).mean()
                row.append(f"{scr:.3f}")
            print(f"  {cid:40s}  {' / '.join(row)}")

        # ── Steered SCR ───────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("STEERED SCR BY ALPHA")
        print("=" * 70)

        for dir_name in steer_directions:
            key = f"steered_{dir_name}"
            if key not in run_results:
                continue
            df_s = run_results[key]
            print(f"\n{dir_name} direction:")
            scr_table = df_s.groupby("alpha").apply(
                lambda g: (g["label"] == "followed_system").mean()
            )
            print(scr_table.to_string())

        print("\n" + "=" * 70)
        print(f"DONE — results cached in {steer_dir}")
        print("Open steering_experiment.ipynb to visualize.")
        print("=" * 70)

    finally:
        _cleanup_nn_model(model_nn, cfg.device)
        del model_nn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
