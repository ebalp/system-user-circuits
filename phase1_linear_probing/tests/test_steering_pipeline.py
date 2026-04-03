"""End-to-end pipeline test for steering experiment.

Mirrors the notebook flow with real model, real data, real verifiers —
but at small scale (2 conflicts, max_new_tokens=64) for speed.

Run with:  uv run python phase1_linear_probing/tests/test_steering_pipeline.py
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

# Ensure imports work
_phase1_dir = Path(__file__).resolve().parent.parent
_repo_root = _phase1_dir.parent
sys.path.insert(0, str(_phase1_dir))
sys.path.insert(0, str(_repo_root))

from data import (
    ProbeConfig,
    _cleanup_nn_model,
    _load_nn_model,
    build_formatted_prompt,
    load_results,
    load_sync_env,
    prepare_condition_c,
)
from probe import load_results as load_probe_results, results_path
from steer import (
    _generate_batch,
    load_steering_directions,
    run_condition_comparison,
    save_experiment_manifest,
    steer_and_generate,
)

# ── Config ────────────────────────────────────────────────────────────────────
# Use only 2 conflicts, 8 samples each, short generation for speed
CONFLICTS = ["json_only_vs_plain", "past_vs_present_tense"]
N_PER_CONFLICT = 8  # subsample for fast testing
MAX_NEW_TOKENS = 32
BATCH_SIZE = 4
ALPHAS = [3.0]
SEED = 42


def main():
    print("=" * 70)
    print("STEERING PIPELINE INTEGRATION TEST")
    print("=" * 70)

    # ── Setup ─────────────────────────────────────────────────────────────
    cfg = ProbeConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        label_mode="binary",
        token_positions=["last_prompt"],
        cv_mode="grouped",
        n_cv_folds=4,
        use_scaler=False,
        probe_C=1.0,
        batch_size=1,
        conflict_ids=CONFLICTS,
        run_id="curated4-8b-v001",
    )
    load_sync_env(cfg.repo_root)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    df_all = load_results(cfg.data_dir, cfg.model_name)
    df_c = prepare_condition_c(df_all, cfg.label_mode, conflict_ids=CONFLICTS)
    df_c = (
        df_c.groupby("conflict_id", group_keys=False)
        .sample(n=N_PER_CONFLICT, random_state=SEED)
        .sort_values("conflict_id")
        .reset_index(drop=True)
    )
    print(f"    Condition C: {len(df_c)} samples, "
          f"{df_c['conflict_id'].nunique()} conflicts")
    print(f"    SCR: {(df_c['label'] == 'followed_system').mean():.3f}")

    # ── Load probe results ────────────────────────────────────────────────
    print("\n[2] Loading probe results...")
    pos = cfg.token_positions[0]
    rpath = results_path(cfg.run_dir, cfg.cv_mode, cfg.use_scaler)
    results = load_probe_results(rpath)
    pr = results[pos]
    best_layer = int(
        pr.cv_scores.loc[pr.cv_scores["roc_auc_mean"].idxmax(), "layer"]
    )
    print(f"    Best layer: {best_layer}")

    # ── Load directions ───────────────────────────────────────────────────
    print("\n[3] Loading steering directions...")
    directions = load_steering_directions(cfg.run_dir, pos, best_layer)
    steer_directions = {"probe": directions["probe"]}
    # Only probe direction for speed in integration test
    print(f"    Directions: {list(steer_directions.keys())}")

    # ── Load model ────────────────────────────────────────────────────────
    print("\n[4] Loading model...")
    model_nn, n_layers = _load_nn_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── Use a temp directory for caching ──────────────────────────────────
    tmpdir = tempfile.mkdtemp(prefix="steer_pipeline_test_")
    steer_dir = Path(tmpdir) / "steering"
    print(f"\n[5] Output dir: {steer_dir}")

    try:
        # ── Run 1: generate everything ────────────────────────────────────
        print("\n[6] Run 1: run_condition_comparison (all fresh)...")
        run_results = run_condition_comparison(
            model_nn, tokenizer, df_c,
            steering_directions=steer_directions,
            layer=best_layer,
            alphas=ALPHAS,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=BATCH_SIZE,
            output_dir=steer_dir,
            seed=SEED,
        )

        # Validate results structure
        expected_keys = {"phase0", "unsteered", "tracer_only"}
        for d in steer_directions:
            expected_keys.add(f"steered_{d}")
        assert set(run_results.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(run_results.keys())}"
        )
        print(f"    Result keys: {list(run_results.keys())}")

        for key, df in run_results.items():
            print(f"    {key}: {len(df)} rows")
            assert len(df) > 0, f"{key} has no rows"

        # Check baselines have same row count
        n_base = len(run_results["phase0"])
        for name in ["unsteered", "tracer_only"]:
            assert len(run_results[name]) == n_base, (
                f"{name} has {len(run_results[name])} rows, expected {n_base}"
            )

        # Check steered has n_base * len(ALPHAS) rows per direction
        for d in steer_directions:
            key = f"steered_{d}"
            expected = n_base * len(ALPHAS)
            assert len(run_results[key]) == expected, (
                f"{key} has {len(run_results[key])} rows, expected {expected}"
            )

        # Divergence analysis
        print("\n[7] Divergence analysis:")
        for name in ["phase0", "unsteered", "tracer_only"]:
            scr = (run_results[name]["label"] == "followed_system").mean()
            print(f"    {name:15s} SCR: {scr:.3f}")

        # Check JSONL files exist
        print("\n[8] Checking cache files...")
        for cond in ["phase0", "unsteered", "tracer_only"]:
            for cid in CONFLICTS:
                fpath = steer_dir / cond / f"{cid}.jsonl"
                assert fpath.exists(), f"Missing: {fpath}"
        for d in steer_directions:
            for alpha in ALPHAS:
                for cid in CONFLICTS:
                    fpath = steer_dir / "steered" / d / f"{cid}_alpha_{alpha}.jsonl"
                    assert fpath.exists(), f"Missing: {fpath}"
        print("    All expected JSONL files present")

        # Save manifest
        manifest_path = save_experiment_manifest(
            steer_dir, seed=SEED, model_name=cfg.model_name,
            conflict_ids=CONFLICTS,
            direction_names=list(steer_directions.keys()),
            alphas=ALPHAS, max_new_tokens=MAX_NEW_TOKENS,
            batch_size=BATCH_SIZE, layer=best_layer,
        )
        manifest = json.loads(manifest_path.read_text())
        print(f"\n[9] Manifest: {len(manifest['files'])} files listed")
        assert manifest["seed"] == SEED
        assert manifest["layer"] == best_layer

        # ── Run 2: verify caching (0 pending) ────────────────────────────
        print("\n[10] Run 2: re-run (everything should be cached)...")
        run_results_2 = run_condition_comparison(
            model_nn, tokenizer, df_c,
            steering_directions=steer_directions,
            layer=best_layer,
            alphas=ALPHAS,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=BATCH_SIZE,
            output_dir=steer_dir,
            seed=SEED,
        )

        # Verify identical results
        for key in run_results:
            assert len(run_results_2[key]) == len(run_results[key]), (
                f"{key}: run2 has {len(run_results_2[key])} rows, "
                f"run1 had {len(run_results[key])}"
            )
        print("    Cached run produced identical row counts")

        print("\n" + "=" * 70)
        print("ALL CHECKS PASSED")
        print("=" * 70)

    finally:
        # Cleanup
        _cleanup_nn_model(model_nn, cfg.device)
        del model_nn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"\nCleaned up: {tmpdir}")


if __name__ == "__main__":
    main()
