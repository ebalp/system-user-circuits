"""GPU smoke tests for activation steering.

Minimal tests to verify steer_and_generate and run_steering_sweep
work end-to-end without syntax/API errors. Short generations (8 tokens).

Run with:  uv run pytest phase1_linear_probing/tests/test_steering_gpu.py -m gpu -v -s
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import (
    ProbeConfig,
    _load_nn_model,
    _cleanup_nn_model,
    build_formatted_prompt,
    load_results as load_data_results,
    load_sync_env,
    prepare_condition_c,
)
from steer import (
    compute_steered_scr,
    load_steering_directions,
    run_steering_sweep,
    steer_and_generate,
)

pytestmark = pytest.mark.gpu

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 24
TOKENS = 8  # minimal generation for speed


@pytest.fixture(scope="module")
def setup():
    """Load model, tokenizer, directions, and 1 sample — once for all tests."""
    cfg = ProbeConfig(model_name=MODEL_NAME, run_id="curated24-8b-v002")
    load_sync_env(cfg.repo_root)

    model, n_layers = _load_nn_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    directions = load_steering_directions(cfg.run_dir, "last_prompt", LAYER)

    # 1 real Condition C sample for sweep tests
    df_all = load_data_results(cfg.data_dir, cfg.model_name)
    df_c = prepare_condition_c(df_all, "binary", conflict_ids=cfg.conflict_ids)
    sample = df_c.head(1).reset_index(drop=True)

    prompt = build_formatted_prompt(
        tokenizer, sample.iloc[0]["system_prompt"], sample.iloc[0]["user_prompt"],
    )

    yield model, tokenizer, directions, sample, prompt, cfg

    _cleanup_nn_model(model, cfg.device)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_generate_alpha_zero(setup):
    """alpha=0: generates non-empty text."""
    model, tokenizer, dirs, _, prompt, _ = setup
    r = steer_and_generate(model, tokenizer, prompt, dirs["probe"], LAYER, 0.0, max_new_tokens=TOKENS)
    assert len(r) > 0
    print(f"  alpha=0: {r}")


def test_generate_with_steering(setup):
    """alpha=5: generates text and differs from alpha=0."""
    model, tokenizer, dirs, _, prompt, _ = setup
    r0 = steer_and_generate(model, tokenizer, prompt, dirs["probe"], LAYER, 0.0, max_new_tokens=TOKENS)
    r5 = steer_and_generate(model, tokenizer, prompt, dirs["probe"], LAYER, 5.0, max_new_tokens=TOKENS)
    assert len(r5) > 0
    assert r0 != r5, "Steering had no effect"
    print(f"  alpha=0: {r0}\n  alpha=5: {r5}")


def test_generate_cmd_direction(setup):
    """CMD direction generates text."""
    model, tokenizer, dirs, _, prompt, _ = setup
    if "cmd_overall" not in dirs:
        pytest.skip("No CMD")
    r = steer_and_generate(model, tokenizer, prompt, dirs["cmd_overall"], LAYER, 3.0, max_new_tokens=TOKENS)
    assert len(r) > 0
    print(f"  CMD alpha=3: {r}")


def test_probe_sweep(setup):
    """Probe sweep: runs, scores, computes SCR."""
    model, tokenizer, dirs, sample, _, _ = setup
    result = run_steering_sweep(
        model, tokenizer, sample, dirs["probe"], LAYER,
        alphas=[0.0, 3.0], max_new_tokens=TOKENS,
    )
    assert len(result) == 2
    assert {"label", "response", "sys_ok"}.issubset(result.columns)
    scr = compute_steered_scr(result)
    print(f"  Probe SCR: {scr.to_dict()}")


def test_cmd_sweep(setup):
    """CMD sweep: runs, scores, computes SCR."""
    model, tokenizer, dirs, sample, _, _ = setup
    if "cmd_overall" not in dirs:
        pytest.skip("No CMD")
    result = run_steering_sweep(
        model, tokenizer, sample, dirs["cmd_overall"], LAYER,
        alphas=[0.0, 3.0], max_new_tokens=TOKENS,
    )
    assert len(result) == 2
    scr = compute_steered_scr(result)
    print(f"  CMD SCR: {scr.to_dict()}")


def test_alpha_zero_baseline_identical(setup):
    """alpha=0 with probe vs CMD gives same response (no intervention)."""
    model, tokenizer, dirs, _, prompt, _ = setup
    if "cmd_overall" not in dirs:
        pytest.skip("No CMD")
    r_probe = steer_and_generate(model, tokenizer, prompt, dirs["probe"], LAYER, 0.0, max_new_tokens=TOKENS)
    r_cmd = steer_and_generate(model, tokenizer, prompt, dirs["cmd_overall"], LAYER, 0.0, max_new_tokens=TOKENS)
    assert r_probe == r_cmd
