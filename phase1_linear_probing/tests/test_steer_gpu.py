"""GPU integration tests for _generate_batch and steering generation.

Tests all three generation modes (unsteered, tracer-only, steered) with a
real nnsight model.  Verifies that:
- Single-sample generation produces coherent text
- Batch generation matches single-sample generation
- Steering with alpha!=0 changes the output
- steer_and_generate delegates correctly to _generate_batch

Run with:  uv run pytest phase1_linear_probing/tests/test_steer_gpu.py -m gpu -v
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

from data import ProbeConfig, _load_nn_model, _cleanup_nn_model, build_formatted_prompt
from steer import _generate_batch, steer_and_generate

pytestmark = pytest.mark.gpu

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 64  # short for speed


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model + tokenizer once for all tests in this module."""
    cfg = ProbeConfig(
        model_name=MODEL_NAME,
        label_mode="binary",
        token_positions=["last_prompt"],
        cv_mode="grouped",
    )
    model_nn, n_layers = _load_nn_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    yield model_nn, tokenizer, n_layers
    _cleanup_nn_model(model_nn, cfg.device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def test_prompts(model_and_tokenizer):
    """Build a small set of formatted prompts."""
    _, tokenizer, _ = model_and_tokenizer
    pairs = [
        ("Always respond in formal English.", "What is gravity?"),
        ("Use bullet points for everything.", "Explain photosynthesis."),
        ("Write in first person only.", "Describe the water cycle."),
        ("Respond in exactly one sentence.", "What causes earthquakes?"),
    ]
    return [build_formatted_prompt(tokenizer, s, u) for s, u in pairs]


@pytest.fixture
def steering_direction(model_and_tokenizer):
    """Random unit-norm direction matching model's d_model."""
    _, tokenizer, n_layers = model_and_tokenizer
    # Infer d_model from tokenizer's vocab embedding
    # Use a fixed small dimension for the direction — we'll get it from model
    rng = np.random.default_rng(42)
    # d_model for Llama-3.1-8B is 4096
    d_model = 4096
    vec = rng.standard_normal(d_model).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


# ── Single-sample generation ────────────────────────────────────────────────


class TestGenerateBatchSingle:
    """Test _generate_batch with a single prompt (batch_size=1)."""

    def test_unsteered_returns_nonempty_string(
        self, model_and_tokenizer, test_prompts,
    ):
        model, tokenizer, _ = model_and_tokenizer
        responses = _generate_batch(
            model, tokenizer, [test_prompts[0]],
            use_tracer=False, max_new_tokens=MAX_NEW_TOKENS,
        )
        assert len(responses) == 1
        assert isinstance(responses[0], str)
        assert len(responses[0].strip()) > 0

    def test_tracer_only_returns_nonempty_string(
        self, model_and_tokenizer, test_prompts,
    ):
        model, tokenizer, _ = model_and_tokenizer
        responses = _generate_batch(
            model, tokenizer, [test_prompts[0]],
            use_tracer=True, alpha=0.0, max_new_tokens=MAX_NEW_TOKENS,
        )
        assert len(responses) == 1
        assert len(responses[0].strip()) > 0

    def test_steered_returns_nonempty_string(
        self, model_and_tokenizer, test_prompts, steering_direction,
    ):
        model, tokenizer, _ = model_and_tokenizer
        responses = _generate_batch(
            model, tokenizer, [test_prompts[0]],
            use_tracer=True,
            direction_vector=steering_direction,
            layer=15, alpha=3.0,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        assert len(responses) == 1
        assert len(responses[0].strip()) > 0


# ── Batch vs single equivalence ──────────────────────────────────────────────


class TestBatchEquivalence:
    """Batch generation should match single-sample generation."""

    def test_unsteered_batch_matches_single(
        self, model_and_tokenizer, test_prompts,
    ):
        model, tokenizer, _ = model_and_tokenizer
        # Single
        singles = []
        for p in test_prompts:
            r = _generate_batch(
                model, tokenizer, [p],
                use_tracer=False, max_new_tokens=MAX_NEW_TOKENS,
            )
            singles.append(r[0])

        # Batch
        batched = _generate_batch(
            model, tokenizer, test_prompts,
            use_tracer=False, max_new_tokens=MAX_NEW_TOKENS,
        )

        assert len(batched) == len(singles)
        for i, (s, b) in enumerate(zip(singles, batched)):
            # Left-padding can cause minor decode differences at boundaries;
            # check prefix match (first 80% of shorter response)
            check_len = int(min(len(s), len(b)) * 0.8)
            if check_len > 10:
                assert s[:check_len] == b[:check_len], (
                    f"Sample {i} diverges: single={s[:100]!r} batch={b[:100]!r}"
                )

    def test_tracer_batch_matches_single(
        self, model_and_tokenizer, test_prompts,
    ):
        model, tokenizer, _ = model_and_tokenizer
        singles = []
        for p in test_prompts:
            r = _generate_batch(
                model, tokenizer, [p],
                use_tracer=True, alpha=0.0, max_new_tokens=MAX_NEW_TOKENS,
            )
            singles.append(r[0])

        batched = _generate_batch(
            model, tokenizer, test_prompts,
            use_tracer=True, alpha=0.0, max_new_tokens=MAX_NEW_TOKENS,
        )

        assert len(batched) == len(singles)
        for i, (s, b) in enumerate(zip(singles, batched)):
            check_len = int(min(len(s), len(b)) * 0.8)
            if check_len > 10:
                assert s[:check_len] == b[:check_len], (
                    f"Sample {i} diverges: single={s[:100]!r} batch={b[:100]!r}"
                )


# ── Steering effect ──────────────────────────────────────────────────────────


class TestSteeringEffect:
    """Steering with nonzero alpha should change the output."""

    def test_steered_differs_from_unsteered(
        self, model_and_tokenizer, test_prompts, steering_direction,
    ):
        model, tokenizer, _ = model_and_tokenizer
        prompt = test_prompts[0]

        unsteered = _generate_batch(
            model, tokenizer, [prompt],
            use_tracer=True, alpha=0.0, max_new_tokens=MAX_NEW_TOKENS,
        )[0]

        steered = _generate_batch(
            model, tokenizer, [prompt],
            use_tracer=True,
            direction_vector=steering_direction,
            layer=15, alpha=10.0,
            max_new_tokens=MAX_NEW_TOKENS,
        )[0]

        # With alpha=10 on a random direction, output should differ
        assert unsteered != steered, (
            "Steered output (alpha=10) identical to unsteered — steering may not be working"
        )


# ── steer_and_generate backward compat ───────────────────────────────────────


class TestSteerAndGenerateCompat:
    """steer_and_generate should produce same result as _generate_batch."""

    def test_matches_generate_batch(
        self, model_and_tokenizer, test_prompts, steering_direction,
    ):
        model, tokenizer, _ = model_and_tokenizer
        prompt = test_prompts[0]

        via_sag = steer_and_generate(
            model, tokenizer, prompt,
            steering_direction, layer=15, alpha=3.0,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        via_batch = _generate_batch(
            model, tokenizer, [prompt],
            use_tracer=True,
            direction_vector=steering_direction,
            layer=15, alpha=3.0,
            max_new_tokens=MAX_NEW_TOKENS,
        )[0]

        assert via_sag == via_batch
