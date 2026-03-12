"""GPU equivalence tests for batched activation extraction.

Verifies that batched extraction produces identical results to single-sample
extraction for both TransformerLens and nnsight backends.

- **TL** uses right-padding; different CUDA kernel paths in fp16 cause small
  numerical differences (~0.03 abs).  Tested via cosine similarity > 0.9999.
- **nnsight** uses multi-invoke with internal left-padding; position indices
  are shifted by pad_offset so results should be near-exact.  Tested via
  cosine > 0.9999 (fp16 accumulation differences possible).

Run with:  uv run pytest phase1_linear_probing/tests/test_extraction_gpu.py -m gpu -v
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import (
    ProbeConfig,
    _pad_right,
    extract_activations_nn,
    extract_activations_tl,
    find_token_positions,
)

pytestmark = pytest.mark.gpu

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
N_SAMPLES = 8  # small for speed; enough to fill a batch


def _make_test_inputs(n: int = N_SAMPLES):
    """Build a small set of tokenized prompts with varying lengths."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompts = [
        ("Always respond in formal English.", "What is gravity?"),
        ("Use bullet points for everything.", "Explain photosynthesis."),
        ("Write in first person only.", "Describe the water cycle in detail."),
        ("Be concise. One sentence max.", "What is DNA?"),
        ("Use casual, friendly language.", "Tell me about black holes and how they form."),
        ("Respond in ALL CAPS.", "What causes earthquakes?"),
        ("Use numbered lists.", "Explain how vaccines work and why they matter."),
        ("Write like a pirate.", "What is machine learning?"),
    ][:n]

    input_ids_list = []
    position_maps = []
    for sys_text, usr_text in prompts:
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": usr_text},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        pm = find_token_positions(tokenizer, sys_text, usr_text)
        input_ids_list.append(ids)
        position_maps.append(pm)

    return input_ids_list, position_maps


def _make_cfg(batch_size: int, run_id: str) -> ProbeConfig:
    return ProbeConfig(
        model_name=MODEL_NAME,
        batch_size=batch_size,
        token_positions=["last_prompt"],
        run_id=run_id,
    )


def _compare_activations_cosine(act_a: dict, act_b: dict, min_cos: float = 0.9999):
    """Assert two activation dicts have high cosine similarity.

    fp16 batched vs unbatched forward passes use different CUDA kernel paths,
    so absolute differences up to ~0.03 are expected.  Cosine similarity is
    the right metric for high-dimensional activation vectors.
    """
    for pos in act_a:
        assert pos in act_b, f"Missing position {pos}"
        a = act_a[pos]
        b = act_b[pos]
        assert a.shape == b.shape, f"{pos}: shape mismatch {a.shape} vs {b.shape}"
        n_samples, n_layers, _ = a.shape
        for layer in range(n_layers):
            for i in range(n_samples):
                cos = cosine_similarity(
                    a[i, layer].reshape(1, -1).astype(np.float64),
                    b[i, layer].reshape(1, -1).astype(np.float64),
                )[0, 0]
                assert cos > min_cos, (
                    f"pos={pos} layer={layer} sample={i}: "
                    f"cosine={cos:.6f} (expected > {min_cos})"
                )


class TestPadRight:
    """Unit tests for _pad_right (no GPU needed, but grouped here)."""

    def test_basic_padding(self):
        ids = [
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4, 5]]),
            torch.tensor([[6, 7, 8, 9]]),
        ]
        batched, lengths = _pad_right(ids, pad_token_id=0)
        assert batched.shape == (3, 4)
        assert lengths == [3, 2, 4]
        assert batched[0].tolist() == [1, 2, 3, 0]
        assert batched[1].tolist() == [4, 5, 0, 0]
        assert batched[2].tolist() == [6, 7, 8, 9]

    def test_same_length_no_padding(self):
        ids = [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])]
        batched, lengths = _pad_right(ids, pad_token_id=0)
        assert batched.shape == (2, 2)
        assert lengths == [2, 2]


class TestNNsightBatchEquivalence:
    """nnsight: batch_size=1 vs batch_size=N produce near-identical activations.

    nnsight multi-invoke internally left-pads; we compensate with pad_offset.
    fp16 accumulation differences from different batch compositions mean
    results are near-identical but not bit-exact.
    """

    def test_nn_batch1_vs_batch8(self):
        input_ids_list, position_maps = _make_test_inputs()

        cfg_b1 = _make_cfg(batch_size=1, run_id="test-nn-b1")
        act_b1 = extract_activations_nn(input_ids_list, position_maps, cfg_b1)

        gc.collect()
        torch.cuda.empty_cache()

        cfg_b8 = _make_cfg(batch_size=8, run_id="test-nn-b8")
        act_b8 = extract_activations_nn(input_ids_list, position_maps, cfg_b8)

        _compare_activations_cosine(act_b1, act_b8, min_cos=0.9999)

    def test_nn_batch1_vs_batch4(self):
        input_ids_list, position_maps = _make_test_inputs()

        cfg_b1 = _make_cfg(batch_size=1, run_id="test-nn-b1-v2")
        act_b1 = extract_activations_nn(input_ids_list, position_maps, cfg_b1)

        gc.collect()
        torch.cuda.empty_cache()

        cfg_b4 = _make_cfg(batch_size=4, run_id="test-nn-b4")
        act_b4 = extract_activations_nn(input_ids_list, position_maps, cfg_b4)

        _compare_activations_cosine(act_b1, act_b4, min_cos=0.9999)


class TestTransformerLensBatchEquivalence:
    """TransformerLens: batch_size=1 vs batch_size=N produce near-identical activations.

    TL uses right-padded batching with different CUDA kernel paths for
    batched vs single-sample matmuls in fp16, so small numerical differences
    (~0.03 abs) are expected. Cosine similarity > 0.9999 confirms equivalence.
    """

    def test_tl_batch1_vs_batch8(self):
        input_ids_list, position_maps = _make_test_inputs()

        cfg_b1 = _make_cfg(batch_size=1, run_id="test-tl-b1")
        act_b1 = extract_activations_tl(input_ids_list, position_maps, cfg_b1)

        gc.collect()
        torch.cuda.empty_cache()

        cfg_b8 = _make_cfg(batch_size=8, run_id="test-tl-b8")
        act_b8 = extract_activations_tl(input_ids_list, position_maps, cfg_b8)

        _compare_activations_cosine(act_b1, act_b8, min_cos=0.9999)

    def test_tl_batch1_vs_batch4(self):
        input_ids_list, position_maps = _make_test_inputs()

        cfg_b1 = _make_cfg(batch_size=1, run_id="test-tl-b1-v2")
        act_b1 = extract_activations_tl(input_ids_list, position_maps, cfg_b1)

        gc.collect()
        torch.cuda.empty_cache()

        cfg_b4 = _make_cfg(batch_size=4, run_id="test-tl-b4")
        act_b4 = extract_activations_tl(input_ids_list, position_maps, cfg_b4)

        _compare_activations_cosine(act_b1, act_b4, min_cos=0.9999)


class TestCrossBackendEquivalence:
    """TL vs nnsight produce consistent activations (may not be bit-identical
    due to implementation differences, but should be very close)."""

    def test_tl_vs_nn_batch1(self):
        input_ids_list, position_maps = _make_test_inputs(4)

        cfg_tl = _make_cfg(batch_size=1, run_id="test-cross-tl")
        act_tl = extract_activations_tl(input_ids_list, position_maps, cfg_tl)

        gc.collect()
        torch.cuda.empty_cache()

        cfg_nn = _make_cfg(batch_size=1, run_id="test-cross-nn")
        act_nn = extract_activations_nn(input_ids_list, position_maps, cfg_nn)

        _compare_activations_cosine(act_tl, act_nn, min_cos=0.99)
