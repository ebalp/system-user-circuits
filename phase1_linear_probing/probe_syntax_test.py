#!/usr/bin/env python3
"""
probe_syntax_test.py – validate nnsight 0.5.x activation-extraction patterns.

Tests 1-6: GPT-2 (always run)
Tests 7-8: Llama-3.1-8B-Instruct (run with --llama)

Tests 2 and 8 are EXPECTED FAILURES that document what NOT to do and why.
Each prints an explanation of the underlying cause and the takeaway.

Key findings:
  1. InterleavingTracer.compile() only creates an Invoker when self.args is truthy.
     · model.trace("text")          → args=("text",)  → Invoker created  ✓
     · model.trace(input_ids=ids)   → args=()          → NO Invoker       ✗
     Fix: pass input positionally, or use tracer.invoke(input_ids=ids).

  2. .save() returns torch.Tensor directly — no .value wrapper in 0.5.x.

  3. Single invoke squeezes batch dim: output[0] shape = (seq_len, d_model).

  4. nnterp.StandardizedTransformer fails on Llama due to FakeTensor
     incompatibility in check_model_renaming. Use LanguageModel directly.

Run:
  uv run python phase1_linear_probing/probe_syntax_test.py
  uv run python phase1_linear_probing/probe_syntax_test.py --llama

Full notes: phase1_linear_probing/NNSIGHT_NOTES.md
"""
import argparse, gc, sys, traceback
import torch
import nnsight
from nnsight import LanguageModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


def ok(msg):
    print(f"  [ok]  {msg}")


def fail(msg, exc):
    print(f"  [FAIL]  {msg}")
    traceback.print_exception(type(exc), exc, exc.__traceback__)


def cleanup():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ────────────────────────────────────────────────────────────
#  Test 1: string prompt (positional arg → invoker created)
# ────────────────────────────────────────────────────────────
def test_string_prompt():
    section("Test 1 · string prompt (positional arg)")
    model = LanguageModel("gpt2", device_map="auto", dispatch=True)
    try:
        with model.trace("The quick brown fox"):
            h0 = model.transformer.h[0].output[0].save()
            h5 = model.transformer.h[5].output[0].save()
            h11 = model.transformer.h[11].output[0].save()

        ok(f"layer 0  shape={tuple(h0.shape)}")
        ok(f"layer 5  shape={tuple(h5.shape)}")
        ok(f"layer 11 shape={tuple(h11.shape)}")
        return True
    except Exception as e:
        fail("string prompt trace", e)
        return False
    finally:
        del model; cleanup()


# ────────────────────────────────────────────────────────────
#  Test 2: input_ids as KWARG (EXPECTED FAILURE – no invoker)
# ────────────────────────────────────────────────────────────
def test_ids_kwarg():
    section("Test 2 · input_ids=tensor as kwarg (EXPECTED FAILURE)")
    print("  model.trace(input_ids=ids) passes input as kwargs only.")
    print("  InterleavingTracer.compile() creates an Invoker only when self.args")
    print("  is truthy. With kwargs-only, args=() → no Invoker → body runs")
    print("  outside the forward pass → .output raises ValueError.")
    print()
    print("  Takeaway: always pass input as a positional arg to model.trace(),")
    print("  or use model.trace() + tracer.invoke(input_ids=ids).")
    print()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok("The quick brown fox", return_tensors="pt").input_ids.to(DEVICE)

    model = LanguageModel("gpt2", device_map="auto", dispatch=True)
    try:
        with model.trace(input_ids=ids):
            h0 = model.transformer.h[0].output[0].save()
        ok(f"unexpectedly succeeded: {tuple(h0.shape)}")
        return True
    except Exception as e:
        ok(f"failed as expected: {type(e).__name__}")
        return True  # expected failure — counts as PASS
    finally:
        del model; cleanup()


# ────────────────────────────────────────────────────────────
#  Test 3: input_ids tensor as POSITIONAL arg
# ────────────────────────────────────────────────────────────
def test_ids_positional():
    section("Test 3 · input_ids tensor as positional arg")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok("The quick brown fox", return_tensors="pt").input_ids.to(DEVICE)

    model = LanguageModel("gpt2", device_map="auto", dispatch=True)
    try:
        with model.trace(ids):
            h0 = model.transformer.h[0].output[0].save()
            h11 = model.transformer.h[11].output[0].save()

        ok(f"layer 0  shape={tuple(h0.shape)}")
        ok(f"layer 11 shape={tuple(h11.shape)}")
        return True
    except Exception as e:
        fail("ids positional trace", e)
        return False
    finally:
        del model; cleanup()


# ────────────────────────────────────────────────────────────
#  Test 4: explicit invoker pattern with input_ids kwarg
#   (use tracer.invoke() to force an Invoker regardless)
# ────────────────────────────────────────────────────────────
def test_explicit_invoke():
    section("Test 4 · explicit tracer.invoke(input_ids=ids)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok("The quick brown fox", return_tensors="pt").input_ids.to(DEVICE)

    model = LanguageModel("gpt2", device_map="auto", dispatch=True)
    try:
        with model.trace() as tracer:
            with tracer.invoke(input_ids=ids):
                h0 = model.transformer.h[0].output[0].save()
                h11 = model.transformer.h[11].output[0].save()

        ok(f"layer 0  shape={tuple(h0.shape)}")
        ok(f"layer 11 shape={tuple(h11.shape)}")
        return True
    except Exception as e:
        fail("explicit invoke trace", e)
        return False
    finally:
        del model; cleanup()


# ────────────────────────────────────────────────────────────
#  Test 5: collect all 12 layers in a loop
# ────────────────────────────────────────────────────────────
def test_all_layers_loop():
    section("Test 5 · all 12 layers via loop (string prompt)")
    model = LanguageModel("gpt2", device_map="auto", dispatch=True)
    n_layers = 12

    try:
        # Save into a pre-allocated list so each .save() object is accessible
        saved = [None] * n_layers
        with model.trace("The quick brown fox"):
            for i in range(n_layers):
                saved[i] = model.transformer.h[i].output[0].save()

        for i in [0, 5, 11]:
            ok(f"layer {i:2d} shape={tuple(saved[i].shape)}")
        return True
    except Exception as e:
        fail("all-layers loop", e)
        return False
    finally:
        del model; cleanup()


# ────────────────────────────────────────────────────────────
#  Test 6: explicit invoke + all layers (for pre-tokenized ids)
# ────────────────────────────────────────────────────────────
def test_invoke_all_layers():
    section("Test 6 · explicit invoke + all layers (pre-tokenized)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok("The quick brown fox", return_tensors="pt").input_ids.to(DEVICE)

    model = LanguageModel("gpt2", device_map="auto", dispatch=True)
    n_layers = 12

    try:
        saved = [None] * n_layers
        with model.trace() as tracer:
            with tracer.invoke(input_ids=ids):
                for i in range(n_layers):
                    saved[i] = model.transformer.h[i].output[0].save()

        for i in [0, 5, 11]:
            ok(f"layer {i:2d} shape={tuple(saved[i].shape)}")
        return True
    except Exception as e:
        fail("invoke all layers", e)
        return False
    finally:
        del model; cleanup()


# ────────────────────────────────────────────────────────────
#  Test 7 (optional): Llama-3.1-8B-Instruct
# ────────────────────────────────────────────────────────────
def test_llama():
    section("Test 7 · Llama-3.1-8B-Instruct · explicit invoke")
    import os, re
    from pathlib import Path
    from transformers import AutoTokenizer

    # Load HF_TOKEN
    for env_file in sorted(Path(__file__).parent.parent.glob("*.sync.env")):
        for line in env_file.read_text().splitlines():
            m = re.match(r'^export\s+(\w+)=(.*)', line.strip())
            if m:
                os.environ.setdefault(m.group(1), m.group(2).strip("'\""))
        break

    MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL)
    messages = [
        {"role": "system", "content": "Reply in English."},
        {"role": "user",   "content": "Say hello."},
    ]
    prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    ok(f"prompt tokens: {ids.shape[1]}")

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = LanguageModel(MODEL, device_map="auto", dispatch=True, torch_dtype=dtype)
    # Llama: model.model.layers[i] → LlamaDecoderLayer → output[0] = hidden_states
    n_layers = len(list(model.model.layers))
    ok(f"loaded | n_layers={n_layers}")

    try:
        saved = [None] * n_layers
        with model.trace() as tracer:
            with tracer.invoke(input_ids=ids):
                for i in range(n_layers):
                    saved[i] = model.model.layers[i].output[0].save()

        for i in [0, n_layers // 2, n_layers - 1]:
            ok(f"layer {i:2d} shape={tuple(saved[i].shape)}")
        return True
    except Exception as e:
        fail("Llama trace", e)
        return False
    finally:
        del model; cleanup()


# ────────────────────────────────────────────────────────────
#  Test 8 (optional): nnterp StandardizedTransformer + Llama
# ────────────────────────────────────────────────────────────
def test_nnterp_llama():
    section("Test 8 · nnterp StandardizedTransformer · Llama (EXPECTED FAILURE)")
    print("  nnterp.StandardizedTransformer wraps nnsight.LanguageModel with")
    print("  architecture-agnostic module renaming (model.layers_output[i],")
    print("  model.attentions_output[i], etc.).")
    print()
    print("  However, its __init__ calls check_model_renaming(), which runs a")
    print("  scan-phase trace to validate the renaming. On nnsight 0.5.x + Llama,")
    print("  this hits a FakeTensor incompatibility:")
    print("    TypeError: FakeTensor.__new__() got unexpected kwarg 'requires_grad'")
    print()
    print("  Takeaway: use nnsight.LanguageModel directly (test 7) and access")
    print("  Llama layers via model.model.layers[i].output[0] instead of the")
    print("  nnterp convenience accessors.")
    print()

    import os, re
    from pathlib import Path
    from transformers import AutoTokenizer
    from nnterp import StandardizedTransformer

    for env_file in sorted(Path(__file__).parent.parent.glob("*.sync.env")):
        for line in env_file.read_text().splitlines():
            m = re.match(r'^export\s+(\w+)=(.*)', line.strip())
            if m:
                os.environ.setdefault(m.group(1), m.group(2).strip("'\""))
        break

    MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL)
    messages = [
        {"role": "system", "content": "Reply in English."},
        {"role": "user",   "content": "Say hello."},
    ]
    prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    try:
        model = StandardizedTransformer(MODEL, dispatch=True, torch_dtype=dtype)
        ok(f"loaded (unexpected success) | n_layers={model.num_layers}")
    except Exception as e:
        ok(f"failed as expected during load: {type(e).__name__}: {e}")
        return True  # expected failure — counts as PASS

    try:
        saved = [None] * model.num_layers
        with model.trace() as tracer:
            with tracer.invoke(input_ids=ids):
                for i in range(model.num_layers):
                    saved[i] = model.layers_output[i].save()

        for i in [0, model.num_layers // 2, model.num_layers - 1]:
            ok(f"layer {i:2d} shape={tuple(saved[i].shape)}")
        return True
    except Exception as e:
        fail("nnterp trace", e)
        return False
    finally:
        del model; cleanup()


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama", action="store_true",
                        help="Also run Llama tests (tests 7 & 8)")
    args = parser.parse_args()

    print(f"nnsight {nnsight.__version__}  |  device={DEVICE}")

    results = {}
    results["1: string prompt"]                     = test_string_prompt()
    results["2: ids kwarg (expected ✗ → lesson)"]   = test_ids_kwarg()
    results["3: ids positional"]                     = test_ids_positional()
    results["4: explicit invoke"]                    = test_explicit_invoke()
    results["5: all layers (string)"]                = test_all_layers_loop()
    results["6: invoke + all layers"]                = test_invoke_all_layers()

    if args.llama:
        results["7: nnsight Llama"]                  = test_llama()
        results["8: nnterp Llama (expected ✗ → lesson)"] = test_nnterp_llama()

    section("Summary")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {name}")
        if not passed:
            all_pass = False

    section("Lessons Learned")
    print("  1. Always pass input as a positional arg or use tracer.invoke().")
    print("     model.trace(input_ids=ids) silently fails (no Invoker created).")
    print()
    print("  2. .save() returns torch.Tensor directly in nnsight 0.5.x.")
    print("     No .value wrapper — use tensor.shape, not tensor.value.shape.")
    print()
    print("  3. Single invoke squeezes batch dim: output[0] is (seq_len, d_model).")
    print()
    print("  4. nnterp.StandardizedTransformer fails on Llama (nnsight 0.5.x)")
    print("     due to FakeTensor incompatibility in check_model_renaming.")
    print("     Use nnsight.LanguageModel directly with model.model.layers[i].")
    print()
    print("  See: phase1_linear_probing/NNSIGHT_NOTES.md")

    sys.exit(0 if all_pass else 1)
