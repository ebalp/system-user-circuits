# nnsight 0.5.x — API Notes

Verified against nnsight 0.5.15 + nnterp 1.2.2 on 2025-02-20.
Test script: `phase1_linear_probing/probe_syntax_test.py`

## Critical Rules

### 1. Input must be a positional arg (or use explicit `tracer.invoke()`)

`InterleavingTracer.compile()` only creates an `Invoker` when `self.args` is truthy.

```python
# WORKS — positional arg creates an Invoker
with model.trace("The quick brown fox"):
    h = model.transformer.h[0].output[0].save()

# WORKS — tensor as positional arg
with model.trace(ids):
    h = model.transformer.h[0].output[0].save()

# WORKS — explicit invoke with kwargs
with model.trace() as tracer:
    with tracer.invoke(input_ids=ids):
        h = model.transformer.h[0].output[0].save()

# FAILS — kwargs-only means args=(), no Invoker, body runs outside forward pass
with model.trace(input_ids=ids):
    h = model.transformer.h[0].output[0].save()  # ValueError: not interleaving
```

### 2. `.save()` returns `torch.Tensor` directly

In nnsight 0.5.x, `.save()` returns real tensors, not proxy objects. There is no `.value` attribute.

```python
with model.trace("text"):
    h = model.transformer.h[0].output[0].save()

# Correct:
print(h.shape)    # torch.Size([seq_len, d_model])

# Wrong (raises AttributeError):
print(h.value)    # no .value in 0.5.x
```

### 3. Single invoke squeezes the batch dimension

With one invoke, the batch dimension is removed from saved tensors:

```python
with model.trace("text"):
    h = model.transformer.h[0].output[0].save()

print(h.shape)  # (seq_len, d_model), NOT (1, seq_len, d_model)
```

Handle this in extraction code:
```python
if hs.dim() == 3:
    hs = hs[0]  # fallback if batch dim present
```

### 4. nnterp `StandardizedTransformer` fails with Llama on nnsight 0.5.x

`StandardizedTransformer` calls `check_model_renaming` which runs a scan-phase trace. This fails with:
```
TypeError: FakeTensor.__new__() got an unexpected keyword argument 'requires_grad'
```

**Workaround**: Use `nnsight.LanguageModel` directly and access Llama layers via `model.model.layers[i]`.

## Working Patterns

### Load model (Llama)

```python
from nnsight import LanguageModel

model = LanguageModel(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    dispatch=True,
    torch_dtype=torch.float16,
)
n_layers = len(list(model.model.layers))
```

### Extract all layer activations (Llama, pre-tokenized)

```python
saved = [None] * n_layers
with model.trace() as tracer:
    with tracer.invoke(input_ids=ids):
        for i in range(n_layers):
            saved[i] = model.model.layers[i].output[0].save()

# saved[i] is a torch.Tensor with shape (seq_len, d_model)
```

### Extract all layer activations (GPT-2, string prompt)

```python
with model.trace("The quick brown fox"):
    saved = [model.transformer.h[i].output[0].save() for i in range(12)]
```

### Module paths by architecture

| Architecture | Layer path | Layer output |
|---|---|---|
| GPT-2 | `model.transformer.h[i]` | `.output[0]` |
| Llama | `model.model.layers[i]` | `.output[0]` |

## Execution Model (for debugging)

nnsight 0.5.x uses **AST capture + thread-based interleaving**:

1. `model.trace(...)` captures the body code via `sys.settrace()` as source text
2. `compile()` creates an `Invoker` (only when positional args exist)
3. The Invoker wraps body code into a `Mediator` running in a daemon thread
4. Main thread runs the forward pass; module hooks provide values to waiting mediators via queues
5. When the trace context exits, saved values are real tensors

Key implication: if no Invoker is created (no positional args), the body runs immediately without interleaving — `.output` raises `ValueError` because the module was never actually called.
