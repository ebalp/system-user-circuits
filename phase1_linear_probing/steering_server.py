"""FastAPI steering server — persistent model, on-demand steering.

Loads the model once at startup and serves steering generation requests
via HTTP.  Agents can explore the steering space interactively without
model loading overhead.

Usage:
  uv run python phase1_linear_probing/steering_server.py
  uv run python phase1_linear_probing/steering_server.py --model meta-llama/Llama-3.1-8B-Instruct
  uv run python phase1_linear_probing/steering_server.py --run-id curated35-8b-v001 --layers 25
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from transformers import AutoTokenizer

# ── Path setup (same pattern as run_steering_pipeline.py) ────────────────────
_phase1_dir = Path(__file__).resolve().parent
_repo_root = _phase1_dir.parent
sys.path.insert(0, str(_phase1_dir))
sys.path.insert(0, str(_repo_root))

from data import (
    ProbeConfig,
    _cleanup_nn_model,
    _load_nn_model,
    build_formatted_prompt,
    load_sync_env,
)
from steer import (
    _generate_batch,
    load_steering_directions,
    score_steered_output,
)

# ── Global state (populated at startup) ──────────────────────────────────────

_model = None
_tokenizer = None
_directions: dict[str, np.ndarray] = {}  # name -> unit-norm vector
_d_model: int = 0
_cfg: ProbeConfig | None = None
_gpu_lock = asyncio.Lock()
_batch_size: int = 48
_layers: list[int] = []


# ── Pydantic models ─────────────────────────────────────────────────────────


class PromptItem(BaseModel):
    raw_prompt: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None

    @model_validator(mode="after")
    def _check_prompt_mode(self):
        has_raw = self.raw_prompt is not None
        has_pair = self.system_prompt is not None and self.user_prompt is not None
        if has_raw == has_pair:
            raise ValueError(
                "Provide either 'raw_prompt' OR both 'system_prompt' and "
                "'user_prompt', not both/neither"
            )
        return self


class ScoreMeta(BaseModel):
    conflict_id: str
    direction: str  # "a_to_b" or "b_to_a"
    instruction_args: dict = {}


class GenerateRequest(BaseModel):
    prompts: list[PromptItem]
    direction: str | list[float] | None = None
    layer: int = 25
    mode: Literal["additive", "projection"] = "additive"
    alpha: float = 0.0
    projection_target: float | None = None
    max_new_tokens: int = 512
    score: bool = False
    score_meta: list[ScoreMeta] | None = None

    @model_validator(mode="after")
    def _check_score_meta(self):
        if self.score:
            if not self.score_meta:
                raise ValueError("score_meta required when score=True")
            if len(self.score_meta) != len(self.prompts):
                raise ValueError(
                    f"score_meta length ({len(self.score_meta)}) must match "
                    f"prompts length ({len(self.prompts)})"
                )
        return self


class ResponseItem(BaseModel):
    text: str
    label: str | None = None
    confidence: float | None = None
    sys_ok: bool | None = None
    usr_ok: bool | None = None


class GenerateResponse(BaseModel):
    responses: list[ResponseItem]
    n_prompts: int
    batch_size: int
    elapsed_s: float


class DirectionInfo(BaseModel):
    name: str
    shape: list[int]
    norm: float
    layer: int


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    gpu_memory_gb: float | None
    directions: list[str]
    layers: list[int]
    d_model: int
    batch_size: int


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_direction(req: GenerateRequest) -> np.ndarray | None:
    """Resolve direction from request — name lookup or raw vector."""
    if req.direction is None:
        return None

    if isinstance(req.direction, str):
        if req.direction not in _directions:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown direction {req.direction!r}. "
                f"Available: {list(_directions.keys())}",
            )
        return _directions[req.direction]

    # Raw vector
    vec = np.array(req.direction, dtype=np.float32)
    if vec.shape != (_d_model,):
        raise HTTPException(
            status_code=400,
            detail=f"Direction vector length {len(req.direction)} != d_model {_d_model}",
        )
    return vec


def _format_prompts(items: list[PromptItem]) -> list[str]:
    """Format prompt items into strings."""
    prompts = []
    for item in items:
        if item.raw_prompt is not None:
            prompts.append(item.raw_prompt)
        else:
            prompts.append(
                build_formatted_prompt(_tokenizer, item.system_prompt, item.user_prompt)
            )
    return prompts


def _generate_chunked(
    prompts: list[str],
    *,
    direction_vector: np.ndarray | None,
    layer: int | None,
    alpha: float,
    projection_target: float | None,
    max_new_tokens: int,
) -> list[str]:
    """Generate with server-side batching."""
    all_responses: list[str] = []
    for start in range(0, len(prompts), _batch_size):
        batch = prompts[start : start + _batch_size]
        responses = _generate_batch(
            _model, _tokenizer, batch,
            direction_vector=direction_vector,
            layer=layer,
            alpha=alpha,
            projection_target=projection_target,
            max_new_tokens=max_new_tokens,
        )
        all_responses.extend(responses)
    return all_responses


# ── Lifespan ─────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Steering server")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--run-id", default=None,
                        help="Probe run ID for loading directions")
    parser.add_argument("--layers", type=int, nargs="*", default=[25],
                        help="Layers to load directions for (default: 25)")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _directions, _d_model, _cfg, _batch_size, _layers

    args = _parse_args()
    _batch_size = args.batch_size
    _layers = args.layers

    # Config
    _cfg = ProbeConfig(
        model_name=args.model,
        run_id=args.run_id or "server",
        batch_size=1,
    )
    load_sync_env(_cfg.repo_root)

    # Model
    print(f"Loading model: {_cfg.model_name}")
    _model, n_layers = _load_nn_model(_cfg)

    # d_model from the model's embedding dimension
    hf_model = _model._model if hasattr(_model, "_model") else _model
    _d_model = hf_model.config.hidden_size
    print(f"d_model: {_d_model}, layers: {n_layers}")

    # Tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(_cfg.model_name)

    # Steering directions (optional — server works without them)
    if args.run_id:
        pos = "last_prompt"
        for layer in args.layers:
            try:
                dirs = load_steering_directions(_cfg.run_dir, pos, layer)
                for name, vec in dirs.items():
                    if isinstance(vec, np.ndarray) and vec.ndim == 1:
                        _directions[f"{name}_L{layer}"] = vec
                        # Also register without layer suffix if only one layer
                        if len(args.layers) == 1:
                            _directions[name] = vec
                print(f"Loaded directions for L{layer}: {[k for k, v in dirs.items() if isinstance(v, np.ndarray) and v.ndim == 1]}")
            except Exception as e:
                print(f"Warning: could not load directions for L{layer}: {e}")

    print(f"Server ready — {len(_directions)} directions loaded")
    print(f"Directions: {list(_directions.keys())}")

    yield

    # Shutdown
    print("Shutting down — cleaning up model")
    if _model is not None:
        _cleanup_nn_model(_model, _cfg.device)
        del _model
        _model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Steering Server", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = round(torch.cuda.memory_allocated() / 1e9, 2)
    return HealthResponse(
        status="ok",
        model=_cfg.model_name,
        device=_cfg.device,
        gpu_memory_gb=gpu_mem,
        directions=list(_directions.keys()),
        layers=_layers,
        d_model=_d_model,
        batch_size=_batch_size,
    )


@app.get("/directions", response_model=list[DirectionInfo])
async def directions():
    result = []
    for name, vec in _directions.items():
        # Infer layer from name suffix (e.g., "probe_L25")
        layer = _layers[0] if _layers else 0
        if "_L" in name:
            try:
                layer = int(name.split("_L")[-1])
            except ValueError:
                pass
        result.append(DirectionInfo(
            name=name,
            shape=list(vec.shape),
            norm=round(float(np.linalg.norm(vec)), 6),
            layer=layer,
        ))
    return result


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    direction_vector = _resolve_direction(req)

    # Determine steering kwargs
    gen_kwargs = dict(
        direction_vector=direction_vector,
        layer=req.layer if direction_vector is not None else None,
        alpha=req.alpha,
        projection_target=None,
        max_new_tokens=req.max_new_tokens,
    )
    if req.mode == "projection" and direction_vector is not None:
        gen_kwargs["projection_target"] = req.projection_target
        gen_kwargs["alpha"] = 0.0  # projection mode ignores alpha

    # Format prompts
    prompts = _format_prompts(req.prompts)

    # Generate under GPU lock
    t0 = time.time()
    async with _gpu_lock:
        loop = asyncio.get_event_loop()
        responses = await loop.run_in_executor(
            None,
            lambda: _generate_chunked(prompts, **gen_kwargs),
        )
    elapsed = time.time() - t0

    # Score if requested
    items: list[ResponseItem] = []
    for i, text in enumerate(responses):
        label = confidence = sys_ok = usr_ok = None
        if req.score and req.score_meta:
            meta = req.score_meta[i]
            try:
                label, confidence, sys_ok, usr_ok = score_steered_output(
                    text, meta.conflict_id, meta.direction, meta.instruction_args,
                )
            except Exception as e:
                label = f"error: {e}"
                confidence = 0.0
                sys_ok = usr_ok = False
        items.append(ResponseItem(
            text=text, label=label, confidence=confidence,
            sys_ok=sys_ok, usr_ok=usr_ok,
        ))

    return GenerateResponse(
        responses=items,
        n_prompts=len(prompts),
        batch_size=_batch_size,
        elapsed_s=round(elapsed, 3),
    )


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    uvicorn.run(
        "steering_server:app",
        host=args.host,
        port=args.port,
        workers=1,
    )
