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
    load_results,
    load_sync_env,
    prepare_condition_c,
)
from steer import (
    _generate_batch,
    load_steering_directions,
    readout_projections,
    score_steered_output,
)

# ── Global state (populated at startup) ──────────────────────────────────────

_model = None
_tokenizer = None
_directions: dict[str, np.ndarray] = {}  # name -> unit-norm vector
_d_model: int = 0
_cfg: ProbeConfig | None = None
_gpu_lock = asyncio.Lock()
_batch_size: int = 96
_layers: list[int] = []
_projection_stats: dict | None = None
_dataset: dict[str, dict] | None = None  # experiment_hash -> sample record
_log_file = None  # append-only JSONL log of all /generate requests+responses
_direction_scales: dict[str, float] = {}  # server name -> projection std


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


class ProjectionSpec(BaseModel):
    """Pin the projection along a direction to a target value at a layer."""
    direction: str | list[float]
    layer: int
    target: float


class GenerateRequest(BaseModel):
    prompts: list[PromptItem] | None = None
    # Reference samples by experiment_hash (alternative to prompts + score_meta)
    sample_ids: list[str] | None = None
    # Single-direction mode (backward compatible)
    direction: str | list[float] | None = None
    layer: int = 25
    mode: Literal["additive", "projection"] = "additive"
    alpha: float = 0.0
    projection_target: float | None = None
    # Multi-projection mode — takes precedence when set
    projections: list[ProjectionSpec] | None = None
    max_new_tokens: int = 512
    score: bool = False
    score_meta: list[ScoreMeta] | None = None

    @model_validator(mode="after")
    def _check_prompts_or_ids(self):
        has_prompts = self.prompts is not None and len(self.prompts) > 0
        has_ids = self.sample_ids is not None and len(self.sample_ids) > 0
        if not has_prompts and not has_ids:
            raise ValueError("Provide either 'prompts' or 'sample_ids'")
        if has_prompts and has_ids:
            raise ValueError("Provide either 'prompts' or 'sample_ids', not both")
        if has_prompts and self.score:
            if not self.score_meta:
                raise ValueError("score_meta required when score=True with prompts")
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
    sample_id: str | None = None
    baseline_label: str | None = None
    conflict_id: str | None = None
    direction: str | None = None  # a_to_b or b_to_a
    system_prompt: str | None = None
    user_prompt: str | None = None


class SteeringConfig(BaseModel):
    direction: str | None = None  # name only, raw vectors omitted
    layer: int | None = None
    mode: str | None = None
    alpha: float | None = None
    projection_target: float | None = None
    projections: list[dict] | None = None  # simplified projection specs


class GenerateResponse(BaseModel):
    responses: list[ResponseItem]
    steering: SteeringConfig
    n_prompts: int
    batch_size: int
    elapsed_s: float


class DirectionType(BaseModel):
    pattern: str
    description: str
    count_per_layer: int


class DirectionsSummary(BaseModel):
    total: int
    n_layers: int
    per_layer: int
    layers: list[int]
    types: list[DirectionType]
    constraints: list[str]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    gpu_memory_gb: float | None
    n_directions: int
    layers: list[int]
    d_model: int
    batch_size: int


# ── Helpers ──────────────────────────────────────────────────────────────────


def _unpack_directions(
    dirs: dict, layer: int,
) -> dict[str, np.ndarray]:
    """Map :func:`load_steering_directions` output to flat server-name → vector.

    Server name conventions (per layer L):
      - probe family: ``probe_L{L}``, ``probe_raw_L{L}``,
        ``probe_{cid}_L{L}``, ``probe_raw_{cid}_L{L}``.
      - cmd family: ``cmd_overall_L{L}``, ``cmd_overall_raw_L{L}``,
        ``cmd_{cid}_L{L}``, ``cmd_raw_{cid}_L{L}``.
      - iid_mm family: ``iid_mm_L{L}``, ``iid_mm_zscored_L{L}``,
        ``iid_mm_mean_L{L}``, ``iid_mm_{cid}_L{L}``,
        ``iid_mm_zscored_{cid}_L{L}``.
    """
    out: dict[str, np.ndarray] = {}
    for name, vec in dirs.items():
        # iid_mm branch — must be checked BEFORE probe/cmd to avoid the
        # "probe" / "cmd" substring tests below misclassifying these keys.
        if name.startswith("iid_mm"):
            if isinstance(vec, np.ndarray) and vec.ndim == 1:
                if name == "iid_mm_overall":
                    server_name = f"iid_mm_L{layer}"
                elif name == "iid_mm_overall_zscored":
                    server_name = f"iid_mm_zscored_L{layer}"
                elif name == "iid_mm_mean":
                    server_name = f"iid_mm_mean_L{layer}"
                else:
                    server_name = f"{name}_L{layer}"
                out[server_name] = vec
            elif isinstance(vec, dict):
                if name == "iid_mm_per_constraint":
                    prefix = "iid_mm"
                elif name == "iid_mm_per_constraint_zscored":
                    prefix = "iid_mm_zscored"
                else:
                    prefix = name
                for cid, cvec in vec.items():
                    if isinstance(cvec, np.ndarray) and cvec.ndim == 1:
                        out[f"{prefix}_{cid}_L{layer}"] = cvec
            continue

        if isinstance(vec, np.ndarray) and vec.ndim == 1:
            out[f"{name}_L{layer}"] = vec
        elif isinstance(vec, dict):
            if "probe" in name:
                prefix = "probe_raw" if name.endswith("_raw") else "probe"
            else:
                prefix = "cmd_raw" if name.endswith("_raw") else "cmd"
            for cname, cvec in vec.items():
                if isinstance(cvec, np.ndarray) and cvec.ndim == 1:
                    out[f"{prefix}_{cname}_L{layer}"] = cvec
    return out


def _proj_std_npz_to_server_name(key: str) -> str | None:
    """Translate an iid_mm NPZ-style key to the corresponding server name.

    Returns ``None`` for keys that do not match any known iid_mm layout.

    Examples
    --------
    ``overall_iid_mm_L12`` -> ``iid_mm_L12``
    ``overall_iid_mm_zscored_L12`` -> ``iid_mm_zscored_L12``
    ``mean_iid_mm_L12`` -> ``iid_mm_mean_L12``
    ``foo_bar_iid_mm_L12`` -> ``iid_mm_foo_bar_L12``
    ``foo_bar_iid_mm_zscored_L12`` -> ``iid_mm_zscored_foo_bar_L12``
    """
    # Extract layer suffix
    if "_L" not in key:
        return None
    body, layer_part = key.rsplit("_L", 1)
    if not layer_part.isdigit():
        return None
    suffix = f"_L{layer_part}"

    if body == "overall_iid_mm":
        return f"iid_mm{suffix}"
    if body == "overall_iid_mm_zscored":
        return f"iid_mm_zscored{suffix}"
    if body == "mean_iid_mm":
        return f"iid_mm_mean{suffix}"
    # Per-conflict: "{cid}_iid_mm" or "{cid}_iid_mm_zscored"
    if body.endswith("_iid_mm_zscored"):
        cid = body[: -len("_iid_mm_zscored")]
        if cid:
            return f"iid_mm_zscored_{cid}{suffix}"
    if body.endswith("_iid_mm"):
        cid = body[: -len("_iid_mm")]
        if cid:
            return f"iid_mm_{cid}{suffix}"
    return None


def _load_direction_scales(run_dir: Path) -> dict[str, float]:
    """Load ``iid_mm_proj_std.json`` and remap to server direction names.

    Returns an empty dict if the file does not exist.
    """
    path = run_dir / "iid_mm_proj_std.json"
    if not path.exists():
        return {}
    import json as _json
    raw = _json.loads(path.read_text())
    out: dict[str, float] = {}
    for k, v in raw.items():
        server_name = _proj_std_npz_to_server_name(k)
        if server_name is not None:
            out[server_name] = float(v)
    return out


def _resolve_direction_value(direction: str | list[float]) -> np.ndarray:
    """Resolve a direction name or raw vector to an ndarray."""
    if isinstance(direction, str):
        if direction not in _directions:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown direction {direction!r}. "
                f"Available: {list(_directions.keys())}",
            )
        return _directions[direction]

    vec = np.array(direction, dtype=np.float32)
    if vec.shape != (_d_model,):
        raise HTTPException(
            status_code=400,
            detail=f"Direction vector length {len(direction)} != d_model {_d_model}",
        )
    return vec


def _resolve_direction(req: GenerateRequest) -> np.ndarray | None:
    """Resolve single direction from request — name lookup or raw vector."""
    if req.direction is None:
        return None
    return _resolve_direction_value(req.direction)


def _resolve_projections(
    specs: list[ProjectionSpec],
) -> list[tuple[np.ndarray, int, float]]:
    """Resolve a list of ProjectionSpec into (vector, layer, target) tuples."""
    return [
        (_resolve_direction_value(s.direction), s.layer, s.target)
        for s in specs
    ]


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
    direction_vector: np.ndarray | None = None,
    layer: int | None = None,
    alpha: float = 0.0,
    projection_target: float | None = None,
    projections: list[tuple[np.ndarray, int, float]] | None = None,
    max_new_tokens: int = 512,
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
            projections=projections,
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
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Layers to load directions for (default: all)")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _directions, _d_model, _cfg, _batch_size, _layers

    args = _parse_args()
    _batch_size = args.batch_size

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
    _layers = args.layers if args.layers is not None else list(range(n_layers))
    print(f"d_model: {_d_model}, layers: {n_layers}")

    # Activate per-model thresholds for verifiers
    from phase0_v2.config.thresholds import activate_model
    activate_model(_cfg.model_name)
    print(f"Activated per-model thresholds for {_cfg.model_name}")

    # Tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(_cfg.model_name)

    # Steering directions (optional — server works without them)
    if args.run_id:
        pos = "last_prompt"
        for layer in _layers:
            try:
                dirs = load_steering_directions(_cfg.run_dir, pos, layer)
                unpacked = _unpack_directions(dirs, layer)
                for server_name, vec in unpacked.items():
                    _directions[server_name] = vec
                    # Convenience aliases (single-layer servers)
                    if len(_layers) == 1 and server_name.endswith(f"_L{layer}"):
                        alias = server_name[: -len(f"_L{layer}")]
                        if alias:
                            _directions[alias] = vec
                loaded = [k for k in _directions if k.endswith(f"_L{layer}")]
                print(f"Loaded {len(loaded)} directions for L{layer}")
            except Exception as e:
                print(f"Warning: could not load directions for L{layer}: {e}")

        # Direction scales (projection std) for iid_mm rawspace dirs
        global _direction_scales
        try:
            _direction_scales = _load_direction_scales(_cfg.run_dir)
            if _direction_scales:
                print(f"Loaded {len(_direction_scales)} direction scales from iid_mm_proj_std.json")
        except Exception as e:
            print(f"Warning: could not load direction scales: {e}")
            _direction_scales = {}

    # Projection stats (optional — pre-computed by compute_projection_stats.py)
    global _projection_stats
    if args.run_id:
        stats_path = _cfg.run_dir / "projection_stats.json"
        if stats_path.exists():
            import json
            _projection_stats = json.loads(stats_path.read_text())
            print(f"Loaded projection stats: {len(_projection_stats)} groups")

    # Dataset (optional — enables sample_ids in /generate requests)
    global _dataset
    if args.run_id:
        try:
            from compute_cmds import load_run_config
            run_cfg = load_run_config(_cfg.run_dir)
            df_all = load_results(run_cfg.data_dir, run_cfg.model_name)
            df_c = prepare_condition_c(df_all, run_cfg.label_mode,
                                       conflict_ids=run_cfg.conflict_ids)
            df_c = df_c.sort_values("conflict_id").reset_index(drop=True)
            _dataset = {}
            for _, row in df_c.iterrows():
                h = row.get("experiment_hash")
                if h is None:
                    continue
                args_val = row.get("instruction_args", {})
                if isinstance(args_val, str):
                    import json as _json
                    args_val = _json.loads(args_val)
                _dataset[h] = {
                    "system_prompt": row["system_prompt"],
                    "user_prompt": row["user_prompt"],
                    "conflict_id": row["conflict_id"],
                    "direction": row["direction"],
                    "instruction_args": args_val,
                    "label": row.get("label", ""),
                    "y": int(row.get("y", 0)),
                }
            print(f"Loaded dataset: {len(_dataset)} samples indexed by experiment_hash")
        except Exception as e:
            print(f"Warning: could not load dataset for sample_ids: {e}")
            _dataset = None

    # Request/response log (append-only JSONL)
    global _log_file
    if args.run_id:
        log_path = _cfg.run_dir / "server_log.jsonl"
        _log_file = open(log_path, "a")
        print(f"Logging to: {log_path}")

    print(f"Server ready — {len(_directions)} directions loaded")
    print(f"Directions: {list(_directions.keys())}")

    yield

    # Shutdown
    if _log_file is not None:
        _log_file.close()
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
        n_directions=len(_directions),
        layers=_layers,
        d_model=_d_model,
        batch_size=_batch_size,
    )


@app.get("/directions", response_model=DirectionsSummary)
async def directions():
    # Extract constraint names from direction keys
    constraints = sorted({
        name.replace("cmd_", "").rsplit("_L", 1)[0]
        for name in _directions
        if name.startswith("cmd_") and not name.startswith("cmd_overall")
        and not name.startswith("cmd_raw_")
    })
    n_constraints = len(constraints)

    # iid_mm constraint set — keys like iid_mm_{cid}_L{layer}, excluding
    # iid_mm_zscored_*, iid_mm_mean_*, and iid_mm_L{layer} (overall).
    iid_mm_constraints = sorted({
        name[len("iid_mm_"):].rsplit("_L", 1)[0]
        for name in _directions
        if name.startswith("iid_mm_")
        and not name.startswith("iid_mm_zscored")
        and not name.startswith("iid_mm_mean")
        and "_L" in name[len("iid_mm_"):]  # body has an _L separator (excludes overall iid_mm_L{layer})
        and name[len("iid_mm_"):].rsplit("_L", 1)[0]  # non-empty cid
    })
    n_iid_constraints = len(iid_mm_constraints)
    has_iid_overall = any(
        name.startswith("iid_mm_L") for name in _directions
    )
    has_iid_zscored_overall = any(
        name.startswith("iid_mm_zscored_L") for name in _directions
    )
    has_iid_mean = any(
        name.startswith("iid_mm_mean_L") for name in _directions
    )

    types = [
        DirectionType(pattern="probe_L{layer}", description="unit-norm logistic regression weight", count_per_layer=1),
        DirectionType(pattern="probe_raw_L{layer}", description="unnormalized logistic regression weight", count_per_layer=1),
        DirectionType(pattern="probe_{constraint}_L{layer}", description=f"unit-norm per-constraint probe ({n_constraints} constraints)", count_per_layer=n_constraints),
        DirectionType(pattern="probe_raw_{constraint}_L{layer}", description=f"unnormalized per-constraint probe ({n_constraints} constraints)", count_per_layer=n_constraints),
        DirectionType(pattern="cmd_overall_L{layer}", description="unit-norm overall class-mean difference", count_per_layer=1),
        DirectionType(pattern="cmd_overall_raw_L{layer}", description="unnormalized overall CMD", count_per_layer=1),
        DirectionType(pattern="cmd_{constraint}_L{layer}", description=f"unit-norm per-constraint CMD ({n_constraints} constraints)", count_per_layer=n_constraints),
        DirectionType(pattern="cmd_raw_{constraint}_L{layer}", description=f"unnormalized per-constraint CMD ({n_constraints} constraints)", count_per_layer=n_constraints),
        DirectionType(pattern="iid_mm_L{layer}", description="unit-norm overall IID Mass-Mean direction (rawspace fit)", count_per_layer=1 if has_iid_overall else 0),
        DirectionType(pattern="iid_mm_zscored_L{layer}", description="unit-norm overall IID Mass-Mean direction (zscored fit)", count_per_layer=1 if has_iid_zscored_overall else 0),
        DirectionType(pattern="iid_mm_mean_L{layer}", description="mean of per-constraint IID-MM rawspace directions, renormalized", count_per_layer=1 if has_iid_mean else 0),
        DirectionType(pattern="iid_mm_{constraint}_L{layer}", description=f"per-constraint IID-MM direction (rawspace fit, {n_iid_constraints} constraints)", count_per_layer=n_iid_constraints),
        DirectionType(pattern="iid_mm_zscored_{constraint}_L{layer}", description=f"per-constraint IID-MM direction (zscored fit, {n_iid_constraints} constraints)", count_per_layer=n_iid_constraints),
    ]

    per_layer = sum(t.count_per_layer for t in types)

    return DirectionsSummary(
        total=len(_directions),
        n_layers=len(_layers),
        per_layer=per_layer,
        layers=_layers,
        types=types,
        constraints=constraints,
    )


@app.get("/direction_scales")
async def direction_scales():
    """Per-direction projection std (rawspace IID-MM directions).

    Keys are server direction names (e.g. ``iid_mm_L12``); values are floats.
    Returns an empty dict if ``iid_mm_proj_std.json`` is not present.
    """
    return _direction_scales


class DirectionVectorResponse(BaseModel):
    name: str
    shape: list[int]
    norm: float
    vector: list[float]


@app.get("/direction_vector/{name}", response_model=DirectionVectorResponse)
async def direction_vector(name: str):
    """Return the actual vector for a named direction."""
    if name not in _directions:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown direction {name!r}. Use /directions for available patterns.",
        )
    vec = _directions[name]
    return DirectionVectorResponse(
        name=name,
        shape=list(vec.shape),
        norm=float(np.linalg.norm(vec)),
        vector=vec.tolist(),
    )


class AddDirectionRequest(BaseModel):
    name: str
    vector: list[float]


@app.post("/add_direction")
async def add_direction(req: AddDirectionRequest):
    """Register a custom direction by name (e.g., mean of per-constraint probes)."""
    vec = np.array(req.vector, dtype=np.float32)
    if vec.shape != (_d_model,):
        raise HTTPException(
            400, f"Vector length {len(req.vector)} != d_model {_d_model}"
        )
    _directions[req.name] = vec
    return {"status": "ok", "name": req.name, "norm": float(np.linalg.norm(vec))}


@app.get("/projection_stats")
async def projection_stats(
    constraint_type: str | None = None,
    layer: int | None = None,
):
    """Serve pre-computed projection distribution statistics."""
    if _projection_stats is None:
        raise HTTPException(
            404, "Projection stats not computed. Run compute_projection_stats.py first.",
        )
    if constraint_type is None and layer is None:
        return _projection_stats

    # Filter by constraint and/or layer
    result = {}
    groups = [constraint_type] if constraint_type else list(_projection_stats.keys())
    for g in groups:
        if g not in _projection_stats:
            continue
        if layer is not None:
            lk = f"L{layer}"
            if lk in _projection_stats[g]:
                result.setdefault(g, {})[lk] = _projection_stats[g][lk]
        else:
            result[g] = _projection_stats[g]
    return result


class ProjectRequest(BaseModel):
    prompts: list[PromptItem]
    directions: list[str]
    layer: int = 25


class ProjectionResult(BaseModel):
    prompt_idx: int
    values: dict[str, float]


class ProjectResponse(BaseModel):
    projections: list[ProjectionResult]
    elapsed_s: float


@app.post("/project", response_model=ProjectResponse)
async def project(req: ProjectRequest):
    """Readout-only: measure projections onto directions without steering."""
    # Resolve direction vectors
    dir_vectors: dict[str, np.ndarray] = {}
    for name in req.directions:
        dir_vectors[name] = _resolve_direction_value(name)

    prompts = _format_prompts(req.prompts)

    t0 = time.time()
    async with _gpu_lock:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: readout_projections(
                _model, _tokenizer, prompts, dir_vectors, req.layer,
            ),
        )
    elapsed = time.time() - t0

    items = [
        ProjectionResult(prompt_idx=i, values=vals)
        for i, vals in enumerate(results)
    ]
    return ProjectResponse(projections=items, elapsed_s=elapsed)


@app.get("/samples")
async def get_samples(
    conflict_id: str | None = None,
    direction: str | None = None,
    baseline_label: str | None = None,
    limit: int | None = None,
    seed: int | None = None,
):
    """List available sample_ids, optionally filtered and shuffled.

    Query params:
      - conflict_id: filter by constraint (e.g., json_only_vs_plain)
      - direction: filter by a_to_b or b_to_a
      - baseline_label: filter by followed_user or followed_system
      - limit: max number of sample_ids to return
      - seed: random seed for deterministic shuffling (omit for stable order)

    Same seed always returns the same subset. Use seed=42 across experiments
    for comparable sample sets, or vary the seed for independent samples.
    """
    if _dataset is None:
        raise HTTPException(400, "Dataset not loaded")
    ids = []
    for h, rec in _dataset.items():
        if conflict_id and rec["conflict_id"] != conflict_id:
            continue
        if direction and rec["direction"] != direction:
            continue
        if baseline_label:
            bl = "followed_system" if rec["y"] == 1 else "followed_user"
            if bl != baseline_label:
                continue
        ids.append(h)
    if seed is not None:
        import random
        rng = random.Random(seed)
        rng.shuffle(ids)
    if limit and limit < len(ids):
        ids = ids[:limit]
    return {
        "sample_ids": ids,
        "total": len(ids),
        "filters": {"conflict_id": conflict_id, "direction": direction,
                     "baseline_label": baseline_label, "seed": seed},
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    # Resolve sample_ids → prompts + score_meta
    if req.sample_ids:
        if _dataset is None:
            raise HTTPException(400, "Dataset not loaded — cannot use sample_ids")
        missing = [h for h in req.sample_ids if h not in _dataset]
        if missing:
            raise HTTPException(400, f"Unknown sample_ids: {missing[:5]}...")
        req.prompts = []
        req.score_meta = []
        for h in req.sample_ids:
            rec = _dataset[h]
            req.prompts.append(PromptItem(
                system_prompt=rec["system_prompt"],
                user_prompt=rec["user_prompt"],
            ))
            req.score_meta.append(ScoreMeta(
                conflict_id=rec["conflict_id"],
                direction=rec["direction"],
                instruction_args=rec["instruction_args"],
            ))
        req.score = True  # always score when using sample_ids

    # Multi-projection mode takes precedence
    if req.projections:
        proj_tuples = _resolve_projections(req.projections)
        gen_kwargs = dict(
            projections=proj_tuples,
            max_new_tokens=req.max_new_tokens,
        )
    else:
        direction_vector = _resolve_direction(req)
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
        sample_id = req.sample_ids[i] if req.sample_ids else None
        baseline_label = conflict_id_val = direction_val = sys_prompt = usr_prompt = None
        if sample_id and _dataset:
            rec = _dataset[sample_id]
            baseline_label = "followed_system" if rec["y"] == 1 else "followed_user"
            conflict_id_val = rec["conflict_id"]
            direction_val = rec["direction"]
            sys_prompt = rec["system_prompt"]
            usr_prompt = rec["user_prompt"]
        items.append(ResponseItem(
            text=text, label=label, confidence=confidence,
            sys_ok=sys_ok, usr_ok=usr_ok,
            sample_id=sample_id, baseline_label=baseline_label,
            conflict_id=conflict_id_val, direction=direction_val,
            system_prompt=sys_prompt, user_prompt=usr_prompt,
        ))

    # Build steering config for response (omit raw vectors for brevity)
    dir_name = req.direction if isinstance(req.direction, str) else (
        f"<vector({len(req.direction)})>" if isinstance(req.direction, list) else None
    )
    proj_summary = None
    if req.projections:
        proj_summary = []
        for p in req.projections:
            d = p.direction if isinstance(p.direction, str) else f"<vector({len(p.direction)})>"
            proj_summary.append({"direction": d, "layer": p.layer, "target": p.target})

    steering = SteeringConfig(
        direction=dir_name,
        layer=req.layer if dir_name else None,
        mode=req.mode if dir_name else None,
        alpha=req.alpha if dir_name and req.mode == "additive" else None,
        projection_target=req.projection_target if dir_name and req.mode == "projection" else None,
        projections=proj_summary,
    )

    response = GenerateResponse(
        responses=items,
        steering=steering,
        n_prompts=len(prompts),
        batch_size=_batch_size,
        elapsed_s=round(elapsed, 3),
    )

    # Append to log
    if _log_file is not None:
        import json as _json
        import datetime
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "steering": steering.model_dump(),
            "n_prompts": len(prompts),
            "elapsed_s": round(elapsed, 3),
            "responses": [item.model_dump(exclude_none=True) for item in items],
        }
        _log_file.write(_json.dumps(log_entry) + "\n")
        _log_file.flush()

    return response


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    uvicorn.run(
        "steering_server:app",
        host=args.host,
        port=args.port,
        workers=1,
    )
