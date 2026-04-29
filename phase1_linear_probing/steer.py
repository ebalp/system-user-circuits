"""Activation steering: add probe/CMD directions during generation.

Provides helpers to load steering directions from probe results,
steer model generation via nnsight, score steered outputs using
Phase 0 v2 verifiers, and run parameter sweeps over steering strength.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ── Direction Loading ─────────────────────────────────────────────────────────


def load_steering_directions(
    run_dir: Path,
    pos_name: str,
    layer: int,
) -> dict[str, np.ndarray]:
    """Load probe and CMD directions for a given layer.

    Parameters
    ----------
    run_dir : Path
        Run directory containing probe results and CMD files.
    pos_name : str
        Token position name (e.g. ``"last_prompt"``).
    layer : int
        Layer index to extract directions from.

    Returns
    -------
    dict with keys:

    - ``"probe"`` — unit-norm probe weight vector at *layer*
    - ``"probe_raw"`` — raw (unnormalized) probe weight vector
    - ``"cmd_overall"`` — mean of fold CMD vectors (unit-normalized)
    - ``"cmd_per_constraint"`` — dict mapping constraint type to CMD vector
    """
    from probe import load_results, results_path

    rpath = results_path(run_dir, cv_mode="grouped", use_scaler=False)
    if not rpath.exists():
        rpath = results_path(run_dir, cv_mode="stratified", use_scaler=False)
    results = load_results(rpath)
    pr = results[pos_name]

    directions: dict[str, np.ndarray] = {
        "probe": pr.weights[layer],
        "probe_raw": pr.weights_raw[layer],
    }

    # Unified CMD file (from compute_cmds.py) — preferred source
    unified_path = run_dir / "constraint_cmds.npz"
    if unified_path.exists():
        cdata = np.load(unified_path)
        suffix = f"_L{layer}"
        suffix_raw = f"_L{layer}_raw"
        for key in cdata.files:
            if key == f"overall{suffix}":
                directions["cmd_overall"] = cdata[key]
            elif key == f"overall{suffix_raw}":
                directions["cmd_overall_raw"] = cdata[key]
            elif key.endswith(suffix_raw):
                name = key[: -len(suffix_raw)]
                directions.setdefault("cmd_per_constraint_raw", {})[name] = cdata[key]
            elif key.endswith(suffix):
                name = key[: -len(suffix)]
                directions.setdefault("cmd_per_constraint", {})[name] = cdata[key]
    else:
        # Legacy per-layer files
        fold_cmds_path = run_dir / f"fold_cmds_L{layer}.npz"
        if fold_cmds_path.exists():
            fold_data = np.load(fold_cmds_path)
            vecs = np.stack([fold_data[k] for k in fold_data.files])
            mean_cmd = vecs.mean(axis=0)
            norm = np.linalg.norm(mean_cmd)
            directions["cmd_overall"] = mean_cmd / norm if norm > 0 else mean_cmd
        else:
            logger.warning("No fold CMDs at %s — cmd_overall unavailable", fold_cmds_path)

        constraint_cmds_path = run_dir / f"constraint_cmds_L{layer}.npz"
        if constraint_cmds_path.exists():
            cdata = np.load(constraint_cmds_path)
            directions["cmd_per_constraint"] = {k: cdata[k] for k in cdata.files}
        else:
            logger.warning("No constraint CMDs at %s", constraint_cmds_path)

    # Per-constraint probes (from compute_per_constraint_probes.py)
    probes_path = run_dir / "constraint_probes.npz"
    if probes_path.exists():
        pdata = np.load(probes_path)
        suffix_probe = f"_probe_L{layer}"
        suffix_probe_raw = f"_probe_raw_L{layer}"
        for key in pdata.files:
            if key.endswith(suffix_probe_raw):
                name = key[: -len(suffix_probe_raw)]
                directions.setdefault("probe_per_constraint_raw", {})[name] = pdata[key]
            elif key.endswith(suffix_probe):
                name = key[: -len(suffix_probe)]
                directions.setdefault("probe_per_constraint", {})[name] = pdata[key]

    # IID Mass-Mean directions (from compute_iid_mm.py) — optional
    iid_mm_path = run_dir / "iid_mm_directions.npz"
    if iid_mm_path.exists():
        idata = np.load(iid_mm_path)
        overall_key = f"overall_iid_mm_L{layer}"
        overall_z_key = f"overall_iid_mm_zscored_L{layer}"
        mean_key = f"mean_iid_mm_L{layer}"
        suffix_raw = f"_iid_mm_L{layer}"
        suffix_z = f"_iid_mm_zscored_L{layer}"
        for key in idata.files:
            if key == overall_key:
                directions["iid_mm_overall"] = idata[key]
            elif key == overall_z_key:
                directions["iid_mm_overall_zscored"] = idata[key]
            elif key == mean_key:
                directions["iid_mm_mean"] = idata[key]
            elif key.endswith(suffix_z):
                cid = key[: -len(suffix_z)]
                directions.setdefault("iid_mm_per_constraint_zscored", {})[cid] = idata[key]
            elif key.endswith(suffix_raw):
                cid = key[: -len(suffix_raw)]
                directions.setdefault("iid_mm_per_constraint", {})[cid] = idata[key]

    return directions


# ── Readout (Projection Measurement) ────────────────────────────────────────


def readout_projections(
    model,
    tokenizer,
    prompts: list[str],
    direction_vectors: dict[str, np.ndarray],
    layer: int,
) -> list[dict[str, float]]:
    """Measure projections of last-token hidden states onto directions.

    Runs a single forward pass (no generation) with a readout hook at
    *layer*, then computes dot products with each direction vector.

    Parameters
    ----------
    model : nnsight.LanguageModel or HF model
        Loaded model.
    tokenizer
        HuggingFace tokenizer.
    prompts : list[str]
        Pre-formatted prompt strings.
    direction_vectors : dict[str, np.ndarray]
        Named direction vectors, each shape ``(d_model,)``.
    layer : int
        Layer index to read activations from.

    Returns
    -------
    list[dict[str, float]]
        One dict per prompt mapping direction name to projection value.
    """
    device = next(model.parameters()).device
    hf_model = model._model if hasattr(model, "_model") else model

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, add_special_tokens=False,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    # Hook to capture last-token hidden state
    captured: dict[str, torch.Tensor] = {}

    def _readout_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        # With left-padding, last token is always at seq_len - 1
        captured["h"] = h[:, -1, :].detach().float()

    handle = _get_decoder_layers(hf_model)[layer].register_forward_hook(_readout_hook)
    try:
        with torch.no_grad():
            hf_model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    h = captured["h"]  # (batch, d_model)

    # Build direction matrix and project
    dir_names = list(direction_vectors.keys())
    D = np.stack([direction_vectors[n] for n in dir_names], axis=1)  # (d_model, k)
    D_t = torch.tensor(D, dtype=torch.float32, device=device)
    projections = (h @ D_t).cpu().numpy()  # (batch, k)

    results = []
    for i in range(len(prompts)):
        results.append({
            name: float(projections[i, j])
            for j, name in enumerate(dir_names)
        })
    return results


# ── Batched Generation ───────────────────────────────────────────────────────


def _get_decoder_layers(hf_model):
    """Resolve the decoder layer list for any supported HF model.

    Handles both standard architectures (Llama, Gemma2, Qwen2, Mistral)
    where layers live at ``model.layers`` and multimodal wrappers (Gemma3)
    where they live at ``model.language_model.layers``.
    """
    if hasattr(hf_model.model, "layers"):
        return hf_model.model.layers
    if hasattr(hf_model.model, "language_model"):
        return hf_model.model.language_model.layers
    raise ValueError(
        f"Cannot find decoder layers on {type(hf_model).__name__}. "
        f"Expected model.model.layers or model.model.language_model.layers."
    )


def _make_multi_projection_hook(
    directions_tensor: torch.Tensor,
    targets_tensor: torch.Tensor,
):
    """Build a forward hook that pins projections along multiple directions.

    For *k* directions at the same layer, solves:

        h' = h + D @ (D^T D)^{-1} @ (targets - D^T h)

    where D is (d_model, k) and targets is (k,).  When k=1 this reduces
    to the single-direction formula ``h + (t - h·v) * v``.
    """
    # D: (d_model, k), targets: (k,)
    D = directions_tensor  # already on device
    t = targets_tensor
    # Precompute (D^T D)^{-1} in float32 — small k×k matrix
    D_f32 = D.float()
    DtD_inv = torch.linalg.inv(D_f32.T @ D_f32)  # (k, k)
    # Precompute D @ (D^T D)^{-1} to avoid repeated matmuls in the hook
    D_DtDinv = (D_f32 @ DtD_inv).half()  # back to fp16: (d_model, k)

    def _hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        # h: (batch, seq, d_model)
        # D^T h: project h onto each direction → (batch, seq, k)
        Dt_h = torch.einsum("dk,bsd->bsk", D, h)
        # residual: (batch, seq, k)
        residual = t.unsqueeze(0).unsqueeze(0) - Dt_h
        # adjustment: D_DtDinv @ residual → (batch, seq, d_model)
        correction = torch.einsum("dk,bsk->bsd", D_DtDinv, residual)
        h_new = h + correction
        if isinstance(output, tuple):
            return (h_new, *output[1:])
        return h_new

    return _hook


def _generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    direction_vector: np.ndarray | None = None,
    layer: int | None = None,
    alpha: float = 0.0,
    projection_target: float | None = None,
    projections: list[tuple[np.ndarray, int, float]] | None = None,
    max_new_tokens: int = 512,
) -> list[str]:
    """Core batched generation primitive.

    All generation flows through this function.  Uses raw HuggingFace
    ``generate()`` for both steered (alpha != 0, via forward hook) and
    unsteered (alpha == 0) paths to ensure identical backends.

    Steering modes (mutually exclusive, checked in this order):

    1. **Multi-projection** (``projections`` is set): pins the component
       along each direction to its target value.  Directions at the same
       layer are handled jointly via least-squares solve.
    2. **Single projection** (``projection_target`` is set):
       ``h' = h + (target - h·v) * v``.
    3. **Additive** (default): ``h' = h + alpha * v``.

    Parameters
    ----------
    model : nnsight.LanguageModel
        Loaded nnsight model (uses ``model._model`` for generation).
    tokenizer
        HuggingFace tokenizer.
    prompts : list[str]
        Pre-formatted prompt strings.
    direction_vector : np.ndarray or None
        Steering direction (single-direction modes).
    layer : int or None
        Layer index (single-direction modes).
    alpha : float
        Steering strength (additive mode).  0 = unsteered baseline.
    projection_target : float or None
        If set, use single-direction projection mode.
    projections : list of (direction_vector, layer, target) or None
        Multi-projection mode: each tuple specifies a direction (d_model,),
        a layer index, and a target projection value.  Directions at the
        same layer are solved jointly.
    max_new_tokens : int
        Maximum tokens to generate.

    Returns
    -------
    list[str]
        Decoded responses (generated tokens only), one per prompt.
    """
    device = next(model.parameters()).device

    # Ensure tokenizer has a pad token and uses left-padding for
    # decoder-only models (required for correct batched generation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    pad_token_id = tokenizer.pad_token_id

    # Batch-tokenize all prompts (used by all paths)
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, add_special_tokens=False,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    max_input_len = input_ids.shape[1]

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    hf_model = model._model if hasattr(model, "_model") else model
    handles: list = []

    try:
        if projections is not None:
            # Multi-projection mode: group by layer, register one hook per layer
            from collections import defaultdict
            layer_groups: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
            for vec, lyr, tgt in projections:
                layer_groups[lyr].append((vec, tgt))

            decoder_layers = _get_decoder_layers(hf_model)
            for lyr, group in layer_groups.items():
                vecs = [v for v, _ in group]
                tgts = [t for _, t in group]
                D = torch.tensor(
                    np.stack(vecs, axis=1), dtype=torch.float16,
                ).to(device)  # (d_model, k)
                t = torch.tensor(tgts, dtype=torch.float16).to(device)  # (k,)
                hook = _make_multi_projection_hook(D, t)
                handles.append(decoder_layers[lyr].register_forward_hook(hook))

            with torch.no_grad():
                output = hf_model.generate(**gen_kwargs)

        elif direction_vector is not None:
            steer_vec = torch.tensor(
                direction_vector, dtype=torch.float16,
            ).to(device)

            if projection_target is not None:
                # Single projection mode: pin h·v to a constant target value.
                target = projection_target

                def _steering_hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    proj = (h * steer_vec).sum(dim=-1, keepdim=True)
                    h_new = h + (target - proj) * steer_vec
                    if isinstance(output, tuple):
                        return (h_new, *output[1:])
                    return h_new
            else:
                # Additive mode: h' = h + alpha * v
                def _steering_hook(module, input, output):
                    if isinstance(output, tuple):
                        return (output[0] + alpha * steer_vec, *output[1:])
                    return output + alpha * steer_vec

            handles.append(
                _get_decoder_layers(hf_model)[layer].register_forward_hook(
                    _steering_hook,
                )
            )
            with torch.no_grad():
                output = hf_model.generate(**gen_kwargs)
        else:
            # Unsteered baseline: raw HuggingFace generate, no hook
            with torch.no_grad():
                output = hf_model.generate(**gen_kwargs)
    finally:
        for h in handles:
            h.remove()

    responses = []
    for i in range(len(prompts)):
        generated_ids = output[i, max_input_len:]
        responses.append(
            tokenizer.decode(generated_ids, skip_special_tokens=True)
        )
    return responses


# ── Steering Core ─────────────────────────────────────────────────────────────


def steer_and_generate(
    model,
    tokenizer,
    prompt: str,
    direction_vector: np.ndarray,
    layer: int,
    alpha: float,
    *,
    max_new_tokens: int = 256,
) -> str:
    """Generate text with a steering vector added at *layer* each step.

    Uses ``model.generate()`` with ``tracer.all()`` to apply the
    intervention at every autoregressive step, adding
    ``alpha * direction_vector`` to the residual stream output at the
    last sequence position.

    Parameters
    ----------
    model : nnsight.LanguageModel
        Loaded nnsight model.
    tokenizer
        HuggingFace tokenizer (used for encoding/decoding).
    prompt : str
        Fully formatted prompt string (with chat template applied).
    direction_vector : np.ndarray
        Steering direction, shape ``(d_model,)``.
    layer : int
        Which transformer layer to intervene on.
    alpha : float
        Steering strength. Positive = push toward followed_system.
    max_new_tokens : int
        Maximum tokens to generate.

    Returns
    -------
    str
        Decoded response (newly generated tokens only).
    """
    return _generate_batch(
        model, tokenizer, [prompt],
        direction_vector=direction_vector,
        layer=layer,
        alpha=alpha,
        max_new_tokens=max_new_tokens,
    )[0]


# ── Scoring ───────────────────────────────────────────────────────────────────


def score_steered_output(
    response: str,
    conflict_id: str,
    direction: str,
    instruction_args: dict | str,
) -> tuple[str, float, bool, bool]:
    """Score a steered response using Phase 0 v2 verifiers.

    Parameters
    ----------
    response : str
        Model-generated text.
    conflict_id : str
        Conflict identifier (e.g. ``"formal_vs_casual_tone"``).
    direction : str
        Prompt direction (``"a_to_b"`` or ``"b_to_a"``).
    instruction_args : dict or str
        Instruction arguments; JSON string is auto-parsed.

    Returns
    -------
    (label, confidence, sys_ok, usr_ok)
    """
    from phase0_v2.conflicts.registry import get_conflict
    from phase0_v2.src.classifiers import classify_response

    if isinstance(instruction_args, str):
        instruction_args = json.loads(instruction_args)

    conflict = get_conflict(conflict_id)
    if conflict is None:
        raise ValueError(f"Unknown conflict_id: {conflict_id!r}")

    label, confidence = classify_response(
        response, conflict, direction, instruction_args
    )

    # Re-derive sys_ok/usr_ok from label
    sys_ok = label in ("followed_system", "followed_both")
    usr_ok = label in ("followed_user", "followed_both")

    return label, confidence, sys_ok, usr_ok


# ── Per-Conflict Generation + Scoring ────────────────────────────────────────


def _build_phase0_df(df_conflict: pd.DataFrame) -> pd.DataFrame:
    """Build a results DataFrame from stored Phase 0 data (no generation).

    Copies columns from the original data and derives ``sys_ok``/``usr_ok``
    from the stored label.
    """
    df = df_conflict[
        ["conflict_id", "direction", "constraint_type",
         "response", "label", "system_prompt", "user_prompt"]
    ].copy()
    df["alpha"] = None
    df["confidence"] = 1.0
    df["sys_ok"] = df["label"].isin(["followed_system", "followed_both"])
    df["usr_ok"] = df["label"].isin(["followed_user", "followed_both"])
    return df.reset_index(drop=True)


def _generate_and_score_conflict(
    model,
    tokenizer,
    df_conflict: pd.DataFrame,
    *,
    direction_vector: np.ndarray | None = None,
    layer: int | None = None,
    alpha: float = 0.0,
    max_new_tokens: int = 512,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Generate and score all samples for one conflict_id.

    Parameters
    ----------
    model : nnsight.LanguageModel
        Loaded model.
    tokenizer
        HuggingFace tokenizer.
    df_conflict : pd.DataFrame
        All samples for one conflict_id.
    direction_vector, layer, alpha
        Steering parameters (direction/layer required when alpha != 0).
    max_new_tokens, batch_size
        Generation parameters.

    Returns
    -------
    pd.DataFrame
        Scored results with columns: conflict_id, direction, constraint_type,
        alpha, response, label, confidence, sys_ok, usr_ok, system_prompt,
        user_prompt.
    """
    from data import build_formatted_prompt

    records = list(df_conflict.itertuples(index=False))
    prompts = [
        build_formatted_prompt(tokenizer, r.system_prompt, r.user_prompt)
        for r in records
    ]

    rows: list[dict] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_records = records[start : start + batch_size]

        responses = _generate_batch(
            model, tokenizer, batch_prompts,
            direction_vector=direction_vector,
            layer=layer,
            alpha=alpha,
            max_new_tokens=max_new_tokens,
        )

        for rec, resp in zip(batch_records, responses):
            label, confidence, sys_ok, usr_ok = score_steered_output(
                resp, rec.conflict_id, rec.direction, rec.instruction_args,
            )
            rows.append({
                "conflict_id": rec.conflict_id,
                "direction": rec.direction,
                "constraint_type": rec.constraint_type,
                "alpha": alpha,
                "response": resp,
                "label": label,
                "confidence": confidence,
                "sys_ok": sys_ok,
                "usr_ok": usr_ok,
                "system_prompt": rec.system_prompt,
                "user_prompt": rec.user_prompt,
            })

    return pd.DataFrame(rows)


# ── Condition Comparison with Caching ────────────────────────────────────────


def run_condition_comparison(
    model,
    tokenizer,
    df_c: pd.DataFrame,
    steering_directions: dict[str, np.ndarray],
    layer: int,
    alphas: list[float] | dict[str, list[float]],
    *,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Run three baseline conditions + steered sweeps with per-conflict caching.

    Parameters
    ----------
    model : nnsight.LanguageModel
        Loaded model.
    tokenizer
        HuggingFace tokenizer.
    df_c : pd.DataFrame
        All Condition C samples (no subsampling).
    steering_directions : dict[str, np.ndarray]
        Named steering directions, e.g. ``{"probe": vec, "cmd_overall": vec}``.
    layer : int
        Layer to steer.
    alphas : list[float] or dict[str, list[float]]
        Nonzero alphas for steered condition.  A plain list applies the same
        alphas to every direction; a dict maps direction name to its own list.
    max_new_tokens, batch_size : int
        Generation parameters.
    output_dir : Path
        Cache directory (e.g. ``{run_dir}/steering/``).
    seed : int
        Random seed (recorded in manifest only).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``phase0``, ``unsteered``, ``unsteered_hooked``,
        ``steered_{dir_name}`` for each direction.
    """
    conflict_ids = sorted(df_c["conflict_id"].unique())
    grouped = df_c.groupby("conflict_id")
    results: dict[str, pd.DataFrame] = {}

    # Normalize alphas to per-direction dict
    if isinstance(alphas, list):
        alphas_per_dir: dict[str, list[float]] = {
            name: alphas for name in steering_directions
        }
    else:
        alphas_per_dir = alphas

    # Direction vector for the hooked-alpha-0 baseline (any direction works
    # since alpha=0 zeroes it out — we just need the hook registered).
    hook_dir_vec = next(iter(steering_directions.values()))

    # 1. Baselines: phase0 (stored), unsteered (raw HF), unsteered_hooked (hook + alpha=0)
    baselines = [
        ("phase0", None),
        ("unsteered", dict(alpha=0.0)),
        # ("unsteered_hooked", dict(
        #     alpha=0.0, direction_vector=hook_dir_vec, layer=layer,
        # )),  # same as unsteered — disabled
    ]
    for condition, gen_kwargs in baselines:
        cond_dir = output_dir / condition
        cond_dir.mkdir(parents=True, exist_ok=True)

        cached, pending = [], []
        for cid in conflict_ids:
            if (cond_dir / f"{cid}.jsonl").exists():
                cached.append(cid)
            else:
                pending.append(cid)

        print(f"{condition}: {len(cached)} cached, {len(pending)} pending")

        for cid in tqdm(pending, desc=condition):
            df_conflict = grouped.get_group(cid)
            if condition == "phase0":
                df_result = _build_phase0_df(df_conflict)
            else:
                df_result = _generate_and_score_conflict(
                    model, tokenizer, df_conflict,
                    max_new_tokens=max_new_tokens, batch_size=batch_size,
                    **gen_kwargs,
                )
            df_result.to_json(
                cond_dir / f"{cid}.jsonl", orient="records", lines=True,
            )

        # Load all (cached + just-computed) and concatenate
        all_dfs = [
            pd.read_json(cond_dir / f"{cid}.jsonl", orient="records", lines=True)
            for cid in conflict_ids
        ]
        results[condition] = pd.concat(all_dfs, ignore_index=True)

    # 2. Steered conditions — per direction (with per-direction alphas)
    for dir_name, dir_vec in steering_directions.items():
        dir_dir = output_dir / "steered" / dir_name
        dir_dir.mkdir(parents=True, exist_ok=True)
        dir_alphas = alphas_per_dir[dir_name]

        cached, pending = [], []
        for alpha in dir_alphas:
            for cid in conflict_ids:
                fname = f"{cid}_alpha_{alpha}.jsonl"
                if (dir_dir / fname).exists():
                    cached.append((alpha, cid))
                else:
                    pending.append((alpha, cid))

        print(f"steered/{dir_name}: {len(cached)} cached, {len(pending)} pending")

        for alpha, cid in tqdm(pending, desc=f"steered/{dir_name}"):
            df_conflict = grouped.get_group(cid)
            df_result = _generate_and_score_conflict(
                model, tokenizer, df_conflict,
                direction_vector=dir_vec,
                layer=layer, alpha=alpha,
                max_new_tokens=max_new_tokens, batch_size=batch_size,
            )
            df_result.to_json(
                dir_dir / f"{cid}_alpha_{alpha}.jsonl",
                orient="records", lines=True,
            )

        # Load all and concatenate
        all_dfs = []
        for alpha in dir_alphas:
            for cid in conflict_ids:
                fname = f"{cid}_alpha_{alpha}.jsonl"
                all_dfs.append(
                    pd.read_json(dir_dir / fname, orient="records", lines=True)
                )
        results[f"steered_{dir_name}"] = pd.concat(all_dfs, ignore_index=True)

    return results


def save_experiment_manifest(
    output_dir: Path,
    *,
    seed: int,
    model_name: str,
    conflict_ids: list[str],
    direction_names: list[str],
    alphas: list[float] | dict[str, list[float]],
    max_new_tokens: int,
    batch_size: int,
    layer: int,
) -> Path:
    """Write ``manifest.json`` with experiment config and file inventory.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_sha = "unknown"

    # Inventory all JSONL files under output_dir
    files = sorted(
        str(p.relative_to(output_dir)) for p in output_dir.rglob("*.jsonl")
    )

    manifest = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "git_sha": git_sha,
        "seed": seed,
        "model_name": model_name,
        "conflict_ids": conflict_ids,
        "direction_names": direction_names,
        "alphas": alphas,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "layer": layer,
        "files": files,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


# ── Sweep ─────────────────────────────────────────────────────────────────────


def run_steering_sweep(
    model,
    tokenizer,
    df_samples: pd.DataFrame,
    direction_vector: np.ndarray,
    layer: int,
    alphas: list[float],
    *,
    max_new_tokens: int = 256,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Run steering across multiple alpha values and samples.

    Parameters
    ----------
    model : nnsight.LanguageModel
        Loaded model.
    tokenizer
        HuggingFace tokenizer.
    df_samples : pd.DataFrame
        Condition C samples with columns: ``conflict_id``, ``direction``,
        ``instruction_args``, ``system_prompt``, ``user_prompt``,
        ``constraint_type``.
    direction_vector : np.ndarray
        Steering direction, shape ``(d_model,)``.
    layer : int
        Layer to steer.
    alphas : list[float]
        Steering strengths to sweep.
    max_new_tokens : int
        Max generation length.

    Returns
    -------
    pd.DataFrame
        Columns: ``conflict_id``, ``direction``, ``alpha``, ``response``,
        ``label``, ``confidence``, ``sys_ok``, ``usr_ok``,
        ``system_prompt``, ``user_prompt``, ``constraint_type``.
    """
    from data import build_formatted_prompt  # noqa: phase1_linear_probing/data.py

    rows: list[dict] = []
    samples_list = list(df_samples.itertuples(index=False))
    prompts = [
        build_formatted_prompt(tokenizer, s.system_prompt, s.user_prompt)
        for s in samples_list
    ]
    total = len(alphas) * len(samples_list)

    with tqdm(total=total, desc="Steering sweep") as pbar:
        for alpha in alphas:
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                batch_samples = samples_list[start : start + batch_size]

                responses = _generate_batch(
                    model, tokenizer, batch_prompts,
                    direction_vector=direction_vector,
                    layer=layer,
                    alpha=alpha,
                    max_new_tokens=max_new_tokens,
                )

                for sample, response in zip(batch_samples, responses):
                    label, confidence, sys_ok, usr_ok = score_steered_output(
                        response,
                        sample.conflict_id,
                        sample.direction,
                        sample.instruction_args,
                    )

                    rows.append({
                        "conflict_id": sample.conflict_id,
                        "direction": sample.direction,
                        "constraint_type": sample.constraint_type,
                        "alpha": alpha,
                        "response": response,
                        "label": label,
                        "confidence": confidence,
                        "sys_ok": sys_ok,
                        "usr_ok": usr_ok,
                        "system_prompt": sample.system_prompt,
                        "user_prompt": sample.user_prompt,
                    })

                pbar.update(len(batch_samples))

    return pd.DataFrame(rows)


# ── SCR Computation ──────────────────────────────────────────────────────────


def compute_steered_scr(
    sweep_df: pd.DataFrame,
    alpha: float | None = None,
) -> float | pd.Series:
    """Compute System Compliance Rate from sweep results.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output of :func:`run_steering_sweep`.
    alpha : float or None
        If given, return SCR for that alpha only.
        If None, return SCR grouped by alpha as a Series.

    Returns
    -------
    float or pd.Series
        SCR = fraction of responses labeled ``followed_system``.
    """
    if alpha is not None:
        sub = sweep_df[sweep_df["alpha"] == alpha]
        if len(sub) == 0:
            raise ValueError(f"No rows for alpha={alpha}")
        return (sub["label"] == "followed_system").mean()

    return sweep_df.groupby("alpha")["label"].apply(
        lambda s: (s == "followed_system").mean()
    ).rename("scr")
