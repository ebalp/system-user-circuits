"""Activation steering: add probe/CMD directions during generation.

Provides helpers to load steering directions from probe results,
steer model generation via nnsight, score steered outputs using
Phase 0 v2 verifiers, and run parameter sweeps over steering strength.
"""

from __future__ import annotations

import json
import logging
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
    results = load_results(rpath)
    pr = results[pos_name]

    directions: dict[str, np.ndarray] = {
        "probe": pr.weights[layer],
        "probe_raw": pr.weights_raw[layer],
    }

    # Fold CMDs
    fold_cmds_path = run_dir / f"fold_cmds_L{layer}.npz"
    if fold_cmds_path.exists():
        fold_data = np.load(fold_cmds_path)
        vecs = np.stack([fold_data[k] for k in fold_data.files])
        mean_cmd = vecs.mean(axis=0)
        norm = np.linalg.norm(mean_cmd)
        directions["cmd_overall"] = mean_cmd / norm if norm > 0 else mean_cmd
    else:
        logger.warning("No fold CMDs at %s — cmd_overall unavailable", fold_cmds_path)

    # Per-constraint CMDs
    constraint_cmds_path = run_dir / f"constraint_cmds_L{layer}.npz"
    if constraint_cmds_path.exists():
        cdata = np.load(constraint_cmds_path)
        directions["cmd_per_constraint"] = {k: cdata[k] for k in cdata.files}
    else:
        logger.warning("No constraint CMDs at %s", constraint_cmds_path)

    return directions


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
    # Prepare steering vector
    device = next(model.parameters()).device
    steer_vec = torch.tensor(
        direction_vector, dtype=torch.float16
    ).to(device)

    encoded = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    n_prompt = input_ids.shape[1]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    output = None
    with model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    ) as tracer:
        if alpha != 0:
            with tracer.all():
                # Add steering vector to all positions in the layer output.
                # During KV-cached decode steps there is only 1 position;
                # during prefill we steer all positions (broadcast handles
                # both 2D and 3D hidden states).
                model.model.layers[layer].output[0][:] += alpha * steer_vec
        output = model.generator.output.save()

    if output is None:
        raise RuntimeError("nnsight generate failed — output proxy not saved")

    generated_ids = output[0, n_prompt:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


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
    total = len(alphas) * len(df_samples)

    with tqdm(total=total, desc="Steering sweep") as pbar:
        for alpha in alphas:
            for _, sample in df_samples.iterrows():
                prompt = build_formatted_prompt(
                    tokenizer,
                    sample["system_prompt"],
                    sample["user_prompt"],
                )

                response = steer_and_generate(
                    model, tokenizer, prompt,
                    direction_vector, layer, alpha,
                    max_new_tokens=max_new_tokens,
                )

                label, confidence, sys_ok, usr_ok = score_steered_output(
                    response,
                    sample["conflict_id"],
                    sample["direction"],
                    sample["instruction_args"],
                )

                rows.append({
                    "conflict_id": sample["conflict_id"],
                    "direction": sample["direction"],
                    "constraint_type": sample["constraint_type"],
                    "alpha": alpha,
                    "response": response,
                    "label": label,
                    "confidence": confidence,
                    "sys_ok": sys_ok,
                    "usr_ok": usr_ok,
                    "system_prompt": sample["system_prompt"],
                    "user_prompt": sample["user_prompt"],
                })

                pbar.update(1)

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
