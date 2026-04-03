"""Quick steering test: 48 followed_user samples × high alphas.

Pick 6 samples per (conflict, direction) that got followed_user in unsteered.
Steer with probe and CMD at high alphas, print label after each pass.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

_phase1_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_phase1_dir))
sys.path.insert(0, str(_phase1_dir.parent))

from data import (
    ProbeConfig,
    _cleanup_nn_model,
    _load_nn_model,
    load_results as load_data,
    load_sync_env,
    prepare_condition_c,
)
from steer import (
    _generate_batch,
    load_steering_directions,
    score_steered_output,
)

CONFLICT_IDS = [
    "json_only_vs_plain",
    "list_bullets_vs_numbered",
    "past_vs_present_tense",
    "starting_word_hello_greetings",
]
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 25
N_PER_GROUP = 6  # per (conflict, direction)

ALPHA_PRESETS = {
    16: {"probe": [1, 2, 5, 10], "cmd": [2, 5, 10, 20]},
    25: {"probe": [2, 5, 10, 20], "cmd": [10, 20, 50, 100]},
}
_preset = ALPHA_PRESETS.get(LAYER, ALPHA_PRESETS[25])
PROBE_ALPHAS = _preset["probe"]
CMD_ALPHAS = _preset["cmd"]
MAX_NEW_TOKENS = 512


def select_samples(df_c: pd.DataFrame) -> pd.DataFrame:
    """Pick N_PER_GROUP followed_user samples per (conflict, direction)."""
    # Load unsteered results to know which samples got followed_user
    steer_dir = _phase1_dir / "data" / "runs" / "curated4-8b-v001" / "steering"
    unsteered_labels = {}
    for f in (steer_dir / "unsteered").glob("*.jsonl"):
        cid = f.stem
        with open(f) as fh:
            for i, line in enumerate(fh):
                row = json.loads(line)
                unsteered_labels[(cid, i)] = row["label"]

    # Assign unsteered labels back to df_c (same order — sorted by conflict_id)
    df_c = df_c.copy()
    df_c["_idx_in_conflict"] = df_c.groupby("conflict_id").cumcount()
    df_c["unsteered_label"] = df_c.apply(
        lambda r: unsteered_labels.get((r["conflict_id"], r["_idx_in_conflict"])),
        axis=1,
    )

    # Filter to followed_user, sample N per (conflict, direction)
    fu = df_c[df_c["unsteered_label"] == "followed_user"]
    samples = (
        fu.groupby(["conflict_id", "direction"])
        .head(N_PER_GROUP)
        .reset_index(drop=True)
    )
    print(f"Selected {len(samples)} samples:")
    print(samples.groupby(["conflict_id", "direction"]).size().to_string())
    return samples


def build_prompts(df: pd.DataFrame, tokenizer) -> list[str]:
    """Build chat prompts from dataframe rows."""
    prompts = []
    for _, row in df.iterrows():
        messages = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_prompt"]},
        ]
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return prompts


def run_and_score(
    model, tokenizer, df: pd.DataFrame, prompts: list[str],
    direction_vector, layer, alpha, dir_name,
):
    """Generate with steering and score, return label series."""
    responses = _generate_batch(
        model, tokenizer, prompts,
        direction_vector=direction_vector,
        layer=layer,
        alpha=alpha,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    labels = []
    for i, resp in enumerate(responses):
        row = df.iloc[i]
        label, conf, _, _ = score_steered_output(
            resp, row["conflict_id"], row["direction"], row["instruction_args"],
        )
        labels.append(label)

    n_sys = sum(1 for l in labels if l == "followed_system")
    n_usr = sum(1 for l in labels if l == "followed_user")
    n_both = sum(1 for l in labels if l == "followed_both")
    n_neither = sum(1 for l in labels if l == "followed_neither")
    print(
        f"  {dir_name} alpha={alpha:>5g}:  "
        f"sys={n_sys:2d}  usr={n_usr:2d}  both={n_both:2d}  neither={n_neither:2d}  "
        f"SCR={n_sys/len(labels):.3f}"
    )

    # Per-conflict breakdown
    df_tmp = df.copy()
    df_tmp["steered_label"] = labels
    df_tmp["response_steered"] = responses
    for cid in CONFLICT_IDS:
        mask = df_tmp["conflict_id"] == cid
        sub = df_tmp[mask]
        if len(sub) == 0:
            continue
        n_s = (sub["steered_label"] == "followed_system").sum()
        n_u = (sub["steered_label"] == "followed_user").sum()
        # Show first response snippet if any flipped
        flipped = sub[sub["steered_label"] == "followed_system"]
        snippet = ""
        if len(flipped) > 0:
            snippet = f'  "{flipped.iloc[0]["response_steered"][:60]}..."'
        print(
            f"    {cid[:30]:30s}  sys={n_s}/{len(sub)}  usr={n_u}/{len(sub)}{snippet}"
        )

    return labels


def main():
    cfg = ProbeConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        label_mode="binary",
        token_positions=["last_prompt"],
        cv_mode="grouped",
        n_cv_folds=4,
        use_scaler=False,
        probe_C=1.0,
        batch_size=1,
        conflict_ids=CONFLICT_IDS,
        run_id="curated4-8b-v001",
    )
    load_sync_env(cfg.repo_root)
    cfg.ensure_dirs()

    # Load data
    df_all = load_data(cfg.data_dir, cfg.model_name)
    df_c = prepare_condition_c(df_all, cfg.label_mode, conflict_ids=cfg.conflict_ids)
    df_c = df_c.sort_values("conflict_id").reset_index(drop=True)

    # Select 48 followed_user samples
    df_test = select_samples(df_c)

    # Load directions
    directions = load_steering_directions(cfg.run_dir, "last_prompt", LAYER)

    # Load model
    model_nn, _ = _load_nn_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Build prompts
    prompts = build_prompts(df_test, tokenizer)

    try:
        # Unsteered baseline on this subset
        print("\n" + "=" * 70)
        print("UNSTEERED BASELINE (48 followed_user samples)")
        print("=" * 70)
        run_and_score(model_nn, tokenizer, df_test, prompts,
                      direction_vector=None, layer=None, alpha=0, dir_name="unsteered")

        # Probe direction
        print("\n" + "=" * 70)
        print("PROBE DIRECTION — high alphas")
        print("=" * 70)
        for alpha in PROBE_ALPHAS:
            run_and_score(model_nn, tokenizer, df_test, prompts,
                          direction_vector=directions["probe"], layer=LAYER,
                          alpha=alpha, dir_name="probe")

        # CMD direction
        print("\n" + "=" * 70)
        print("CMD DIRECTION — high alphas")
        print("=" * 70)
        for alpha in CMD_ALPHAS:
            run_and_score(model_nn, tokenizer, df_test, prompts,
                          direction_vector=directions["cmd_overall"], layer=LAYER,
                          alpha=alpha, dir_name="cmd")

    finally:
        _cleanup_nn_model(model_nn, cfg.device)
        del model_nn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
