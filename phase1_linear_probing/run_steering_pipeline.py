"""Full steering pipeline with round-robin conflict interleaving.

Processes one batch (default 48 samples) per conflict through all conditions
(unsteered + steered at each alpha), then moves to the next conflict.  After
one full round you have partial SCR for every conflict.

Results are appended incrementally to per-conflict JSONL files in
{run_dir}/steering/, so restarts resume from where they left off.

Usage:
  uv run python phase1_linear_probing/run_steering_pipeline.py
  uv run python phase1_linear_probing/run_steering_pipeline.py --steer-only-user
  uv run python phase1_linear_probing/run_steering_pipeline.py --batch-size 24
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

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
    _build_phase0_df,
    _generate_batch,
    load_steering_directions,
    save_experiment_manifest,
    score_steered_output,
)

# ── Config ────────────────────────────────────────────────────────────────────
PROBE_CONFLICTS_35 = [
    "starting_word_hello_greetings",
    "list_bullets_vs_numbered",
    "json_only_vs_plain",
    "keyword_avoidance",
    "imperative_vs_declarative",
    "first_vs_third_person",
    "language_en_zh",
    "number_density",
    "each_word_new_line",
    "bullets_and_sub_bullets",
    "language_en_es",
    "spanish_loanwords",
    "response_length",
    "capitalization_all_caps",
    "format_json_markdown",
    "address_reader_directly",
    "alphabetical_sentences",
    "paragraph_start_word",
    "self_reference_ai_mention",
    "leetspeak_encoding",
    "html_emphasis_tags",
    "short_paragraphs_vs_single_block",
    "sentence_connector_density",
    "disclaimer_first_vs_none",
    "formal_vs_casual_tone",
    "short_vs_long_sentences",
    "numbered_sections_vs_prose",
    "template_response",
    "questions_vs_statements",
    "lowercase_vs_capitalized",
    "past_vs_present_tense",
    "emoji_use_vs_avoid",
    "direct_answer_vs_hedging",
    "pronoun_density",
    "parenthetical_asides",
]

SEED = 42
MAX_NEW_TOKENS = 512
STEER_LAYER = 25


# ── Helpers ──────────────────────────────────────────────────────────────────


def _count_jsonl_rows(path: Path) -> int:
    """Count lines in a JSONL file (0 if missing)."""
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def _generate_and_score_batch(
    model, tokenizer, records, prompts,
    *, direction_vector=None, layer=None, alpha=0.0,
    projection_target=None, max_new_tokens=512,
) -> list[dict]:
    """Generate and score a single batch. Returns list of row dicts."""
    responses = _generate_batch(
        model, tokenizer, prompts,
        direction_vector=direction_vector,
        layer=layer,
        alpha=alpha,
        projection_target=projection_target,
        max_new_tokens=max_new_tokens,
    )
    # For labeling: use projection_target as alpha if in projection mode
    alpha_label = f"proj{projection_target}" if projection_target is not None else alpha
    rows = []
    for rec, resp in zip(records, responses):
        label, confidence, sys_ok, usr_ok = score_steered_output(
            resp, rec.conflict_id, rec.direction, rec.instruction_args,
        )
        rows.append({
            "conflict_id": rec.conflict_id,
            "direction": rec.direction,
            "constraint_type": rec.constraint_type,
            "alpha": alpha_label,
            "response": resp,
            "label": label,
            "confidence": confidence,
            "sys_ok": sys_ok,
            "usr_ok": usr_ok,
            "system_prompt": rec.system_prompt,
            "user_prompt": rec.user_prompt,
        })
    return rows


def _append_rows(path: Path, rows: list[dict]):
    """Append rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_json(path, orient="records", lines=True,
               mode="a" if path.exists() else "w")


def _print_progress(steer_dir: Path, conflict_ids: list[str],
                     conditions: list[tuple[str, Path]],
                     n_total_per_conflict: dict[str, int]):
    """Print current SCR across all conflicts and conditions."""
    print(f"\n{'conflict_id':40s}", end="")
    for cond_name, _ in conditions:
        print(f"  {cond_name:>12s}", end="")
    print(f"  {'n':>5s}")
    print("-" * (40 + 14 * len(conditions) + 7))

    for cid in sorted(conflict_ids):
        print(f"  {cid:40s}", end="")
        for cond_name, cond_path in conditions:
            fpath = cond_path / f"{cid}.jsonl"
            if fpath.exists():
                df = pd.read_json(fpath, orient="records", lines=True)
                scr = (df["label"] == "followed_system").mean()
                print(f"  {scr:>11.3f}", end="")
            else:
                print(f"  {'---':>12s}", end="")
        # n completed (from unsteered as reference)
        unsteered_path = steer_dir / "unsteered" / f"{cid}.jsonl"
        n = _count_jsonl_rows(unsteered_path)
        total = n_total_per_conflict[cid]
        print(f"  {n:>3d}/{total}")


def main():
    parser = argparse.ArgumentParser(description="Interleaved steering pipeline")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--steer-only-user", action="store_true",
                        help="Only steer samples labeled followed_user in phase0")
    parser.add_argument("--steer-layer", type=int, default=STEER_LAYER)
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.5, 1, 1.5],
                        help="Additive steering strengths (default: 0.5 1 1.5)")
    parser.add_argument("--projection-targets", type=float, nargs="*", default=[],
                        help="Projection targets — pin h·v to these values (default: none)")
    parser.add_argument("--no-additive", action="store_true",
                        help="Skip additive steering conditions")
    parser.add_argument("--no-projection", action="store_true",
                        help="Skip projection steering conditions")
    args = parser.parse_args()

    batch_size = args.batch_size
    steer_only_user = args.steer_only_user

    cfg = ProbeConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        label_mode="binary",
        token_positions=["last_prompt"],
        cv_mode="grouped",
        n_cv_folds=10,
        use_scaler=False,
        probe_C=1.0,
        batch_size=1,
        conflict_ids=PROBE_CONFLICTS_35,
        run_id="curated35-8b-v001",
    )
    load_sync_env(cfg.repo_root)
    cfg.ensure_dirs()

    steer_layer = args.steer_layer

    print(f"Run ID           : {cfg.run_id}")
    print(f"Run dir          : {cfg.run_dir}")
    print(f"Steer layer      : {steer_layer}")
    print(f"Batch size       : {batch_size}")
    print(f"Steer only user  : {steer_only_user}")
    print(f"Alphas           : {args.alphas}")
    print(f"Proj targets     : {args.projection_targets}")

    # ── Load steering directions ──────────────────────────────────────────
    pos = cfg.token_positions[0]
    directions = load_steering_directions(cfg.run_dir, pos, steer_layer)
    steer_directions = {}
    if "probe" in directions:
        steer_directions["probe"] = directions["probe"]
    # if "cmd_overall" in directions:
    #     steer_directions["cmd_overall"] = directions["cmd_overall"]
    print(f"Directions       : {list(steer_directions.keys())}")

    # Same alphas/targets for all directions
    alphas_per_dir: dict[str, list[float]] = {
        name: args.alphas for name in steer_directions
    }
    proj_per_dir: dict[str, list[float]] = {
        name: args.projection_targets for name in steer_directions
    }

    # ── Load Condition C samples ──────────────────────────────────────────
    df_all = load_results(cfg.data_dir, cfg.model_name)
    df_c = prepare_condition_c(
        df_all, cfg.label_mode, conflict_ids=cfg.conflict_ids
    )
    df_c = df_c.sort_values("conflict_id").reset_index(drop=True)
    conflict_ids = sorted(df_c["conflict_id"].unique())

    print(
        f"Condition C      : {len(df_c)} samples across "
        f"{len(conflict_ids)} conflicts"
    )
    print(f"Phase 0 SCR      : {(df_c['label'] == 'followed_system').mean():.3f}")

    # ── Phase 0 baselines (no generation needed) ─────────────────────────
    steer_dir = cfg.run_dir / "steering"
    phase0_dir = steer_dir / "phase0"
    phase0_dir.mkdir(parents=True, exist_ok=True)
    for cid in conflict_ids:
        fpath = phase0_dir / f"{cid}.jsonl"
        if not fpath.exists():
            df_conflict = df_c[df_c["conflict_id"] == cid]
            _build_phase0_df(df_conflict).to_json(
                fpath, orient="records", lines=True,
            )
    print("Phase 0 baselines: cached")

    # ── Prepare per-conflict batches ──────────────────────────────────────
    grouped = df_c.groupby("conflict_id")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Pre-build records and prompts per conflict
    conflict_data: dict[str, list] = {}
    n_total_per_conflict: dict[str, int] = {}
    for cid in conflict_ids:
        df_conflict = grouped.get_group(cid)
        records = list(df_conflict.itertuples(index=False))
        prompts = [
            build_formatted_prompt(tokenizer, r.system_prompt, r.user_prompt)
            for r in records
        ]
        conflict_data[cid] = (records, prompts, df_conflict)
        n_total_per_conflict[cid] = len(records)

    # Build list of (condition_name, output_dir, gen_kwargs) for each condition
    # Steered conditions are per (direction, alpha)
    # Use hook with alpha=0 for unsteered (same result, avoids hook churn)
    first_dir_vec = next(iter(steer_directions.values()))
    conditions: list[tuple[str, Path, dict]] = [
        ("unsteered", steer_dir / "unsteered", dict(
            direction_vector=first_dir_vec, layer=steer_layer, alpha=0.0,
        )),
    ]
    # Additive steering: h' = h + alpha * v
    if not args.no_additive:
        for dir_name, dir_vec in steer_directions.items():
            for alpha in alphas_per_dir.get(dir_name, []):
                cond_name = f"{dir_name}_a{alpha}"
                cond_dir = steer_dir / "steered" / dir_name
                conditions.append((cond_name, cond_dir, dict(
                    direction_vector=dir_vec, layer=steer_layer, alpha=alpha,
                )))
    # Projection steering: pin h·v to target value
    if not args.no_projection:
        for dir_name, dir_vec in steer_directions.items():
            for target in proj_per_dir.get(dir_name, []):
                cond_name = f"{dir_name}_proj{target}"
                cond_dir = steer_dir / "projected" / dir_name
                conditions.append((cond_name, cond_dir, dict(
                    direction_vector=dir_vec, layer=steer_layer,
                    projection_target=target,
                )))

    print(f"Conditions       : {[c[0] for c in conditions]}")

    # Determine how many batches per conflict (max across all)
    max_samples = max(n_total_per_conflict.values())
    n_rounds = math.ceil(max_samples / batch_size)
    print(f"Rounds           : {n_rounds} (batches of {batch_size})")

    # ── Load model ────────────────────────────────────────────────────────
    model_nn, _n_layers = _load_nn_model(cfg)

    # ── Round-robin loop ──────────────────────────────────────────────────
    t0 = time.time()
    total_generated = 0

    try:
        for round_idx in range(n_rounds):
            round_t0 = time.time()
            round_generated = 0

            for cid in conflict_ids:
                records_all, prompts_all, df_conflict = conflict_data[cid]
                n_total = len(records_all)
                start = round_idx * batch_size
                if start >= n_total:
                    continue
                end = min(start + batch_size, n_total)

                # Build the batch once — same samples for all conditions
                batch_records = records_all[start:end]
                batch_prompts = prompts_all[start:end]

                if steer_only_user:
                    filtered = [
                        (r, p) for r, p in zip(batch_records, batch_prompts)
                        if r.label == "followed_user"
                    ]
                    if not filtered:
                        continue
                    batch_records, batch_prompts = zip(*filtered)
                    batch_records = list(batch_records)
                    batch_prompts = list(batch_prompts)

                for cond_name, cond_dir, gen_kwargs in conditions:
                    # Determine output file and check resume
                    has_projection = gen_kwargs.get("projection_target") is not None
                    has_additive = gen_kwargs.get("alpha", 0.0) != 0.0

                    if has_projection:
                        fname = f"{cid}_proj_{gen_kwargs['projection_target']}.jsonl"
                    elif has_additive:
                        fname = f"{cid}_alpha_{gen_kwargs['alpha']}.jsonl"
                    else:
                        fname = f"{cid}.jsonl"

                    fpath = cond_dir / fname
                    existing = _count_jsonl_rows(fpath)
                    if existing >= end:
                        continue  # already done this batch

                    cond_dir.mkdir(parents=True, exist_ok=True)
                    rows = _generate_and_score_batch(
                        model_nn, tokenizer, batch_records, batch_prompts,
                        max_new_tokens=MAX_NEW_TOKENS, **gen_kwargs,
                    )
                    _append_rows(fpath, rows)
                    round_generated += len(rows)

            total_generated += round_generated
            elapsed = time.time() - round_t0
            total_elapsed = time.time() - t0
            print(
                f"Round {round_idx + 1}/{n_rounds}: "
                f"{round_generated} samples in {elapsed:.0f}s "
                f"(total: {total_generated}, {total_elapsed / 60:.1f}min)"
            )

            # Print progress table every 5 rounds
            if (round_idx + 1) % 5 == 0 or round_idx == n_rounds - 1:
                cond_display = [
                    ("phase0", phase0_dir),
                    *[(c[0], c[1]) for c in conditions],
                ]
                _print_progress(
                    steer_dir, conflict_ids, cond_display, n_total_per_conflict,
                )

        # ── Save manifest ─────────────────────────────────────────────────
        save_experiment_manifest(
            steer_dir,
            seed=SEED,
            model_name=cfg.model_name,
            conflict_ids=list(cfg.conflict_ids),
            direction_names=list(steer_directions.keys()),
            alphas={name: args.alphas for name in steer_directions},
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=batch_size,
            layer=steer_layer,
        )

        total_elapsed = time.time() - t0
        print(f"\nDONE — {total_generated} samples in {total_elapsed / 60:.1f}min")
        print(f"Results cached in {steer_dir}")

    finally:
        _cleanup_nn_model(model_nn, cfg.device)
        del model_nn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
