#!/usr/bin/env python3
"""
extract_activations.py — Activation extraction for conflict resolution probing.

Reads a model's results.jsonl, filters to Condition C binary-label samples,
runs forward passes with nnsight to extract residual stream activations at
specified token positions, and saves everything to disk.

Usage:
    # Dry-run: inspect dataset without loading model
    python extract_activations.py --model meta-llama/Llama-3.1-8B-Instruct --data-dir ./data --dry-run

    # Extract one model
    python extract_activations.py --model meta-llama/Llama-3.1-8B-Instruct --data-dir ./data --out-dir ./activations

    # Extract all supported models sequentially
    python extract_activations.py --all --data-dir ./data --out-dir ./activations

    # Custom positions
    python extract_activations.py --model ... --positions last_prompt last_system first_generated last_generated
"""

import argparse
import json
import gc
import os
import sys
import time
import traceback
from pathlib import Path

# ── Default HF token ──
HF_TOKEN = os.environ.get("HF_TOKEN", "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")
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Specification for a supported model."""
    hf_name: str                          # HuggingFace model ID
    results_stem: str                     # filename stem in data dir (without _results.jsonl)
    layer_path: str = "model.model.layers"  # nnsight path to decoder layers
    device_map: str = "auto"              # device_map for nnsight
    dtype: torch.dtype = torch.bfloat16

    @property
    def safe_name(self) -> str:
        return self.hf_name.replace("/", "_")

SUPPORTED_MODELS = {
    "gemma-27b": ModelSpec(
        hf_name="google/gemma-3-27b-it",
        results_stem="google_gemma-3-27b-it",
        # Gemma3 is multimodal: model.language_model.model.layers
        layer_path="model.model.language_model.layers",
    ),
    #"llama-70b": ModelSpec(
    #    hf_name="meta-llama/Llama-3.3-70B-Instruct",
    #    results_stem="meta-llama_Llama-3.3-70B-Instruct",
    #    layer_path="model.model.layers",
    #),
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & filtering
# ─────────────────────────────────────────────────────────────────────────────

def load_results(data_dir: Path, results_stem: str) -> pd.DataFrame:
    """Load results.jsonl into a DataFrame."""
    path = data_dir / f"{results_stem}_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    records, bad_lines = [], []
    decoder = json.JSONDecoder()
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(decoder.decode(line))
            except json.JSONDecodeError:
                bad_lines.append(i)

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from {path.name}")
    if bad_lines:
        print(f"  Warning: {len(bad_lines)} bad lines skipped")
    return df


def filter_condition_c_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Condition C with binary labels, then balance to 50/50."""
    df_c = df[df["condition"] == "C"].copy()
    df_c = df_c[df_c["label"].isin(["followed_system", "followed_user"])].copy()
    df_c["y"] = (df_c["label"] == "followed_system").astype(int)

    # Balance: downsample majority class to match minority
    n_sys = (df_c["y"] == 1).sum()
    n_usr = (df_c["y"] == 0).sum()
    n_per_class = min(n_sys, n_usr)

    if n_per_class == 0:
        print(f"  ⚠ One class is empty (system={n_sys}, user={n_usr})")
        return df_c.reset_index(drop=True)

    rng = np.random.RandomState(42)
    sys_idx = rng.choice(df_c[df_c["y"] == 1].index, size=n_per_class, replace=False)
    usr_idx = rng.choice(df_c[df_c["y"] == 0].index, size=n_per_class, replace=False)
    df_c = df_c.loc[np.sort(np.concatenate([sys_idx, usr_idx]))].copy()

    print(f"  Balanced: {n_per_class} per class ({n_per_class*2} total) "
          f"from {n_sys} sys + {n_usr} usr")

    return df_c.reset_index(drop=True)


def print_dry_run_report(df_all: pd.DataFrame, df: pd.DataFrame):
    """Print dataset statistics without loading model."""
    print("\n" + "=" * 70)
    print("DRY RUN REPORT")
    print("=" * 70)

    print(f"\nTotal records: {len(df_all)}")
    print(f"Conditions: {df_all['condition'].value_counts().to_dict()}")

    print(f"\nCondition C binary-label samples: {len(df)}")
    print(f"  followed_system: {df['y'].sum()} ({df['y'].mean():.1%})")
    print(f"  followed_user:   {(1 - df['y']).sum().astype(int)} ({1 - df['y'].mean():.1%})")

    print(f"\n{'─' * 70}")
    print(f"{'constraint_type':<45} {'n':>6}  {'SCR':>6}")
    print(f"{'─' * 70}")
    scr = df.groupby("constraint_type").agg(
        n=("y", "count"),
        scr=("y", "mean"),
    ).sort_values("scr", ascending=False)
    for ct, row in scr.iterrows():
        bar = "█" * int(row["scr"] * 20)
        print(f"  {ct:<43} {row['n']:>6}  {row['scr']:.3f}  {bar}")

    print(f"\n{'─' * 70}")
    print("SCR by system_style:")
    for style, grp in df.groupby("system_style"):
        print(f"  {style:<20} n={len(grp):>6}  SCR={grp['y'].mean():.3f}")

    print(f"\nSCR by user_style:")
    for style, grp in df.groupby("user_style"):
        print(f"  {style:<20} n={len(grp):>6}  SCR={grp['y'].mean():.3f}")

    # Positive sample feasibility
    pos_counts = df[df["y"] == 1].groupby("constraint_type").size().sort_values(ascending=False)
    zero_pos = set(df["constraint_type"].unique()) - set(pos_counts.index)
    print(f"\nConstraint types with 0 positive samples: {len(zero_pos)}")
    if zero_pos:
        for ct in sorted(zero_pos):
            print(f"  ⚠ {ct}")

    print(f"\n{'=' * 70}")
    print("Dry run complete. No model loaded, no activations extracted.")
    print(f"{'=' * 70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatting & position finding
# ─────────────────────────────────────────────────────────────────────────────

def build_formatted_prompt(tokenizer, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def find_input_positions(tokenizer, system_text: str, user_text: str) -> dict:
    """Find token positions for system, user, and prompt boundaries."""
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    encode_len = lambda text: len(tokenizer.encode(text, add_special_tokens=False))

    full_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sys_str = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text}],
        tokenize=False, add_generation_prompt=False,
    )
    sys_user_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )

    n_full = encode_len(full_str)
    n_sys = min(encode_len(sys_str), n_full - 1)
    n_sys_user = min(encode_len(sys_user_str), n_full - 1)
    n_sys = max(n_sys, 1)
    n_sys_user = max(n_sys_user, n_sys + 1)

    return {
        "last_prompt": n_full - 1,
        "last_system": n_sys - 1,
        "last_user": n_sys_user - 1,
        "n_full": n_full,
        "n_sys": n_sys,
        "n_sys_user": n_sys_user,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Activation extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_layer_accessor(model, spec: ModelSpec):
    """
    Return a function that accesses layer outputs via nnsight.
    Handles different model architectures:
      - LLaMA/Qwen/Gemma: model.model.layers[i]
      - GPT-NeoX:         model.gpt_neox.layers[i]
    """
    parts = spec.layer_path.split(".")
    obj = model
    # Skip the first 'model' since nnsight wraps it
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj  # this is the ModuleList of layers


def extract_single_model(
    spec: ModelSpec,
    data_dir: Path,
    out_dir: Path,
    positions: list[str],
    hf_token: str | None = None,
):
    """Extract activations for a single model. Saves incrementally."""
    print(f"\n{'=' * 70}")
    print(f"Extracting: {spec.hf_name}")
    print(f"{'=' * 70}")

    # ── Load data ──
    df_all = load_results(data_dir, spec.results_stem)
    df = filter_condition_c_binary(df_all)
    print(f"Filtered to {len(df)} Condition C binary-label samples")
    print(f"  SCR = {df['y'].mean():.3f}")

    if len(df) == 0:
        print("  ⚠ No samples after filtering. Skipping.")
        return

    # ── Set up tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_name,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Prepare prompts & positions ──
    print("Preparing prompts...")
    formatted_prompts = []
    input_ids_list = []
    position_maps = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        fp = build_formatted_prompt(tokenizer, row["system_prompt"], row["user_prompt"])
        ids = tokenizer(fp, return_tensors="pt", add_special_tokens=False).input_ids
        pm = find_input_positions(tokenizer, row["system_prompt"], row["user_prompt"])
        formatted_prompts.append(fp)
        input_ids_list.append(ids)
        position_maps.append(pm)

    # ── Load model ──
    print(f"Loading model (device_map={spec.device_map}, dtype={spec.dtype})...")
    from nnsight import LanguageModel
    model = LanguageModel(
        spec.hf_name,
        device_map=spec.device_map,
        dtype=spec.dtype,
        dispatch=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=hf_token,
    )

    # Detect n_layers and d_model — handle Gemma3 which nests under text_config
    config = model.config
    if hasattr(config, "num_hidden_layers"):
        n_layers = config.num_hidden_layers
        d_model = config.hidden_size
    elif hasattr(config, "text_config"):
        n_layers = config.text_config.num_hidden_layers
        d_model = config.text_config.hidden_size
    else:
        raise AttributeError(
            f"Cannot find num_hidden_layers in config: {list(config.__dict__.keys())}"
        )
    print(f"  Layers: {n_layers}, d_model: {d_model}")

    # ── Verify layer access ──
    layers_module = get_layer_accessor(model, spec)
    sample_ids = input_ids_list[0]
    response_ids = tokenizer(
        df.iloc[0]["response"], add_special_tokens=False, return_tensors="pt"
    ).input_ids
    test_ids = torch.cat([sample_ids, response_ids], dim=1)

    with model.trace(test_ids):
        test_out = layers_module[0].output[0].save()
    print(f"  Layer access verified: shape={test_out.shape}")
    del test_out
    torch.cuda.empty_cache()

    # ── Filter positions to those requested ──
    # Only keep positions that are available (input vs output positions)
    input_positions = [p for p in positions if p in ("last_prompt", "last_system", "last_user")]
    output_positions = [p for p in positions if p in ("first_generated", "mid_generated", "last_generated")]
    token_positions = input_positions + output_positions
    print(f"  Token positions: {token_positions}")

    # ── Allocate buffers ──
    n_samples = len(df)
    buffers = {
        pos: np.zeros((n_samples, n_layers, d_model), dtype=np.float32)
        for pos in token_positions
    }
    gen_lengths = []
    skipped = 0

    # ── Extract ──
    t0 = time.time()
    for sample_idx in tqdm(range(n_samples), desc="Extracting activations"):
        row = df.iloc[sample_idx]
        prompt_ids = input_ids_list[sample_idx]
        prompt_len = prompt_ids.shape[1]
        pm = position_maps[sample_idx]

        response_ids = tokenizer(
            row["response"], add_special_tokens=False, return_tensors="pt"
        ).input_ids
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        total_len = full_ids.shape[1]
        gen_len = total_len - prompt_len
        gen_lengths.append(gen_len)
        if gen_len < 2:
            skipped += 1

        pos_indices = {
            "last_prompt": pm["last_prompt"],
            "last_system": pm["last_system"],
            "last_user": pm.get("last_user", pm["last_system"]),
            "first_generated": prompt_len,
            "mid_generated": prompt_len + gen_len // 2,
            "last_generated": total_len - 1,
        }

        saved = {}
        with model.trace(full_ids):
            for layer in range(n_layers):
                saved[layer] = layers_module[layer].output[0].save()

        for layer in range(n_layers):
            resid = saved[layer]
            for pos_name in token_positions:
                idx = min(pos_indices[pos_name], resid.shape[0] - 1)
                buffers[pos_name][sample_idx, layer, :] = (
                    resid[idx, :].float().detach().cpu().numpy()
                )

        del saved
        torch.cuda.empty_cache()

    extract_time = time.time() - t0
    print(f"\nExtraction: {extract_time:.1f}s ({extract_time / n_samples:.2f}s/sample)")
    print(f"Response lengths — min: {min(gen_lengths)}, median: {int(np.median(gen_lengths))}, "
          f"max: {max(gen_lengths)}, std: {np.std(gen_lengths):.1f}")
    print(f"Skipped (gen_len < 2): {skipped}")

    # ── Unload model ──
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded.")

    # ── Save ──
    save_dir = out_dir / spec.safe_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for pos_name in token_positions:
        path = save_dir / f"activations_{pos_name}.npy"
        np.save(path, buffers[pos_name])
        size_gb = buffers[pos_name].nbytes / 1e9
        print(f"  Saved {path.name} ({size_gb:.1f} GB)")

    # Save labels and metadata
    y = df["y"].values
    np.savez(
        save_dir / "labels.npz",
        y=y,
        label=df["label"].values,
    )

    # Save metadata CSV (for probing script)
    meta_cols = [
        "constraint_type", "system_style", "user_style", "direction",
        "condition", "task_id", "conflict_id",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    df[meta_cols].to_csv(save_dir / "metadata.csv", index=False)

    # Save position maps
    np.savez(save_dir / "position_maps.npz", **{
        f"pm_{i}": json.dumps(pm) for i, pm in enumerate(position_maps)
    })

    # Save extraction config
    config = {
        "model": spec.hf_name,
        "n_samples": n_samples,
        "n_layers": n_layers,
        "d_model": d_model,
        "token_positions": token_positions,
        "extract_time_s": extract_time,
        "skipped": skipped,
        "gen_length_stats": {
            "min": int(min(gen_lengths)),
            "median": int(np.median(gen_lengths)),
            "max": int(max(gen_lengths)),
            "std": float(np.std(gen_lengths)),
        },
        "scr": float(df["y"].mean()),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
    }
    with open(save_dir / "extraction_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAll outputs saved to {save_dir}/")
    print(f"  activations: {len(token_positions)} files")
    print(f"  labels.npz, metadata.csv, position_maps.npz, extraction_config.json")

    # Free buffers
    del buffers
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract residual stream activations for conflict resolution probing."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model key (e.g. llama-8b) or full HF name (e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Extract all supported models sequentially",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing *_results.jsonl files",
    )
    parser.add_argument(
        "--out-dir", type=str, default="./activations",
        help="Output directory for activations (default: ./activations)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print dataset statistics, do not load model or extract",
    )
    parser.add_argument(
        "--positions", nargs="+",
        default=["last_prompt", "last_system", "first_generated", "last_generated"],
        help="Token positions to extract",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    hf_token = args.hf_token or HF_TOKEN
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    # ── Resolve models to process ──
    if args.all:
        specs = list(SUPPORTED_MODELS.values())
    elif args.model:
        if args.model in SUPPORTED_MODELS:
            specs = [SUPPORTED_MODELS[args.model]]
        else:
            # Try to match by HF name
            matched = [s for s in SUPPORTED_MODELS.values() if s.hf_name == args.model]
            if matched:
                specs = matched
            else:
                # Assume it's a custom HF name — construct a spec
                safe = args.model.replace("/", "_")
                specs = [ModelSpec(
                    hf_name=args.model,
                    results_stem=safe,
                )]
                print(f"Note: '{args.model}' not in registry, using default layer_path. "
                      f"If extraction fails, add it to SUPPORTED_MODELS.")
    else:
        print("Error: specify --model or --all")
        parser.print_help()
        sys.exit(1)

    # ── Dry run ──
    if args.dry_run:
        for spec in specs:
            print(f"\n{'━' * 70}")
            print(f"  {spec.hf_name}")
            print(f"{'━' * 70}")
            try:
                df_all = load_results(data_dir, spec.results_stem)
                df = filter_condition_c_binary(df_all)
                print_dry_run_report(df_all, df)
            except FileNotFoundError as e:
                print(f"  ⚠ {e}")
        return

    # ── Extract ──
    results_summary = {}
    for spec in specs:
        try:
            extract_single_model(
                spec=spec,
                data_dir=data_dir,
                out_dir=out_dir,
                positions=args.positions,
                hf_token=hf_token,
            )
            results_summary[spec.hf_name] = "✓ success"
        except FileNotFoundError as e:
            print(f"\n⚠ Skipping {spec.hf_name}: {e}")
            results_summary[spec.hf_name] = f"⚠ skipped: {e}"
        except Exception as e:
            print(f"\n✗ FAILED {spec.hf_name}: {e}")
            traceback.print_exc()
            results_summary[spec.hf_name] = f"✗ failed: {e}"
            # Clean up GPU memory before next model
            gc.collect()
            torch.cuda.empty_cache()

    # ── Print summary ──
    if len(specs) > 1:
        print(f"\n{'=' * 70}")
        print("EXTRACTION SUMMARY")
        print(f"{'=' * 70}")
        for model_name, status in results_summary.items():
            print(f"  {model_name:<50} {status}")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()