#!/usr/bin/env python3
"""
extract_token_projections.py — Extract per-token projections for separation animation.

Loads a model, runs forward pass on all samples, and at ONE fixed layer
projects every token's activation onto two fixed axes:
  - X axis: IID-MM conflict resolution direction (d)
  - Y axis: Residual PC1 direction (v)

Then generates animations showing how class separation emerges across token positions.

Usage:
    python -u extract_token_projections.py \
        --results-jsonl ./data/meta-llama_Llama-3.1-8B-Instruct_results.jsonl \
        --probe-dir ./results/llama-8b \
        --act-dir /workspace/activations/meta-llama_Llama-3.1-8B-Instruct \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --out-dir ./results/llama-8b/token_animation
"""

import argparse
import json
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import to_rgba


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (reuse from run_probing / extract_activations)
# ─────────────────────────────────────────────────────────────────────────────

def load_results_jsonl(path):
    records = []
    decoder = json.JSONDecoder()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(decoder.decode(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(records)


def filter_condition_c_binary(df):
    df_c = df[df["condition"] == "C"].copy()
    df_c = df_c[df_c["label"].isin(["followed_system", "followed_user"])].copy()
    df_c["y"] = (df_c["label"] == "followed_system").astype(int)
    return df_c.reset_index(drop=True)


def build_formatted_prompt(tokenizer, system_text, user_text):
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _balance_fallback(df_c, y):
    """Balance df_c to match len(y) using seed=42."""
    n_per_class = len(y) // 2
    rng = np.random.RandomState(42)
    pos_idx = np.where(df_c["y"].values == 1)[0]
    neg_idx = np.where(df_c["y"].values == 0)[0]
    if len(pos_idx) > n_per_class:
        pos_idx = rng.choice(pos_idx, size=n_per_class, replace=False)
    neg_idx = rng.choice(neg_idx, size=n_per_class, replace=False)
    keep = np.sort(np.concatenate([pos_idx, neg_idx]))
    return df_c.iloc[keep].reset_index(drop=True)


def find_token_boundaries(tokenizer, system_text, user_text):
    """Find where system, user, and generation tokens start/end."""
    encode_len = lambda text: len(tokenizer.encode(text, add_special_tokens=False))

    messages_sys = [{"role": "system", "content": system_text}]
    messages_full = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    sys_str = tokenizer.apply_chat_template(
        messages_sys, tokenize=False, add_generation_prompt=False
    )
    sys_user_str = tokenizer.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False
    )
    full_str = tokenizer.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=True
    )

    n_sys = encode_len(sys_str)
    n_sys_user = encode_len(sys_user_str)
    n_full = encode_len(full_str)

    return {
        "system_end": n_sys,        # tokens [0, system_end) are system
        "user_end": n_sys_user,     # tokens [system_end, user_end) are user  
        "prompt_end": n_full,       # tokens [user_end, prompt_end) are template/generation prompt
    }


# ─────────────────────────────────────────────────────────────────────────────
# Compute fixed axes from existing activations
# ─────────────────────────────────────────────────────────────────────────────

def compute_fixed_axes(act_dir, probe_dir, method="iid_mm", position="last_prompt"):
    """
    Compute the two fixed axes (d, v) from existing run_probing results.
    d = IID-MM direction at peak layer
    v = residual PC1 after projecting out d
    Also returns peak_layer and scaler stats (mean, std).
    """
    # Load peak layer
    with open(Path(probe_dir) / "metrics" / "peak_layers.json") as f:
        peak_layers = json.load(f)
    peak_layer = peak_layers[method][position]
    print(f"  Peak layer: {peak_layer} ({method} @ {position})")

    # Load direction
    d_path = Path(probe_dir) / "artifacts" / "directions" / f"{method}_{position}_L{peak_layer}.npy"
    d = np.load(d_path)
    d = d / (np.linalg.norm(d) + 1e-12)
    print(f"  Direction loaded: {d_path.name}")

    # Load activations at peak position to compute residual PC1
    KNOWN_POSITIONS = ["last_prompt", "last_system", "last_user",
                       "first_generated", "mid_generated", "last_generated"]
    act_path = None
    for f in Path(act_dir).glob("activations_*.npy"):
        if f.stem.endswith(f"_{position}"):
            act_path = f
            break
    if act_path is None:
        raise FileNotFoundError(f"No activation file for position '{position}' in {act_dir}")

    X_full = np.load(act_path)  # (n_samples, n_layers, d_model)
    X_peak = X_full[:, peak_layer, :]  # (n_samples, d_model)
    del X_full
    gc.collect()

    # Load labels
    y_files = sorted(Path(act_dir).glob("y_*.npy"))
    labels_file = Path(act_dir) / "labels.npz"
    if labels_file.exists():
        y = np.load(labels_file)["y"]
    elif y_files:
        y = np.load(y_files[0])
    else:
        raise FileNotFoundError(f"No label file in {act_dir}")

    # Compute scaler stats from peak position (will apply to all tokens)
    mu = X_peak.mean(axis=0)
    std = X_peak.std(axis=0)
    std[std < 1e-12] = 1e-12

    # Scale
    X_scaled = (X_peak - mu) / std

    # Project onto d
    proj_d = X_scaled @ d

    # Residual after removing d component
    residual = X_scaled - np.outer(proj_d, d)

    # PCA on residual → v
    pca = PCA(n_components=1, random_state=42)
    pca.fit(residual)
    v = pca.components_[0]
    v = v / (np.linalg.norm(v) + 1e-12)
    print(f"  Residual PC1 computed (var explained: {pca.explained_variance_ratio_[0]:.3f})")

    return d, v, mu, std, peak_layer, y


# ─────────────────────────────────────────────────────────────────────────────
# Layer path helpers (same as extract_activations.py)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_LAYER_PATHS = {
    "llama": "model.layers",
    "qwen": "model.layers",
    "gemma": "model.layers",
    "gpt_neox": "gpt_neox.layers",
}

def guess_layer_path(model_name):
    name_lower = model_name.lower()
    if "gpt-neox" in name_lower or "gpt-oss" in name_lower:
        return "gpt_neox.layers"
    return "model.layers"


def get_layers_module(model, model_name):
    """Get the layers ModuleList from an nnsight model."""
    path = guess_layer_path(model_name)
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Extraction: per-token projections
# ─────────────────────────────────────────────────────────────────────────────

def extract_token_projections(
    model_name, results_jsonl, d, v, mu, std, peak_layer, y,
    out_dir, act_dir, hf_token=None,
):
    """
    Run forward pass on all samples, extract per-token projections at one layer.
    For each sample, stores:
      - proj_x: projection onto d (conflict resolution axis) per token
      - proj_y: projection onto v (residual PC1) per token
      - boundaries: {system_end, user_end, prompt_end}
      - label: 0 or 1
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data — replicate EXACT notebook filter logic to get same rows in same order
    print("Loading results.jsonl...")
    df_all = load_results_jsonl(results_jsonl)

    # Step 1: filter Condition C + binary labels (same as notebook cell 14)
    df = df_all[df_all["condition"] == "C"].copy()
    df = df[df["label"].isin(["followed_system", "followed_user"])].copy()
    df["y"] = (df["label"] == "followed_system").astype(int)
    df = df.reset_index(drop=True)
    print(f"  Condition C binary: {len(df)} samples (SCR={df['y'].mean():.3f})")

    # Step 2: balance — keep ALL minority class + sample majority to match (seed=42)
    n_pos = int(df["y"].sum())
    n_neg = int((df["y"] == 0).sum())
    n_per_class = min(n_pos, n_neg)

    if n_per_class * 2 != len(df):
        rng = np.random.RandomState(42)
        pos_idx = np.where(df["y"].values == 1)[0]
        neg_idx = np.where(df["y"].values == 0)[0]

        if n_pos <= n_neg:
            # Positives are minority — keep all, sample negatives
            keep_pos = pos_idx
            keep_neg = rng.choice(neg_idx, size=n_per_class, replace=False)
        else:
            # Negatives are minority — keep all, sample positives
            keep_neg = neg_idx
            keep_pos = rng.choice(pos_idx, size=n_per_class, replace=False)

        keep_idx = np.sort(np.concatenate([keep_pos, keep_neg]))
        df = df.iloc[keep_idx].reset_index(drop=True)
        print(f"  Balanced to {len(df)} samples ({n_per_class} per class)")

    # Verify against saved labels
    y_check = df["y"].values
    if len(y_check) == len(y):
        match_rate = (y_check == y).mean()
        print(f"  Label match rate vs saved y: {match_rate:.3f}")
        if match_rate < 0.99:
            print(f"  ⚠ WARNING: labels don't fully match ({match_rate:.1%}). "
                  f"Using saved y for class coloring in animation.")
    else:
        print(f"  ⚠ Size mismatch: filtered={len(y_check)}, saved={len(y)}. "
              f"Proceeding with filtered df ({len(df)} samples).")
        y = y_check

    print(f"  {len(df)} samples ready")

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare prompts
    print("Tokenizing prompts...")
    input_ids_list = []
    boundaries_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        fp = build_formatted_prompt(tokenizer, row["system_prompt"], row["user_prompt"])
        ids = tokenizer(fp, return_tensors="pt", add_special_tokens=False).input_ids
        bounds = find_token_boundaries(tokenizer, row["system_prompt"], row["user_prompt"])
        input_ids_list.append(ids)
        boundaries_list.append(bounds)

    # Load model
    print(f"Loading model {model_name}...")
    from nnsight import LanguageModel
    model = LanguageModel(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        dispatch=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    layers_module = get_layers_module(model, model_name)
    print(f"  Model loaded. Extracting layer {peak_layer} only.")

    # Convert axes to torch
    d_t = torch.tensor(d, dtype=torch.float32)
    v_t = torch.tensor(v, dtype=torch.float32)
    mu_t = torch.tensor(mu, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32)

    # Extract
    all_proj_x = []   # list of 1D arrays
    all_proj_y = []
    all_seq_lens = []
    all_prompt_lens = []

    t0 = time.time()
    for idx in tqdm(range(len(df)), desc="Forward pass"):
        row = df.iloc[idx]
        prompt_ids = input_ids_list[idx]
        prompt_len = prompt_ids.shape[1]

        response_ids = tokenizer(
            row["response"], add_special_tokens=False, return_tensors="pt"
        ).input_ids
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        seq_len = full_ids.shape[1]

        # Forward pass — only save one layer
        with model.trace(full_ids):
            layer_out = layers_module[peak_layer].output[0].save()

        # layer_out shape: (seq_len, d_model)
        h = layer_out.float().detach().cpu()  # (seq_len, d_model)

        # Scale with fixed mu/std from peak position
        h_scaled = (h - mu_t) / std_t

        # Project onto both axes
        px = (h_scaled @ d_t).detach().numpy()  # (seq_len,)
        # Y axis = residual PC1: remove d component first, then project onto v
        h_residual = h_scaled - torch.outer(h_scaled @ d_t, d_t)
        py = (h_residual @ v_t).detach().numpy()  # (seq_len,)

        all_proj_x.append(px)
        all_proj_y.append(py)
        all_seq_lens.append(seq_len)
        all_prompt_lens.append(prompt_len)

        del layer_out, h, h_scaled, h_residual
        if idx % 500 == 0:
            torch.cuda.empty_cache()

    extract_time = time.time() - t0
    print(f"\nExtraction: {extract_time:.1f}s ({extract_time/len(df):.2f}s/sample)")

    # Unload model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Save raw projections
    print("Saving projections...")
    np.savez(
        out_dir / "token_projections.npz",
        # Can't store ragged arrays directly, so use object array
        proj_x=np.array(all_proj_x, dtype=object),
        proj_y=np.array(all_proj_y, dtype=object),
        seq_lens=np.array(all_seq_lens),
        prompt_lens=np.array(all_prompt_lens),
        y=y,
        d=d,
        v=v,
        mu=mu,
        std=std,
        peak_layer=np.array(peak_layer),
    )

    # Save boundaries
    with open(out_dir / "token_boundaries.json", "w") as f:
        json.dump(boundaries_list, f)

    print(f"  Saved to {out_dir}/")

    return all_proj_x, all_proj_y, all_seq_lens, all_prompt_lens, boundaries_list


# ─────────────────────────────────────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────────────────────────────────────

def make_animations(
    all_proj_x, all_proj_y, all_seq_lens, all_prompt_lens,
    boundaries_list, y, out_dir, n_frames_pct=100, fps=10,
):
    """
    Generate two animations:
      1. Absolute position: frame = token index, samples disappear when seq ends
      2. Normalized position: frame = percentage (0-100%), all samples in every frame
    """
    out_dir = Path(out_dir)
    n_samples = len(y)
    sys_mask = y == 1  # followed_system
    usr_mask = y == 0  # followed_user

    # Compute axis limits from exactly the last_prompt token position
    lp_x, lp_y = [], []
    for i in range(n_samples):
        prompt_len = all_prompt_lens[i]
        tok = prompt_len - 1  # last prompt token
        if tok < all_seq_lens[i]:
            lp_x.append(all_proj_x[i][tok])
            lp_y.append(all_proj_y[i][tok])
    lp_x = np.array(lp_x)
    lp_y = np.array(lp_y)
    x_lo, x_hi = np.percentile(lp_x, [0.5, 99.5])
    y_lo, y_hi = np.percentile(lp_y, [0.5, 99.5])
    x_margin = (x_hi - x_lo) * 0.15
    y_margin = (y_hi - y_lo) * 0.15
    xlim = (x_lo - x_margin, x_hi + x_margin)
    ylim = (y_lo - y_margin, y_hi + y_margin)

    max_seq = max(all_seq_lens)

    # Colors
    clr_sys = to_rgba("#d62728", alpha=0.15)
    clr_usr = to_rgba("#1f77b4", alpha=0.15)

    # ── Animation 1: Absolute position ──
    print("\nGenerating absolute position animation...")

    # Subsample frames for absolute (every 2 tokens, otherwise too many frames)
    step = max(1, max_seq // 200)
    abs_frames = list(range(0, max_seq, step))

    fig, ax = plt.subplots(figsize=(10, 7))
    scat_sys = ax.scatter([], [], c=[clr_sys], s=4, label="follow system", rasterized=True)
    scat_usr = ax.scatter([], [], c=[clr_usr], s=4, label="follow user", rasterized=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("IID-MM projection (conflict resolution axis)", fontsize=12)
    ax.set_ylabel("Residual PC1", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, markerscale=4)
    title = ax.set_title("", fontsize=13)

    # Precompute which region each token falls in
    def get_region_label(tok_idx, bounds, prompt_len, seq_len):
        if tok_idx < bounds.get("system_end", 0):
            return "system"
        elif tok_idx < bounds.get("user_end", prompt_len):
            return "user"
        elif tok_idx < prompt_len:
            return "template"
        else:
            return "generated"

    def update_abs(frame_idx):
        tok = abs_frames[frame_idx]
        xs_sys, ys_sys = [], []
        xs_usr, ys_usr = [], []
        n_alive = 0
        for i in range(n_samples):
            if tok >= all_seq_lens[i]:
                continue
            n_alive += 1
            px = all_proj_x[i][tok]
            py = all_proj_y[i][tok]
            if sys_mask[i]:
                xs_sys.append(px)
                ys_sys.append(py)
            else:
                xs_usr.append(px)
                ys_usr.append(py)

        scat_sys.set_offsets(np.c_[xs_sys, ys_sys] if xs_sys else np.empty((0, 2)))
        scat_usr.set_offsets(np.c_[xs_usr, ys_usr] if xs_usr else np.empty((0, 2)))

        # Determine region for display
        median_prompt_len = int(np.median(all_prompt_lens))
        if tok < int(np.median([b.get("system_end", 0) for b in boundaries_list])):
            region = "system prompt"
        elif tok < int(np.median([b.get("user_end", 0) for b in boundaries_list])):
            region = "user prompt"
        elif tok < median_prompt_len:
            region = "template"
        else:
            region = "generated"

        title.set_text(f"Token {tok}/{max_seq}  |  region: {region}  |  samples: {n_alive}")
        return scat_sys, scat_usr, title

    ani1 = animation.FuncAnimation(
        fig, update_abs, frames=len(abs_frames), blit=False, interval=1000 // fps
    )
    ani1.save(str(out_dir / "separation_absolute.gif"), writer="pillow", fps=fps, dpi=100)
    plt.close(fig)
    print(f"  Saved separation_absolute.gif ({len(abs_frames)} frames)")

    # ── Animation 2: Normalized position (0%–100%) ──
    print("Generating normalized position animation...")

    fig, ax = plt.subplots(figsize=(10, 7))
    scat_sys = ax.scatter([], [], c=[clr_sys], s=4, label="follow system", rasterized=True)
    scat_usr = ax.scatter([], [], c=[clr_usr], s=4, label="follow user", rasterized=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("IID-MM projection (conflict resolution axis)", fontsize=12)
    ax.set_ylabel("Residual PC1", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, markerscale=4)
    title = ax.set_title("", fontsize=13)

    # Precompute normalized token indices for each sample
    # For pct p (0-100), each sample maps to token = int(p/100 * (seq_len-1))
    pct_frames = list(range(0, 101, max(1, 100 // n_frames_pct)))

    def update_pct(frame_idx):
        pct = pct_frames[frame_idx]
        xs_sys, ys_sys = [], []
        xs_usr, ys_usr = [], []
        for i in range(n_samples):
            tok = int(pct / 100.0 * (all_seq_lens[i] - 1))
            tok = min(tok, all_seq_lens[i] - 1)
            px = all_proj_x[i][tok]
            py = all_proj_y[i][tok]
            if sys_mask[i]:
                xs_sys.append(px)
                ys_sys.append(py)
            else:
                xs_usr.append(px)
                ys_usr.append(py)

        scat_sys.set_offsets(np.c_[xs_sys, ys_sys] if xs_sys else np.empty((0, 2)))
        scat_usr.set_offsets(np.c_[xs_usr, ys_usr] if xs_usr else np.empty((0, 2)))

        # Estimate what region this % corresponds to
        median_prompt_pct = np.median([
            b.get("system_end", 0) / sl for b, sl in zip(boundaries_list, all_seq_lens)
        ]) * 100
        median_user_pct = np.median([
            b.get("user_end", 0) / sl for b, sl in zip(boundaries_list, all_seq_lens)
        ]) * 100
        median_gen_pct = np.median([
            pl / sl for pl, sl in zip(all_prompt_lens, all_seq_lens)
        ]) * 100

        if pct < median_prompt_pct:
            region = "system prompt"
        elif pct < median_user_pct:
            region = "user prompt"
        elif pct < median_gen_pct:
            region = "template"
        else:
            region = "generated"

        title.set_text(f"Position: {pct}%  |  region: {region}  |  samples: {n_samples}")
        return scat_sys, scat_usr, title

    ani2 = animation.FuncAnimation(
        fig, update_pct, frames=len(pct_frames), blit=False, interval=1000 // fps
    )
    ani2.save(str(out_dir / "separation_normalized.gif"), writer="pillow", fps=fps, dpi=100)
    plt.close(fig)
    print(f"  Saved separation_normalized.gif ({len(pct_frames)} frames)")

    # ── Animations 3a/3b/3c: Per-segment normalized (system / user / generation) ──
    segments = {
        "system": lambda i: (0, boundaries_list[i].get("system_end", 0)),
        "user": lambda i: (boundaries_list[i].get("system_end", 0),
                           boundaries_list[i].get("user_end", all_prompt_lens[i])),
        "generation": lambda i: (all_prompt_lens[i], all_seq_lens[i]),
    }

    for seg_name, seg_range_fn in segments.items():
        print(f"Generating {seg_name} segment animation...")

        fig, ax = plt.subplots(figsize=(10, 7))
        scat_sys_seg = ax.scatter([], [], c=[clr_sys], s=4, label="follow system", rasterized=True)
        scat_usr_seg = ax.scatter([], [], c=[clr_usr], s=4, label="follow user", rasterized=True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("IID-MM projection (conflict resolution axis)", fontsize=12)
        ax.set_ylabel("Residual PC1", fontsize=12)
        ax.legend(loc="upper right", fontsize=10, markerscale=4)
        title_seg = ax.set_title("", fontsize=13)

        seg_pct_frames = list(range(0, 101))

        def make_update_seg(seg_fn, seg_label):
            def update_seg(frame_idx):
                pct = seg_pct_frames[frame_idx]
                xs_sys, ys_sys = [], []
                xs_usr, ys_usr = [], []
                n_valid = 0
                for i in range(n_samples):
                    start, end = seg_fn(i)
                    seg_len = end - start
                    if seg_len <= 0:
                        continue
                    tok = start + int(pct / 100.0 * (seg_len - 1))
                    tok = min(tok, all_seq_lens[i] - 1)
                    n_valid += 1
                    px = all_proj_x[i][tok]
                    py = all_proj_y[i][tok]
                    if sys_mask[i]:
                        xs_sys.append(px)
                        ys_sys.append(py)
                    else:
                        xs_usr.append(px)
                        ys_usr.append(py)

                scat_sys_seg.set_offsets(np.c_[xs_sys, ys_sys] if xs_sys else np.empty((0, 2)))
                scat_usr_seg.set_offsets(np.c_[xs_usr, ys_usr] if xs_usr else np.empty((0, 2)))
                title_seg.set_text(f"{seg_label}: {pct}%  |  samples: {n_valid}")
                return scat_sys_seg, scat_usr_seg, title_seg
            return update_seg

        ani_seg = animation.FuncAnimation(
            fig, make_update_seg(seg_range_fn, seg_name),
            frames=len(seg_pct_frames), blit=False, interval=1000 // fps
        )
        ani_seg.save(str(out_dir / f"separation_{seg_name}.gif"), writer="pillow", fps=fps, dpi=100)
        plt.close(fig)
        print(f"  Saved separation_{seg_name}.gif ({len(seg_pct_frames)} frames)")

        # Static ridge plot for this segment
        n_slices_seg = 8
        pct_slices_seg = np.linspace(0, 100, n_slices_seg).astype(int)

        fig, axes = plt.subplots(1, n_slices_seg, figsize=(n_slices_seg * 2.5, 5), sharey=True)
        for idx, pct in enumerate(pct_slices_seg):
            ax = axes[idx]
            xs_s, ys_s = [], []
            xs_u, ys_u = [], []
            for i in range(n_samples):
                start, end = seg_range_fn(i)
                seg_len = end - start
                if seg_len <= 0:
                    continue
                tok = start + int(pct / 100.0 * (seg_len - 1))
                tok = min(tok, all_seq_lens[i] - 1)
                px = all_proj_x[i][tok]
                py = all_proj_y[i][tok]
                if sys_mask[i]:
                    xs_s.append(px)
                    ys_s.append(py)
                else:
                    xs_u.append(px)
                    ys_u.append(py)

            ax.scatter(xs_s, ys_s, c="#d62728", s=1, alpha=0.08, rasterized=True)
            ax.scatter(xs_u, ys_u, c="#1f77b4", s=1, alpha=0.08, rasterized=True)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title(f"{pct}%", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Residual PC1", fontsize=10)
            ax.set_xlabel("IID-MM proj.", fontsize=8)
            ax.tick_params(labelsize=7)

        fig.suptitle(f"Activation separation — {seg_name} prompt (0-100%)", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(str(out_dir / f"separation_static_{seg_name}.png"), dpi=200, bbox_inches="tight")
        fig.savefig(str(out_dir / f"separation_static_{seg_name}.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved separation_static_{seg_name}.png/pdf")

    # ── Animation 4: Combined sequential (sys → tmpl → user → tmpl → generation) ──
    print("Generating combined sequential animation...")

    # Define 5 segments with template tokens between them
    combined_segments = [
        ("system",    lambda i: (0, boundaries_list[i].get("system_end", 0))),
        ("template₁", lambda i: (boundaries_list[i].get("system_end", 0),
                                  boundaries_list[i].get("system_end", 0))),  # placeholder, will check
        ("user",      lambda i: (boundaries_list[i].get("system_end", 0),
                                  boundaries_list[i].get("user_end", all_prompt_lens[i]))),
        ("template₂", lambda i: (boundaries_list[i].get("user_end", all_prompt_lens[i]),
                                  all_prompt_lens[i])),
        ("generation", lambda i: (all_prompt_lens[i], all_seq_lens[i])),
    ]

    # Frames per segment: proportional to median token count
    seg_median_lens = []
    for seg_name_c, seg_fn_c in combined_segments:
        lens = []
        for i in range(n_samples):
            s, e = seg_fn_c(i)
            lens.append(max(0, e - s))
        seg_median_lens.append(max(1, int(np.median(lens))))

    total_median = sum(seg_median_lens)
    # Allocate frames: minimum 5 per segment, rest proportional
    n_total_frames = 200
    frames_per_seg = []
    for ml in seg_median_lens:
        frames_per_seg.append(max(5, int(n_total_frames * ml / total_median)))

    # Build frame list: (segment_index, pct_within_segment)
    combined_frames = []
    for seg_idx, n_frames in enumerate(frames_per_seg):
        for f in range(n_frames):
            pct = int(f / max(1, n_frames - 1) * 100)
            combined_frames.append((seg_idx, pct))

    fig, ax = plt.subplots(figsize=(10, 7))
    scat_sys_c = ax.scatter([], [], c=[clr_sys], s=4, label="follow system", rasterized=True)
    scat_usr_c = ax.scatter([], [], c=[clr_usr], s=4, label="follow user", rasterized=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("IID-MM projection (conflict resolution axis)", fontsize=12)
    ax.set_ylabel("Residual PC1", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, markerscale=4)
    title_c = ax.set_title("", fontsize=13)

    seg_colors = {
        "system": "#2ecc71",
        "template₁": "#95a5a6",
        "user": "#3498db",
        "template₂": "#e67e22",
        "generation": "#e74c3c",
    }

    def update_combined(frame_idx):
        seg_idx, pct = combined_frames[frame_idx]
        seg_name_c, seg_fn_c = combined_segments[seg_idx]
        xs_sys, ys_sys = [], []
        xs_usr, ys_usr = [], []
        n_valid = 0
        for i in range(n_samples):
            start, end = seg_fn_c(i)
            seg_len = end - start
            if seg_len <= 0:
                continue
            tok = start + int(pct / 100.0 * (seg_len - 1))
            tok = min(tok, all_seq_lens[i] - 1)
            n_valid += 1
            px_val = all_proj_x[i][tok]
            py_val = all_proj_y[i][tok]
            if sys_mask[i]:
                xs_sys.append(px_val)
                ys_sys.append(py_val)
            else:
                xs_usr.append(px_val)
                ys_usr.append(py_val)

        scat_sys_c.set_offsets(np.c_[xs_sys, ys_sys] if xs_sys else np.empty((0, 2)))
        scat_usr_c.set_offsets(np.c_[xs_usr, ys_usr] if xs_usr else np.empty((0, 2)))

        # Progress bar showing which segment we're in
        total_so_far = sum(frames_per_seg[:seg_idx]) + (frame_idx - sum(frames_per_seg[:seg_idx]))
        overall_pct = int(total_so_far / len(combined_frames) * 100)

        title_c.set_text(
            f"[{seg_name_c}] {pct}%  |  overall: {overall_pct}%  |  samples: {n_valid}"
        )
        return scat_sys_c, scat_usr_c, title_c

    ani_combined = animation.FuncAnimation(
        fig, update_combined, frames=len(combined_frames),
        blit=False, interval=1000 // fps
    )
    ani_combined.save(str(out_dir / "separation_combined.gif"), writer="pillow", fps=fps, dpi=100)
    plt.close(fig)
    print(f"  Saved separation_combined.gif ({len(combined_frames)} frames)")

    # Static version: one subplot per segment, showing 0%, 50%, 100%
    print("Generating combined static plot...")
    n_seg = len(combined_segments)
    slices_per = 3  # 0%, 50%, 100%
    n_cols = n_seg * slices_per
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2, 5), sharey=True)

    col = 0
    for seg_idx, (seg_name_c, seg_fn_c) in enumerate(combined_segments):
        for pct in [0, 50, 100]:
            ax = axes[col]
            xs_s, ys_s, xs_u, ys_u = [], [], [], []
            for i in range(n_samples):
                start, end = seg_fn_c(i)
                seg_len = end - start
                if seg_len <= 0:
                    continue
                tok = start + int(pct / 100.0 * (seg_len - 1))
                tok = min(tok, all_seq_lens[i] - 1)
                px_val = all_proj_x[i][tok]
                py_val = all_proj_y[i][tok]
                if sys_mask[i]:
                    xs_s.append(px_val)
                    ys_s.append(py_val)
                else:
                    xs_u.append(px_val)
                    ys_u.append(py_val)

            ax.scatter(xs_s, ys_s, c="#d62728", s=1, alpha=0.08, rasterized=True)
            ax.scatter(xs_u, ys_u, c="#1f77b4", s=1, alpha=0.08, rasterized=True)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title(f"{seg_name_c}\n{pct}%", fontsize=8)
            if col == 0:
                ax.set_ylabel("Residual PC1", fontsize=10)
            ax.set_xlabel("IID-MM proj.", fontsize=7)
            ax.tick_params(labelsize=6)
            col += 1

    fig.suptitle("Activation separation — sequential segments", fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(str(out_dir / "separation_static_combined.png"), dpi=200, bbox_inches="tight")
    fig.savefig(str(out_dir / "separation_static_combined.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved separation_static_combined.png/pdf")

    # ── Also save a static ridge plot for the paper ──
    print("Generating static ridge plot...")
    n_slices = 8
    pct_slices = np.linspace(0, 100, n_slices).astype(int)

    fig, axes = plt.subplots(1, n_slices, figsize=(n_slices * 2.5, 5), sharey=True)
    for idx, pct in enumerate(pct_slices):
        ax = axes[idx]
        xs_sys, ys_sys = [], []
        xs_usr, ys_usr = [], []
        for i in range(n_samples):
            tok = int(pct / 100.0 * (all_seq_lens[i] - 1))
            tok = min(tok, all_seq_lens[i] - 1)
            px = all_proj_x[i][tok]
            py = all_proj_y[i][tok]
            if sys_mask[i]:
                xs_sys.append(px)
                ys_sys.append(py)
            else:
                xs_usr.append(px)
                ys_usr.append(py)

        ax.scatter(xs_sys, ys_sys, c="#d62728", s=1, alpha=0.08, rasterized=True)
        ax.scatter(xs_usr, ys_usr, c="#1f77b4", s=1, alpha=0.08, rasterized=True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{pct}%", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Residual PC1", fontsize=10)
        ax.set_xlabel("IID-MM proj.", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle("Activation separation across sequence positions", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(str(out_dir / "separation_static.png"), dpi=200, bbox_inches="tight")
    fig.savefig(str(out_dir / "separation_static.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved separation_static.png/pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract per-token projections and generate separation animation."
    )
    parser.add_argument("--results-jsonl", type=str, required=True,
                        help="Path to *_results.jsonl")
    parser.add_argument("--probe-dir", type=str, required=True,
                        help="Directory with run_probing.py outputs (peak_layers.json, directions/)")
    parser.add_argument("--act-dir", type=str, required=True,
                        help="Directory with existing activations (for computing residual PC1)")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--out-dir", type=str, default="./token_animation")
    parser.add_argument("--method", type=str, default="iid_mm",
                        help="Direction method to use (default: iid_mm)")
    parser.add_argument("--position", type=str, default="last_prompt",
                        help="Position used for peak layer / direction (default: last_prompt)")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction, only generate animation from saved projections")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Compute fixed axes from existing results
    print("── Computing fixed axes ──")
    d, v, mu, std, peak_layer, y = compute_fixed_axes(
        act_dir=args.act_dir,
        probe_dir=args.probe_dir,
        method=args.method,
        position=args.position,
    )

    if not args.skip_extract:
        # Step 2: Extract per-token projections
        print(f"\n── Extracting per-token projections (layer {peak_layer}) ──")
        all_proj_x, all_proj_y, all_seq_lens, all_prompt_lens, boundaries = \
            extract_token_projections(
                model_name=args.model,
                results_jsonl=args.results_jsonl,
                d=d, v=v, mu=mu, std=std,
                peak_layer=peak_layer,
                y=y,
                out_dir=out_dir,
                act_dir=args.act_dir,
                hf_token=hf_token,
            )
    else:
        # Load saved projections
        print("\n── Loading saved projections ──")
        data = np.load(out_dir / "token_projections.npz", allow_pickle=True)
        all_proj_x = list(data["proj_x"])
        all_proj_y = list(data["proj_y"])
        all_seq_lens = data["seq_lens"].tolist()
        all_prompt_lens = data["prompt_lens"].tolist()
        with open(out_dir / "token_boundaries.json") as f:
            boundaries = json.load(f)

        # Use d and v from npz (the ones used during extraction) — NOT the freshly
        # computed ones from compute_fixed_axes, because PCA sign/direction can differ
        # between runs.
        if "d" in data and "v" in data:
            d_saved = data["d"]
            v_saved = data["v"]
            # Check if saved d/v differ from freshly computed ones
            d_cos = abs(np.dot(d, d_saved))
            v_cos = abs(np.dot(v, v_saved))
            print(f"  d alignment (saved vs fresh): {d_cos:.6f}")
            print(f"  v alignment (saved vs fresh): {v_cos:.6f}")
            # Use fresh d (same direction method) but fix proj_y using saved v
            # Actually: proj_x was computed with the EXTRACTION-TIME d, proj_y with EXTRACTION-TIME v.
            # Both are stored. We should use them as-is, no correction needed.
            # But the extraction-time v may have been computed WITHOUT removing d component.
            # So we correct: py_new = py_old - px * dot(d_saved, v_saved)
            d_dot_v_saved = np.dot(d_saved, v_saved)
            if abs(d_dot_v_saved) > 1e-6:
                print(f"  Correcting proj_y: removing d component (d·v = {d_dot_v_saved:.4f})")
                for i in range(len(all_proj_x)):
                    all_proj_y[i] = all_proj_y[i] - all_proj_x[i] * d_dot_v_saved

            # Override d, v with saved versions for axis limit computation
            d = d_saved
            v = v_saved

        print(f"  Loaded {len(all_proj_x)} samples")

    # Step 3: Generate animations
    print("\n── Generating animations ──")
    make_animations(
        all_proj_x, all_proj_y, all_seq_lens, all_prompt_lens,
        boundaries, y, out_dir, fps=args.fps,
    )

    print("\n── Done ──")
    print(f"Outputs in {out_dir}/:")
    print(f"  token_projections.npz (raw data)")
    print(f"  token_boundaries.json")
    print(f"  separation_absolute.gif")
    print(f"  separation_normalized.gif")
    print(f"  separation_static.png/pdf")


if __name__ == "__main__":
    main()