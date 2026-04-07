"""Utilities for steering exploration experiments.

Provides helpers for querying sample IDs, generating steered text,
computing informed steering parameters, summarizing results with
coherence scoring, and saving experiment artifacts.

Usage:
    from explore_utils import (
        get_sample_ids, generate, summarize, save_experiment,
        steering_clues, get_projection_stats,
    )
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import requests

from coherence import score_coherence, compute_genuine_scr

BASE = "http://localhost:8000"

CONFLICT_IDS = [
    "json_only_vs_plain",
    "list_bullets_vs_numbered",
    "past_vs_present_tense",
    "starting_word_hello_greetings",
]


def set_base_url(url: str):
    """Override the server base URL (default http://localhost:8000)."""
    global BASE
    BASE = url


# ── Sample selection ───────────────────────────────────────────────────────


def get_sample_ids(
    conflict_ids: list[str] | None = None,
    baseline_label: str = "followed_user",
    seed: int = 42,
    limit: int | None = 96,
) -> list[str]:
    """Get sample_ids from the server for all constraints and directions.

    Parameters
    ----------
    conflict_ids : constraints to include (default: all 4 curated4 constraints)
    baseline_label : "followed_user" or "followed_system"
    seed : deterministic shuffle seed (same seed = same subset = comparable)
    limit : max samples per (conflict_id, direction) cell. None = all.

    Returns
    -------
    Flat list of experiment_hash strings.
    """
    cids = conflict_ids or CONFLICT_IDS
    all_ids = []
    for cid in cids:
        for d in ["a_to_b", "b_to_a"]:
            params: dict = {
                "conflict_id": cid,
                "direction": d,
                "baseline_label": baseline_label,
                "seed": seed,
            }
            if limit is not None:
                params["limit"] = limit
            resp = requests.get(f"{BASE}/samples", params=params, timeout=30)
            resp.raise_for_status()
            all_ids.extend(resp.json()["sample_ids"])
    return all_ids


# ── Generation ─────────────────────────────────────────────────────────────


def generate(
    sample_ids: list[str],
    direction: str | list[float] | None = None,
    layer: int | None = None,
    alpha: float = 0.0,
    mode: str = "additive",
    projection_target: float | None = None,
    projections: list[dict] | None = None,
    max_new_tokens: int = 512,
) -> dict:
    """Generate steered text for a list of sample_ids.

    Returns the full server response dict with keys:
    ``steering``, ``responses``, ``n_prompts``, ``batch_size``, ``elapsed_s``.

    Each response contains: text, label, confidence, sys_ok, usr_ok,
    sample_id, baseline_label, conflict_id, direction, system_prompt,
    user_prompt.
    """
    body: dict = {
        "sample_ids": sample_ids,
        "max_new_tokens": max_new_tokens,
    }
    if projections is not None:
        body["projections"] = projections
    elif direction is not None:
        body["direction"] = direction
        body["layer"] = layer
        body["mode"] = mode
        body["alpha"] = alpha
        if projection_target is not None:
            body["projection_target"] = projection_target
    r = requests.post(f"{BASE}/generate", json=body, timeout=600)
    r.raise_for_status()
    return r.json()


# ── Projection stats & informed parameters ─────────────────────────────────


def get_projection_stats() -> dict:
    """Fetch pre-computed projection stats from the server."""
    r = requests.get(f"{BASE}/projection_stats", timeout=30)
    r.raise_for_status()
    return r.json()


def steering_clues(
    stats: dict,
    constraint: str,
    direction_name: str,
    layer: int,
    alpha_multiplier: float = 3.0,
    target_n_std: float = 2.0,
) -> dict:
    """Get orientation from the activation distributions before steering.

    Returns where followed_system and followed_user activations sit along
    this direction, the class separation, and suggested starting points for
    alpha and projection target. These are **starting points** — adapt based
    on what you observe (coherence, genuine_scr, response quality).

    Parameters
    ----------
    stats : from :func:`get_projection_stats`
    constraint : conflict_id
    direction_name : e.g. "probe", "cmd_overall"
    layer : layer index
    alpha_multiplier : suggested alpha = multiplier * |separation|
    target_n_std : suggested target = y1_mean + n_std * y1_std

    Returns
    -------
    dict with ``suggested_alpha``, ``suggested_target``, ``separation``,
    ``y0_mean``, ``y0_std``, ``y1_mean``, ``y1_std``, ``cohens_d``.
    """
    s = stats[constraint][f"L{layer}"][direction_name]
    y0_mean = s["y0"]["mean"]
    y0_std = s["y0"]["std"]
    y1_mean = s["y1"]["mean"]
    y1_std = s["y1"]["std"]
    separation = y1_mean - y0_mean
    return {
        "suggested_alpha": alpha_multiplier * abs(separation),
        "suggested_target": y1_mean + target_n_std * y1_std,
        "separation": separation,
        "y0_mean": y0_mean,
        "y0_std": y0_std,
        "y1_mean": y1_mean,
        "y1_std": y1_std,
        "cohens_d": s.get("cohens_d"),
    }


# ── Summarization ──────────────────────────────────────────────────────────


def summarize(responses: list[dict], label: str = "") -> dict:
    """Coherence-score responses and print per-constraint breakdown.

    Parameters
    ----------
    responses : list of response dicts from :func:`generate`
    label : description of this experiment (printed in header)

    Returns
    -------
    dict with ``metrics`` (overall genuine_scr etc.) and ``per_cell``
    (per constraint×direction breakdown).
    """
    scores = [score_coherence(r["text"]) for r in responses]
    labels = [r["label"] for r in responses]
    metrics = compute_genuine_scr(labels, scores)

    print(f"  {label}: genuine_scr={metrics['genuine_scr']:.3f} "
          f"raw_scr={metrics['raw_scr']:.3f} "
          f"quality={metrics['quality_breakdown']}")

    # Group by (conflict_id, direction)
    groups: dict[tuple, list] = defaultdict(list)
    for r, s in zip(responses, scores):
        groups[(r.get("conflict_id", "?"), r.get("direction", "?"))].append((r, s))

    per_cell = {}
    for (cid, d), items in sorted(groups.items()):
        sub_labels = [r["label"] for r, _ in items]
        sub_scores = [s for _, s in items]
        sub_metrics = compute_genuine_scr(sub_labels, sub_scores)
        n = len(items)
        n_gen_sys = sub_metrics["genuine_system"]
        n_gen = sub_metrics["n_genuine"]
        # Count baseline flips
        n_flipped = sum(
            1 for r, _ in items
            if r.get("baseline_label") == "followed_user" and r["label"] == "followed_system"
        )
        print(f"    {cid:35s} {d:6s}: "
              f"gen_sys={n_gen_sys:2d}/{n_gen} "
              f"genuine_scr={sub_metrics['genuine_scr']:.3f} "
              f"flipped={n_flipped}/{n}")
        per_cell[f"{cid}_{d}"] = {
            **sub_metrics,
            "n_flipped": n_flipped,
            "n_total": n,
        }

    return {"metrics": metrics, "per_cell": per_cell}


# ── Saving results ─────────────────────────────────────────────────────────


def save_experiment(
    name: str,
    config: dict,
    responses: list[dict],
    out_dir: str | Path,
    summary: dict | None = None,
    notes: str = "",
) -> Path:
    """Save one experiment's results with coherence scoring and agent notes.

    Parameters
    ----------
    name : experiment name (used as filename)
    config : steering config dict
    responses : list of response dicts from :func:`generate`
    out_dir : directory to save to
    summary : optional pre-computed summary from :func:`summarize`
    notes : free-form observations, hypotheses, qualitative impressions.
        Write what you noticed: which responses looked genuine, what
        surprised you, what to try next, why you chose this config.

    Returns
    -------
    Path to saved file.
    """
    if summary is None:
        summary = summarize(responses, name)

    # Annotate responses with coherence scores
    scores = [score_coherence(r["text"]) for r in responses]
    for r, s in zip(responses, scores):
        r["quality"] = s.quality.value
        r["rep3"] = s.repetition_3gram
        r["rep5"] = s.repetition_5gram

    out_path = Path(out_dir) / f"{name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config,
        "notes": notes,
        "n_samples": len(responses),
        **summary["metrics"],
        "per_cell": summary["per_cell"],
        "responses": responses,
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"  Saved: {out_path} ({len(responses)} samples)")
    return out_path
