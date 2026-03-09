"""Data loading, configuration, tokenization, activation I/O, and extraction.

Provides :class:`ProbeConfig` for experiment tracking, utilities to load
behavioral results and prepare Condition C data, tokenizer helpers for
prompt formatting and token position mapping, and activation extraction
via TransformerLens or nnsight backends.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────


def find_repo_root() -> Path:
    """Walk up from cwd until ``pyproject.toml`` is found."""
    p = Path.cwd()
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return p


def load_sync_env(repo_root: Path) -> Path | None:
    """Auto-load the first ``*.sync.env`` from *repo_root* into ``os.environ``.

    Lines matching ``export KEY=VALUE`` are loaded via
    :func:`os.environ.setdefault` so they don't clobber existing variables.

    Returns the loaded file path, or ``None`` if no file was found.
    """
    pattern = re.compile(r"^export\s+(\w+)=(.*)")
    for env_file in sorted(repo_root.glob("*.sync.env")):
        with open(env_file) as f:
            for line in f:
                m = pattern.match(line.strip())
                if m:
                    key, val = m.group(1), m.group(2).strip("\"'")
                    os.environ.setdefault(key, val)
        return env_file  # load first file found, stop
    return None


@dataclass
class ProbeConfig:
    """Central configuration for a linear probing experiment run.

    Derived paths (``data_dir``, ``run_dir``, ``reports_dir``) are computed
    in ``__post_init__``.  If ``run_id`` is not set explicitly it is
    auto-generated as a short hash of the parameters that affect results.
    """

    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    label_mode: Literal["binary", "one_vs_rest"] = "binary"
    token_positions: list[str] = field(
        default_factory=lambda: ["last_prompt"]
    )
    cv_mode: Literal["stratified", "grouped"] = "grouped"
    n_cv_folds: int = 5
    max_samples: int | None = None
    use_scaler: bool = False
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    repo_root: Path = field(default_factory=find_repo_root)
    run_id: str | None = None

    # Auto-derived in __post_init__
    data_dir: Path = field(init=False)
    run_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = (
            self.repo_root / "phase0_behavioral_analysis" / "data" / "results"
        )
        if self.run_id is None:
            self.run_id = self._compute_run_id()
        self.run_dir = (
            self.repo_root
            / "phase1_linear_probing"
            / "data"
            / "runs"
            / self.run_id
        )
        self.reports_dir = self.run_dir / "reports"

    @property
    def safe_model_name(self) -> str:
        """Model name with ``/`` replaced by ``_``."""
        return self.model_name.replace("/", "_")

    def _data_hash(self) -> str:
        """SHA-256 hash of the source JSONL file (first 16 hex chars).

        Returns ``"no_data"`` if the file does not exist yet (e.g. in tests
        or before any experiments have been run).
        """
        safe_name = self.model_name.replace("/", "_")
        path = self.data_dir / f"{safe_name}_results.jsonl"
        if not path.exists():
            return "no_data"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def _compute_run_id(self) -> str:
        """Short hash of config parameters + data that affect experiment results."""
        key = json.dumps(
            {
                "model": self.model_name,
                "label": self.label_mode,
                "positions": sorted(self.token_positions),
                "cv_mode": self.cv_mode,
                "folds": self.n_cv_folds,
                "scaler": self.use_scaler,
                "max_samples": self.max_samples,
                "data": self._data_hash(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def save_config(self) -> Path:
        """Save config as JSON in *run_dir* for reproducibility."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        serializable = {
            "model_name": self.model_name,
            "label_mode": self.label_mode,
            "token_positions": self.token_positions,
            "cv_mode": self.cv_mode,
            "n_cv_folds": self.n_cv_folds,
            "max_samples": self.max_samples,
            "use_scaler": self.use_scaler,
            "device": self.device,
            "run_id": self.run_id,
        }
        path = self.run_dir / "config.json"
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        return path

    def ensure_dirs(self) -> None:
        """Create *run_dir* and *reports_dir* if they don't exist."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_results(
    data_dir: Path,
    model_name: str,
    config_path: str | Path = "phase0_v2/config/experiment.yaml",
) -> pd.DataFrame:
    """Load JSONL experiment results for *model_name* from *data_dir*.

    Filters out unregistered conflicts (stale/redesigned) and per-model
    excluded conflicts from the experiment config.
    """
    safe_name = model_name.replace("/", "_")
    path = Path(data_dir) / f"{safe_name}_results.jsonl"
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Filter to registered conflicts only
    from phase0_v2.conflicts.registry import get_all_conflicts
    valid_ids = {c.conflict_id for c in get_all_conflicts()}
    before = len(df)
    df = df[df["conflict_id"].isin(valid_ids)]

    # Apply per-model exclusions from experiment config
    config_path = Path(config_path)
    if config_path.exists():
        from phase0_v2.src.config import load_config
        config = load_config(config_path)
        model_config = next((m for m in config.models if m.id == model_name), None)
        if model_config and model_config.exclude_conflicts:
            excluded = set(model_config.exclude_conflicts)
            df = df[~df["conflict_id"].isin(excluded)]

    filtered = before - len(df)
    if filtered > 0:
        logger.info("Filtered %d/%d records (unregistered or excluded conflicts)", filtered, before)

    return df


def prepare_condition_c(
    df_all: pd.DataFrame,
    label_mode: str = "binary",
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Filter to Condition C, assign labels, and optionally subsample.

    Parameters
    ----------
    df_all : pd.DataFrame
        Full experiment results (all conditions).
    label_mode : str
        ``"binary"`` keeps only ``followed_system`` and ``followed_user``;
        ``"one_vs_rest"`` keeps all Condition C labels.
    max_samples : int or None
        Cap the number of samples (for fast debugging).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with a ``y`` column (1 = followed_system).
    """
    df = df_all[df_all["condition"] == "C"].copy()
    if label_mode == "binary":
        df = df[df["label"].isin(["followed_system", "followed_user"])].copy()
    df["y"] = (df["label"] == "followed_system").astype(int)
    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
    return df.reset_index(drop=True)


# ── Tokenizer Utilities ──────────────────────────────────────────────────────


def build_formatted_prompt(
    tokenizer, system_text: str, user_text: str
) -> str:
    """Apply the chat template to *system_text* and *user_text*.

    Parameters
    ----------
    tokenizer
        A HuggingFace tokenizer with ``apply_chat_template``.
    system_text, user_text : str
        System and user message contents.
    """
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def find_token_positions(
    tokenizer, system_text: str, user_text: str
) -> dict:
    """Compute a token-position map for the formatted prompt.

    Returns a dict with keys:

    - ``last_prompt`` (int): index of the last token
    - ``last_system`` (int): index of the last system-segment token
    - ``last_user`` (int): index of the last user-segment token
    - ``mean_all`` (tuple[int, int]): range covering all tokens
    - ``mean_system`` (tuple[int, int]): range covering system tokens
    - ``mean_user`` (tuple[int, int]): range covering user tokens
    """

    def _encode_len(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    full_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sys_str = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text}],
        tokenize=False,
        add_generation_prompt=False,
    )
    sys_user_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    n_full = _encode_len(full_str)
    n_sys = min(_encode_len(sys_str), n_full - 1)
    n_sys_user = min(_encode_len(sys_user_str), n_full - 1)

    n_sys = max(n_sys, 1)
    n_sys_user = max(n_sys_user, n_sys + 1)

    return {
        "last_prompt": n_full - 1,
        "last_system": n_sys - 1,
        "last_user": n_sys_user - 1,
        "mean_all": (0, n_full),
        "mean_system": (0, n_sys),
        "mean_user": (n_sys, n_sys_user),
    }


def precompute_prompts(
    df: pd.DataFrame, tokenizer
) -> tuple[list[str], list[dict], list[torch.Tensor]]:
    """Tokenize all rows and compute position maps.

    Parameters
    ----------
    df : pd.DataFrame
        Condition C dataframe with ``system_prompt`` and ``user_prompt``.
    tokenizer
        A HuggingFace tokenizer.

    Returns
    -------
    formatted_prompts : list[str]
    position_maps : list[dict]
    input_ids_list : list[torch.Tensor]
    """
    formatted_prompts: list[str] = []
    position_maps: list[dict] = []
    input_ids_list: list[torch.Tensor] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing"):
        fp = build_formatted_prompt(
            tokenizer, row["system_prompt"], row["user_prompt"]
        )
        pm = find_token_positions(
            tokenizer, row["system_prompt"], row["user_prompt"]
        )
        ids = tokenizer(
            fp, return_tensors="pt", add_special_tokens=False
        ).input_ids
        formatted_prompts.append(fp)
        position_maps.append(pm)
        input_ids_list.append(ids)

    return formatted_prompts, position_maps, input_ids_list


# ── Activation Helpers ────────────────────────────────────────────────────────


def slice_activation(
    act_tensor: torch.Tensor, pos_val: int | tuple[int, int]
) -> np.ndarray:
    """Extract activation(s) at a given token position.

    Parameters
    ----------
    act_tensor : torch.Tensor
        Shape ``(1, seq_len, d_model)``.
    pos_val : int or tuple[int, int]
        If ``int``, extract a single token vector.
        If ``(start, end)``, return the mean over ``[start, end)``.
    """
    if isinstance(pos_val, int):
        return act_tensor[0, pos_val, :].float().cpu().numpy()
    start, end = pos_val
    return act_tensor[0, start:end, :].float().mean(dim=0).cpu().numpy()


def build_activation_array(
    buffers: dict,
    n_samples: int,
    n_layers: int,
    token_positions: list[str],
) -> dict[str, np.ndarray]:
    """Convert per-layer buffers into ``{pos: (n_samples, n_layers, d_model)}`` arrays.

    Parameters
    ----------
    buffers : dict
        ``{pos_name: list[list[np.ndarray]]}`` — outer list indexed by layer,
        inner list indexed by sample.
    n_samples, n_layers : int
        Expected dimensions.
    token_positions : list[str]
        Position names to include.
    """
    return {
        pos: np.array(
            [
                [buffers[pos][layer][i] for layer in range(n_layers)]
                for i in range(n_samples)
            ]
        )
        for pos in token_positions
    }


def save_activations(activations: dict[str, np.ndarray], path: Path) -> None:
    """Save activation arrays as a compressed ``.npz`` file."""
    np.savez_compressed(path, **activations)
    print(f"Saved: {path}")


def load_activations(path: Path) -> dict[str, np.ndarray]:
    """Load activation arrays from a ``.npz`` file."""
    loaded = np.load(path)
    return {k: loaded[k] for k in loaded.files}


def filter_activations(
    activations: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    """Subset activation arrays by *indices* along the sample axis."""
    return {pos: arr[indices] for pos, arr in activations.items()}


# ── Activation Extraction ─────────────────────────────────────────────────────


def extract_activations_tl(
    input_ids_list: list[torch.Tensor],
    position_maps: list[dict],
    cfg: ProbeConfig,
) -> dict[str, np.ndarray]:
    """Extract residual-stream activations using TransformerLens.

    Loads a :class:`HookedTransformer`, runs each sample through the model,
    collects ``resid_post`` at every layer for each token position, saves
    the result to ``cfg.run_dir``, and frees GPU memory.

    Parameters
    ----------
    input_ids_list : list[torch.Tensor]
        Pre-tokenized input ids (each shape ``(1, seq_len)``).
    position_maps : list[dict]
        Per-sample token position dicts from :func:`find_token_positions`.
    cfg : ProbeConfig
        Experiment configuration.

    Returns
    -------
    dict[str, np.ndarray]
        ``{pos_name: array of shape (n_samples, n_layers, d_model)}``.
    """
    from transformer_lens import HookedTransformer

    model_tl = HookedTransformer.from_pretrained(
        cfg.model_name,
        dtype=torch.float16 if cfg.device == "cuda" else torch.float32,
        device=cfg.device,
    )
    model_tl.eval()

    n_layers = model_tl.cfg.n_layers
    d_model = model_tl.cfg.d_model
    print(f"Loaded via TL  |  layers={n_layers}  d_model={d_model}")

    buffers: dict[str, list[list]] = {
        pos: [[] for _ in range(n_layers)] for pos in cfg.token_positions
    }

    t0 = time.time()
    for ids, pm in tqdm(
        zip(input_ids_list, position_maps),
        total=len(input_ids_list),
        desc="TL extract",
    ):
        ids_gpu = ids.to(cfg.device)
        with torch.no_grad():
            _, cache = model_tl.run_with_cache(
                ids_gpu,
                prepend_bos=False,
                names_filter=lambda name: name.endswith("resid_post"),
            )
        for layer in range(n_layers):
            act = cache["resid_post", layer]
            for pos_name in cfg.token_positions:
                buffers[pos_name][layer].append(
                    slice_activation(act, pm[pos_name])
                )
        del cache
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(
        f"TL extraction: {elapsed:.1f}s  "
        f"({elapsed / len(input_ids_list):.2f}s/sample)"
    )

    activations = build_activation_array(
        buffers, len(input_ids_list), n_layers, cfg.token_positions
    )
    del buffers

    cfg.ensure_dirs()
    save_path = cfg.run_dir / f"act_tl_{cfg.safe_model_name}.npz"
    save_activations(activations, save_path)

    # Cleanup
    del model_tl
    gc.collect()
    if cfg.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(
            f"VRAM freed. Current usage: "
            f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )

    return activations


def extract_activations_nn(
    input_ids_list: list[torch.Tensor],
    position_maps: list[dict],
    cfg: ProbeConfig,
) -> dict[str, np.ndarray]:
    """Extract residual-stream activations using nnsight.

    Uses ``nnsight.LanguageModel`` with the explicit ``tracer.invoke()``
    pattern (see ``NNSIGHT_NOTES.md``).  Saves the result to
    ``cfg.run_dir`` and frees GPU memory.

    Parameters
    ----------
    input_ids_list : list[torch.Tensor]
        Pre-tokenized input ids (each shape ``(1, seq_len)``).
    position_maps : list[dict]
        Per-sample token position dicts from :func:`find_token_positions`.
    cfg : ProbeConfig
        Experiment configuration.

    Returns
    -------
    dict[str, np.ndarray]
        ``{pos_name: array of shape (n_samples, n_layers, d_model)}``.
    """
    from nnsight import LanguageModel

    model_nn = LanguageModel(
        cfg.model_name,
        device_map="auto",
        dispatch=True,
        torch_dtype=torch.float16 if cfg.device == "cuda" else torch.float32,
    )

    n_layers = len(list(model_nn.model.layers))
    print(f"Loaded via nnsight LanguageModel  |  layers={n_layers}")

    buffers: dict[str, list[list]] = {
        pos: [[] for _ in range(n_layers)] for pos in cfg.token_positions
    }

    t0 = time.time()
    for ids, pm in tqdm(
        zip(input_ids_list, position_maps),
        total=len(input_ids_list),
        desc="nnsight extract",
    ):
        ids_gpu = ids.to(cfg.device)

        saved = [None] * n_layers
        with model_nn.trace() as tracer:
            with tracer.invoke(input_ids=ids_gpu):
                for layer in range(n_layers):
                    saved[layer] = model_nn.model.layers[layer].output[0].save()

        for layer in range(n_layers):
            hs = saved[layer]  # (seq_len, d_model) — batch dim squeezed
            if hs.dim() == 3:
                hs = hs[0]
            for pos_name in cfg.token_positions:
                pos_val = pm[pos_name]
                if isinstance(pos_val, int):
                    vec = hs[pos_val, :].detach().float().cpu().numpy()
                else:
                    start, end = pos_val
                    vec = (
                        hs[start:end, :]
                        .detach()
                        .float()
                        .mean(0)
                        .cpu()
                        .numpy()
                    )
                buffers[pos_name][layer].append(vec)

        if cfg.device == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(
        f"nnsight extraction: {elapsed:.1f}s  "
        f"({elapsed / len(input_ids_list):.2f}s/sample)"
    )

    activations = build_activation_array(
        buffers, len(input_ids_list), n_layers, cfg.token_positions
    )
    del buffers

    cfg.ensure_dirs()
    save_path = cfg.run_dir / f"act_nn_{cfg.safe_model_name}.npz"
    save_activations(activations, save_path)

    # Cleanup
    from accelerate.hooks import remove_hook_from_submodules

    remove_hook_from_submodules(model_nn._model)
    del model_nn
    gc.collect()
    if cfg.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(
            f"VRAM after cleanup: "
            f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )

    return activations


def compare_backends(
    activations_tl: dict[str, np.ndarray],
    activations_nn: dict[str, np.ndarray],
    token_positions: list[str],
) -> None:
    """Print per-position cosine similarity between TL and nnsight activations.

    Reports similarity at layers 0, mid, and last, plus the mean across all
    layers. Values of 1.0 indicate identical activations.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    n_layers = activations_tl[token_positions[0]].shape[1]
    mid = n_layers // 2

    print(
        f"{'position':<16} {'layer 0':>10} {'layer mid':>10} "
        f"{'layer last':>10} {'mean all':>10}"
    )
    print("-" * 60)

    for pos in token_positions:
        tl = activations_tl[pos]
        nn = activations_nn[pos]

        def mean_cos(layer: int) -> float:
            a = tl[:, layer, :].astype(np.float32)
            b = nn[:, layer, :].astype(np.float32)
            sims = np.array(
                [
                    cosine_similarity(a[[i]], b[[i]])[0, 0]
                    for i in range(len(a))
                ]
            )
            return float(sims.mean())

        c0 = mean_cos(0)
        cm = mean_cos(mid)
        cl = mean_cos(n_layers - 1)
        ca = float(np.mean([mean_cos(l) for l in range(n_layers)]))
        print(f"{pos:<16} {c0:>10.4f} {cm:>10.4f} {cl:>10.4f} {ca:>10.4f}")

    print(
        "\n1.0 = identical, <1.0 = divergence between TL reimplementation "
        "and HF"
    )
