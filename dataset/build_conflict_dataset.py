"""Build system-user conflict dataset from config YAML (WildChat prompts × N conflicts per prompt)."""

import yaml
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

DEFAULT_CONFIG = _SCRIPT_DIR / "dataset_config.yaml"

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and return config dict from YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_filtered_wildchat(jsonl_path: str | Path) -> list[dict[str, str]]:
    """Load JSONL rows with wildchat_id and clean_prompt."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Filtered WildChat file not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_dataset_from_registry(
    wildchat: list[dict[str, str]],
    num_conflicts_per_prompt: int,
    seed: int,
    max_prompts: int | None,
) -> list[dict[str, Any]]:
    """Build M×N examples: each prompt gets N conflicts; conflict is prefix or suffix 50/50."""
    from conflicts.registry import get_all_conflicts

    random.seed(seed)
    conflicts_list = get_all_conflicts()
    if len(conflicts_list) == 0:
        raise ValueError("No conflicts in registry")

    n_conflicts = len(conflicts_list)
    n_prompts = len(wildchat)
    if max_prompts is not None and max_prompts > 0:
        n_prompts = min(n_prompts, max_prompts)
    if n_prompts == 0:
        raise ValueError("No WildChat prompts to use")

    n_per_prompt = min(num_conflicts_per_prompt, n_conflicts) if num_conflicts_per_prompt > 0 else n_conflicts
    conflict_indices_pool = list(range(n_conflicts))

    dataset = []
    for w_idx in range(n_prompts):
        task_prompt = wildchat[w_idx]["clean_prompt"].strip()
        wildchat_id = wildchat[w_idx]["wildchat_id"]

        if n_per_prompt <= n_conflicts:
            chosen = random.sample(conflict_indices_pool, n_per_prompt)
        else:
            chosen = list(conflict_indices_pool) + random.choices(conflict_indices_pool, k=n_per_prompt - n_conflicts)

        for c_idx in chosen:
            conflict = conflicts_list[c_idx]
            args = conflict.sample_args() if conflict.get_instruction_args_keys() else {}
            system_prompt = conflict.build_system_prompt(**args)
            conflict_instruction = conflict.build_user_conflict_prompt(**args)
            instruction_args = conflict.get_instruction_args() or {}

            if random.random() < 0.5:
                user_prompt = f"{conflict_instruction}\n\n{task_prompt}"
                conflict_position = "prefix"
            else:
                user_prompt = f"{task_prompt}\n\n{conflict_instruction}"
                conflict_position = "append"

            canonical = json.dumps(
                [wildchat_id, conflict.conflict_id, conflict_position, instruction_args, task_prompt],
                sort_keys=True,
                ensure_ascii=False,
            )
            sample_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

            dataset.append({
                "sample_id": sample_id,
                "constraint_id": conflict.conflict_id,
                "verification_class": conflict.__class__.__name__,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "wildchat_id": wildchat_id,
                "task_prompt": task_prompt,
                "conflict_instruction": conflict_instruction,
                "conflict_position": conflict_position,
                "instruction_args": instruction_args,
            })

    return dataset


def run(config_path: str | Path, output_path_override: str | Path | None = None) -> tuple[list[dict[str, Any]], str]:
    """Load config, build dataset, write JSONL. Return (dataset, output path). Config paths are resolved relative to the config file directory."""
    config_path_resolved = Path(config_path).resolve()
    if not config_path_resolved.exists():
        raise FileNotFoundError(f"Config file not found: {config_path_resolved}")
    config = load_config(config_path_resolved)
    config_dir = config_path_resolved.parent

    def resolve_path(raw: str | Path, default: str) -> Path:
        p = Path(raw if raw else default)
        return p if p.is_absolute() else (config_dir / p).resolve()

    wildchat_path = resolve_path(config.get("wildchat_path"), "wildchat_filtered.jsonl")
    output_path = output_path_override if output_path_override is not None else resolve_path(config.get("output_path"), "conflict_dataset.jsonl")
    num_conflicts_per_prompt = int(config.get("num_conflicts_per_prompt", 5))
    max_prompts = config.get("max_prompts")
    if max_prompts is not None:
        max_prompts = int(max_prompts)
    seed = int(config.get("seed", 42))

    wildchat = load_filtered_wildchat(wildchat_path)
    if len(wildchat) == 0:
        raise ValueError(f"No rows in {wildchat_path}")

    dataset = build_dataset_from_registry(
        wildchat=wildchat,
        num_conflicts_per_prompt=num_conflicts_per_prompt,
        seed=seed,
        max_prompts=max_prompts,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return dataset, str(out_path)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build conflict dataset from config YAML (dataset size = M × N)")
    p.add_argument("--config", "-c", default=str(DEFAULT_CONFIG), help="Path to config YAML")
    p.add_argument("--output", "-o", default=None, help="Override output path from config")
    args = p.parse_args()

    dataset, out_path = run(args.config, output_path_override=args.output)
    m = len({r["wildchat_id"] for r in dataset})
    n = len(dataset) // m if m else 0
    print(f"Wrote {len(dataset)} examples (M={m} prompts × N={n} conflicts per prompt) to {out_path}")


if __name__ == "__main__":
    main()
