"""Quick smoke test for a single conflict against a vLLM server.

Generates a small set of prompts, sends them to vLLM, scores responses,
and computes per-condition accuracy and overall BA. Used to validate new
or redesigned conflicts before committing to a full experiment run.

Usage:
    uv run python -m phase0_v2.calibration.smoke_test \
      --conflict forbidden_words \
      --vllm-url http://localhost:8000/v1 \
      --model meta-llama/Llama-3.1-8B-Instruct

    # Focus on baseline conditions with more tasks:
    uv run python -m phase0_v2.calibration.smoke_test \
      --conflict forbidden_words \
      --vllm-url http://localhost:8000/v1 \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --conditions A,B \
      --n-tasks 10

    # Save raw results for inspection:
    uv run python -m phase0_v2.calibration.smoke_test \
      --conflict forbidden_words \
      --vllm-url http://localhost:8000/v1 \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --output smoke_results.jsonl
"""

import argparse
import json
import logging
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..conflicts.registry import get_conflict
from ..src.api_client import VLLMClient
from ..src.config import ExperimentConfig, load_config, Task
from ..src.experiment import build_record
from ..src.prompts import PromptGenerator

logger = logging.getLogger(__name__)

# Default: first task from each of the 4 categories in synthetic_tasks.yaml
_DEFAULT_TASK_IDS = ["photosynthesis", "used_car", "renewable_energy", "french_revolution"]


def _select_tasks(config: ExperimentConfig, n_tasks: int | None) -> list[Task]:
    """Select tasks for smoke testing.

    Strategy: pick the first task from each category for diversity,
    then fill remaining slots sequentially. If n_tasks is specified,
    take the first n_tasks from the full list.
    """
    all_tasks = list(config.tasks)
    if not all_tasks:
        raise ValueError("No synthetic tasks found in config")

    if n_tasks is not None:
        return all_tasks[:n_tasks]

    # Default: pick the 4 category-representative tasks
    by_id = {t.id: t for t in all_tasks}
    selected = []
    for tid in _DEFAULT_TASK_IDS:
        if tid in by_id:
            selected.append(by_id[tid])
    # Fallback if tasks.yaml changed and defaults don't match
    if not selected:
        selected = all_tasks[:4]
    return selected


def _compute_metrics(records: list[dict]) -> dict:
    """Compute per-condition accuracy and overall BA from scored records."""
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if rec.get("error"):
            continue
        by_condition[rec["condition"]].append(rec)

    metrics: dict = {"conditions": {}, "total_records": len(records)}
    errors = sum(1 for r in records if r.get("error"))
    metrics["errors"] = errors

    for cond in sorted(by_condition.keys()):
        recs = by_condition[cond]
        total = len(recs)
        correct = sum(1 for r in recs if r["label"] == r["expected_label"])
        acc = correct / total if total else 0.0

        # Per-direction breakdown
        by_dir: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            by_dir[r["direction"]].append(r)
        dir_acc = {}
        for d, d_recs in sorted(by_dir.items()):
            d_correct = sum(1 for r in d_recs if r["label"] == r["expected_label"])
            dir_acc[d] = d_correct / len(d_recs) if d_recs else 0.0

        # Label distribution
        label_counts = Counter(r["label"] for r in recs)

        metrics["conditions"][cond] = {
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "by_direction": dir_acc,
            "labels": dict(label_counts),
        }

    # BA for condition C (the main metric)
    c_recs = by_condition.get("C", [])
    if c_recs:
        # BA = average of per-direction accuracy
        by_dir_c: dict[str, list[dict]] = defaultdict(list)
        for r in c_recs:
            by_dir_c[r["direction"]].append(r)
        dir_accs = []
        for d, d_recs in by_dir_c.items():
            correct = sum(1 for r in d_recs if r["label"] == r["expected_label"])
            dir_accs.append(correct / len(d_recs) if d_recs else 0.0)
        metrics["ba"] = sum(dir_accs) / len(dir_accs) if dir_accs else 0.0
    else:
        metrics["ba"] = None

    # Per-constraint SBR/UCR from condition A/B baselines
    # SBR(a) = Cond A, a_to_b (system has constraint_a)
    # SBR(b) = Cond A, b_to_a (system has constraint_b)
    # UCR(a) = Cond B, b_to_a (user has constraint_a)
    # UCR(b) = Cond B, a_to_b (user has constraint_b)
    a_recs = by_condition.get("A", [])
    b_recs = by_condition.get("B", [])

    def _dir_acc(recs: list[dict], direction: str, expected: str) -> float | None:
        subset = [r for r in recs if r["direction"] == direction]
        if not subset:
            return None
        return sum(1 for r in subset if r["label"] == expected) / len(subset)

    if a_recs:
        metrics["sbr_a"] = _dir_acc(a_recs, "a_to_b", "followed_system")
        metrics["sbr_b"] = _dir_acc(a_recs, "b_to_a", "followed_system")
    if b_recs:
        metrics["ucr_a"] = _dir_acc(b_recs, "b_to_a", "followed_user")
        metrics["ucr_b"] = _dir_acc(b_recs, "a_to_b", "followed_user")

    # BA from baselines (mean of 4 component BAs, matching calibration-report)
    rates = [metrics.get(k) for k in ("sbr_a", "ucr_a", "sbr_b", "ucr_b")]
    valid = [r for r in rates if r is not None]
    if valid:
        metrics["baseline_ba"] = sum(valid) / len(valid)

    return metrics


def _print_report(metrics: dict, conflict_id: str) -> None:
    """Print a human-readable smoke test report."""
    print(f"\n{'=' * 60}")
    print(f"SMOKE TEST: {conflict_id}")
    print(f"{'=' * 60}")
    print(f"Total records: {metrics['total_records']}, Errors: {metrics['errors']}")

    # Per-constraint baseline rates
    has_baselines = any(k in metrics for k in ("sbr_a", "ucr_a", "sbr_b", "ucr_b"))
    if has_baselines:
        print(f"\nBaseline rates (per constraint):")
        print(f"{'':>12} {'SBR':>8} {'UCR':>8}")
        print(f"{'':>12} {'(system)':>8} {'(user)':>8}")
        print("-" * 32)
        for label in ("a", "b"):
            sbr = metrics.get(f"sbr_{label}")
            ucr = metrics.get(f"ucr_{label}")
            sbr_s = f"{sbr:.3f}" if sbr is not None else "  n/a"
            ucr_s = f"{ucr:.3f}" if ucr is not None else "  n/a"
            print(f"  {label:>10} {sbr_s:>8} {ucr_s:>8}")
        if "baseline_ba" in metrics:
            print(f"\n  Baseline BA: {metrics['baseline_ba']:.3f}")
            if metrics["baseline_ba"] >= 0.95:
                print("  PASS — scorer is reliable, ready for full experiments")
            elif metrics["baseline_ba"] >= 0.90:
                print("  MARGINAL — check failures, scorer may need tuning")
            else:
                print("  FAIL — scorer or constraint issue, investigate before proceeding")

    # Per-condition detail (always shown)
    print(f"\nPer-condition breakdown:")
    print(f"{'Cond':<6} {'Total':>6} {'Correct':>8} {'Acc':>8}  Direction detail")
    print("-" * 60)
    for cond, info in sorted(metrics["conditions"].items()):
        dir_str = ", ".join(
            f"{d}={a:.2f}" for d, a in sorted(info["by_direction"].items())
        )
        print(
            f"{cond:<6} {info['total']:>6} {info['correct']:>8} "
            f"{info['accuracy']:>7.3f}  {dir_str}"
        )
        if info["labels"]:
            label_str = ", ".join(
                f"{l}={c}" for l, c in sorted(info["labels"].items())
            )
            print(f"{'':>6} labels: {label_str}")


def run_smoke_test(
    conflict_id: str,
    vllm_url: str,
    model_id: str,
    config: ExperimentConfig,
    conditions: list[str] | None = None,
    n_tasks: int | None = None,
    concurrent: int = 4,
) -> tuple[list[dict], dict]:
    """Run smoke test and return (records, metrics).

    Args:
        conflict_id: ID of the conflict to test.
        vllm_url: vLLM server URL (e.g. http://localhost:8000/v1).
        model_id: Model ID (must match vLLM's loaded model).
        config: Experiment configuration.
        conditions: List of conditions to test (default: all).
        n_tasks: Number of tasks (default: 4 category-representative).
        concurrent: Number of concurrent API calls.

    Returns:
        Tuple of (list of result records, metrics dict).
    """
    conflict = get_conflict(conflict_id)
    if conflict is None:
        raise ValueError(f"Unknown conflict_id: {conflict_id}")

    # Select tasks
    tasks = _select_tasks(config, n_tasks)
    logger.info("Using %d tasks: %s", len(tasks), [t.id for t in tasks])

    # Generate prompts
    generator = PromptGenerator(config)
    prompts = generator.generate_for_conflict(conflict, tasks)

    # Filter conditions
    if conditions:
        allowed = set(conditions)
        prompts = [p for p in prompts if p.condition in allowed]

    logger.info("Generated %d prompts", len(prompts))
    if not prompts:
        raise ValueError("No prompts generated — check conditions and conflict configuration")

    # Run through vLLM
    client = VLLMClient(base_url=vllm_url)
    records: list[dict] = []
    lock = threading.Lock()

    def run_one(prompt):
        messages = []
        if prompt.system_message:
            messages.append({"role": "system", "content": prompt.system_message})
        messages.append({"role": "user", "content": prompt.user_message})

        try:
            response = client.chat_completion(
                model_id=model_id,
                messages=messages,
                temperature=config.generation.temperature,
                max_tokens=config.generation.max_tokens,
            )
            response_text = response.content if hasattr(response, "content") else str(response)
            error = response.error if hasattr(response, "error") else None
        except Exception as e:
            response_text = ""
            error = str(e)

        record = build_record(
            prompt=prompt,
            response=response_text,
            conflict=conflict,
            model=model_id,
            config=config,
            error=error,
        )
        return record

    print(f"Running {len(prompts)} prompts against {model_id}...")
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = {executor.submit(run_one, p): p for p in prompts}
        done = 0
        for future in as_completed(futures):
            try:
                record = future.result()
                with lock:
                    records.append(record)
            except Exception as e:
                logger.error("Prompt failed: %s", e)
            done += 1
            if done % 10 == 0 or done == len(prompts):
                print(f"  {done}/{len(prompts)} done", flush=True)

    metrics = _compute_metrics(records)
    return records, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Smoke test a conflict against vLLM")
    parser.add_argument("--conflict", required=True, help="Conflict ID to test")
    parser.add_argument("--vllm-url", required=True, help="vLLM server URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--model", required=True, help="Model ID (must match vLLM)")
    parser.add_argument("--config", default="phase0_v2/config/experiment.yaml", help="Experiment config YAML")
    parser.add_argument("--conditions", default=None, help="Comma-separated conditions (default: A,B,C,D)")
    parser.add_argument("--n-tasks", type=int, default=None, help="Number of tasks (default: 4 representative)")
    parser.add_argument("--concurrent", type=int, default=16, help="Concurrent API calls")
    parser.add_argument("--output", default=None, help="Save raw results to JSONL file")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    conditions = None
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",")]

    records, metrics = run_smoke_test(
        conflict_id=args.conflict,
        vllm_url=args.vllm_url,
        model_id=args.model,
        config=config,
        conditions=conditions,
        n_tasks=args.n_tasks,
        concurrent=args.concurrent,
    )

    _print_report(metrics, args.conflict)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"\nRaw results saved to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
