"""Phase 0 v2: Run instruction hierarchy experiments with class-based conflicts.

Usage:
    uv run python -m phase0_v2.run_experiments --dry-run
    uv run python -m phase0_v2.run_experiments --dry-run --conflicts forbidden_words,language_en_es
    uv run python phase0_v2/run_experiments.py --config phase0_v2/config/experiment.yaml
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
import logging
import random
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from phase0_v2.src.config import load_config
from phase0_v2.src.prompts import PromptGenerator, _deterministic_seed
from phase0_v2.src.experiment import ExperimentRunner, _make_experiment_key, compute_experiment_hash
from phase0_v2.src.api_client import HFClient, VLLMClient
from phase0_v2.src.lambda_cloud import LambdaCloudManager, load_lambda_config
from phase0_v2.conflicts.registry import get_all_conflicts, get_conflict
from phase0_v2.tasks.wildchat_tasks import load_wildchat_tasks, filter_compatible_tasks

logger = logging.getLogger(__name__)


def _run_work_items(
    work_items: list[tuple],
    runner: ExperimentRunner,
    model_semaphores: dict,
    model_locks: dict,
    max_workers: int,
) -> int:
    """Run work items through ThreadPoolExecutor with tqdm progress.

    Returns:
        Number of successfully completed items.
    """
    def run_one(item: tuple) -> dict | None:
        model_id, prompt, conflict, threshold_overrides = item
        with model_semaphores[model_id]:
            record = runner.run_single(
                prompt, conflict, model_id,
                threshold_overrides=threshold_overrides,
            )
            if record.get("error"):
                logger.warning(
                    "Skipping save for %s: %s",
                    record.get("prompt_id"), record["error"][:80],
                )
                return record
            with model_locks[model_id]:
                runner.append_record(model_id, record)
            return record

    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(run_one, item): item for item in work_items}
        for future in tqdm(
            as_completed(future_to_item),
            total=len(work_items),
            desc="Running experiments",
        ):
            item = future_to_item[future]
            try:
                future.result()
                completed_count += 1
            except Exception as e:
                model_id, prompt, _, _ = item
                logger.error("Error running %s on %s: %s", prompt.id, model_id, e)

    return completed_count


def main():
    parser = argparse.ArgumentParser(description="Run Phase 0 v2 experiments")
    parser.add_argument("--config", default="phase0_v2/config/experiment.yaml")
    parser.add_argument("--output-dir", default="phase0_v2/data/results")
    parser.add_argument("--model", type=str, help="Run only this model (default: all)")
    parser.add_argument(
        "--conflicts", type=str,
        help="Comma-separated conflict IDs (default: all)",
    )
    parser.add_argument("--n-tasks", type=int, help="Max synthetic tasks per conflict")
    parser.add_argument(
        "--conditions", type=str,
        help="Comma-separated conditions (default: A,B,C,D)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate prompts only, no API calls",
    )
    parser.add_argument(
        "--backend", choices=["hf", "vllm", "lambda"], default="hf",
        help="Inference backend: hf (HuggingFace API), vllm (self-hosted vLLM), lambda (auto-launch Lambda GPU)",
    )
    parser.add_argument(
        "--vllm-url", type=str,
        help="vLLM server URL (required for --backend vllm), e.g. http://localhost:8000/v1",
    )
    parser.add_argument(
        "--lambda-config", type=str, default="phase0_v2/config/lambda.yaml",
        help="Lambda config YAML file (for --backend lambda)",
    )
    args = parser.parse_args()

    if args.backend == "vllm" and not args.vllm_url:
        parser.error("--vllm-url is required when using --backend vllm")

    config = load_config(args.config)

    # Select conflicts
    if args.conflicts:
        conflict_ids = [cid.strip() for cid in args.conflicts.split(",")]
        conflicts = [get_conflict(cid) for cid in conflict_ids]
        conflicts = [c for c in conflicts if c is not None]
        if not conflicts:
            logger.error("No valid conflicts found for IDs: %s", conflict_ids)
            return
    else:
        conflicts = get_all_conflicts()

    # Filter non-invertible conflicts if configured
    if config.counterbalancing.require_invertible:
        before = len(conflicts)
        conflicts = [c for c in conflicts if c.supports_counterbalancing()]
        logger.info(
            "require_invertible: kept %d/%d conflicts (dropped %d non-invertible)",
            len(conflicts), before, before - len(conflicts),
        )

    # Load tasks
    synthetic_tasks = list(config.tasks)
    if args.n_tasks:
        synthetic_tasks = synthetic_tasks[: args.n_tasks]

    # Optionally load WildChat tasks
    wildchat_tasks = []
    wc_config = config.task_sources.get("wildchat")
    if wc_config and wc_config.file and wc_config.k_tasks_per_conflict:
        wc_path = Path(args.config).parent.parent / wc_config.file
        if wc_path.exists():
            wildchat_tasks = load_wildchat_tasks(wc_path)
            logger.info("Loaded %d WildChat tasks", len(wildchat_tasks))

    # Generate prompts
    generator = PromptGenerator(config)
    all_prompts = []
    for conflict in conflicts:
        # Synthetic tasks (all compatible)
        prompts = generator.generate_for_conflict(conflict, synthetic_tasks)
        all_prompts.extend(prompts)

        # WildChat tasks (filtered by compatibility)
        if wildchat_tasks:
            compatible = filter_compatible_tasks(wildchat_tasks, conflict.conflict_id)
            k = wc_config.k_tasks_per_conflict if wc_config else None
            if k and len(compatible) > k:
                random.seed(_deterministic_seed(config.seed, conflict.conflict_id, "wildchat_selection"))
                compatible = random.sample(compatible, k)
            wc_prompts = generator.generate_for_conflict(conflict, compatible)
            all_prompts.extend(wc_prompts)

    # Filter conditions if specified
    if args.conditions:
        allowed = set(c.strip() for c in args.conditions.split(","))
        all_prompts = [p for p in all_prompts if p.condition in allowed]

    # Shuffle prompts so that early stopping still gives good coverage
    # across conflicts, conditions, and directions.
    random.seed(config.seed)
    random.shuffle(all_prompts)

    logger.info(
        "Generated %d prompts for %d conflicts", len(all_prompts), len(conflicts)
    )

    if args.dry_run:
        print(f"DRY RUN: {len(all_prompts)} prompts generated")
        # Print summary
        cond_counts = Counter(p.condition for p in all_prompts)
        for cond, count in sorted(cond_counts.items()):
            print(f"  Condition {cond}: {count} prompts")
        dir_counts = Counter(p.direction for p in all_prompts)
        for d, count in sorted(dir_counts.items()):
            print(f"  Direction {d}: {count} prompts")
        conflict_counts = Counter(p.conflict_id for p in all_prompts)
        print(f"  Conflicts: {len(conflict_counts)}")
        return

    # Select models
    model_configs = config.models
    if args.model:
        model_configs = [mc for mc in model_configs if mc.id == args.model]
        if not model_configs:
            logger.error("Model '%s' not found in config", args.model)
            return

    # Build work items: (model_id, prompt, conflict, threshold_overrides) tuples, skipping completed
    # Use a temporary runner (no client) just for loading completed hashes
    _hash_runner = ExperimentRunner(
        config=config, client=None, output_dir=args.output_dir,
    )
    work_items: list[tuple[str, object, object, dict | None]] = []
    skipped = 0
    for model_config in model_configs:
        model_id = model_config.id

        # Filter prompts to exclude this model's excluded conflicts
        excluded = set(model_config.exclude_conflicts)
        if excluded:
            model_prompts = [p for p in all_prompts if p.conflict_id not in excluded]
            logger.info(
                "Model %s: excluded %d conflicts (%s), %d/%d prompts remain",
                model_id, len(excluded), ", ".join(sorted(excluded)),
                len(model_prompts), len(all_prompts),
            )
        else:
            model_prompts = all_prompts

        # Get threshold overrides for this model
        threshold_overrides = model_config.thresholds if model_config.thresholds else None

        completed_counts = _hash_runner.load_completed_hash_counts(model_id)
        logger.info(
            "Model %s: %d experiments already completed",
            model_id, sum(completed_counts.values()),
        )
        for prompt in model_prompts:
            conflict = get_conflict(prompt.conflict_id)
            if conflict is None:
                logger.warning("Conflict not found: %s", prompt.conflict_id)
                continue
            key = _make_experiment_key(prompt, model_id, config)
            h = compute_experiment_hash(key)
            if completed_counts.get(h, 0) > 0:
                skipped += 1
                continue
            work_items.append((model_id, prompt, conflict, threshold_overrides))

    logger.info("Pending: %d, Already done: %d", len(work_items), skipped)

    if not work_items:
        logger.info("All experiments already completed.")
        return

    # Interleave work items round-robin by model so all models get
    # dispatched to workers immediately rather than one model at a time.
    by_model_interleave: dict[str, list] = defaultdict(list)
    for item in work_items:
        by_model_interleave[item[0]].append(item)
    interleaved: list[tuple[str, object, object, dict | None]] = []
    queues = list(by_model_interleave.values())
    while any(queues):
        for q in queues:
            if q:
                interleaved.append(q.pop(0))
    work_items = interleaved

    # ── Backend-aware execution ──
    if args.backend == "hf":
        client = HFClient()
        concurrent_per_model = config.api.concurrent_per_model
        runner = ExperimentRunner(
            config=config, client=client,
            output_dir=args.output_dir,
            per_model_concurrency=concurrent_per_model,
        )
        model_ids = [mc.id for mc in model_configs]
        model_semaphores = {m: threading.Semaphore(concurrent_per_model) for m in model_ids}
        model_locks = {m: threading.Lock() for m in model_ids}
        max_workers = len(model_configs) * concurrent_per_model

        completed_count = _run_work_items(
            work_items, runner, model_semaphores, model_locks, max_workers,
        )
        logger.info("Completed: %d, Skipped: %d", completed_count, skipped)

    elif args.backend == "vllm":
        client = VLLMClient(base_url=args.vllm_url)
        concurrent_per_model = config.api.concurrent_per_model
        runner = ExperimentRunner(
            config=config, client=client,
            output_dir=args.output_dir,
            per_model_concurrency=concurrent_per_model,
        )
        model_ids = [mc.id for mc in model_configs]
        model_semaphores = {m: threading.Semaphore(concurrent_per_model) for m in model_ids}
        model_locks = {m: threading.Lock() for m in model_ids}
        max_workers = len(model_configs) * concurrent_per_model

        completed_count = _run_work_items(
            work_items, runner, model_semaphores, model_locks, max_workers,
        )
        logger.info("Completed: %d, Skipped: %d", completed_count, skipped)

    elif args.backend == "lambda":
        lambda_yaml_path = Path(args.lambda_config)
        import yaml as _yaml
        _lambda_data = _yaml.safe_load(lambda_yaml_path.read_text())
        lambda_concurrent = _lambda_data.get("defaults", {}).get("concurrent_per_model", 10)

        total_completed = 0
        by_model: dict[str, list] = defaultdict(list)
        for item in work_items:
            by_model[item[0]].append(item)

        for model_id, model_items in by_model.items():
            logger.info("Lambda: launching for %s (%d items)", model_id, len(model_items))
            lconfig = load_lambda_config(args.lambda_config, model_id)
            with LambdaCloudManager(lconfig) as mgr:
                client = mgr.get_client()
                runner = ExperimentRunner(
                    config=config, client=client,
                    output_dir=args.output_dir,
                    per_model_concurrency=lambda_concurrent,
                )
                sems = {model_id: threading.Semaphore(lambda_concurrent)}
                locks = {model_id: threading.Lock()}
                completed = _run_work_items(
                    model_items, runner, sems, locks,
                    max_workers=lambda_concurrent,
                )
                total_completed += completed
                logger.info("Model %s: %d/%d done", model_id, completed, len(model_items))

        logger.info("Lambda total: %d completed, %d skipped", total_completed, skipped)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
