"""
Run all Phase 0 experiments using the hash-based dedup runner.

Usage:
    source .venv/bin/activate && python run_experiments.py

Results are saved to data/results/{model_safe_name}.jsonl
"""
import logging
import sys

from src.config import load_config
from src.api_client import HFClient
from src.experiment import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    config = load_config("config/experiment.yaml")
    client = HFClient()
    runner = ExperimentRunner(config, client)

    keys = runner.generate_experiment_keys()
    logger.info(f"Generated {len(keys)} experiment keys")
    logger.info(f"Models: {config.models}")
    logger.info(f"Instances per cell: {config.generation.instances_per_cell}")

    results = runner.run_all_with_dedup()

    total = sum(len(v) for v in results.values())
    logger.info(f"Done. {total} new results across {len(results)} models.")


if __name__ == "__main__":
    main()
