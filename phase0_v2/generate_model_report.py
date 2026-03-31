#!/usr/bin/env python3
"""CLI: generate a single-model Phase 0 v2 interactive HTML report."""

import argparse
import logging
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from phase0_v2.src.model_report import generate_model_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate single-model Phase 0 v2 HTML report"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--results-dir",
        default="phase0_v2/data/results",
        help="Directory containing JSONL result files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path (auto-generated if omitted)",
    )
    parser.add_argument(
        "--config",
        default="phase0_v2/config/experiment.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help=(
            "Condition C style filter as key=value pairs, comma-separated. "
            "Default: no filter (all styles pooled). "
            "Example: --filter system_style=bare,user_style=with_instruction"
        ),
    )
    args = parser.parse_args()

    # Parse filter
    cond_c_filter = None
    if args.filter:
        cond_c_filter = {}
        for pair in args.filter.split(","):
            k, v = pair.strip().split("=", 1)
            cond_c_filter[k.strip()] = v.strip()

    path = generate_model_report(
        model_id=args.model,
        results_dir=args.results_dir,
        output_path=args.output,
        config_path=args.config,
        cond_c_filter=cond_c_filter,
    )
    print(f"Report generated: {path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
