#!/usr/bin/env python3
"""CLI entry point: generate the Phase 0 v2 interactive HTML report.

Reads JSONL result files from the output directory, computes metrics
(SCR, UCR, SBR, Hierarchy Index, etc.), and generates an HTML report
with interactive Plotly figures.

Usage:
    uv run python phase0_v2/generate_report.py --results-dir phase0_v2/data/results
    uv run python phase0_v2/generate_report.py --results-dir phase0_v2/data/results --output reports/report.html
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from phase0_v2.src.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 0 v2 HTML report")
    parser.add_argument(
        "--results-dir",
        default="phase0_v2/data/results",
        help="Directory containing JSONL result files",
    )
    parser.add_argument(
        "--output",
        default="phase0_v2/reports/report.html",
        help="Output HTML file path",
    )
    parser.add_argument(
        "--config",
        default="phase0_v2/config/experiment.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()

    path = generate_report(
        results_dir=args.results_dir,
        output_path=args.output,
        config_path=args.config,
    )
    print(f"Report generated: {path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
