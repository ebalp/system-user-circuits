#!/usr/bin/env python3
"""CLI entry point: generate the interactive HTML report."""

import argparse
from src.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 0 HTML report")
    parser.add_argument("--results-dir", default="data/results",
                        help="Directory containing JSONL result files")
    parser.add_argument("--output", default="reports/report.html",
                        help="Output HTML file path")
    args = parser.parse_args()

    path = generate_report(results_dir=args.results_dir, output_path=args.output)
    print(f"Report generated: {path}")


if __name__ == "__main__":
    main()
