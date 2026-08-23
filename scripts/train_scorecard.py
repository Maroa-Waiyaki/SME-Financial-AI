"""Train the credit-risk scorecard from the engineered feature set.

This script is designed to work in the lean Docker image because it only needs
numpy and pandas. It produces a persisted scorecard at `models/credit_scorecard.json`
that is used by `agents/credit_agent.py`.

Usage:
    python scripts/train_scorecard.py
    python scripts/train_scorecard.py --data-dir data/synthetic --output models/credit_scorecard.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.features.credit_features import build_features, load_data
from src.ml.scorecard import train_scorecard

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SME credit-risk scorecard")
    parser.add_argument(
        "--data-dir",
        default="data/synthetic",
        help="Directory containing the synthetic CSV files",
    )
    parser.add_argument(
        "--output",
        default="models/credit_scorecard.json",
        help="Path to write the trained scorecard JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--write-report",
        default="models/credit_scorecard_report.json",
        help="Path to write the full train/test/CV metrics report",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            "Run `python -m src.data.generate` first."
        )

    logger.info("Loading data from %s", data_dir)
    dfs = load_data(data_dir)
    features = build_features(dfs)
    logger.info("Feature frame shape: %s; positive rate: %.3f", features.shape, features["target"].mean())

    model, report = train_scorecard(features, seed=args.seed)
    report_path = Path(args.write_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    output_path = model.save(args.output)
    logger.info("Scorecard saved to %s", output_path)
    logger.info(
        "Test AUC=%.3f, Accuracy=%.3f, Precision=%.3f, Recall=%.3f, F1=%.3f",
        report["test"]["roc_auc"],
        report["test"]["accuracy"],
        report["test"]["precision"],
        report["test"]["recall"],
        report["test"]["f1"],
    )
    logger.info("Full report: %s", report_path)


if __name__ == "__main__":
    main()
