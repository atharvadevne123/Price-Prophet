"""Standalone training script with configurable parameters."""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train(n_samples: int, output: str, metrics_out: str) -> dict[str, Any]:
    """Train the model with n_samples synthetic rows, persist to output paths.

    Args:
        n_samples: Number of synthetic training samples to generate.
        output: Path to write the pickled model.
        metrics_out: Path to write the metrics JSON.

    Returns:
        Metrics dict with mae, rmse, and r2.
    """
    from app.features import generate_synthetic_training_data
    from app.model import train_model

    logger.info("Generating %d synthetic training samples...", n_samples)
    t0 = time.perf_counter()
    df = generate_synthetic_training_data(n_samples)
    logger.info("Data generated in %.2fs", time.perf_counter() - t0)

    logger.info("Training ensemble model...")
    t1 = time.perf_counter()
    metrics = train_model(df, model_path=output)
    logger.info("Training complete in %.2fs", time.perf_counter() - t1)
    logger.info("MAE=%.4f  RMSE=%.4f  R2=%.4f", metrics["mae"], metrics["rmse"], metrics["r2"])

    pathlib.Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(metrics_out).write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved to %s", metrics_out)
    return metrics


def main() -> None:
    """Entry point for the standalone training script."""
    parser = argparse.ArgumentParser(description="Train the Price-Prophet model")
    parser.add_argument("--n-samples", type=int, default=5000, help="Training samples")
    parser.add_argument("--output", default="models/price_model.pkl", help="Model output path")
    parser.add_argument("--metrics-out", default="models/metrics.json", help="Metrics output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    try:
        metrics = train(args.n_samples, args.output, args.metrics_out)
    except ImportError as exc:
        logger.error("Import error: %s — run from project root with pip install -e .", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)

    print(json.dumps({"status": "ok", "metrics": metrics}))


if __name__ == "__main__":
    main()
