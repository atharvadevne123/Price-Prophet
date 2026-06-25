"""Standalone training script with configurable parameters."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Price-Prophet model")
    parser.add_argument("--n-samples", type=int, default=5000, help="Training samples")
    parser.add_argument("--output", default="models/price_model.pkl", help="Model output path")
    parser.add_argument("--metrics-out", default="models/metrics.json", help="Metrics output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    try:
        from app.features import generate_synthetic_training_data
        from app.model import train_model
    except ImportError as exc:
        logger.error("Import error: %s — run from project root with pip install -e .", exc)
        sys.exit(1)

    logger.info("Generating %d synthetic training samples (seed=%d)...", args.n_samples, args.seed)
    t0 = time.perf_counter()
    df = generate_synthetic_training_data(args.n_samples)
    elapsed = time.perf_counter() - t0
    logger.info("Data generated in %.2fs — %d rows, %d columns", elapsed, len(df), len(df.columns))

    logger.info("Training ensemble model...")
    t1 = time.perf_counter()
    metrics = train_model(df, model_path=args.output)
    elapsed = time.perf_counter() - t1
    logger.info("Training complete in %.2fs", elapsed)
    logger.info("MAE=%.4f  RMSE=%.4f  R2=%.4f", metrics["mae"], metrics["rmse"], metrics["r2"])

    import pathlib, json as _json  # noqa: E401
    pathlib.Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.metrics_out).write_text(_json.dumps(metrics, indent=2))
    logger.info("Metrics saved to %s", args.metrics_out)

    print(json.dumps({"status": "ok", "metrics": metrics}))


if __name__ == "__main__":
    main()
