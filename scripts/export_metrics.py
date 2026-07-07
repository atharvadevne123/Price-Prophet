"""Export model metrics and drift stats to JSON/CSV."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_predictions_from_db(database_url: str) -> list[dict[str, Any]]:
    """Load recent predictions from the database.

    Args:
        database_url: SQLAlchemy database URL.

    Returns:
        List of prediction dicts with id, category, predicted_price, created_at.
    """
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(database_url)
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT id, category, predicted_price, created_at "
                "FROM predictions ORDER BY created_at DESC LIMIT 1000"
            )).fetchall()
        return [
            {"id": r[0], "category": r[1], "predicted_price": r[2], "created_at": str(r[3])}
            for r in rows
        ]
    except Exception as exc:
        logger.error("DB error: %s", exc)
        return []


def main() -> None:
    """Entry point: export model metrics and predictions to JSON/CSV."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Export Price-Prophet metrics")
    parser.add_argument("--db", default="sqlite:///./prices.db", help="Database URL")
    parser.add_argument("--model-metrics", default="models/metrics.json", help="Model metrics JSON")
    parser.add_argument("--output-json", help="Write combined output to JSON file")
    parser.add_argument("--output-csv", help="Write predictions to CSV file")
    args = parser.parse_args()

    try:
        with open(args.model_metrics) as f:
            model_metrics = json.load(f)
    except (OSError, json.JSONDecodeError):
        model_metrics = {}

    predictions = load_predictions_from_db(args.db)

    output = {
        "model_metrics": model_metrics,
        "total_predictions": len(predictions),
        "predictions_sample": predictions[:10],
    }

    print(json.dumps(output, indent=2))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("JSON written to %s", args.output_json)

    if args.output_csv and predictions:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
            writer.writeheader()
            writer.writerows(predictions)
        logger.info("CSV written to %s", args.output_csv)


if __name__ == "__main__":
    main()
