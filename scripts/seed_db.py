#!/usr/bin/env python3
"""Seed the database with synthetic predictions for development."""
from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def seed(n: int = 50, db_url: str = "sqlite:///./price_prophet.db") -> None:
    """Populate the database with n synthetic predictions.

    Args:
        n: Number of synthetic predictions to create.
        db_url: SQLAlchemy database URL.
    """
    os.environ.setdefault("DATABASE_URL", db_url)

    from app.database import Prediction, SessionLocal, init_db
    from app.features import CATEGORY_MAP, engineer_features, generate_synthetic_training_data
    from app.model import load_model, predict, optimize_price
    import numpy as np

    init_db()
    model = load_model()
    X, _ = generate_synthetic_training_data(n_samples=n)
    categories = list(CATEGORY_MAP.keys())
    np.random.seed(0)

    db = SessionLocal()
    try:
        for i in range(n):
            features = X[i]
            demand = float(max(predict(model, features.reshape(1, -1))[0], 0.0))
            cost = float(features[0]) * 0.6
            opt = optimize_price(model, features, cost)
            cat = categories[int(X[i, 7]) % len(categories)]
            pred = Prediction(
                product_id=f"SEED-{i:04d}",
                category=cat,
                predicted_demand=round(demand, 2),
                optimized_price=opt["optimized_price"],
                confidence=min(1.0, demand / 100),
                features_used={"seeded": True},
            )
            db.add(pred)
        db.commit()
        logger.info("Seeded %d predictions into %s", n, db_url)
    finally:
        db.close()


def main() -> None:
    """Entry point for the seed script."""
    parser = argparse.ArgumentParser(description="Seed Price-Prophet database")
    parser.add_argument("-n", "--count", type=int, default=50, help="Number of predictions to seed")
    parser.add_argument("--db", default="sqlite:///./price_prophet.db", help="Database URL")
    args = parser.parse_args()
    seed(n=args.count, db_url=args.db)


if __name__ == "__main__":
    main()
