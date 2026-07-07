#!/usr/bin/env python3
"""Seed the database with synthetic predictions for development."""
from __future__ import annotations

import argparse
import logging
import os
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def seed(n: int = 50, db_url: str = "sqlite:///./prices.db") -> None:
    """Populate the database with n synthetic predictions.

    Args:
        n: Number of synthetic predictions to create.
        db_url: SQLAlchemy database URL.
    """
    os.environ.setdefault("DATABASE_URL", db_url)

    from app.database import Prediction, SessionLocal, init_db
    from app.features import CATEGORY_MAP, generate_synthetic_training_data
    from app.model import predict, train_model

    init_db()

    df = generate_synthetic_training_data(max(n * 2, 200))
    train_model(df, run_cv=False)

    categories = list(CATEGORY_MAP.keys())
    random.seed(0)

    with SessionLocal() as db:
        for i in range(n):
            category = random.choice(categories)
            competitor_price = round(random.uniform(10.0, 500.0), 2)
            stock_level = random.randint(0, 200)
            demand_trend = round(random.uniform(0.5, 2.5), 2)
            margin_ratio = round(random.uniform(0.1, 0.5), 2)
            features = {
                "category": category,
                "stock_level": stock_level,
                "competitor_price": competitor_price,
                "demand_trend": demand_trend,
                "margin_ratio": margin_ratio,
            }
            price = predict(features)
            pred = Prediction(
                category=category,
                input_price=competitor_price,
                predicted_price=round(price, 2),
                confidence_low=round(price * 0.94, 2),
                confidence_high=round(price * 1.06, 2),
                stock_level=stock_level,
                demand_trend=demand_trend,
            )
            db.add(pred)
        db.commit()
    logger.info("Seeded %d predictions into %s", n, db_url)


def main() -> None:
    """Entry point for the seed script."""
    parser = argparse.ArgumentParser(description="Seed Price-Prophet database")
    parser.add_argument("-n", "--count", type=int, default=50, help="Number of predictions to seed")
    parser.add_argument("--db", default="sqlite:///./prices.db", help="Database URL")
    args = parser.parse_args()
    seed(n=args.count, db_url=args.db)


if __name__ == "__main__":
    main()
