#!/usr/bin/env python3
"""Generate ASCII architecture diagram for Price-Prophet."""

from __future__ import annotations

DIAGRAM = """
price-prophet Architecture
==========================

  HTTP Clients / Dashboard (index.html)
              |
              v
    FastAPI Application (app/main.py)
    |--------------------------------------------|
    | POST /forecast   -> demand + optimal price |
    | POST /batch-forecast  -> bulk predictions  |
    | POST /train      -> retrain ensemble model |
    | POST /similar    -> FAISS similarity search|
    | GET  /drift      -> KS-test drift analysis |
    | GET  /metrics    -> training + health stats|
    | GET  /predictions -> history (SQLite/PG)   |
    | GET  /categories -> supported categories   |
    | GET  /summary    -> aggregate DB stats     |
    | GET  /health     -> liveness check         |
    | GET  /version    -> API version            |
    |--------------------------------------------|
              |
    +---------+----------+----------------+
    |                    |                |
    v                    v                v
app/model.py       app/retrieval.py  app/monitoring.py
(VotingRegressor   (ProductIndex /   (KS-test drift
 XGBoost +          FAISS cosine      detection,
 LightGBM +         similarity)       reference stats)
 RandomForest)
    |                                    |
    v                                    v
app/features.py                  app/database.py
(15 engineered                   (SQLAlchemy ORM
 feature pipeline)                predictions /
                                  model_metrics /
                                  drift_logs)
                                         |
                              +----------+----------+
                              |                     |
                              v                     v
                          SQLite (dev)         PostgreSQL (prod)
                              |
                              v
                    dags/retrain_dag.py
                    (Airflow daily DAG
                     drift-gated retrain)
"""


def main() -> None:
    """Print the architecture diagram."""
    print(DIAGRAM)


if __name__ == "__main__":
    main()
