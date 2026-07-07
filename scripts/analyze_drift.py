"""Offline drift analysis tool: compare training vs production distributions."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def load_values(path: str) -> list[float]:
    """Load a JSON file containing a list of float values."""
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return [float(v) for v in data]
        if isinstance(data, dict) and "values" in data:
            return [float(v) for v in data["values"]]
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.error("Error loading %s: %s", path, exc)
        sys.exit(1)
    logger.error("Unrecognised format in %s", path)
    sys.exit(1)


def run_analysis(ref: list[float], cur: list[float]) -> dict[str, Any]:
    """Compute drift metrics for two value lists.

    Args:
        ref: Reference distribution values.
        cur: Current distribution values.

    Returns:
        Dict with drift results, alerts, and health metrics.
    """
    from app.monitoring import check_alerts, compute_drift, prediction_health
    drift = compute_drift(ref, cur)
    return {
        "drift": drift,
        "alerts": check_alerts(drift),
        "current_health": prediction_health(cur),
        "n_reference": len(ref),
        "n_current": len(cur),
    }


def main() -> None:
    """Entry point: parse args, analyse drift, print and optionally save results."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Analyse distributional drift between two value lists"
    )
    parser.add_argument("reference", help="Path to JSON file with reference values")
    parser.add_argument("current", help="Path to JSON file with current values")
    parser.add_argument("--output", help="Write JSON results to this file")
    parser.add_argument("--threshold-ks", type=float, default=0.05, help="KS p-value threshold")
    parser.add_argument("--threshold-psi", type=float, default=0.2, help="PSI drift threshold")
    args = parser.parse_args()

    ref = load_values(args.reference)
    cur = load_values(args.current)

    result = run_analysis(ref, cur)

    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Results written to %s", args.output)

    if result["alerts"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
