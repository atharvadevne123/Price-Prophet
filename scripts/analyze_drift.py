"""Offline drift analysis tool: compare training vs production distributions."""
from __future__ import annotations

import argparse
import json
import sys


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
        print(f"Error loading {path}: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Unrecognised format in {path}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
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

    from app.monitoring import compute_drift, check_alerts, prediction_health

    drift = compute_drift(ref, cur)
    alerts = check_alerts(drift)
    health = prediction_health(cur)

    result = {
        "drift": drift,
        "alerts": alerts,
        "current_health": health,
        "n_reference": len(ref),
        "n_current": len(cur),
    }

    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults written to {args.output}", file=sys.stderr)

    if alerts:
        sys.exit(1)


if __name__ == "__main__":
    main()
