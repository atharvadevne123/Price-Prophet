"""Benchmark Price-Prophet API response times."""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
import urllib.request

logger = logging.getLogger(__name__)


def make_payload(category: str = "Electronics") -> bytes:
    """Serialise a forecast request payload for the given category.

    Args:
        category: Product category to include in the payload.

    Returns:
        JSON-encoded bytes ready to POST to /forecast.
    """
    data = {
        "category": category,
        "stock_level": 50,
        "competitor_price": 299.99,
        "demand_trend": 1.2,
        "margin_ratio": 0.3,
    }
    return json.dumps(data).encode()


def benchmark(base_url: str, n: int, endpoint: str) -> dict[str, float | int]:
    """Run n requests against endpoint and return latency statistics.

    Args:
        base_url: Base URL of the API (e.g., ``http://localhost:8000``).
        n: Number of requests to issue.
        endpoint: API endpoint path (e.g., ``/forecast``).

    Returns:
        Dict with min, max, mean, p50, p95, p99 latencies (ms) and error count.
    """
    url = f"{base_url}{endpoint}"
    payload = make_payload()
    latencies: list[float] = []
    errors = 0
    for _ in range(n):
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception as exc:
            logger.debug("Request failed: %s", exc)
            errors += 1
        else:
            latencies.append((time.perf_counter() - t0) * 1000)

    if not latencies:
        return {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0, "errors": errors}

    latencies.sort()
    n_l = len(latencies)
    return {
        "min": round(latencies[0], 2),
        "max": round(latencies[-1], 2),
        "mean": round(statistics.mean(latencies), 2),
        "p50": round(latencies[int(n_l * 0.50)], 2),
        "p95": round(latencies[int(n_l * 0.95)], 2),
        "p99": round(latencies[int(n_l * 0.99)], 2),
        "errors": errors,
    }


def main() -> None:
    """Entry point: parse args, run benchmark, print results."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Benchmark Price-Prophet API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--n", type=int, default=100, help="Number of requests")
    parser.add_argument("--endpoint", default="/forecast", help="Endpoint to benchmark")
    args = parser.parse_args()

    logger.info("Benchmarking %s%s with %d requests...", args.url, args.endpoint, args.n)
    stats = benchmark(args.url, args.n, args.endpoint)

    logger.info("  min:    %s ms", stats["min"])
    logger.info("  max:    %s ms", stats["max"])
    logger.info("  mean:   %s ms", stats["mean"])
    logger.info("  p50:    %s ms", stats["p50"])
    logger.info("  p95:    %s ms", stats["p95"])
    logger.info("  p99:    %s ms", stats["p99"])
    logger.info("  errors: %s/%d", stats["errors"], args.n)


if __name__ == "__main__":
    main()
