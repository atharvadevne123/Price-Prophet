"""Benchmark Price-Prophet API response times."""
from __future__ import annotations

import argparse
import statistics
import time
import urllib.request
import json


def make_payload(category: str = "Electronics") -> bytes:
    data = {
        "category": category,
        "stock_level": 50,
        "competitor_price": 299.99,
        "demand_trend": 1.2,
        "margin_ratio": 0.3,
    }
    return json.dumps(data).encode()


def benchmark(base_url: str, n: int, endpoint: str) -> dict[str, float]:
    """Run n requests against endpoint and return latency stats."""
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
        except Exception:
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
    parser = argparse.ArgumentParser(description="Benchmark Price-Prophet API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--n", type=int, default=100, help="Number of requests")
    parser.add_argument("--endpoint", default="/forecast", help="Endpoint to benchmark")
    args = parser.parse_args()

    # Warm up
    make_payload()

    print(f"Benchmarking {args.url}{args.endpoint} with {args.n} requests...")
    stats = benchmark(args.url, args.n, args.endpoint)

    print(f"  min:    {stats[min]} ms")
    print(f"  max:    {stats[max]} ms")
    print(f"  mean:   {stats[mean]} ms")
    print(f"  p50:    {stats[p50]} ms")
    print(f"  p95:    {stats[p95]} ms")
    print(f"  p99:    {stats[p99]} ms")
    print(f"  errors: {stats[errors]}/{args.n}")


if __name__ == "__main__":
    main()
