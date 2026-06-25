#!/usr/bin/env python3
"""Health check script for Price-Prophet API."""
from __future__ import annotations

import argparse
import sys
import urllib.request
import json


def check_health(base_url: str) -> int:
    """Check the API health endpoint.

    Args:
        base_url: Base URL of the Price-Prophet API.

    Returns:
        Exit code: 0 for healthy, 1 for unhealthy.
    """
    try:
        url = f"{base_url.rstrip(/)}/health"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
        if data.get("status") == "ok":
            print(f"OK: API is healthy (model_loaded={data.get(model_loaded)})")
            return 0
        else:
            print(f"WARN: Unexpected response: {data}")
            return 1
    except Exception as exc:
        print(f"ERROR: Health check failed: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    """Run the API health check."""
    parser = argparse.ArgumentParser(description="Check Price-Prophet API health")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    sys.exit(check_health(args.url))


if __name__ == "__main__":
    main()
