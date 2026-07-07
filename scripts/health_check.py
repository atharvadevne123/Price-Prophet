#!/usr/bin/env python3
"""Health check script for Price-Prophet API."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request

logger = logging.getLogger(__name__)


def check_health(base_url: str) -> int:
    """Check the API health endpoint.

    Args:
        base_url: Base URL of the Price-Prophet API.

    Returns:
        Exit code: 0 for healthy, 1 for unhealthy.
    """
    try:
        url = f"{base_url.rstrip('/')}/health"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
        if data.get("status") == "ok":
            logger.info("OK: API is healthy (model_trained=%s)", data.get("model_trained"))
            return 0
        logger.warning("Unexpected response: %s", data)
        return 1
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return 1


def main() -> None:
    """Run the API health check."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Check Price-Prophet API health")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    sys.exit(check_health(args.url))


if __name__ == "__main__":
    main()
