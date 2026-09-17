#!/usr/bin/env python3
"""Deprecated — daily_drug_demand table was removed with SHIELD-XR."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "seed_daily_drug_demand.py is deprecated.\n"
        "Load pharmacy data via receipt upload (data ingestion) into drug_receipts instead."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
