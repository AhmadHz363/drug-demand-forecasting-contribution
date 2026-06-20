#!/usr/bin/env bash
# Run full Step 1 test suite (unit + DB integration).
set -euo pipefail
cd "$(dirname "$0")/.."
source bin/activate
export PYTHONPATH=src
pip install -q pytest 2>/dev/null || true
pytest tests/test_cold_start_step1.py -v --tb=short "$@"
