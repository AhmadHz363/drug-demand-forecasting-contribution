#!/usr/bin/env bash
# Run Cold Start Step 6 end-to-end integration validation.
set -euo pipefail
cd "$(dirname "$0")/.."
source bin/activate
export PYTHONPATH=src
alembic upgrade head
python scripts/validate_cold_start_step6.py --fast "$@"
