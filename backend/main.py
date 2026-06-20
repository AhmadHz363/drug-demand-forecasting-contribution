"""Shim so `uvicorn main:app` works when cwd is backend/ (app package lives under src/)."""

from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from app.main import app  # noqa: E402

__all__ = ["app"]
