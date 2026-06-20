"""Pytest configuration — disable JWT gate so existing integration tests stay unchanged."""

from __future__ import annotations

import os

os.environ.setdefault("AUTH_ENABLED", "false")
