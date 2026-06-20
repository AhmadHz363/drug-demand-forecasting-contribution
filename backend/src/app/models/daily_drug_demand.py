"""Daily aggregated drug demand — materialized from ``drug_receipts``.

Each row is the sum of ``drug_receipts.quantity`` for a ``drug_code`` on a
calendar day. Refresh via ``sync_daily_demand_from_receipts`` after receipt
uploads or before batch training.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DailyDrugDemand(Base):
    """One row per drug per calendar day — total dispensed/receipt quantity."""

    __tablename__ = "daily_drug_demand"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_code: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    demand_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    total_quantity: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
