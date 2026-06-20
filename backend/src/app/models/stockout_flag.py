"""Stockout flags — censored demand correction audit trail."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class StockoutFlag(Base):
    """One row per stockout day where demand was corrected for censoring."""

    __tablename__ = "stockout_flags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_code: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    center_syn_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    flag_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    observed_quantity: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    estimated_true_demand: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    correction_method: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
