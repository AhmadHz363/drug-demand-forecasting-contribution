"""Supplier lead times — per-drug delivery reliability constants."""

from __future__ import annotations

from sqlalchemy import Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SupplierLeadTime(Base):
    """Historical supplier lead-time statistics per drug."""

    __tablename__ = "supplier_lead_times"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_code: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    avg_lead_time_days: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    lead_time_std_days: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    reliability_score: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
