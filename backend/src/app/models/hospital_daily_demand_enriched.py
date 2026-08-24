"""Cleaned hospital daily demand panel for SHIELD-XR training."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, Integer, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class HospitalDailyDemandEnriched(Base):
    """One SKU-day row in the enriched training panel."""

    __tablename__ = "hospital_daily_demand_enriched"
    __table_args__ = (
        UniqueConstraint(
            "source_file",
            "demand_date",
            "drug_code",
            name="uq_hospital_daily_demand_enriched_source_drug_date",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    source_file: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    demand_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    drug_code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    article: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    cat: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    demand: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    n_patients_drug: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    n_doctors_drug: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    top_cr_drug: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    top_cs_drug: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    top_dr_drug: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    n_unique_patients: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    n_admissions: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    n_unique_doctors: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    n_unique_cr: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    n_unique_cs: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    n_transactions: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    n_demand_txns: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    hospital_total_demand: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    age_mean: Mapped[Optional[float]] = mapped_column(Numeric(10, 4), nullable=True)
    age_median: Mapped[Optional[float]] = mapped_column(Numeric(10, 4), nullable=True)

    top1_cr: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    top1_cr_share: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False, server_default="0")
    top2_cr: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    top2_cr_share: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False, server_default="0")
    top3_cr: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    top3_cr_share: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False, server_default="0")

    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
