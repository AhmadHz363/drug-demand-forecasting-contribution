"""Cleaned SHIELD-XR training panel audit rows."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ForecastTrainingData(Base):
    __tablename__ = "forecast_training_data"
    __table_args__ = (
        UniqueConstraint(
            "training_run_id",
            "drug_code",
            "demand_date",
            name="uq_forecast_training_data_run_drug_date",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    training_run_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    drug_code: Mapped[str] = mapped_column(String(255), nullable=False)
    demand_date: Mapped[date] = mapped_column(Date, nullable=False)
    demand_raw: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    demand_clean: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    drug_name: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    drug_category: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    sb_class: Mapped[str] = mapped_column(String(32), nullable=False)
    is_anomaly_day: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    is_dropout_day: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    ceiling: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
