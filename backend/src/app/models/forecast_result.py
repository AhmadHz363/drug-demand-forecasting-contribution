"""Forecast results — persisted ensemble predictions per drug and date."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, Integer, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ForecastResult(Base):
    """Calibrated forecast quantiles stored at inference time."""

    __tablename__ = "forecast_results"
    __table_args__ = (
        UniqueConstraint(
            "drug_code",
            "center_syn_id",
            "generated_at",
            "forecast_date",
            name="uq_forecast_results_drug_center_generated_date",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_code: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    center_syn_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    forecast_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    p5: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    p10: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    p50: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    p90: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    p95: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    model_weight_sarima: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    model_weight_lgbm: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    model_weight_classical: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
