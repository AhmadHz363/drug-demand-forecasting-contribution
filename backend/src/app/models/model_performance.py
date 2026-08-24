"""Persisted hold-out metrics per drug and model."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    drug_code: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(32), nullable=False)
    smape: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False)
    smape_normal_supply: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    mase: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    mase_normal_supply: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    coverage_90: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False, server_default="0")
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    training_run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    demand_segment: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    data_quality_status: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    weight_sarima: Mapped[Optional[float]] = mapped_column(Numeric(6, 4), nullable=True)
    weight_lgbm: Mapped[Optional[float]] = mapped_column(Numeric(6, 4), nullable=True)
    weight_classical: Mapped[Optional[float]] = mapped_column(Numeric(6, 4), nullable=True)
