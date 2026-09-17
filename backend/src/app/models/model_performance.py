"""Model performance — walk-forward validation metrics per drug and model."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ModelPerformance(Base):
    """Walk-forward validation metrics persisted after each training run."""

    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_code: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(32), nullable=False)
    smape: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False)
    smape_normal_supply: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    coverage_90: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    mase: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    mase_normal_supply: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    smape_7day_full: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    smape_30day_full: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    mase_7day_full: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    mase_30day_full: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    smape_7day_normal: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    smape_30day_normal: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    mase_7day_normal: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    mase_30day_normal: Mapped[Optional[float]] = mapped_column(Numeric(8, 4), nullable=True)
    training_run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    demand_segment: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    data_quality_status: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    weight_sarima: Mapped[Optional[float]] = mapped_column(Numeric(6, 4), nullable=True)
    weight_lgbm: Mapped[Optional[float]] = mapped_column(Numeric(6, 4), nullable=True)
    weight_classical: Mapped[Optional[float]] = mapped_column(Numeric(6, 4), nullable=True)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
