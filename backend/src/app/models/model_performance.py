"""Model performance — walk-forward validation metrics per drug and model."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ModelPerformance(Base):
    """sMAPE and interval coverage from walk-forward validation."""

    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_code: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(32), nullable=False)
    smape: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False)
    coverage_90: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
