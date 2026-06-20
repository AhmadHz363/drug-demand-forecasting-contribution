"""Unique drugs derived from receipt ingestion."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.drug_receipt import DrugReceipt


class Drug(Base):
    """One row per unique drug_code seen in drug_receipts."""

    __tablename__ = "drugs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_code: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    drug_name: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    drug_category: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    receipt_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    receipts: Mapped[list[DrugReceipt]] = relationship(
        "DrugReceipt",
        back_populates="drug",
    )
