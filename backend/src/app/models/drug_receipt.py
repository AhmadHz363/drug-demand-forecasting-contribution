from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Date, DateTime, ForeignKey, Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.category import Category
    from app.models.drug import Drug


class DrugReceipt(Base):
    """Persisted drug receipt line from hospital pharmacy data."""

    __tablename__ = "drug_receipts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("drugs.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    category_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("categories.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )

    receipt_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    line_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    drug_category: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    center_receipt: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    receipt_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    movement_number: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    movement_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    drug_code: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    drug_name: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    month: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    center_syn_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    quantity: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    unit_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    total_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    admission_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    room_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    bed_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    doctor_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

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

    drug: Mapped[Drug] = relationship("Drug", back_populates="receipts")
    category: Mapped[Optional[Category]] = relationship("Category", back_populates="receipts")
