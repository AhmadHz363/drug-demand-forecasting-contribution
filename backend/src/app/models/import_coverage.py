"""Ledger of successfully ingested pharmacy receipt files and their date coverage."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ImportCoverage(Base):
    """One row per successfully ingested receipt file (or backfilled year).

    Coverage is hospital/file based: days inside ``[min_receipt_date, max_receipt_date]``
    are treated as observed ledger periods. Years with no import are coverage gaps.
    """

    __tablename__ = "import_coverage"
    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_import_coverage_content_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    min_receipt_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    max_receipt_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    center_scope: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="success")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

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
