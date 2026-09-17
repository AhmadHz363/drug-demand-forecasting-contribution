"""Raw hospital pharmacy Excel export rows (unparsed source data)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class HospitalReceiptRaw(Base):
    """One row from a hospital pharmacy Excel export, stored as-is."""

    __tablename__ = "hospital_receipt_raw"
    __table_args__ = (
        UniqueConstraint(
            "source_file",
            "source_sheet",
            "excel_row_number",
            name="uq_hospital_receipt_raw_source_row",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    source_file: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    source_sheet: Mapped[str] = mapped_column(String(255), nullable=False)
    excel_row_number: Mapped[int] = mapped_column(Integer, nullable=False)

    doc: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    line: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cat: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    c_r: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    date_raw: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mov_num: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mov_des: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    article: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    m: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    c_s: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    qty: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    u_p: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    t_p: Mapped[Optional[float]] = mapped_column(Numeric(18, 4), nullable=True)
    mrn: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    ad_date_raw: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    r: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    u: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    age_raw: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    dr: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
