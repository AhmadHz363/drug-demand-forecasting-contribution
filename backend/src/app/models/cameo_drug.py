"""Drug attribute registry for CAMEO cold-start (from training Excel)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class CameoDrug(Base):
    """One hospital SKU with matched pharmacological attributes."""

    __tablename__ = "cameo_drugs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)

    generic_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    drug_class: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    indications: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    dosage_form: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    strength: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    route_of_administration: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    side_effects: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    contraindications: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    interaction_warnings_precautions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    storage_conditions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pregnancy_category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    reference: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    availability: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    input_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    input_drug_name: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    resolved_generic: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    matched_source_generic: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    match_status: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    match_method: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    receipt_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    synthetic_cells_filled: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    training_quality_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

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
