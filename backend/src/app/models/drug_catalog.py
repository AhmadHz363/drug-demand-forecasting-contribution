"""Drug catalog — metadata for known drugs used by the Cold Start module."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DrugCatalog(Base):
    """Known drug metadata for embedding, KNN similarity, and catalog management."""

    __tablename__ = "drug_catalog"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_code: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    drug_name: Mapped[str] = mapped_column(String(512), nullable=False)
    therapeutic_class: Mapped[str] = mapped_column(String(128), nullable=False)
    atc_category: Mapped[str] = mapped_column(String(32), nullable=False)
    pharmaceutical_form: Mapped[str] = mapped_column(String(64), nullable=False)
    ven_class: Mapped[str] = mapped_column(String(1), nullable=False)
    abc_class: Mapped[str] = mapped_column(String(1), nullable=False)
    unit_price_tier: Mapped[int] = mapped_column(Integer, nullable=False)
    requires_refrigeration: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_controlled_substance: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    average_shelf_life_days: Mapped[int] = mapped_column(Integer, nullable=False)
    route_of_administration: Mapped[str] = mapped_column(String(32), nullable=False)
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
