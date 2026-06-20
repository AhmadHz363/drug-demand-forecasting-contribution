"""Hospital census — bed occupancy and surgery volume by date."""

from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class HospitalCensus(Base):
    """Daily hospital operational metrics used as external demand predictors."""

    __tablename__ = "hospital_census"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    census_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    bed_occupancy_rate: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    weekly_surgery_count: Mapped[int] = mapped_column(Integer, nullable=False)
    center_syn_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
