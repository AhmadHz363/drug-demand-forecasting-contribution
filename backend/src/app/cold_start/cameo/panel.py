"""Weekly demand panel + Syntetos–Boylan classification from enriched hospital data."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models.hospital_daily_demand_enriched import HospitalDailyDemandEnriched

MIN_WEEKS = 30


def sb_class(series: np.ndarray) -> str:
    s = np.asarray(series, dtype=float)
    nz = s[s > 0]
    if len(nz) < 2:
        return "intermittent"
    adi = len(s) / len(nz)
    cv2 = (nz.std() / (nz.mean() + 1e-9)) ** 2
    if adi <= 1.32 and cv2 <= 0.49:
        return "smooth"
    if adi > 1.32 and cv2 <= 0.49:
        return "intermittent"
    if adi <= 1.32 and cv2 > 0.49:
        return "erratic"
    return "lumpy"


def load_weekly_series(db: Session, drug_codes: list[str]) -> dict[str, np.ndarray]:
    if not drug_codes:
        return {}

    rows = (
        db.query(
            HospitalDailyDemandEnriched.drug_code,
            HospitalDailyDemandEnriched.demand_date,
            HospitalDailyDemandEnriched.demand,
        )
        .filter(HospitalDailyDemandEnriched.drug_code.in_(drug_codes))
        .order_by(
            HospitalDailyDemandEnriched.drug_code.asc(),
            HospitalDailyDemandEnriched.demand_date.asc(),
        )
        .all()
    )
    if not rows:
        return {}

    frame = pd.DataFrame(rows, columns=["drug_code", "demand_date", "demand"])
    frame["demand_date"] = pd.to_datetime(frame["demand_date"])
    frame["week"] = frame["demand_date"].dt.to_period("W")
    weekly = (
        frame.groupby(["drug_code", "week"], as_index=False)["demand"]
        .sum()
        .sort_values(["drug_code", "week"])
    )

    series_by_code: dict[str, np.ndarray] = {}
    for code, group in weekly.groupby("drug_code"):
        series_by_code[str(code)] = group["demand"].to_numpy(dtype=float)
    return series_by_code


def load_weekly_series_for_drug(db: Session, drug_code: str) -> np.ndarray:
    return load_weekly_series(db, [drug_code]).get(drug_code, np.array([], dtype=float))


def count_observation_days(db: Session, drug_code: str) -> int:
    count = (
        db.query(HospitalDailyDemandEnriched.demand_date)
        .filter(
            HospitalDailyDemandEnriched.drug_code == drug_code,
            HospitalDailyDemandEnriched.demand > 0,
        )
        .distinct()
        .count()
    )
    return int(count)


def load_daily_demand_series(
    db: Session,
    drug_code: str,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[tuple[date, float]]:
    query = db.query(
        HospitalDailyDemandEnriched.demand_date,
        HospitalDailyDemandEnriched.demand,
    ).filter(HospitalDailyDemandEnriched.drug_code == drug_code)
    if start_date is not None:
        query = query.filter(HospitalDailyDemandEnriched.demand_date >= start_date)
    if end_date is not None:
        query = query.filter(HospitalDailyDemandEnriched.demand_date <= end_date)
    rows = query.order_by(HospitalDailyDemandEnriched.demand_date.asc()).all()
    return [(row.demand_date, float(row.demand)) for row in rows]
