"""Build the cleaned daily demand panel from raw hospital receipt rows."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.services.movement_demand import clip_daily_demand

PATIENT_SALE_MOVEMENT_DESCRIPTION = "مـبـيع الـى مـرضـى داخلـي"

_ENRICHED_COLUMNS = [
    "demand_date",
    "drug_code",
    "article",
    "cat",
    "demand",
    "n_patients_drug",
    "n_doctors_drug",
    "top_cr_drug",
    "top_cs_drug",
    "top_dr_drug",
    "n_unique_patients",
    "n_admissions",
    "n_unique_doctors",
    "n_unique_cr",
    "n_unique_cs",
    "n_transactions",
    "n_demand_txns",
    "hospital_total_demand",
    "age_mean",
    "age_median",
    "top1_cr",
    "top1_cr_share",
    "top2_cr",
    "top2_cr_share",
    "top3_cr",
    "top3_cr_share",
]


def normalize_movement_description(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def parse_hospital_date(value: Any) -> Optional[date]:
    """Parse hospital export dates (DD/MM/YY, Excel datetimes, placeholders)."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.date()

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text in {"00/00/00", "0000/00/00"}:
        return None

    compact = re.sub(r"\s+", "", text)
    parts = compact.split("/")
    if len(parts) == 3:
        day_part, month_part, year_part = parts
        try:
            day_val = int(day_part)
            month_val = int(month_part)
            year_val = int(year_part)
            if year_val < 100:
                year_val += 2000
            return date(year_val, month_val, day_val)
        except (TypeError, ValueError):
            pass

    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if not pd.isna(parsed):
            return parsed.date()
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    return parsed.date()


def _clean_doctor_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})


def _mode_value(series: pd.Series) -> Any:
    values = series.dropna()
    if values.empty:
        return None
    return values.mode().iloc[0]


def _count_unique_doctors(series: pd.Series) -> int:
    cleaned = _clean_doctor_series(series).dropna()
    return int(cleaned.nunique())


def _patient_age_years(receipt_date: date, birth_date: Optional[date]) -> Optional[float]:
    if birth_date is None:
        return None
    years = receipt_date.year - birth_date.year
    if (receipt_date.month, receipt_date.day) < (birth_date.month, birth_date.day):
        years -= 1
    if years < 0 or years > 120:
        return None
    return float(years)


def _build_hospital_day_drivers(day_frame: pd.DataFrame) -> dict[str, Any]:
    total_rows = len(day_frame)
    cr_series = pd.to_numeric(day_frame["C.R"], errors="coerce").dropna().astype(int)
    cr_counts = cr_series.value_counts()
    total_cr = float(cr_counts.sum()) if not cr_counts.empty else 0.0

    top_cr_values: list[Optional[int]] = []
    top_cr_shares: list[float] = []
    for rank in range(3):
        if rank < len(cr_counts):
            cr_val = cr_counts.index[rank]
            try:
                top_cr_values.append(int(cr_val))
            except (TypeError, ValueError):
                top_cr_values.append(None)
            top_cr_shares.append(float(cr_counts.iloc[rank] / total_cr) if total_cr > 0 else 0.0)
        else:
            top_cr_values.append(None)
            top_cr_shares.append(0.0)

    ages = day_frame["patient_age"].dropna()
    age_mean = float(ages.mean()) if not ages.empty else None
    age_median = float(ages.median()) if not ages.empty else None

    return {
        "n_unique_patients": int(day_frame["MRN"].dropna().nunique()),
        "n_admissions": int(day_frame.loc[day_frame["admission_date"] == day_frame["DATE"], "MRN"].dropna().nunique()),
        "n_unique_doctors": _count_unique_doctors(day_frame["DR"]),
        "n_unique_cr": int(day_frame["C.R"].dropna().nunique()),
        "n_unique_cs": int(day_frame["C.S"].dropna().nunique()),
        "n_transactions": total_rows,
        "n_demand_txns": total_rows,
        "age_mean": age_mean,
        "age_median": age_median,
        "top1_cr": top_cr_values[0],
        "top1_cr_share": top_cr_shares[0],
        "top2_cr": top_cr_values[1],
        "top2_cr_share": top_cr_shares[1],
        "top3_cr": top_cr_values[2],
        "top3_cr_share": top_cr_shares[2],
    }


def build_enriched_daily_panel(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Filter to inpatient sales and build the dense SKU × day enriched panel."""
    if raw_df.empty:
        return pd.DataFrame(columns=_ENRICHED_COLUMNS), 0

    work = raw_df.copy()
    work.columns = [str(col).strip() for col in work.columns]
    skipped_rows = int(len(work))

    work["mov_des_norm"] = work["Mov des"].map(normalize_movement_description)
    work = work[work["mov_des_norm"] == PATIENT_SALE_MOVEMENT_DESCRIPTION].copy()
    skipped_rows -= len(work)
    if work.empty:
        return pd.DataFrame(columns=_ENRICHED_COLUMNS), skipped_rows

    work["DATE"] = work["DATE"].map(parse_hospital_date)
    work["admission_date"] = work["AD DATE"].map(parse_hospital_date)
    work["birth_date"] = work["AGE"].map(parse_hospital_date)
    work = work[work["DATE"].notna()].copy()
    work["CODE"] = work["CODE"].astype(str).str.strip()
    work = work[work["CODE"].astype(bool)].copy()
    work["patient_age"] = [
        _patient_age_years(receipt_date, birth_date)
        for receipt_date, birth_date in zip(work["DATE"], work["birth_date"])
    ]

    drug_meta = (
        work.groupby("CODE", as_index=False)
        .agg(
            article=("ARTICLE", "first"),
            cat=("CAT", _mode_value),
        )
    )

    drug_day = (
        work.groupby(["DATE", "CODE"], as_index=False)
        .agg(
            demand=("QTY", lambda s: int(round(clip_daily_demand(-float(s.sum()))))),
            n_patients_drug=("MRN", lambda s: int(s.dropna().nunique())),
            n_doctors_drug=("DR", _count_unique_doctors),
            top_cr_drug=("C.R", _mode_value),
            top_cs_drug=("C.S", _mode_value),
            top_dr_drug=("DR", lambda s: _mode_value(_clean_doctor_series(s))),
        )
    )

    hospital_days: list[dict[str, Any]] = []
    for day_value, day_frame in work.groupby("DATE", sort=True):
        drivers = _build_hospital_day_drivers(day_frame)
        day_demand = int(
            drug_day.loc[drug_day["DATE"] == day_value, "demand"].sum(),
        )
        drivers["DATE"] = day_value
        drivers["hospital_total_demand"] = day_demand
        hospital_days.append(drivers)
    hospital = pd.DataFrame(hospital_days)

    all_codes = drug_meta["CODE"].tolist()
    all_dates = pd.date_range(work["DATE"].min(), work["DATE"].max(), freq="D")
    skeleton = pd.MultiIndex.from_product([all_codes, all_dates], names=["CODE", "DATE"]).to_frame(index=False)
    skeleton["DATE"] = skeleton["DATE"].dt.date

    panel = skeleton.merge(drug_meta, on="CODE", how="left")
    panel = panel.merge(drug_day, on=["DATE", "CODE"], how="left")
    panel = panel.merge(hospital, on="DATE", how="left")

    int_cols = [
        "demand",
        "n_patients_drug",
        "n_doctors_drug",
        "n_unique_patients",
        "n_admissions",
        "n_unique_doctors",
        "n_unique_cr",
        "n_unique_cs",
        "n_transactions",
        "n_demand_txns",
        "hospital_total_demand",
    ]
    for col in int_cols:
        panel[col] = panel[col].fillna(0).astype(int)

    share_cols = ["top1_cr_share", "top2_cr_share", "top3_cr_share"]
    for col in share_cols:
        panel[col] = panel[col].fillna(0.0).astype(float)

    nullable_int_cols = ["cat", "top1_cr", "top2_cr", "top3_cr"]
    for col in nullable_int_cols:
        numeric = pd.to_numeric(panel[col], errors="coerce")
        panel[col] = numeric.apply(lambda value: int(value) if pd.notna(value) else None)

    for col in ("top_cr_drug", "top_cs_drug", "top_dr_drug", "article"):
        panel[col] = panel[col].where(panel[col].notna(), None)

    panel = panel.rename(
        columns={
            "DATE": "demand_date",
            "CODE": "drug_code",
            "article": "article",
            "cat": "cat",
        },
    )
    return panel[_ENRICHED_COLUMNS], skipped_rows
