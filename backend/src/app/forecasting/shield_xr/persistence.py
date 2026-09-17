"""Persist cleaned training panel rows to ``forecast_training_data``."""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy.orm import Session

from app.models.forecast_training_data import ForecastTrainingData

logger = logging.getLogger(__name__)


def persist_training_panel(
    db_session: Session,
    training_run_id: str,
    panel: pd.DataFrame,
    *,
    replace_previous: bool = True,
) -> int:
    """Bulk-insert cleaned panel rows used for SHIELD-XR training."""
    if replace_previous:
        deleted = (
            db_session.query(ForecastTrainingData)
            .filter(ForecastTrainingData.training_run_id != training_run_id)
            .delete(synchronize_session=False)
        )
        if deleted:
            logger.info("Removed %d rows from prior forecast_training_data runs", deleted)

    records = panel.to_dict(orient="records")
    rows: list[ForecastTrainingData] = []
    for record in records:
        rows.append(
            ForecastTrainingData(
                training_run_id=training_run_id,
                drug_code=str(record["CODE"]),
                demand_date=pd.Timestamp(record["DATE"]).date(),
                demand_raw=float(record["demand_raw"]),
                demand_clean=float(record["demand_clean"]),
                drug_name=record.get("ARTICLE"),
                drug_category=str(record["CAT"]) if record.get("CAT") is not None else None,
                sb_class=str(record["sb_class"]),
                is_anomaly_day=bool(record.get("is_anomaly_day", False)),
                is_dropout_day=bool(record.get("is_dropout_day", False)),
                ceiling=float(record["ceiling"]) if record.get("ceiling") is not None else None,
            )
        )

    db_session.bulk_save_objects(rows)
    db_session.flush()
    logger.info(
        "Persisted %d forecast_training_data rows for run %s",
        len(rows),
        training_run_id,
    )
    return len(rows)
