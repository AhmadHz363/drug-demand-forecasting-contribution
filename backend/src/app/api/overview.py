"""System overview — aggregated KPIs across all modules."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.overview import OverviewResponse
from app.services.overview import get_overview

router = APIRouter(prefix="/overview", tags=["overview"])


@router.get(
    "",
    response_model=OverviewResponse,
    summary="System overview KPIs",
    response_description=(
        "Aggregated metrics from all modules: ingestion, drug registry, "
        "category registry, cold start, and forecasting."
    ),
)
def overview(db: Annotated[Session, Depends(get_db)]) -> OverviewResponse:
    return get_overview(db)
