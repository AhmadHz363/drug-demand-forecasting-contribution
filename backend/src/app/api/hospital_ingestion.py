import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.receipt_upload import UploadHospitalReceiptsResponse
from app.services.hospital_receipt_ingestion import ingest_hospital_receipt_file

logger = logging.getLogger(__name__)

router = APIRouter(tags=["hospital-ingestion"])


@router.post(
    "/upload-hospital-receipts",
    response_model=UploadHospitalReceiptsResponse,
    summary="Upload hospital pharmacy Excel export",
    response_description="Raw and enriched row counts from the ingestion run.",
)
async def upload_hospital_receipts(
    db: Annotated[Session, Depends(get_db)],
    file: Annotated[UploadFile, File(...)],
) -> UploadHospitalReceiptsResponse:
    """Accept hospital Excel exports, store raw rows, and build the cleaned training panel."""
    fname = file.filename or ""
    lower = fname.lower()
    if not lower.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only `.xlsx`, `.xls`, and `.csv` files are accepted",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )

    logger.info("Processing hospital upload filename=%s size=%s bytes", fname, len(raw))
    outcome = ingest_hospital_receipt_file(raw, db, filename=fname)
    return UploadHospitalReceiptsResponse(
        raw_inserted_rows=outcome.raw_inserted_rows,
        raw_failed_rows=outcome.raw_failed_rows,
        enriched_inserted_rows=outcome.enriched_inserted_rows,
        filtered_out_rows=outcome.filtered_out_rows,
        failed_rows=outcome.raw_failed_rows,
        inserted_rows=outcome.raw_inserted_rows,
        errors=outcome.errors,
    )
