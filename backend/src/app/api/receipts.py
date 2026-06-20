import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.receipt_upload import UploadReceiptsResponse
from app.services.receipt_ingestion import ingest_receipt_file

logger = logging.getLogger(__name__)

router = APIRouter(tags=["receipts"])


@router.post(
    "/upload-receipts",
    response_model=UploadReceiptsResponse,
    summary="Upload receipt spreadsheet",
    response_description="Row counts and per-row validation errors from the ingestion run.",
)
async def upload_receipts(
    db: Annotated[Session, Depends(get_db)],
    file: Annotated[UploadFile, File(...)],
) -> UploadReceiptsResponse:
    """Accept `.xlsx`, `.xls`, or `.csv` pharmacy receipts and bulk-load into PostgreSQL."""
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

    logger.info("Processing upload filename=%s size=%s bytes", fname, len(raw))

    outcome = ingest_receipt_file(raw, db, filename=fname)
    return UploadReceiptsResponse(
        inserted_rows=outcome.inserted_rows,
        failed_rows=outcome.failed_rows,
        errors=outcome.errors,
    )
