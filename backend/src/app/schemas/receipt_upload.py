from pydantic import BaseModel, Field


class ReceiptRowError(BaseModel):
    row_index: int = Field(..., description="1-based Excel data row (excluding header)")
    message: str


class UploadHospitalReceiptsResponse(BaseModel):
    raw_inserted_rows: int
    raw_failed_rows: int
    enriched_inserted_rows: int
    filtered_out_rows: int
    failed_rows: int = Field(..., description="Alias of raw_failed_rows for legacy clients")
    inserted_rows: int = Field(..., description="Alias of raw_inserted_rows for legacy clients")
    errors: list[ReceiptRowError]
