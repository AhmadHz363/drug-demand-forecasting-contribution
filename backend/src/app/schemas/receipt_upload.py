from pydantic import BaseModel, Field


class ReceiptRowError(BaseModel):
    row_index: int = Field(..., description="1-based Excel data row (excluding header)")
    message: str


class UploadReceiptsResponse(BaseModel):
    inserted_rows: int
    failed_rows: int
    errors: list[ReceiptRowError]
