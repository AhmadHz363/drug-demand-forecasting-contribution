"""Excel column labels → snake_case DB field names."""

EXCEL_TO_DB_COLUMNS: dict[str, str] = {
    "Doc": "receipt_id",
    "Line": "line_count",
    "Cat": "drug_category",
    "C.R": "center_receipt",
    "Date": "receipt_date",
    "Mov#": "movement_number",
    "Mov": "movement_type",
    "Code": "drug_code",
    "Article": "drug_name",
    "M": "month",
    "C.S": "center_syn_id",
    "QTY": "quantity",
    "U.p": "unit_price",
    "T.P": "total_price",
    "Ad": "admission_date",
    "R": "room_number",
    "U": "bed_number",
    "Dr": "doctor_name",
}

# After header canonicalization, these labels must exist for a valid upload.
# `Doc` (receipt id) is optional — missing / blank values get a per-row incremental id during import.
# Other Excel columns are optional (e.g. Ad / R / U / Dr are often absent in exports).
REQUIRED_CANONICAL_EXCEL_COLUMNS: frozenset[str] = frozenset({"Code", "Date"})

# Integers and amounts may be negative in source exports (returns, reversals, adjustments).
INTEGER_FIELDS = frozenset({"line_count", "month"})
FLOAT_FIELDS = frozenset({"quantity", "unit_price", "total_price"})
DATE_FIELDS = frozenset({"receipt_date", "admission_date"})
STRING_FIELDS = frozenset(
    {
        "receipt_id",
        "drug_category",
        "center_receipt",
        "movement_number",
        "movement_type",
        "drug_code",
        "drug_name",
        "center_syn_id",
        "room_number",
        "bed_number",
        "doctor_name",
    }
)
