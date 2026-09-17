"""Pharmacy movement taxonomy for inpatient demand aggregation.

Hospital ledger exports encode stock movements with ``MOV#`` / ``movement_number``.
Only patient-facing movements should feed the forecasting target:

* **5** — sale to inpatient (consumption)
* **6** — return from patient (reduces net demand)
* **8** — cancel patient sale (reduces net demand)

Transfers, inventory adjustments, and other ledger events are kept in
``drug_receipts`` for audit but excluded from daily demand.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class MovementRole(str, Enum):
    PATIENT_SALE = "patient_sale"
    PATIENT_RETURN = "patient_return"
    CANCEL_SALE = "cancel_sale"
    TRANSFER = "transfer"
    OTHER = "other"
    UNKNOWN = "unknown"


# Integer MOV# values observed in hospital 2023/2024 pharmacy exports.
_PATIENT_SALE_MOVEMENTS = frozenset({5})
_PATIENT_RETURN_MOVEMENTS = frozenset({6})
_CANCEL_SALE_MOVEMENTS = frozenset({8})
_TRANSFER_MOVEMENTS = frozenset({2, 7})  # transfer / cancel transfer

PATIENT_SALE_MOVEMENTS = _PATIENT_SALE_MOVEMENTS
PATIENT_RETURN_OR_CANCEL_MOVEMENTS = _PATIENT_RETURN_MOVEMENTS | _CANCEL_SALE_MOVEMENTS
PATIENT_DEMAND_MOVEMENTS = PATIENT_SALE_MOVEMENTS | PATIENT_RETURN_OR_CANCEL_MOVEMENTS


def normalize_movement_number(raw: Any) -> Optional[int]:
    """Parse ``movement_number`` from Excel/DB forms like ``5``, ``5.0``, ``\"5.0\"``."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if raw != raw:  # NaN
            return None
        return int(raw)
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def classify_movement(raw: Any) -> MovementRole:
    """Map a raw movement number to a demand role."""
    mov = normalize_movement_number(raw)
    if mov is None:
        return MovementRole.UNKNOWN
    if mov in _PATIENT_SALE_MOVEMENTS:
        return MovementRole.PATIENT_SALE
    if mov in _PATIENT_RETURN_MOVEMENTS:
        return MovementRole.PATIENT_RETURN
    if mov in _CANCEL_SALE_MOVEMENTS:
        return MovementRole.CANCEL_SALE
    if mov in _TRANSFER_MOVEMENTS:
        return MovementRole.TRANSFER
    return MovementRole.OTHER


def is_patient_demand_movement(raw: Any) -> bool:
    return classify_movement(raw) in {
        MovementRole.PATIENT_SALE,
        MovementRole.PATIENT_RETURN,
        MovementRole.CANCEL_SALE,
    }


def patient_demand_contribution(movement_number: Any, quantity: Any) -> float:
    """Convert one receipt line into a signed inpatient demand contribution.

    Uses movement **role** with absolute quantities so both Excel-signed stock
    ledgers (sales negative) and abs-normalized DB rows work the same way:

    * sale → ``+abs(qty)``
    * return / cancel sale → ``-abs(qty)``
    * transfer / other → ``0``
    * unknown / missing MOV → ``+quantity`` (legacy seed rows without MOV#)
    """
    if quantity is None:
        return 0.0
    try:
        qty = float(quantity)
    except (TypeError, ValueError):
        return 0.0
    if qty != qty:  # NaN
        return 0.0

    role = classify_movement(movement_number)
    if role is MovementRole.PATIENT_SALE:
        return abs(qty)
    if role in {MovementRole.PATIENT_RETURN, MovementRole.CANCEL_SALE}:
        return -abs(qty)
    if role is MovementRole.UNKNOWN:
        # Legacy / test receipts without movement_number: keep signed qty.
        return qty
    return 0.0


def clip_daily_demand(value: float) -> float:
    """Net patient demand cannot be negative for forecasting."""
    if value != value:  # NaN
        return 0.0
    return float(max(0.0, value))
