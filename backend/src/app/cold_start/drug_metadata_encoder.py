"""Deterministic encoding of drug metadata into a fixed-length feature vector."""

from __future__ import annotations

from app.cold_start.constants import (
    METADATA_INPUT_DIM,
    SHELF_LIFE_MAX_DAYS,
    SHELF_LIFE_MIN_DAYS,
    UNIT_PRICE_TIER_MAX,
    UNIT_PRICE_TIER_MIN,
)
from app.cold_start.schemas import DrugMetadataInput

# Label maps — extend via "other" fallback for unseen values (scales to large catalogs).
THERAPEUTIC_CLASS_MAP: dict[str, int] = {
    "antibiotic": 0,
    "analgesic": 1,
    "antifungal": 2,
    "antiviral": 3,
    "anticoagulant": 4,
    "antihypertensive": 5,
    "antidiabetic": 6,
    "antiemetic": 7,
    "bronchodilator": 8,
    "corticosteroid": 9,
    "diuretic": 10,
    "immunosuppressant": 11,
    "other": 12,
}
TOTAL_THERAPEUTIC_CLASSES = 13

PHARMA_FORM_MAP: dict[str, int] = {
    "tablet": 0,
    "capsule": 1,
    "injection": 2,
    "syrup": 3,
    "suspension": 4,
    "cream": 5,
    "ointment": 6,
    "patch": 7,
    "inhaler": 8,
    "suppository": 9,
    "drops": 10,
    "other": 11,
}
TOTAL_PHARMA_FORMS = 12

ROUTE_MAP: dict[str, int] = {
    "oral": 0,
    "iv": 1,
    "im": 2,
    "sc": 3,
    "topical": 4,
    "inhalation": 5,
    "rectal": 6,
    "ophthalmic": 7,
    "other": 8,
}
TOTAL_ROUTES = 9

VEN_CLASS_VALUES: dict[str, float] = {"V": 1.0, "E": 0.5, "N": 0.0}
ABC_CLASS_VALUES: dict[str, float] = {"A": 1.0, "B": 0.5, "C": 0.0}
VEN_WEIGHT: dict[str, float] = {"V": 3.0, "E": 2.0, "N": 1.0}
ABC_WEIGHT: dict[str, float] = {"A": 3.0, "B": 2.0, "C": 1.0}

_FALLBACK_KEY = "other"


def _normalize_key(value: str) -> str:
    return value.strip().lower()


def _label_index(value: str, mapping: dict[str, int], fallback: str = _FALLBACK_KEY) -> int:
    key = _normalize_key(value)
    if key in mapping:
        return mapping[key]
    return mapping[fallback]


def _normalize_label_index(index: int, total_classes: int) -> float:
    if total_classes <= 1:
        return 0.0
    return index / (total_classes - 1)


def _encode_atc_category(atc_category: str) -> float:
    code = atc_category.strip().upper()
    if not code:
        return 0.0
    ordinal = ord(code[0]) - ord("A")
    return max(0.0, min(ordinal / 25.0, 1.0))


def _encode_shelf_life(days: int) -> float:
    clipped = max(SHELF_LIFE_MIN_DAYS, min(days, SHELF_LIFE_MAX_DAYS))
    span = SHELF_LIFE_MAX_DAYS - SHELF_LIFE_MIN_DAYS
    return (clipped - SHELF_LIFE_MIN_DAYS) / span


def _encode_price_tier(tier: int) -> float:
    clamped = max(UNIT_PRICE_TIER_MIN, min(tier, UNIT_PRICE_TIER_MAX))
    return (clamped - UNIT_PRICE_TIER_MIN) / (UNIT_PRICE_TIER_MAX - UNIT_PRICE_TIER_MIN)


def _lookup_class_value(value: str, mapping: dict[str, float], default: float = 0.0) -> float:
    key = _normalize_key(value)
    if len(key) == 1 and key.upper() in mapping:
        return mapping[key.upper()]
    return default


def _lookup_weight(value: str, mapping: dict[str, float], default: float = 1.0) -> float:
    key = _normalize_key(value)
    if len(key) == 1 and key.upper() in mapping:
        return mapping[key.upper()] / 3.0
    return default / 3.0


def encode_drug_metadata(drug: DrugMetadataInput) -> list[float]:
    """Return exactly METADATA_INPUT_DIM floats in [0, 1] for autoencoder input."""
    features: list[float] = [
        _normalize_label_index(
            _label_index(drug.therapeutic_class, THERAPEUTIC_CLASS_MAP),
            TOTAL_THERAPEUTIC_CLASSES,
        ),
        _encode_atc_category(drug.atc_category),
        _normalize_label_index(
            _label_index(drug.pharmaceutical_form, PHARMA_FORM_MAP),
            TOTAL_PHARMA_FORMS,
        ),
        _lookup_class_value(drug.ven_class, VEN_CLASS_VALUES),
        _lookup_class_value(drug.abc_class, ABC_CLASS_VALUES),
        _encode_price_tier(drug.unit_price_tier),
        1.0 if drug.requires_refrigeration else 0.0,
        1.0 if drug.is_controlled_substance else 0.0,
        _encode_shelf_life(drug.average_shelf_life_days),
        _normalize_label_index(
            _label_index(drug.route_of_administration, ROUTE_MAP),
            TOTAL_ROUTES,
        ),
        _lookup_weight(drug.abc_class, ABC_WEIGHT),
        _lookup_weight(drug.ven_class, VEN_WEIGHT),
    ]

    if len(features) != METADATA_INPUT_DIM:
        raise ValueError(
            f"Encoder produced {len(features)} features; expected {METADATA_INPUT_DIM}"
        )

    return features
