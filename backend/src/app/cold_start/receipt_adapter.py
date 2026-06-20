"""Bridge between ``drug_receipts`` rows and Cold Start Pydantic schemas."""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.cold_start.catalog_adapter import drug_catalog_to_metadata
from app.cold_start.schemas import DrugMetadataInput
from app.models.drug_catalog import DrugCatalog
from app.models.drug_receipt import DrugReceipt


def infer_metadata_from_receipt(
    drug_code: str,
    drug_name: str | None,
    unit_price: float | None,
) -> DrugMetadataInput:
    """
    Heuristic metadata for drugs sourced from ``drug_receipts``.

    Receipt lines lack full catalog attributes, so drug names and unit prices
    are mapped to plausible embedding inputs.
    """
    name = (drug_name or "").upper()
    form = "tablet"
    route = "oral"
    therapeutic = "other"
    atc = "A01"
    refrigeration = False
    controlled = False

    if any(x in name for x in (" IV", " VI", " INJ", "INFUS", "VIAL")):
        form, route = "injection", "iv"
    elif "TAB" in name or "COMP" in name or "TABLE" in name:
        form, route = "tablet", "oral"
    elif "SYRUP" in name or "SUSP" in name:
        form, route = "syrup", "oral"
    elif "CREAM" in name or "OINT" in name:
        form, route = "cream", "topical"
    elif "INHAL" in name:
        form, route = "inhaler", "inhalation"

    if any(x in name for x in ("VANCO", "CEF", "TRIAX", "AMOX", "CIPRO", "LEVOFLOX", "LINEZOL", "AROPEM")):
        therapeutic, atc = "antibiotic", "J01"
    elif any(x in name for x in ("LASIX", "FUROSEM")):
        therapeutic, atc = "diuretic", "C03"
    elif any(x in name for x in ("WARFAR", "HEPAR")):
        therapeutic, atc = "anticoagulant", "B01"
    elif any(x in name for x in ("METFORM", "INSUL")):
        therapeutic, atc = "antidiabetic", "A10"
    elif any(x in name for x in ("SALBUT", "VENTOL")):
        therapeutic, atc = "bronchodilator", "R03"

    price = float(unit_price or 0)
    if price >= 500_000:
        tier = 5
    elif price >= 100_000:
        tier = 4
    elif price >= 20_000:
        tier = 3
    elif price >= 5_000:
        tier = 2
    else:
        tier = 1

    if "VANCO" in name or "INSUL" in name:
        refrigeration = True

    return DrugMetadataInput(
        drug_code=str(drug_code),
        drug_name=str(drug_name or drug_code)[:512],
        therapeutic_class=therapeutic,
        atc_category=atc,
        pharmaceutical_form=form,
        ven_class="V",
        abc_class="A",
        unit_price_tier=tier,
        requires_refrigeration=refrigeration,
        is_controlled_substance=controlled,
        average_shelf_life_days=365,
        route_of_administration=route,
    )


def _receipt_summary_rows(db_session: Session) -> list[tuple[str, str | None, float | None]]:
    return (
        db_session.query(
            DrugReceipt.drug_code,
            func.max(DrugReceipt.drug_name).label("drug_name"),
            func.max(DrugReceipt.unit_price).label("unit_price"),
        )
        .group_by(DrugReceipt.drug_code)
        .order_by(DrugReceipt.drug_code.asc())
        .all()
    )


def load_receipt_drug_metadata(db_session: Session) -> list[DrugMetadataInput]:
    """Build Cold Start metadata for every distinct drug in ``drug_receipts``."""
    catalog_by_code = {
        row.drug_code: row
        for row in db_session.query(DrugCatalog).all()
    }

    metadata_list: list[DrugMetadataInput] = []
    for drug_code, drug_name, unit_price in _receipt_summary_rows(db_session):
        catalog_row = catalog_by_code.get(drug_code)
        if catalog_row is not None:
            metadata_list.append(drug_catalog_to_metadata(catalog_row))
        else:
            metadata_list.append(
                infer_metadata_from_receipt(
                    drug_code,
                    drug_name,
                    float(unit_price) if unit_price is not None else None,
                )
            )
    return metadata_list
