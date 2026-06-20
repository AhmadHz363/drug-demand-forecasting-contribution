"""Bridge between DrugCatalog ORM rows and Cold Start Pydantic schemas."""

from __future__ import annotations

from app.cold_start.schemas import DrugMetadataInput
from app.models.drug_catalog import DrugCatalog


def drug_catalog_to_metadata(row: DrugCatalog) -> DrugMetadataInput:
    """Convert a persisted catalog row to embedding input (batch-friendly for large catalogs)."""
    return DrugMetadataInput(
        drug_code=row.drug_code,
        drug_name=row.drug_name,
        therapeutic_class=row.therapeutic_class,
        atc_category=row.atc_category,
        pharmaceutical_form=row.pharmaceutical_form,
        ven_class=row.ven_class,
        abc_class=row.abc_class,
        unit_price_tier=row.unit_price_tier,
        requires_refrigeration=row.requires_refrigeration,
        is_controlled_substance=row.is_controlled_substance,
        average_shelf_life_days=row.average_shelf_life_days,
        route_of_administration=row.route_of_administration,
    )
