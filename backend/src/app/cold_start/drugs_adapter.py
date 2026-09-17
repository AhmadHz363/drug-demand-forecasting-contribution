"""Bridge between ``drugs`` ORM rows and CAMEO cold-start schemas."""

from __future__ import annotations

import re

import numpy as np
from sqlalchemy.orm import Session

from app.cold_start.schemas import DrugMetadataInput
from app.models.cameo_drug import CameoDrug


def parse_leading_number(value: str | None) -> float:
    if not value:
        return float("nan")
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else float("nan")


def drug_row_to_metadata(row: CameoDrug) -> DrugMetadataInput:
    return DrugMetadataInput(
        drug_code=row.drug_code,
        drug_name=row.input_drug_name or row.generic_name or row.drug_code,
        generic_name=row.generic_name,
        drug_class=row.drug_class or "MISSING",
        dosage_form=row.dosage_form or "MISSING",
        strength=row.strength or "",
        route_of_administration=row.route_of_administration or "MISSING",
        pregnancy_category=row.pregnancy_category or "MISSING",
        availability=row.availability or "MISSING",
        indications=row.indications or "",
        side_effects=row.side_effects or "",
        contraindications=row.contraindications or "",
    )


def resolve_matched_source_attributes(db: Session) -> list[DrugMetadataInput]:
    """
    Notebook Section 4 — only ``matched_source`` rows with strength-aware
    generic resolution when multiple attribute rows share a generic name.
    """
    matched_rows = (
        db.query(CameoDrug)
        .filter(CameoDrug.match_status == "matched_source")
        .order_by(CameoDrug.drug_code.asc())
        .all()
    )
    if not matched_rows:
        return []

    by_code: dict[str, DrugMetadataInput] = {}
    article_strength = {
        row.drug_code: parse_leading_number(row.input_drug_name or row.strength)
        for row in matched_rows
    }

    generic_groups: dict[str, list[CameoDrug]] = {}
    for row in matched_rows:
        generic = row.matched_source_generic or row.generic_name or row.resolved_generic
        if not generic:
            by_code[row.drug_code] = drug_row_to_metadata(row)
            continue
        generic_groups.setdefault(generic, []).append(row)

    for generic, rows in generic_groups.items():
        if len(rows) == 1:
            by_code[rows[0].drug_code] = drug_row_to_metadata(rows[0])
            continue

        best_by_code: dict[str, tuple[float, CameoDrug]] = {}
        for row in rows:
            di_strength = parse_leading_number(row.strength)
            article = article_strength.get(row.drug_code, float("nan"))
            diff = abs(article - di_strength) if np.isfinite(article) and np.isfinite(di_strength) else 1e9
            current = best_by_code.get(row.drug_code)
            if current is None or diff < current[0]:
                best_by_code[row.drug_code] = (diff, row)

        for code, (_, row) in best_by_code.items():
            by_code[code] = drug_row_to_metadata(row)

    return [by_code[row.drug_code] for row in matched_rows if row.drug_code in by_code]


def load_library_drug_metadata(db: Session) -> list[DrugMetadataInput]:
    """Historical library — matched_source drugs only (0% synthetic placeholders)."""
    return resolve_matched_source_attributes(db)


def load_all_drug_metadata(db: Session) -> list[DrugMetadataInput]:
    rows = db.query(CameoDrug).order_by(CameoDrug.drug_code.asc()).all()
    return [drug_row_to_metadata(row) for row in rows]


def get_drug_metadata(db: Session, drug_code: str) -> DrugMetadataInput | None:
    row = db.query(CameoDrug).filter(CameoDrug.drug_code == drug_code).one_or_none()
    if row is None:
        return None
    return drug_row_to_metadata(row)
