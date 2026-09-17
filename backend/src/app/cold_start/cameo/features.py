"""CAMEO cold-start — attribute feature engineering (notebook Section 6)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.cold_start.schemas import DrugMetadataInput


def parse_leading_number(value: str | None) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else float("nan")


def count_items(text: str | None) -> int:
    if not text:
        return 0
    parts = re.split(r"[;,]", str(text))
    return len([part for part in parts if part.strip()])


def onehot_topk(series: pd.Series, k: int, prefix: str) -> pd.DataFrame:
    top = series.value_counts().head(k).index
    normalized = series.where(series.isin(top), other="OTHER").fillna("MISSING")
    return pd.get_dummies(normalized, prefix=prefix)


@dataclass
class CameoFeatureEncoder:
    """Fit on historical library attributes; transform new drugs with fixed columns."""

    column_names: list[str] = field(default_factory=list)
    strength_median: float = 0.0

    def _attrs_frame(self, rows: list[DrugMetadataInput]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Drug Class": [row.drug_class or "MISSING" for row in rows],
                "Dosage Form": [row.dosage_form or "MISSING" for row in rows],
                "Route of Administration": [
                    row.route_of_administration or "MISSING" for row in rows
                ],
                "Pregnancy Category": [row.pregnancy_category or "MISSING" for row in rows],
                "Availability": [row.availability or "MISSING" for row in rows],
                "Strength": [row.strength for row in rows],
                "Indications": [row.indications or "" for row in rows],
                "Side Effects": [row.side_effects or "" for row in rows],
                "Contraindications": [row.contraindications or "" for row in rows],
            }
        )

    def fit(self, rows: list[DrugMetadataInput]) -> "CameoFeatureEncoder":
        frame = self._attrs_frame(rows)
        numeric_strength = frame["Strength"].apply(parse_leading_number)
        self.strength_median = float(numeric_strength.median(skipna=True) or 0.0)

        parts = [
            onehot_topk(frame["Drug Class"], k=15, prefix="class"),
            onehot_topk(frame["Dosage Form"], k=10, prefix="form"),
            onehot_topk(frame["Route of Administration"], k=8, prefix="route"),
            onehot_topk(frame["Pregnancy Category"], k=6, prefix="preg"),
            onehot_topk(frame["Availability"], k=4, prefix="avail"),
            pd.DataFrame(
                {
                    "strength": numeric_strength.fillna(self.strength_median),
                    "n_indications": frame["Indications"].apply(count_items),
                    "n_side_effects": frame["Side Effects"].apply(count_items),
                    "n_contraindications": frame["Contraindications"].apply(count_items),
                }
            ),
        ]
        feature_table = pd.concat(parts, axis=1).fillna(0.0)
        self.column_names = list(feature_table.columns)
        return self

    def transform(self, rows: list[DrugMetadataInput]) -> np.ndarray:
        if not self.column_names:
            raise ValueError("CameoFeatureEncoder is not fit yet.")

        frame = self._attrs_frame(rows)
        numeric_strength = frame["Strength"].apply(parse_leading_number)
        parts = [
            onehot_topk(frame["Drug Class"], k=15, prefix="class"),
            onehot_topk(frame["Dosage Form"], k=10, prefix="form"),
            onehot_topk(frame["Route of Administration"], k=8, prefix="route"),
            onehot_topk(frame["Pregnancy Category"], k=6, prefix="preg"),
            onehot_topk(frame["Availability"], k=4, prefix="avail"),
            pd.DataFrame(
                {
                    "strength": numeric_strength.fillna(self.strength_median),
                    "n_indications": frame["Indications"].apply(count_items),
                    "n_side_effects": frame["Side Effects"].apply(count_items),
                    "n_contraindications": frame["Contraindications"].apply(count_items),
                }
            ),
        ]
        feature_table = pd.concat(parts, axis=1).fillna(0.0)
        aligned = feature_table.reindex(columns=self.column_names, fill_value=0.0)
        return aligned.to_numpy(dtype=float)

    def transform_one(self, row: DrugMetadataInput) -> np.ndarray:
        return self.transform([row])[0]
