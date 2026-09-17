"""Unit tests for CAMEO cold-start feature engineering and pipeline."""

from __future__ import annotations

import numpy as np

from app.cold_start.cameo.features import CameoFeatureEncoder, count_items, parse_leading_number
from app.cold_start.cameo.metric_net import MetricNet, demand_shape_features, train_metric_net
from app.cold_start.cameo.pipeline import run_cameo
from app.cold_start.schemas import DrugMetadataInput


def _sample_drug(code: str = "P100390") -> DrugMetadataInput:
    return DrugMetadataInput(
        drug_code=code,
        drug_name="DIOVON 160MG TAB",
        generic_name="Valsartan",
        drug_class="ARB",
        dosage_form="Tablet",
        strength="80 mg",
        route_of_administration="Oral",
        pregnancy_category="Contraindicated (not allowed)",
        availability="Prescription",
        indications="Hypertension",
        side_effects="heart failure",
        contraindications="post-MI",
    )


def test_parse_leading_number():
    assert parse_leading_number("160MG TAB") == 160.0
    assert np.isnan(parse_leading_number(None))


def test_count_items():
    assert count_items("a; b, c") == 3
    assert count_items("") == 0


def test_feature_encoder_fit_transform_stable_columns():
    rows = [_sample_drug("A"), _sample_drug("B")]
    rows[1].drug_class = "Beta blocker"
    encoder = CameoFeatureEncoder().fit(rows)
    matrix = encoder.transform(rows)
    assert matrix.shape[0] == 2
    assert matrix.shape[1] == len(encoder.column_names)
    assert encoder.transform([rows[0]]).shape == (1, matrix.shape[1])


def test_metric_net_train_and_cameo_pipeline_runs():
    rng = np.random.default_rng(0)
    attrs = rng.normal(size=(12, 20))
    shapes = rng.normal(size=(12, 5))
    net = train_metric_net(attrs, shapes, epochs=30, seed=0)
    embeds = net.embed(attrs)
    hist_series = [rng.integers(0, 20, size=40).astype(float) for _ in range(12)]
    actual = rng.integers(0, 15, size=8).astype(float)
    forecasts, drift = run_cameo(attrs[0], embeds, hist_series, net, actual, topk=3)
    assert forecasts.shape == actual.shape
    assert len(drift) == len(actual)
