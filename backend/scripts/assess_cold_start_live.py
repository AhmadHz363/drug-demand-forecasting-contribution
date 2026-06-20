#!/usr/bin/env python3
"""Live assessment of cold-start forecasting against the real database (no seed reset)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.cold_start.constants import ARTIFACTS_DIR
from app.cold_start.drug_metadata_encoder import encode_drug_metadata
from app.cold_start.schemas import DrugMetadataInput
from app.core.database import SessionLocal, engine
from app.main import app

NEW_DRUG = {
    "drug_metadata": {
        "drug_code": "ASSESS-NEW-001",
        "drug_name": "Ceftriaxone 1g Injection",
        "therapeutic_class": "antibiotic",
        "atc_category": "J01",
        "pharmaceutical_form": "injection",
        "ven_class": "V",
        "abc_class": "A",
        "unit_price_tier": 4,
        "requires_refrigeration": False,
        "is_controlled_substance": False,
        "average_shelf_life_days": 730,
        "route_of_administration": "iv",
    },
    "forecast_horizon_days": 7,
}


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> int:
    issues: list[str] = []
    warnings: list[str] = []
    passes: list[str] = []

    section("1. Database readiness")
    db = SessionLocal()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        passes.append("PostgreSQL connected")

        catalog = db.execute(text("SELECT COUNT(*) FROM drug_catalog")).scalar() or 0
        demand_rows = db.execute(text("SELECT COUNT(*) FROM daily_drug_demand")).scalar() or 0
        demand_drugs = db.execute(
            text("SELECT COUNT(DISTINCT drug_code) FROM daily_drug_demand")
        ).scalar() or 0
        ge7 = db.execute(
            text(
                """
                SELECT COUNT(*) FROM (
                    SELECT drug_code FROM daily_drug_demand GROUP BY drug_code HAVING COUNT(*) >= 7
                ) s
                """
            )
        ).scalar() or 0
        ge90 = db.execute(
            text(
                """
                SELECT COUNT(*) FROM (
                    SELECT drug_code FROM daily_drug_demand GROUP BY drug_code HAVING COUNT(*) >= 90
                ) s
                """
            )
        ).scalar() or 0
        maml_eligible = db.execute(
            text(
                """
                SELECT COUNT(*) FROM (
                    SELECT drug_code FROM daily_drug_demand
                    GROUP BY drug_code HAVING COUNT(*) >= 26
                ) s
                """
            )
        ).scalar() or 0
        catalog_ge7 = db.execute(
            text(
                """
                SELECT COUNT(*) FROM drug_catalog dc
                WHERE (SELECT COUNT(*) FROM daily_drug_demand d WHERE d.drug_code = dc.drug_code) >= 7
                """
            )
        ).scalar() or 0

        print(f"  drug_catalog rows:           {catalog}")
        print(f"  daily_drug_demand rows:      {demand_rows} ({demand_drugs} drugs)")
        print(f"  drugs with >= 7 days:        {ge7} (in catalog: {catalog_ge7})")
        print(f"  drugs with >= 90 days:       {ge90}")
        print(f"  drugs MAML-eligible (>=26d): {maml_eligible}")

        if catalog < 5:
            issues.append("drug_catalog has fewer than 5 drugs")
        else:
            passes.append(f"drug_catalog has {catalog} drugs")

        if catalog_ge7 < 5:
            issues.append(
                f"Only {catalog_ge7} catalog drugs have >=7 days demand — KNN may fail or degrade"
            )
        else:
            passes.append(f"{catalog_ge7} catalog drugs usable for KNN neighbours")

        if ge90 < 5:
            warnings.append(
                f"Only {ge90} drugs have 90+ days history (spec expects 90-day lookback for KNN)"
            )

        if maml_eligible < 5:
            warnings.append(
                f"Only {maml_eligible} drugs MAML-eligible — meta-training quality may be limited"
            )
    finally:
        db.close()

    section("2. Model artifacts")
    for name in ("autoencoder.pt", "maml_base.pt"):
        path = os.path.join(ARTIFACTS_DIR, name)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            passes.append(f"{name} exists ({size} bytes)")
            print(f"  OK  {name} ({size} bytes)")
        else:
            issues.append(f"{name} missing — train via POST /cold-start/train-embedder or train-maml")
            print(f"  MISSING  {name}")

    section("3. Encoder sanity")
    meta = DrugMetadataInput(**NEW_DRUG["drug_metadata"])
    vec = encode_drug_metadata(meta)
    if len(vec) == 12 and all(0.0 <= v <= 1.0 for v in vec):
        passes.append("encode_drug_metadata returns 12 values in [0,1]")
        print(f"  OK  vector length={len(vec)} sample={vec[:4]}...")
    else:
        issues.append(f"Bad encoder output: len={len(vec)}")

    section("4. Live predict (cold_start_only)")
    client = TestClient(app)
    db = SessionLocal()
    try:
        db.execute(
            text("DELETE FROM daily_drug_demand WHERE drug_code = 'ASSESS-NEW-001'")
        )
        db.commit()
    finally:
        db.close()

    resp = client.post("/cold-start/predict", json=NEW_DRUG)
    print(f"  HTTP {resp.status_code}")
    if resp.status_code != 200:
        issues.append(f"predict failed: {resp.status_code} {resp.text[:300]}")
        print(resp.text[:500])
    else:
        body = resp.json()
        print(json.dumps({k: body[k] for k in (
            "drug_code", "stage_used", "observation_count",
            "nearest_neighbours", "similarity_scores",
        )}, indent=2))
        fc = body.get("forecast", [])
        if body.get("stage_used") == "cold_start_only":
            passes.append("predict returns cold_start_only for new drug")
        else:
            issues.append(f"expected cold_start_only, got {body.get('stage_used')}")

        if len(body.get("embedding", [])) == 32:
            passes.append("embedding has 32 dimensions")
        else:
            issues.append("embedding length != 32")

        nn = body.get("nearest_neighbours", [])
        if 0 < len(nn) <= 5:
            passes.append(f"found {len(nn)} KNN neighbours")
        elif len(nn) == 0:
            issues.append("no KNN neighbours returned — bootstrap likely used pharmacist fallback or failed silently")
        else:
            issues.append(f"unexpected neighbour count: {len(nn)}")

        if len(fc) == 7:
            passes.append("7-day forecast returned")
        else:
            issues.append(f"forecast length {len(fc)} != 7")

        intervals_ok = all(
            d["p10"] >= 0 and d["p10"] <= d["p50"] <= d["p90"] for d in fc
        )
        if intervals_ok:
            passes.append("P10 <= P50 <= P90 on all days")
            print(f"  P50 day 1: {fc[0]['p50']:.2f}  (P10={fc[0]['p10']:.2f}, P90={fc[0]['p90']:.2f})")
        else:
            issues.append("invalid prediction intervals")

        p50_baseline = fc[0]["p50"] if fc else None

        section("5. Pharmacist estimate blend")
        body_pharm = {
            **NEW_DRUG,
            "pharmacist_estimate": {"weekly_units": 200, "confidence": 0.8},
        }
        resp2 = client.post("/cold-start/predict", json=body_pharm)
        if resp2.status_code == 200 and p50_baseline is not None:
            p50_pharm = resp2.json()["forecast"][0]["p50"]
            shift = abs(p50_pharm - p50_baseline)
            print(f"  P50 baseline={p50_baseline:.2f}  with pharmacist={p50_pharm:.2f}  delta={shift:.2f}")
            if shift > 0.01:
                passes.append("pharmacist estimate shifts P50")
            else:
                warnings.append("pharmacist estimate did not materially shift P50")
        else:
            issues.append(f"pharmacist predict failed: {resp2.status_code}")

        section("6. Graduation (insert synthetic observations)")
        db = SessionLocal()
        try:
            from datetime import date, timedelta

            base = date.today() - timedelta(days=20)
            for i in range(5):
                db.execute(
                    text(
                        """
                        INSERT INTO daily_drug_demand (drug_code, demand_date, total_quantity)
                        VALUES (:code, :d, :q)
                        ON CONFLICT DO NOTHING
                        """
                    ),
                    {"code": "ASSESS-NEW-001", "d": base + timedelta(days=i), "q": 10.0 + i},
                )
            db.commit()
        finally:
            db.close()

        resp3 = client.post("/cold-start/predict", json=NEW_DRUG)
        if resp3.status_code == 200:
            stage5 = resp3.json()["stage_used"]
            print(f"  5 observations → stage_used={stage5}")
            if stage5 == "blended":
                passes.append("graduation: 5 obs → blended")
            else:
                warnings.append(f"expected blended at 5 obs, got {stage5}")
        else:
            issues.append(f"graduation predict (5 obs) failed: {resp3.status_code}")

        section("7. Latency (10 predicts)")
        start = time.perf_counter()
        ok_lat = True
        for _ in range(10):
            r = client.post("/cold-start/predict", json=NEW_DRUG)
            if r.status_code != 200:
                ok_lat = False
                break
        elapsed = time.perf_counter() - start
        print(f"  10 predicts in {elapsed:.3f}s")
        if ok_lat and elapsed < 5.0:
            passes.append(f"10 predicts in {elapsed:.2f}s (<5s with 10k catalog)")
        elif ok_lat:
            warnings.append(f"10 predicts took {elapsed:.2f}s — slow with large catalog")

    section("SUMMARY")
    print(f"  PASS:   {len(passes)}")
    for p in passes:
        print(f"    ✓ {p}")
    print(f"  WARN:   {len(warnings)}")
    for w in warnings:
        print(f"    ! {w}")
    print(f"  ISSUES: {len(issues)}")
    for i in issues:
        print(f"    ✗ {i}")

    if issues:
        print("\nVerdict: NEEDS ATTENTION — pipeline runs but has blocking issues.")
        return 1
    if warnings:
        print("\nVerdict: PARTIALLY WORKING — core pipeline OK; data/training gaps limit quality.")
        return 0
    print("\nVerdict: WORKING WELL — cold start forecasting operational.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
