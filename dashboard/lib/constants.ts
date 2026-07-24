import type { DrugMetadataInput, GraduationStage } from "./types";

export const THERAPEUTIC_CLASSES = [
  "antibiotic",
  "analgesic",
  "antifungal",
  "antiviral",
  "anticoagulant",
  "antihypertensive",
  "antidiabetic",
  "antiemetic",
  "bronchodilator",
  "corticosteroid",
  "diuretic",
  "immunosuppressant",
  "other",
] as const;

export const PHARMA_FORMS = [
  "tablet",
  "capsule",
  "injection",
  "syrup",
  "suspension",
  "cream",
  "ointment",
  "patch",
  "inhaler",
  "suppository",
  "drops",
  "other",
] as const;

export const ROUTES = [
  "oral",
  "iv",
  "im",
  "sc",
  "topical",
  "inhalation",
  "rectal",
  "ophthalmic",
  "other",
] as const;

export const DEFAULT_DRUG_METADATA: DrugMetadataInput = {
  drug_code: "NEW-001",
  drug_name: "Ceftriaxone 1g Injection",
  therapeutic_class: "antibiotic",
  atc_category: "J01",
  pharmaceutical_form: "injection",
  ven_class: "V",
  abc_class: "A",
  unit_price_tier: 4,
  requires_refrigeration: false,
  is_controlled_substance: false,
  average_shelf_life_days: 730,
  route_of_administration: "iv",
};

export const COLD_START_ONLY_BELOW = 4;
export const BLEND_UNTIL = 12;

export function decideStageFromCount(observationCount: number): GraduationStage {
  if (observationCount < COLD_START_ONLY_BELOW) return "cold_start_only";
  if (observationCount < BLEND_UNTIL) return "blended";
  return "full_ensemble";
}

export function mamlWeightPercent(observationCount: number): number {
  if (observationCount < COLD_START_ONLY_BELOW) return 0;
  if (observationCount >= BLEND_UNTIL) return 90;
  return ((observationCount - COLD_START_ONLY_BELOW) / (BLEND_UNTIL - COLD_START_ONLY_BELOW)) * 90;
}

export function coldStartRemainingPercent(observationCount: number): number {
  if (observationCount < COLD_START_ONLY_BELOW) return 100;
  if (observationCount >= BLEND_UNTIL) return 0;
  const alpha = (observationCount - COLD_START_ONLY_BELOW) / (BLEND_UNTIL - COLD_START_ONLY_BELOW);
  return Math.round((1 - alpha) * 100);
}

export const MAX_FORECAST_HORIZON = 30;

export const FORECAST_MODELS = ["sarima", "lgbm", "classical"] as const;

export type ForecastModelId = (typeof FORECAST_MODELS)[number];

export const FORECAST_MODEL_LABELS: Record<ForecastModelId, string> = {
  sarima: "SARIMA",
  lgbm: "LightGBM",
  classical: "Classical",
};

export const EXAMPLE_FORECAST_DRUGS = [
  { code: "E2E-AMOX", name: "Amoxicillin 500mg" },
  { code: "E2E-PARA", name: "Paracetamol 500mg" },
  { code: "E2E-METF", name: "Metformin 850mg" },
  { code: "E2E-WARF", name: "Warfarin 5mg" },
  { code: "E2E-SALB", name: "Salbutamol inhaler" },
  { code: "E2E-OMEP", name: "Omeprazole 20mg" },
] as const;

export const DEFAULT_FORECAST_DRUG = EXAMPLE_FORECAST_DRUGS[0].code;
