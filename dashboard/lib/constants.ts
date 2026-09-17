import type { DrugMetadataInput, GraduationStage } from "./types";

export const DRUG_CLASSES = [
  "ARB",
  "ACE inhibitor",
  "Beta blocker",
  "Calcium channel blocker",
  "Antibiotic",
  "Analgesic",
  "Anticoagulant",
  "Antidiabetic",
  "Corticosteroid",
  "other",
] as const;

export const DOSAGE_FORMS = [
  "Tablet",
  "Capsule",
  "Injection",
  "Syrup",
  "Suspension",
  "Cream",
  "Ointment",
  "Inhaler",
  "Drops",
  "other",
] as const;

export const ROUTES = [
  "Oral",
  "IV",
  "IM",
  "SC",
  "Topical",
  "Inhalation",
  "Rectal",
  "Ophthalmic",
  "other",
] as const;

export const PREGNANCY_CATEGORIES = [
  "Contraindicated (not allowed)",
  "Use with caution",
  "Generally safe",
  "Unknown",
  "MISSING",
] as const;

export const AVAILABILITY_OPTIONS = [
  "Prescription",
  "OTC",
  "Hospital use only",
  "MISSING",
] as const;

/** @deprecated use DRUG_CLASSES — kept for any stale imports */
export const THERAPEUTIC_CLASSES = DRUG_CLASSES;
/** @deprecated use DOSAGE_FORMS */
export const PHARMA_FORMS = DOSAGE_FORMS;

export const DEFAULT_DRUG_METADATA: DrugMetadataInput = {
  drug_code: "NEW-001",
  drug_name: "Ceftriaxone 1g Injection",
  generic_name: "Ceftriaxone",
  drug_class: "Antibiotic",
  dosage_form: "Injection",
  strength: "1 g",
  route_of_administration: "IV",
  pregnancy_category: "Use with caution",
  availability: "Prescription",
  indications: "Severe bacterial infections",
  side_effects: "Diarrhea; rash",
  contraindications: "Hypersensitivity to cephalosporins",
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
