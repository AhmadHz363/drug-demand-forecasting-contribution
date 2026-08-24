import type {
  ForecastResponse,
  ModelPerformanceRow,
  ShieldXRAccuracySummary,
} from "./types";

/** SHIELD-XR stores MASE as (1 − WAPE accuracy) for legacy API compatibility. */
export function accuracyPctFromMase(mase: number | null | undefined): number | null {
  if (mase == null) return null;
  return Math.max(0, (1 - mase) * 100);
}

/** SHIELD-XR maps WAPE to a 0–200 sMAPE-like scale in `smape` fields. */
export function accuracyPctFromSmape(smape: number | null | undefined): number | null {
  if (smape == null) return null;
  return Math.max(0, (1 - smape / 200) * 100);
}

export function performanceRowAccuracy(row: ModelPerformanceRow): number | null {
  return accuracyPctFromMase(row.mase) ?? accuracyPctFromSmape(row.smape);
}

export function forecastResponseAccuracy(data: ForecastResponse): number | null {
  return (
    accuracyPctFromMase(data.mase_validation_full_window) ??
    accuracyPctFromSmape(data.smape_last_validation)
  );
}

export function trainSummaryAccuracy(smapeSummary: Record<string, number>): number | null {
  const values = Object.values(smapeSummary);
  if (values.length === 0) return null;
  const avgSmape = values.reduce((s, v) => s + v, 0) / values.length;
  return accuracyPctFromSmape(avgSmape);
}

export function meetsWeeklyAccuracyTarget(summary: ShieldXRAccuracySummary | null | undefined): boolean {
  const weekly = summary?.hospital_weekly_accuracy_pct;
  return weekly != null && weekly >= 90;
}

export function meetsPerDrugAccuracyTarget(summary: ShieldXRAccuracySummary | null | undefined): boolean {
  const nonLumpy = summary?.non_lumpy_weekly_accuracy_pct;
  if (nonLumpy != null) return nonLumpy >= 70;
  const mean = summary?.reconciled_weekly_per_drug_mean_accuracy_pct;
  return mean != null && mean >= 70;
}

export function meetsHybridAbcTarget(summary: ShieldXRAccuracySummary | null | undefined): boolean {
  const hybrid = summary?.hybrid_abc_combined_accuracy_pct;
  return hybrid != null && hybrid >= 85;
}

export function formatAccuracySummaryLine(summary: ShieldXRAccuracySummary): string {
  const parts: string[] = [];
  if (summary.hospital_weekly_accuracy_pct != null) {
    parts.push(`Hospital weekly ${summary.hospital_weekly_accuracy_pct.toFixed(1)}%`);
  }
  if (summary.reconciled_weekly_per_drug_mean_accuracy_pct != null) {
    parts.push(
      `Per-drug weekly mean ${summary.reconciled_weekly_per_drug_mean_accuracy_pct.toFixed(1)}%`,
    );
  }
  if (summary.daily_ensemble_accuracy_pct != null) {
    parts.push(`Daily ensemble ${summary.daily_ensemble_accuracy_pct.toFixed(1)}%`);
  }
  return parts.join(" · ");
}

export function accuracyAccent(pct: number): "green" | "amber" | "blue" | "slate" {
  if (pct >= 85) return "green";
  if (pct >= 70) return "blue";
  if (pct >= 55) return "amber";
  return "slate";
}
