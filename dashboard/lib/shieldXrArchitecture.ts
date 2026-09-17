/** Architecture reference values from SHIELD-XR thesis (July 2026). */
export const SHIELD_XR_BENCHMARKS = {
  hospitalWeeklyAccuracy: 89.0,
  hybridAbcCombinedAccuracy: 86.4,
  dailyEnsembleAccuracy: 85.0,
  reconciledAllSkusAccuracy: 58.5,
  top50VolumeMeanAccuracy: 83.6,
  nonLumpySubsetAccuracy: 70.4,
} as const;

export const SHIELD_XR_SB_BENCHMARKS: Record<string, number> = {
  smooth: 76.0,
  intermittent: 66.0,
  erratic: 66.2,
  lumpy: 30.0,
};

export const SHIELD_XR_PIPELINE_STAGES = [
  {
    id: "anomaly-guard",
    title: "AnomalyGuard",
    detail: "Spike winsorization & dropout imputation on hospital-day artifacts",
  },
  {
    id: "sb-classification",
    title: "Demand classification",
    detail: "Syntetos–Boylan classes · lag & hospital-activity features",
  },
  {
    id: "daily-ensemble",
    title: "Daily 3-way ensemble",
    detail: "SHIELD-XR hurdle + EVT · Plain-Tweedie · Plain-L1 per class",
  },
  {
    id: "weekly-reconcile",
    title: "Weekly reconciliation",
    detail: "SKU weekly model scaled to trusted hospital-week total",
  },
  {
    id: "hybrid-abc",
    title: "Hybrid ABC report",
    detail: "Top-15 stable high-volume drugs + pooled tail bucket (~86%)",
  },
] as const;

export const SHIELD_XR_ACCURACY_VIEWS = [
  {
    label: "Hospital weekly total",
    target: "≥90%",
    benchmark: SHIELD_XR_BENCHMARKS.hospitalWeeklyAccuracy,
    description: "Bottom-up daily ensemble aggregated to hospital-week WAPE",
  },
  {
    label: "Hybrid ABC (operational)",
    target: "≥85%",
    benchmark: SHIELD_XR_BENCHMARKS.hybridAbcCombinedAccuracy,
    description: "Top-15 named drugs + All Other Drugs bucket for procurement",
  },
  {
    label: "All SKUs (reconciled)",
    target: "Transparency",
    benchmark: SHIELD_XR_BENCHMARKS.reconciledAllSkusAccuracy,
    description: "Full ~893-SKU breakdown — lumpy demand sets statistical floor",
  },
  {
    label: "Top-50 by volume",
    target: "High value",
    benchmark: SHIELD_XR_BENCHMARKS.top50VolumeMeanAccuracy,
    description: "Most procurement-critical drugs individually accurate",
  },
] as const;
