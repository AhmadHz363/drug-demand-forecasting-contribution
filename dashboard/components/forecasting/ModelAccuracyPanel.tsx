"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { CheckCircle2, Gauge, Loader2, RefreshCw, Target, XCircle } from "lucide-react";

import { fetchForecastingPerformance } from "@/lib/api";
import {
  accuracyAccent,
  meetsPerDrugAccuracyTarget,
  meetsWeeklyAccuracyTarget,
  performanceRowAccuracy,
} from "@/lib/forecastMetrics";
import type { ForecastResponse, ModelPerformanceRow, ShieldXRAccuracySummary } from "@/lib/types";
import { StatChip, Panel, PanelBody, PanelHeader } from "@/components/cold-start/ui";

interface ModelAccuracyPanelProps {
  selectedDrugCode?: string | null;
  forecastData?: ForecastResponse | null;
  accuracySummary?: ShieldXRAccuracySummary | null;
  trainingRunId?: string | null;
}

function formatPct(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${value.toFixed(1)}%`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function TargetBadge({
  met,
  label,
}: {
  met: boolean;
  label: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium ${
        met ? "bg-emerald-50 text-emerald-800" : "bg-amber-50 text-amber-800"
      }`}
    >
      {met ? <CheckCircle2 className="h-3 w-3" /> : <XCircle className="h-3 w-3" />}
      {label}
    </span>
  );
}

export function ModelAccuracyPanel({
  selectedDrugCode,
  forecastData,
  accuracySummary,
  trainingRunId,
}: ModelAccuracyPanelProps) {
  const [rows, setRows] = useState<ModelPerformanceRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchForecastingPerformance({
        model_name: "shield_xr_ensemble",
        training_run_id: trainingRunId ?? undefined,
        limit: 200,
      });
      setRows(result.items);
    } catch (err) {
      setRows([]);
      setError(err instanceof Error ? err.message : "Failed to load per-drug metrics");
    } finally {
      setLoading(false);
    }
  }, [trainingRunId]);

  useEffect(() => {
    if (trainingRunId) {
      void load();
    }
  }, [load, trainingRunId]);

  const latestByDrug = useMemo(() => {
    const map = new Map<string, ModelPerformanceRow>();
    for (const row of rows) {
      const existing = map.get(row.drug_code);
      if (!existing || new Date(row.evaluated_at) > new Date(existing.evaluated_at)) {
        map.set(row.drug_code, row);
      }
    }
    return [...map.values()].sort((a, b) => a.drug_code.localeCompare(b.drug_code));
  }, [rows]);

  const selectedRow = selectedDrugCode
    ? latestByDrug.find((r) => r.drug_code === selectedDrugCode)
    : undefined;
  const selectedAccuracy = selectedRow ? performanceRowAccuracy(selectedRow) : null;

  const weeklyTargetMet = meetsWeeklyAccuracyTarget(accuracySummary);
  const perDrugTargetMet = meetsPerDrugAccuracyTarget(accuracySummary);
  const sbClassEntries = Object.entries(accuracySummary?.per_sb_class_weekly_accuracy_pct ?? {});

  const hasSummary = Boolean(
    accuracySummary &&
      (accuracySummary.hospital_weekly_accuracy_pct != null ||
        accuracySummary.reconciled_weekly_per_drug_mean_accuracy_pct != null),
  );

  return (
    <Panel id="forecasting-accuracy-panel">
      <PanelHeader
        title="Model accuracy"
        description="Notebook-aligned WAPE accuracy: hospital weekly total (target ≥90%) and reconciled per-drug weekly mean (target ≥70%)."
        action={
          <button
            type="button"
            onClick={() => void load()}
            disabled={loading || !trainingRunId}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
          >
            {loading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
            Refresh drugs
          </button>
        }
      />
      <PanelBody className="space-y-5">
        {!hasSummary && !loading && (
          <div className="flex items-start gap-3 rounded-xl border border-dashed border-slate-200 bg-slate-50/80 px-4 py-4">
            <Gauge className="mt-0.5 h-5 w-5 shrink-0 text-slate-400" />
            <div>
              <p className="text-sm font-medium text-slate-800">No accuracy metrics yet</p>
              <p className="mt-1 text-xs leading-relaxed text-slate-500">
                Train SHIELD-XR on ingested hospital data to evaluate weekly hospital accuracy and
                per-drug weekly breakdown accuracy.
              </p>
            </div>
          </div>
        )}

        {hasSummary && accuracySummary && (
          <>
            <div className="flex flex-wrap gap-2">
              <TargetBadge met={weeklyTargetMet} label="Weekly hospital ≥90%" />
              <TargetBadge met={perDrugTargetMet} label="Per-drug weekly mean ≥70%" />
            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <StatChip
                label="Hospital weekly"
                value={formatPct(accuracySummary.hospital_weekly_accuracy_pct)}
                accent={
                  accuracySummary.hospital_weekly_accuracy_pct != null
                    ? accuracyAccent(accuracySummary.hospital_weekly_accuracy_pct)
                    : "slate"
                }
              />
              <StatChip
                label="Per-drug weekly (mean)"
                value={formatPct(accuracySummary.reconciled_weekly_per_drug_mean_accuracy_pct)}
                accent={
                  accuracySummary.reconciled_weekly_per_drug_mean_accuracy_pct != null
                    ? accuracyAccent(accuracySummary.reconciled_weekly_per_drug_mean_accuracy_pct)
                    : "slate"
                }
              />
              <StatChip
                label="Per-drug weekly (median)"
                value={formatPct(accuracySummary.reconciled_weekly_per_drug_median_accuracy_pct)}
                accent="blue"
              />
              <StatChip
                label="Daily ensemble"
                value={formatPct(accuracySummary.daily_ensemble_accuracy_pct)}
                accent="slate"
              />
            </div>

            {sbClassEntries.length > 0 && (
              <div className="rounded-xl border border-slate-200 bg-slate-50/60 px-4 py-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Weekly accuracy by demand segment
                </p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {sbClassEntries.map(([segment, pct]) => (
                    <span
                      key={segment}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-white px-2.5 py-1 text-xs ring-1 ring-slate-200"
                    >
                      <span className="capitalize text-slate-600">{segment}</span>
                      <span
                        className={`font-semibold ${
                          pct >= 70 ? "text-emerald-700" : pct >= 55 ? "text-blue-700" : "text-slate-700"
                        }`}
                      >
                        {pct.toFixed(1)}%
                      </span>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {selectedDrugCode && (
          <StatChip
            label={`Selected drug (${selectedDrugCode})`}
            value={formatPct(selectedAccuracy)}
            accent={selectedAccuracy != null ? accuracyAccent(selectedAccuracy) : "slate"}
          />
        )}

        {forecastData?.validation_metrics_note && (
          <p className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs leading-relaxed text-slate-600">
            {forecastData.validation_metrics_note}
          </p>
        )}

        {error && trainingRunId && (
          <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
            Per-drug table unavailable ({error}). Summary metrics above are from the latest training
            run.
          </p>
        )}

        {latestByDrug.length > 0 && (
          <>
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
              <span className="inline-flex items-center gap-1 rounded-lg bg-slate-100 px-2.5 py-1">
                <Target className="h-3 w-3" />
                Per-drug weekly accuracy (reconciled breakdown)
              </span>
              {trainingRunId && (
                <span className="rounded-lg bg-slate-100 px-2.5 py-1 font-mono text-slate-500">
                  run {trainingRunId.slice(0, 8)}
                </span>
              )}
            </div>

            <div className="max-h-72 overflow-auto rounded-xl border border-slate-200">
              <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
                <thead className="sticky top-0 bg-slate-50 text-slate-500">
                  <tr>
                    <th className="px-3 py-2 font-medium">Drug</th>
                    <th className="px-3 py-2 font-medium">Weekly accuracy</th>
                    <th className="px-3 py-2 font-medium">Segment</th>
                    <th className="px-3 py-2 font-medium">Evaluated</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white text-slate-800">
                  {latestByDrug.map((row) => {
                    const accuracy = performanceRowAccuracy(row);
                    const isSelected = row.drug_code === selectedDrugCode;
                    return (
                      <tr
                        key={`${row.drug_code}-${row.evaluated_at}`}
                        className={isSelected ? "bg-blue-50/60" : undefined}
                      >
                        <td className="px-3 py-2 font-medium">{row.drug_code}</td>
                        <td className="px-3 py-2">
                          <span
                            className={
                              accuracy != null && accuracy >= 85
                                ? "font-semibold text-emerald-700"
                                : accuracy != null && accuracy >= 70
                                  ? "font-semibold text-blue-700"
                                  : ""
                            }
                          >
                            {formatPct(accuracy)}
                          </span>
                        </td>
                        <td className="px-3 py-2 capitalize">{row.demand_segment ?? "—"}</td>
                        <td className="px-3 py-2 text-slate-500">
                          {formatDate(row.evaluated_at)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </PanelBody>
    </Panel>
  );
}
