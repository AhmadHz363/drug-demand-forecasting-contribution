"use client";

import { Info } from "lucide-react";

import type { CameoAccuracySummary } from "@/lib/types";

interface CameoAccuracyReportProps {
  summary: CameoAccuracySummary | null;
  trainedAt?: string | null;
}

function ComparisonRow({
  label,
  cameo,
  analogous,
}: {
  label: string;
  cameo: number | null | undefined;
  analogous: number | null | undefined;
}) {
  return (
    <div className="grid grid-cols-[1fr_auto_auto] items-center gap-3 rounded-xl bg-slate-50 px-4 py-3 ring-1 ring-slate-100">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <span className="text-sm font-bold tabular-nums text-blue-700">
        CAMEO {cameo != null ? `${cameo.toFixed(1)}%` : "—"}
      </span>
      <span className="text-sm tabular-nums text-slate-500">
        Analogous {analogous != null ? `${analogous.toFixed(1)}%` : "—"}
      </span>
    </div>
  );
}

export function CameoAccuracyReport({ summary, trainedAt }: CameoAccuracyReportProps) {
  if (!summary || summary.n_coldstart_test_drugs === 0) {
    return null;
  }

  const formattedTrainedAt = trainedAt
    ? new Date(trainedAt).toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : null;

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-slate-900">Validation methodology</h3>
          <p className="mt-1 max-w-2xl text-sm text-slate-500">
            Leave-drugs-out cold-start simulation on matched-source library drugs — same protocol
            as the CAMEO real-data notebook. {summary.n_coldstart_test_drugs} held-out drugs,{" "}
            {summary.forecast_horizon_weeks}-week horizon, WAPE accuracy = 1 − WAPE.
          </p>
        </div>
        {formattedTrainedAt && (
          <span className="rounded-lg bg-slate-100 px-3 py-1.5 text-xs text-slate-600">
            Trained {formattedTrainedAt}
          </span>
        )}
      </div>

      <div className="mt-5 space-y-2">
        <ComparisonRow
          label="Pooled (volume-weighted) accuracy"
          cameo={summary.pooled_accuracy_pct.CAMEO}
          analogous={summary.pooled_accuracy_pct.Analogous}
        />
        <ComparisonRow
          label="Median per-drug accuracy (robust)"
          cameo={summary.median_accuracy_pct.CAMEO}
          analogous={summary.median_accuracy_pct.Analogous}
        />
        <ComparisonRow
          label="Win rate (lowest WAPE per drug)"
          cameo={summary.win_rate_pct.CAMEO}
          analogous={summary.win_rate_pct.Analogous}
        />
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        {summary.empirical_conformal_coverage_pct != null && (
          <div className="rounded-xl border border-slate-200 px-4 py-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Conformal coverage
            </p>
            <p className="mt-1 text-2xl font-bold text-slate-900">
              {summary.empirical_conformal_coverage_pct.toFixed(1)}%
            </p>
            <p className="mt-1 text-xs text-slate-500">Target 90% · leave-one-out on library</p>
          </div>
        )}
        {summary.conformal_half_width_weekly != null && (
          <div className="rounded-xl border border-slate-200 px-4 py-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Conformal half-width (weekly)
            </p>
            <p className="mt-1 text-2xl font-bold text-slate-900">
              ±{summary.conformal_half_width_weekly.toFixed(1)} units
            </p>
            <p className="mt-1 text-xs text-slate-500">Used for prediction intervals</p>
          </div>
        )}
      </div>

      <div className="mt-5 flex items-start gap-2 rounded-xl border border-blue-100 bg-blue-50/60 px-4 py-3 text-xs leading-relaxed text-blue-900">
        <Info className="mt-0.5 h-4 w-4 shrink-0" />
        <p>
          Prefer pooled and median views over naive per-drug means — near-zero demand drugs can
          distort aggregate MASE/WAPE. Pooled accuracy sums all held-out drug-weeks before
          computing WAPE, analogous to hospital-total accuracy in SHIELD-XR.
          {summary.cameo_vs_analogous_wape_improvement_pct != null && (
            <>
              {" "}
              CAMEO reduces pooled WAPE by{" "}
              <strong>{summary.cameo_vs_analogous_wape_improvement_pct.toFixed(1)}%</strong> vs.
              the analogous baseline.
            </>
          )}
        </p>
      </div>
    </div>
  );
}
