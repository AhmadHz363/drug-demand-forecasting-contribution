"use client";

import { useState } from "react";
import { BarChart3, CalendarDays, Sparkles } from "lucide-react";

import { accuracyAccent, forecastResponseAccuracy } from "@/lib/forecastMetrics";
import type { ForecastResponse } from "@/lib/types";
import { aggregateForecastByWeek } from "@/lib/weeklyForecast";

import { ForecastChart } from "@/components/cold-start/ForecastChart";
import { EmptyState, Skeleton, StatChip } from "@/components/cold-start/ui";
import { WeeklyForecastChart } from "@/components/forecasting/WeeklyForecastChart";

interface ForecastingResultsTabsProps {
  data: ForecastResponse | null;
  isLoading: boolean;
  onScrollToForm: () => void;
}

type ViewMode = "weekly" | "daily";

export function ForecastingResultsTabs({
  data,
  isLoading,
  onScrollToForm,
}: ForecastingResultsTabsProps) {
  const [view, setView] = useState<ViewMode>("weekly");

  if (isLoading) {
    return (
      <div id="forecasting-results-section" className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
        <Skeleton className="mb-4 h-8 w-48" />
        <Skeleton className="h-72 w-full" />
      </div>
    );
  }

  if (!data) {
    return (
      <div id="forecasting-results-section" className="rounded-2xl border border-slate-200 bg-white shadow-sm">
        <EmptyState
          icon={<Sparkles className="h-6 w-6" />}
          title="No forecast yet"
          description="Train SHIELD-XR, pick a drug from the enriched panel, and generate a weekly forecast."
          action={
            <button
              type="button"
              onClick={onScrollToForm}
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
            >
              Configure forecast
            </button>
          }
        />
      </div>
    );
  }

  const drugAccuracy = forecastResponseAccuracy(data);
  const weeklyBuckets = aggregateForecastByWeek(data.forecast);
  const weeklyTotal = weeklyBuckets.reduce((s, w) => s + w.p50, 0);

  return (
    <div id="forecasting-results-section" className="space-y-4">
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-blue-600">Results</p>
            <h2 className="mt-1 text-xl font-bold text-slate-900">{data.drug_code}</h2>
            <p className="mt-0.5 text-sm text-slate-500">
              {weeklyBuckets.length} week{weeklyBuckets.length === 1 ? "" : "s"} ·{" "}
              {data.demand_segment ?? "unknown"} segment
              {data.center_syn_id ? ` · ${data.center_syn_id}` : ""}
            </p>
          </div>

          <div className="flex rounded-xl border border-slate-200 bg-slate-50 p-1">
            <button
              type="button"
              onClick={() => setView("weekly")}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                view === "weekly"
                  ? "bg-white text-blue-700 shadow-sm"
                  : "text-slate-600 hover:text-slate-800"
              }`}
            >
              <BarChart3 className="h-3.5 w-3.5" />
              Weekly
            </button>
            <button
              type="button"
              onClick={() => setView("daily")}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                view === "daily"
                  ? "bg-white text-blue-700 shadow-sm"
                  : "text-slate-600 hover:text-slate-800"
              }`}
            >
              <CalendarDays className="h-3.5 w-3.5" />
              Daily
            </button>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatChip
            label="Weekly total (P50)"
            value={`${weeklyTotal.toFixed(0)} u`}
            accent="green"
          />
          <StatChip
            label="Hold-out WAPE accuracy"
            value={drugAccuracy != null ? `${drugAccuracy.toFixed(1)}%` : "—"}
            accent={drugAccuracy != null ? accuracyAccent(drugAccuracy) : "slate"}
          />
          <StatChip label="SB demand class" value={data.demand_segment ?? "—"} accent="amber" />
          <StatChip
            label="Class winner"
            value={
              data.model_weights.lgbm >= 0.99
                ? "SHIELD-XR"
                : data.model_weights.sarima >= 0.99
                  ? "Plain-Tweedie"
                  : "Plain-L1"
            }
            accent="blue"
          />
        </div>

        {data.uncertainty_note && (
          <p className="mt-4 rounded-xl border border-amber-100 bg-amber-50/80 px-4 py-2.5 text-xs text-amber-900">
            {data.uncertainty_note}
          </p>
        )}

        <div className="mt-6">
          {view === "weekly" ? (
            <WeeklyForecastChart
              forecast={data.forecast}
              history={data.history}
              drugCode={data.drug_code}
            />
          ) : (
            <ForecastChart forecast={data.forecast} history={data.history} embedded />
          )}
        </div>
      </div>
    </div>
  );
}
