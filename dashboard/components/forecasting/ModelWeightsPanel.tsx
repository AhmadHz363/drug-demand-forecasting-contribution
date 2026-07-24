"use client";

import { Layers } from "lucide-react";

import { FORECAST_MODEL_LABELS, type ForecastModelId } from "@/lib/constants";
import type { ModelWeightBreakdown } from "@/lib/types";

import { EmptyState } from "@/components/cold-start/ui";

const MODEL_COLORS: Record<ForecastModelId, string> = {
  sarima: "bg-violet-500",
  lgbm: "bg-blue-500",
  classical: "bg-emerald-500",
};

interface ModelWeightsPanelProps {
  weights: ModelWeightBreakdown | null;
  embedded?: boolean;
}

export function ModelWeightsPanel({ weights, embedded = false }: ModelWeightsPanelProps) {
  if (!weights) {
    return (
      <EmptyState
        icon={<Layers className="h-6 w-6" />}
        title="No model weights"
        description="Run a forecast to see how SARIMA, LightGBM, and Classical contribute to the ensemble."
      />
    );
  }

  const entries = (Object.keys(MODEL_COLORS) as ForecastModelId[]).map((key) => ({
    key,
    label: FORECAST_MODEL_LABELS[key],
    value: weights[key],
    color: MODEL_COLORS[key],
  }));

  const total = entries.reduce((s, e) => s + e.value, 0);
  const normalized = total > 0 ? entries.map((e) => ({ ...e, pct: (e.value / total) * 100 })) : entries.map((e) => ({ ...e, pct: 0 }));

  const wrapperClass = embedded ? "" : "rounded-xl border border-slate-200 bg-white p-5 shadow-sm";

  return (
    <div className={wrapperClass}>
      {!embedded && (
        <h2 className="mb-4 text-lg font-semibold text-slate-900">Ensemble Model Weights</h2>
      )}

      <div className="mb-6 flex h-4 overflow-hidden rounded-full bg-slate-100">
        {normalized.map((entry) =>
          entry.pct > 0 ? (
            <div
              key={entry.key}
              className={`${entry.color} transition-all`}
              style={{ width: `${entry.pct}%` }}
              title={`${entry.label}: ${entry.pct.toFixed(1)}%`}
            />
          ) : null,
        )}
      </div>

      <div className="space-y-3">
        {normalized.map((entry) => (
          <div key={entry.key} className="flex items-center gap-3">
            <span className={`h-3 w-3 shrink-0 rounded-full ${entry.color}`} />
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm font-medium text-slate-800">{entry.label}</span>
                <span className="text-sm font-semibold text-slate-900">
                  {entry.pct.toFixed(1)}%
                </span>
              </div>
              <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-slate-100">
                <div
                  className={`h-full rounded-full ${entry.color} transition-all`}
                  style={{ width: `${entry.pct}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      <p className="mt-4 text-xs leading-relaxed text-slate-500">
        Weights are derived from inverse validation sMAPE via the stacking meta-learner. Higher
        weight means the model contributed more reliably on recent holdout data.
      </p>
    </div>
  );
}
