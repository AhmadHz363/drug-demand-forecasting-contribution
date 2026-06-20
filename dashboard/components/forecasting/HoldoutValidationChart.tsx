"use client";

import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { LineChart } from "lucide-react";

import type { HoldoutResponse } from "@/lib/types";

import { EmptyState } from "@/components/cold-start/ui";

type SeriesKey = "ensemble" | "sarima" | "lgbm" | "tft";

interface HoldoutValidationChartProps {
  data: HoldoutResponse | null;
  showSeries?: SeriesKey[];
  embedded?: boolean;
}

function formatDate(iso: string): string {
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

const SERIES_CONFIG: Record<
  Exclude<SeriesKey, "ensemble">,
  { key: keyof HoldoutResponse["series"][0]; label: string; color: string; dash?: string }
> = {
  sarima: { key: "sarima_p50", label: "SARIMA (P50)", color: "#7c3aed" },
  lgbm: { key: "lgbm_p50", label: "LightGBM (P50)", color: "#059669" },
  tft: { key: "tft_p50", label: "TFT (P50)", color: "#d97706" },
};

export function HoldoutValidationChart({
  data,
  showSeries = ["ensemble"],
  embedded = false,
}: HoldoutValidationChartProps) {
  const { chartData, ensembleMetrics } = useMemo(() => {
    if (!data?.series.length) {
      return { chartData: [], ensembleMetrics: null };
    }
    const chartData = data.series.map((point) => ({
      ...point,
      label: formatDate(point.date),
      bandBase: point.ensemble_p10 ?? 0,
      bandRange: (point.ensemble_p90 ?? 0) - (point.ensemble_p10 ?? 0),
    }));
    return { chartData, ensembleMetrics: data.metrics.ensemble ?? null };
  }, [data]);

  if (!data?.series.length) {
    return (
      <EmptyState
        icon={<LineChart className="h-6 w-6" />}
        title="No hold-out data"
        description="Run hold-out validation to compare predicted demand against actual historical values."
      />
    );
  }

  const showEnsemble = showSeries.includes("ensemble");
  const wrapperClass = embedded ? "" : "rounded-xl border border-slate-200 bg-white p-5 shadow-sm";

  return (
    <div className={wrapperClass}>
      {!embedded && (
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Actual vs Predicted</h2>
      )}
      <p className="mb-4 text-xs text-slate-500">
        Trained on {data.train_period.start} → {data.train_period.end} · Tested on{" "}
        {data.test_period.start} → {data.test_period.end}
      </p>

      {ensembleMetrics && showEnsemble && (
        <div className="mb-4 grid gap-3 sm:grid-cols-3">
          <SummaryCard label="Ensemble sMAPE" value={`${ensembleMetrics.smape.toFixed(1)}%`} highlight />
          <SummaryCard label="Ensemble MAE" value={`${ensembleMetrics.mae.toFixed(1)} units`} />
          <SummaryCard
            label="90% coverage"
            value={`${(ensembleMetrics.coverage_90 * 100).toFixed(0)}%`}
          />
        </div>
      )}

      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#64748b" }} />
          <YAxis tick={{ fontSize: 11, fill: "#64748b" }} />
          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as (typeof chartData)[0];
              return (
                <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg">
                  <p className="mb-1.5 font-semibold text-slate-800">{label}</p>
                  <p className="text-slate-800">
                    Actual: <span className="font-medium">{row.actual.toFixed(2)}</span>
                  </p>
                  {row.ensemble_p50 != null && (
                    <p className="text-blue-700">
                      Ensemble P50: <span className="font-medium">{row.ensemble_p50.toFixed(2)}</span>
                    </p>
                  )}
                  {row.sarima_p50 != null && (
                    <p className="text-violet-700">SARIMA: {row.sarima_p50.toFixed(2)}</p>
                  )}
                  {row.lgbm_p50 != null && (
                    <p className="text-emerald-700">LightGBM: {row.lgbm_p50.toFixed(2)}</p>
                  )}
                  {row.tft_p50 != null && (
                    <p className="text-amber-700">TFT: {row.tft_p50.toFixed(2)}</p>
                  )}
                </div>
              );
            }}
          />
          <Legend
            formatter={(value) => {
              const labels: Record<string, string> = {
                actual: "Actual demand",
                ensemble_p50: "Ensemble (P50)",
                bandBase: "Ensemble band (P10–P90)",
                sarima_p50: "SARIMA (P50)",
                lgbm_p50: "LightGBM (P50)",
                tft_p50: "TFT (P50)",
              };
              return labels[value] ?? value;
            }}
          />

          {showEnsemble && (
            <>
              <Area
                type="monotone"
                dataKey="bandBase"
                stackId="band"
                stroke="none"
                fill="transparent"
                legendType="none"
              />
              <Area
                type="monotone"
                dataKey="bandRange"
                stackId="band"
                stroke="none"
                fill="#93c5fd"
                fillOpacity={0.25}
                name="bandBase"
              />
              <Line
                type="monotone"
                dataKey="ensemble_p50"
                stroke="#2563eb"
                strokeWidth={2}
                strokeDasharray="6 4"
                dot={{ r: 2, fill: "#2563eb" }}
                name="ensemble_p50"
              />
            </>
          )}

          <Line
            type="monotone"
            dataKey="actual"
            stroke="#0f172a"
            strokeWidth={2.5}
            dot={{ r: 3, fill: "#0f172a" }}
            name="actual"
          />

          {(Object.keys(SERIES_CONFIG) as Exclude<SeriesKey, "ensemble">[]).map((model) => {
            if (!showSeries.includes(model)) return null;
            const cfg = SERIES_CONFIG[model];
            return (
              <Line
                key={model}
                type="monotone"
                dataKey={cfg.key}
                stroke={cfg.color}
                strokeWidth={1.5}
                strokeDasharray="4 4"
                dot={false}
                name={cfg.key}
              />
            );
          })}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function SummaryCard({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={`rounded-xl border px-4 py-3 ${
        highlight ? "border-blue-200 bg-blue-50" : "border-slate-200 bg-slate-50"
      }`}
    >
      <p className="text-xs font-medium text-slate-500">{label}</p>
      <p className={`mt-0.5 text-lg font-bold ${highlight ? "text-blue-700" : "text-slate-800"}`}>
        {value}
      </p>
    </div>
  );
}
