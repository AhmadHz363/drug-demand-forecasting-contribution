"use client";

import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { LineChart } from "lucide-react";

import type { DailyDemandPoint, DailyForecast } from "@/lib/types";

import { EmptyState } from "./ui";

interface ForecastChartProps {
  forecast: DailyForecast[] | null;
  history?: DailyDemandPoint[];
  embedded?: boolean;
}

type ChartPoint = {
  date: string;
  label: string;
  p10?: number;
  p50?: number;
  p90?: number;
  bandBase?: number;
  bandRange?: number;
  historyQuantity?: number;
};

function formatDate(iso: string): string {
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function formatFullDate(iso: string): string {
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString(undefined, {
    weekday: "short",
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function ForecastChart({
  forecast,
  history,
  embedded = false,
}: ForecastChartProps) {
  const { chartData, p50Total, p10Total, p90Total, historyEndLabel, hasHistory } = useMemo(() => {
    if (!forecast?.length) {
      return {
        chartData: [] as ChartPoint[],
        p50Total: 0,
        p10Total: 0,
        p90Total: 0,
        historyEndLabel: null as string | null,
        hasHistory: false,
      };
    }

    const historyPoints: ChartPoint[] = (history ?? []).map((point) => ({
      date: point.date,
      label: formatDate(point.date),
      historyQuantity: point.quantity,
    }));

    const forecastPoints: ChartPoint[] = forecast.map((day) => ({
      ...day,
      label: formatDate(day.date),
      bandBase: day.p10,
      bandRange: day.p90 - day.p10,
    }));

    const chartData = [...historyPoints, ...forecastPoints];
    const p50Total = forecast.reduce((s, d) => s + d.p50, 0);
    const p10Total = forecast.reduce((s, d) => s + d.p10, 0);
    const p90Total = forecast.reduce((s, d) => s + d.p90, 0);
    const historyEndLabel =
      historyPoints.length > 0 ? historyPoints[historyPoints.length - 1].label : null;

    return {
      chartData,
      p50Total,
      p10Total,
      p90Total,
      historyEndLabel,
      hasHistory: historyPoints.length > 0,
    };
  }, [forecast, history]);

  if (!forecast) {
    return (
      <EmptyState
        icon={<LineChart className="h-6 w-6" />}
        title="No forecast data"
        description="Run a prediction to see P10, P50, and P90 demand bands over your chosen horizon."
      />
    );
  }

  const days = forecast.length;
  const wrapperClass = embedded ? "" : "rounded-xl border border-slate-200 bg-white p-5 shadow-sm";

  return (
    <div className={wrapperClass}>
      {!embedded && (
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Demand Forecast</h2>
      )}
      {hasHistory && (
        <p className="mb-4 text-sm text-slate-500">
          Slate line = historical demand · Blue dashed line = forecast P50 with uncertainty band
        </p>
      )}
      <div className="mb-4 grid gap-3 sm:grid-cols-3">
        <SummaryCard label={`${days}-day total (P50)`} value={`${p50Total.toFixed(1)} units`} highlight />
        <SummaryCard label="Pessimistic (P10)" value={`${p10Total.toFixed(1)} units`} />
        <SummaryCard label="Optimistic (P90)" value={`${p90Total.toFixed(1)} units`} />
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#64748b" }} minTickGap={24} />
          <YAxis tick={{ fontSize: 11, fill: "#64748b" }} />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as ChartPoint;
              const isHistory = row.historyQuantity != null;

              return (
                <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg">
                  <p className="mb-1.5 font-semibold text-slate-800">{formatFullDate(row.date)}</p>
                  {isHistory ? (
                    <p className="text-slate-700">
                      Historical demand:{" "}
                      <span className="font-medium">{row.historyQuantity!.toFixed(2)}</span>
                    </p>
                  ) : (
                    <>
                      <p className="text-slate-600">
                        P10: <span className="font-medium">{row.p10!.toFixed(2)}</span>
                      </p>
                      <p className="text-blue-700">
                        P50: <span className="font-medium">{row.p50!.toFixed(2)}</span>
                      </p>
                      <p className="text-slate-600">
                        P90: <span className="font-medium">{row.p90!.toFixed(2)}</span>
                      </p>
                    </>
                  )}
                </div>
              );
            }}
          />
          <Legend
            formatter={(value) => {
              const labels: Record<string, string> = {
                historyQuantity: "Historical demand",
                bandBase: "Uncertainty band (P10–P90)",
                p50: "Forecast (P50)",
                p10: "Pessimistic (P10)",
                p90: "Optimistic (P90)",
              };
              return labels[value] ?? value;
            }}
          />

          {hasHistory && historyEndLabel && (
            <ReferenceLine
              x={historyEndLabel}
              stroke="#cbd5e1"
              strokeDasharray="4 4"
              label={{
                value: "Forecast start",
                position: "insideTopRight",
                fill: "#94a3b8",
                fontSize: 10,
              }}
            />
          )}

          <Area
            type="monotone"
            dataKey="bandBase"
            stackId="band"
            stroke="none"
            fill="transparent"
            legendType="none"
            connectNulls={false}
          />
          <Area
            type="monotone"
            dataKey="bandRange"
            stackId="band"
            stroke="none"
            fill="#93c5fd"
            fillOpacity={0.35}
            name="bandBase"
            connectNulls={false}
          />

          {hasHistory && (
            <Line
              type="monotone"
              dataKey="historyQuantity"
              stroke="#475569"
              strokeWidth={2.5}
              dot={chartData.length <= 120 ? { r: 2.5, fill: "#475569" } : false}
              activeDot={{ r: 4 }}
              name="historyQuantity"
              connectNulls
            />
          )}

          <Line
            type="monotone"
            dataKey="p50"
            stroke="#2563eb"
            strokeWidth={2.5}
            strokeDasharray={hasHistory ? "6 4" : undefined}
            dot={{ r: 3, fill: "#2563eb" }}
            name="p50"
            connectNulls={false}
          />
          <Line
            type="monotone"
            dataKey="p10"
            stroke="#94a3b8"
            strokeWidth={1.5}
            strokeDasharray="4 4"
            dot={false}
            name="p10"
            connectNulls={false}
          />
          <Line
            type="monotone"
            dataKey="p90"
            stroke="#94a3b8"
            strokeWidth={1.5}
            strokeDasharray="4 4"
            dot={false}
            name="p90"
            connectNulls={false}
          />
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

export function ForecastChartSkeleton() {
  return (
    <div className="animate-pulse space-y-4">
      <div className="grid gap-3 sm:grid-cols-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-16 rounded-xl bg-slate-200" />
        ))}
      </div>
      <div className="h-80 rounded-xl bg-slate-200" />
    </div>
  );
}
