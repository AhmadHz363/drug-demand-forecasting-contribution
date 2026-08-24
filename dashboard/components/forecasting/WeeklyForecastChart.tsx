"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CalendarRange } from "lucide-react";

import type { DailyDemandPoint, DailyForecast } from "@/lib/types";
import {
  aggregateForecastByWeek,
  aggregateHistoryByWeek,
  weekRangeLabel,
  type WeeklyForecastBucket,
} from "@/lib/weeklyForecast";

interface WeeklyForecastChartProps {
  forecast: DailyForecast[];
  history?: DailyDemandPoint[];
  drugCode?: string;
}

type ChartRow = {
  label: string;
  kind: "history" | "forecast";
  quantity?: number;
  p50?: number;
  p10?: number;
  p90?: number;
  weekRange?: string;
};

function buildChartRows(
  history: DailyDemandPoint[],
  weeklyForecast: WeeklyForecastBucket[],
): ChartRow[] {
  const historyWeeks = aggregateHistoryByWeek(history).slice(-8);
  const rows: ChartRow[] = historyWeeks.map((w) => ({
    label: w.weekLabel,
    kind: "history",
    quantity: w.quantity,
  }));

  for (const w of weeklyForecast) {
    rows.push({
      label: w.weekLabel,
      kind: "forecast",
      p50: w.p50,
      p10: w.p10,
      p90: w.p90,
      weekRange: weekRangeLabel(w.weekStart, w.weekEnd),
    });
  }
  return rows;
}

export function WeeklyForecastChart({ forecast, history = [], drugCode }: WeeklyForecastChartProps) {
  const weeklyForecast = aggregateForecastByWeek(forecast);
  const chartData = buildChartRows(history, weeklyForecast);
  const weeklyTotal = weeklyForecast.reduce((s, w) => s + w.p50, 0);
  const historyDivider = history.length
    ? aggregateHistoryByWeek(history).slice(-8).length - 0.5
    : -0.5;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-blue-600">
            Weekly breakdown
          </p>
          <h3 className="mt-1 text-lg font-bold text-slate-900">
            {drugCode ? `${drugCode} — ` : ""}
            {weeklyForecast.length} week forecast
          </h3>
        </div>
        <div className="rounded-xl bg-blue-50 px-4 py-2 text-right ring-1 ring-blue-100">
          <p className="text-xs font-medium text-blue-600">Reconciled weekly total (P50)</p>
          <p className="text-xl font-bold tabular-nums text-blue-900">{weeklyTotal.toFixed(0)} units</p>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
          <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#64748b" }} />
          <YAxis tick={{ fontSize: 11, fill: "#64748b" }} />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as ChartRow;
              return (
                <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg">
                  <p className="font-semibold text-slate-800">{row.label}</p>
                  {row.weekRange && <p className="text-slate-500">{row.weekRange}</p>}
                  {row.kind === "history" ? (
                    <p className="mt-1 text-slate-700">
                      Actual: <span className="font-medium">{row.quantity?.toFixed(0)}</span> units
                    </p>
                  ) : (
                    <>
                      <p className="mt-1 text-blue-700">
                        P50: <span className="font-medium">{row.p50?.toFixed(0)}</span>
                      </p>
                      <p className="text-slate-600">
                        P10–P90: {row.p10?.toFixed(0)} – {row.p90?.toFixed(0)}
                      </p>
                    </>
                  )}
                </div>
              );
            }}
          />
          <Legend
            formatter={(value) =>
              value === "quantity" ? "Historical weekly" : "Forecast weekly (P50)"
            }
          />
          {history.length > 0 && weeklyForecast.length > 0 && (
            <ReferenceLine
              x={chartData[Math.max(0, Math.floor(historyDivider))]?.label}
              stroke="#cbd5e1"
              strokeDasharray="4 4"
              label={{ value: "Forecast →", position: "insideTopRight", fontSize: 10, fill: "#94a3b8" }}
            />
          )}
          <Bar dataKey="quantity" name="quantity" radius={[6, 6, 0, 0]}>
            {chartData.map((row, i) => (
              <Cell key={`h-${i}`} fill={row.kind === "history" ? "#64748b" : "transparent"} />
            ))}
          </Bar>
          <Bar dataKey="p50" name="p50" radius={[6, 6, 0, 0]}>
            {chartData.map((row, i) => (
              <Cell key={`f-${i}`} fill={row.kind === "forecast" ? "#2563eb" : "transparent"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="grid gap-2 sm:grid-cols-3">
        {weeklyForecast.map((w) => (
          <div
            key={w.weekKey}
            className="rounded-xl border border-slate-200 bg-slate-50/80 px-3 py-2.5"
          >
            <div className="flex items-center gap-1.5 text-xs font-medium text-slate-500">
              <CalendarRange className="h-3.5 w-3.5" />
              {w.weekLabel}
              <span className="text-slate-400">· {w.dayCount}d</span>
            </div>
            <p className="mt-1 text-lg font-bold tabular-nums text-slate-900">{w.p50.toFixed(0)}</p>
            <p className="text-xs text-slate-500">
              {weekRangeLabel(w.weekStart, w.weekEnd)} · P10–P90 {w.p10.toFixed(0)}–{w.p90.toFixed(0)}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
