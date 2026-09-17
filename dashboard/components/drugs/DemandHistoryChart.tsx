"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart as RechartsLineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { LineChart } from "lucide-react";

import { EmptyState } from "@/components/cold-start/ui";
import type { DailyDemandPoint } from "@/lib/types";

interface DemandHistoryChartProps {
  series: DailyDemandPoint[];
  lookbackDays: number;
  embedded?: boolean;
}

function formatDate(iso: string): string {
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
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

export function DemandHistoryChart({
  series,
  lookbackDays,
  embedded = false,
}: DemandHistoryChartProps) {
  const { chartData, totalQuantity, peakDay } = useMemo(() => {
    if (!series.length) {
      return { chartData: [], totalQuantity: 0, peakDay: null as DailyDemandPoint | null };
    }

    let peak = series[0];
    let total = 0;
    const chartData = series.map((point) => {
      total += point.quantity;
      if (point.quantity > peak.quantity) {
        peak = point;
      }
      return {
        ...point,
        label: formatDate(point.date),
      };
    });

    return { chartData, totalQuantity: total, peakDay: peak };
  }, [series]);

  if (!series.length) {
    return (
      <EmptyState
        icon={<LineChart className="h-6 w-6" />}
        title="No demand history"
        description="This drug has no receipt quantities in the selected lookback window."
      />
    );
  }

  const wrapperClass = embedded ? "" : "rounded-xl border border-slate-200 bg-white p-5 shadow-sm";

  return (
    <div className={wrapperClass}>
      {!embedded && (
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Historical demand</h2>
      )}
      <p className="mb-4 text-sm text-slate-500">
        Daily quantities aggregated from receipt lines over the last {lookbackDays} days with data.
      </p>

      <div className="mb-4 grid gap-3 sm:grid-cols-3">
        <SummaryCard label="Total units" value={totalQuantity.toFixed(1)} highlight />
        <SummaryCard label="Days with demand" value={String(series.length)} />
        <SummaryCard
          label="Peak day"
          value={
            peakDay
              ? `${peakDay.quantity.toFixed(1)} on ${formatDate(peakDay.date)}`
              : "—"
          }
        />
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <RechartsLineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: "#64748b" }}
            minTickGap={24}
          />
          <YAxis tick={{ fontSize: 11, fill: "#64748b" }} />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as DailyDemandPoint & { label: string };
              return (
                <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg">
                  <p className="mb-1.5 font-semibold text-slate-800">{formatFullDate(row.date)}</p>
                  <p className="text-blue-700">
                    Quantity: <span className="font-medium">{row.quantity.toFixed(2)}</span>
                  </p>
                </div>
              );
            }}
          />
          <Line
            type="monotone"
            dataKey="quantity"
            stroke="#2563eb"
            strokeWidth={2}
            dot={series.length <= 60 ? { r: 2.5, fill: "#2563eb" } : false}
            activeDot={{ r: 4 }}
            name="Daily demand"
          />
        </RechartsLineChart>
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
      <p
        className={`mt-0.5 text-lg font-semibold tabular-nums ${
          highlight ? "text-blue-800" : "text-slate-900"
        }`}
      >
        {value}
      </p>
    </div>
  );
}
