"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { EmptyState } from "@/components/cold-start/ui";

interface RankedBarChartProps {
  data: { label: string; value: number; sublabel?: string }[];
  valueLabel: string;
  color?: string;
  emptyTitle: string;
  emptyDescription: string;
}

function truncate(text: string, max = 22): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

export function RankedBarChart({
  data,
  valueLabel,
  color = "#2563eb",
  emptyTitle,
  emptyDescription,
}: RankedBarChartProps) {
  const chartData = useMemo(
    () =>
      [...data]
        .sort((a, b) => b.value - a.value)
        .map((item) => ({
          ...item,
          shortLabel: truncate(item.label),
        })),
    [data],
  );

  if (!chartData.length) {
    return (
      <EmptyState
        icon={<span className="text-lg">📊</span>}
        title={emptyTitle}
        description={emptyDescription}
      />
    );
  }

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 40 + 24)}>
      <BarChart
        data={chartData}
        layout="vertical"
        margin={{ top: 4, right: 16, left: 4, bottom: 4 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 11, fill: "#64748b" }} />
        <YAxis
          type="category"
          dataKey="shortLabel"
          width={120}
          tick={{ fontSize: 11, fill: "#64748b" }}
        />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const row = payload[0].payload as (typeof chartData)[0];
            return (
              <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg">
                <p className="mb-1 font-semibold text-slate-800">{row.label}</p>
                {row.sublabel ? <p className="mb-1 text-slate-500">{row.sublabel}</p> : null}
                <p className="text-blue-700">
                  {valueLabel}:{" "}
                  <span className="font-medium">{row.value.toLocaleString()}</span>
                </p>
              </div>
            );
          }}
        />
        <Bar dataKey="value" fill={color} radius={[0, 4, 4, 0]} maxBarSize={22} />
      </BarChart>
    </ResponsiveContainer>
  );
}
