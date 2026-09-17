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

import { EmptyState } from "@/components/cold-start/ui";
import type { TimeSeriesPoint } from "@/lib/types";

function formatPeriod(period: string): string {
  const [year, month] = period.split("-");
  const d = new Date(Number(year), Number(month) - 1, 1);
  return d.toLocaleDateString(undefined, { month: "short", year: "2-digit" });
}

export function ReceiptTrendChart({ data }: { data: TimeSeriesPoint[] }) {
  const chartData = useMemo(
    () =>
      data.map((point) => ({
        ...point,
        label: formatPeriod(point.period),
      })),
    [data],
  );

  if (!chartData.length) {
    return (
      <EmptyState
        icon={<span className="text-lg">📈</span>}
        title="No receipt history"
        description="Upload receipt spreadsheets to see volume trends over time."
      />
    );
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="receiptQtyGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#2563eb" stopOpacity={0.25} />
            <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#64748b" }} minTickGap={20} />
        <YAxis
          yAxisId="left"
          tick={{ fontSize: 11, fill: "#64748b" }}
          tickFormatter={(v) => Number(v).toLocaleString()}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tick={{ fontSize: 11, fill: "#64748b" }}
          tickFormatter={(v) => Number(v).toLocaleString()}
        />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const row = payload[0].payload as TimeSeriesPoint & { label: string };
            return (
              <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg">
                <p className="mb-1.5 font-semibold text-slate-800">{row.period}</p>
                <p className="text-blue-700">
                  Receipt rows:{" "}
                  <span className="font-medium">{row.receipt_count.toLocaleString()}</span>
                </p>
                <p className="text-emerald-700">
                  Total quantity:{" "}
                  <span className="font-medium">
                    {Math.round(row.total_quantity).toLocaleString()}
                  </span>
                </p>
              </div>
            );
          }}
        />
        <Legend
          wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
          formatter={(value) => <span className="text-slate-600">{value}</span>}
        />
        <Area
          yAxisId="right"
          type="monotone"
          dataKey="total_quantity"
          name="Total quantity"
          stroke="#10b981"
          fill="url(#receiptQtyGradient)"
          strokeWidth={2}
        />
        <Line
          yAxisId="left"
          type="monotone"
          dataKey="receipt_count"
          name="Receipt rows"
          stroke="#2563eb"
          strokeWidth={2}
          dot={{ r: 3, fill: "#2563eb" }}
          activeDot={{ r: 5 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
