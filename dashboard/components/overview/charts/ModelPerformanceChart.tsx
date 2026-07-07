"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { EmptyState } from "@/components/cold-start/ui";
import type { ModelPerformanceSummary } from "@/lib/types";

export function ModelPerformanceChart({ data }: { data: ModelPerformanceSummary[] }) {
  const chartData = useMemo(
    () =>
      data.map((row) => ({
        model: row.model_name.toUpperCase(),
        smape: Number((row.avg_smape * 100).toFixed(2)),
        coverage: Number((row.avg_coverage_90 * 100).toFixed(1)),
        records: row.record_count,
      })),
    [data],
  );

  if (!chartData.length) {
    return (
      <EmptyState
        icon={<span className="text-lg">📉</span>}
        title="No model metrics yet"
        description="Train forecasting models to compare sMAPE and coverage across SARIMA, LGBM, and TFT."
      />
    );
  }

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="model" tick={{ fontSize: 11, fill: "#64748b" }} />
        <YAxis
          tick={{ fontSize: 11, fill: "#64748b" }}
          tickFormatter={(v) => `${v}%`}
          domain={[0, "auto"]}
        />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const row = payload[0].payload as (typeof chartData)[0];
            return (
              <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg">
                <p className="mb-1.5 font-semibold text-slate-800">{row.model}</p>
                <p className="text-blue-700">
                  Avg sMAPE: <span className="font-medium">{row.smape.toFixed(2)}%</span>
                </p>
                <p className="text-emerald-700">
                  Avg coverage 90: <span className="font-medium">{row.coverage.toFixed(1)}%</span>
                </p>
                <p className="text-slate-500">
                  Records: <span className="font-medium">{row.records.toLocaleString()}</span>
                </p>
              </div>
            );
          }}
        />
        <Legend
          wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
          formatter={(value) => <span className="text-slate-600">{value}</span>}
        />
        <Bar dataKey="smape" name="Avg sMAPE (%)" fill="#2563eb" radius={[4, 4, 0, 0]} maxBarSize={48} />
        <Bar
          dataKey="coverage"
          name="Avg coverage 90 (%)"
          fill="#10b981"
          radius={[4, 4, 0, 0]}
          maxBarSize={48}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
