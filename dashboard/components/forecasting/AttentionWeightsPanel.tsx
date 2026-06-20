"use client";

import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Eye } from "lucide-react";

import type { AttentionWeight } from "@/lib/types";

import { EmptyState } from "@/components/cold-start/ui";

interface AttentionWeightsPanelProps {
  weights: AttentionWeight[] | null | undefined;
  embedded?: boolean;
}

export function AttentionWeightsPanel({
  weights,
  embedded = false,
}: AttentionWeightsPanelProps) {
  const chartData = useMemo(() => {
    if (!weights?.length) return [];
    return [...weights]
      .sort((a, b) => a.week_offset - b.week_offset)
      .map((w) => ({
        label: w.week_offset === 0 ? "Current" : `−${w.week_offset}w`,
        week_offset: w.week_offset,
        weight: w.weight,
      }));
  }, [weights]);

  if (!weights?.length) {
    return (
      <EmptyState
        icon={<Eye className="h-6 w-6" />}
        title="No attention weights"
        description="Enable “Include TFT attention” when running a forecast to see which past weeks the TFT model focused on."
      />
    );
  }

  const wrapperClass = embedded ? "" : "rounded-xl border border-slate-200 bg-white p-5 shadow-sm";

  return (
    <div className={wrapperClass}>
      {!embedded && (
        <h2 className="mb-2 text-lg font-semibold text-slate-900">TFT Temporal Attention</h2>
      )}
      <p className="mb-4 text-sm text-slate-500">
        How much the TFT model attends to each past week when forming its forecast.
      </p>

      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#64748b" }} />
          <YAxis tick={{ fontSize: 11, fill: "#64748b" }} domain={[0, "auto"]} />
          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as (typeof chartData)[0];
              return (
                <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg">
                  <p className="mb-1 font-semibold text-slate-800">{label}</p>
                  <p className="text-emerald-700">
                    Attention: <span className="font-medium">{row.weight.toFixed(4)}</span>
                  </p>
                </div>
              );
            }}
          />
          <Area
            type="monotone"
            dataKey="weight"
            stroke="none"
            fill="#6ee7b7"
            fillOpacity={0.35}
          />
          <Line
            type="monotone"
            dataKey="weight"
            stroke="#059669"
            strokeWidth={2}
            dot={{ r: 3, fill: "#059669" }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
