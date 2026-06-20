"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { GitBranch } from "lucide-react";

import type { ShapFeature } from "@/lib/types";

import { EmptyState } from "@/components/cold-start/ui";

interface ShapFeaturesPanelProps {
  features: ShapFeature[] | null | undefined;
  embedded?: boolean;
}

export function ShapFeaturesPanel({ features, embedded = false }: ShapFeaturesPanelProps) {
  const chartData = useMemo(() => {
    if (!features?.length) return [];
    return [...features]
      .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
      .slice(0, 12)
      .map((f) => ({
        name: f.feature_name.replace(/_/g, " "),
        shap: f.shap_value,
        value: f.feature_value,
        fill: f.shap_value >= 0 ? "#2563eb" : "#94a3b8",
      }));
  }, [features]);

  if (!features?.length) {
    return (
      <EmptyState
        icon={<GitBranch className="h-6 w-6" />}
        title="No SHAP features"
        description="Enable “Include SHAP features” when running a forecast to see LightGBM feature attributions."
      />
    );
  }

  const wrapperClass = embedded ? "" : "rounded-xl border border-slate-200 bg-white p-5 shadow-sm";

  return (
    <div className={wrapperClass}>
      {!embedded && (
        <h2 className="mb-2 text-lg font-semibold text-slate-900">SHAP Feature Attribution</h2>
      )}
      <p className="mb-4 text-sm text-slate-500">
        Top features driving the LightGBM forecast. Blue bars push demand up; grey bars push it down.
      </p>

      <ResponsiveContainer width="100%" height={Math.max(280, chartData.length * 36)}>
        <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 11, fill: "#64748b" }} />
          <YAxis
            type="category"
            dataKey="name"
            width={140}
            tick={{ fontSize: 11, fill: "#64748b" }}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as (typeof chartData)[0];
              return (
                <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg">
                  <p className="mb-1 font-semibold capitalize text-slate-800">{row.name}</p>
                  <p className="text-slate-600">
                    SHAP: <span className="font-medium">{row.shap.toFixed(4)}</span>
                  </p>
                  <p className="text-slate-600">
                    Value: <span className="font-medium">{row.value.toFixed(4)}</span>
                  </p>
                </div>
              );
            }}
          />
          <Bar dataKey="shap" radius={[0, 4, 4, 0]}>
            {chartData.map((entry) => (
              <Cell key={entry.name} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
