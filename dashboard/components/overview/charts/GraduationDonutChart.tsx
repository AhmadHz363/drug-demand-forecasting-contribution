"use client";

import { useMemo } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

import { EmptyState } from "@/components/cold-start/ui";
import type { GraduationDistribution } from "@/lib/types";

const SEGMENTS = [
  { key: "cold_start_only" as const, label: "Cold-start only", color: "#fbbf24" },
  { key: "blended" as const, label: "Blended", color: "#60a5fa" },
  { key: "full_ensemble" as const, label: "Full ensemble", color: "#10b981" },
];

export function GraduationDonutChart({ distribution }: { distribution: GraduationDistribution }) {
  const chartData = useMemo(
    () =>
      SEGMENTS.map(({ key, label, color }) => ({
        name: label,
        value: distribution[key],
        color,
      })).filter((item) => item.value > 0),
    [distribution],
  );

  const total = distribution.cold_start_only + distribution.blended + distribution.full_ensemble;

  if (total === 0) {
    return (
      <EmptyState
        icon={<span className="text-lg">🎯</span>}
        title="No drugs registered"
        description="Ingest receipt data to see graduation stage distribution."
      />
    );
  }

  return (
    <div className="flex flex-col items-center gap-4 sm:flex-row sm:items-start">
      <div className="relative h-52 w-full sm:w-52">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              innerRadius={52}
              outerRadius={78}
              paddingAngle={2}
              stroke="none"
            >
              {chartData.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const row = payload[0].payload as (typeof chartData)[0];
                const pct = ((row.value / total) * 100).toFixed(1);
                return (
                  <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg">
                    <p className="font-semibold text-slate-800">{row.name}</p>
                    <p className="text-slate-600">
                      {row.value.toLocaleString()} drugs ({pct}%)
                    </p>
                  </div>
                );
              }}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <p className="text-2xl font-bold tabular-nums text-slate-900">{total.toLocaleString()}</p>
          <p className="text-[10px] font-medium uppercase tracking-wide text-slate-400">Drugs</p>
        </div>
      </div>
      <ul className="flex w-full flex-1 flex-col gap-2">
        {SEGMENTS.map(({ key, label, color }) => {
          const value = distribution[key];
          const pct = total > 0 ? ((value / total) * 100).toFixed(0) : "0";
          return (
            <li
              key={key}
              className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2 text-sm"
            >
              <span className="flex items-center gap-2 text-slate-700">
                <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
                {label}
              </span>
              <span className="tabular-nums text-slate-600">
                {value.toLocaleString()}{" "}
                <span className="text-slate-400">({pct}%)</span>
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
