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
import { BarChart3 } from "lucide-react";

import type { ColdStartPredictResponse } from "@/lib/types";

import { EmptyState } from "./ui";

interface NeighboursPanelProps {
  data: Pick<
    ColdStartPredictResponse,
    "nearest_neighbours" | "similarity_scores"
  > | null;
  embedded?: boolean;
}

function similarityColor(score: number): string {
  const intensity = Math.round(score * 180 + 75);
  return `rgb(37, 99, ${Math.min(235, intensity)})`;
}

export function NeighboursPanel({ data, embedded = false }: NeighboursPanelProps) {
  const { chartData, totalSimilarity } = useMemo(() => {
    if (!data) return { chartData: [], totalSimilarity: 0 };
    const total = data.similarity_scores.reduce((a, b) => a + b, 0);
    const rows = data.nearest_neighbours.map((code, i) => ({
      drug_code: code,
      similarity: data.similarity_scores[i] ?? 0,
      weight: total > 0 ? ((data.similarity_scores[i] ?? 0) / total) * 100 : 0,
    }));
    return { chartData: rows.sort((a, b) => a.similarity - b.similarity), totalSimilarity: total };
  }, [data]);

  if (!data) {
    return (
      <EmptyState
        icon={<BarChart3 className="h-6 w-6" />}
        title="No neighbour data"
        description="Similar catalog drugs and their influence on the forecast will appear here."
      />
    );
  }

  return (
    <div className={embedded ? "" : "rounded-xl border border-slate-200 bg-white p-5 shadow-sm"}>
      {!embedded && (
        <h2 className="mb-4 text-lg font-semibold text-slate-900">Similar catalog drugs</h2>
      )}
      <p className="mb-4 text-sm text-slate-500">
        The {data.nearest_neighbours.length} most similar known drugs, ranked by cosine similarity.
      </p>
      <ResponsiveContainer width="100%" height={Math.max(160, chartData.length * 40)}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 4, right: 16 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
          <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => v.toFixed(2)} tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="drug_code" width={96} tick={{ fontSize: 11 }} />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as (typeof chartData)[0];
              return (
                <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg">
                  <p className="font-semibold text-slate-800">{row.drug_code}</p>
                  <p className="text-slate-600">Similarity: {row.similarity.toFixed(3)}</p>
                  <p className="text-blue-700">Forecast weight: {row.weight.toFixed(1)}%</p>
                </div>
              );
            }}
          />
          <Bar dataKey="similarity" radius={[0, 6, 6, 0]} barSize={20}>
            {chartData.map((entry) => (
              <Cell key={entry.drug_code} fill={similarityColor(entry.similarity)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="mt-5 overflow-x-auto rounded-xl border border-slate-200">
        <table className="w-full min-w-[320px] text-left text-sm">
          <thead className="bg-slate-50">
            <tr className="text-xs uppercase tracking-wide text-slate-500">
              <th className="px-4 py-2.5 font-medium">Drug code</th>
              <th className="px-4 py-2.5 font-medium">Similarity</th>
              <th className="px-4 py-2.5 font-medium">Weight</th>
              <th className="px-4 py-2.5 font-medium">Avg daily demand</th>
            </tr>
          </thead>
          <tbody>
            {data.nearest_neighbours.map((code, i) => {
              const sim = data.similarity_scores[i] ?? 0;
              const weight = totalSimilarity > 0 ? (sim / totalSimilarity) * 100 : 0;
              return (
                <tr key={code} className="border-t border-slate-100">
                  <td className="px-4 py-2.5 font-mono text-xs text-slate-800">{code}</td>
                  <td className="px-4 py-2.5 text-slate-600">{sim.toFixed(3)}</td>
                  <td className="px-4 py-2.5 font-medium text-blue-700">{weight.toFixed(1)}%</td>
                  <td
                    className="px-4 py-2.5 text-slate-400"
                    title="Fetch from /cold-start/neighbours/{drug_code} (not yet implemented)"
                  >
                    —
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function NeighboursPanelSkeleton() {
  return (
    <div className="animate-pulse space-y-3">
      {[1, 2, 3, 4, 5].map((i) => (
        <div key={i} className="h-6 rounded-lg bg-slate-200" style={{ width: `${100 - i * 12}%` }} />
      ))}
    </div>
  );
}
