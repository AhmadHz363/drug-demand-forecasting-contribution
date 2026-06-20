"use client";

import { useEffect, useMemo } from "react";
import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
} from "recharts";
import { Radar as RadarIcon } from "lucide-react";

import { EmptyState, Skeleton } from "./ui";

const BUCKET_LABELS = [
  "Therapeutic",
  "ATC & Form",
  "Criticality",
  "Cost & Access",
  "Storage",
  "Route",
  "Temporal",
  "Economic",
] as const;

function groupEmbedding(embedding: number[]) {
  return BUCKET_LABELS.map((subject, i) => {
    const start = i * 4;
    const slice = embedding.slice(start, start + 4);
    const avg = slice.length > 0 ? slice.reduce((a, b) => a + b, 0) / slice.length : 0;
    return { subject, value: Math.max(0, Math.min(1, avg)) };
  });
}

interface EmbeddingRadarChartProps {
  embedding: number[] | null;
  previousEmbedding?: number[] | null;
  embedded?: boolean;
}

export function EmbeddingRadarChart({
  embedding,
  previousEmbedding,
  embedded = false,
}: EmbeddingRadarChartProps) {
  useEffect(() => {
    if (embedding) {
      console.log("Raw 32-dim embedding:", embedding);
    }
  }, [embedding]);

  const data = useMemo(() => {
    if (!embedding) return [];
    const current = groupEmbedding(embedding);
    if (!previousEmbedding) {
      return current.map((row) => ({ ...row, current: row.value, previous: 0 }));
    }
    const previous = groupEmbedding(previousEmbedding);
    return current.map((row, i) => ({
      subject: row.subject,
      value: row.value,
      current: row.value,
      previous: previous[i]?.value ?? 0,
    }));
  }, [embedding, previousEmbedding]);

  if (!embedding) {
    return (
      <EmptyState
        icon={<RadarIcon className="h-6 w-6" />}
        title="No embedding data"
        description="The 32-dimensional drug fingerprint will appear here after prediction."
      />
    );
  }

  const hasComparison = Boolean(previousEmbedding);

  return (
    <div className={embedded ? "" : "rounded-xl border border-slate-200 bg-white p-5 shadow-sm"}>
      {!embedded && (
        <h2 className="mb-4 text-lg font-semibold text-slate-900">Drug embedding</h2>
      )}
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="78%">
          <PolarGrid stroke="#e2e8f0" />
          <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: "#64748b" }} />
          <PolarRadiusAxis domain={[0, 1]} tick={{ fontSize: 9 }} axisLine={false} />
          {hasComparison ? (
            <>
              <Radar name="Previous" dataKey="previous" stroke="#94a3b8" fill="#94a3b8" fillOpacity={0.15} />
              <Radar name="Current" dataKey="current" stroke="#2563eb" fill="#2563eb" fillOpacity={0.3} />
            </>
          ) : (
            <Radar name="Embedding" dataKey="current" stroke="#2563eb" fill="#2563eb" fillOpacity={0.3} />
          )}
        </RadarChart>
      </ResponsiveContainer>
      <p className="mt-3 text-xs leading-relaxed text-slate-500">
        32 dimensions grouped into 8 buckets for readability. Open the browser console for the raw
        vector.
        {hasComparison && " Shaded overlay shows your previous prediction for comparison."}
      </p>
    </div>
  );
}

export function EmbeddingRadarChartSkeleton() {
  return (
    <div className="flex flex-col items-center">
      <Skeleton className="h-64 w-64 rounded-full" />
    </div>
  );
}
