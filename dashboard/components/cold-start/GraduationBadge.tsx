"use client";

import { Snowflake, TrendingUp, Zap } from "lucide-react";

import { BLEND_UNTIL } from "@/lib/constants";
import type { GraduationStage } from "@/lib/types";

const STAGE_CONFIG: Record<
  GraduationStage,
  { label: string; description: string; className: string; icon: React.ReactNode }
> = {
  cold_start_only: {
    label: "Cold Start Only",
    description: "Using KNN similarity — no real demand history yet",
    className: "bg-blue-50 text-blue-800 border-blue-200",
    icon: <Snowflake className="h-4 w-4" />,
  },
  blended: {
    label: "Blended Mode",
    description: "Mixing KNN baseline with MAML adaptation",
    className: "bg-amber-50 text-amber-800 border-amber-200",
    icon: <Zap className="h-4 w-4" />,
  },
  full_ensemble: {
    label: "Full Ensemble",
    description: "Ready for the production forecasting engine",
    className: "bg-emerald-50 text-emerald-800 border-emerald-200",
    icon: <TrendingUp className="h-4 w-4" />,
  },
};

interface GraduationBadgeProps {
  stage: GraduationStage;
  observationCount: number;
  uncertaintyNote?: string;
  mamlWeightPercent?: number;
  embedded?: boolean;
}

export function GraduationBadge({
  stage,
  observationCount,
  uncertaintyNote,
  mamlWeightPercent,
  embedded = false,
}: GraduationBadgeProps) {
  const config = STAGE_CONFIG[stage];
  const progressPct = Math.min(100, (observationCount / BLEND_UNTIL) * 100);

  const content = (
    <>
      <div className="flex items-center gap-2">
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-sm font-semibold ${config.className}`}
        >
          {config.icon}
          {config.label}
        </span>
      </div>
      <p className="mt-2 text-sm text-slate-600">{config.description}</p>

      <div className="mt-4">
        <div className="mb-1.5 flex justify-between text-xs text-slate-500">
          <span>{observationCount} observation{observationCount === 1 ? "" : "s"}</span>
          <span>Graduates at {BLEND_UNTIL}</span>
        </div>
        <div className="overflow-hidden rounded-full bg-slate-100">
          <div
            className="h-2 rounded-full bg-emerald-500 transition-all"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      {stage === "blended" && mamlWeightPercent !== undefined && (
        <p className="mt-3 text-sm font-medium text-amber-700">
          MAML weight in forecast: {mamlWeightPercent.toFixed(0)}%
        </p>
      )}

      {uncertaintyNote && (
        <p className="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-xs leading-relaxed text-slate-500">
          {uncertaintyNote}
        </p>
      )}
    </>
  );

  if (embedded) {
    return <div>{content}</div>;
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="mb-3 text-base font-semibold text-slate-900">Graduation stage</h2>
      {content}
    </div>
  );
}
