"use client";

import { GitCompare } from "lucide-react";

import {
  BLEND_UNTIL,
  COLD_START_ONLY_BELOW,
  coldStartRemainingPercent,
  decideStageFromCount,
  mamlWeightPercent,
} from "@/lib/constants";
import type { GraduationStage } from "@/lib/types";

import { GraduationBadge } from "./GraduationBadge";
import { Panel, PanelBody, PanelHeader, Skeleton } from "./ui";

interface ObservationSimulatorProps {
  simulatedCount: number;
  onSimulatedCountChange: (count: number) => void;
  onRerunPredict: () => void;
  apiObservationCount?: number;
  apiStage?: GraduationStage;
  apiUncertaintyNote?: string;
  isLoading: boolean;
  hasPrediction?: boolean;
}

function explanationText(count: number): string {
  if (count < COLD_START_ONLY_BELOW) {
    return "No real data yet — forecast relies entirely on similar known drugs.";
  }
  if (count < BLEND_UNTIL) {
    const remaining = coldStartRemainingPercent(count);
    return `MAML is learning from ${count} observations. Cold Start influence at ${remaining}%.`;
  }
  return "Enough history accumulated — this drug graduates to the full ensemble.";
}

export function ObservationSimulator({
  simulatedCount,
  onSimulatedCountChange,
  onRerunPredict,
  apiObservationCount,
  apiStage,
  apiUncertaintyNote,
  isLoading,
  hasPrediction,
}: ObservationSimulatorProps) {
  const simulatedStage = decideStageFromCount(simulatedCount);
  const mamlPct = mamlWeightPercent(simulatedCount);
  const progressPct = Math.min(100, (simulatedCount / BLEND_UNTIL) * 100);

  const displayCount =
    apiObservationCount !== undefined ? apiObservationCount : simulatedCount;
  const displayStage = apiStage ?? simulatedStage;
  const showApiNote = apiUncertaintyNote && apiStage !== undefined;

  if (isLoading) {
    return (
      <Panel>
        <PanelBody className="space-y-3">
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-2 w-full" />
          <Skeleton className="h-20 w-full" />
        </PanelBody>
      </Panel>
    );
  }

  return (
    <div className="space-y-4">
      <Panel>
        <PanelHeader
          title="Graduation preview"
          description="Drag to simulate how many days of real demand data this drug has."
        />
        <PanelBody className="space-y-4">
          <div>
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="font-medium text-slate-700">Observations</span>
              <span className="rounded-full bg-slate-100 px-2.5 py-0.5 font-bold text-slate-800">
                {simulatedCount}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={20}
              value={simulatedCount}
              onChange={(e) => onSimulatedCountChange(Number(e.target.value))}
              className="w-full"
            />
            <div className="mt-1 flex justify-between text-[10px] text-slate-400">
              <span>0 · Cold start</span>
              <span>4 · Blended</span>
              <span>12+ · Ensemble</span>
            </div>
          </div>

          <div className="overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-2 rounded-full bg-gradient-to-r from-blue-500 via-amber-400 to-emerald-500 transition-all"
              style={{ width: `${progressPct}%` }}
            />
          </div>

          <div className="rounded-xl bg-slate-50 p-3 text-sm leading-relaxed text-slate-600">
            {explanationText(simulatedCount)}
            {simulatedStage === "blended" && (
              <p className="mt-1.5 font-semibold text-amber-700">
                MAML influence: {mamlPct.toFixed(0)}%
              </p>
            )}
          </div>

          {hasPrediction && (
            <button
              type="button"
              onClick={onRerunPredict}
              disabled={isLoading}
              className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
            >
              <GitCompare className="h-4 w-4" />
              Re-run with latest data
            </button>
          )}
        </PanelBody>
      </Panel>

      {(hasPrediction || showApiNote) && (
        <GraduationBadge
          stage={displayStage}
          observationCount={displayCount}
          uncertaintyNote={showApiNote ? apiUncertaintyNote : undefined}
          mamlWeightPercent={
            displayStage === "blended" ? mamlWeightPercent(displayCount) : undefined
          }
        />
      )}
    </div>
  );
}
