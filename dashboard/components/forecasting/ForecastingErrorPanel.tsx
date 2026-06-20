"use client";

import { AlertCircle, ArrowUp } from "lucide-react";

import { Panel, PanelBody } from "@/components/cold-start/ui";

import { scrollToForecastingTraining } from "./ForecastingTrainingControls";

interface ForecastingErrorPanelProps {
  error: string;
}

export function ForecastingErrorPanel({ error }: ForecastingErrorPanelProps) {
  const needsTraining =
    error.toLowerCase().includes("no trained models") ||
    error.toLowerCase().includes("call post /forecasting/train");

  return (
    <Panel className="border-red-200 bg-red-50/50">
      <PanelBody>
        <div className="flex items-start gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-red-100">
            <AlertCircle className="h-5 w-5 text-red-600" />
          </span>
          <div className="flex-1">
            <h3 className="font-semibold text-red-900">Something went wrong</h3>
            <p className="mt-1 text-sm text-red-800">{error}</p>
            {needsTraining && (
              <div className="mt-4 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={scrollToForecastingTraining}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700"
                >
                  <ArrowUp className="h-4 w-4" />
                  Train models first
                </button>
              </div>
            )}
          </div>
        </div>
      </PanelBody>
    </Panel>
  );
}
