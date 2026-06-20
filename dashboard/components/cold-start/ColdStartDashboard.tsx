"use client";

import { useCallback, useRef, useState } from "react";
import { Activity, Pill } from "lucide-react";

import { DrugMetadataForm } from "@/components/cold-start/DrugMetadataForm";
import { ErrorPanel } from "@/components/cold-start/ErrorPanel";
import { ObservationSimulator } from "@/components/cold-start/ObservationSimulator";
import { ResultsTabs } from "@/components/cold-start/ResultsTabs";
import { TrainingControls } from "@/components/cold-start/TrainingControls";
import { StatChip } from "@/components/cold-start/ui";
import {
  WorkflowStepper,
  scrollToSection,
  type WorkflowStep,
} from "@/components/cold-start/WorkflowStepper";
import { useColdStartPredict } from "@/hooks/useColdStartPredict";
import type { ColdStartPredictRequest } from "@/lib/types";

const STAGE_LABELS = {
  cold_start_only: "Cold start",
  blended: "Blended",
  full_ensemble: "Full ensemble",
} as const;

export function ColdStartDashboard() {
  const { data, error, isLoading, predict } = useColdStartPredict();
  const [simulatedCount, setSimulatedCount] = useState(0);
  const [previousEmbedding, setPreviousEmbedding] = useState<number[] | null>(null);
  const [embedderReady, setEmbedderReady] = useState(false);
  const lastRequestRef = useRef<ColdStartPredictRequest | null>(null);

  const activeStep: WorkflowStep = data
    ? "results"
    : embedderReady
      ? "configure"
      : "setup";

  const handlePredict = useCallback(
    async (request: ColdStartPredictRequest) => {
      lastRequestRef.current = request;
      if (data?.embedding) {
        setPreviousEmbedding(data.embedding);
      }
      await predict(request);
      setTimeout(() => scrollToSection("results-section"), 100);
    },
    [data?.embedding, predict],
  );

  const handleRerun = useCallback(() => {
    if (lastRequestRef.current) {
      void handlePredict(lastRequestRef.current);
    }
  }, [handlePredict]);

  const p50Total = data?.forecast.reduce((s, d) => s + d.p50, 0);

  return (
    <div className="min-h-full">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
          <div className="flex items-start gap-3">
            <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white shadow-sm">
              <Pill className="h-5 w-5" />
            </span>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
                Cold Start Forecasting
              </h1>
              <p className="mt-1 max-w-2xl text-sm leading-relaxed text-slate-500">
                Test demand predictions for new drugs — from metadata embedding through KNN
                matching to MAML adaptation and graduation.
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-6 px-4 py-6 sm:px-6">
        <WorkflowStepper
          embedderReady={embedderReady}
          hasPrediction={Boolean(data)}
          activeStep={activeStep}
        />

        <TrainingControls onEmbedderReady={setEmbedderReady} />

        {error && !isLoading && <ErrorPanel error={error} />}

        {data && !isLoading && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatChip label="Drug" value={data.drug_code} accent="slate" />
            <StatChip label="Stage" value={STAGE_LABELS[data.stage_used]} accent="blue" />
            <StatChip
              label="Similar drugs"
              value={data.nearest_neighbours.length}
              accent="amber"
            />
            <StatChip
              label={`${data.forecast.length}-day forecast`}
              value={p50Total !== undefined ? `${p50Total.toFixed(0)} units` : "—"}
              accent="green"
            />
          </div>
        )}

        <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
          <DrugMetadataForm onSubmit={handlePredict} isLoading={isLoading} />

          <aside className="space-y-4 xl:sticky xl:top-6 xl:self-start">
            <ObservationSimulator
              simulatedCount={simulatedCount}
              onSimulatedCountChange={setSimulatedCount}
              onRerunPredict={handleRerun}
              apiObservationCount={data?.observation_count}
              apiStage={data?.stage_used}
              apiUncertaintyNote={data?.uncertainty_note}
              isLoading={isLoading}
              hasPrediction={Boolean(data)}
            />

            {!data && !isLoading && (
              <div className="rounded-2xl border border-blue-100 bg-blue-50/60 p-4">
                <div className="flex items-start gap-2">
                  <Activity className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" />
                  <div>
                    <p className="text-sm font-semibold text-blue-900">Quick tip</p>
                    <p className="mt-1 text-xs leading-relaxed text-blue-800/80">
                      Train the embedder first, then click <strong>Generate forecast</strong> with
                      the pre-filled example. Results open in tabs below.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </aside>
        </div>

        <ResultsTabs
          data={data}
          previousEmbedding={previousEmbedding}
          isLoading={isLoading}
          onScrollToForm={() => scrollToSection("configure-section")}
        />
      </main>
    </div>
  );
}
