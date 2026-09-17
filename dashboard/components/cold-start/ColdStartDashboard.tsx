"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Activity, Pill, Upload } from "lucide-react";

import { CameoAccuracyHero } from "@/components/cold-start/CameoAccuracyHero";
import { CameoAccuracyReport } from "@/components/cold-start/CameoAccuracyReport";
import { CameoPerDrugAccuracyTable } from "@/components/cold-start/CameoPerDrugAccuracyTable";
import { DrugMetadataForm } from "@/components/cold-start/DrugMetadataForm";
import { ErrorPanel } from "@/components/cold-start/ErrorPanel";
import { ObservationSimulator } from "@/components/cold-start/ObservationSimulator";
import { ResultsTabs } from "@/components/cold-start/ResultsTabs";
import {
  TrainingControls,
  type TrainingCompletePayload,
} from "@/components/cold-start/TrainingControls";
import { StatChip } from "@/components/cold-start/ui";
import {
  ColdStartWorkflowStepper,
  scrollToSection,
  type ColdStartWorkflowStep,
} from "@/components/cold-start/ColdStartWorkflowStepper";
import { useColdStartPredict } from "@/hooks/useColdStartPredict";
import { fetchColdStartStatus } from "@/lib/api";
import type { CameoAccuracySummary, ColdStartPredictRequest } from "@/lib/types";

const STAGE_LABELS = {
  cold_start_only: "Cold start",
  blended: "Blended",
  full_ensemble: "Full ensemble",
} as const;

function applyStatusPayload(
  status: Awaited<ReturnType<typeof fetchColdStartStatus>>,
  setters: {
    setModelReady: (v: boolean) => void;
    setAccuracySummary: (v: CameoAccuracySummary | null) => void;
    setDrugsTrained: (v: number | undefined) => void;
    setTrainedAt: (v: string | null) => void;
  },
) {
  if (status.model_ready) setters.setModelReady(true);
  if (status.accuracy_summary) setters.setAccuracySummary(status.accuracy_summary);
  if (status.drugs_trained_on != null) setters.setDrugsTrained(status.drugs_trained_on);
  if (status.trained_at) setters.setTrainedAt(status.trained_at);
}

export function ColdStartDashboard() {
  const { data, error, isLoading, predict } = useColdStartPredict();
  const [activeStep, setActiveStep] = useState<ColdStartWorkflowStep>("train");
  const [modelReady, setModelReady] = useState(false);
  const [accuracySummary, setAccuracySummary] = useState<CameoAccuracySummary | null>(null);
  const [drugsTrained, setDrugsTrained] = useState<number | undefined>();
  const [trainedAt, setTrainedAt] = useState<string | null>(null);
  const [simulatedCount, setSimulatedCount] = useState(0);
  const [previousEmbedding, setPreviousEmbedding] = useState<number[] | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const lastRequestRef = useRef<ColdStartPredictRequest | null>(null);

  const hasValidationMetrics = Boolean(
    accuracySummary && accuracySummary.n_coldstart_test_drugs > 0,
  );

  const refreshStatus = useCallback(async () => {
    try {
      const status = await fetchColdStartStatus();
      applyStatusPayload(status, {
        setModelReady,
        setAccuracySummary,
        setDrugsTrained,
        setTrainedAt,
      });
    } catch {
      // Backend may be offline during dev.
    } finally {
      setStatusLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  const handleTrainComplete = useCallback(
    (payload: TrainingCompletePayload) => {
      if (payload.accuracySummary) setAccuracySummary(payload.accuracySummary);
      if (payload.drugsTrained != null) setDrugsTrained(payload.drugsTrained);
      setModelReady(true);
      setActiveStep("validate");
      void refreshStatus();
    },
    [refreshStatus],
  );

  const handlePredict = useCallback(
    async (request: ColdStartPredictRequest) => {
      lastRequestRef.current = request;
      if (data?.embedding) {
        setPreviousEmbedding(data.embedding);
      }
      await predict(request);
      setActiveStep("forecast");
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

  const stepTabs: { id: ColdStartWorkflowStep; label: string }[] = [
    { id: "train", label: "1 · Train" },
    { id: "validate", label: "2 · Validate" },
    { id: "forecast", label: "3 · Forecast" },
  ];

  return (
    <div className="min-h-full">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="flex items-start gap-3">
              <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white shadow-sm">
                <Pill className="h-5 w-5" />
              </span>
              <div>
                <p className="text-xs font-semibold uppercase tracking-widest text-slate-500">
                  CAMEO
                </p>
                <h1 className="mt-0.5 text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
                  Cold Start Forecasting
                </h1>
                <p className="mt-1 max-w-2xl text-sm leading-relaxed text-slate-500">
                  Metric-learning analog search, zero-inflated Bayesian tracking, and conformal
                  intervals for new drugs — validated with leave-drugs-out hold-out accuracy on
                  real matched-source attributes.
                </p>
              </div>
            </div>
            <Link
              href="/data-ingestion"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
            >
              <Upload className="h-4 w-4" />
              Data ingestion
            </Link>
          </div>

          <div className="mt-5">
            <ColdStartWorkflowStepper
              modelReady={modelReady}
              hasValidationMetrics={hasValidationMetrics}
              hasPrediction={Boolean(data)}
              activeStep={activeStep}
            />
          </div>

          <div className="mt-4 flex gap-1 rounded-xl border border-slate-200 bg-slate-100 p-1">
            {stepTabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveStep(tab.id)}
                className={`flex-1 rounded-lg px-4 py-2.5 text-sm font-semibold transition ${
                  activeStep === tab.id
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 px-4 py-6 sm:px-6">
        {activeStep === "train" && (
          <div className="space-y-6">
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
              <h2 className="text-lg font-semibold text-slate-900">Train the CAMEO library</h2>
              <p className="mt-1 text-sm text-slate-500">
                Trains the metric-learning embedder on matched-source drugs from the Excel library
                plus weekly demand shapes from enriched hospital data. Hold-out validation runs
                automatically after training.
              </p>
              <div className="mt-5">
                <TrainingControls
                  onModelReady={setModelReady}
                  onTrainComplete={handleTrainComplete}
                />
              </div>
            </section>

            {!statusLoading && hasValidationMetrics && (
              <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 px-4 py-3 text-sm text-emerald-900">
                Trained artifacts detected — open{" "}
                <button
                  type="button"
                  onClick={() => setActiveStep("validate")}
                  className="font-semibold underline"
                >
                  Validate
                </button>{" "}
                to review hold-out accuracy, or{" "}
                <button
                  type="button"
                  onClick={() => setActiveStep("forecast")}
                  className="font-semibold underline"
                >
                  Forecast
                </button>{" "}
                a new drug.
              </div>
            )}
          </div>
        )}

        {activeStep === "validate" && (
          <div className="space-y-6">
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
              <h2 className="text-lg font-semibold text-slate-900">Hold-out validation accuracy</h2>
              <p className="mt-1 text-sm text-slate-500">
                Leave-drugs-out cold-start simulation — WAPE-based accuracy (Accuracy = 1 − WAPE)
                aligned with the CAMEO real-data notebook evaluation.
              </p>
              <div className="mt-5">
                <CameoAccuracyHero summary={accuracySummary} drugsTrained={drugsTrained} />
              </div>
            </section>

            <CameoAccuracyReport summary={accuracySummary} trainedAt={trainedAt} />

            <CameoPerDrugAccuracyTable
              rows={accuracySummary?.per_drug ?? []}
              selectedDrugCode={data?.drug_code}
            />

            {!hasValidationMetrics && !statusLoading && (
              <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                Train CAMEO on the{" "}
                <button
                  type="button"
                  onClick={() => setActiveStep("train")}
                  className="font-semibold underline"
                >
                  Train
                </button>{" "}
                tab first to compute validation metrics.
              </div>
            )}
          </div>
        )}

        {activeStep === "forecast" && (
          <div className="space-y-6">
            {!modelReady && (
              <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                Train CAMEO first on the{" "}
                <button
                  type="button"
                  onClick={() => setActiveStep("train")}
                  className="font-semibold underline"
                >
                  Train
                </button>{" "}
                tab before generating forecasts.
              </div>
            )}

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
              <div className="space-y-6">
                <DrugMetadataForm
                  onSubmit={handlePredict}
                  isLoading={isLoading}
                  disabled={!modelReady}
                />
                <ResultsTabs
                  data={data}
                  previousEmbedding={previousEmbedding}
                  isLoading={isLoading}
                  onScrollToForm={() => scrollToSection("configure-section")}
                />
              </div>

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

                {!data && !isLoading && modelReady && (
                  <div className="rounded-2xl border border-blue-100 bg-blue-50/60 p-4">
                    <div className="flex items-start gap-2">
                      <Activity className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" />
                      <div>
                        <p className="text-sm font-semibold text-blue-900">Quick tip</p>
                        <p className="mt-1 text-xs leading-relaxed text-blue-800/80">
                          Click <strong>Generate forecast</strong> with the pre-filled example to
                          see CAMEO analog matching, embedding, and graduation results.
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </aside>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
