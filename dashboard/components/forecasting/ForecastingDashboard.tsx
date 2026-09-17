"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { LineChart, Upload } from "lucide-react";

import { ForecastRequestForm } from "@/components/forecasting/ForecastRequestForm";
import { ForecastingErrorPanel } from "@/components/forecasting/ForecastingErrorPanel";
import {
  ForecastingWorkflowStepper,
  type ForecastingWorkflowStep,
} from "@/components/forecasting/ForecastingWorkflowStepper";
import { ForecastingResultsTabs } from "@/components/forecasting/ForecastingResultsTabs";
import {
  ForecastingTrainingControls,
  type TrainingCompletePayload,
} from "@/components/forecasting/ForecastingTrainingControls";
import { PerDrugAccuracyTable } from "@/components/forecasting/PerDrugAccuracyTable";
import { ShieldXRAccuracyReport } from "@/components/forecasting/ShieldXRAccuracyReport";
import { ShieldXRPipelineStrip } from "@/components/forecasting/ShieldXRPipelineStrip";
import { WeeklyAccuracyHero } from "@/components/forecasting/WeeklyAccuracyHero";
import { useForecastingPredict } from "@/hooks/useForecastingPredict";
import { fetchShieldXRStatus } from "@/lib/api";
import type { ForecastRequest, ShieldXRAccuracySummary } from "@/lib/types";

function applyStatusPayload(
  status: Awaited<ReturnType<typeof fetchShieldXRStatus>>,
  setters: {
    setModelsReady: (v: boolean) => void;
    setAccuracySummary: (v: ShieldXRAccuracySummary | null) => void;
    setTrainingRunId: (v: string | null) => void;
    setDrugsTrained: (v: number | undefined) => void;
    setTrainEnd: (v: string | null) => void;
    setValidEnd: (v: string | null) => void;
  },
) {
  if (status.models_ready) setters.setModelsReady(true);
  if (status.accuracy_summary) setters.setAccuracySummary(status.accuracy_summary);
  if (status.training_run_id) setters.setTrainingRunId(status.training_run_id);
  if (status.n_skus != null) setters.setDrugsTrained(status.n_skus);
  if (status.train_end) setters.setTrainEnd(status.train_end);
  if (status.valid_end) setters.setValidEnd(status.valid_end);
}

export function ForecastingDashboard() {
  const { data, error, isLoading, predict } = useForecastingPredict();
  const [activeStep, setActiveStep] = useState<ForecastingWorkflowStep>("train");
  const [modelsReady, setModelsReady] = useState(false);
  const [accuracySummary, setAccuracySummary] = useState<ShieldXRAccuracySummary | null>(null);
  const [trainingRunId, setTrainingRunId] = useState<string | null>(null);
  const [drugsTrained, setDrugsTrained] = useState<number | undefined>();
  const [trainEnd, setTrainEnd] = useState<string | null>(null);
  const [validEnd, setValidEnd] = useState<string | null>(null);
  const [prefillDrug, setPrefillDrug] = useState<string | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const lastRequestRef = useRef<ForecastRequest | null>(null);

  const hasValidationMetrics = Boolean(
    accuracySummary?.hospital_weekly_accuracy_pct != null ||
      accuracySummary?.reconciled_weekly_per_drug_mean_accuracy_pct != null,
  );

  const refreshStatus = useCallback(async () => {
    try {
      const status = await fetchShieldXRStatus();
      applyStatusPayload(status, {
        setModelsReady,
        setAccuracySummary,
        setTrainingRunId,
        setDrugsTrained,
        setTrainEnd,
        setValidEnd,
      });
    } catch {
      // Backend may be offline during dev — training flow still works.
    } finally {
      setStatusLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  const handleTrainComplete = useCallback((payload: TrainingCompletePayload) => {
    if (payload.accuracySummary) setAccuracySummary(payload.accuracySummary);
    if (payload.trainingRunId) setTrainingRunId(payload.trainingRunId);
    if (payload.drugsTrained != null) setDrugsTrained(payload.drugsTrained);
    setModelsReady(true);
    setActiveStep("validate");
    void refreshStatus();
  }, [refreshStatus]);

  const handlePredict = useCallback(
    async (request: ForecastRequest) => {
      lastRequestRef.current = request;
      await predict(request);
      setActiveStep("forecast");
      setTimeout(() => {
        document.getElementById("forecasting-results-section")?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }, 100);
    },
    [predict],
  );

  const handleSelectDrugFromTable = useCallback((code: string) => {
    setPrefillDrug(code);
    setActiveStep("forecast");
    setTimeout(() => {
      document.getElementById("forecast-configure-section")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 50);
  }, []);

  const stepTabs: { id: ForecastingWorkflowStep; label: string }[] = [
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
                <LineChart className="h-5 w-5" />
              </span>
              <div>
                <p className="text-xs font-semibold uppercase tracking-widest text-slate-500">
                  SHIELD-XR
                </p>
                <h1 className="mt-0.5 text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
                  Hospital Demand Forecasting
                </h1>
                <p className="mt-1 max-w-2xl text-sm leading-relaxed text-slate-500">
                  Hybrid architecture with AnomalyGuard, class-conditional hurdle + EVT ensemble,
                  weekly reconciliation, and hybrid ABC reporting — targeting ~89% hospital-week
                  and ~86% operational accuracy.
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
            <ForecastingWorkflowStepper
              modelsReady={modelsReady}
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
        <ShieldXRPipelineStrip modelsReady={modelsReady} />

        {activeStep === "train" && (
          <div className="space-y-6">
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
              <h2 className="text-lg font-semibold text-slate-900">Train the SHIELD-XR stack</h2>
              <p className="mt-1 text-sm text-slate-500">
                Runs the full five-stage pipeline on enriched inpatient demand. All thresholds,
                ensemble weights, and bias corrections are fit on the training window only — no
                leakage into validation or test.
              </p>
              <div className="mt-5 max-w-2xl">
                <ForecastingTrainingControls
                  onModelsReady={setModelsReady}
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
                a specific drug.
              </div>
            )}
          </div>
        )}

        {activeStep === "validate" && (
          <div className="space-y-6">
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
              <h2 className="text-lg font-semibold text-slate-900">Hold-out validation accuracy</h2>
              <p className="mt-1 text-sm text-slate-500">
                WAPE-based accuracy (Accuracy = 1 − WAPE) on the chronological test split — aligned
                with the SHIELD-XR thesis evaluation framework.
              </p>
              <div className="mt-5">
                <WeeklyAccuracyHero
                  summary={accuracySummary}
                  drugsTrained={drugsTrained}
                  trainingRunId={trainingRunId}
                />
              </div>
            </section>

            <ShieldXRAccuracyReport
              summary={accuracySummary}
              trainEnd={trainEnd}
              validEnd={validEnd}
            />

            <PerDrugAccuracyTable
              trainingRunId={trainingRunId}
              selectedDrugCode={data?.drug_code ?? lastRequestRef.current?.drug_code ?? prefillDrug}
              onSelectDrug={handleSelectDrugFromTable}
            />

            {!hasValidationMetrics && !statusLoading && (
              <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                Train SHIELD-XR on the{" "}
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
            {!modelsReady && (
              <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                Train SHIELD-XR first on the{" "}
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

            <div className="grid gap-6 xl:grid-cols-[380px_1fr]">
              <ForecastRequestForm
                key={prefillDrug ?? "forecast-form"}
                onSubmit={handlePredict}
                isLoading={isLoading}
                disabled={!modelsReady}
                initialDrugCode={prefillDrug}
              />
              <div className="space-y-4">
                {error && !isLoading && <ForecastingErrorPanel error={error} />}
                <ForecastingResultsTabs
                  data={data}
                  isLoading={isLoading}
                  onScrollToForm={() =>
                    document
                      .getElementById("forecast-configure-section")
                      ?.scrollIntoView({ behavior: "smooth" })
                  }
                />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
