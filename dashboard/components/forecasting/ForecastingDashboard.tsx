"use client";

import { useCallback, useRef, useState } from "react";
import { Activity, LineChart } from "lucide-react";

import { ForecastRequestForm } from "@/components/forecasting/ForecastRequestForm";
import { HoldoutValidationPanel } from "@/components/forecasting/HoldoutValidationPanel";
import { ModelHealthMonitoringPanel } from "@/components/forecasting/ModelHealthMonitoringPanel";
import { ForecastingErrorPanel } from "@/components/forecasting/ForecastingErrorPanel";
import { ForecastingResultsTabs } from "@/components/forecasting/ForecastingResultsTabs";
import { ForecastingTrainingControls } from "@/components/forecasting/ForecastingTrainingControls";
import {
  ForecastingWorkflowStepper,
  scrollToSection,
  type ForecastingWorkflowStep,
} from "@/components/forecasting/ForecastingWorkflowStepper";
import { StatChip } from "@/components/cold-start/ui";
import { useForecastingPredict } from "@/hooks/useForecastingPredict";
import { FORECAST_MODEL_LABELS, type ForecastModelId } from "@/lib/constants";
import type { ForecastRequest } from "@/lib/types";

export function ForecastingDashboard() {
  const { data, error, isLoading, predict } = useForecastingPredict();
  const [modelsReady, setModelsReady] = useState(false);
  const lastRequestRef = useRef<ForecastRequest | null>(null);

  const activeStep: ForecastingWorkflowStep = data
    ? "results"
    : modelsReady
      ? "configure"
      : "setup";

  const handlePredict = useCallback(
    async (request: ForecastRequest) => {
      lastRequestRef.current = request;
      await predict(request);
      setTimeout(() => scrollToSection("forecasting-results-section"), 100);
    },
    [predict],
  );

  const p50Total = data?.forecast.reduce((s, d) => s + d.p50, 0);

  const dominantModel = data
    ? FORECAST_MODEL_LABELS[
        (Object.entries(data.model_weights) as [ForecastModelId, number][]).reduce(
          (best, [key, val]) => (val > best.val ? { key, val } : best),
          { key: "sarima" as ForecastModelId, val: 0 },
        ).key
      ]
    : null;

  return (
    <div className="min-h-full">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
          <div className="flex items-start gap-3">
            <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white shadow-sm">
              <LineChart className="h-5 w-5" />
            </span>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
                Ensemble Forecasting
              </h1>
              <p className="mt-1 max-w-2xl text-sm leading-relaxed text-slate-500">
                Train SARIMA, LightGBM, and TFT models — then generate calibrated quantile forecasts
                with conformal uncertainty bands and ensemble stacking.
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-6 px-4 py-6 sm:px-6">
        <ForecastingWorkflowStepper
          modelsReady={modelsReady}
          hasPrediction={Boolean(data)}
          activeStep={activeStep}
        />

        <ForecastingTrainingControls onModelsReady={setModelsReady} />

        <HoldoutValidationPanel />

        <ModelHealthMonitoringPanel />

        {error && !isLoading && <ForecastingErrorPanel error={error} />}

        {data && !isLoading && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatChip label="Drug" value={data.drug_code} accent="slate" />
            <StatChip
              label="Validation sMAPE"
              value={
                data.smape_last_validation != null
                  ? `${data.smape_last_validation.toFixed(1)}%`
                  : "—"
              }
              accent="amber"
            />
            <StatChip
              label="Top model"
              value={dominantModel ?? "—"}
              accent="blue"
            />
            <StatChip
              label={`${data.forecast.length}-day forecast`}
              value={p50Total !== undefined ? `${p50Total.toFixed(0)} units` : "—"}
              accent="green"
            />
          </div>
        )}

        <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
          <ForecastRequestForm onSubmit={handlePredict} isLoading={isLoading} />

          <aside className="space-y-4 xl:sticky xl:top-6 xl:self-start">
            {!data && !isLoading && (
              <div className="rounded-2xl border border-blue-100 bg-blue-50/60 p-4">
                <div className="flex items-start gap-2">
                  <Activity className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" />
                  <div>
                    <p className="text-sm font-semibold text-blue-900">Quick tip</p>
                    <p className="mt-1 text-xs leading-relaxed text-blue-800/80">
                      Train models first, then select <strong>E2E-AMOX</strong> and click{" "}
                      <strong>Generate forecast</strong>. Enable SHAP or attention for deeper
                      model insights.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {data && data.center_syn_id && (
              <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                  Center filter
                </p>
                <p className="mt-1 text-sm font-semibold text-slate-900">{data.center_syn_id}</p>
              </div>
            )}
          </aside>
        </div>

        <ForecastingResultsTabs
          data={data}
          isLoading={isLoading}
          onScrollToForm={() => scrollToSection("forecast-configure-section")}
        />
      </main>
    </div>
  );
}
