"use client";

import { useCallback, useState } from "react";
import { CheckCircle2, Loader2, Shield, Sparkles, XCircle } from "lucide-react";
import Link from "next/link";

import { searchEnrichedDrugCodes, trainForecasting } from "@/lib/api";
import { formatAccuracySummaryLine } from "@/lib/forecastMetrics";
import type { ShieldXRAccuracySummary } from "@/lib/types";

import { DrugMultiSelectSearch } from "@/components/forecasting/DrugSearchCombobox";

type TrainStatus = "idle" | "loading" | "success" | "error";

interface TrainState {
  status: TrainStatus;
  message: string;
  summary?: string;
}

const initialState: TrainState = { status: "idle", message: "" };

export interface TrainingCompletePayload {
  accuracySummary?: ShieldXRAccuracySummary;
  trainingRunId?: string;
  drugsTrained?: number;
}

interface ForecastingTrainingControlsProps {
  onModelsReady?: (ready: boolean) => void;
  onTrainComplete?: (payload: TrainingCompletePayload) => void;
}

export function ForecastingTrainingControls({
  onModelsReady,
  onTrainComplete,
}: ForecastingTrainingControlsProps) {
  const [trainState, setTrainState] = useState<TrainState>(initialState);
  const [selectedDrugCodes, setSelectedDrugCodes] = useState<string[]>([]);
  const [forceRetrain, setForceRetrain] = useState(false);

  const searchEnrichedDrugs = useCallback(async (query: string, page: number) => {
    const result = await searchEnrichedDrugCodes(query, page);
    return {
      items: result.items,
      total: result.total,
      page: result.page,
      page_size: result.page_size,
      total_pages: result.total_pages,
    };
  }, []);

  const runTrain = useCallback(async () => {
    setTrainState({ status: "loading", message: "" });
    try {
      const result = await trainForecasting({
        drug_codes: selectedDrugCodes.length > 0 ? selectedDrugCodes : undefined,
        models: ["shield_xr"],
        force_retrain: forceRetrain,
      });

      const summary = result.accuracy_summary;
      const skipped = result.skipped_drugs?.length ?? 0;
      const flagged = result.flagged_drugs?.length ?? 0;
      const drift = result.drift_alerts?.length ?? 0;
      const qualityNote =
        skipped || flagged || drift
          ? ` · Skipped ${skipped} · Flagged ${flagged}${drift ? ` · Drift ${drift}` : ""}`
          : "";

      const accuracyLine = summary ? formatAccuracySummaryLine(summary) : null;

      setTrainState({
        status: "success",
        message: `${result.drugs_trained} SKU${result.drugs_trained === 1 ? "" : "s"} trained`,
        summary:
          (accuracyLine
            ? `${accuracyLine} · Run ${result.training_run_id?.slice(0, 8) ?? "—"}`
            : result.status === "skipped"
              ? "Using existing SHIELD-XR artifacts"
              : "SHIELD-XR stack trained") + qualityNote,
      });
      onModelsReady?.(true);
      onTrainComplete?.({
        accuracySummary: summary,
        trainingRunId: result.training_run_id || undefined,
        drugsTrained: result.drugs_trained || undefined,
      });
    } catch (err) {
      setTrainState({
        status: "error",
        message: err instanceof Error ? err.message : "Training failed",
      });
    }
  }, [forceRetrain, onModelsReady, onTrainComplete, selectedDrugCodes]);

  const isLoading = trainState.status === "loading";

  return (
    <div id="forecasting-training-controls" className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
      <div className="flex items-start gap-3">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white">
          <Shield className="h-5 w-5" />
        </span>
        <div>
          <h2 className="text-base font-semibold text-slate-900">Train SHIELD-XR</h2>
          <p className="mt-0.5 text-sm text-slate-500">
            Hospital-wide ensemble on enriched inpatient demand. Weekly reconciliation runs after daily
            ensemble training.
          </p>
        </div>
      </div>

      <div className="mt-5 space-y-4">
        <label className="block">
          <span className="mb-1.5 block text-sm font-medium text-slate-700">
            Limit to specific drugs <span className="font-normal text-slate-400">(optional)</span>
          </span>
          <DrugMultiSelectSearch
            selectedCodes={selectedDrugCodes}
            onChange={setSelectedDrugCodes}
            search={searchEnrichedDrugs}
            disabled={isLoading}
            placeholder="Search enriched panel (3+ chars)…"
          />
        </label>

        <label className="flex cursor-pointer items-center justify-between rounded-xl border border-slate-200 bg-slate-50/60 px-4 py-3">
          <div>
            <p className="text-sm font-medium text-slate-800">Force retrain</p>
            <p className="text-xs text-slate-500">Overwrite existing model artifacts</p>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={forceRetrain}
            disabled={isLoading}
            onClick={() => setForceRetrain((v) => !v)}
            className={`relative h-6 w-11 shrink-0 rounded-full transition ${
              forceRetrain ? "bg-blue-600" : "bg-slate-300"
            } disabled:opacity-60`}
          >
            <span
              className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition ${
                forceRetrain ? "left-5" : "left-0.5"
              }`}
            />
          </button>
        </label>

        <button
          type="button"
          onClick={() => void runTrain()}
          disabled={isLoading}
          className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isLoading && <Loader2 className="h-4 w-4 animate-spin" />}
          {trainState.status === "success" && <CheckCircle2 className="h-4 w-4" />}
          {trainState.status === "error" && <XCircle className="h-4 w-4" />}
          {!isLoading && trainState.status === "idle" && <Sparkles className="h-4 w-4" />}
          {isLoading ? "Training…" : "Run SHIELD-XR training"}
        </button>

        {isLoading && (
          <p className="text-center text-xs text-slate-500">
            Training can take several minutes for large catalogs.
          </p>
        )}

        {trainState.status === "success" && (
          <div className="rounded-xl border border-emerald-200 bg-emerald-50/80 px-4 py-3">
            <p className="text-sm font-medium text-emerald-800">{trainState.message}</p>
            {trainState.summary && (
              <p className="mt-1 text-xs text-emerald-700">{trainState.summary}</p>
            )}
          </div>
        )}

        {trainState.status === "error" && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {trainState.message.includes("No enriched") ? (
              <>
                {trainState.message}{" "}
                <Link href="/data-ingestion" className="font-semibold underline">
                  Upload data first
                </Link>
              </>
            ) : (
              trainState.message
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export function scrollToForecastingTraining() {
  document.getElementById("forecasting-training-controls")?.scrollIntoView({ behavior: "smooth" });
}
