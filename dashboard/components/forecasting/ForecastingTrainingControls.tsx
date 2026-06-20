"use client";

import { useCallback, useState } from "react";
import {
  BarChart3,
  Brain,
  CheckCircle2,
  Loader2,
  Sparkles,
  TrendingUp,
  XCircle,
} from "lucide-react";

import { searchReceiptDrugCodes, trainForecasting } from "@/lib/api";
import {
  FORECAST_MODEL_LABELS,
  FORECAST_MODELS,
  type ForecastModelId,
} from "@/lib/constants";

import { DrugMultiSelectSearch } from "@/components/forecasting/DrugSearchCombobox";
import { Panel, PanelBody, PanelHeader } from "@/components/cold-start/ui";

type TrainStatus = "idle" | "loading" | "success" | "error";

interface TrainState {
  status: TrainStatus;
  message: string;
  summary?: string;
}

const initialState: TrainState = { status: "idle", message: "" };

interface ForecastingTrainingControlsProps {
  onModelsReady?: (ready: boolean) => void;
}

export function ForecastingTrainingControls({
  onModelsReady,
}: ForecastingTrainingControlsProps) {
  const [trainState, setTrainState] = useState<TrainState>(initialState);
  const [selectedModels, setSelectedModels] = useState<ForecastModelId[]>([
    ...FORECAST_MODELS,
  ]);
  const [selectedDrugCodes, setSelectedDrugCodes] = useState<string[]>([]);
  const [forceRetrain, setForceRetrain] = useState(false);

  const searchReceiptDrugs = useCallback(async (query: string, page: number) => {
    const result = await searchReceiptDrugCodes(query, page);
    return {
      items: result.items,
      total: result.total,
      page: result.page,
      page_size: result.page_size,
      total_pages: result.total_pages,
    };
  }, []);

  const toggleModel = (model: ForecastModelId) => {
    setSelectedModels((prev) => {
      if (prev.includes(model)) {
        return prev.length === 1 ? prev : prev.filter((m) => m !== model);
      }
      return [...prev, model];
    });
  };

  const runTrain = useCallback(async () => {
    setTrainState({ status: "loading", message: "" });
    try {
      const result = await trainForecasting({
        drug_codes: selectedDrugCodes.length > 0 ? selectedDrugCodes : undefined,
        models: selectedModels,
        force_retrain: forceRetrain,
      });

      const smapeEntries = Object.entries(result.smape_summary);
      const avgSmape =
        smapeEntries.length > 0
          ? smapeEntries.reduce((s, [, v]) => s + v, 0) / smapeEntries.length
          : null;

      setTrainState({
        status: "success",
        message: `Ready — ${result.drugs_trained} drug${result.drugs_trained === 1 ? "" : "s"} trained`,
        summary:
          avgSmape !== null
            ? `Avg validation sMAPE: ${avgSmape.toFixed(1)}% · Models: ${result.models_trained.join(", ")}`
            : `Models: ${result.models_trained.join(", ")}`,
      });
      onModelsReady?.(true);
    } catch (err) {
      setTrainState({
        status: "error",
        message: err instanceof Error ? err.message : "Training failed",
      });
    }
  }, [forceRetrain, onModelsReady, selectedDrugCodes, selectedModels]);

  const isLoading = trainState.status === "loading";

  return (
    <Panel id="forecasting-training-controls">
      <PanelHeader
        step={1}
        title="Model setup"
        description="Train the ensemble forecasting stack. All three models are recommended for calibrated quantile predictions."
      />
      <PanelBody className="space-y-5">
        <div className="grid gap-3 sm:grid-cols-3">
          {FORECAST_MODELS.map((model) => {
            const selected = selectedModels.includes(model);
            const icons = {
              sarima: TrendingUp,
              lgbm: BarChart3,
              tft: Brain,
            };
            const Icon = icons[model];

            return (
              <button
                key={model}
                type="button"
                onClick={() => toggleModel(model)}
                disabled={isLoading}
                className={`flex flex-col rounded-xl border p-4 text-left transition ${
                  selected
                    ? "border-blue-300 bg-blue-50 ring-1 ring-blue-200"
                    : "border-slate-200 bg-slate-50/60 hover:border-slate-300"
                } disabled:cursor-not-allowed disabled:opacity-60`}
              >
                <span
                  className={`mb-2 flex h-9 w-9 items-center justify-center rounded-lg shadow-sm ring-1 ${
                    selected
                      ? "bg-blue-600 text-white ring-blue-500"
                      : "bg-white text-slate-500 ring-slate-200"
                  }`}
                >
                  <Icon className="h-5 w-5" />
                </span>
                <span className="text-sm font-semibold text-slate-900">
                  {FORECAST_MODEL_LABELS[model]}
                </span>
                <span className="mt-0.5 text-xs text-slate-500">
                  {model === "sarima" && "Seasonal ARIMA baseline"}
                  {model === "lgbm" && "Gradient boosting with lags"}
                  {model === "tft" && "Temporal fusion transformer"}
                </span>
              </button>
            );
          })}
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block">
            <span className="mb-1.5 block text-sm font-medium text-slate-700">
              Drug codes <span className="font-normal text-slate-400">(optional)</span>
            </span>
            <p className="mb-1.5 text-xs text-slate-500">
              Search and add codes from drug_receipts. Leave empty to train on all receipt drugs.
            </p>
            <DrugMultiSelectSearch
              selectedCodes={selectedDrugCodes}
              onChange={setSelectedDrugCodes}
              search={searchReceiptDrugs}
              disabled={isLoading}
              placeholder="Type 3+ letters to search drug_receipts…"
            />
          </label>

          <label className="flex cursor-pointer items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3 sm:mt-6">
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
        </div>

        <button
          type="button"
          onClick={() => void runTrain()}
          disabled={isLoading || selectedModels.length === 0}
          className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isLoading && <Loader2 className="h-4 w-4 animate-spin" />}
          {trainState.status === "success" && <CheckCircle2 className="h-4 w-4" />}
          {trainState.status === "error" && <XCircle className="h-4 w-4" />}
          {!isLoading && trainState.status === "idle" && <Sparkles className="h-4 w-4" />}
          {isLoading ? "Training models…" : "Train forecasting models"}
        </button>

        {isLoading && (
          <p className="text-center text-xs text-slate-500">
            Training can take several minutes per drug. Keep this tab open.
          </p>
        )}
        {trainState.status === "success" && (
          <div className="rounded-xl border border-emerald-200 bg-emerald-50/80 px-4 py-3 text-center">
            <p className="text-sm font-medium text-emerald-800">{trainState.message}</p>
            {trainState.summary && (
              <p className="mt-1 text-xs text-emerald-700">{trainState.summary}</p>
            )}
          </div>
        )}
        {trainState.status === "error" && (
          <p className="text-center text-xs text-red-600">{trainState.message}</p>
        )}
      </PanelBody>
    </Panel>
  );
}

export function scrollToForecastingTraining() {
  document.getElementById("forecasting-training-controls")?.scrollIntoView({ behavior: "smooth" });
}
