"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Loader2, Play, RotateCcw } from "lucide-react";

import {
  fetchHoldoutDateSuggestions,
  runHoldoutValidation,
  searchReceiptDrugCodes,
} from "@/lib/api";
import { FORECAST_MODEL_LABELS, type ForecastModelId } from "@/lib/constants";
import {
  buildTestEndOptions,
  buildTestStartOptions,
  buildTrainEndOptions,
  pickValidSelection,
} from "@/lib/holdoutDateOptions";
import type { HoldoutDateSuggestions, HoldoutResponse, HoldoutValidationRequest } from "@/lib/types";

import {
  DrugSearchCombobox,
} from "@/components/forecasting/DrugSearchCombobox";
import {
  FormField,
  Panel,
  PanelBody,
  PanelHeader,
  selectClass,
  StatChip,
} from "@/components/cold-start/ui";

import { HoldoutValidationChart } from "./HoldoutValidationChart";

type SeriesKey = "ensemble" | "sarima" | "lgbm" | "classical";

const SERIES_OPTIONS: { id: SeriesKey; label: string }[] = [
  { id: "ensemble", label: "Ensemble" },
  { id: "sarima", label: "SARIMA" },
  { id: "lgbm", label: "LightGBM" },
  { id: "classical", label: "Classical" },
];

function accuracyAccent(pct: number): "green" | "amber" | "blue" | "slate" {
  if (pct >= 70) return "green";
  if (pct >= 40) return "amber";
  return "slate";
}

export function HoldoutValidationPanel() {
  const [drugCode, setDrugCode] = useState("");
  const [dateSuggestions, setDateSuggestions] = useState<HoldoutDateSuggestions | null>(null);
  const [trainEnd, setTrainEnd] = useState("");
  const [testStart, setTestStart] = useState("");
  const [testEnd, setTestEnd] = useState("");
  const [showSeries, setShowSeries] = useState<SeriesKey[]>(["ensemble", "sarima", "lgbm"]);
  const [data, setData] = useState<HoldoutResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingDates, setIsLoadingDates] = useState(false);

  const searchDrugs = useCallback(async (query: string, page: number) => {
    const result = await searchReceiptDrugCodes(query, page);
    return {
      items: result.items,
      total: result.total,
      page: result.page,
      page_size: result.page_size,
      total_pages: result.total_pages,
    };
  }, []);

  const applyDateSuggestions = useCallback((suggestions: HoldoutDateSuggestions) => {
    setDateSuggestions(suggestions);
    setTrainEnd(suggestions.defaults.train_end);
    setTestStart(suggestions.defaults.test_start);
    setTestEnd(suggestions.defaults.test_end);
  }, []);

  const loadDateSuggestions = useCallback(
    async (code: string) => {
      const trimmed = code.trim();
      if (!trimmed) {
        setDateSuggestions(null);
        setTrainEnd("");
        setTestStart("");
        setTestEnd("");
        return;
      }

      setIsLoadingDates(true);
      try {
        const suggestions = await fetchHoldoutDateSuggestions(trimmed);
        applyDateSuggestions(suggestions);
        setError(null);
      } catch (err) {
        setDateSuggestions(null);
        setTrainEnd("");
        setTestStart("");
        setTestEnd("");
        setError(
          err instanceof Error
            ? err.message
            : "Could not load date suggestions for this drug.",
        );
      } finally {
        setIsLoadingDates(false);
      }
    },
    [applyDateSuggestions],
  );

  const handleDrugSelect = useCallback(
    (code: string) => {
      setDrugCode(code);
      void loadDateSuggestions(code);
    },
    [loadDateSuggestions],
  );

  const trainEndOptions = useMemo(
    () => (dateSuggestions ? buildTrainEndOptions(dateSuggestions) : []),
    [dateSuggestions],
  );

  const testStartOptions = useMemo(
    () =>
      dateSuggestions && trainEnd
        ? buildTestStartOptions(trainEnd, dateSuggestions)
        : [],
    [dateSuggestions, trainEnd],
  );

  const testEndOptions = useMemo(
    () =>
      dateSuggestions && testStart
        ? buildTestEndOptions(testStart, dateSuggestions)
        : [],
    [dateSuggestions, testStart],
  );

  useEffect(() => {
    if (!dateSuggestions || !trainEnd) return;
    const nextTestStart = pickValidSelection(testStart, testStartOptions);
    if (nextTestStart !== testStart) {
      setTestStart(nextTestStart);
    }
  }, [dateSuggestions, trainEnd, testStart, testStartOptions]);

  useEffect(() => {
    if (!dateSuggestions || !testStart) return;
    const nextTestEnd = pickValidSelection(testEnd, testEndOptions);
    if (nextTestEnd !== testEnd) {
      setTestEnd(nextTestEnd);
    }
  }, [dateSuggestions, testStart, testEnd, testEndOptions]);

  const toggleSeries = (id: SeriesKey) => {
    setShowSeries((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id],
    );
  };

  const reset = () => {
    setDrugCode("");
    setDateSuggestions(null);
    setTrainEnd("");
    setTestStart("");
    setTestEnd("");
    setData(null);
    setError(null);
  };

  const handleRun = async () => {
    const code = drugCode.trim();
    if (!code) {
      setError("Please select a drug code.");
      return;
    }
    if (!trainEnd || !testStart || !testEnd) {
      setError("Choose train and test windows for the selected drug.");
      return;
    }
    setError(null);
    setIsLoading(true);
    try {
      const req: HoldoutValidationRequest = {
        drug_code: code,
        train_end: trainEnd,
        test_start: testStart,
        test_end: testEnd,
        models: ["sarima", "lgbm", "classical"],
      };
      const result = await runHoldoutValidation(req);
      setData(result);
    } catch (err) {
      setData(null);
      setError(err instanceof Error ? err.message : "Hold-out validation failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const datesDisabled = isLoading || isLoadingDates || !dateSuggestions;

  return (
    <Panel id="holdout-validation-section">
      <PanelHeader
        title="Hold-out validation"
        description="Train on historical data, forecast a held-out test window, and compare predictions to actual demand."
        action={
          <button
            type="button"
            onClick={reset}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset
          </button>
        }
      />
      <PanelBody className="space-y-6">
        <div className="rounded-xl border border-slate-200 bg-slate-50/60 px-4 py-3 text-sm text-slate-600">
          Select a drug to auto-fill train/test windows from its receipt history. Negative daily
          totals (inter-department transfers) are treated as positive consumption in training and
          evaluation. Adjust the dropdowns if you want a different split.
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <FormField
            label="Drug code"
            required
            hint="Search drugs with receipt history (min 3 characters)."
          >
            <DrugSearchCombobox
              value={drugCode}
              onChange={setDrugCode}
              onSelect={(option) => handleDrugSelect(option.drug_code)}
              search={searchDrugs}
              disabled={isLoading}
              placeholder="Search receipt drug codes…"
            />
          </FormField>

          <div className="grid gap-3 sm:grid-cols-3">
            <FormField
              label="Train through"
              required
              hint={
                dateSuggestions
                  ? `History from ${dateSuggestions.data_start}`
                  : "Select a drug first"
              }
            >
              <select
                value={trainEnd}
                onChange={(e) => setTrainEnd(e.target.value)}
                className={selectClass}
                disabled={datesDisabled}
              >
                {!trainEnd && <option value="">Choose train end…</option>}
                {trainEndOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </FormField>
            <FormField
              label="Test from"
              required
              hint={trainEnd ? "Must be after train end" : "Depends on train end"}
            >
              <select
                value={testStart}
                onChange={(e) => setTestStart(e.target.value)}
                className={selectClass}
                disabled={datesDisabled || !trainEnd}
              >
                {!testStart && <option value="">Choose test start…</option>}
                {testStartOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </FormField>
            <FormField
              label="Test through"
              required
              hint={
                dateSuggestions
                  ? `Latest data: ${dateSuggestions.data_end}`
                  : "Depends on test start"
              }
            >
              <select
                value={testEnd}
                onChange={(e) => setTestEnd(e.target.value)}
                className={selectClass}
                disabled={datesDisabled || !testStart}
              >
                {!testEnd && <option value="">Choose test end…</option>}
                {testEndOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </FormField>
          </div>
        </div>

        {isLoadingDates && (
          <p className="flex items-center gap-2 text-xs text-slate-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Loading date suggestions for {drugCode}…
          </p>
        )}

        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
            Show on chart
          </p>
          <div className="flex flex-wrap gap-2">
            {SERIES_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => toggleSeries(opt.id)}
                className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
                  showSeries.includes(opt.id)
                    ? "border-blue-600 bg-blue-50 text-blue-700"
                    : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <button
          type="button"
          onClick={handleRun}
          disabled={isLoading || isLoadingDates}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-slate-900 px-4 py-3.5 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800 disabled:opacity-60 sm:w-auto sm:min-w-[220px]"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Running hold-out validation…
            </>
          ) : (
            <>
              <Play className="h-4 w-4" />
              Run hold-out validation
            </>
          )}
        </button>

        {data && (
          <>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <StatChip
                label={
                  data.smape_unreliable ||
                  data.demand_segment === "intermittent" ||
                  data.demand_segment === "lumpy"
                    ? "MASE skill (ensemble)"
                    : "Total accuracy (ensemble)"
                }
                value={
                  data.smape_unreliable ||
                  data.demand_segment === "intermittent" ||
                  data.demand_segment === "lumpy"
                    ? `${data.total_accuracy_skill_pct.toFixed(1)}%`
                    : `${data.total_accuracy_pct.toFixed(1)}%`
                }
                accent={accuracyAccent(
                  data.smape_unreliable ||
                    data.demand_segment === "intermittent" ||
                    data.demand_segment === "lumpy"
                    ? data.total_accuracy_skill_pct
                    : data.total_accuracy_pct,
                )}
              />
              {data.metrics.ensemble && (
                <>
                  <StatChip
                    label="Ensemble MAE"
                    value={data.metrics.ensemble.mae.toFixed(1)}
                    accent="blue"
                  />
                  <StatChip
                    label="Ensemble sMAPE"
                    value={`${data.metrics.ensemble.smape.toFixed(1)}%`}
                    accent="slate"
                  />
                  <StatChip
                    label="Ensemble MASE skill"
                    value={`${data.metrics.ensemble.accuracy_skill_pct.toFixed(1)}%`}
                    accent="blue"
                  />
                  <StatChip
                    label="Interval coverage (90%)"
                    value={`${(data.metrics.ensemble.coverage_90 * 100).toFixed(0)}%`}
                    accent={
                      data.metrics.ensemble.coverage_90 >= 0.85
                        ? "green"
                        : data.metrics.ensemble.coverage_90 >= 0.6
                          ? "amber"
                          : "slate"
                    }
                  />
                </>
              )}
            </div>

            {data.smape_unreliable && (
              <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                MASE skill is the primary accuracy metric for {data.demand_segment} demand
                {data.validation_zero_actual_fraction != null && (
                  <> ({(data.validation_zero_actual_fraction * 100).toFixed(0)}% zero-actual days)</>
                )}
                . sMAPE is shown for reference only on sparse series.
              </p>
            )}

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {(["ensemble", "sarima", "lgbm", "classical"] as const).map((model) => {
                const metrics = data.metrics[model];
                if (!metrics) return null;
                const label =
                  model === "ensemble"
                    ? "Ensemble"
                    : FORECAST_MODEL_LABELS[model as ForecastModelId];
                const useMaseSkill =
                  data.smape_unreliable ||
                  data.demand_segment === "intermittent" ||
                  data.demand_segment === "lumpy";
                return (
                  <StatChip
                    key={model}
                    label={`${label} ${useMaseSkill ? "MASE skill" : "accuracy"}`}
                    value={
                      useMaseSkill
                        ? `${metrics.accuracy_skill_pct.toFixed(1)}%`
                        : `${metrics.accuracy_pct.toFixed(1)}%`
                    }
                    accent={accuracyAccent(
                      useMaseSkill ? metrics.accuracy_skill_pct : metrics.accuracy_pct,
                    )}
                  />
                );
              })}
            </div>

            {Object.keys(data.model_errors).length > 0 && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-900">
                {Object.entries(data.model_errors).map(([model, msg]) => (
                  <p key={model}>
                    <strong>{model.toUpperCase()}</strong>: {msg}
                  </p>
                ))}
              </div>
            )}

            <HoldoutValidationChart data={data} showSeries={showSeries} embedded />
          </>
        )}
      </PanelBody>
    </Panel>
  );
}
