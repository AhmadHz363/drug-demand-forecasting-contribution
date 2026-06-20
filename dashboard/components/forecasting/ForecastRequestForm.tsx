"use client";

import { useCallback, useState } from "react";
import { Loader2, RotateCcw, Sparkles } from "lucide-react";

import { searchForecastedDrugs } from "@/lib/api";
import { MAX_FORECAST_HORIZON } from "@/lib/constants";
import type { ForecastRequest } from "@/lib/types";

import {
  DrugSearchCombobox,
  PaginatedDrugPicker,
} from "@/components/forecasting/DrugSearchCombobox";
import {
  FormField,
  inputClass,
  Panel,
  PanelBody,
  PanelHeader,
} from "@/components/cold-start/ui";

interface ForecastRequestFormProps {
  onSubmit: (request: ForecastRequest) => void;
  isLoading: boolean;
}

export function ForecastRequestForm({ onSubmit, isLoading }: ForecastRequestFormProps) {
  const [drugCode, setDrugCode] = useState("");
  const [customDrugCode, setCustomDrugCode] = useState("");
  const [useCustomDrug, setUseCustomDrug] = useState(false);
  const [horizon, setHorizon] = useState(7);
  const [centerSynId, setCenterSynId] = useState("");
  const [includeShap, setIncludeShap] = useState(false);
  const [includeAttention, setIncludeAttention] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const loadForecastedDrugsPage = useCallback(async (page: number) => {
    const result = await searchForecastedDrugs(undefined, page);
    return {
      items: result.items.map((code) => ({ drug_code: code })),
      total: result.total,
      page: result.page,
      page_size: result.page_size,
      total_pages: result.total_pages,
    };
  }, []);

  const searchForecastedDrugCodes = useCallback(async (query: string, page: number) => {
    const result = await searchForecastedDrugs(query, page);
    return {
      items: result.items.map((code) => ({ drug_code: code })),
      total: result.total,
      page: result.page,
      page_size: result.page_size,
      total_pages: result.total_pages,
    };
  }, []);

  const resetForm = () => {
    setDrugCode("");
    setCustomDrugCode("");
    setUseCustomDrug(false);
    setHorizon(7);
    setCenterSynId("");
    setIncludeShap(false);
    setIncludeAttention(false);
    setValidationError(null);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const resolvedCode = useCustomDrug ? customDrugCode.trim() : drugCode.trim();
    if (!resolvedCode) {
      setValidationError("Please select or enter a drug code.");
      return;
    }
    setValidationError(null);

    onSubmit({
      drug_code: resolvedCode,
      horizon_days: horizon,
      center_syn_id: centerSynId.trim() || undefined,
      include_shap: includeShap,
      include_attention: includeAttention,
    });
  };

  return (
    <Panel id="forecast-configure-section">
      <PanelHeader
        step={2}
        title="Forecast configuration"
        description="Select a drug that already has forecast results, or search for one by code."
        action={
          <button
            type="button"
            onClick={resetForm}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset
          </button>
        }
      />
      <PanelBody>
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-3">
            <FormField
              label="Drug selection"
              required
              hint="Drugs with rows in the forecast_results table. Ten shown per page."
            >
              <PaginatedDrugPicker
                selectedCode={drugCode}
                onSelect={(code) => {
                  setUseCustomDrug(false);
                  setDrugCode(code);
                }}
                loadPage={loadForecastedDrugsPage}
                disabled={useCustomDrug || isLoading}
              />
            </FormField>

            <label className="flex cursor-pointer items-center justify-between rounded-xl border border-slate-200 bg-slate-50/50 px-4 py-3">
              <div>
                <p className="text-sm font-medium text-slate-800">Use custom drug code</p>
                <p className="text-xs text-slate-500">
                  Search among drugs that already have forecast results
                </p>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={useCustomDrug}
                onClick={() => setUseCustomDrug((v) => !v)}
                className={`relative h-6 w-11 shrink-0 rounded-full transition ${
                  useCustomDrug ? "bg-blue-600" : "bg-slate-300"
                }`}
              >
                <span
                  className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition ${
                    useCustomDrug ? "left-5" : "left-0.5"
                  }`}
                />
              </button>
            </label>

            {useCustomDrug && (
              <FormField
                label="Custom drug code"
                required
                hint="Type at least 3 letters to search forecast_results."
              >
                <DrugSearchCombobox
                  value={customDrugCode}
                  onChange={setCustomDrugCode}
                  onSelect={(option) => setCustomDrugCode(option.drug_code)}
                  search={searchForecastedDrugCodes}
                  disabled={isLoading}
                  showDrugName={false}
                  placeholder="Search forecasted drug codes…"
                  emptyHint="Enter at least 3 letters to search drugs with forecast results."
                />
              </FormField>
            )}
          </div>

          <FormField
            label={`Forecast horizon — ${horizon} day${horizon === 1 ? "" : "s"}`}
            hint={`Maximum ${MAX_FORECAST_HORIZON} days`}
          >
            <div className="flex items-center gap-4">
              <input
                type="range"
                min={1}
                max={MAX_FORECAST_HORIZON}
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
                className="flex-1"
              />
              <span className="w-12 rounded-lg bg-blue-50 px-2 py-1 text-center text-sm font-bold text-blue-700">
                {horizon}d
              </span>
            </div>
          </FormField>

          <FormField
            label="Center ID"
            hint="Optional — filter demand history to a specific hospital center"
          >
            <input
              type="text"
              value={centerSynId}
              onChange={(e) => setCenterSynId(e.target.value)}
              placeholder="Leave empty for all centers"
              className={`${inputClass} sm:max-w-md`}
            />
          </FormField>

          <div className="grid gap-3 sm:grid-cols-2">
            <ToggleCard
              label="Include SHAP features"
              description="Explain which features drove the LightGBM forecast"
              checked={includeShap}
              onChange={setIncludeShap}
            />
            <ToggleCard
              label="Include TFT attention"
              description="Show temporal attention weights from the TFT model"
              checked={includeAttention}
              onChange={setIncludeAttention}
            />
          </div>

          {validationError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {validationError}
            </div>
          )}

          <button
            type="submit"
            disabled={isLoading}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3.5 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:opacity-60"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Running forecast…
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" />
                Generate forecast
              </>
            )}
          </button>
        </form>
      </PanelBody>
    </Panel>
  );
}

function ToggleCard({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-start justify-between gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3">
      <div>
        <p className="text-sm font-medium text-slate-800">{label}</p>
        <p className="mt-0.5 text-xs text-slate-500">{description}</p>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative mt-0.5 h-6 w-11 shrink-0 rounded-full transition ${
          checked ? "bg-blue-600" : "bg-slate-300"
        }`}
      >
        <span
          className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition ${
            checked ? "left-5" : "left-0.5"
          }`}
        />
      </button>
    </label>
  );
}
