"use client";

import { useCallback, useState } from "react";
import { Loader2, RotateCcw, Sparkles } from "lucide-react";

import { searchEnrichedDrugCodes } from "@/lib/api";
import type { ForecastRequest } from "@/lib/types";
import { WEEK_HORIZON_OPTIONS } from "@/lib/weeklyForecast";

import {
  DrugSearchCombobox,
} from "@/components/forecasting/DrugSearchCombobox";
import { FormField, inputClass } from "@/components/cold-start/ui";

interface ForecastRequestFormProps {
  onSubmit: (request: ForecastRequest) => void;
  isLoading: boolean;
  disabled?: boolean;
  initialDrugCode?: string | null;
}

export function ForecastRequestForm({
  onSubmit,
  isLoading,
  disabled,
  initialDrugCode,
}: ForecastRequestFormProps) {
  const [drugCode, setDrugCode] = useState(initialDrugCode ?? "");
  const [drugName, setDrugName] = useState<string | null>(null);
  const [weeks, setWeeks] = useState<(typeof WEEK_HORIZON_OPTIONS)[number]["weeks"]>(1);
  const [centerSynId, setCenterSynId] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);

  const searchEnriched = useCallback(async (query: string, page: number) => {
    const result = await searchEnrichedDrugCodes(query, page);
    return {
      items: result.items,
      total: result.total,
      page: result.page,
      page_size: result.page_size,
      total_pages: result.total_pages,
    };
  }, []);

  const selectedHorizon = WEEK_HORIZON_OPTIONS.find((o) => o.weeks === weeks) ?? WEEK_HORIZON_OPTIONS[0];

  const resetForm = () => {
    setDrugCode("");
    setDrugName(null);
    setWeeks(1);
    setCenterSynId("");
    setValidationError(null);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const resolvedCode = drugCode.trim();
    if (!resolvedCode) {
      setValidationError("Select a drug from the enriched training panel.");
      return;
    }
    setValidationError(null);
    onSubmit({
      drug_code: resolvedCode,
      horizon_days: selectedHorizon.days,
      center_syn_id: centerSynId.trim() || undefined,
    });
  };

  return (
    <div id="forecast-configure-section" className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-slate-900">Weekly forecast</h2>
          <p className="mt-0.5 text-sm text-slate-500">
            Pick a drug from the enriched panel and choose a week-based horizon.
          </p>
        </div>
        <button
          type="button"
          onClick={resetForm}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Reset
        </button>
      </div>

      <form onSubmit={handleSubmit} className="mt-5 space-y-5">
        <FormField
          label="Drug code"
          required
          hint="Search drugs in hospital_daily_demand_enriched (inpatient sales panel)."
        >
          <DrugSearchCombobox
            value={drugCode}
            onChange={setDrugCode}
            onSelect={(option) => {
              setDrugCode(option.drug_code);
              setDrugName(option.drug_name ?? null);
            }}
            search={searchEnriched}
            disabled={isLoading || disabled}
            placeholder="Type 3+ characters…"
            emptyHint="Enter at least 3 characters to search the enriched panel."
          />
          {drugName && (
            <p className="mt-1.5 text-xs text-slate-500">{drugName}</p>
          )}
        </FormField>

        <FormField label="Forecast horizon" hint="Daily predictions aggregated into weekly buckets.">
          <div className="grid grid-cols-3 gap-2">
            {WEEK_HORIZON_OPTIONS.map((option) => {
              const selected = weeks === option.weeks;
              return (
                <button
                  key={option.weeks}
                  type="button"
                  disabled={isLoading || disabled}
                  onClick={() => setWeeks(option.weeks)}
                  className={`rounded-xl border px-3 py-3 text-center transition ${
                    selected
                      ? "border-blue-300 bg-blue-50 ring-2 ring-blue-200"
                      : "border-slate-200 bg-white hover:border-slate-300"
                  } disabled:opacity-50`}
                >
                  <p className={`text-lg font-bold ${selected ? "text-blue-700" : "text-slate-800"}`}>
                    {option.weeks}
                  </p>
                  <p className="text-xs text-slate-500">{option.label}</p>
                  <p className="mt-0.5 text-[10px] text-slate-400">{option.days} days</p>
                </button>
              );
            })}
          </div>
        </FormField>

        <FormField label="Center filter" hint="Optional — restrict history to one hospital center.">
          <input
            type="text"
            value={centerSynId}
            onChange={(e) => setCenterSynId(e.target.value)}
            placeholder="All centers"
            disabled={isLoading || disabled}
            className={`${inputClass} sm:max-w-md`}
          />
        </FormField>

        {validationError && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {validationError}
          </div>
        )}

        <button
          type="submit"
          disabled={isLoading || disabled}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3.5 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:opacity-60"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Generating weekly forecast…
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              Generate {selectedHorizon.label} forecast
            </>
          )}
        </button>
      </form>
    </div>
  );
}
