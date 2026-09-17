"use client";

import { useState } from "react";
import { Loader2, RotateCcw, Sparkles } from "lucide-react";

import {
  AVAILABILITY_OPTIONS,
  DEFAULT_DRUG_METADATA,
  DOSAGE_FORMS,
  DRUG_CLASSES,
  PREGNANCY_CATEGORIES,
  ROUTES,
} from "@/lib/constants";
import type {
  ColdStartPredictRequest,
  DrugMetadataInput,
  PharmacistEstimate,
} from "@/lib/types";

import {
  CollapsibleSection,
  FormField,
  inputClass,
  Panel,
  PanelBody,
  PanelHeader,
  selectClass,
} from "./ui";

interface DrugMetadataFormProps {
  onSubmit: (request: ColdStartPredictRequest) => void;
  isLoading: boolean;
  disabled?: boolean;
}

export function DrugMetadataForm({ onSubmit, isLoading, disabled = false }: DrugMetadataFormProps) {
  const [metadata, setMetadata] = useState<DrugMetadataInput>(DEFAULT_DRUG_METADATA);
  const [horizon, setHorizon] = useState(7);
  const [showPharmacist, setShowPharmacist] = useState(false);
  const [pharmacist, setPharmacist] = useState<PharmacistEstimate>({
    weekly_units: 200,
    confidence: 0.8,
  });
  const [validationError, setValidationError] = useState<string | null>(null);

  const update = <K extends keyof DrugMetadataInput>(
    key: K,
    value: DrugMetadataInput[K],
  ) => {
    setMetadata((prev) => ({ ...prev, [key]: value }));
  };

  const resetForm = () => {
    setMetadata(DEFAULT_DRUG_METADATA);
    setHorizon(7);
    setShowPharmacist(false);
    setValidationError(null);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!metadata.drug_code.trim()) {
      setValidationError("Please enter a drug code.");
      return;
    }
    if (!metadata.drug_name.trim()) {
      setValidationError("Please enter a drug name.");
      return;
    }
    setValidationError(null);

    const request: ColdStartPredictRequest = {
      drug_metadata: { ...metadata, drug_code: metadata.drug_code.trim() },
      forecast_horizon_days: horizon,
    };
    if (showPharmacist) {
      request.pharmacist_estimate = pharmacist;
    }
    onSubmit(request);
  };

  return (
    <Panel id="configure-section">
      <PanelHeader
        step={2}
        title="New drug configuration"
        description="Enter CAMEO attribute fields (same as the real-data cold-start notebook)."
        action={
          <button
            type="button"
            onClick={resetForm}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset example
          </button>
        }
      />
      <PanelBody>
        <form onSubmit={handleSubmit} className="space-y-4">
          <CollapsibleSection title="Identity" subtitle="Hospital code and product name" defaultOpen>
            <div className="grid gap-4 sm:grid-cols-2">
              <FormField label="Drug code" required>
                <input
                  type="text"
                  value={metadata.drug_code}
                  onChange={(e) => update("drug_code", e.target.value)}
                  placeholder="e.g. NEW-001"
                  className={inputClass}
                />
              </FormField>
              <FormField label="Drug name (ARTICLE)" required>
                <input
                  type="text"
                  value={metadata.drug_name}
                  onChange={(e) => update("drug_name", e.target.value)}
                  placeholder="e.g. Ceftriaxone 1g Injection"
                  className={inputClass}
                />
              </FormField>
              <FormField label="Generic name">
                <input
                  type="text"
                  value={metadata.generic_name ?? ""}
                  onChange={(e) => update("generic_name", e.target.value)}
                  className={inputClass}
                />
              </FormField>
            </div>
          </CollapsibleSection>

          <CollapsibleSection title="Pharmacological attributes" subtitle="Used for metric-learning analog search" defaultOpen>
            <div className="grid gap-4 sm:grid-cols-2">
              <FormField label="Drug class">
                <select
                  value={metadata.drug_class}
                  onChange={(e) => update("drug_class", e.target.value)}
                  className={selectClass}
                >
                  {DRUG_CLASSES.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </FormField>
              <FormField label="Dosage form">
                <select
                  value={metadata.dosage_form}
                  onChange={(e) => update("dosage_form", e.target.value)}
                  className={selectClass}
                >
                  {DOSAGE_FORMS.map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </FormField>
              <FormField label="Strength">
                <input
                  type="text"
                  value={metadata.strength}
                  onChange={(e) => update("strength", e.target.value)}
                  placeholder="e.g. 500 mg"
                  className={inputClass}
                />
              </FormField>
              <FormField label="Route of administration">
                <select
                  value={metadata.route_of_administration}
                  onChange={(e) => update("route_of_administration", e.target.value)}
                  className={selectClass}
                >
                  {ROUTES.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </FormField>
              <FormField label="Pregnancy category">
                <select
                  value={metadata.pregnancy_category ?? "MISSING"}
                  onChange={(e) => update("pregnancy_category", e.target.value)}
                  className={selectClass}
                >
                  {PREGNANCY_CATEGORIES.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </FormField>
              <FormField label="Availability">
                <select
                  value={metadata.availability ?? "Prescription"}
                  onChange={(e) => update("availability", e.target.value)}
                  className={selectClass}
                >
                  {AVAILABILITY_OPTIONS.map((a) => (
                    <option key={a} value={a}>
                      {a}
                    </option>
                  ))}
                </select>
              </FormField>
            </div>
          </CollapsibleSection>

          <CollapsibleSection title="Clinical text (optional)" subtitle="Parsed into complexity counts for embedding">
            <div className="grid gap-4">
              <FormField label="Indications">
                <textarea
                  value={metadata.indications ?? ""}
                  onChange={(e) => update("indications", e.target.value)}
                  rows={2}
                  className={inputClass}
                />
              </FormField>
              <FormField label="Side effects">
                <textarea
                  value={metadata.side_effects ?? ""}
                  onChange={(e) => update("side_effects", e.target.value)}
                  rows={2}
                  className={inputClass}
                />
              </FormField>
              <FormField label="Contraindications">
                <textarea
                  value={metadata.contraindications ?? ""}
                  onChange={(e) => update("contraindications", e.target.value)}
                  rows={2}
                  className={inputClass}
                />
              </FormField>
            </div>
          </CollapsibleSection>

          <CollapsibleSection title="Forecast options" subtitle="Horizon and optional pharmacist input">
            <div className="space-y-4">
              <FormField label={`Forecast horizon — ${horizon} day${horizon === 1 ? "" : "s"}`}>
                <div className="flex items-center gap-4">
                  <input
                    type="range"
                    min={1}
                    max={140}
                    value={horizon}
                    onChange={(e) => setHorizon(Number(e.target.value))}
                    className="flex-1"
                  />
                  <span className="w-12 rounded-lg bg-blue-50 px-2 py-1 text-center text-sm font-bold text-blue-700">
                    {horizon}d
                  </span>
                </div>
              </FormField>

              <div className="rounded-xl border border-slate-200 bg-slate-50/50">
                <label className="flex cursor-pointer items-center justify-between px-4 py-3">
                  <div>
                    <p className="text-sm font-medium text-slate-800">Include pharmacist estimate</p>
                    <p className="text-xs text-slate-500">Blends your manual weekly demand guess into the forecast</p>
                  </div>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={showPharmacist}
                    onClick={() => setShowPharmacist((s) => !s)}
                    className={`relative h-6 w-11 shrink-0 rounded-full transition ${
                      showPharmacist ? "bg-blue-600" : "bg-slate-300"
                    }`}
                  >
                    <span
                      className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition ${
                        showPharmacist ? "left-5" : "left-0.5"
                      }`}
                    />
                  </button>
                </label>
                {showPharmacist && (
                  <div className="space-y-3 border-t border-slate-200 bg-white px-4 py-4">
                    <FormField label="Expected weekly units">
                      <input
                        type="number"
                        min={0.01}
                        step={0.01}
                        value={pharmacist.weekly_units}
                        onChange={(e) =>
                          setPharmacist((p) => ({ ...p, weekly_units: Number(e.target.value) }))
                        }
                        className={`${inputClass} sm:max-w-xs`}
                      />
                    </FormField>
                    <FormField label={`Confidence — ${Math.round(pharmacist.confidence * 100)}%`}>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={pharmacist.confidence}
                        onChange={(e) =>
                          setPharmacist((p) => ({ ...p, confidence: Number(e.target.value) }))
                        }
                        className="w-full"
                      />
                    </FormField>
                  </div>
                )}
              </div>
            </div>
          </CollapsibleSection>

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
                Running CAMEO…
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
