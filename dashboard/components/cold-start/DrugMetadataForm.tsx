"use client";

import { useState } from "react";
import { Loader2, RotateCcw, Sparkles } from "lucide-react";

import {
  DEFAULT_DRUG_METADATA,
  PHARMA_FORMS,
  ROUTES,
  THERAPEUTIC_CLASSES,
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
}

const PRICE_TIER_LABELS = ["Budget", "Low", "Mid", "High", "Premium"] as const;

export function DrugMetadataForm({ onSubmit, isLoading }: DrugMetadataFormProps) {
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
        description="Describe the drug you want to forecast. Defaults are pre-filled with a sample antibiotic."
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
          <CollapsibleSection
            title="Basic information"
            subtitle="Drug identity and therapeutic profile"
            defaultOpen
          >
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
              <FormField label="Drug name" required>
                <input
                  type="text"
                  value={metadata.drug_name}
                  onChange={(e) => update("drug_name", e.target.value)}
                  placeholder="e.g. Ceftriaxone 1g Injection"
                  className={inputClass}
                />
              </FormField>
              <FormField label="Therapeutic class">
                <select
                  value={metadata.therapeutic_class}
                  onChange={(e) => update("therapeutic_class", e.target.value)}
                  className={selectClass}
                >
                  {THERAPEUTIC_CLASSES.map((c) => (
                    <option key={c} value={c}>
                      {c.charAt(0).toUpperCase() + c.slice(1)}
                    </option>
                  ))}
                </select>
              </FormField>
              <FormField label="ATC category" hint="WHO anatomical classification, e.g. J01 for antibacterials">
                <input
                  type="text"
                  placeholder="J01"
                  value={metadata.atc_category}
                  onChange={(e) => update("atc_category", e.target.value)}
                  className={inputClass}
                />
              </FormField>
              <FormField label="Pharmaceutical form">
                <select
                  value={metadata.pharmaceutical_form}
                  onChange={(e) => update("pharmaceutical_form", e.target.value)}
                  className={selectClass}
                >
                  {PHARMA_FORMS.map((f) => (
                    <option key={f} value={f}>
                      {f.charAt(0).toUpperCase() + f.slice(1)}
                    </option>
                  ))}
                </select>
              </FormField>
              <FormField label="Route of administration">
                <select
                  value={metadata.route_of_administration}
                  onChange={(e) => update("route_of_administration", e.target.value)}
                  className={selectClass}
                >
                  {ROUTES.map((r) => (
                    <option key={r} value={r}>
                      {r.toUpperCase()}
                    </option>
                  ))}
                </select>
              </FormField>
            </div>
          </CollapsibleSection>

          <CollapsibleSection
            title="Priority & classification"
            subtitle="VEN criticality, ABC economic value, and cost tier"
          >
            <div className="space-y-4">
              <FormField
                label="VEN class"
                hint="Vital · Essential · Non-essential — indicates clinical criticality"
              >
                <RadioGroup
                  options={[
                    { value: "V", label: "Vital", sub: "Critical", color: "border-red-500 bg-red-50 text-red-700 ring-red-200" },
                    { value: "E", label: "Essential", sub: "Important", color: "border-amber-500 bg-amber-50 text-amber-800 ring-amber-200" },
                    { value: "N", label: "Non-essential", sub: "Routine", color: "border-emerald-500 bg-emerald-50 text-emerald-800 ring-emerald-200" },
                  ]}
                  value={metadata.ven_class}
                  onChange={(v) => update("ven_class", v as DrugMetadataInput["ven_class"])}
                />
              </FormField>
              <FormField
                label="ABC class"
                hint="A = high value · B = medium · C = low consumption value"
              >
                <RadioGroup
                  options={[
                    { value: "A", label: "Class A", sub: "High value", color: "border-violet-500 bg-violet-50 text-violet-800 ring-violet-200" },
                    { value: "B", label: "Class B", sub: "Medium", color: "border-blue-500 bg-blue-50 text-blue-800 ring-blue-200" },
                    { value: "C", label: "Class C", sub: "Low value", color: "border-slate-400 bg-slate-50 text-slate-700 ring-slate-200" },
                  ]}
                  value={metadata.abc_class}
                  onChange={(v) => update("abc_class", v as DrugMetadataInput["abc_class"])}
                />
              </FormField>
              <FormField label="Unit price tier">
                <div className="flex flex-wrap gap-2">
                  {([1, 2, 3, 4, 5] as const).map((tier) => (
                    <button
                      key={tier}
                      type="button"
                      onClick={() => update("unit_price_tier", tier)}
                      className={`flex min-w-[4.5rem] flex-col items-center rounded-xl border px-3 py-2 text-center transition ${
                        metadata.unit_price_tier === tier
                          ? "border-blue-600 bg-blue-600 text-white shadow-sm"
                          : "border-slate-200 bg-white text-slate-600 hover:border-blue-300"
                      }`}
                    >
                      <span className="text-sm font-bold">{tier}</span>
                      <span className="text-[10px] opacity-80">{PRICE_TIER_LABELS[tier - 1]}</span>
                    </button>
                  ))}
                </div>
              </FormField>
            </div>
          </CollapsibleSection>

          <CollapsibleSection
            title="Storage & compliance"
            subtitle="Handling requirements and shelf life"
          >
            <div className="grid gap-4 sm:grid-cols-2">
              <Toggle
                label="Requires refrigeration"
                checked={metadata.requires_refrigeration}
                onChange={(v) => update("requires_refrigeration", v)}
              />
              <Toggle
                label="Controlled substance"
                checked={metadata.is_controlled_substance}
                onChange={(v) => update("is_controlled_substance", v)}
              />
              <div className="sm:col-span-2">
                <FormField label="Average shelf life" hint="Typical shelf life in days (minimum 30)">
                  <input
                    type="number"
                    min={30}
                    value={metadata.average_shelf_life_days}
                    onChange={(e) => update("average_shelf_life_days", Number(e.target.value))}
                    className={`${inputClass} sm:max-w-xs`}
                  />
                </FormField>
              </div>
            </div>
          </CollapsibleSection>

          <CollapsibleSection
            title="Forecast options"
            subtitle="Horizon length and optional pharmacist input"
          >
            <div className="space-y-4">
              <FormField label={`Forecast horizon — ${horizon} day${horizon === 1 ? "" : "s"}`}>
                <div className="flex items-center gap-4">
                  <input
                    type="range"
                    min={1}
                    max={30}
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
            disabled={isLoading}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3.5 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:opacity-60"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Running prediction…
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

function RadioGroup({
  options,
  value,
  onChange,
}: {
  options: { value: string; label: string; sub: string; color: string }[];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="grid gap-2 sm:grid-cols-3">
      {options.map((opt) => {
        const selected = value === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={`rounded-xl border px-3 py-2.5 text-left transition ring-1 ${
              selected ? opt.color : "border-slate-200 bg-white text-slate-500 ring-transparent hover:border-slate-300"
            }`}
          >
            <span className="block text-sm font-semibold">{opt.label}</span>
            <span className="block text-[11px] opacity-75">{opt.sub}</span>
          </button>
        );
      })}
    </div>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative h-6 w-11 rounded-full transition ${
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
