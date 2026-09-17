"use client";

import { useState } from "react";
import { Eye, GitBranch, Layers, LineChart, Sparkles } from "lucide-react";

import type { ForecastResponse } from "@/lib/types";

import { ForecastChart } from "@/components/cold-start/ForecastChart";
import { EmptyState, Panel, PanelBody, PanelHeader, Skeleton } from "@/components/cold-start/ui";

import { AttentionWeightsPanel } from "./AttentionWeightsPanel";
import { ModelWeightsPanel } from "./ModelWeightsPanel";
import { ShapFeaturesPanel } from "./ShapFeaturesPanel";

type TabId = "forecast" | "weights" | "shap" | "attention";

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: "forecast", label: "Forecast", icon: <LineChart className="h-4 w-4" /> },
  { id: "weights", label: "Model weights", icon: <Layers className="h-4 w-4" /> },
  { id: "shap", label: "SHAP", icon: <GitBranch className="h-4 w-4" /> },
  { id: "attention", label: "Attention", icon: <Eye className="h-4 w-4" /> },
];

interface ForecastingResultsTabsProps {
  data: ForecastResponse | null;
  isLoading: boolean;
  onScrollToForm: () => void;
}

export function ForecastingResultsTabs({
  data,
  isLoading,
  onScrollToForm,
}: ForecastingResultsTabsProps) {
  const [activeTab, setActiveTab] = useState<TabId>("forecast");

  if (isLoading) {
    return (
      <Panel>
        <PanelHeader step={3} title="Results" description="Generating forecast…" />
        <PanelBody className="space-y-4">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-72 w-full" />
        </PanelBody>
      </Panel>
    );
  }

  if (!data) {
    return (
      <Panel id="forecasting-results-section">
        <PanelHeader
          step={3}
          title="Results"
          description="Calibrated quantile forecasts and model insights will appear here."
        />
        <PanelBody>
          <EmptyState
            icon={<Sparkles className="h-6 w-6" />}
            title="No forecast yet"
            description="Complete steps 1 and 2, then click Generate forecast to see the ensemble prediction."
            action={
              <button
                type="button"
                onClick={onScrollToForm}
                className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
              >
                Go to configuration
              </button>
            }
          />
        </PanelBody>
      </Panel>
    );
  }

  return (
    <Panel id="forecasting-results-section">
      <PanelHeader
        step={3}
        title="Results"
        description={`Forecast for ${data.drug_code} · ${data.forecast.length}-day horizon`}
      />
      <div className="border-b border-slate-100 px-4 sm:px-6">
        <div className="flex gap-1 overflow-x-auto pb-px" role="tablist">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex shrink-0 items-center gap-2 border-b-2 px-4 py-3 text-sm font-medium transition ${
                activeTab === tab.id
                  ? "border-blue-600 text-blue-700"
                  : "border-transparent text-slate-500 hover:border-slate-200 hover:text-slate-700"
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </div>
      <PanelBody className="space-y-4 pt-4">
        {data.uncertainty_note && (
          <div className="rounded-xl border border-amber-200 bg-amber-50/80 px-4 py-3 text-sm text-amber-900">
            {data.uncertainty_note}
          </div>
        )}

        {activeTab === "forecast" && (
          <ForecastChart forecast={data.forecast} history={data.history} embedded />
        )}
        {activeTab === "weights" && (
          <ModelWeightsPanel weights={data.model_weights} embedded />
        )}
        {activeTab === "shap" && (
          <ShapFeaturesPanel features={data.shap_features} embedded />
        )}
        {activeTab === "attention" && (
          <AttentionWeightsPanel weights={data.attention_weights} embedded />
        )}
      </PanelBody>
    </Panel>
  );
}
