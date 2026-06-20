"use client";

import { useState } from "react";
import {
  BarChart3,
  GitCompare,
  LineChart,
  Radar,
  Sparkles,
} from "lucide-react";

import type { ColdStartPredictResponse } from "@/lib/types";

import { EmbeddingRadarChart } from "./EmbeddingRadarChart";
import { ForecastChart } from "./ForecastChart";
import { GraduationBadge } from "./GraduationBadge";
import { NeighboursPanel } from "./NeighboursPanel";
import { EmptyState, Panel, PanelBody, PanelHeader, Skeleton } from "./ui";

type TabId = "forecast" | "neighbours" | "embedding" | "stage";

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: "forecast", label: "Forecast", icon: <LineChart className="h-4 w-4" /> },
  { id: "neighbours", label: "Similar drugs", icon: <BarChart3 className="h-4 w-4" /> },
  { id: "embedding", label: "Embedding", icon: <Radar className="h-4 w-4" /> },
  { id: "stage", label: "Graduation", icon: <GitCompare className="h-4 w-4" /> },
];

interface ResultsTabsProps {
  data: ColdStartPredictResponse | null;
  previousEmbedding: number[] | null;
  isLoading: boolean;
  onScrollToForm: () => void;
}

export function ResultsTabs({
  data,
  previousEmbedding,
  isLoading,
  onScrollToForm,
}: ResultsTabsProps) {
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
      <Panel id="results-section">
        <PanelHeader
          step={3}
          title="Results"
          description="Your forecast, similar drugs, and model insights will appear here."
        />
        <PanelBody>
          <EmptyState
            icon={<Sparkles className="h-6 w-6" />}
            title="No prediction yet"
            description="Complete steps 1 and 2, then click Predict to see the demand forecast and analysis."
            action={
              <button
                type="button"
                onClick={onScrollToForm}
                className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
              >
                Go to drug form
              </button>
            }
          />
        </PanelBody>
      </Panel>
    );
  }

  return (
    <Panel id="results-section">
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
      <PanelBody className="pt-4">
        {activeTab === "forecast" && <ForecastChart forecast={data.forecast} embedded />}
        {activeTab === "neighbours" && (
          <NeighboursPanel
            data={{
              nearest_neighbours: data.nearest_neighbours,
              similarity_scores: data.similarity_scores,
            }}
            embedded
          />
        )}
        {activeTab === "embedding" && (
          <EmbeddingRadarChart
            embedding={data.embedding}
            previousEmbedding={previousEmbedding}
            embedded
          />
        )}
        {activeTab === "stage" && (
          <GraduationBadge
            stage={data.stage_used}
            observationCount={data.observation_count}
            uncertaintyNote={data.uncertainty_note}
            embedded
          />
        )}
      </PanelBody>
    </Panel>
  );
}
