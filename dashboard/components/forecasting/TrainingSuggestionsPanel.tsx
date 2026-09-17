"use client";

import { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  Lightbulb,
  Loader2,
  RefreshCw,
  TrendingDown,
} from "lucide-react";

import { fetchForecastingPerformance } from "@/lib/api";
import { EXAMPLE_FORECAST_DRUGS } from "@/lib/constants";
import type { ModelPerformanceRow } from "@/lib/types";
import { Panel, PanelBody, PanelHeader } from "@/components/cold-start/ui";

// ── Types ──────────────────────────────────────────────────────────────────────

interface DrugSuggestion {
  drug_code: string;
  drug_name?: string;
  avg_smape: number | null;
  best_smape: number | null;
  has_drift: boolean;
  avg_coverage_90: number | null;
  demand_segment: string | null;
  data_quality_status: string | null;
  models_count: number;
  score: number;
  source: "performance" | "example";
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function computeScore(s: Omit<DrugSuggestion, "score">): number {
  let score = 50;
  if (s.avg_smape != null) {
    // Lower sMAPE → higher score. Under 10 % is excellent (+40), at 50 % it gives 0.
    score += Math.max(0, Math.min(40, 40 - (s.avg_smape - 10) * 0.8));
  }
  if (!s.has_drift) score += 15;
  if (s.avg_coverage_90 != null) {
    // Coverage 90 in [0,1]; reward > 0.8
    score += Math.max(0, (s.avg_coverage_90 - 0.5) * 30);
  }
  if (s.source === "example") score = Math.max(score, 60); // always show examples as decent
  return Math.round(score);
}

function buildSuggestionsFromPerformance(items: ModelPerformanceRow[]): DrugSuggestion[] {
  const drugMap = new Map<string, ModelPerformanceRow[]>();
  for (const row of items) {
    const existing = drugMap.get(row.drug_code) ?? [];
    existing.push(row);
    drugMap.set(row.drug_code, existing);
  }

  const derived: DrugSuggestion[] = [];
  for (const [drug_code, rows] of drugMap) {
    const smapes = rows.map((r) => r.smape).filter((s): s is number => s != null);
    const coverages = rows
      .map((r) => r.coverage_90)
      .filter((c): c is number => c != null);

    const avg_smape = smapes.length > 0 ? smapes.reduce((a, b) => a + b, 0) / smapes.length : null;
    const best_smape = smapes.length > 0 ? Math.min(...smapes) : null;
    const has_drift = rows.some((r) => r.drift_detected);
    const avg_coverage_90 =
      coverages.length > 0 ? coverages.reduce((a, b) => a + b, 0) / coverages.length : null;
    const demand_segment = rows[0]?.demand_segment ?? null;
    const data_quality_status = rows[0]?.data_quality_status ?? null;

    const base: Omit<DrugSuggestion, "score"> = {
      drug_code,
      avg_smape,
      best_smape,
      has_drift,
      avg_coverage_90,
      demand_segment,
      data_quality_status,
      models_count: rows.length,
      source: "performance",
    };
    derived.push({ ...base, score: computeScore(base) });
  }

  derived.sort((a, b) => b.score - a.score);
  return derived.slice(0, 8);
}

function buildExampleSuggestions(): DrugSuggestion[] {
  return EXAMPLE_FORECAST_DRUGS.map((d, i) => {
    const base: Omit<DrugSuggestion, "score"> = {
      drug_code: d.code,
      drug_name: d.name,
      avg_smape: null,
      best_smape: null,
      has_drift: false,
      avg_coverage_90: null,
      demand_segment: null,
      data_quality_status: null,
      models_count: 0,
      source: "example",
    };
    return { ...base, score: 90 - i * 5 };
  });
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function ScoreBadge({ score }: { score: number }) {
  let label: string;
  let cls: string;

  if (score >= 85) {
    label = "Top pick";
    cls = "bg-blue-600 text-white";
  } else if (score >= 70) {
    label = "Recommended";
    cls = "bg-emerald-600 text-white";
  } else if (score >= 55) {
    label = "Good";
    cls = "bg-amber-500 text-white";
  } else {
    label = "Suggested";
    cls = "bg-slate-400 text-white";
  }

  return (
    <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold leading-none ${cls}`}>
      {label}
    </span>
  );
}

function SmapeBadge({ smape }: { smape: number }) {
  let cls: string;
  let label: string;

  if (smape < 15) {
    cls = "bg-emerald-50 text-emerald-700 border border-emerald-200";
    label = "Excellent";
  } else if (smape < 30) {
    cls = "bg-amber-50 text-amber-700 border border-amber-200";
    label = "Good";
  } else if (smape < 50) {
    cls = "bg-orange-50 text-orange-700 border border-orange-200";
    label = "Fair";
  } else {
    cls = "bg-red-50 text-red-700 border border-red-200";
    label = "Poor";
  }

  return (
    <span className={`rounded-md px-1.5 py-0.5 text-[10px] font-medium ${cls}`}>
      {label} · {smape.toFixed(1)}% sMAPE
    </span>
  );
}

function SuggestionCard({
  suggestion,
  onSelect,
}: {
  suggestion: DrugSuggestion;
  onSelect: (code: string) => void;
}) {
  const isTop = suggestion.score >= 85;

  return (
    <div
      className={`flex flex-col gap-3 rounded-xl border p-4 transition ${
        isTop
          ? "border-blue-200 bg-blue-50/40 shadow-sm"
          : "border-slate-200 bg-white hover:border-slate-300"
      }`}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-slate-900">{suggestion.drug_code}</p>
          {suggestion.drug_name && (
            <p className="mt-0.5 truncate text-xs text-slate-500">{suggestion.drug_name}</p>
          )}
        </div>
        <ScoreBadge score={suggestion.score} />
      </div>

      {/* Metrics row */}
      <div className="flex flex-wrap gap-1.5">
        {suggestion.avg_smape != null ? (
          <SmapeBadge smape={suggestion.avg_smape} />
        ) : (
          <span className="rounded-md border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[10px] font-medium text-slate-500">
            Not yet trained
          </span>
        )}

        {suggestion.source === "performance" && (
          <>
            {suggestion.has_drift ? (
              <span className="inline-flex items-center gap-1 rounded-md border border-amber-200 bg-amber-50 px-1.5 py-0.5 text-[10px] font-medium text-amber-700">
                <AlertTriangle className="h-2.5 w-2.5" />
                Drift
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 rounded-md border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700">
                <CheckCircle2 className="h-2.5 w-2.5" />
                Stable
              </span>
            )}

            {suggestion.avg_coverage_90 != null && (
              <span className="rounded-md border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[10px] font-medium text-slate-600">
                {(suggestion.avg_coverage_90 * 100).toFixed(0)}% cov
              </span>
            )}

            {suggestion.demand_segment && (
              <span className="rounded-md border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[10px] font-medium text-slate-500 capitalize">
                {suggestion.demand_segment}
              </span>
            )}
          </>
        )}

        {suggestion.source === "example" && (
          <span className="rounded-md border border-blue-100 bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-600">
            Seed drug
          </span>
        )}
      </div>

      {/* CTA */}
      <button
        type="button"
        onClick={() => onSelect(suggestion.drug_code)}
        className="mt-auto inline-flex w-full items-center justify-center gap-1 rounded-lg border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700 transition hover:bg-blue-100 active:bg-blue-200"
      >
        Use for training
        <ChevronRight className="h-3 w-3" />
      </button>
    </div>
  );
}

// ── Why panel ──────────────────────────────────────────────────────────────────

function ScoringLegend() {
  const items = [
    {
      icon: <TrendingDown className="h-3.5 w-3.5 text-emerald-600" />,
      label: "Low sMAPE",
      detail: "< 15 % → Excellent accuracy signal",
    },
    {
      icon: <CheckCircle2 className="h-3.5 w-3.5 text-blue-600" />,
      label: "No drift",
      detail: "Stable demand over time",
    },
    {
      icon: <CheckCircle2 className="h-3.5 w-3.5 text-purple-600" />,
      label: "High coverage",
      detail: "≥ 80 % of actuals within P10–P90 band",
    },
    {
      icon: <Lightbulb className="h-3.5 w-3.5 text-amber-500" />,
      label: "Seed drugs",
      detail: "Known-good example drugs included by default",
    },
  ];

  return (
    <div className="rounded-xl border border-slate-100 bg-slate-50/80 px-4 py-3">
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        How scores are calculated
      </p>
      <ul className="space-y-1.5">
        {items.map(({ icon, label, detail }) => (
          <li key={label} className="flex items-start gap-2 text-xs text-slate-600">
            <span className="mt-0.5 shrink-0">{icon}</span>
            <span>
              <strong>{label}</strong> — {detail}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export function TrainingSuggestionsPanel() {
  const [suggestions, setSuggestions] = useState<DrugSuggestion[]>([]);
  const [loading, setLoading] = useState(true);
  const [hasPerformanceData, setHasPerformanceData] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const result = await fetchForecastingPerformance({ limit: 100 });
      if (result.items.length > 0) {
        setHasPerformanceData(true);
        setSuggestions(buildSuggestionsFromPerformance(result.items));
      } else {
        setHasPerformanceData(false);
        setSuggestions(buildExampleSuggestions());
      }
    } catch {
      setHasPerformanceData(false);
      setSuggestions(buildExampleSuggestions());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const handleSelect = useCallback((code: string) => {
    document
      .getElementById("forecasting-training-controls")
      ?.scrollIntoView({ behavior: "smooth" });
    // Copy drug code to clipboard for easy paste into the multi-select
    void navigator.clipboard.writeText(code).catch(() => undefined);
  }, []);

  return (
    <Panel>
      <PanelHeader
        title="Training suggestions"
        description={
          hasPerformanceData
            ? "Drugs ranked by predicted model accuracy based on historical sMAPE, drift, and demand coverage."
            : "Recommended seed drugs to start training. Scores update automatically after your first training run."
        }
        action={
          <button
            type="button"
            onClick={() => void load()}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
          >
            {loading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
            Refresh
          </button>
        }
      />

      <PanelBody className="space-y-5">
        {/* Contextual banner */}
        {!hasPerformanceData && !loading && (
          <div className="flex items-start gap-2 rounded-xl border border-blue-100 bg-blue-50/60 px-4 py-3">
            <Lightbulb className="mt-0.5 h-4 w-4 shrink-0 text-blue-500" />
            <p className="text-xs leading-relaxed text-blue-800/90">
              No training history yet. The drugs below are curated seed examples with stable,
              predictable demand — an ideal starting point for calibrated ensemble forecasting.
              Scores will reflect real accuracy metrics after your first training run.
            </p>
          </div>
        )}

        {loading ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div
                key={i}
                className="animate-pulse rounded-xl border border-slate-200 bg-slate-100 p-4 h-36"
              />
            ))}
          </div>
        ) : (
          <>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {suggestions.map((s) => (
                <SuggestionCard key={s.drug_code} suggestion={s} onSelect={handleSelect} />
              ))}
            </div>

            <ScoringLegend />

            <p className="text-center text-xs text-slate-400">
              {hasPerformanceData
                ? `Based on ${suggestions.reduce((t, s) => t + s.models_count, 0)} performance records across ${suggestions.length} drugs. Click "Use for training" to scroll to the training panel — the drug code is also copied to your clipboard.`
                : 'Click "Use for training" to scroll to the training panel — the drug code is copied to your clipboard for easy paste.'}
            </p>
          </>
        )}
      </PanelBody>
    </Panel>
  );
}
