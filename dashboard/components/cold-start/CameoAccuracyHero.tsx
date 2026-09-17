"use client";

import { AlertTriangle, CheckCircle2, Target, TrendingUp, XCircle } from "lucide-react";

import { accuracyAccent } from "@/lib/forecastMetrics";
import type { CameoAccuracySummary } from "@/lib/types";
import { StatChip } from "@/components/cold-start/ui";

interface CameoAccuracyHeroProps {
  summary: CameoAccuracySummary | null;
  drugsTrained?: number;
}

function HeroMetric({
  label,
  value,
  subtitle,
  met,
  featured,
}: {
  label: string;
  value: number | null | undefined;
  subtitle: string;
  met?: boolean;
  featured?: boolean;
}) {
  const display = value != null ? `${value.toFixed(1)}%` : "—";

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border p-5 ${
        featured
          ? "border-blue-200 bg-gradient-to-br from-blue-50/80 to-white shadow-sm"
          : "border-slate-200 bg-white"
      }`}
    >
      {featured && (
        <span className="absolute right-3 top-3 rounded-full bg-blue-600 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-white">
          Headline
        </span>
      )}
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">{label}</p>
          <p
            className={`mt-2 font-bold tabular-nums tracking-tight text-slate-900 ${
              featured ? "text-5xl" : "text-4xl"
            }`}
          >
            {display}
          </p>
          <p className="mt-1 text-sm text-slate-500">{subtitle}</p>
        </div>
        {met != null && value != null && (
          <span
            className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold ${
              met ? "bg-emerald-100 text-emerald-800" : "bg-slate-100 text-slate-600"
            }`}
          >
            {met ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
            {met ? "Beats baseline" : "Mixed"}
          </span>
        )}
      </div>
    </div>
  );
}

export function CameoAccuracyHero({ summary, drugsTrained }: CameoAccuracyHeroProps) {
  if (!summary || summary.n_coldstart_test_drugs === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/80 px-6 py-10 text-center">
        <Target className="mx-auto h-8 w-8 text-slate-400" />
        <p className="mt-3 text-sm font-semibold text-slate-800">No validation metrics yet</p>
        <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
          Seed the drugs library and enriched demand, then train CAMEO to run the leave-drugs-out
          hold-out evaluation from the real-data notebook.
        </p>
        {summary?.note && (
          <p className="mx-auto mt-3 max-w-md text-xs text-amber-700">{summary.note}</p>
        )}
      </div>
    );
  }

  const cameoPooled = summary.pooled_accuracy_pct.CAMEO;
  const analogousPooled = summary.pooled_accuracy_pct.Analogous;
  const cameoMedian = summary.median_accuracy_pct.CAMEO;
  const cameoWinRate = summary.win_rate_pct.CAMEO;
  const analogousWinRate = summary.win_rate_pct.Analogous;
  const operationalPooled = summary.operational_pooled_accuracy_pct;
  const nonLumpyOperational = summary.non_lumpy_operational_pooled_accuracy_pct;
  const smoothPooled = summary.smooth_pooled_accuracy_pct;
  const headlineAccuracy =
    nonLumpyOperational ?? smoothPooled ?? operationalPooled ?? cameoPooled;
  const sbEntries = Object.entries(summary.per_sb_class_pooled_accuracy_pct ?? {});

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <HeroMetric
          label="CAMEO procurement accuracy"
          value={headlineAccuracy}
          subtitle="Smooth + erratic held-out drugs with meaningful launch volume (≥5 units / 20 weeks)"
          met={headlineAccuracy != null && headlineAccuracy >= 70}
          featured
        />
        <HeroMetric
          label="CAMEO win rate"
          value={cameoWinRate}
          subtitle="Share of held-out drugs where CAMEO beats the analogous baseline"
          met={cameoWinRate != null && analogousWinRate != null && cameoWinRate >= analogousWinRate}
        />
      </div>

      <div className="rounded-xl border border-amber-200 bg-amber-50/70 px-4 py-3 text-sm text-amber-950">
        <div className="flex items-start gap-2">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <p>
            All-drugs pooled accuracy ({cameoPooled?.toFixed(1) ?? "—"}%) includes near-dormant launch
            windows (1–2 units over 20 weeks) where WAPE is unstable. Operational accuracy focuses on
            SKUs with meaningful launch volume.
            {smoothPooled != null && (
              <>
                {" "}
                Smooth-class pooled: <strong>{smoothPooled.toFixed(1)}%</strong>.
              </>
            )}
            {(summary.zero_demand_test_drugs ?? 0) > 0 && (
              <>
                {" "}
                {summary.zero_demand_test_drugs} test drug(s) had zero demand in the hold-out window.
              </>
            )}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatChip
          label="All-drugs pooled"
          value={cameoPooled != null ? `${cameoPooled.toFixed(1)}%` : "—"}
          accent="slate"
        />
        <StatChip
          label="Analogous pooled"
          value={analogousPooled != null ? `${analogousPooled.toFixed(1)}%` : "—"}
          accent="slate"
        />
        <StatChip
          label="Median per drug"
          value={cameoMedian != null ? `${cameoMedian.toFixed(1)}%` : "—"}
          accent={cameoMedian != null ? accuracyAccent(cameoMedian) : "slate"}
        />
        <StatChip
          label="Held-out test drugs"
          value={String(summary.n_coldstart_test_drugs)}
          accent="slate"
        />
        <StatChip
          label="Library drugs"
          value={String(drugsTrained ?? summary.n_historical_library_drugs)}
          accent="slate"
        />
      </div>

      {sbEntries.length > 0 && (
        <div className="rounded-2xl border border-slate-200 bg-white p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-800">
            <TrendingUp className="h-4 w-4 text-blue-600" />
            Pooled accuracy by Syntetos–Boylan class
          </div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {sbEntries.map(([segment, pct]) => (
              <div
                key={segment}
                className="flex items-center justify-between rounded-xl bg-slate-50 px-3 py-2.5 ring-1 ring-slate-100"
              >
                <span className="text-sm capitalize text-slate-600">{segment}</span>
                <span
                  className={`text-sm font-bold tabular-nums ${
                    pct >= 50
                      ? "text-emerald-700"
                      : pct >= 30
                        ? "text-blue-700"
                        : "text-slate-700"
                  }`}
                >
                  {pct.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
