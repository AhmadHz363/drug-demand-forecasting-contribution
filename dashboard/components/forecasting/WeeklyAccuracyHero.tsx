"use client";

import { CheckCircle2, Target, TrendingUp, XCircle } from "lucide-react";

import {
  meetsPerDrugAccuracyTarget,
  meetsWeeklyAccuracyTarget,
} from "@/lib/forecastMetrics";
import { SHIELD_XR_BENCHMARKS } from "@/lib/shieldXrArchitecture";
import type { ShieldXRAccuracySummary } from "@/lib/types";
import { StatChip } from "@/components/cold-start/ui";

interface WeeklyAccuracyHeroProps {
  summary: ShieldXRAccuracySummary | null;
  drugsTrained?: number;
  trainingRunId?: string | null;
}

function HeroMetric({
  label,
  value,
  target,
  benchmark,
  met,
  featured,
}: {
  label: string;
  value: number | null | undefined;
  target: number;
  benchmark: number;
  met: boolean;
  featured?: boolean;
}) {
  const display = value != null ? `${value.toFixed(1)}%` : "—";

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border p-5 ${
        featured
          ? met
            ? "border-emerald-300 bg-gradient-to-br from-emerald-50 via-white to-blue-50/40 shadow-sm"
            : "border-blue-200 bg-gradient-to-br from-blue-50/80 to-white shadow-sm"
          : met
            ? "border-emerald-200 bg-gradient-to-br from-emerald-50 to-white"
            : "border-amber-200 bg-gradient-to-br from-amber-50 to-white"
      }`}
    >
      {featured && (
        <span className="absolute right-3 top-3 rounded-full bg-blue-600 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-white">
          Primary KPI
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
          <p className="mt-1 text-sm text-slate-500">
            Target ≥{target}% · Thesis {benchmark.toFixed(1)}%
          </p>
        </div>
        {value != null && (
          <span
            className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold ${
              met ? "bg-emerald-100 text-emerald-800" : "bg-amber-100 text-amber-800"
            }`}
          >
            {met ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
            {met ? "On target" : "Below target"}
          </span>
        )}
      </div>
      {value != null && (
        <div className="mt-4 h-2.5 overflow-hidden rounded-full bg-slate-200/80">
          <div
            className={`h-full rounded-full transition-all ${met ? "bg-emerald-500" : "bg-blue-500"}`}
            style={{ width: `${Math.min(100, value)}%` }}
          />
        </div>
      )}
    </div>
  );
}

export function WeeklyAccuracyHero({ summary, drugsTrained, trainingRunId }: WeeklyAccuracyHeroProps) {
  if (!summary) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/80 px-6 py-10 text-center">
        <Target className="mx-auto h-8 w-8 text-slate-400" />
        <p className="mt-3 text-sm font-semibold text-slate-800">No validation metrics yet</p>
        <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
          Upload hospital Excel data via Data Ingestion, then train SHIELD-XR to evaluate
          leakage-free hold-out accuracy on hospital weekly totals and reconciled per-drug
          breakdown.
        </p>
      </div>
    );
  }

  const weeklyMet = meetsWeeklyAccuracyTarget(summary);
  const perDrugMet = meetsPerDrugAccuracyTarget(summary);
  const sbEntries = Object.entries(summary.per_sb_class_weekly_accuracy_pct ?? {});

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <HeroMetric
          label="Hospital weekly accuracy"
          value={summary.hospital_weekly_accuracy_pct}
          target={90}
          benchmark={SHIELD_XR_BENCHMARKS.hospitalWeeklyAccuracy}
          met={weeklyMet}
          featured
        />
        <HeroMetric
          label="Non-lumpy weekly (reconciled)"
          value={summary.non_lumpy_weekly_accuracy_pct ?? summary.reconciled_weekly_per_drug_mean_accuracy_pct}
          target={70}
          benchmark={SHIELD_XR_BENCHMARKS.nonLumpySubsetAccuracy}
          met={perDrugMet}
        />
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatChip
          label="Per-drug median"
          value={
            summary.reconciled_weekly_per_drug_median_accuracy_pct != null
              ? `${summary.reconciled_weekly_per_drug_median_accuracy_pct.toFixed(1)}%`
              : "—"
          }
          accent="blue"
        />
        <StatChip
          label="Daily ensemble"
          value={
            summary.daily_ensemble_accuracy_pct != null
              ? `${summary.daily_ensemble_accuracy_pct.toFixed(1)}%`
              : "—"
          }
          accent="slate"
        />
        <StatChip
          label="SKUs trained"
          value={drugsTrained != null ? String(drugsTrained) : "—"}
          accent="slate"
        />
        <StatChip
          label="Unweighted SKU mean"
          value={
            summary.reconciled_weekly_per_drug_mean_accuracy_pct != null
              ? `${summary.reconciled_weekly_per_drug_mean_accuracy_pct.toFixed(1)}%`
              : "—"
          }
          accent="slate"
        />
        <StatChip
          label="Hybrid ABC (thesis)"
          value={
            summary.hybrid_abc_combined_accuracy_pct != null
              ? `${summary.hybrid_abc_combined_accuracy_pct.toFixed(1)}%`
              : `${SHIELD_XR_BENCHMARKS.hybridAbcCombinedAccuracy}%`
          }
          accent="green"
        />
      </div>

      {sbEntries.length > 0 && (
        <div className="rounded-2xl border border-slate-200 bg-white p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-800">
            <TrendingUp className="h-4 w-4 text-blue-600" />
            Weekly accuracy by Syntetos–Boylan class
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
                    pct >= 70
                      ? "text-emerald-700"
                      : pct >= 55
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
