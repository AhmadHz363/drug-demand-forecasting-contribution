"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Loader2, RefreshCw, Search } from "lucide-react";

import { fetchForecastingPerformance } from "@/lib/api";
import { accuracyAccent, performanceRowAccuracy } from "@/lib/forecastMetrics";
import type { ModelPerformanceRow } from "@/lib/types";

interface PerDrugAccuracyTableProps {
  trainingRunId?: string | null;
  selectedDrugCode?: string | null;
  onSelectDrug?: (code: string) => void;
}

function formatPct(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${value.toFixed(1)}%`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function PerDrugAccuracyTable({
  trainingRunId,
  selectedDrugCode,
  onSelectDrug,
}: PerDrugAccuracyTableProps) {
  const [rows, setRows] = useState<ModelPerformanceRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchForecastingPerformance({
        model_name: "shield_xr_ensemble",
        training_run_id: trainingRunId ?? undefined,
        limit: 500,
      });
      setRows(result.items);
    } catch (err) {
      setRows([]);
      setError(err instanceof Error ? err.message : "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }, [trainingRunId]);

  useEffect(() => {
    if (trainingRunId) void load();
  }, [load, trainingRunId]);

  const latestByDrug = useMemo(() => {
    const map = new Map<string, ModelPerformanceRow>();
    for (const row of rows) {
      const existing = map.get(row.drug_code);
      if (!existing || new Date(row.evaluated_at) > new Date(existing.evaluated_at)) {
        map.set(row.drug_code, row);
      }
    }
    return [...map.values()].sort((a, b) => a.drug_code.localeCompare(b.drug_code));
  }, [rows]);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return latestByDrug;
    return latestByDrug.filter((r) => r.drug_code.toLowerCase().includes(q));
  }, [filter, latestByDrug]);

  const above70 = latestByDrug.filter((r) => (performanceRowAccuracy(r) ?? 0) >= 70).length;
  const topPerformers = useMemo(() => {
    return [...latestByDrug]
      .map((row) => ({ row, accuracy: performanceRowAccuracy(row) }))
      .filter((entry) => entry.accuracy != null)
      .sort((a, b) => (b.accuracy ?? 0) - (a.accuracy ?? 0))
      .slice(0, 5);
  }, [latestByDrug]);

  if (!trainingRunId) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50/80 px-4 py-8 text-center text-sm text-slate-500">
        Per-drug weekly accuracy appears after training completes.
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-4 py-3 sm:px-5">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">Per-drug weekly accuracy</h3>
          <p className="text-xs text-slate-500">
            Reconciled breakdown · {above70}/{latestByDrug.length} drugs ≥70%
          </p>
        </div>
        <button
          type="button"
          onClick={() => void load()}
          disabled={loading}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
        >
          {loading ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <RefreshCw className="h-3.5 w-3.5" />
          )}
          Refresh
        </button>
      </div>

      {topPerformers.length > 0 && (
        <div className="border-b border-slate-100 px-4 py-3 sm:px-5">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Top performers (weekly reconciled)
          </p>
          <div className="flex flex-wrap gap-2">
            {topPerformers.map(({ row, accuracy }) => (
              <button
                key={row.drug_code}
                type="button"
                onClick={() => onSelectDrug?.(row.drug_code)}
                className="inline-flex items-center gap-2 rounded-lg bg-emerald-50 px-3 py-1.5 text-xs ring-1 ring-emerald-100 transition hover:bg-emerald-100"
              >
                <span className="font-medium text-slate-800">{row.drug_code}</span>
                <span className="font-bold tabular-nums text-emerald-700">
                  {accuracy!.toFixed(1)}%
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="border-b border-slate-100 px-4 py-2 sm:px-5">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            type="search"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter drug codes…"
            className="w-full rounded-lg border border-slate-200 py-2 pl-9 pr-3 text-sm"
          />
        </div>
      </div>

      {error && (
        <p className="px-4 py-3 text-xs text-amber-800 sm:px-5">{error}</p>
      )}

      <div className="max-h-80 overflow-auto">
        <table className="min-w-full divide-y divide-slate-100 text-left text-xs">
          <thead className="sticky top-0 bg-slate-50 text-slate-500">
            <tr>
              <th className="px-4 py-2 font-medium sm:px-5">Drug</th>
              <th className="px-4 py-2 font-medium">Weekly acc.</th>
              <th className="px-4 py-2 font-medium">Segment</th>
              <th className="hidden px-4 py-2 font-medium sm:table-cell">Evaluated</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50 bg-white text-slate-800">
            {loading && filtered.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center text-slate-400 sm:px-5">
                  <Loader2 className="mx-auto h-5 w-5 animate-spin" />
                </td>
              </tr>
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center text-slate-400 sm:px-5">
                  No drugs match your filter.
                </td>
              </tr>
            ) : (
              filtered.map((row) => {
                const accuracy = performanceRowAccuracy(row);
                const isSelected = row.drug_code === selectedDrugCode;
                const accent = accuracy != null ? accuracyAccent(accuracy) : "slate";
                const accentClass =
                  accent === "green"
                    ? "text-emerald-700 font-semibold"
                    : accent === "blue"
                      ? "text-blue-700 font-semibold"
                      : accent === "amber"
                        ? "text-amber-700"
                        : "";

                return (
                  <tr
                    key={`${row.drug_code}-${row.evaluated_at}`}
                    onClick={() => onSelectDrug?.(row.drug_code)}
                    className={`${onSelectDrug ? "cursor-pointer hover:bg-slate-50" : ""} ${
                      isSelected ? "bg-blue-50/70" : ""
                    }`}
                  >
                    <td className="px-4 py-2.5 font-medium sm:px-5">{row.drug_code}</td>
                    <td className={`px-4 py-2.5 tabular-nums ${accentClass}`}>
                      {formatPct(accuracy)}
                    </td>
                    <td className="px-4 py-2.5 capitalize text-slate-600">
                      {row.demand_segment ?? "—"}
                    </td>
                    <td className="hidden px-4 py-2.5 text-slate-500 sm:table-cell">
                      {formatDate(row.evaluated_at)}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
