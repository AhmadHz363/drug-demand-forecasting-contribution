"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, Loader2, RefreshCw } from "lucide-react";

import { fetchForecastingPerformance } from "@/lib/api";
import type { ModelPerformanceRow } from "@/lib/types";
import { Panel, PanelBody, PanelHeader } from "@/components/cold-start/ui";

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

function DriftBadge({ row }: { row: ModelPerformanceRow }) {
  if (!row.drift_detected) {
    return (
      <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-700">
        Stable
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-800">
      <AlertTriangle className="h-3 w-3" />
      Drift
    </span>
  );
}

export function ModelHealthMonitoringPanel() {
  const [rows, setRows] = useState<ModelPerformanceRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchForecastingPerformance({ limit: 50 });
      setRows(result.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load performance data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const driftCount = rows.filter((r) => r.drift_detected).length;

  return (
    <Panel id="forecasting-health-panel">
      <PanelHeader
        title="Model health monitoring"
        description="Walk-forward sMAPE/MASE, coverage, and drift vs prior training runs."
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
      <PanelBody className="space-y-4">
        {error && (
          <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </p>
        )}

        {!error && !loading && rows.length === 0 && (
          <p className="text-sm text-slate-500">
            No training metrics yet. Run model training to populate performance history.
          </p>
        )}

        {rows.length > 0 && (
          <>
            <div className="flex flex-wrap gap-3 text-xs text-slate-600">
              <span className="rounded-lg bg-slate-100 px-2.5 py-1">
                {rows.length} recent records
              </span>
              {driftCount > 0 && (
                <span className="rounded-lg bg-amber-50 px-2.5 py-1 text-amber-800">
                  {driftCount} with metric drift
                </span>
              )}
            </div>

            <div className="overflow-x-auto rounded-xl border border-slate-200">
              <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
                <thead className="bg-slate-50 text-slate-500">
                  <tr>
                    <th className="px-3 py-2 font-medium">Drug</th>
                    <th className="px-3 py-2 font-medium">Model</th>
                    <th className="px-3 py-2 font-medium">sMAPE</th>
                    <th className="px-3 py-2 font-medium">MASE</th>
                    <th className="px-3 py-2 font-medium">Coverage</th>
                    <th className="px-3 py-2 font-medium">Segment</th>
                    <th className="px-3 py-2 font-medium">Drift</th>
                    <th className="px-3 py-2 font-medium">Evaluated</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white text-slate-800">
                  {rows.map((row) => (
                    <tr key={`${row.drug_code}-${row.model_name}-${row.evaluated_at}`}>
                      <td className="px-3 py-2 font-medium">{row.drug_code}</td>
                      <td className="px-3 py-2 uppercase">{row.model_name}</td>
                      <td className="px-3 py-2">{formatPct(row.smape)}</td>
                      <td className="px-3 py-2">
                        {row.mase != null ? row.mase.toFixed(3) : "—"}
                      </td>
                      <td className="px-3 py-2">{formatPct(row.coverage_90 * 100)}</td>
                      <td className="px-3 py-2">{row.demand_segment ?? "—"}</td>
                      <td className="px-3 py-2">
                        <DriftBadge row={row} />
                      </td>
                      <td className="px-3 py-2 text-slate-500">{formatDate(row.evaluated_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </PanelBody>
    </Panel>
  );
}
