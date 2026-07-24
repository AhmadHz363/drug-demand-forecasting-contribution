"use client";

import { BarChart3, Pill, Search } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { Panel, PanelBody, PanelHeader, inputClass } from "@/components/cold-start/ui";
import { DrugDetailModal } from "@/components/drugs/DrugDetailModal";
import { DrugUsageStatsModal } from "@/components/drugs/DrugUsageStatsModal";
import { listDrugs } from "@/lib/api";
import type { DrugItem, PaginatedDrugListResponse } from "@/lib/types";

const PAGE_SIZE = 20;

function formatDateRange(first?: string | null, last?: string | null): string {
  if (!first && !last) return "—";
  const fmt = (iso: string) =>
    new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  if (first && last) return `${fmt(first)} → ${fmt(last)}`;
  return fmt(first ?? last ?? "");
}

export function DrugsDashboard() {
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [page, setPage] = useState(1);
  const [data, setData] = useState<PaginatedDrugListResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedDrug, setSelectedDrug] = useState<DrugItem | null>(null);
  const [usageDrug, setUsageDrug] = useState<DrugItem | null>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedQuery(query.trim());
      setPage(1);
    }, 300);
    return () => window.clearTimeout(timer);
  }, [query]);

  const loadDrugs = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await listDrugs(debouncedQuery || undefined, page, PAGE_SIZE);
      setData(result);
    } catch (err) {
      setData(null);
      setError(err instanceof Error ? err.message : "Failed to load drugs");
    } finally {
      setIsLoading(false);
    }
  }, [debouncedQuery, page]);

  useEffect(() => {
    void loadDrugs();
  }, [loadDrugs]);

  const totalPages = data?.total_pages ?? 0;

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-4 py-8 sm:px-6 lg:px-8">
      <header className="flex items-start gap-4">
        <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-blue-600 text-white shadow-sm">
          <Pill className="h-6 w-6" />
        </span>
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Drug Registry</h1>
          <p className="mt-1 max-w-2xl text-sm text-slate-500">
            Unique drugs collected from receipt ingestion, with receipt counts and metadata.
          </p>
        </div>
      </header>

      <Panel>
        <PanelHeader
          title="Known drugs"
          description="Search by code, name, or category. Open details for specs, or usage stats to inspect daily quantities and data gaps."
          action={
            data ? (
              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
                {data.total.toLocaleString()} unique drug{data.total === 1 ? "" : "s"}
              </span>
            ) : null
          }
        />
        <PanelBody className="space-y-4">
          <div className="relative max-w-md">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search drugs…"
              className={`${inputClass} pl-9`}
              aria-label="Search drugs"
            />
          </div>

          {error && (
            <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}

          <div className="overflow-x-auto rounded-xl border border-slate-200">
            <table className="min-w-full divide-y divide-slate-200 text-sm">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold text-slate-700">Code</th>
                  <th className="px-4 py-3 text-left font-semibold text-slate-700">Name</th>
                  <th className="px-4 py-3 text-left font-semibold text-slate-700">Category</th>
                  <th className="px-4 py-3 text-right font-semibold text-slate-700">Data days</th>
                  <th className="px-4 py-3 text-left font-semibold text-slate-700">Date range</th>
                  <th className="px-4 py-3 text-right font-semibold text-slate-700">Receipts</th>
                  <th className="px-4 py-3 text-right font-semibold text-slate-700">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {isLoading ? (
                  <tr>
                    <td colSpan={7} className="px-4 py-8 text-center text-slate-500">
                      Loading drugs…
                    </td>
                  </tr>
                ) : !data?.items.length ? (
                  <tr>
                    <td colSpan={7} className="px-4 py-8 text-center text-slate-500">
                      {debouncedQuery ? "No drugs match your search." : "No drugs ingested yet."}
                    </td>
                  </tr>
                ) : (
                  data.items.map((drug: DrugItem) => (
                    <tr key={drug.id} className="hover:bg-slate-50/80">
                      <td className="whitespace-nowrap px-4 py-3 font-mono text-xs text-slate-900">
                        {drug.drug_code}
                      </td>
                      <td className="max-w-md truncate px-4 py-3 text-slate-800">
                        {drug.drug_name ?? "—"}
                      </td>
                      <td className="whitespace-nowrap px-4 py-3 text-slate-600">
                        {drug.drug_category ?? "—"}
                      </td>
                      <td className="whitespace-nowrap px-4 py-3 text-right tabular-nums text-slate-900">
                        {drug.distinct_receipt_days.toLocaleString()}
                      </td>
                      <td className="whitespace-nowrap px-4 py-3 text-xs text-slate-600">
                        {formatDateRange(drug.first_receipt_date, drug.last_receipt_date)}
                      </td>
                      <td className="whitespace-nowrap px-4 py-3 text-right tabular-nums text-slate-900">
                        {drug.receipt_count.toLocaleString()}
                      </td>
                      <td className="whitespace-nowrap px-4 py-3 text-right">
                        <div className="flex justify-end gap-2">
                          <button
                            type="button"
                            onClick={() => setSelectedDrug(drug)}
                            className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 transition hover:bg-slate-50"
                          >
                            Details
                          </button>
                          <button
                            type="button"
                            onClick={() => setUsageDrug(drug)}
                            className="inline-flex items-center gap-1 rounded-lg border border-blue-200 bg-blue-50 px-2.5 py-1 text-xs font-medium text-blue-700 transition hover:bg-blue-100"
                          >
                            <BarChart3 className="h-3.5 w-3.5" aria-hidden="true" />
                            Usage
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
              <p className="text-xs text-slate-500">
                Page {page} of {totalPages}
              </p>
              <div className="flex gap-2">
                <button
                  type="button"
                  disabled={page <= 1 || isLoading}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Previous
                </button>
                <button
                  type="button"
                  disabled={page >= totalPages || isLoading}
                  onClick={() => setPage((p) => p + 1)}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </PanelBody>
      </Panel>

      {selectedDrug && (
        <DrugDetailModal drug={selectedDrug} onClose={() => setSelectedDrug(null)} />
      )}

      {usageDrug && (
        <DrugUsageStatsModal drug={usageDrug} onClose={() => setUsageDrug(null)} />
      )}
    </div>
  );
}
