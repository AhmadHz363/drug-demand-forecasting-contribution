"use client";

import { Activity, Loader2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { getDrugDetail } from "@/lib/api";
import type { DrugDetailResponse, DrugItem } from "@/lib/types";

import { DrugUsageChart } from "./DrugUsageChart";

interface DrugUsageStatsModalProps {
  drug: DrugItem;
  onClose: () => void;
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function DrugUsageStatsModal({ drug, onClose }: DrugUsageStatsModalProps) {
  const [detail, setDetail] = useState<DrugDetailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const loadUsage = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await getDrugDetail(drug.drug_code, 3650);
      setDetail(result);
    } catch (err) {
      setDetail(null);
      setError(err instanceof Error ? err.message : "Failed to load usage history");
    } finally {
      setIsLoading(false);
    }
  }, [drug.drug_code]);

  useEffect(() => {
    void loadUsage();
  }, [loadUsage]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [onClose]);

  const title = drug.drug_name?.trim() || drug.drug_code;

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center bg-slate-900/40 p-0 backdrop-blur-sm sm:items-center sm:p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="drug-usage-title"
      onClick={onClose}
    >
      <div
        className="flex max-h-[92vh] w-full max-w-5xl flex-col overflow-hidden rounded-t-2xl border border-slate-200 bg-white shadow-2xl sm:rounded-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b border-slate-100 px-5 py-4 sm:px-6">
          <div className="min-w-0">
            <div className="mb-1 flex items-center gap-2 text-xs text-slate-500">
              <Activity className="h-3.5 w-3.5" aria-hidden="true" />
              <span className="font-mono">{drug.drug_code}</span>
            </div>
            <h2 id="drug-usage-title" className="truncate text-lg font-semibold text-slate-900">
              {title} — usage & gaps
            </h2>
            <p className="mt-0.5 text-sm text-slate-500">
              {drug.distinct_receipt_days.toLocaleString()} day
              {drug.distinct_receipt_days === 1 ? "" : "s"} with receipts
              {drug.first_receipt_date && drug.last_receipt_date
                ? ` · ${formatDate(drug.first_receipt_date)} → ${formatDate(drug.last_receipt_date)}`
                : ""}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-slate-200 p-2 text-slate-500 transition hover:bg-slate-50 hover:text-slate-700"
            aria-label="Close usage stats"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="overflow-y-auto px-5 py-5 sm:px-6">
          {isLoading ? (
            <div className="flex items-center justify-center gap-2 py-16 text-sm text-slate-500">
              <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
              Loading usage history…
            </div>
          ) : error ? (
            <div className="space-y-4">
              <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
              <button
                type="button"
                onClick={() => void loadUsage()}
                className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
              >
                Retry
              </button>
            </div>
          ) : detail ? (
            <DrugUsageChart
              series={detail.demand_series}
              firstReceiptDate={detail.first_receipt_date}
              lastReceiptDate={detail.last_receipt_date}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}
