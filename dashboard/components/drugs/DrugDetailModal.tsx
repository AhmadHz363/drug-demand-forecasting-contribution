"use client";

import { Loader2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { GraduationBadge } from "@/components/cold-start/GraduationBadge";
import { getDrugDetail } from "@/lib/api";
import type { DrugDetailResponse, DrugItem } from "@/lib/types";

import { DemandHistoryChart } from "./DemandHistoryChart";

interface DrugDetailModalProps {
  drug: DrugItem;
  onClose: () => void;
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function formatDateTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function SpecRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-100 bg-slate-50/80 px-3 py-2.5">
      <dt className="text-xs font-medium text-slate-500">{label}</dt>
      <dd className="mt-0.5 text-sm font-medium text-slate-900">{value}</dd>
    </div>
  );
}

export function DrugDetailModal({ drug, onClose }: DrugDetailModalProps) {
  const [detail, setDetail] = useState<DrugDetailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const loadDetail = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await getDrugDetail(drug.drug_code);
      setDetail(result);
    } catch (err) {
      setDetail(null);
      setError(err instanceof Error ? err.message : "Failed to load drug details");
    } finally {
      setIsLoading(false);
    }
  }, [drug.drug_code]);

  useEffect(() => {
    void loadDetail();
  }, [loadDetail]);

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
      aria-labelledby="drug-detail-title"
      onClick={onClose}
    >
      <div
        className="flex max-h-[92vh] w-full max-w-4xl flex-col overflow-hidden rounded-t-2xl border border-slate-200 bg-white shadow-2xl sm:rounded-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b border-slate-100 px-5 py-4 sm:px-6">
          <div className="min-w-0">
            <p className="font-mono text-xs text-slate-500">{drug.drug_code}</p>
            <h2 id="drug-detail-title" className="truncate text-lg font-semibold text-slate-900">
              {title}
            </h2>
            {drug.drug_category && (
              <p className="mt-0.5 text-sm text-slate-500">{drug.drug_category}</p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-slate-200 p-2 text-slate-500 transition hover:bg-slate-50 hover:text-slate-700"
            aria-label="Close drug details"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="overflow-y-auto px-5 py-5 sm:px-6">
          {isLoading ? (
            <div className="flex items-center justify-center gap-2 py-16 text-sm text-slate-500">
              <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
              Loading drug details…
            </div>
          ) : error ? (
            <div className="space-y-4">
              <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
              <button
                type="button"
                onClick={() => void loadDetail()}
                className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
              >
                Retry
              </button>
            </div>
          ) : detail ? (
            <div className="space-y-6">
              <section>
                <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
                  Registry
                </h3>
                <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  <SpecRow label="Receipt lines" value={detail.receipt_count.toLocaleString()} />
                  <SpecRow
                    label="Distinct receipt days"
                    value={detail.distinct_receipt_days.toLocaleString()}
                  />
                  <SpecRow label="Centers" value={detail.center_count.toLocaleString()} />
                  <SpecRow
                    label="Total quantity"
                    value={`${detail.total_quantity.toLocaleString(undefined, {
                      maximumFractionDigits: 1,
                    })} units`}
                  />
                  <SpecRow
                    label="Avg daily quantity"
                    value={
                      detail.avg_daily_quantity != null
                        ? `${detail.avg_daily_quantity.toLocaleString(undefined, {
                            maximumFractionDigits: 2,
                          })} units`
                        : "—"
                    }
                  />
                  <SpecRow
                    label="First receipt"
                    value={formatDate(detail.first_receipt_date)}
                  />
                  <SpecRow
                    label="Last receipt"
                    value={formatDate(detail.last_receipt_date)}
                  />
                  <SpecRow label="First seen" value={formatDateTime(detail.created_at)} />
                  <SpecRow label="Last updated" value={formatDateTime(detail.updated_at)} />
                </dl>
              </section>

              <section className="rounded-xl border border-slate-200 bg-white p-4">
                <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
                  Forecasting readiness
                </h3>
                <GraduationBadge
                  embedded
                  stage={detail.graduation_stage}
                  observationCount={detail.observation_count}
                />
              </section>

              <DemandHistoryChart
                embedded
                series={detail.demand_series}
                lookbackDays={detail.lookback_days}
              />
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
