"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  Activity,
  BarChart2,
  CheckCircle2,
  ClipboardList,
  Database,
  FlaskConical,
  LayoutDashboard,
  LineChart,
  RefreshCw,
  Tags,
  TrendingUp,
  Upload,
  XCircle,
} from "lucide-react";

import { Panel, PanelBody, PanelHeader, Skeleton } from "@/components/cold-start/ui";
import { GraduationDonutChart } from "@/components/overview/charts/GraduationDonutChart";
import { ModelPerformanceChart } from "@/components/overview/charts/ModelPerformanceChart";
import { RankedBarChart } from "@/components/overview/charts/RankedBarChart";
import { ReceiptTrendChart } from "@/components/overview/charts/ReceiptTrendChart";
import { fetchOverview } from "@/lib/api";
import type { OverviewResponse } from "@/lib/types";

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmt(n: number) {
  return n.toLocaleString();
}

function fmtDate(d: string | null | undefined) {
  if (!d) return "—";
  return new Date(d).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function fmtPct(v: number | null | undefined) {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtSmape(v: number | null | undefined) {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

// ── Sub-components ────────────────────────────────────────────────────────────

function HeroKpi({
  label,
  value,
  icon: Icon,
  accent,
  href,
}: {
  label: string;
  value: string | number;
  icon: React.ElementType;
  accent: "blue" | "violet" | "emerald" | "amber" | "rose";
  href: string;
}) {
  const palette = {
    blue: {
      wrap: "bg-blue-50 border-blue-100",
      icon: "bg-blue-600 text-white",
      value: "text-blue-900",
      label: "text-blue-600",
    },
    violet: {
      wrap: "bg-violet-50 border-violet-100",
      icon: "bg-violet-600 text-white",
      value: "text-violet-900",
      label: "text-violet-600",
    },
    emerald: {
      wrap: "bg-emerald-50 border-emerald-100",
      icon: "bg-emerald-600 text-white",
      value: "text-emerald-900",
      label: "text-emerald-600",
    },
    amber: {
      wrap: "bg-amber-50 border-amber-100",
      icon: "bg-amber-500 text-white",
      value: "text-amber-900",
      label: "text-amber-600",
    },
    rose: {
      wrap: "bg-rose-50 border-rose-100",
      icon: "bg-rose-600 text-white",
      value: "text-rose-900",
      label: "text-rose-600",
    },
  }[accent];

  return (
    <Link
      href={href}
      className={`flex items-center gap-4 rounded-2xl border p-5 transition hover:shadow-md ${palette.wrap}`}
    >
      <span
        className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl shadow-sm ${palette.icon}`}
      >
        <Icon className="h-6 w-6" />
      </span>
      <div>
        <p className={`text-2xl font-bold tabular-nums ${palette.value}`}>{value}</p>
        <p className={`text-xs font-medium ${palette.label}`}>{label}</p>
      </div>
    </Link>
  );
}

function ChartPanel({
  title,
  description,
  children,
  className = "",
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Panel className={className}>
      <PanelHeader title={title} description={description} />
      <PanelBody>{children}</PanelBody>
    </Panel>
  );
}

function SectionCard({
  title,
  icon: Icon,
  href,
  children,
}: {
  title: string;
  icon: React.ElementType;
  href: string;
  children: React.ReactNode;
}) {
  return (
    <Panel>
      <PanelHeader
        title={title}
        action={
          <Link
            href={href}
            className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:bg-slate-50 hover:text-slate-900"
          >
            <Icon className="h-3.5 w-3.5" />
            Open
          </Link>
        }
      />
      <PanelBody>{children}</PanelBody>
    </Panel>
  );
}

function StatusBadge({ ok, labelOk, labelNo }: { ok: boolean; labelOk: string; labelNo: string }) {
  return ok ? (
    <span className="flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700 ring-1 ring-emerald-100">
      <CheckCircle2 className="h-3.5 w-3.5" />
      {labelOk}
    </span>
  ) : (
    <span className="flex items-center gap-1.5 rounded-full bg-rose-50 px-3 py-1 text-xs font-semibold text-rose-600 ring-1 ring-rose-100">
      <XCircle className="h-3.5 w-3.5" />
      {labelNo}
    </span>
  );
}

function KpiRow({
  items,
}: {
  items: { label: string; value: string | number }[];
}) {
  return (
    <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      {items.map(({ label, value }) => (
        <div key={label} className="rounded-xl bg-slate-50 px-4 py-3">
          <dt className="text-[11px] font-medium uppercase tracking-wide text-slate-500">{label}</dt>
          <dd className="mt-0.5 text-base font-semibold tabular-nums text-slate-900">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function OverviewDashboard() {
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await fetchOverview();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load overview");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  return (
    <div className="mx-auto max-w-6xl space-y-8 px-4 py-8 sm:px-6 lg:px-8">
      {/* Page header */}
      <header className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-4">
          <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-blue-600 text-white shadow-sm">
            <LayoutDashboard className="h-6 w-6" />
          </span>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900">Overview</h1>
            <p className="mt-1 max-w-2xl text-sm text-slate-500">
              A snapshot of the entire supply-chain forecasting pipeline — ingestion, registries,
              models, and predictions.
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => void load()}
          disabled={isLoading}
          className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-600 shadow-sm transition hover:bg-slate-50 disabled:opacity-60"
        >
          <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </header>

      {/* Error state */}
      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Hero KPI strip */}
      {isLoading ? (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-24 rounded-2xl" />
          ))}
        </div>
      ) : data ? (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          <HeroKpi
            label="Receipt Rows"
            value={fmt(data.ingestion.total_receipt_rows)}
            icon={Upload}
            accent="blue"
            href="/data-ingestion"
          />
          <HeroKpi
            label="Unique Drugs"
            value={fmt(data.drug_registry.total_drugs)}
            icon={ClipboardList}
            accent="violet"
            href="/drugs"
          />
          <HeroKpi
            label="Categories"
            value={fmt(data.category_registry.total_categories)}
            icon={Tags}
            accent="amber"
            href="/categories"
          />
          <HeroKpi
            label="Forecasted Drugs"
            value={fmt(data.forecasting.total_forecasted_drugs)}
            icon={LineChart}
            accent="emerald"
            href="/forecasting"
          />
          <HeroKpi
            label="Perf. Records"
            value={fmt(data.forecasting.total_performance_records)}
            icon={Activity}
            accent="rose"
            href="/forecasting"
          />
        </div>
      ) : null}

      {/* Analytics charts */}
      {isLoading ? (
        <div className="space-y-6">
          <Skeleton className="h-80 rounded-2xl" />
          <div className="grid gap-6 lg:grid-cols-2">
            <Skeleton className="h-72 rounded-2xl" />
            <Skeleton className="h-72 rounded-2xl" />
            <Skeleton className="h-72 rounded-2xl" />
            <Skeleton className="h-72 rounded-2xl" />
          </div>
        </div>
      ) : data ? (
        <div className="space-y-6">
          <ChartPanel
            title="Receipt volume trend"
            description="Monthly receipt rows and total dispensed quantity over the last 18 months."
          >
            <ReceiptTrendChart data={data.ingestion.receipt_trend} />
          </ChartPanel>

          <div className="grid gap-6 lg:grid-cols-2">
            <ChartPanel
              title="Graduation stages"
              description="How drugs are distributed across cold-start, blended, and full-ensemble tiers."
            >
              <GraduationDonutChart distribution={data.drug_registry.graduation_distribution} />
            </ChartPanel>

            <ChartPanel
              title="Top drugs by receipts"
              description="Highest-volume drugs in the receipt registry."
            >
              <RankedBarChart
                data={data.drug_registry.top_drugs.map((d) => ({
                  label: d.drug_name ?? d.drug_code,
                  sublabel: d.drug_code,
                  value: d.receipt_count,
                }))}
                valueLabel="Receipts"
                color="#7c3aed"
                emptyTitle="No drugs yet"
                emptyDescription="Ingest receipt data to see top drugs by volume."
              />
            </ChartPanel>

            <ChartPanel
              title="Top categories"
              description="Categories with the most distinct drugs."
            >
              <RankedBarChart
                data={data.category_registry.top_categories.map((c) => ({
                  label: c.name ?? c.category_code,
                  sublabel: `${c.receipt_count.toLocaleString()} receipts`,
                  value: c.drug_count,
                }))}
                valueLabel="Drugs"
                color="#d97706"
                emptyTitle="No categories yet"
                emptyDescription="Ingest receipt data to see category breakdown."
              />
            </ChartPanel>

            <ChartPanel
              title="Top dispensing centers"
              description="Centers with the highest receipt row counts."
            >
              <RankedBarChart
                data={data.ingestion.top_centers.map((c) => ({
                  label: c.center_syn_id,
                  value: c.receipt_count,
                }))}
                valueLabel="Receipt rows"
                color="#2563eb"
                emptyTitle="No center data"
                emptyDescription="Receipt rows need a center ID to show center rankings."
              />
            </ChartPanel>
          </div>

          <ChartPanel
            title="Model performance comparison"
            description="Average walk-forward sMAPE and 90% interval coverage by forecasting model."
          >
            <ModelPerformanceChart data={data.forecasting.model_performance} />
          </ChartPanel>
        </div>
      ) : null}

      {/* Section cards — 2-column grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Data Ingestion */}
        <SectionCard title="Data Ingestion" icon={Upload} href="/data-ingestion">
          {isLoading ? (
            <div className="space-y-3">
              <Skeleton className="h-20" />
            </div>
          ) : data ? (
            <div className="space-y-4">
              <KpiRow
                items={[
                  { label: "Receipt rows", value: fmt(data.ingestion.total_receipt_rows) },
                  { label: "Distinct drugs", value: fmt(data.ingestion.distinct_drug_codes) },
                  { label: "Dispensing centers", value: fmt(data.ingestion.distinct_centers) },
                  {
                    label: "Total quantity",
                    value: fmt(Math.round(data.ingestion.total_quantity)),
                  },
                  { label: "First receipt", value: fmtDate(data.ingestion.first_receipt_date) },
                  { label: "Last receipt", value: fmtDate(data.ingestion.last_receipt_date) },
                ]}
              />
            </div>
          ) : null}
        </SectionCard>

        {/* Drug Registry */}
        <SectionCard title="Drug Registry" icon={ClipboardList} href="/drugs">
          {isLoading ? (
            <Skeleton className="h-28" />
          ) : data ? (
            <KpiRow
              items={[
                { label: "Unique drugs", value: fmt(data.drug_registry.total_drugs) },
                {
                  label: "Cold-start only",
                  value: fmt(data.drug_registry.graduation_distribution.cold_start_only),
                },
                {
                  label: "Blended",
                  value: fmt(data.drug_registry.graduation_distribution.blended),
                },
                {
                  label: "Full ensemble",
                  value: fmt(data.drug_registry.graduation_distribution.full_ensemble),
                },
              ]}
            />
          ) : null}
        </SectionCard>

        {/* Category Registry */}
        <SectionCard title="Category Registry" icon={Tags} href="/categories">
          {isLoading ? (
            <Skeleton className="h-28" />
          ) : data ? (
            <KpiRow
              items={[
                {
                  label: "Total categories",
                  value: fmt(data.category_registry.total_categories),
                },
                ...data.category_registry.top_categories.slice(0, 3).map((c) => ({
                  label: c.name ?? c.category_code,
                  value: `${fmt(c.drug_count)} drugs`,
                })),
              ]}
            />
          ) : null}
        </SectionCard>

        {/* Cold Start */}
        <SectionCard title="Cold Start" icon={FlaskConical} href="/cold-start">
          {isLoading ? (
            <Skeleton className="h-28" />
          ) : data ? (
            <div className="space-y-4">
              <p className="text-sm text-slate-500">
                Cold-start module status — both models must be trained before predicting demand for
                new drugs.
              </p>
              <div className="flex flex-wrap gap-3">
                <div className="space-y-1">
                  <p className="text-xs font-medium text-slate-500">Embedder (autoencoder)</p>
                  <StatusBadge
                    ok={data.cold_start.embedder_trained}
                    labelOk="Trained"
                    labelNo="Not trained"
                  />
                </div>
                <div className="space-y-1">
                  <p className="text-xs font-medium text-slate-500">MAML base forecaster</p>
                  <StatusBadge
                    ok={data.cold_start.maml_trained}
                    labelOk="Trained"
                    labelNo="Not trained"
                  />
                </div>
              </div>
              {(!data.cold_start.embedder_trained || !data.cold_start.maml_trained) && (
                <p className="rounded-xl border border-amber-100 bg-amber-50 px-4 py-2.5 text-xs text-amber-700">
                  Visit the Cold Start page to train the missing models before making new-drug
                  predictions.
                </p>
              )}
            </div>
          ) : null}
        </SectionCard>

        {/* Forecasting — full-width */}
        <div className="lg:col-span-2">
          <SectionCard title="Forecasting" icon={TrendingUp} href="/forecasting">
            {isLoading ? (
              <Skeleton className="h-24" />
            ) : data ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <div className="rounded-xl bg-emerald-50 px-4 py-3 ring-1 ring-emerald-100">
                    <p className="text-[11px] font-medium uppercase tracking-wide text-emerald-600">
                      Forecasted drugs
                    </p>
                    <p className="mt-0.5 text-xl font-bold tabular-nums text-emerald-900">
                      {fmt(data.forecasting.total_forecasted_drugs)}
                    </p>
                  </div>
                  <div className="rounded-xl bg-blue-50 px-4 py-3 ring-1 ring-blue-100">
                    <p className="text-[11px] font-medium uppercase tracking-wide text-blue-600">
                      Avg sMAPE
                    </p>
                    <p className="mt-0.5 text-xl font-bold tabular-nums text-blue-900">
                      {fmtSmape(data.forecasting.avg_smape)}
                    </p>
                  </div>
                  <div className="rounded-xl bg-violet-50 px-4 py-3 ring-1 ring-violet-100">
                    <p className="text-[11px] font-medium uppercase tracking-wide text-violet-600">
                      Avg coverage 90
                    </p>
                    <p className="mt-0.5 text-xl font-bold tabular-nums text-violet-900">
                      {fmtPct(data.forecasting.avg_coverage_90)}
                    </p>
                  </div>
                  <div className="rounded-xl bg-slate-50 px-4 py-3 ring-1 ring-slate-200">
                    <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
                      Perf. records
                    </p>
                    <p className="mt-0.5 text-xl font-bold tabular-nums text-slate-900">
                      {fmt(data.forecasting.total_performance_records)}
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-4">
                  {data.forecasting.models_evaluated.length > 0 && (
                    <div className="flex items-center gap-2">
                      <Database className="h-4 w-4 text-slate-400" />
                      <span className="text-xs text-slate-500">Models evaluated:</span>
                      <div className="flex gap-1.5">
                        {data.forecasting.models_evaluated.map((m) => (
                          <span
                            key={m}
                            className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-semibold uppercase text-slate-600"
                          >
                            {m}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {data.forecasting.latest_training_run_id && (
                    <div className="flex items-center gap-2">
                      <BarChart2 className="h-4 w-4 text-slate-400" />
                      <span className="text-xs text-slate-500">Latest run:</span>
                      <span className="font-mono text-xs text-slate-700">
                        {data.forecasting.latest_training_run_id}
                      </span>
                    </div>
                  )}
                  {data.forecasting.models_evaluated.length === 0 &&
                    !data.forecasting.latest_training_run_id && (
                      <p className="text-sm text-slate-400">
                        No forecasting models trained yet. Visit the Forecasting page to begin.
                      </p>
                    )}
                </div>
              </div>
            ) : null}
          </SectionCard>
        </div>
      </div>

      {/* Footer timestamp */}
      {data && (
        <p className="text-right text-xs text-slate-400">
          Generated at{" "}
          {new Date(data.generated_at).toLocaleString(undefined, {
            dateStyle: "medium",
            timeStyle: "short",
          })}
        </p>
      )}
    </div>
  );
}
