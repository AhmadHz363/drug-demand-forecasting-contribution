"use client";

import { useMemo, useState } from "react";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity } from "lucide-react";

import { EmptyState, selectClass } from "@/components/cold-start/ui";
import type { DailyDemandPoint } from "@/lib/types";

interface DrugUsageChartProps {
  series: DailyDemandPoint[];
  firstReceiptDate?: string | null;
  lastReceiptDate?: string | null;
}

interface GapSegment {
  start: string;
  end: string;
  days: number;
}

interface YearUsageStats {
  year: number;
  chartData: Array<DailyDemandPoint & { label: string; hasData: number }>;
  daysWithData: number;
  spanDays: number;
  gapDays: number;
  coveragePct: number;
  longestGap: number;
  totalQuantity: number;
  peakDay: DailyDemandPoint | null;
  gaps: GapSegment[];
}

function parseLocalDate(iso: string): Date {
  return new Date(`${iso}T00:00:00`);
}

function toIsoDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function formatShortDate(iso: string): string {
  return parseLocalDate(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function formatFullDate(iso: string): string {
  return parseLocalDate(iso).toLocaleDateString(undefined, {
    weekday: "short",
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function addDays(date: Date, days: number): Date {
  const next = new Date(date);
  next.setDate(next.getDate() + days);
  return next;
}

function computeYearStats(series: DailyDemandPoint[], year: number): YearUsageStats | null {
  const yearPoints = series
    .filter((point) => parseLocalDate(point.date).getFullYear() === year)
    .sort((a, b) => a.date.localeCompare(b.date));

  if (!yearPoints.length) {
    return null;
  }

  const dataByDate = new Map(yearPoints.map((point) => [point.date, point.quantity]));
  const firstDate = yearPoints[0].date;
  const lastDate = yearPoints[yearPoints.length - 1].date;

  let spanDays = 0;
  let gapDays = 0;
  let longestGap = 0;
  let currentGap = 0;
  const gaps: GapSegment[] = [];
  let gapStart: string | null = null;

  for (
    let cursor = parseLocalDate(firstDate);
    cursor <= parseLocalDate(lastDate);
    cursor = addDays(cursor, 1)
  ) {
    spanDays += 1;
    const iso = toIsoDate(cursor);
    if (!dataByDate.has(iso)) {
      gapDays += 1;
      currentGap += 1;
      if (gapStart === null) {
        gapStart = iso;
      }
    } else if (gapStart !== null) {
      gaps.push({
        start: gapStart,
        end: toIsoDate(addDays(cursor, -1)),
        days: currentGap,
      });
      gapStart = null;
      currentGap = 0;
    }
  }

  if (gapStart !== null && currentGap > 0) {
    gaps.push({
      start: gapStart,
      end: lastDate,
      days: currentGap,
    });
  }

  longestGap = gaps.reduce((max, gap) => Math.max(max, gap.days), 0);

  let totalQuantity = 0;
  let peakDay: DailyDemandPoint | null = null;
  const chartData = yearPoints.map((point) => {
    totalQuantity += point.quantity;
    if (!peakDay || point.quantity > peakDay.quantity) {
      peakDay = point;
    }
    return {
      ...point,
      label: formatShortDate(point.date),
      hasData: 1,
    };
  });

  return {
    year,
    chartData,
    daysWithData: yearPoints.length,
    spanDays,
    gapDays,
    coveragePct: spanDays > 0 ? (yearPoints.length / spanDays) * 100 : 0,
    longestGap,
    totalQuantity,
    peakDay,
    gaps: gaps.sort((a, b) => b.days - a.days),
  };
}

export function DrugUsageChart({ series, firstReceiptDate, lastReceiptDate }: DrugUsageChartProps) {
  const availableYears = useMemo(() => {
    const years = new Set<number>();
    for (const point of series) {
      years.add(parseLocalDate(point.date).getFullYear());
    }
    if (firstReceiptDate) {
      years.add(parseLocalDate(firstReceiptDate).getFullYear());
    }
    if (lastReceiptDate) {
      years.add(parseLocalDate(lastReceiptDate).getFullYear());
    }
    return [...years].sort((a, b) => b - a);
  }, [series, firstReceiptDate, lastReceiptDate]);

  const [selectedYear, setSelectedYear] = useState<number | null>(availableYears[0] ?? null);

  const activeYear = selectedYear ?? availableYears[0] ?? null;
  const stats = activeYear != null ? computeYearStats(series, activeYear) : null;

  if (!series.length) {
    return (
      <EmptyState
        icon={<Activity className="h-6 w-6" />}
        title="No usage history"
        description="This drug has no receipt quantities recorded yet."
      />
    );
  }

  if (!availableYears.length || !stats) {
    return (
      <EmptyState
        icon={<Activity className="h-6 w-6" />}
        title="No data for selected year"
        description="Choose another year to inspect daily quantities and coverage gaps."
      />
    );
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
            Daily usage
          </h3>
          <p className="mt-1 text-sm text-slate-500">
            Quantity used on days with receipts. Stats below highlight missing days in the span.
          </p>
        </div>
        <label className="block min-w-[140px]">
          <span className="mb-1 block text-xs font-medium text-slate-500">Year</span>
          <select
            value={activeYear}
            onChange={(event) => setSelectedYear(Number(event.target.value))}
            className={selectClass}
            aria-label="Filter usage by year"
          >
            {availableYears.map((year) => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <StatCard label="Days with data" value={String(stats.daysWithData)} highlight />
        <StatCard label="Span (first→last)" value={String(stats.spanDays)} />
        <StatCard
          label="Gap days"
          value={String(stats.gapDays)}
          tone={stats.gapDays > 0 ? "warning" : "default"}
        />
        <StatCard label="Coverage" value={`${stats.coveragePct.toFixed(1)}%`} />
        <StatCard
          label="Longest gap"
          value={stats.longestGap > 0 ? `${stats.longestGap} days` : "None"}
          tone={stats.longestGap > 7 ? "warning" : "default"}
        />
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <StatCard
          label={`Total units (${activeYear})`}
          value={stats.totalQuantity.toFixed(1)}
          highlight
        />
        <StatCard
          label="Peak day"
          value={
            stats.peakDay
              ? `${stats.peakDay.quantity.toFixed(1)} on ${formatShortDate(stats.peakDay.date)}`
              : "—"
          }
        />
      </div>

      <CoverageBar coveragePct={stats.coveragePct} gapDays={stats.gapDays} />

      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={stats.chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#64748b" }} minTickGap={24} />
          <YAxis tick={{ fontSize: 11, fill: "#64748b" }} />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as DailyDemandPoint & { label: string };
              return (
                <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg">
                  <p className="mb-1.5 font-semibold text-slate-800">{formatFullDate(row.date)}</p>
                  <p className="text-blue-700">
                    Quantity: <span className="font-medium">{row.quantity.toFixed(2)}</span>
                  </p>
                </div>
              );
            }}
          />
          <Bar dataKey="hasData" fill="#dbeafe" barSize={8} name="Has data" />
          <Line
            type="monotone"
            dataKey="quantity"
            stroke="#2563eb"
            strokeWidth={2}
            dot={stats.chartData.length <= 60 ? { r: 2.5, fill: "#2563eb" } : false}
            activeDot={{ r: 4 }}
            name="Quantity used"
          />
        </ComposedChart>
      </ResponsiveContainer>

      {stats.gaps.length > 0 ? (
        <section className="rounded-xl border border-amber-200 bg-amber-50/60 p-4">
          <h4 className="text-sm font-semibold text-amber-900">Coverage gaps</h4>
          <p className="mt-1 text-xs text-amber-800/90">
            Missing receipt days between the first and last usage day in {activeYear}.
          </p>
          <div className="mt-3 overflow-x-auto rounded-lg border border-amber-200/80 bg-white">
            <table className="min-w-full text-xs">
              <thead className="bg-amber-50 text-left text-amber-900">
                <tr>
                  <th className="px-3 py-2 font-semibold">From</th>
                  <th className="px-3 py-2 font-semibold">To</th>
                  <th className="px-3 py-2 text-right font-semibold">Days missing</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-amber-100">
                {stats.gaps.slice(0, 8).map((gap) => (
                  <tr key={`${gap.start}-${gap.end}`}>
                    <td className="px-3 py-2 text-slate-700">{formatFullDate(gap.start)}</td>
                    <td className="px-3 py-2 text-slate-700">{formatFullDate(gap.end)}</td>
                    <td className="px-3 py-2 text-right font-medium tabular-nums text-amber-900">
                      {gap.days}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {stats.gaps.length > 8 && (
            <p className="mt-2 text-xs text-amber-800/80">
              Showing the 8 longest gaps out of {stats.gaps.length} total.
            </p>
          )}
        </section>
      ) : (
        <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 px-4 py-3 text-sm text-emerald-800">
          No gaps detected between the first and last usage day in {activeYear}.
        </div>
      )}
    </div>
  );
}

function CoverageBar({
  coveragePct,
  gapDays,
}: {
  coveragePct: number;
  gapDays: number;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50/80 px-4 py-3">
      <div className="mb-2 flex items-center justify-between text-xs font-medium text-slate-600">
        <span>Data coverage in span</span>
        <span>
          {coveragePct.toFixed(1)}% filled
          {gapDays > 0 ? ` · ${gapDays} gap day${gapDays === 1 ? "" : "s"}` : ""}
        </span>
      </div>
      <div className="h-3 overflow-hidden rounded-full bg-slate-200">
        <div
          className={`h-full rounded-full transition-all ${
            coveragePct >= 90 ? "bg-emerald-500" : coveragePct >= 70 ? "bg-amber-500" : "bg-red-500"
          }`}
          style={{ width: `${Math.min(100, Math.max(0, coveragePct))}%` }}
        />
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  highlight,
  tone = "default",
}: {
  label: string;
  value: string;
  highlight?: boolean;
  tone?: "default" | "warning";
}) {
  const toneClass =
    tone === "warning"
      ? "border-amber-200 bg-amber-50"
      : highlight
        ? "border-blue-200 bg-blue-50"
        : "border-slate-200 bg-slate-50";
  const valueClass =
    tone === "warning"
      ? "text-amber-900"
      : highlight
        ? "text-blue-800"
        : "text-slate-900";

  return (
    <div className={`rounded-xl border px-4 py-3 ${toneClass}`}>
      <p className="text-xs font-medium text-slate-500">{label}</p>
      <p className={`mt-0.5 text-lg font-semibold tabular-nums ${valueClass}`}>{value}</p>
    </div>
  );
}
