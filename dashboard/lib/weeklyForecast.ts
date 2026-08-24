import type { DailyDemandPoint, DailyForecast } from "./types";

export interface WeeklyForecastBucket {
  weekKey: string;
  weekLabel: string;
  weekStart: string;
  weekEnd: string;
  p10: number;
  p50: number;
  p90: number;
  dayCount: number;
}

export interface WeeklyHistoryBucket {
  weekKey: string;
  weekLabel: string;
  quantity: number;
  dayCount: number;
}

function isoWeekKey(isoDate: string): string {
  const d = new Date(isoDate + "T00:00:00");
  const day = d.getDay() || 7;
  d.setDate(d.getDate() + 4 - day);
  const yearStart = new Date(d.getFullYear(), 0, 1);
  const week = Math.ceil(((d.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
  return `${d.getFullYear()}-W${String(week).padStart(2, "0")}`;
}

function weekLabelFromKey(key: string): string {
  const match = key.match(/^(\d{4})-W(\d{2})$/);
  if (!match) return key;
  return `W${match[2]} ${match[1]}`;
}

function formatShortDate(iso: string): string {
  return new Date(iso + "T00:00:00").toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export function aggregateForecastByWeek(forecast: DailyForecast[]): WeeklyForecastBucket[] {
  const buckets = new Map<
    string,
    WeeklyForecastBucket & { dates: string[] }
  >();

  for (const day of forecast) {
    const key = isoWeekKey(day.date);
    const existing = buckets.get(key);
    if (existing) {
      existing.p10 += day.p10;
      existing.p50 += day.p50;
      existing.p90 += day.p90;
      existing.dayCount += 1;
      existing.dates.push(day.date);
      if (day.date < existing.weekStart) existing.weekStart = day.date;
      if (day.date > existing.weekEnd) existing.weekEnd = day.date;
    } else {
      buckets.set(key, {
        weekKey: key,
        weekLabel: weekLabelFromKey(key),
        weekStart: day.date,
        weekEnd: day.date,
        p10: day.p10,
        p50: day.p50,
        p90: day.p90,
        dayCount: 1,
        dates: [day.date],
      });
    }
  }

  return [...buckets.values()]
    .sort((a, b) => a.weekStart.localeCompare(b.weekStart))
    .map(({ dates, ...rest }) => {
      void dates;
      return rest;
    });
}

export function aggregateHistoryByWeek(history: DailyDemandPoint[]): WeeklyHistoryBucket[] {
  const buckets = new Map<string, WeeklyHistoryBucket>();

  for (const point of history) {
    const key = isoWeekKey(point.date);
    const existing = buckets.get(key);
    if (existing) {
      existing.quantity += point.quantity;
      existing.dayCount += 1;
    } else {
      buckets.set(key, {
        weekKey: key,
        weekLabel: weekLabelFromKey(key),
        quantity: point.quantity,
        dayCount: 1,
      });
    }
  }

  return [...buckets.values()].sort((a, b) => a.weekKey.localeCompare(b.weekKey));
}

export function weekRangeLabel(start: string, end: string): string {
  return `${formatShortDate(start)} – ${formatShortDate(end)}`;
}

export const WEEK_HORIZON_OPTIONS = [
  { weeks: 1, days: 7, label: "1 week" },
  { weeks: 2, days: 14, label: "2 weeks" },
  { weeks: 4, days: 28, label: "4 weeks" },
] as const;
