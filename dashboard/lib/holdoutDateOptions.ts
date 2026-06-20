import type { HoldoutDateSuggestions } from "@/lib/types";

const MIN_TEST_DAYS = 28;

export interface DateOption {
  value: string;
  label: string;
}

function toIso(d: Date): string {
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function parseIso(value: string): Date {
  const [y, m, d] = value.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d));
}

function addDays(value: string, days: number): string {
  const d = parseIso(value);
  d.setUTCDate(d.getUTCDate() + days);
  return toIso(d);
}

function monthEnd(year: number, month: number): Date {
  return new Date(Date.UTC(year, month, 0));
}

function formatLabel(value: string, dataEnd?: string): string {
  const d = parseIso(value);
  const formatted = d.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
  if (dataEnd && value === dataEnd) {
    return `Latest available (${formatted})`;
  }
  return formatted;
}

function monthStartsBetween(startIso: string, endIso: string): string[] {
  const start = parseIso(startIso);
  const end = parseIso(endIso);
  const values: string[] = [];
  const cursor = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth(), 1));

  while (cursor <= end) {
    const candidate = toIso(cursor);
    if (candidate >= startIso && candidate <= endIso) {
      values.push(candidate);
    }
    cursor.setUTCMonth(cursor.getUTCMonth() + 1);
  }
  return values;
}

function monthEndsBetween(startIso: string, endIso: string): string[] {
  const start = parseIso(startIso);
  const end = parseIso(endIso);
  const values: string[] = [];
  let year = start.getUTCFullYear();
  let month = start.getUTCMonth() + 1;

  while (true) {
    const last = monthEnd(year, month);
    const candidate = toIso(last);
    if (candidate >= startIso && candidate <= endIso) {
      values.push(candidate);
    }
    if (last >= end) break;
    month += 1;
    if (month > 12) {
      month = 1;
      year += 1;
    }
  }
  return values;
}

export function buildTrainEndOptions(suggestions: HoldoutDateSuggestions): DateOption[] {
  return suggestions.train_end_options.map((value) => ({
    value,
    label: formatLabel(value),
  }));
}

export function buildTestStartOptions(
  trainEnd: string,
  suggestions: HoldoutDateSuggestions,
): DateOption[] {
  const earliest = addDays(trainEnd, 1);
  const latest = addDays(suggestions.data_end, -(MIN_TEST_DAYS - 1));
  const upper = latest < earliest ? earliest : latest;

  const dayAfter: DateOption = {
    value: earliest,
    label: `Day after training (${formatLabel(earliest)})`,
  };

  const monthStarts = monthStartsBetween(earliest, upper).filter((value) => value !== earliest);
  const options = [
    dayAfter,
    ...monthStarts.map((value) => ({
      value,
      label: `Month start (${formatLabel(value)})`,
    })),
  ];

  return options.filter(
    (option, index, all) => all.findIndex((item) => item.value === option.value) === index,
  );
}

export function buildTestEndOptions(
  testStart: string,
  suggestions: HoldoutDateSuggestions,
): DateOption[] {
  const monthEnds = monthEndsBetween(testStart, suggestions.data_end);
  const values = new Set(monthEnds);
  values.add(suggestions.data_end);

  return [...values]
    .sort()
    .map((value) => ({
      value,
      label: formatLabel(value, suggestions.data_end),
    }));
}

export function pickValidSelection(
  current: string,
  options: DateOption[],
): string {
  if (options.some((option) => option.value === current)) {
    return current;
  }
  return options[0]?.value ?? current;
}
