"use client";

import { useMemo, useState } from "react";
import { Search } from "lucide-react";

import { accuracyAccent } from "@/lib/forecastMetrics";
import type { CameoPerDrugAccuracy } from "@/lib/types";

interface CameoPerDrugAccuracyTableProps {
  rows: CameoPerDrugAccuracy[];
  selectedDrugCode?: string | null;
}

function formatPct(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${value.toFixed(1)}%`;
}

export function CameoPerDrugAccuracyTable({
  rows,
  selectedDrugCode,
}: CameoPerDrugAccuracyTableProps) {
  const [filter, setFilter] = useState("");

  const filtered = useMemo(() => {
    const query = filter.trim().toLowerCase();
    const sorted = [...rows].sort((a, b) => a.drug_code.localeCompare(b.drug_code));
    if (!query) return sorted;
    return sorted.filter(
      (row) =>
        row.drug_code.toLowerCase().includes(query) ||
        (row.sb_class ?? "").toLowerCase().includes(query),
    );
  }, [filter, rows]);

  if (rows.length === 0) {
    return null;
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-slate-900">Per-drug hold-out accuracy</h3>
          <p className="mt-1 text-sm text-slate-500">
            CAMEO WAPE accuracy on each held-out cold-start drug ({rows.length} drugs).
          </p>
        </div>
        <label className="relative block min-w-[200px]">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            type="search"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter drugs…"
            className="w-full rounded-lg border border-slate-200 py-2 pl-9 pr-3 text-sm"
          />
        </label>
      </div>

      <div className="mt-4 max-h-80 overflow-auto rounded-xl border border-slate-200">
        <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
          <thead className="sticky top-0 bg-slate-50 text-slate-500">
            <tr>
              <th className="px-3 py-2 font-medium">Drug</th>
              <th className="px-3 py-2 font-medium">Accuracy</th>
              <th className="px-3 py-2 font-medium">SB class</th>
              <th className="px-3 py-2 font-medium">WAPE</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white text-slate-800">
            {filtered.map((row) => {
              const isSelected = row.drug_code === selectedDrugCode;
              const accent =
                row.accuracy_pct != null ? accuracyAccent(row.accuracy_pct) : "slate";
              return (
                <tr key={row.drug_code} className={isSelected ? "bg-blue-50/60" : undefined}>
                  <td className="px-3 py-2 font-medium">{row.drug_code}</td>
                  <td className="px-3 py-2">
                    <span
                      className={
                        accent === "green"
                          ? "font-semibold text-emerald-700"
                          : accent === "blue"
                            ? "font-semibold text-blue-700"
                            : ""
                      }
                    >
                      {formatPct(row.accuracy_pct)}
                    </span>
                  </td>
                  <td className="px-3 py-2 capitalize">{row.sb_class ?? "—"}</td>
                  <td className="px-3 py-2 text-slate-500">
                    {row.wape != null ? row.wape.toFixed(3) : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
