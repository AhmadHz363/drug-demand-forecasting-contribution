"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, Circle, Loader2 } from "lucide-react";

const STEPS = [
  {
    id: "upload",
    label: "Uploading file",
    detail: "Sending spreadsheet to the server",
    afterMs: 0,
  },
  {
    id: "validate",
    label: "Validating rows",
    detail: "Checking columns, types, and required fields",
    afterMs: 2500,
  },
  {
    id: "insert",
    label: "Inserting into database",
    detail: "Bulk-loading validated receipt lines",
    afterMs: 8000,
  },
  {
    id: "sync",
    label: "Syncing daily demand",
    detail: "Refreshing aggregated demand history",
    afterMs: 20000,
  },
] as const;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatElapsed(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes === 0) return `${seconds}s`;
  return `${minutes}m ${seconds.toString().padStart(2, "0")}s`;
}

interface IngestionLoadingOverlayProps {
  fileName: string;
  fileSize: number;
}

export function IngestionLoadingOverlay({ fileName, fileSize }: IngestionLoadingOverlayProps) {
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    const started = Date.now();
    const timer = window.setInterval(() => {
      setElapsedMs(Date.now() - started);
    }, 250);
    return () => window.clearInterval(timer);
  }, []);

  const activeIndex = STEPS.reduce(
    (idx, step, i) => (elapsedMs >= step.afterMs ? i : idx),
    0,
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="ingestion-loading-title"
      aria-busy="true"
    >
      <div className="w-full max-w-lg overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl">
        <div className="border-b border-slate-100 bg-gradient-to-r from-blue-50 to-white px-6 py-5">
          <div className="flex items-start gap-3">
            <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white shadow-sm">
              <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
            </span>
            <div className="min-w-0 flex-1">
              <h2 id="ingestion-loading-title" className="text-base font-semibold text-slate-900">
                Ingesting receipt data
              </h2>
              <p className="mt-0.5 truncate text-sm text-slate-500" title={fileName}>
                {fileName}
              </p>
              <p className="mt-1 text-xs text-slate-400">
                {formatBytes(fileSize)} · Elapsed {formatElapsed(elapsedMs)}
              </p>
            </div>
          </div>

          <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-slate-200">
            <div className="relative h-full w-1/3 rounded-full bg-blue-600 progress-indeterminate" />
          </div>
        </div>

        <div className="space-y-4 px-6 py-5">
          <ol className="space-y-3" aria-label="Ingestion progress">
            {STEPS.map((step, index) => {
              const isComplete = index < activeIndex;
              const isActive = index === activeIndex;
              const isPending = index > activeIndex;

              return (
                <li
                  key={step.id}
                  className={`flex items-start gap-3 rounded-xl border px-3 py-3 transition ${
                    isActive
                      ? "border-blue-200 bg-blue-50/80"
                      : isComplete
                        ? "border-emerald-100 bg-emerald-50/50"
                        : "border-slate-100 bg-slate-50/50"
                  }`}
                >
                  <span
                    className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
                      isActive
                        ? "bg-blue-600 text-white"
                        : isComplete
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-slate-100 text-slate-400"
                    }`}
                  >
                    {isComplete ? (
                      <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
                    ) : isActive ? (
                      <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                    ) : (
                      <Circle className="h-4 w-4" aria-hidden="true" />
                    )}
                  </span>
                  <div className="min-w-0">
                    <p
                      className={`text-sm font-medium ${
                        isPending ? "text-slate-400" : "text-slate-800"
                      }`}
                    >
                      {step.label}
                    </p>
                    <p
                      className={`text-xs ${
                        isActive ? "text-blue-700" : isComplete ? "text-emerald-700" : "text-slate-400"
                      }`}
                    >
                      {step.detail}
                    </p>
                  </div>
                </li>
              );
            })}
          </ol>

          <div className="rounded-xl border border-slate-100 bg-slate-50/80 p-4" aria-hidden="true">
            <p className="mb-3 text-xs font-medium uppercase tracking-wide text-slate-400">
              Preview
            </p>
            <div className="grid grid-cols-3 gap-3">
              <div className="shimmer h-14 rounded-lg" />
              <div className="shimmer h-14 rounded-lg" />
              <div className="shimmer h-14 rounded-lg" />
            </div>
            <div className="mt-3 space-y-2">
              <div className="shimmer h-3 w-full rounded" />
              <div className="shimmer h-3 w-5/6 rounded" />
              <div className="shimmer h-3 w-4/6 rounded" />
            </div>
          </div>

          <p className="text-center text-xs leading-relaxed text-slate-500">
            Large files can take several minutes. Keep this tab open — the server is processing
            your upload in a single transaction.
          </p>
        </div>
      </div>
    </div>
  );
}
