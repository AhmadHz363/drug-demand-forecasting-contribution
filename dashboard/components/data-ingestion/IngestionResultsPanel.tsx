"use client";

import { useState } from "react";
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  RotateCcw,
  XCircle,
} from "lucide-react";

import { EmptyState, Panel, PanelBody, PanelHeader, Skeleton, StatChip } from "@/components/cold-start/ui";
import type { UploadReceiptsResponse } from "@/lib/types";

interface IngestionResultsPanelProps {
  result: UploadReceiptsResponse | null;
  error: string | null;
  fileName: string | null;
  isUploading: boolean;
  onReset: () => void;
}

export function IngestionResultsPanel({
  result,
  error,
  fileName,
  isUploading,
  onReset,
}: IngestionResultsPanelProps) {
  const [showAllErrors, setShowAllErrors] = useState(false);

  const visibleErrors =
    !result?.errors.length
      ? []
      : showAllErrors
        ? result.errors
        : result.errors.slice(0, 5);

  if (!result && !error) {
    return (
      <Panel>
        <PanelHeader
          step={2}
          title="Ingestion results"
          description={
            isUploading
              ? "Processing your upload…"
              : "Upload a file to see row counts and any validation issues."
          }
        />
        <PanelBody>
          {isUploading ? (
            <div className="space-y-3" aria-hidden="true">
              <div className="grid grid-cols-3 gap-3">
                <Skeleton className="h-16 rounded-xl" />
                <Skeleton className="h-16 rounded-xl" />
                <Skeleton className="h-16 rounded-xl" />
              </div>
              <Skeleton className="h-24 rounded-xl" />
            </div>
          ) : (
            <EmptyState
              icon={<BarChart3 className="h-6 w-6" />}
              title="No ingestion results yet"
              description="Upload a receipt spreadsheet and start ingestion to see row counts and any validation issues here."
            />
          )}
        </PanelBody>
      </Panel>
    );
  }

  if (error) {
    return (
      <Panel>
        <PanelHeader
          step={2}
          title="Ingestion failed"
          description={fileName ? `Could not process ${fileName}` : "The upload did not complete."}
          action={
            <button
              type="button"
              onClick={onReset}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-50"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Try again
            </button>
          }
        />
        <PanelBody>
          <div className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-4">
            <XCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-600" />
            <div>
              <p className="text-sm font-semibold text-red-800">Upload error</p>
              <p className="mt-1 text-sm leading-relaxed text-red-700">{error}</p>
            </div>
          </div>
        </PanelBody>
      </Panel>
    );
  }

  if (!result) return null;

  const allSucceeded =
    result.raw_inserted_rows > 0 &&
    result.enriched_inserted_rows > 0 &&
    result.failed_rows === 0;
  const partialSuccess =
    (result.raw_inserted_rows > 0 || result.enriched_inserted_rows > 0) &&
    (result.failed_rows > 0 || result.enriched_inserted_rows === 0);
  const allFailed = result.raw_inserted_rows === 0;

  return (
    <Panel>
      <PanelHeader
        step={2}
        title="Ingestion results"
        description={
          fileName
            ? `Processed ${fileName}`
            : "Row counts and validation details from the latest upload."
        }
        action={
          <button
            type="button"
            onClick={onReset}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-50"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Upload another
          </button>
        }
      />
      <PanelBody className="space-y-5">
        <div
          className={`flex items-start gap-3 rounded-xl border px-4 py-4 ${
            allSucceeded
              ? "border-emerald-200 bg-emerald-50/80"
              : partialSuccess
                ? "border-amber-200 bg-amber-50/80"
                : "border-red-200 bg-red-50/80"
          }`}
        >
          {allSucceeded ? (
            <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" />
          ) : partialSuccess ? (
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
          ) : (
            <XCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-600" />
          )}
          <div>
            <p
              className={`text-sm font-semibold ${
                allSucceeded
                  ? "text-emerald-800"
                  : partialSuccess
                    ? "text-amber-800"
                    : "text-red-800"
              }`}
            >
              {allSucceeded && "Ingestion completed successfully"}
              {partialSuccess && "Ingestion completed with warnings"}
              {allFailed && "No rows were inserted"}
            </p>
            <p
              className={`mt-1 text-sm ${
                allSucceeded
                  ? "text-emerald-700"
                  : partialSuccess
                    ? "text-amber-700"
                    : "text-red-700"
              }`}
            >
              {allSucceeded &&
                `${result.raw_inserted_rows.toLocaleString()} raw rows and ${result.enriched_inserted_rows.toLocaleString()} enriched SKU-day rows stored (${result.filtered_out_rows.toLocaleString()} lines filtered by movement type).`}
              {partialSuccess &&
                `${result.raw_inserted_rows.toLocaleString()} raw and ${result.enriched_inserted_rows.toLocaleString()} enriched rows stored; ${result.failed_rows.toLocaleString()} issue${result.failed_rows === 1 ? "" : "s"}.`}
              {allFailed &&
                (result.failed_rows > 0
                  ? `${result.failed_rows.toLocaleString()} row${result.failed_rows === 1 ? "" : "s"} failed validation. Review the errors below.`
                  : "The file contained no valid rows to insert.")}
            </p>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <StatChip
            label="Raw rows"
            value={result.raw_inserted_rows.toLocaleString()}
            accent="green"
          />
          <StatChip
            label="Enriched rows"
            value={result.enriched_inserted_rows.toLocaleString()}
            accent="green"
          />
          <StatChip
            label="Filtered out"
            value={result.filtered_out_rows.toLocaleString()}
            accent="slate"
          />
          <StatChip
            label="Failed"
            value={result.failed_rows.toLocaleString()}
            accent={result.failed_rows > 0 ? "amber" : "slate"}
          />
        </div>

        {result.errors.length > 0 && (
          <StatChip
            label="Error details"
            value={result.errors.length.toLocaleString()}
            accent="amber"
          />
        )}

        {result.errors.length > 0 && (
          <div className="overflow-hidden rounded-xl border border-slate-200">
            <div className="border-b border-slate-100 bg-slate-50 px-4 py-3">
              <p className="text-sm font-semibold text-slate-800">Validation errors</p>
              <p className="text-xs text-slate-500">
                Row numbers refer to spreadsheet data rows (excluding the header).
              </p>
            </div>
            <ul className="divide-y divide-slate-100">
              {visibleErrors.map((err) => (
                <li key={`${err.row_index}-${err.message}`} className="px-4 py-3 text-sm">
                  <span className="font-medium text-slate-800">Row {err.row_index}</span>
                  <span className="text-slate-400"> · </span>
                  <span className="text-slate-600">{err.message}</span>
                </li>
              ))}
            </ul>
            {result.errors.length > 5 && (
              <button
                type="button"
                onClick={() => setShowAllErrors((v) => !v)}
                className="flex w-full items-center justify-center gap-1 border-t border-slate-100 bg-white px-4 py-2.5 text-xs font-medium text-blue-600 transition hover:bg-slate-50"
              >
                {showAllErrors
                  ? "Show fewer errors"
                  : `Show all ${result.errors.length} errors`}
                <ChevronDown
                  className={`h-3.5 w-3.5 transition ${showAllErrors ? "rotate-180" : ""}`}
                />
              </button>
            )}
          </div>
        )}
      </PanelBody>
    </Panel>
  );
}
