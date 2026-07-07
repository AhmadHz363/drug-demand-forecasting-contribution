"use client";

import { useCallback } from "react";
import { Database } from "lucide-react";

import { IngestionLoadingOverlay } from "@/components/data-ingestion/IngestionLoadingOverlay";
import { IngestionResultsPanel } from "@/components/data-ingestion/IngestionResultsPanel";
import { ReceiptUploadPanel } from "@/components/data-ingestion/ReceiptUploadPanel";
import { useReceiptUpload } from "@/hooks/useReceiptUpload";

export function DataIngestionDashboard() {
  const { isUploading, status, result, error, fileName, fileSize, upload, reset } =
    useReceiptUpload();

  const handleUpload = useCallback(
    (file: File) => {
      upload(file).catch(() => {
        /* error is stored in hook state */
      });
    },
    [upload],
  );

  return (
    <div className="min-h-full">
      {isUploading && fileName && fileSize !== null && (
        <IngestionLoadingOverlay fileName={fileName} fileSize={fileSize} />
      )}

      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
          <div className="flex items-start gap-3">
            <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white shadow-sm">
              <Database className="h-5 w-5" />
            </span>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
                Data Ingestion
              </h1>
              <p className="mt-1 max-w-2xl text-sm leading-relaxed text-slate-500">
                Upload pharmacy receipt spreadsheets to populate drug_receipts and refresh daily
                demand aggregates used by forecasting and cold-start modules.
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-6 px-4 py-6 sm:px-6">
        <div className="grid gap-6 lg:grid-cols-2">
          <ReceiptUploadPanel
            disabled={isUploading || status === "success"}
            onUpload={handleUpload}
          />
          <IngestionResultsPanel
            result={result}
            error={error}
            fileName={fileName}
            isUploading={isUploading}
            onReset={reset}
          />
        </div>
      </main>
    </div>
  );
}
