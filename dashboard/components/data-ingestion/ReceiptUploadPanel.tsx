"use client";

import { useCallback, useRef, useState } from "react";
import { AlertCircle, FileSpreadsheet, Upload, X } from "lucide-react";

import { Panel, PanelBody, PanelHeader } from "@/components/cold-start/ui";

const ACCEPTED_EXTENSIONS = [".xlsx", ".xls", ".csv"];
const ACCEPTED_MIME =
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel,text/csv,.xlsx,.xls,.csv";

function isAcceptedFile(file: File): boolean {
  const lower = file.name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface ReceiptUploadPanelProps {
  disabled?: boolean;
  onUpload: (file: File) => void;
}

export function ReceiptUploadPanel({ disabled = false, onUpload }: ReceiptUploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleFile = useCallback((file: File | null) => {
    if (!file) {
      setSelectedFile(null);
      setValidationError(null);
      return;
    }

    if (!isAcceptedFile(file)) {
      setSelectedFile(null);
      setValidationError("Only .xlsx, .xls, and .csv files are supported.");
      return;
    }

    setValidationError(null);
    setSelectedFile(file);
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const file = event.dataTransfer.files[0];
      handleFile(file ?? null);
    },
    [disabled, handleFile],
  );

  const onBrowse = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0] ?? null;
      handleFile(file);
      event.target.value = "";
    },
    [handleFile],
  );

  const clearFile = useCallback(() => {
    handleFile(null);
    if (inputRef.current) inputRef.current.value = "";
  }, [handleFile]);

  const submit = useCallback(() => {
    if (selectedFile && !disabled) {
      onUpload(selectedFile);
    }
  }, [disabled, onUpload, selectedFile]);

  return (
    <Panel>
      <PanelHeader
        step={1}
        title="Upload hospital Excel export"
        description="Import raw pharmacy receipt rows, then build the cleaned daily demand panel used for SHIELD-XR training. Only inpatient sales (مـبـيع الـى مـرضـى داخلـي) contribute to demand."
      />
      <PanelBody className="space-y-4">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            if (!disabled) setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          className={`relative rounded-2xl border-2 border-dashed px-6 py-10 text-center transition ${
            dragOver
              ? "border-blue-400 bg-blue-50/80"
              : "border-slate-200 bg-slate-50/60 hover:border-slate-300"
          } ${disabled ? "pointer-events-none opacity-60" : "cursor-pointer"}`}
          onClick={() => !disabled && inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              if (!disabled) inputRef.current?.click();
            }
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_MIME}
            className="sr-only"
            disabled={disabled}
            onChange={onBrowse}
          />

          <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-white shadow-sm ring-1 ring-slate-200">
            <Upload className="h-6 w-6 text-blue-600" />
          </div>
          <p className="text-sm font-semibold text-slate-800">
            Drag and drop your file here, or click to browse
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Supports .xlsx, .xls, and .csv. Large files (100k+ rows) may take 10–30+ minutes;
            keep this tab open until ingestion completes.
          </p>
        </div>

        {validationError && (
          <div className="flex items-start gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{validationError}</span>
          </div>
        )}

        {selectedFile && (
          <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-emerald-50 text-emerald-700 ring-1 ring-emerald-100">
              <FileSpreadsheet className="h-5 w-5" />
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-slate-900">{selectedFile.name}</p>
              <p className="text-xs text-slate-500">{formatBytes(selectedFile.size)}</p>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                clearFile();
              }}
              disabled={disabled}
              className="rounded-lg p-2 text-slate-400 transition hover:bg-slate-100 hover:text-slate-600 disabled:opacity-50"
              aria-label="Remove selected file"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        <button
          type="button"
          onClick={submit}
          disabled={disabled || !selectedFile}
          className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Upload className="h-4 w-4" />
          Start ingestion
        </button>

        <div className="rounded-xl border border-slate-100 bg-slate-50/80 px-4 py-3">
          <p className="text-xs font-medium text-slate-700">Expected columns</p>
          <p className="mt-1 text-xs leading-relaxed text-slate-500">
            DOC, LINE, CAT, C.R, DATE, MOV#, Mov des, CODE, ARTICLE, M, C.S, QTY, U.P, T.P, MRN,
            AD DATE, R, U, AGE, DR. Re-uploading the same filename replaces prior raw and enriched
            rows for that file.
          </p>
        </div>
      </PanelBody>
    </Panel>
  );
}
