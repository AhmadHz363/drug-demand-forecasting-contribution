"use client";

import { ChevronLeft, ChevronRight, Loader2, Search, X } from "lucide-react";
import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";

import { inputClass } from "@/components/cold-start/ui";

const DEFAULT_MIN_QUERY_LENGTH = 3;
const DEFAULT_PAGE_SIZE = 10;

export interface DrugSearchOption {
  drug_code: string;
  drug_name?: string | null;
}

export interface PaginatedDrugSearchResult {
  items: DrugSearchOption[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

interface DrugSearchComboboxProps {
  value: string;
  onChange: (value: string) => void;
  onSelect: (option: DrugSearchOption) => void;
  search: (query: string, page: number) => Promise<PaginatedDrugSearchResult>;
  placeholder?: string;
  disabled?: boolean;
  minQueryLength?: number;
  pageSize?: number;
  showDrugName?: boolean;
  emptyHint?: string;
}

export function DrugSearchCombobox({
  value,
  onChange,
  onSelect,
  search,
  placeholder = "Type at least 3 characters to search…",
  disabled = false,
  minQueryLength = DEFAULT_MIN_QUERY_LENGTH,
  pageSize = DEFAULT_PAGE_SIZE,
  showDrugName = true,
  emptyHint,
}: DrugSearchComboboxProps) {
  const listboxId = useId();
  const containerRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [results, setResults] = useState<PaginatedDrugSearchResult | null>(null);
  const [activeIndex, setActiveIndex] = useState(-1);

  const canSearch = value.trim().length >= minQueryLength;

  const runSearch = useCallback(
    async (query: string, nextPage: number) => {
      if (query.trim().length < minQueryLength) {
        setResults(null);
        setError(null);
        return;
      }

      setLoading(true);
      setError(null);
      try {
        const response = await search(query.trim(), nextPage);
        setResults(response);
        setPage(nextPage);
        setActiveIndex(response.items.length > 0 ? 0 : -1);
      } catch (err) {
        setResults(null);
        setError(err instanceof Error ? err.message : "Search failed");
      } finally {
        setLoading(false);
      }
    },
    [minQueryLength, search],
  );

  useEffect(() => {
    if (!open || !canSearch) {
      return;
    }

    const timer = window.setTimeout(() => {
      void runSearch(value, 1);
    }, 250);

    return () => window.clearTimeout(timer);
  }, [canSearch, open, runSearch, value]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (option: DrugSearchOption) => {
    onSelect(option);
    onChange(option.drug_code);
    setOpen(false);
    setResults(null);
    setActiveIndex(-1);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (!open || !results?.items.length) {
      if (event.key === "ArrowDown" && canSearch) {
        setOpen(true);
      }
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((prev) => (prev + 1) % results.items.length);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((prev) =>
        prev <= 0 ? results.items.length - 1 : prev - 1,
      );
    } else if (event.key === "Enter" && activeIndex >= 0) {
      event.preventDefault();
      handleSelect(results.items[activeIndex]);
    } else if (event.key === "Escape") {
      setOpen(false);
    }
  };

  const showDropdown = open && (canSearch || loading || !!error);

  return (
    <div ref={containerRef} className="relative">
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
        <input
          type="text"
          value={value}
          disabled={disabled}
          role="combobox"
          aria-expanded={showDropdown}
          aria-controls={listboxId}
          aria-autocomplete="list"
          onFocus={() => setOpen(true)}
          onChange={(event) => {
            onChange(event.target.value);
            setOpen(true);
            setPage(1);
          }}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className={`${inputClass} pl-9`}
        />
      </div>

      {showDropdown && (
        <div className="absolute z-20 mt-1 w-full overflow-hidden rounded-xl border border-slate-200 bg-white shadow-lg">
          {!canSearch && (
            <p className="px-3 py-2 text-xs text-slate-500">
              {emptyHint ?? `Enter at least ${minQueryLength} characters to search.`}
            </p>
          )}

          {canSearch && loading && (
            <div className="flex items-center gap-2 px-3 py-3 text-sm text-slate-500">
              <Loader2 className="h-4 w-4 animate-spin" />
              Searching…
            </div>
          )}

          {canSearch && error && (
            <p className="px-3 py-2 text-xs text-red-600">{error}</p>
          )}

          {canSearch && !loading && !error && results && results.items.length === 0 && (
            <p className="px-3 py-2 text-xs text-slate-500">No matching drugs found.</p>
          )}

          {canSearch && !loading && !error && results && results.items.length > 0 && (
            <>
              <ul id={listboxId} role="listbox" className="max-h-56 overflow-y-auto py-1">
                {results.items.map((option, index) => (
                  <li key={option.drug_code} role="presentation">
                    <button
                      type="button"
                      role="option"
                      aria-selected={index === activeIndex}
                      onMouseEnter={() => setActiveIndex(index)}
                      onClick={() => handleSelect(option)}
                      className={`flex w-full flex-col px-3 py-2 text-left text-sm transition ${
                        index === activeIndex
                          ? "bg-blue-50 text-blue-900"
                          : "text-slate-700 hover:bg-slate-50"
                      }`}
                    >
                      <span className="font-semibold">{option.drug_code}</span>
                      {showDrugName && option.drug_name && (
                        <span className="text-xs text-slate-500">{option.drug_name}</span>
                      )}
                    </button>
                  </li>
                ))}
              </ul>

              {results.total_pages > 1 && (
                <div className="flex items-center justify-between border-t border-slate-100 px-3 py-2">
                  <button
                    type="button"
                    disabled={page <= 1 || loading}
                    onClick={() => void runSearch(value, page - 1)}
                    className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-40"
                  >
                    <ChevronLeft className="h-3.5 w-3.5" />
                    Prev
                  </button>
                  <span className="text-xs text-slate-500">
                    Page {page} of {results.total_pages} · {results.total} total
                  </span>
                  <button
                    type="button"
                    disabled={page >= results.total_pages || loading}
                    onClick={() => void runSearch(value, page + 1)}
                    className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-40"
                  >
                    Next
                    <ChevronRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

interface DrugMultiSelectSearchProps {
  selectedCodes: string[];
  onChange: (codes: string[]) => void;
  search: (query: string, page: number) => Promise<PaginatedDrugSearchResult>;
  disabled?: boolean;
  placeholder?: string;
}

export function DrugMultiSelectSearch({
  selectedCodes,
  onChange,
  search,
  disabled = false,
  placeholder = "Search drug codes from receipts…",
}: DrugMultiSelectSearchProps) {
  const [query, setQuery] = useState("");

  const addCode = (option: DrugSearchOption) => {
    if (selectedCodes.includes(option.drug_code)) {
      return;
    }
    onChange([...selectedCodes, option.drug_code]);
    setQuery("");
  };

  const removeCode = (code: string) => {
    onChange(selectedCodes.filter((item) => item !== code));
  };

  return (
    <div className="space-y-2">
      <DrugSearchCombobox
        value={query}
        onChange={setQuery}
        onSelect={addCode}
        search={search}
        disabled={disabled}
        placeholder={placeholder}
        emptyHint="Type at least 3 letters to search drug_receipts."
      />

      {selectedCodes.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {selectedCodes.map((code) => (
            <span
              key={code}
              className="inline-flex items-center gap-1 rounded-full border border-blue-200 bg-blue-50 px-2.5 py-1 text-xs font-medium text-blue-800"
            >
              {code}
              <button
                type="button"
                disabled={disabled}
                onClick={() => removeCode(code)}
                className="rounded-full p-0.5 hover:bg-blue-100 disabled:opacity-50"
                aria-label={`Remove ${code}`}
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

interface PaginatedDrugPickerProps {
  selectedCode: string;
  onSelect: (code: string) => void;
  loadPage: (page: number) => Promise<PaginatedDrugSearchResult>;
  disabled?: boolean;
}

export function PaginatedDrugPicker({
  selectedCode,
  onSelect,
  loadPage,
  disabled = false,
}: PaginatedDrugPickerProps) {
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<PaginatedDrugSearchResult | null>(null);

  const fetchPage = useCallback(
    async (nextPage: number) => {
      setLoading(true);
      setError(null);
      try {
        const response = await loadPage(nextPage);
        setResults(response);
        setPage(nextPage);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load drugs");
      } finally {
        setLoading(false);
      }
    },
    [loadPage],
  );

  useEffect(() => {
    void fetchPage(1);
  }, [fetchPage]);

  if (loading && !results) {
    return (
      <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-500">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading forecasted drugs…
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
        {error}
      </div>
    );
  }

  if (!results || results.items.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-500">
        No drugs with forecast results yet. Train models and run a forecast first.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        {results.items.map((option) => (
          <button
            key={option.drug_code}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(option.drug_code)}
            className={`rounded-xl border px-3 py-2 text-left transition ${
              selectedCode === option.drug_code
                ? "border-blue-600 bg-blue-600 text-white shadow-sm"
                : "border-slate-200 bg-white text-slate-700 hover:border-blue-300 disabled:opacity-50"
            }`}
          >
            <span className="block text-xs font-bold">{option.drug_code}</span>
          </button>
        ))}
      </div>

      {results.total_pages > 1 && (
        <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
          <button
            type="button"
            disabled={disabled || loading || page <= 1}
            onClick={() => void fetchPage(page - 1)}
            className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-slate-600 hover:bg-white disabled:opacity-40"
          >
            <ChevronLeft className="h-3.5 w-3.5" />
            Previous
          </button>
          <span className="text-xs text-slate-500">
            Page {page} of {results.total_pages} · {results.total} drugs
          </span>
          <button
            type="button"
            disabled={disabled || loading || page >= results.total_pages}
            onClick={() => void fetchPage(page + 1)}
            className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-slate-600 hover:bg-white disabled:opacity-40"
          >
            Next
            <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>
      )}
    </div>
  );
}
