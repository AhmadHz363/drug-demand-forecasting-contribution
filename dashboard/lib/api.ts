import { authHeaders } from "@/lib/auth-token";
import type {
  AuthUser,
  ColdStartPredictRequest,
  ColdStartPredictResponse,
  ColdStartStatusResponse,
  ForecastRequest,
  ForecastResponse,
  HoldoutDateSuggestions,
  HoldoutResponse,
  HoldoutValidationRequest,
  OverviewResponse,
  PaginatedDrugCodeSearchResponse,
  PaginatedDrugListResponse,
  PaginatedCategoryListResponse,
  DrugDetailResponse,
  PaginatedReceiptDrugSearchResponse,
  PerformanceMonitoringResponse,
  ShieldXRStatusResponse,
  TrainForecastingRequest,
  TrainForecastingResponse,
  LoginPayload,
  LoginResponse,
  RegisterPayload,
  TrainResponse,
  UploadReceiptsResponse,
} from "./types";

const BASE = "/api/backend";

/** Direct origin for large multipart uploads (bypasses Next body buffer + proxy timeout). */
function backendOrigin(): string {
  const url = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  return url.replace(/\/$/, "");
}

const TRAINING_PATHS = new Set([
  "/cold-start/train-cameo",
  "/cold-start/train-embedder",
  "/cold-start/train-maml",
  "/forecasting/train",
  "/forecasting/holdout",
  "/upload-hospital-receipts",
]);

async function parseError(res: Response, path: string): Promise<string> {
  const body = await res.json().catch(() => ({}));
  const detail = body?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ");
  }
  if (res.status >= 500 && TRAINING_PATHS.has(path)) {
    const longRunningHint =
      path === "/upload-hospital-receipts"
        ? "Large spreadsheets can take several minutes — check the backend terminal; " +
          "if ingestion finished there, refresh the page."
        : "Large catalogs can take 1–2 minutes — check the backend terminal; " +
          "if training finished there, click Train again or refresh the page.";
    return `Request failed (proxy or server error). ${longRunningHint}`;
  }
  return `HTTP ${res.status}`;
}

function mergeHeaders(init?: RequestInit): Headers {
  const headers = new Headers(init?.headers);
  const auth = authHeaders();
  if ("Authorization" in auth && auth.Authorization) {
    headers.set("Authorization", String(auth.Authorization));
  }
  return headers;
}

async function backendFetch(path: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(`${BASE}${path}`, {
      ...init,
      headers: mergeHeaders(init),
    });
  } catch {
    throw new Error(
      "Could not reach the API. Ensure the backend is running on port 8000.",
    );
  }
}

export async function fetchOverview(): Promise<OverviewResponse> {
  const path = "/overview";
  const res = await backendFetch(path);
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function login(payload: LoginPayload): Promise<LoginResponse> {
  const path = "/auth/login";
  const res = await backendFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function register(payload: RegisterPayload): Promise<AuthUser> {
  const path = "/auth/register";
  const res = await backendFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function fetchCurrentUser(): Promise<AuthUser> {
  const path = "/auth/me";
  const res = await backendFetch(path);
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function predictColdStart(
  req: ColdStartPredictRequest,
): Promise<ColdStartPredictResponse> {
  const path = "/cold-start/predict";
  const res = await backendFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function fetchColdStartStatus(): Promise<ColdStartStatusResponse> {
  const path = "/cold-start/status";
  const res = await backendFetch(path);
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function trainCameo(): Promise<TrainResponse> {
  const path = "/cold-start/train-cameo";
  const res = await backendFetch(path, { method: "POST" });
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function trainEmbedder(): Promise<TrainResponse> {
  return trainCameo();
}

export async function trainMaml(): Promise<TrainResponse> {
  return trainCameo();
}

export async function trainForecasting(
  req: TrainForecastingRequest,
): Promise<TrainForecastingResponse> {
  const path = "/forecasting/train";
  const res = await backendFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function fetchShieldXRStatus(): Promise<ShieldXRStatusResponse> {
  const path = "/forecasting/status";
  const res = await backendFetch(path);
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function fetchForecastingPerformance(params?: {
  drug_code?: string;
  model_name?: string;
  training_run_id?: string;
  limit?: number;
}): Promise<PerformanceMonitoringResponse> {
  const search = new URLSearchParams();
  if (params?.drug_code) search.set("drug_code", params.drug_code);
  if (params?.model_name) search.set("model_name", params.model_name);
  if (params?.training_run_id) search.set("training_run_id", params.training_run_id);
  if (params?.limit) search.set("limit", String(params.limit));
  const qs = search.toString();
  const path = `/forecasting/performance${qs ? `?${qs}` : ""}`;
  const res = await backendFetch(path);
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function predictForecast(
  req: ForecastRequest,
): Promise<ForecastResponse> {
  const path = "/forecasting/predict";
  const res = await backendFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function fetchHoldoutDateSuggestions(
  drugCode: string,
): Promise<HoldoutDateSuggestions> {
  const params = new URLSearchParams({ drug_code: drugCode.trim() });
  const path = `/forecasting/holdout/date-suggestions?${params}`;
  const res = await backendFetch(path);
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function runHoldoutValidation(
  req: HoldoutValidationRequest,
): Promise<HoldoutResponse> {
  const path = "/forecasting/holdout";
  const res = await backendFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function searchEnrichedDrugCodes(
  query: string,
  page = 1,
  pageSize = 10,
): Promise<PaginatedReceiptDrugSearchResponse> {
  const params = new URLSearchParams({
    q: query,
    page: String(page),
    page_size: String(pageSize),
  });
  const path = `/forecasting/enriched-drugs/search?${params}`;
  const res = await backendFetch(path);
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function searchReceiptDrugCodes(
  query: string,
  page = 1,
  pageSize = 10,
): Promise<PaginatedReceiptDrugSearchResponse> {
  const params = new URLSearchParams({
    q: query,
    page: String(page),
    page_size: String(pageSize),
  });
  const path = `/forecasting/receipt-drug-codes/search?${params}`;
  const res = await backendFetch(path);
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function searchForecastedDrugs(
  query?: string,
  page = 1,
  pageSize = 10,
): Promise<PaginatedDrugCodeSearchResponse> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  const trimmed = query?.trim();
  if (trimmed) {
    params.set("q", trimmed);
  }
  const path = `/forecasting/forecasted-drugs/search?${params}`;
  const res = await backendFetch(path);
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function uploadReceipts(file: File): Promise<UploadReceiptsResponse> {
  const path = "/upload-hospital-receipts";
  const form = new FormData();
  form.append("file", file);

  let res: Response;
  try {
    res = await fetch(`${backendOrigin()}${path}`, {
      method: "POST",
      headers: mergeHeaders(),
      body: form,
    });
  } catch {
    throw new Error(
      "Could not reach the API. Ensure the backend is running on port 8000.",
    );
  }
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function listDrugs(
  query?: string,
  page = 1,
  pageSize = 20,
): Promise<PaginatedDrugListResponse> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  const trimmed = query?.trim();
  if (trimmed) {
    params.set("q", trimmed);
  }
  const path = `/drugs?${params}`;
  const res = await backendFetch(path);
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function getDrugDetail(
  drugCode: string,
  lookbackDays = 365,
): Promise<DrugDetailResponse> {
  const params = new URLSearchParams({
    lookback_days: String(lookbackDays),
  });
  const path = `/drugs/${encodeURIComponent(drugCode.trim())}?${params}`;
  const res = await backendFetch(path);
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}

export async function listCategories(
  query?: string,
  page = 1,
  pageSize = 20,
): Promise<PaginatedCategoryListResponse> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  const trimmed = query?.trim();
  if (trimmed) {
    params.set("q", trimmed);
  }
  const path = `/categories?${params}`;
  const res = await backendFetch(path);
  if (!res.ok) {
    throw new Error(await parseError(res, path));
  }
  return res.json();
}
