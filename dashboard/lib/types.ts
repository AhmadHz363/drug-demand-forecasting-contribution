export interface DrugMetadataInput {
  drug_code: string;
  drug_name: string;
  therapeutic_class: string;
  atc_category: string;
  pharmaceutical_form: string;
  ven_class: "V" | "E" | "N";
  abc_class: "A" | "B" | "C";
  unit_price_tier: 1 | 2 | 3 | 4 | 5;
  requires_refrigeration: boolean;
  is_controlled_substance: boolean;
  average_shelf_life_days: number;
  route_of_administration: string;
}

export interface PharmacistEstimate {
  weekly_units: number;
  confidence: number;
}

export interface ColdStartPredictRequest {
  drug_metadata: DrugMetadataInput;
  pharmacist_estimate?: PharmacistEstimate;
  forecast_horizon_days: number;
}

export interface DailyForecast {
  date: string;
  p10: number;
  p50: number;
  p90: number;
}

export type GraduationStage =
  | "cold_start_only"
  | "blended"
  | "full_ensemble";

export interface ColdStartPredictResponse {
  drug_code: string;
  stage_used: GraduationStage;
  observation_count: number;
  embedding: number[];
  nearest_neighbours: string[];
  similarity_scores: number[];
  forecast: DailyForecast[];
  uncertainty_note: string;
}

export interface TrainResponse {
  status: string;
  drugs_trained_on?: number;
  tasks_trained_on?: number;
  artifact_path: string;
}

export interface ForecastRequest {
  drug_code: string;
  horizon_days: number;
  center_syn_id?: string;
  include_shap: boolean;
  include_attention: boolean;
}

export interface TrainForecastingRequest {
  drug_codes?: string[];
  models: string[];
  force_retrain: boolean;
}

export interface ModelWeightBreakdown {
  sarima: number;
  lgbm: number;
  tft: number;
}

export interface ShapFeature {
  feature_name: string;
  shap_value: number;
  feature_value: number;
}

export interface AttentionWeight {
  week_offset: number;
  weight: number;
}

export interface ForecastResponse {
  drug_code: string;
  center_syn_id: string | null;
  horizon_days: number;
  model_weights: ModelWeightBreakdown;
  forecast: DailyForecast[];
  shap_features?: ShapFeature[] | null;
  attention_weights?: AttentionWeight[] | null;
  uncertainty_note: string;
  smape_last_validation?: number | null;
  error?: string | null;
}

export interface TrainForecastingResponse {
  status: string;
  drugs_trained: number;
  models_trained: string[];
  smape_summary: Record<string, number>;
  artifacts_saved: string[];
}

export interface PeriodRange {
  start: string;
  end: string;
}

export interface HoldoutMetrics {
  smape: number;
  mae: number;
  coverage_90: number;
  accuracy_pct: number;
}

export interface HoldoutSeriesPoint {
  date: string;
  actual: number;
  sarima_p50?: number | null;
  lgbm_p50?: number | null;
  tft_p50?: number | null;
  ensemble_p50?: number | null;
  ensemble_p10?: number | null;
  ensemble_p90?: number | null;
}

export interface HoldoutDateDefaults {
  train_end: string;
  test_start: string;
  test_end: string;
}

export interface HoldoutDateSuggestions {
  data_start: string;
  data_end: string;
  defaults: HoldoutDateDefaults;
  train_end_options: string[];
  test_start_options: string[];
  test_end_options: string[];
}

export interface HoldoutValidationRequest {
  drug_code: string;
  train_end?: string;
  test_start?: string;
  test_end?: string;
  train_start?: string;
  center_syn_id?: string;
  models?: string[];
}

export interface HoldoutResponse {
  drug_code: string;
  center_syn_id: string | null;
  train_period: PeriodRange;
  test_period: PeriodRange;
  models_evaluated: string[];
  model_errors: Record<string, string>;
  metrics: Record<string, HoldoutMetrics>;
  total_accuracy_pct: number;
  model_weights: ModelWeightBreakdown;
  series: HoldoutSeriesPoint[];
}

export interface ReceiptDrugOption {
  drug_code: string;
  drug_name?: string | null;
}

export interface PaginatedDrugCodeSearchResponse {
  items: string[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface PaginatedReceiptDrugSearchResponse {
  items: ReceiptDrugOption[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ReceiptRowError {
  row_index: number;
  message: string;
}

export interface UploadReceiptsResponse {
  inserted_rows: number;
  failed_rows: number;
  errors: ReceiptRowError[];
}

export interface DrugItem {
  id: number;
  drug_code: string;
  drug_name?: string | null;
  drug_category?: string | null;
  receipt_count: number;
  created_at: string;
  updated_at: string;
}

export interface PaginatedDrugListResponse {
  items: DrugItem[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface CategoryItem {
  id: number;
  category_code: string;
  name?: string | null;
  receipt_count: number;
  drug_count: number;
  created_at: string;
  updated_at: string;
}

export interface PaginatedCategoryListResponse {
  items: CategoryItem[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface AuthUser {
  id: number;
  email: string;
  is_active: boolean;
  created_at: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export type RegisterPayload = LoginPayload;

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: AuthUser;
}
