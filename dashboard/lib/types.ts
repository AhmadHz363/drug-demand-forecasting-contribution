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
  recommended_quantity?: number | null;
}

export interface InferenceHealth {
  demand_segment: string;
  used_stacking: boolean;
  used_conformal: boolean;
  used_spread_fallback: boolean;
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
  include_shap?: boolean;
  include_attention?: boolean;
}

export interface TrainForecastingRequest {
  drug_codes?: string[];
  models: string[];
  force_retrain: boolean;
}

export interface ModelWeightBreakdown {
  sarima: number;
  lgbm: number;
  classical: number;
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
  history?: DailyDemandPoint[];
  shap_features?: ShapFeature[] | null;
  attention_weights?: AttentionWeight[] | null;
  uncertainty_note: string;
  smape_last_validation?: number | null;
  smape_validation_full_window?: number | null;
  smape_validation_normal_supply?: number | null;
  mase_validation_full_window?: number | null;
  mase_validation_normal_supply?: number | null;
  mase_beats_baseline_full_window?: boolean | null;
  mase_beats_baseline_normal_supply?: boolean | null;
  mase_skill_validation_full_window?: number | null;
  smape_validation_7day_total?: number | null;
  smape_validation_30day_total?: number | null;
  mase_validation_7day_total?: number | null;
  mase_validation_30day_total?: number | null;
  smape_validation_7day_normal_supply?: number | null;
  smape_validation_30day_normal_supply?: number | null;
  mase_validation_7day_normal_supply?: number | null;
  mase_validation_30day_normal_supply?: number | null;
  ven_class?: string | null;
  operating_quantile?: number | null;
  recommended_quantity_total?: number | null;
  inference_health?: InferenceHealth | null;
  demand_segment?: string | null;
  validation_zero_actual_fraction?: number | null;
  smape_validation_unreliable?: boolean | null;
  primary_validation_metric?: string | null;
  validation_metrics_note?: string | null;
  error?: string | null;
}

export interface DrugQualitySummary {
  drug_code: string;
  status: string;
  reasons: string[];
}

export interface ShieldXRAccuracySummary {
  daily_ensemble_accuracy_pct?: number;
  hospital_weekly_accuracy_pct?: number;
  reconciled_weekly_per_drug_mean_accuracy_pct?: number;
  reconciled_weekly_per_drug_median_accuracy_pct?: number;
  volume_weighted_weekly_accuracy_pct?: number;
  non_lumpy_weekly_accuracy_pct?: number;
  hybrid_abc_combined_accuracy_pct?: number;
  per_sb_class_weekly_accuracy_pct?: Record<string, number>;
}

export interface ShieldXRStatusResponse {
  models_ready: boolean;
  training_run_id?: string | null;
  train_end?: string | null;
  valid_end?: string | null;
  n_skus?: number | null;
  accuracy_summary?: ShieldXRAccuracySummary;
}

export interface TrainForecastingResponse {
  status: string;
  training_run_id?: string;
  drugs_trained: number;
  models_trained: string[];
  smape_summary: Record<string, number>;
  accuracy_summary?: ShieldXRAccuracySummary;
  artifacts_saved: string[];
  skipped_drugs?: DrugQualitySummary[];
  flagged_drugs?: DrugQualitySummary[];
  drift_alerts?: string[];
}

export interface ModelPerformanceRow {
  drug_code: string;
  model_name: string;
  smape: number;
  mase?: number | null;
  coverage_90: number;
  demand_segment?: string | null;
  data_quality_status?: string | null;
  training_run_id?: string | null;
  weight_sarima?: number | null;
  weight_lgbm?: number | null;
  weight_classical?: number | null;
  weights_as_of?: string | null;
  smape_drift_pct?: number | null;
  mase_drift_pct?: number | null;
  drift_detected: boolean;
  evaluated_at: string;
}

export interface PerformanceMonitoringResponse {
  items: ModelPerformanceRow[];
  total: number;
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
  mase?: number;
  rmsse?: number;
  pinball_p10?: number;
  pinball_p50?: number;
  pinball_p90?: number;
  accuracy_skill_pct?: number;
}

export interface HoldoutSeriesPoint {
  date: string;
  actual: number;
  sarima_p50?: number | null;
  lgbm_p50?: number | null;
  classical_p50?: number | null;
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
  demand_segment: string;
  total_accuracy_pct: number;
  total_accuracy_skill_pct: number;
  primary_accuracy_metric?: string;
  smape_unreliable?: boolean;
  validation_zero_actual_fraction?: number;
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
  raw_inserted_rows: number;
  raw_failed_rows: number;
  enriched_inserted_rows: number;
  filtered_out_rows: number;
  /** Legacy alias for raw_inserted_rows */
  inserted_rows: number;
  /** Legacy alias for raw_failed_rows */
  failed_rows: number;
  errors: ReceiptRowError[];
}

export interface DrugItem {
  id: number;
  drug_code: string;
  drug_name?: string | null;
  drug_category?: string | null;
  receipt_count: number;
  distinct_receipt_days: number;
  first_receipt_date?: string | null;
  last_receipt_date?: string | null;
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

export interface DailyDemandPoint {
  date: string;
  quantity: number;
}

export interface DrugDetailResponse extends DrugItem {
  total_quantity: number;
  distinct_receipt_days: number;
  first_receipt_date?: string | null;
  last_receipt_date?: string | null;
  center_count: number;
  avg_daily_quantity?: number | null;
  observation_count: number;
  graduation_stage: GraduationStage;
  lookback_days: number;
  demand_series: DailyDemandPoint[];
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

// ── Overview ──────────────────────────────────────────────────────────────────

export interface TimeSeriesPoint {
  period: string;
  receipt_count: number;
  total_quantity: number;
}

export interface CenterSummary {
  center_syn_id: string;
  receipt_count: number;
}

export interface ModelPerformanceSummary {
  model_name: string;
  avg_smape: number;
  avg_coverage_90: number;
  record_count: number;
}

export interface DrugSummary {
  drug_code: string;
  drug_name?: string | null;
  drug_category?: string | null;
  receipt_count: number;
}

export interface CategorySummary {
  category_code: string;
  name?: string | null;
  drug_count: number;
  receipt_count: number;
}

export interface GraduationDistribution {
  cold_start_only: number;
  blended: number;
  full_ensemble: number;
}

export interface IngestionKPIs {
  total_receipt_rows: number;
  distinct_drug_codes: number;
  distinct_centers: number;
  first_receipt_date?: string | null;
  last_receipt_date?: string | null;
  total_quantity: number;
  receipt_trend: TimeSeriesPoint[];
  top_centers: CenterSummary[];
}

export interface DrugRegistryKPIs {
  total_drugs: number;
  graduation_distribution: GraduationDistribution;
  top_drugs: DrugSummary[];
}

export interface CategoryRegistryKPIs {
  total_categories: number;
  top_categories: CategorySummary[];
}

export interface ColdStartKPIs {
  embedder_trained: boolean;
  maml_trained: boolean;
}

export interface ForecastingKPIs {
  total_forecasted_drugs: number;
  avg_smape?: number | null;
  avg_coverage_90?: number | null;
  models_evaluated: string[];
  latest_training_run_id?: string | null;
  total_performance_records: number;
  model_performance: ModelPerformanceSummary[];
}

export interface OverviewResponse {
  ingestion: IngestionKPIs;
  drug_registry: DrugRegistryKPIs;
  category_registry: CategoryRegistryKPIs;
  cold_start: ColdStartKPIs;
  forecasting: ForecastingKPIs;
  generated_at: string;
}

// ── Auth ──────────────────────────────────────────────────────────────────────

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
