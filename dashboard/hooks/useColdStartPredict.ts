"use client";

import { useCallback, useState } from "react";

import { predictColdStart } from "@/lib/api";
import type { ColdStartPredictRequest, ColdStartPredictResponse } from "@/lib/types";

interface UseColdStartPredictResult {
  data: ColdStartPredictResponse | null;
  error: string | null;
  isLoading: boolean;
  predict: (request: ColdStartPredictRequest) => Promise<void>;
  reset: () => void;
}

export function useColdStartPredict(): UseColdStartPredictResult {
  const [data, setData] = useState<ColdStartPredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const predict = useCallback(async (request: ColdStartPredictRequest) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await predictColdStart(request);
      setData(response);
    } catch (err) {
      setData(null);
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return { data, error, isLoading, predict, reset };
}
