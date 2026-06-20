"use client";

import { useCallback, useState } from "react";

import { predictForecast } from "@/lib/api";
import type { ForecastRequest, ForecastResponse } from "@/lib/types";

interface UseForecastingPredictResult {
  data: ForecastResponse | null;
  error: string | null;
  isLoading: boolean;
  predict: (request: ForecastRequest) => Promise<void>;
  reset: () => void;
}

export function useForecastingPredict(): UseForecastingPredictResult {
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const predict = useCallback(async (request: ForecastRequest) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await predictForecast(request);
      if (response.error) {
        setData(null);
        setError(response.error);
        return;
      }
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
