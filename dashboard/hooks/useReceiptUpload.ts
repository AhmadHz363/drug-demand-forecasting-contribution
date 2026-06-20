"use client";

import { useCallback, useState } from "react";

import { uploadReceipts } from "@/lib/api";
import type { UploadReceiptsResponse } from "@/lib/types";

export type UploadStatus = "idle" | "uploading" | "success" | "error";

export interface ReceiptUploadState {
  status: UploadStatus;
  result: UploadReceiptsResponse | null;
  error: string | null;
  fileName: string | null;
  fileSize: number | null;
}

const initialState: ReceiptUploadState = {
  status: "idle",
  result: null,
  error: null,
  fileName: null,
  fileSize: null,
};

export function useReceiptUpload() {
  const [state, setState] = useState<ReceiptUploadState>(initialState);

  const upload = useCallback(async (file: File) => {
    setState({
      status: "uploading",
      result: null,
      error: null,
      fileName: file.name,
      fileSize: file.size,
    });

    try {
      const result = await uploadReceipts(file);
      setState({
        status: "success",
        result,
        error: null,
        fileName: file.name,
        fileSize: file.size,
      });
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Upload failed";
      setState({
        status: "error",
        result: null,
        error: message,
        fileName: file.name,
        fileSize: file.size,
      });
      throw err;
    }
  }, []);

  const reset = useCallback(() => {
    setState(initialState);
  }, []);

  return {
    ...state,
    isUploading: state.status === "uploading",
    upload,
    reset,
  };
}
