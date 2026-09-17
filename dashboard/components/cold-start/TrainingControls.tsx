"use client";

import { useCallback, useRef, useState } from "react";
import { Brain, CheckCircle2, Loader2, Sparkles, XCircle } from "lucide-react";

import { trainEmbedder, trainMaml } from "@/lib/api";
import type { CameoAccuracySummary } from "@/lib/types";

import { Panel, PanelBody, PanelHeader } from "./ui";

type TrainStatus = "idle" | "loading" | "success" | "error";

interface TrainState {
  status: TrainStatus;
  message: string;
}

const initialState: TrainState = { status: "idle", message: "" };

export interface TrainingCompletePayload {
  drugsTrained?: number;
  accuracySummary?: CameoAccuracySummary | null;
}

interface TrainingControlsProps {
  onModelReady?: (ready: boolean) => void;
  onTrainComplete?: (payload: TrainingCompletePayload) => void;
  /** @deprecated use onModelReady */
  onEmbedderReady?: (ready: boolean) => void;
}

export function TrainingControls({
  onModelReady,
  onTrainComplete,
  onEmbedderReady,
}: TrainingControlsProps) {
  const controlsRef = useRef<HTMLDivElement>(null);
  const [embedder, setEmbedder] = useState<TrainState>(initialState);
  const [maml, setMaml] = useState<TrainState>(initialState);

  const isBusy = embedder.status === "loading" || maml.status === "loading";

  const runTrain = useCallback(
    async (kind: "embedder" | "maml") => {
      const setState = kind === "embedder" ? setEmbedder : setMaml;
      setState({ status: "loading", message: "" });
      try {
        const result =
          kind === "embedder" ? await trainEmbedder() : await trainMaml();
        const count =
          kind === "embedder"
            ? result.drugs_trained_on
            : result.tasks_trained_on;
        const label = kind === "embedder" ? "drugs" : "tasks";
        setState({
          status: "success",
          message: `Ready — trained on ${count ?? 0} ${label}`,
        });
        if (kind === "embedder") {
          onModelReady?.(true);
          onEmbedderReady?.(true);
          onTrainComplete?.({
            drugsTrained: result.drugs_trained_on,
            accuracySummary: result.accuracy_summary,
          });
        }
      } catch (err) {
        setState({
          status: "error",
          message: err instanceof Error ? err.message : "Training failed",
        });
      }
    },
    [onEmbedderReady, onModelReady, onTrainComplete],
  );

  return (
    <Panel id="training-controls">
      <PanelHeader
        step={1}
        title="CAMEO model setup"
        description="Train the metric-learning cold-start model on matched-source drugs from the drugs table."
      />
      <PanelBody>
        <div ref={controlsRef} className="grid gap-4 sm:grid-cols-2">
          <TrainCard
            icon={<Brain className="h-5 w-5 text-blue-600" />}
            title="CAMEO library"
            description="Trains Module A (metric net) on real matched-source attributes + weekly demand shapes. Required before prediction."
            badge="Required"
            badgeColor="red"
            buttonLabel="Train CAMEO"
            state={embedder}
            disabled={isBusy}
            onClick={() => runTrain("embedder")}
          />
          <TrainCard
            icon={<Sparkles className="h-5 w-5 text-violet-600" />}
            title="Retrain (optional)"
            description="Re-run after updating the drugs Excel import or enriched demand panel."
            badge="Optional"
            badgeColor="slate"
            buttonLabel="Retrain CAMEO"
            state={maml}
            disabled={isBusy}
            onClick={() => runTrain("maml")}
          />
        </div>
      </PanelBody>
    </Panel>
  );
}

function TrainCard({
  icon,
  title,
  description,
  badge,
  badgeColor,
  buttonLabel,
  state,
  disabled,
  onClick,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  badge: string;
  badgeColor: "red" | "slate";
  buttonLabel: string;
  state: TrainState;
  disabled: boolean;
  onClick: () => void;
}) {
  const badgeStyles =
    badgeColor === "red"
      ? "bg-red-50 text-red-700 ring-red-100"
      : "bg-slate-100 text-slate-600 ring-slate-200";

  return (
    <div className="flex flex-col rounded-xl border border-slate-200 bg-slate-50/60 p-4">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-white shadow-sm ring-1 ring-slate-200">
            {icon}
          </span>
          <div>
            <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
            <span
              className={`mt-0.5 inline-block rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ring-1 ${badgeStyles}`}
            >
              {badge}
            </span>
          </div>
        </div>
      </div>
      <p className="mb-4 flex-1 text-sm leading-relaxed text-slate-600">{description}</p>
      <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {state.status === "loading" && <Loader2 className="h-4 w-4 animate-spin" />}
        {state.status === "success" && <CheckCircle2 className="h-4 w-4" />}
        {state.status === "error" && <XCircle className="h-4 w-4" />}
        {state.status === "loading" ? "Training…" : buttonLabel}
      </button>
      {state.status === "loading" && (
        <p className="mt-2 text-center text-xs text-slate-500">
          Large catalogs can take 1–2 minutes. Keep this tab open.
        </p>
      )}
      {state.status === "success" && (
        <p className="mt-2 text-center text-xs font-medium text-emerald-700">{state.message}</p>
      )}
      {state.status === "error" && (
        <p className="mt-2 text-center text-xs text-red-600">{state.message}</p>
      )}
    </div>
  );
}

export function scrollToTrainingControls() {
  document.getElementById("training-controls")?.scrollIntoView({ behavior: "smooth" });
}
