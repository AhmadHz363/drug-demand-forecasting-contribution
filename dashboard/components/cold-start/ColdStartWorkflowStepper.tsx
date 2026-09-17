"use client";

import { Check, Circle } from "lucide-react";

export type ColdStartWorkflowStep = "train" | "validate" | "forecast";

interface ColdStartWorkflowStepperProps {
  modelReady: boolean;
  hasValidationMetrics: boolean;
  hasPrediction: boolean;
  activeStep: ColdStartWorkflowStep;
}

const STEPS: { id: ColdStartWorkflowStep; label: string; hint: string }[] = [
  {
    id: "train",
    label: "Train CAMEO",
    hint: "Metric-learning library on matched-source drugs",
  },
  {
    id: "validate",
    label: "Review validation",
    hint: "Leave-drugs-out WAPE · SB class breakdown",
  },
  {
    id: "forecast",
    label: "Cold-start forecast",
    hint: "New drug metadata + optional observations",
  },
];

function stepComplete(
  id: ColdStartWorkflowStep,
  modelReady: boolean,
  hasValidationMetrics: boolean,
  hasPrediction: boolean,
): boolean {
  if (id === "train") return modelReady;
  if (id === "validate") return hasValidationMetrics;
  if (id === "forecast") return hasPrediction;
  return false;
}

export function ColdStartWorkflowStepper({
  modelReady,
  hasValidationMetrics,
  hasPrediction,
  activeStep,
}: ColdStartWorkflowStepperProps) {
  return (
    <nav
      aria-label="CAMEO workflow"
      className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm"
    >
      <ol className="grid gap-3 sm:grid-cols-3 sm:gap-4">
        {STEPS.map((step, index) => {
          const done = stepComplete(
            step.id,
            modelReady,
            hasValidationMetrics,
            hasPrediction,
          );
          const isActive = step.id === activeStep;
          const stepNum = index + 1;

          return (
            <li
              key={step.id}
              className={`flex items-center gap-3 rounded-xl px-3 py-2.5 transition ${
                isActive ? "bg-blue-50 ring-1 ring-blue-200" : "bg-transparent"
              }`}
            >
              <span
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-semibold ${
                  done
                    ? "bg-emerald-500 text-white"
                    : isActive
                      ? "bg-blue-600 text-white"
                      : "bg-slate-100 text-slate-500"
                }`}
              >
                {done ? <Check className="h-4 w-4" /> : stepNum}
              </span>
              <div className="min-w-0">
                <p
                  className={`text-sm font-semibold ${
                    isActive ? "text-blue-900" : done ? "text-emerald-800" : "text-slate-700"
                  }`}
                >
                  {step.label}
                </p>
                <p className="truncate text-xs text-slate-500">{step.hint}</p>
              </div>
              {!done && !isActive && index > 0 && (
                <Circle className="ml-auto hidden h-3 w-3 text-slate-300 sm:block" />
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

export function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}
