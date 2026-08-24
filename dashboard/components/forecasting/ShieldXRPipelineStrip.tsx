"use client";

import { Check } from "lucide-react";

import { SHIELD_XR_PIPELINE_STAGES } from "@/lib/shieldXrArchitecture";

interface ShieldXRPipelineStripProps {
  modelsReady: boolean;
  activeStageIndex?: number;
}

export function ShieldXRPipelineStrip({
  modelsReady,
  activeStageIndex,
}: ShieldXRPipelineStripProps) {
  const resolvedActive =
    activeStageIndex ?? (modelsReady ? SHIELD_XR_PIPELINE_STAGES.length - 1 : 0);

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-5 py-3 sm:px-6">
        <p className="text-xs font-semibold uppercase tracking-widest text-slate-500">
          SHIELD-XR architecture
        </p>
        <p className="mt-0.5 text-sm text-slate-600">
          Five-stage pipeline — AnomalyGuard → class-conditional ensemble → weekly reconciliation →
          hybrid ABC reporting
        </p>
      </div>
      <ol className="grid gap-px bg-slate-100 lg:grid-cols-5">
        {SHIELD_XR_PIPELINE_STAGES.map((step, index) => {
          const done = modelsReady && index <= resolvedActive;
          const active = !modelsReady && index === resolvedActive;

          return (
            <li
              key={step.id}
              className={`relative bg-white px-3 py-4 sm:px-4 ${
                active ? "ring-1 ring-inset ring-blue-200" : ""
              }`}
            >
              <div className="flex items-start gap-2.5">
                <span
                  className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-xs font-bold ${
                    done
                      ? "bg-emerald-50 text-emerald-600"
                      : active
                        ? "bg-blue-50 text-blue-600"
                        : "bg-slate-100 text-slate-400"
                  }`}
                >
                  {done ? <Check className="h-4 w-4" /> : index + 1}
                </span>
                <div className="min-w-0">
                  <p className="text-xs font-semibold leading-tight text-slate-900 sm:text-sm">
                    {step.title}
                  </p>
                  <p className="mt-1 text-[11px] leading-snug text-slate-500 sm:text-xs">
                    {step.detail}
                  </p>
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
