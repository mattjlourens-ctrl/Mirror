"use client";

import { useCallback, useState } from "react";
import type { ReadyGlasses } from "@/lib/api";
import { CameraPreview } from "./camera-preview";
import { SourceTabs, type TabKey } from "./source-tabs";
import type { JobPhase } from "./use-glasses-job";

/** Connects the source forms to the preview: a finished model goes straight to try-on. */
export function TryOnExperience() {
  const [tab, setTab] = useState<TabKey>("link");
  const [glasses, setGlasses] = useState<ReadyGlasses | null>(null);
  const [jobPhase, setJobPhase] = useState<JobPhase>("idle");
  const handleReady = useCallback((ready: ReadyGlasses) => setGlasses(ready), []);
  const handlePhase = useCallback((phase: JobPhase) => setJobPhase(phase), []);

  return (
    <>
      <section aria-labelledby="page-heading" className="mx-auto flex w-full max-w-2xl flex-col gap-8">
        <div className="flex flex-col gap-3 text-center">
          <h1 id="page-heading" className="text-3xl font-semibold tracking-tight text-neutral-950 sm:text-5xl">
            Try on any pair of glasses
          </h1>
          <p className="text-base text-neutral-500 sm:text-lg">Paste a product link or upload a picture to get started.</p>
        </div>
        <SourceTabs active={tab} onActiveChange={setTab} onReady={handleReady} onPhase={handlePhase} />
      </section>
      <CameraPreview glasses={glasses} jobPhase={jobPhase} />
    </>
  );
}
