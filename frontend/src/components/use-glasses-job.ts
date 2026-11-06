import { useCallback, useEffect, useRef, useState } from "react";
import { readyGlasses, waitForJob, type Job, type ReadyGlasses } from "@/lib/api";
import { errorMessage } from "./describe-api-result";

export type JobPhase = "idle" | "uploading" | "preparing" | "ready" | "failed";

export type JobState =
  | { phase: "idle" }
  | { phase: "uploading"; sourceImageUrl?: undefined }
  | { phase: "preparing"; sourceImageUrl: string }
  | { phase: "ready"; sourceImageUrl: string }
  | { phase: "failed"; sourceImageUrl?: string; message: string };

/**
 * Follows a reconstruction job to the end and hands a finished model to
 * `onReady`. `start` takes a function that creates the job, so callers can
 * retry by calling it again.
 */
export function useGlassesJob(onReady: (glasses: ReadyGlasses) => void, onPhase?: (phase: JobPhase) => void) {
  const [state, setState] = useState<JobState>({ phase: "idle" });
  const controller = useRef<AbortController | null>(null);

  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => onPhase?.(state.phase), [state.phase, onPhase]);

  const cancel = useCallback(() => {
    controller.current?.abort();
    controller.current = null;
    setState({ phase: "idle" });
  }, []);

  const follow = useCallback(
    async (job: Job, signal: AbortSignal) => {
      setState({ phase: "preparing", sourceImageUrl: job.sourceImageUrl });
      try {
        const finished = await waitForJob(job, signal);
        const ready = readyGlasses(finished);
        if (ready) {
          setState({ phase: "ready", sourceImageUrl: finished.sourceImageUrl });
          onReady(ready);
        } else {
          setState({
            phase: "failed",
            sourceImageUrl: finished.sourceImageUrl,
            message: finished.error?.message ?? "Mirror couldn't prepare these glasses. Please try again.",
          });
        }
      } catch (error) {
        if (signal.aborted) return;
        setState({
          phase: "failed",
          sourceImageUrl: job.sourceImageUrl,
          message: errorMessage(error),
        });
      }
    },
    [onReady],
  );

  /** Abort anything in flight and return a fresh signal for a new attempt. */
  const begin = useCallback(() => {
    controller.current?.abort();
    controller.current = new AbortController();
    return controller.current.signal;
  }, []);

  return { state, setState, follow, begin, cancel };
}
