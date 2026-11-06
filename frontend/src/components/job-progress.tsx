import { CircleAlert, CircleCheck, LoaderCircle, RotateCcw } from "lucide-react";
import type { JobState } from "./use-glasses-job";

/** The source photo with uploading, generating, ready or failed status, and a retry button. */
export function JobProgress({ state, onRetry }: { state: JobState; onRetry?: () => void }) {
  if (state.phase === "idle") return null;
  const image = state.sourceImageUrl;
  return (
    <div className="flex items-center gap-4 rounded-xl border border-neutral-200 p-3" role="status" aria-live="polite">
      {image && (
        // eslint-disable-next-line @next/next/no-img-element -- backend-served photo of unknown size
        <img src={image} alt="Photo used for your glasses" className="h-16 w-24 shrink-0 rounded-lg bg-neutral-50 object-contain" />
      )}
      <div className="min-w-0 flex-1 text-sm">
        {(state.phase === "uploading" || state.phase === "preparing") && (
          <p className="flex items-center gap-2 font-medium text-neutral-900">
            <LoaderCircle aria-hidden className="size-4 animate-spin text-neutral-500" />
            {state.phase === "uploading" ? "Uploading image…" : "Generating glasses…"}
          </p>
        )}
        {state.phase === "ready" && (
          <p className="flex items-center gap-2 font-medium text-emerald-700">
            <CircleCheck aria-hidden className="size-4" />
            Glasses generated. They appear in the live preview below.
          </p>
        )}
        {state.phase === "failed" && (
          <p className="flex items-start gap-2 text-red-600" role="alert">
            <CircleAlert aria-hidden className="mt-0.5 size-4 shrink-0" />
            <span>{state.message}</span>
          </p>
        )}
      </div>
      {state.phase === "failed" && onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex h-10 shrink-0 items-center gap-2 rounded-lg border border-neutral-200 px-3 text-sm font-medium text-neutral-700 hover:bg-neutral-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-950"
        >
          <RotateCcw aria-hidden className="size-4" />
          Try again
        </button>
      )}
    </div>
  );
}
