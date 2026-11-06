import { useId, useState, type SubmitEvent } from "react";
import { ImageUp, Link2 } from "lucide-react";
import { chooseCandidate, lookUpProductUrl, type Candidate, type ReadyGlasses } from "@/lib/api";
import { validateProductUrl } from "@/lib/validation";
import { StatusMessage, type Status } from "./status-message";
import { SubmitButton } from "./submit-button";
import { errorMessage, failureStatus, suggestsUpload } from "./describe-api-result";
import { CandidatePicker } from "./candidate-picker";
import { JobProgress } from "./job-progress";
import { useGlassesJob } from "./use-glasses-job";

export function ProductLinkForm({
  onReady,
  onUseUpload,
}: {
  onReady: (glasses: ReadyGlasses) => void;
  onUseUpload: () => void;
}) {
  const inputId = useId();
  const statusId = useId();
  const [value, setValue] = useState("");
  const [status, setStatus] = useState<Status>({ tone: "idle" });
  const [offerUpload, setOfferUpload] = useState(false);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [chosen, setChosen] = useState<Candidate | null>(null);
  const job = useGlassesJob(onReady);

  async function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    const result = validateProductUrl(value);
    if (!result.ok) {
      setStatus({ tone: "error", message: result.error });
      return;
    }

    job.cancel();
    setCandidates([]);
    setChosen(null);
    setOfferUpload(false);
    const signal = job.begin();
    setStatus({ tone: "loading", message: "Finding product photos on that page…" });
    try {
      const lookup = await lookUpProductUrl(result.value, signal);
      setStatus({ tone: "idle" });
      setCandidates(lookup.candidates);
      if (lookup.status === "processing") await job.follow(lookup.job, signal);
    } catch (error) {
      if (signal.aborted) return;
      setStatus(failureStatus(error, "That link looks good."));
      setOfferUpload(suggestsUpload(error));
    }
  }

  async function choose(candidate: Candidate) {
    setChosen(candidate);
    const signal = job.begin();
    job.setState({ phase: "preparing", sourceImageUrl: candidate.imageUrl });
    try {
      await job.follow(await chooseCandidate(candidate.id, signal), signal);
    } catch (error) {
      if (signal.aborted) return;
      job.setState({ phase: "failed", sourceImageUrl: candidate.imageUrl, message: errorMessage(error) });
    }
  }

  const invalid = status.tone === "error";
  const busy = status.tone === "loading" || job.state.phase === "preparing";
  const picking = candidates.length > 1 && !chosen && job.state.phase === "idle";
  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={handleSubmit} noValidate className="flex flex-col gap-3">
        <label htmlFor={inputId} className="sr-only">
          Product page URL
        </label>
        <div className="flex flex-col gap-3 sm:flex-row">
          <div className="relative flex-1">
            <Link2 aria-hidden className="pointer-events-none absolute top-1/2 left-4 size-5 -translate-y-1/2 text-neutral-400" />
            <input
              id={inputId}
              type="url"
              inputMode="url"
              autoComplete="url"
              spellCheck={false}
              placeholder="https://www.example.com/glasses/frame-name"
              value={value}
              onChange={(event) => {
                setValue(event.target.value);
                if (status.tone === "error") setStatus({ tone: "idle" });
              }}
              aria-invalid={invalid}
              aria-describedby={statusId}
              className={`h-14 w-full rounded-xl border bg-white pr-4 pl-12 text-base text-neutral-900 placeholder:text-neutral-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-900/10 ${
                invalid ? "border-red-400 focus:border-red-500" : "border-neutral-200 focus:border-neutral-400"
              }`}
            />
          </div>
          <SubmitButton loading={busy}>Find Glasses</SubmitButton>
        </div>
        <StatusMessage id={statusId} status={status} />
        {offerUpload && (
          <button
            type="button"
            onClick={onUseUpload}
            className="inline-flex h-10 items-center gap-2 self-start rounded-lg border border-neutral-200 px-3 text-sm font-medium text-neutral-700 hover:bg-neutral-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-950"
          >
            <ImageUp aria-hidden className="size-4" />
            Upload a photo instead
          </button>
        )}
      </form>

      {picking && <CandidatePicker candidates={candidates} onChoose={choose} disabled={busy} />}
      <JobProgress state={job.state} onRetry={chosen ? () => choose(chosen) : undefined} />
      {job.state.phase === "failed" && candidates.length > 1 && (
        <button
          type="button"
          onClick={() => {
            job.cancel();
            setChosen(null);
          }}
          className="self-start text-sm font-medium text-neutral-700 underline underline-offset-4 hover:text-neutral-950"
        >
          Choose a different photo
        </button>
      )}
    </div>
  );
}
