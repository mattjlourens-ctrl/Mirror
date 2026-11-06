import { useEffect, useId, useRef, useState, type DragEvent, type SubmitEvent } from "react";
import { ImageUp, RefreshCw, X } from "lucide-react";
import { uploadGlassesImage, type ReadyGlasses } from "@/lib/api";
import { ACCEPTED_IMAGE_INPUT, MAX_IMAGE_BYTES, formatBytes, validateImageFile } from "@/lib/validation";
import { StatusMessage, type Status } from "./status-message";
import { SubmitButton } from "./submit-button";
import { failureStatus } from "./describe-api-result";
import { JobProgress } from "./job-progress";
import { useGlassesJob } from "./use-glasses-job";

type Selected = { file: File; previewUrl: string };

export function ImageUpload({ onReady }: { onReady: (glasses: ReadyGlasses) => void }) {
  const inputId = useId();
  const statusId = useId();
  const fileInput = useRef<HTMLInputElement>(null);
  const [selected, setSelected] = useState<Selected | null>(null);
  const [status, setStatus] = useState<Status>({ tone: "idle" });
  const [dragging, setDragging] = useState(false);
  const job = useGlassesJob(onReady);

  // Release the object URL whenever the preview changes or the component unmounts.
  useEffect(() => () => {
    if (selected) URL.revokeObjectURL(selected.previewUrl);
  }, [selected]);

  function choose(file: File | undefined) {
    if (!file) return;
    const result = validateImageFile(file);
    if (!result.ok) {
      setStatus({ tone: "error", message: result.error });
      return;
    }
    job.cancel();
    setSelected({ file, previewUrl: URL.createObjectURL(file) });
    setStatus({ tone: "idle" });
  }

  function remove() {
    job.cancel();
    setSelected(null);
    setStatus({ tone: "idle" });
    if (fileInput.current) fileInput.current.value = "";
    fileInput.current?.focus();
  }

  function handleDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    setDragging(false);
    choose(event.dataTransfer.files[0]);
  }

  async function submit() {
    if (!selected) {
      setStatus({ tone: "error", message: "Choose an image first." });
      return;
    }
    job.cancel();
    const signal = job.begin();
    setStatus({ tone: "loading", message: "Uploading your photo…" });
    try {
      const started = await uploadGlassesImage(selected.file, signal);
      setStatus({ tone: "idle" });
      await job.follow(started, signal);
    } catch (error) {
      if (signal.aborted) return;
      setStatus(failureStatus(error, "Your image is ready."));
    }
  }

  function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    void submit();
  }

  return (
    <form onSubmit={handleSubmit} noValidate className="flex flex-col gap-3">
      {/* The input stays in the DOM (visually hidden) so the label and keyboard focus work. */}
      <input
        ref={fileInput}
        id={inputId}
        type="file"
        accept={ACCEPTED_IMAGE_INPUT}
        className="peer sr-only"
        aria-describedby={statusId}
        onChange={(event) => choose(event.target.files?.[0])}
      />

      {selected ? (
        <div className="flex flex-col gap-4 rounded-xl border border-neutral-200 p-3 sm:flex-row sm:items-center">
          {/* eslint-disable-next-line @next/next/no-img-element -- local blob preview; next/image can't optimize it */}
          <img
            src={selected.previewUrl}
            alt={`Preview of ${selected.file.name}`}
            className="h-40 w-full rounded-lg bg-neutral-50 object-contain sm:h-24 sm:w-36"
          />
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-neutral-900">{selected.file.name}</p>
            <p className="text-sm text-neutral-500">{formatBytes(selected.file.size)}</p>
          </div>
          <div className="flex gap-2">
            <label
              htmlFor={inputId}
              className="inline-flex h-10 cursor-pointer items-center gap-2 rounded-lg border border-neutral-200 px-3 text-sm font-medium text-neutral-700 hover:bg-neutral-50 peer-focus-visible:ring-2 peer-focus-visible:ring-neutral-900/20"
            >
              <RefreshCw aria-hidden className="size-4" />
              Replace
            </label>
            <button
              type="button"
              onClick={remove}
              className="inline-flex h-10 items-center gap-2 rounded-lg border border-neutral-200 px-3 text-sm font-medium text-neutral-700 hover:bg-neutral-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-950"
            >
              <X aria-hidden className="size-4" />
              Remove
            </button>
          </div>
        </div>
      ) : (
        <label
          htmlFor={inputId}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed px-6 py-8 text-center transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-neutral-900/20 ${
            dragging ? "border-neutral-500 bg-neutral-50" : "border-neutral-300 hover:bg-neutral-50"
          }`}
        >
          <ImageUp aria-hidden className="size-6 text-neutral-400" />
          <span className="text-base font-medium text-neutral-900">Choose an image or drag it here</span>
          <span className="text-sm text-neutral-500">
            PNG or JPG, up to {formatBytes(MAX_IMAGE_BYTES)}. A front view on a plain white background works best.
          </span>
        </label>
      )}

      <div className="flex justify-end">
        <SubmitButton loading={status.tone === "loading" || job.state.phase === "preparing"} disabled={!selected}>
          Find Glasses
        </SubmitButton>
      </div>
      <StatusMessage id={statusId} status={status} />
      <JobProgress state={job.state} onRetry={() => void submit()} />
    </form>
  );
}
