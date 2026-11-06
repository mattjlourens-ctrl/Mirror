/**
 * Client for the Mirror backend (FastAPI, in ../backend). Until
 * NEXT_PUBLIC_MIRROR_API_URL is set, every call fails with a "not-configured"
 * ApiError so the UI never fakes a result.
 */

export type JobStatus = "queued" | "processing" | "succeeded" | "failed";

export type Job = {
  id: string;
  status: JobStatus;
  /** Absolute URL of the photo being turned into glasses. */
  sourceImageUrl: string;
  /** Absolute URL of the generated model; only for the browser renderer. */
  modelUrl: string | null;
  parts: string[];
  error: { code: string; message: string } | null;
};

export type Candidate = {
  id: string;
  imageUrl: string;
  width: number;
  height: number;
};

export type ProductLookup =
  | { status: "processing"; job: Job; candidates: Candidate[] }
  | { status: "choose"; candidates: Candidate[] };

/** A finished model, ready to hand to the try-on view. */
export type ReadyGlasses = { jobId: string; modelUrl: string; sourceImageUrl: string };

export type ApiErrorKind = "not-configured" | "network" | "http" | "invalid-response" | "timeout";

export class ApiError extends Error {
  constructor(
    readonly kind: ApiErrorKind,
    message: string,
    readonly status?: number,
    readonly code?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// Read directly so Next.js inlines the value at build time.
const API_URL = process.env.NEXT_PUBLIC_MIRROR_API_URL?.trim().replace(/\/+$/, "") ?? "";

export function isBackendConfigured(): boolean {
  return API_URL !== "";
}

function absolute(path: string): string {
  return /^https?:\/\//.test(path) ? path : `${API_URL}${path}`;
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  if (!isBackendConfigured()) {
    throw new ApiError("not-configured", "The Mirror backend isn't connected yet.");
  }

  let response: Response;
  try {
    response = await fetch(absolute(path), init);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError("network", "Couldn't reach Mirror. Check your connection and try again.");
  }

  if (!response.ok) {
    let message = `Something went wrong (${response.status}). Please try again.`;
    let code: string | undefined;
    try {
      const detail = (await response.json())?.detail;
      if (typeof detail === "string") message = detail;
      else if (detail && typeof detail.message === "string") {
        message = detail.message;
        code = typeof detail.code === "string" ? detail.code : undefined;
      }
    } catch {
      // Not JSON; keep the generic message.
    }
    throw new ApiError("http", message, response.status, code);
  }

  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError("invalid-response", "Mirror sent a response the page couldn't read.");
  }
}

type RawJob = {
  id: string;
  status: JobStatus;
  source_image_url: string;
  model_url: string | null;
  parts: string[];
  error: { code: string; message: string } | null;
};

type RawCandidate = { id: string; image_url: string; width: number; height: number };

function toJob(raw: RawJob): Job {
  return {
    id: raw.id,
    status: raw.status,
    sourceImageUrl: absolute(raw.source_image_url),
    modelUrl: raw.model_url ? absolute(raw.model_url) : null,
    parts: raw.parts,
    error: raw.error,
  };
}

function toCandidate(raw: RawCandidate): Candidate {
  return { id: raw.id, imageUrl: absolute(raw.image_url), width: raw.width, height: raw.height };
}

/** POST /api/jobs/from-image: start turning an uploaded photo into glasses. */
export async function uploadGlassesImage(image: File, signal?: AbortSignal): Promise<Job> {
  const body = new FormData();
  body.append("image", image);
  return toJob(await request<RawJob>("/api/jobs/from-image", { method: "POST", body, signal }));
}

/** POST /api/sources/from-url: find front-view product photos; starts a job when there is exactly one. */
export async function lookUpProductUrl(url: string, signal?: AbortSignal): Promise<ProductLookup> {
  const raw = await request<{ status: "processing" | "choose"; job: RawJob | null; candidates: RawCandidate[] }>(
    "/api/sources/from-url",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }), signal },
  );
  const candidates = raw.candidates.map(toCandidate);
  if (raw.status === "processing" && raw.job) return { status: "processing", job: toJob(raw.job), candidates };
  return { status: "choose", candidates };
}

/** POST /api/jobs/from-candidate: start turning a chosen product photo into glasses. */
export async function chooseCandidate(candidateId: string, signal?: AbortSignal): Promise<Job> {
  return toJob(
    await request<RawJob>("/api/jobs/from-candidate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ candidate_id: candidateId }),
      signal,
    }),
  );
}

export async function getJob(jobId: string, signal?: AbortSignal): Promise<Job> {
  return toJob(await request<RawJob>(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "GET", signal }));
}

const POLL_INTERVAL_MS = 700;
const JOB_TIMEOUT_MS = 2 * 60 * 1000;

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

/** Poll a job until it succeeds or fails. */
export async function waitForJob(job: Job, signal?: AbortSignal): Promise<Job> {
  const started = Date.now();
  let current = job;
  while (current.status === "queued" || current.status === "processing") {
    if (Date.now() - started > JOB_TIMEOUT_MS) {
      throw new ApiError("timeout", "Preparing your glasses is taking too long. Please try again.");
    }
    await sleep(POLL_INTERVAL_MS, signal);
    current = await getJob(current.id, signal);
  }
  return current;
}

export function readyGlasses(job: Job): ReadyGlasses | null {
  return job.status === "succeeded" && job.modelUrl
    ? { jobId: job.id, modelUrl: job.modelUrl, sourceImageUrl: job.sourceImageUrl }
    : null;
}
