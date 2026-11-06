import { resolveApiUrl } from "./api-config";
import { httpError, networkFailureMessage } from "./api-errors";
import type { FitHints } from "./ar/auto-fit";

/**
 * Client for the Mirror backend (FastAPI, in ../backend), at
 * NEXT_PUBLIC_MIRROR_API_URL (http://localhost:8000 during development). A
 * production build without it fails every call with a "not-configured"
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
  /** Hints for placing the model on a face; null if unavailable. */
  fitHints: FitHints | null;
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
export type ReadyGlasses = { jobId: string; modelUrl: string; sourceImageUrl: string; fitHints: FitHints | null };

/** "network": backend not answering; "blocked": backend up but refused this page (CORS). */
export type ApiErrorKind = "not-configured" | "network" | "blocked" | "http" | "invalid-response" | "timeout";

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

// Read directly so Next.js inlines the values at build time.
const API_URL = resolveApiUrl(process.env.NEXT_PUBLIC_MIRROR_API_URL, process.env.NODE_ENV);

export function isBackendConfigured(): boolean {
  return API_URL !== "";
}

function absolute(path: string): string {
  return /^https?:\/\//.test(path) ? path : `${API_URL}${path}`;
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  if (!isBackendConfigured()) {
    throw new ApiError(
      "not-configured",
      "This site has no Mirror backend configured. Set NEXT_PUBLIC_MIRROR_API_URL and rebuild.",
    );
  }

  let response: Response;
  try {
    response = await fetch(absolute(path), init);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    // The browser reports "server down" and "CORS refused" the same way; a
    // no-cors probe tells them apart without reading anything.
    const backendResponds = await fetch(absolute("/api/health"), { mode: "no-cors", cache: "no-store" }).then(
      () => true,
      () => false,
    );
    throw new ApiError(
      backendResponds ? "blocked" : "network",
      networkFailureMessage({
        backendResponds,
        apiUrl: API_URL,
        pageOrigin: window.location.origin,
        dev: process.env.NODE_ENV !== "production",
      }),
    );
  }

  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      // Not JSON; httpError falls back to the status code.
    }
    const { message, code } = httpError(response.status, body, absolute(path));
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
  fit: { lens_centers_cm: [number, number][] } | null;
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
    fitHints: raw.fit ? { lensCentersCm: raw.fit.lens_centers_cm } : null,
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
      throw new ApiError("timeout", "Generating glasses is taking too long. Please try again.");
    }
    await sleep(POLL_INTERVAL_MS, signal);
    current = await getJob(current.id, signal);
  }
  return current;
}

export function readyGlasses(job: Job): ReadyGlasses | null {
  return job.status === "succeeded" && job.modelUrl
    ? { jobId: job.id, modelUrl: job.modelUrl, sourceImageUrl: job.sourceImageUrl, fitHints: job.fitHints }
    : null;
}
