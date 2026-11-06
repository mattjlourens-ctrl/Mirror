/**
 * Messages for failed backend calls. Each kind of failure gets its own
 * message: backend down, backend refusing this page, endpoint missing,
 * invalid request, server error, or Mirror's own explanation.
 */

const START_COMMAND = "uv run uvicorn backend.app:app --port 8000";
const DEV_SITE = "http://localhost:3000";

type ValidationItem = { loc?: unknown[]; msg?: unknown };

/** Message and code for a non-2xx response with an (optionally) parsed JSON body. */
export function httpError(status: number, body: unknown, url?: string): { message: string; code?: string } {
  const detail = (body as { detail?: unknown } | null)?.detail;

  // Mirror's own errors: {"detail": {"code", "message"}}.
  if (detail && typeof detail === "object" && !Array.isArray(detail)) {
    const { code, message } = detail as { code?: unknown; message?: unknown };
    if (typeof message === "string") return { message, code: typeof code === "string" ? code : undefined };
  }

  // FastAPI request validation: {"detail": [{"loc": [...], "msg": "..."}]}.
  if (Array.isArray(detail) && detail.length > 0) {
    const problems = (detail as ValidationItem[]).map(({ loc, msg }) => {
      const field = Array.isArray(loc) ? loc.filter((part) => part !== "body").at(-1) : undefined;
      const text = typeof msg === "string" ? msg : "invalid value";
      return field !== undefined ? `${String(field)}: ${text}` : text;
    });
    return { message: `Mirror rejected the request (${status}): ${problems.join("; ")}.`, code: "validation_error" };
  }

  // FastAPI's default 404 means the route doesn't exist on that server.
  if (status === 404 && detail === "Not Found") {
    return {
      message: `The server has no ${url ?? "such"} endpoint (404). Check that NEXT_PUBLIC_MIRROR_API_URL points at the Mirror backend.`,
      code: "endpoint_not_found",
    };
  }
  if (typeof detail === "string") return { message: `${detail} (${status})` };
  if (status >= 500) return { message: `Mirror's server hit an internal error (${status}). Please try again.` };
  return { message: `Mirror's server returned an unexpected error (${status}). Please try again.` };
}

/** Message for a request that never got a readable response. */
export function networkFailureMessage({
  backendResponds,
  apiUrl,
  pageOrigin,
  dev,
}: {
  /** Whether the backend answered a CORS-free probe, i.e. it's up but refused this page. */
  backendResponds: boolean;
  apiUrl: string;
  pageOrigin: string;
  dev: boolean;
}): string {
  if (backendResponds) {
    if (!dev) return "Mirror's server didn't accept a request from this page. Please try again later.";
    const openLocal = pageOrigin === DEV_SITE ? "" : `Open the site at ${DEV_SITE}, or `;
    const fix = `${openLocal}add ${pageOrigin} to MIRROR_ALLOWED_ORIGINS and restart the backend.`;
    return `The Mirror backend at ${apiUrl} is running but refused requests from this page (${pageOrigin}). ${fix[0].toUpperCase()}${fix.slice(1)}`;
  }
  return dev
    ? `Couldn't reach the Mirror backend at ${apiUrl}. Start it with "${START_COMMAND}" and try again.`
    : "Couldn't reach Mirror. Check your connection and try again.";
}
