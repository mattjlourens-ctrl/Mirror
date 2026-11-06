/** Where the backend runs when you start it with the README's command. */
export const LOCAL_API_URL = "http://localhost:8000";

/**
 * The backend base URL: NEXT_PUBLIC_MIRROR_API_URL if set, otherwise the local
 * backend during development. A production build without it stays unconfigured
 * rather than pointing visitors at their own localhost.
 */
export function resolveApiUrl(value: string | undefined, nodeEnv: string | undefined): string {
  const configured = value?.trim().replace(/\/+$/, "") ?? "";
  if (configured) return configured;
  return nodeEnv === "production" ? "" : LOCAL_API_URL;
}
