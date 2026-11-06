import { ApiError } from "@/lib/api";
import type { Status } from "./status-message";

/** Turns a failed backend call into a status line. */
export function failureStatus(error: unknown): Status {
  if (error instanceof ApiError) {
    return { tone: "error", message: error.message };
  }
  return { tone: "error", message: "Something went wrong. Please try again." };
}

export function errorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : "Something went wrong. Please try again.";
}

/** Errors where uploading a photo is the useful next step. */
export function suggestsUpload(error: unknown): boolean {
  return (
    error instanceof ApiError &&
    ["no_product_images", "no_front_view", "access_restricted", "page_not_found", "page_unavailable", "not_a_page", "unreachable", "timeout"].includes(
      error.code ?? "",
    )
  );
}
