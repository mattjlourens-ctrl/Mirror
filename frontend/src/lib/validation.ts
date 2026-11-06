export type ValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: string };

/**
 * Accepts an http(s) product URL. A missing scheme is treated as https, so
 * "shop.example.com/frames" works; any other scheme is rejected.
 */
export function validateProductUrl(input: string): ValidationResult<string> {
  const trimmed = input.trim();
  if (!trimmed) {
    return { ok: false, error: "Enter a product link." };
  }
  if (/\s/.test(trimmed)) {
    return { ok: false, error: "Links can't contain spaces." };
  }

  const hasScheme = /^[a-z][a-z0-9+.-]*:/i.test(trimmed);
  let url: URL;
  try {
    url = new URL(hasScheme ? trimmed : `https://${trimmed}`);
  } catch {
    return { ok: false, error: "That doesn't look like a valid link." };
  }

  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return { ok: false, error: "Use a link starting with http:// or https://." };
  }
  const host = url.hostname;
  if (host !== "localhost" && !/^[^.]+(\.[^.]+)+$/.test(host)) {
    return { ok: false, error: "Include the full website address, like shop.com." };
  }

  return { ok: true, value: url.toString() };
}

export const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
export const ACCEPTED_IMAGE_TYPES = ["image/png", "image/jpeg"] as const;
export const ACCEPTED_IMAGE_INPUT = ".png,.jpg,.jpeg,image/png,image/jpeg";

type FileLike = { name: string; type: string; size: number };

/** Accepts PNG or JPG files up to MAX_IMAGE_BYTES. */
export function validateImageFile<T extends FileLike>(file: T): ValidationResult<T> {
  const extension = file.name.toLowerCase().split(".").pop() ?? "";
  const typeOk = (ACCEPTED_IMAGE_TYPES as readonly string[]).includes(file.type);
  // Some systems report an empty type, so fall back to the extension.
  const extensionOk = ["png", "jpg", "jpeg"].includes(extension);
  if (!(typeOk || (file.type === "" && extensionOk))) {
    return { ok: false, error: "Choose a PNG or JPG image." };
  }
  if (file.size === 0) {
    return { ok: false, error: "That file is empty." };
  }
  if (file.size > MAX_IMAGE_BYTES) {
    return { ok: false, error: `Images must be ${formatBytes(MAX_IMAGE_BYTES)} or smaller.` };
  }
  return { ok: true, value: file };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(bytes % (1024 * 1024) === 0 ? 0 : 1)} MB`;
}
