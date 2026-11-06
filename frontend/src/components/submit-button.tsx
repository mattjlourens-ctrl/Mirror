import { LoaderCircle } from "lucide-react";
import type { ReactNode } from "react";

export function SubmitButton({
  children,
  loading = false,
  disabled = false,
}: {
  children: ReactNode;
  loading?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      type="submit"
      disabled={disabled || loading}
      aria-busy={loading}
      className="inline-flex h-14 shrink-0 items-center justify-center gap-2 rounded-xl bg-neutral-950 px-7 text-base font-medium text-white transition-colors hover:bg-neutral-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-950 disabled:cursor-not-allowed disabled:bg-neutral-300"
    >
      {loading && <LoaderCircle aria-hidden className="size-4 animate-spin" />}
      {children}
    </button>
  );
}
