import { CircleAlert, CircleCheck, Info, LoaderCircle } from "lucide-react";

export type Status =
  | { tone: "idle" }
  | { tone: "loading"; message: string }
  | { tone: "info" | "error" | "success"; message: string };

const STYLES = {
  loading: { icon: LoaderCircle, className: "text-neutral-500", iconClass: "animate-spin" },
  info: { icon: Info, className: "text-neutral-600", iconClass: "" },
  error: { icon: CircleAlert, className: "text-red-600", iconClass: "" },
  success: { icon: CircleCheck, className: "text-emerald-700", iconClass: "" },
} as const;

/** One-line status under a form. The region is always rendered so screen readers announce changes. */
export function StatusMessage({ status, id }: { status: Status; id?: string }) {
  const style = status.tone === "idle" ? null : STYLES[status.tone];
  const Icon = style?.icon;
  return (
    <div
      id={id}
      role={status.tone === "error" ? "alert" : "status"}
      aria-live="polite"
      className="min-h-5 text-sm"
    >
      {style && Icon && status.tone !== "idle" && (
        <p className={`flex items-start gap-2 ${style.className}`}>
          <Icon aria-hidden className={`mt-0.5 size-4 shrink-0 ${style.iconClass}`} />
          <span>{status.message}</span>
        </p>
      )}
    </div>
  );
}
