import Link from "next/link";

/** Wordmark with a simple glasses glyph. */
export function Logo() {
  return (
    <Link
      href="/"
      className="inline-flex items-center gap-2 rounded-md text-neutral-950 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-neutral-950"
    >
      <svg aria-hidden viewBox="0 0 32 16" className="h-4 w-8" fill="none" stroke="currentColor" strokeWidth="2.2">
        <circle cx="8" cy="8" r="6" />
        <circle cx="24" cy="8" r="6" />
        <path d="M14 7.5c1.2-1.3 2.8-1.3 4 0" strokeLinecap="round" />
      </svg>
      <span className="text-lg font-semibold tracking-tight">Mirror</span>
    </Link>
  );
}
