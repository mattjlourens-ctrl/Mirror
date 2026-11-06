import type { Candidate } from "@/lib/api";

/** Grid of front-view product photos found on a page, best first. */
export function CandidatePicker({
  candidates,
  onChoose,
  disabled = false,
}: {
  candidates: Candidate[];
  onChoose: (candidate: Candidate) => void;
  disabled?: boolean;
}) {
  return (
    <fieldset className="flex flex-col gap-3" disabled={disabled}>
      <legend className="text-sm text-neutral-600">
        We found several photos of these glasses. Choose the one to try on.
      </legend>
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {candidates.map((candidate, index) => (
          <li key={candidate.id}>
            <button
              type="button"
              onClick={() => onChoose(candidate)}
              className="flex w-full rounded-xl border border-neutral-200 p-2 text-left hover:border-neutral-400 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-950 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {/* eslint-disable-next-line @next/next/no-img-element -- backend-served photo of unknown size */}
              <img
                src={candidate.imageUrl}
                alt={`Product photo ${index + 1}`}
                className="aspect-[4/3] w-full rounded-lg bg-neutral-50 object-contain"
              />
            </button>
          </li>
        ))}
      </ul>
    </fieldset>
  );
}
